from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import random
import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI()

@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat() + "Z"}

@app.get("/")
def root():
    # Preserve your Quick Check contract
    return {"ok": True}

# Serve ./web at /web
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

# ---------- Models ----------
class PortfolioPosition(BaseModel):
    ticker: str
    quantity: float
    trade_price: float

class RunRequest(BaseModel):
    tickers: List[str] = Field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_cash: float = 100_000
    portfolio_positions: Optional[List[PortfolioPosition]] = None

class RunResult(BaseModel):
    decisions: Dict[str, Any]
    analyst_signals: Dict[str, Any]

class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100_000
    portfolio_positions: Optional[List[PortfolioPosition]] = None
    # NOTE: we keep schema the same; strategy params are fixed below (fast=20, slow=50)

class BacktestSummary(BaseModel):
    performance_metrics: Dict[str, Any]
    final_portfolio: Dict[str, Any]
    total_days: int

# New: prices endpoint (kept simple and consistent with your naming)
class PricesRequest(BaseModel):
    tickers: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: Optional[str] = "1d"

# ---------- Helpers ----------
def _date_or_default(start: Optional[str], end: Optional[str]) -> tuple[str, str]:
    end_s = end or datetime.now().strftime("%Y-%m-%d")
    if start:
        start_s = start
    else:
        dt_end = datetime.strptime(end_s, "%Y-%m-%d")
        start_s = (dt_end - timedelta(days=30)).strftime("%Y-%m-%d")
    return start_s, end_s

def _parse_date_str(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def _compute_metrics(equity: pd.Series) -> Dict[str, float]:
    rets = equity.pct_change().dropna()
    ann = 252
    vol = float(rets.std() * np.sqrt(ann)) if not rets.empty else 0.0
    sharpe = float((rets.mean() * ann) / vol) if vol > 0 else None
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    max_dd = float(dd.min()) if not dd.empty else None
    years = (equity.index[-1] - equity.index[0]).days / 365.25 if len(equity) > 1 else 0
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1) if years > 0 else None
    return {"sharpe_ratio": sharpe, "max_drawdown": max_dd, "volatility": vol, "cagr": cagr}

# ---------- Endpoints ----------

@app.post("/hedge-fund/run", response_model=RunResult)
def hedge_run(req: RunRequest):
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    _ = _date_or_default(req.start_date, req.end_date)

    actions = ["buy", "hold", "sell"]
    decisions: Dict[str, Any] = {}
    analyst: Dict[str, Any] = {}
    for t in req.tickers:
        action = random.choice(actions)
        qty = random.randint(1, 5)
        decisions[t] = {"action": action, "quantity": qty}
        analyst[t] = {
            "signal": random.choice(["bullish", "bearish", "neutral"]),
            "confidence": round(random.uniform(0.55, 0.9), 2),
        }
    return RunResult(decisions=decisions, analyst_signals=analyst)

@app.post("/data/prices")
def get_prices(req: PricesRequest):
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    start_s, end_s = _date_or_default(req.start_date, req.end_date)

    data = yf.download(
        tickers=" ".join(req.tickers),
        start=start_s,
        end=end_s,
        interval=req.interval or "1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    out: List[Dict[str, Any]] = []
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker frame
        for t in req.tickers:
            try:
                df_t = data.xs(t, axis=1, level=0)
            except Exception:
                continue
            if "Close" not in df_t.columns:
                continue
            ser = df_t["Close"].dropna()
            if ser.empty:
                continue
            out.append({
                "ticker": t,
                "dates": ser.index.strftime("%Y-%m-%d").tolist(),
                "closes": ser.round(6).astype(float).tolist()
            })
    else:
        # Single-ticker frame
        if "Close" not in data.columns:
            raise HTTPException(status_code=404, detail="No price data.")
        ser = data["Close"].dropna()
        t = req.tickers[0]
        out.append({
            "ticker": t,
            "dates": ser.index.strftime("%Y-%m-%d").tolist(),
            "closes": ser.round(6).astype(float).tolist()
        })

    return {"data": out}

@app.post("/hedge-fund/backtest", response_model=BacktestSummary)
def hedge_backtest(req: BacktestRequest):
    # Validate dates
    try:
        start_dt = _parse_date_str(req.start_date)
        end_dt = _parse_date_str(req.end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD.")
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")

    # Download prices
    start_s = req.start_date
    end_s = req.end_date
    data = yf.download(
        tickers=" ".join(req.tickers),
        start=start_s,
        end=end_s,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Build aligned returns and strategy returns for each ticker (SMA 20/50)
    fast, slow = 20, 50
    all_strat_rets = []
    all_dates = None

    def strat_returns(series: pd.Series) -> pd.Series:
        fast_ma = series.rolling(window=fast, min_periods=fast).mean()
        slow_ma = series.rolling(window=slow, min_periods=slow).mean()
        signal = (fast_ma > slow_ma).astype(int)
        position = signal.shift(1).fillna(0)
        rets = series.pct_change().fillna(0.0)
        return (position * rets).rename("strat_ret")

    if isinstance(data.columns, pd.MultiIndex):
        # data[ticker][Close]
        for t in req.tickers:
            try:
                df_t = data.xs(t, axis=1, level=0)
                if "Close" not in df_t.columns:
                    continue
                strat = strat_returns(df_t["Close"].dropna())
                all_strat_rets.append(strat)
                if all_dates is None:
                    all_dates = strat.index
            except Exception:
                continue
    else:
        if "Close" not in data.columns:
            raise HTTPException(status_code=404, detail="No price data for selected range.")
        strat = strat_returns(data["Close"].dropna())
        all_strat_rets.append(strat)
        all_dates = strat.index

    if not all_strat_rets:
        raise HTTPException(status_code=404, detail="No price data for any ticker.")

    # Align and equal-weight
    df_rets = pd.concat(all_strat_rets, axis=1).fillna(0.0)
    port_rets = df_rets.mean(axis=1)  # equal weight
    equity = (1 + port_rets).cumprod() * float(req.initial_capital)

    metrics = _compute_metrics(equity)

    # Preserve your response structure
    final_port = {
        "cash": round(float(equity.iloc[-1]), 2),
        "positions": {t: 0 for t in req.tickers},  # strategy modeled as fractional exposure; no discrete shares tracked
        "portfolio_value": round(float(equity.iloc[-1]), 2),
    }
    total_days = int((end_dt - start_dt).days + 1)

    return BacktestSummary(
        performance_metrics=metrics,
        final_portfolio=final_port,
        total_days=total_days
    )

