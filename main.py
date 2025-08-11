from __future__ import annotations
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field

app = FastAPI(docs_url=None, redoc_url=None, title="AI Hedge Fund API")

# Serve ./web at /web
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(WEB_DIR):
    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")


# ---------- Root & Docs ----------
@app.get("/", include_in_schema=False)
def root(request: Request):
    # If a browser (Accept: text/html or typical UA), send them to the UI.
    accept = (request.headers.get("accept") or "").lower()
    ua = (request.headers.get("user-agent") or "").lower()
    if "text/html" in accept or "mozilla" in ua or "chrome" in ua or "safari" in ua:
        return RedirectResponse(url="/web/")
    # Otherwise, keep returning simple JSON for health checks/bots.
    return {"ok": True}


@app.get("/docs", include_in_schema=False)
async def custom_docs():
    html = get_swagger_ui_html(openapi_url="/openapi.json", title="Docs")
    inject = """
    <div style="position:fixed;top:10px;right:10px;z-index:9999">
      <a href="/web" style="text-decoration:none;font-weight:700;">⬅ Back to UI</a>
    </div>
    """
    content = html.body.decode("utf-8").replace("</body>", inject + "</body>")
    return HTMLResponse(content=content, status_code=200)

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat() + "Z"}


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

class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100_000
    portfolio_positions: Optional[List[PortfolioPosition]] = None

class PricesRequest(BaseModel):
    tickers: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: Optional[str] = "1d"


# ---------- Helpers ----------
def _parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).tz_localize(None)

def _metrics(equity: pd.Series) -> Dict[str, float]:
    rets = equity.pct_change().dropna()
    ann = 252
    vol = float(rets.std() * np.sqrt(ann)) if not rets.empty else 0.0
    sharpe = float((rets.mean() * ann) / vol) if vol > 0 else None
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    max_dd = float(dd.min()) if not dd.empty else None
    if len(equity) > 1:
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else None
    else:
        cagr = None
    return {"sharpe_ratio": sharpe, "volatility": vol, "max_drawdown": max_dd, "cagr": cagr}

def _sma_returns(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    fast_ma = close.rolling(window=fast, min_periods=fast).mean()
    slow_ma = close.rolling(window=slow, min_periods=slow).mean()
    signal = (fast_ma > slow_ma).astype(int)
    position = signal.shift(1).fillna(0)
    rets = close.pct_change().fillna(0.0)
    return position * rets

def _latest_sma_signal(close: pd.Series, fast: int = 20, slow: int = 50) -> Dict[str, Any]:
    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(slow, min_periods=slow).mean()
    if fast_ma.dropna().empty or slow_ma.dropna().empty:
        return {"signal": "hold", "confidence": 0.0}
    f, s = float(fast_ma.iloc[-1]), float(slow_ma.iloc[-1])
    price = float(close.iloc[-1])
    spread = abs(f - s) / max(price, 1e-9)
    conf = max(0.0, min(0.95, round(spread * 10, 2)))
    if f > s:
        return {"signal": "buy", "confidence": conf, "price": price}
    elif f < s:
        return {"signal": "sell", "confidence": conf, "price": price}
    else:
        return {"signal": "hold", "confidence": 0.0, "price": price}


# ---------- Endpoints ----------
@app.post("/hedge-fund/run")
def hedge_run(req: RunRequest):
    """
    Produce real buy/sell/hold signals using latest SMA(20/50).
    Keeps the original response shape: { decisions: {t: {action, quantity}}, analyst_signals: {...} }.
    """
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=220)  # enough for 50-day MA warmup

    df = yf.download(
        tickers=" ".join(req.tickers),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    decisions: Dict[str, Any] = {}
    analyst: Dict[str, Any] = {}

    if isinstance(df.columns, pd.MultiIndex):
        for t in req.tickers:
            try:
                sub = df.xs(t, axis=1, level=0)
                close = sub["Close"].dropna()
            except Exception:
                close = pd.Series(dtype=float)
            sig = _latest_sma_signal(close)
            action = {"buy": "buy", "sell": "sell"}.get(sig.get("signal"), "hold")
            qty = 0 if action == "hold" else 1
            decisions[t] = {"action": action, "quantity": qty}
            analyst[t] = {"signal": sig.get("signal", "hold"),
                          "confidence": sig.get("confidence", 0.0),
                          "price": sig.get("price", None)}
    else:
        # single ticker frame
        if "Close" in df.columns and req.tickers:
            close = df["Close"].dropna()
            t = req.tickers[0]
            sig = _latest_sma_signal(close)
            action = {"buy": "buy", "sell": "sell"}.get(sig.get("signal"), "hold")
            qty = 0 if action == "hold" else 1
            decisions[t] = {"action": action, "quantity": qty}
            analyst[t] = {"signal": sig.get("signal", "hold"),
                          "confidence": sig.get("confidence", 0.0),
                          "price": sig.get("price", None)}

    return {"decisions": decisions, "analyst_signals": analyst, "generated_at": datetime.utcnow().isoformat() + "Z"}

@app.post("/data/prices")
def get_prices(req: PricesRequest):
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    start = req.start_date or (pd.Timestamp.today() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    end = req.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    df = yf.download(
        tickers=" ".join(req.tickers),
        start=start,
        end=end,
        interval=req.interval or "1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out: List[Dict[str, Any]] = []
    if isinstance(df.columns, pd.MultiIndex):
        for t in req.tickers:
            try:
                sub = df.xs(t, axis=1, level=0)
            except Exception:
                continue
            if "Close" not in sub.columns:
                continue
            ser = sub["Close"].dropna()
            if ser.empty:
                continue
            out.append({"ticker": t,
                        "dates": ser.index.strftime("%Y-%m-%d").tolist(),
                        "closes": ser.round(6).astype(float).tolist()})
    else:
        if "Close" not in df.columns:
            raise HTTPException(status_code=404, detail="No price data.")
        ser = df["Close"].dropna()
        out.append({"ticker": req.tickers[0],
                    "dates": ser.index.strftime("%Y-%m-%d").tolist(),
                    "closes": ser.round(6).astype(float).tolist()})
    return {"data": out}

@app.post("/hedge-fund/backtest")
def backtest(req: BacktestRequest):
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    try:
        start_dt = _parse_date(req.start_date)
        end_dt = _parse_date(req.end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD.")
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    df = yf.download(
        tickers=" ".join(req.tickers),
        start=req.start_date,
        end=req.end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    strat_rets = []
    bench_rets = []
    if isinstance(df.columns, pd.MultiIndex):
        for t in req.tickers:
            try:
                sub = df.xs(t, axis=1, level=0)
                if "Close" not in sub.columns:
                    continue
                close = sub["Close"].dropna()
            except Exception:
                continue
            strat_rets.append(_sma_returns(close))
            bench_rets.append(close.pct_change().fillna(0.0))
    else:
        if "Close" not in df.columns:
            raise HTTPException(status_code=404, detail="No price data for selected range.")
        close = df["Close"].dropna()
        strat_rets.append(_sma_returns(close))
        bench_rets.append(close.pct_change().fillna(0.0))

    if not strat_rets:
        raise HTTPException(status_code=404, detail="No price data for any ticker.")

    strat_df = pd.concat(strat_rets, axis=1).fillna(0.0)
    bench_df = pd.concat(bench_rets, axis=1).fillna(0.0)
    port_rets = strat_df.mean(axis=1)          # equal-weight strategy
    bench_rets_eq = bench_df.mean(axis=1)      # equal-weight benchmark

    equity = (1 + port_rets).cumprod() * float(req.initial_capital)
    bench = (1 + bench_rets_eq).cumprod() * float(req.initial_capital)

    metrics = _metrics(equity)

    equity_curve = {
        "dates": equity.index.strftime("%Y-%m-%d").tolist(),
        "equity": equity.round(2).astype(float).tolist(),
        "benchmark": bench.round(2).astype(float).tolist(),
    }

    final_port = {
        "cash": round(float(equity.iloc[-1]), 2),
        "positions": {t: 0 for t in req.tickers},
        "portfolio_value": round(float(equity.iloc[-1]), 2),
    }

    return {
        "performance_metrics": metrics,
        "final_portfolio": final_port,
        "total_days": int((end_dt - start_dt).days + 1),
        "equity_curve": equity_curve,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# Local dev entrypoint (Render uses gunicorn via Procfile)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
