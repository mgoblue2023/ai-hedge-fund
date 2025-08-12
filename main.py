

from __future__ import annotations
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
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
    accept = (request.headers.get("accept") or "").lower()
    ua = (request.headers.get("user-agent") or "").lower()
    if "text/html" in accept or "mozilla" in ua or "chrome" in ua or "safari" in ua:
        return RedirectResponse(url="/web/")
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
    fast: int = 20
    slow: int = 50

class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100_000
    portfolio_positions: Optional[List[PortfolioPosition]] = None
    fast: int = 20
    slow: int = 50

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
        return {"signal": "hold", "confidence": 0.0, "price": None}
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

def _extract_closes(df: pd.DataFrame, tickers: List[str]) -> Tuple[Dict[str, pd.Series], List[str]]:
    """Return {ticker: close_series} and a list of missing tickers."""
    closes: Dict[str, pd.Series] = {}
    missing: List[str] = []

    if df is None or df.empty:
        return closes, tickers[:]  # everything missing

    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            try:
                sub = df.xs(t, axis=1, level=0)
                ser = sub["Close"].dropna()
                if ser.empty:
                    missing.append(t)
                else:
                    closes[t] = ser
            except Exception:
                missing.append(t)
    else:
        t = tickers[0]
        if "Close" in df.columns:
            ser = df["Close"].dropna()
            if ser.empty:
                missing.append(t)
            else:
                closes[t] = ser
        else:
            missing.append(t)

    return closes, missing

def _safe_download(tickers: List[str],
                   start_dt: pd.Timestamp,
                   end_dt: pd.Timestamp,
                   interval: str = "1d") -> pd.DataFrame:
    """
    Robust price fetcher:
      1) Bulk yf.download(start/end)
      2) Bulk yf.download(period=window)
      3) Per-ticker .history(start/end)
      4) Per-ticker .history(period=window)
      5) Per-ticker Stooq CSV (https://stooq.com)
    Returns a DataFrame with MultiIndex columns (ticker, 'Close') or empty df.
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return pd.DataFrame()

    start_s = start_dt.strftime("%Y-%m-%d")
    end_s_excl = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    days = max(365, (end_dt - start_dt).days + 30)
    period = "2y" if days > 365 else "1y"

    # STEP 1: bulk start/end
    try:
        print(f"[DL] step1 bulk download start/end tickers={tickers} {start_s}..{end_s_excl}")
        df = yf.download(
            tickers=" ".join(tickers),
            start=start_s,
            end=end_s_excl,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if df is not None and not df.empty:
            print(f"[DL] step1 ok shape={getattr(df,'shape',None)}")
            return df
        print("[DL] step1 empty")
    except Exception as e:
        print(f"[DL] step1 error: {e}")

    # STEP 2: bulk with period
    try:
        print(f"[DL] step2 bulk download period={period} tickers={tickers}")
        df = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if df is not None and not df.empty:
            print(f"[DL] step2 ok shape={getattr(df,'shape',None)}")
            return df
        print("[DL] step2 empty")
    except Exception as e:
        print(f"[DL] step2 error: {e}")

    # helper to combine per-ticker 'Close' into MultiIndex frame
    def _combine(frames: List[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1).sort_index()
        if not isinstance(out.columns, pd.MultiIndex):
            out.columns = pd.MultiIndex.from_tuples(out.columns)
        return out

    # STEP 3: per-ticker start/end
    frames = []
    for t in tickers:
        try:
            print(f"[DL] step3 {t} history start/end {start_s}..{end_s_excl}")
            h = yf.Ticker(t).history(start=start_s, end=end_s_excl, interval=interval, auto_adjust=True)
            if h is not None and not h.empty and "Close" in h.columns:
                ser = h["Close"].dropna().copy()
                ser.index = pd.to_datetime(ser.index).tz_localize(None)
                frames.append(pd.DataFrame({(t, "Close"): ser}))
            else:
                print(f"[DL] step3 {t} empty")
        except Exception as e:
            print(f"[DL] step3 {t} error: {e}")
    df = _combine(frames)
    if df is not None and not df.empty:
        print(f"[DL] step3 combined ok shape={df.shape}")
        return df

    # STEP 4: per-ticker period
    frames = []
    for t in tickers:
        try:
            print(f"[DL] step4 {t} history period={period}")
            h = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
            if h is not None and not h.empty and "Close" in h.columns:
                ser = h["Close"].dropna().copy()
                ser.index = pd.to_datetime(ser.index).tz_localize(None)
                frames.append(pd.DataFrame({(t, "Close"): ser}))
            else:
                print(f"[DL] step4 {t} empty")
        except Exception as e:
            print(f"[DL] step4 {t} error: {e}")
    df = _combine(frames)
    if df is not None and not df.empty:
        print(f"[DL] step4 combined ok shape={df.shape}")
        return df

    # STEP 5: Stooq per-ticker (CSV)
    frames = []
    for t in tickers:
        ser = _fetch_stooq_series(t, start_dt, end_dt)
        if ser is not None and not ser.empty:
            frames.append(pd.DataFrame({(t, "Close"): ser}))
    if frames:
        df = _combine(frames)
        print(f"[DL] step5 stooq combined ok shape={df.shape}")
        return df
    print("[DL] step5 stooq empty")

    print("[DL] all steps empty → returning empty df")
    return pd.DataFrame()


def _stooq_symbol(t: str) -> str:
    # Stooq uses e.g. spy.us, aapl.us, brk-b.us
    t = t.strip().lower().replace(".", "-")
    if not t.endswith(".us"):
        t = f"{t}.us"
    return t

def _fetch_stooq_series(ticker: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> Optional[pd.Series]:
    sym = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"[DL] stooq {ticker} error: {e}")
        return None
    if df is None or df.empty or "Date" not in df or "Close" not in df:
        print(f"[DL] stooq {ticker} empty")
        return None
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date")
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
    if df.empty:
        print(f"[DL] stooq {ticker} filtered empty")
        return None
    ser = pd.Series(pd.to_numeric(df["Close"], errors="coerce").values, index=df["Date"])
    ser = ser.dropna()
    return ser if not ser.empty else None



# ---------- Endpoints ----------
@app.post("/hedge-fund/run")
def hedge_run(req: RunRequest):
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")

# Validate SMA inputs
fast = int(max(2, req.fast))
slow = int(max(3, req.slow))
if fast >= slow:
    raise HTTPException(status_code=400, detail="Invalid SMA inputs: fast must be less than slow.")
req.fast, req.slow = fast, slow

# Ensure enough history for SMAs
min_hist_days = max(120, slow * 2)
fetch_start = min(_parse_date(req.start_date) if req.start_date else (pd.Timestamp.today().normalize() - pd.Timedelta(days=420)),
                  (pd.Timestamp.today().normalize() if not req.end_date else _parse_date(req.end_date)) - pd.Timedelta(days=min_hist_days))
df = _safe_download(req.tickers, fetch_start, _parse_date(req.end_date) if req.end_date else pd.Timestamp.today().normalize(), interval="1d")


    # Default lookback ~ 300 trading days for MAs
    today = pd.Timestamp.today().normalize()
    end_dt = _parse_date(req.end_date) if req.end_date else today
    if end_dt > today:
        end_dt = today
    start_dt = _parse_date(req.start_date) if req.start_date else (end_dt - pd.Timedelta(days=420))

    df = _safe_download(req.tickers, start_dt, end_dt, interval="1d")
    closes, missing = _extract_closes(df, req.tickers)
    if not closes:
        raise HTTPException(status_code=404, detail=f"No price data for: {', '.join(missing)}")

    decisions: Dict[str, Any] = {}
    analyst: Dict[str, Any] = {}
    for t, close in closes.items():
        sig = _latest_sma_signal(close, req.fast, req.slow)
        action = {"buy": "buy", "sell": "sell"}.get(sig.get("signal"), "hold")
        qty = 0 if action == "hold" else 1
        decisions[t] = {"action": action, "quantity": qty}
        analyst[t] = {"signal": sig.get("signal", "hold"),
                      "confidence": sig.get("confidence", 0.0),
                      "price": sig.get("price", None)}

    if missing:
        analyst["_warnings"] = {"missing": missing}

    return {"decisions": decisions, "analyst_signals": analyst, "generated_at": datetime.utcnow().isoformat() + "Z"}

@app.post("/data/prices")
def get_prices(req: PricesRequest):
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    today = pd.Timestamp.today().normalize()
    start_dt = _parse_date(req.start_date) if req.start_date else (today - pd.DateOffset(years=1))
    end_dt = _parse_date(req.end_date) if req.end_date else today
    if end_dt > today:
        end_dt = today

    df = _safe_download(req.tickers, start_dt, end_dt, req.interval or "1d")
    closes, missing = _extract_closes(df, req.tickers)
    if not closes:
        raise HTTPException(status_code=404, detail=f"No price data for: {', '.join(missing)}")

    out: List[Dict[str, Any]] = []
    for t, ser in closes.items():
        out.append({"ticker": t,
                    "dates": ser.index.strftime("%Y-%m-%d").tolist(),
                    "closes": ser.round(6).astype(float).tolist()})
    return {"data": out, "missing": missing}

@app.post("/hedge-fund/backtest")
def backtest(req: BacktestRequest):
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    try:
        start_dt = _parse_date(req.start_date)
        end_dt = _parse_date(req.end_date)
        today = pd.Timestamp.today().normalize()
        if end_dt > today:
            end_dt = today
    except Exception:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD.")
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")
# Validate SMA inputs
fast = int(max(2, req.fast))
slow = int(max(3, req.slow))
if fast >= slow:
    raise HTTPException(status_code=400, detail="Invalid SMA inputs: fast must be less than slow.")
req.fast, req.slow = fast, slow


    df = _safe_download(req.tickers, start_dt, end_dt, interval="1d")
    closes, missing = _extract_closes(df, req.tickers)
    if not closes:
        raise HTTPException(status_code=404, detail=f"No price data for: {', '.join(missing)}")

    strat_rets, bench_rets = [], []
    for t, close in closes.items():
        strat_rets.append(_sma_returns(close, req.fast, req.slow))
        bench_rets.append(close.pct_change().fillna(0.0))

    strat_df = pd.concat(strat_rets, axis=1).fillna(0.0)
    bench_df = pd.concat(bench_rets, axis=1).fillna(0.0)
    port_rets = strat_df.mean(axis=1)
    bench_rets_eq = bench_df.mean(axis=1)

    if port_rets.dropna().empty:
        eq_index = pd.date_range(start_dt, end_dt, freq="B")
        equity = pd.Series([float(req.initial_capital)] * len(eq_index), index=eq_index)
        bench = equity.copy()
    else:
        equity = (1 + port_rets).cumprod() * float(req.initial_capital)
        bench  = (1 + bench_rets_eq).cumprod() * float(req.initial_capital)

    metrics = _metrics(equity)
    equity_curve = {
        "dates": equity.index.strftime("%Y-%m-%d").tolist(),
        "equity": equity.round(2).astype(float).tolist(),
        "benchmark": bench.round(2).astype(float).tolist(),
    }
    end_val = float(equity.iloc[-1]) if len(equity) else float(req.initial_capital)
    final_port = {
        "cash": round(end_val, 2),
        "positions": {t: 0 for t in req.tickers},
        "portfolio_value": round(end_val, 2),
    }
    return {
        "performance_metrics": metrics,
        "final_portfolio": final_port,
        "total_days": int((end_dt - start_dt).days + 1),
        "equity_curve": equity_curve,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "missing": missing,
    }

# Local dev entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
