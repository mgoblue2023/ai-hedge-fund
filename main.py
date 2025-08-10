from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import random

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI()
from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/web")


# ---------- Base route (health) ----------

# Serve ./web at /web
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

# ---------- Models ----------
class PortfolioPosition(BaseModel):
    ticker: str
    quantity: float
    trade_price: float

class RunRequest(BaseModel):
    tickers: List[str] = Field(default_factory=list)
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD
    initial_cash: float = 100_000
    portfolio_positions: Optional[List[PortfolioPosition]] = None

class RunResult(BaseModel):
    decisions: Dict[str, Any]
    analyst_signals: Dict[str, Any]

class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: str   # YYYY-MM-DD
    end_date: str     # YYYY-MM-DD
    initial_capital: float = 100_000
    portfolio_positions: Optional[List[PortfolioPosition]] = None

class BacktestSummary(BaseModel):
    performance_metrics: Dict[str, Any]
    final_portfolio: Dict[str, Any]
    total_days: int

class EquityCurveRequest(BaseModel):
    tickers: List[str]
    start_date: str   # YYYY-MM-DD
    end_date: str     # YYYY-MM-DD
    initial_capital: float = 100_000

class EquityCurveResponse(BaseModel):
    dates: List[str]        # ISO strings YYYY-MM-DD
    equity: List[float]     # portfolio value per day

# ---------- Helpers ----------
def _date_or_default(start: Optional[str], end: Optional[str]) -> tuple[str, str]:
    end_s = end or datetime.now().strftime("%Y-%m-%d")
    if start:
        start_s = start
    else:
        dt_end = datetime.strptime(end_s, "%Y-%m-%d")
        start_s = (dt_end - timedelta(days=30)).strftime("%Y-%m-%d")
    return start_s, end_s

def _date_range(start_s: str, end_s: str):
    d0 = datetime.strptime(start_s, "%Y-%m-%d")
    d1 = datetime.strptime(end_s, "%Y-%m-%d")
    step = timedelta(days=1)
    cur = d0
    while cur <= d1:
        yield cur
        cur += step

# ---------- Endpoints ----------
@app.post("/hedge-fund/run", response_model=RunResult)
def hedge_run(req: RunRequest):
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    _ = _date_or_default(req.start_date, req.end_date)

    actions = ["buy", "hold", "sell"]
    decisions = {}
    analyst = {}
    for t in req.tickers:
        action = random.choice(actions)
        qty = random.randint(1, 5)
        decisions[t] = {"action": action, "quantity": qty}
        analyst[t] = {
            "signal": random.choice(["bullish", "bearish", "neutral"]),
            "confidence": round(random.uniform(0.55, 0.9), 2),
        }
    return RunResult(decisions=decisions, analyst_signals=analyst)

@app.post("/hedge-fund/backtest", response_model=BacktestSummary)
def hedge_backtest(req: BacktestRequest):
    try:
        start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(req.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD.")
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    # Toy simulation
    num_days = (end_dt - start_dt).days + 1
    cash = req.initial_capital
    holdings = {t: 0 for t in req.tickers}
    price = {t: 100.0 for t in req.tickers}

    for _ in range(num_days):
        for t in req.tickers:
            # random walk price
            price[t] = round(price[t] + random.uniform(-2.0, 2.0), 2)
            price[t] = max(1.0, price[t])
            # trivial strategy
            move = random.choice(["buy", "hold", "sell"])
            if move == "buy" and cash > price[t]:
                cash -= price[t]
                holdings[t] += 1
            elif move == "sell" and holdings[t] > 0:
                cash += price[t]
                holdings[t] -= 1

    portfolio_value = cash + sum(holdings[t]*price[t] for t in req.tickers)
    perf = {
        "sharpe_ratio": round(random.uniform(0.5, 2.0), 2),
        "max_drawdown": round(random.uniform(-0.2, -0.05), 3),
    }
    final_port = {"cash": round(cash, 2), "positions": holdings, "portfolio_value": round(portfolio_value, 2)}
    return BacktestSummary(performance_metrics=perf, final_portfolio=final_port, total_days=num_days)

@app.post("/hedge-fund/equity-curve", response_model=EquityCurveResponse)
def equity_curve(req: EquityCurveRequest):
    """Generate a simple equity time series between start_date and end_date."""
    try:
        start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(req.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD.")
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")

    dates: List[str] = []
    equity: List[float] = []

    # Start near initial_capital; do a mean-reverting drift + noise
    cur = float(req.initial_capital)
    target = cur * random.uniform(0.95, 1.15)  # drift target
    for d in _date_range(req.start_date, req.end_date):
        # small mean reversion toward target plus noise
        cur += (target - cur) * 0.03 + random.uniform(-0.004, 0.004) * max(cur, 1.0)
        cur = max(1000.0, cur)
        dates.append(d.strftime("%Y-%m-%d"))
        equity.append(round(cur, 2))

    return EquityCurveResponse(dates=dates, equity=equity)



