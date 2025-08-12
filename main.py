
import os, math, sqlite3, datetime as dt, pathlib
from typing import Optional, Literal, List, Dict, Any

from fastapi import FastAPI, APIRouter, Query, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_NAME = "AI Hedge Fund"
app = FastAPI(title=APP_NAME)

# -------------------------
# Static files (/web) + root redirect
# -------------------------
BASE_DIR = pathlib.Path(__file__).parent.resolve()
WEB_DIR = BASE_DIR / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/web/")

# -------------------------
# Health endpoint (for pill + uptime checks)
# -------------------------
@app.get("/health")
def health():
    return {
        "ok": True,                     # <-- add this for the old UI
        "status": "OK",
        "utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "app": APP_NAME,
    }


# -------------------------
# Stubs you already expose (safe to keep)
# -------------------------
@app.post("/hedge-fund/run")
def run_job():
    return {"ok": True, "action": "run", "utc": dt.datetime.utcnow().isoformat() + "Z"}

@app.post("/hedge-fund/backtest")
def backtest_job():
    return {
        "ok": True,
        "action": "backtest",
        "metrics": {"trades": 0, "pnl": 0.0},
        "utc": dt.datetime.utcnow().isoformat() + "Z",
    }

# -------------------------
# Trade sizing + execution log
# -------------------------
import pathlib, os, sqlite3  # (ensure these are imported at top)

BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Prefer ENV overrides; otherwise use a local .data folder (works on Windows & Render)
DATA_DIR = pathlib.Path(os.getenv("EXEC_DB_DIR", BASE_DIR / ".data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = os.getenv("EXEC_DB_PATH", str(DATA_DIR / "executions.db"))

def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _init_db():
    with _conn() as cx:
        cx.execute("""
        CREATE TABLE IF NOT EXISTS execution_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_utc TEXT NOT NULL,
          strategy TEXT,
          symbol TEXT NOT NULL,
          side TEXT CHECK(side IN ('BUY','SELL')) NOT NULL,
          qty INTEGER NOT NULL,
          price REAL NOT NULL,
          fill_price REAL NOT NULL,
          slippage_bps REAL DEFAULT 0,
          fee REAL DEFAULT 0,
          order_id TEXT,
          status TEXT DEFAULT 'FILLED',
          notes TEXT
        );
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_exec_ts ON execution_log (ts_utc);")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_exec_symbol ON execution_log (symbol);")

@app.on_event("startup")
def on_startup():
    _init_db()

class TradeSizeRequest(BaseModel):
    price: float = Field(gt=0)
    equity: float = Field(gt=0)
    mode: Literal["risk_pct","allocation_pct","fixed_cash"] = "risk_pct"

    # risk_pct mode
    risk_pct: Optional[float] = Field(default=None, gt=0, description="e.g., 1 for 1%")
    stop_price: Optional[float] = Field(default=None, gt=0)

    # allocation_pct mode
    allocation_pct: Optional[float] = Field(default=None, gt=0)

    # fixed_cash mode
    cash_to_use: Optional[float] = Field(default=None, gt=0)

    # constraints
    lot_size: int = Field(default=1, gt=0)
    max_qty: Optional[int] = Field(default=None, gt=0)
    min_qty: int = Field(default=1, gt=0)

    # misc (context only)
    symbol: Optional[str] = None
    side: Optional[Literal["BUY","SELL"]] = None
    strategy: Optional[str] = None

class TradeSizeResponse(BaseModel):
    qty: int
    dollars: float
    mode_used: str
    details: Dict[str, Any]

class ExecutionRequest(BaseModel):
    symbol: str
    side: Literal["BUY","SELL"]
    qty: int = Field(gt=0)
    price: float = Field(gt=0, description="Intended price")
    slippage_bps: float = 0.0
    fee: float = 0.0
    strategy: Optional[str] = None
    notes: Optional[str] = None
    order_id: Optional[str] = None
    status: Literal["FILLED","PARTIAL","REJECTED","CANCELLED"] = "FILLED"
    ts_utc: Optional[str] = None  # ISO 8601

class ExecutionRecord(ExecutionRequest):
    id: int
    fill_price: float

class ExecQueryOut(BaseModel):
    id: int
    ts_utc: str
    strategy: Optional[str]
    symbol: str
    side: str
    qty: int
    price: float
    fill_price: float
    slippage_bps: float
    fee: float
    order_id: Optional[str]
    status: str
    notes: Optional[str]

def _slipped_price(price: float, side: str, slippage_bps: float) -> float:
    # Slippage moves against you: BUY pays more; SELL receives less.
    sign = 1 if side == "BUY" else -1
    return round(price * (1 + sign * (slippage_bps / 10_000.0)), 6)

trading = APIRouter(prefix="/hedge-fund", tags=["trading"])

@trading.post("/size", response_model=TradeSizeResponse)
def size_trade(req: TradeSizeRequest):
    qty: int = 0
    details: Dict[str, Any] = {}

    if req.mode == "risk_pct":
        if req.risk_pct is None or req.stop_price is None:
            raise HTTPException(400, "risk_pct and stop_price required for risk_pct mode")
        risk_amount = req.equity * (req.risk_pct / 100.0)
        per_share_risk = abs(req.price - req.stop_price)
        qty = 0 if per_share_risk <= 0 else math.floor(risk_amount / per_share_risk)
        details.update(risk_amount=risk_amount, per_share_risk=per_share_risk)

    elif req.mode == "allocation_pct":
        if req.allocation_pct is None:
            raise HTTPException(400, "allocation_pct required for allocation_pct mode")
        alloc_cash = req.equity * (req.allocation_pct / 100.0)
        qty = math.floor(alloc_cash / req.price)
        details.update(allocation_cash=alloc_cash)

    elif req.mode == "fixed_cash":
        if req.cash_to_use is None:
            raise HTTPException(400, "cash_to_use required for fixed_cash mode")
        qty = math.floor(req.cash_to_use / req.price)
        details.update(cash_to_use=req.cash_to_use)

    # lot size & bounds
    if req.lot_size > 1 and qty > 0:
        qty = qty - (qty % req.lot_size)
    if req.max_qty is not None:
        qty = min(qty, req.max_qty)
    qty = max(qty, req.min_qty if qty > 0 else 0)

    dollars = round(qty * req.price, 2)
    return TradeSizeResponse(qty=qty, dollars=dollars, mode_used=req.mode, details=details)

@trading.post("/execute", response_model=ExecutionRecord)
def execute(req: ExecutionRequest):
    ts = req.ts_utc or dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    fill_price = _slipped_price(req.price, req.side, req.slippage_bps)

    with _conn() as cx:
        cur = cx.execute("""
            INSERT INTO execution_log
            (ts_utc, strategy, symbol, side, qty, price, fill_price, slippage_bps, fee, order_id, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, req.strategy, req.symbol, req.side, req.qty, req.price, fill_price,
              req.slippage_bps, req.fee, req.order_id, req.status, req.notes))
        exec_id = cur.lastrowid

    dump = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    return ExecutionRecord(id=exec_id, fill_price=fill_price, **dump)

@trading.get("/db-path")
def db_path():
    return {"db_path": DB_PATH}

@trading.get("/db-info")
def db_info():
    import os, datetime as dt
    exists = os.path.exists(DB_PATH)
    size = os.path.getsize(DB_PATH) if exists else 0
    mtime = dt.datetime.fromtimestamp(os.path.getmtime(DB_PATH)).isoformat() if exists else None

    quick_check = None
    tables = []
    try:
        with _conn() as cx:
            quick_check = cx.execute("PRAGMA quick_check").fetchone()[0]
            tables = [r[0] for r in cx.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            ).fetchall()]
    except Exception as e:
        quick_check = f"error: {type(e).__name__}: {e}"

    return {
        "db_path": DB_PATH,
        "exists": exists,
        "size_bytes": size,
        "modified_iso": mtime,
        "quick_check": quick_check,
        "tables": tables,
    }


@trading.get("/executions", response_model=List[ExecQueryOut])
def executions(
    limit: int = Query(200, ge=1, le=2000),
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    since_iso: Optional[str] = Query(None, description="Return trades at/after this UTC ISO timestamp"),
):
    q = "SELECT id, ts_utc, strategy, symbol, side, qty, price, fill_price, slippage_bps, fee, order_id, status, notes FROM execution_log"
    clauses, args = [], []
    if symbol:
        clauses.append("symbol = ?"); args.append(symbol)
    if strategy:
        clauses.append("strategy = ?"); args.append(strategy)
    if since_iso:
        clauses.append("ts_utc >= ?"); args.append(since_iso)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY ts_utc DESC, id DESC LIMIT ?"; args.append(limit)

    with _conn() as cx:
        rows = cx.execute(q, tuple(args)).fetchall()
    return [ExecQueryOut(
        id=r[0], ts_utc=r[1], strategy=r[2], symbol=r[3], side=r[4], qty=r[5],
        price=r[6], fill_price=r[7], slippage_bps=r[8], fee=r[9], order_id=r[10],
        status=r[11], notes=r[12]) for r in rows]

# -------------------------
# SMA signal + backtest (no extra deps; uses Stooq CSV)
# -------------------------
def _canon_symbol_for_stooq(symbol: str) -> str:
    s = (symbol or "").strip().lower()
    if not s:
        raise HTTPException(400, "symbol is required")
    # Stooq uses .us for U.S. tickers (aapl.us, msft.us)
    if "." not in s:
        s = s + ".us"
    return s

def _fetch_daily_from_stooq(symbol: str):
    sym = _canon_symbol_for_stooq(symbol)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        raise HTTPException(502, f"data fetch failed for {symbol}: {e}")

    rdr = csv.DictReader(io.StringIO(text))
    rows = []
    for r in rdr:
        try:
            d = r["Date"]
            c = float(r["Close"])
            if d and c > 0:
                rows.append({"date": d, "close": c})
        except Exception:
            continue
    # Stooq returns newest last already, but sort to be safe
    rows.sort(key=lambda x: x["date"])
    if len(rows) < 30:
        raise HTTPException(404, f"not enough history for {symbol}")
    return rows  # [{date:'YYYY-MM-DD', close: float}, ...]

def _sma(values, window: int):
    if window <= 0:
        raise ValueError("window must be > 0")
    out, s = [], 0.0
    from collections import deque
    q = deque()
    for v in values:
        q.append(v); s += v
        if len(q) > window:
            s -= q.popleft()
        out.append(s / len(q))
    return out

@trading.get("/signal")
def signal(symbol: str, fast: int = 10, slow: int = 20):
    if fast >= slow:
        raise HTTPException(400, "fast must be < slow")
    rows = _fetch_daily_from_stooq(symbol)
    closes = [r["close"] for r in rows]
    sma_fast = _sma(closes, fast)
    sma_slow = _sma(closes, slow)

    # Determine state & most recent cross
    above_now = sma_fast[-1] > sma_slow[-1]
    above_prev = sma_fast[-2] > sma_slow[-2]
    if above_now and not above_prev:
        sig = "BUY"
    elif (not above_now) and above_prev:
        sig = "SELL"
    else:
        sig = "HOLD"

    return {
        "symbol": symbol.upper(),
        "asof": rows[-1]["date"],
        "price": closes[-1],
        "fast": fast, "slow": slow,
        "sma_fast": round(sma_fast[-1], 4),
        "sma_slow": round(sma_slow[-1], 4),
        "state": "above" if above_now else ("equal" if sma_fast[-1] == sma_slow[-1] else "below"),
        "signal": sig,
        "n_bars": len(rows),
    }

@trading.get("/backtest")
def backtest(
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
    fast: int = 10,
    slow: int = 20,
    initial_cash: float = 10_000.0,
):
    # Backward-compat: if no symbol provided, keep the old stub response
    if not symbol:
        return {
            "ok": True,
            "action": "backtest",
            "metrics": {"trades": 0, "pnl": 0.0},
            "utc": dt.datetime.utcnow().isoformat() + "Z",
        }

    if fast >= slow:
        raise HTTPException(400, "fast must be < slow")

    rows = _fetch_daily_from_stooq(symbol)
    # Apply optional date slicing
    if start:
        rows = [r for r in rows if r["date"] >= start]
    if end:
        rows = [r for r in rows if r["date"] <= end]
    if len(rows) < slow + 10:
        raise HTTPException(400, "not enough bars for the chosen windows/range")

    closes = [r["close"] for r in rows]
    dates = [r["date"] for r in rows]
    sma_fast = _sma(closes, fast)
    sma_slow = _sma(closes, slow)

    # Simple long-only: in market when fast > slow; out otherwise
    equity = [initial_cash]
    position = 0
    trades = 0
    max_equity = initial_cash
    max_drawdown = 0.0
    wins = 0
    loss_trades = 0

    # Track trade outcome on each entry/exit pair
    entry_equity = None

    for i in range(1, len(closes)):
        in_market = 1 if sma_fast[i - 1] > sma_slow[i - 1] else 0
        ret = (closes[i] / closes[i - 1]) - 1.0
        new_equity = equity[-1] * (1 + in_market * ret)
        equity.append(new_equity)

        # detect cross (entry/exit) at the OPEN of next day approximation
        prev_state = sma_fast[i - 1] > sma_slow[i - 1]
        now_state = sma_fast[i] > sma_slow[i]
        if now_state != prev_state:
            trades += 1
            if now_state and entry_equity is None:
                entry_equity = new_equity
            elif (not now_state) and entry_equity is not None:
                if new_equity > entry_equity:
                    wins += 1
                else:
                    loss_trades += 1
                entry_equity = None

        if new_equity > max_equity:
            max_equity = new_equity
        dd = (max_equity - new_equity) / max_equity
        if dd > max_drawdown:
            max_drawdown = dd

    total_return = (equity[-1] / initial_cash) - 1.0
    n_days = len(equity) - 1
    ann_ret = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0
    win_rate = wins / max(1, (wins + loss_trades))

    # Trim equity curve to keep payload small (last 365 points)
    curve = [{"date": d, "equity": round(e, 2)} for d, e in zip(dates, equity)][-365:]

    return {
        "symbol": symbol.upper(),
        "asof": dates[-1],
        "params": {"fast": fast, "slow": slow, "initial_cash": initial_cash},
        "metrics": {
            "total_return": round(total_return, 4),
            "annualized_return": round(ann_ret, 4),
            "max_drawdown": round(max_drawdown, 4),
            "trades": trades,
            "win_rate": round(win_rate, 4),
            "bars": len(dates),
        },
        "equity_curve": curve,
    }


app.include_router(trading)

# -------------------------
# Local dev runner
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
