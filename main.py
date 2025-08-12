
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

app.include_router(trading)

# -------------------------
# Local dev runner
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
