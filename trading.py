# trading.py
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any
import sqlite3, os, math, datetime as dt

router = APIRouter()

DB_PATH = os.getenv("EXEC_DB_PATH", "/tmp/executions.db")

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

    # misc (not used in sizing math but useful context)
    symbol: Optional[str] = None
    side: Optional[Literal["BUY","SELL"]] = None
    strategy: Optional[str] = None

class TradeSizeResponse(BaseModel):
    qty: int
    dollars: float
    mode_used: str
    details: Dict[str, Any]

@router.post("/size", response_model=TradeSizeResponse)
def size_trade(req: TradeSizeRequest):
    qty: int = 0
    details: Dict[str, Any] = {}

    if req.mode == "risk_pct":
        if req.risk_pct is None or req.stop_price is None:
            raise ValueError("risk_pct and stop_price required for risk_pct mode")
        risk_amount = req.equity * (req.risk_pct / 100.0)
        per_share_risk = abs(req.price - req.stop_price)
        qty = 0 if per_share_risk <= 0 else math.floor(risk_amount / per_share_risk)
        details.update(risk_amount=risk_amount, per_share_risk=per_share_risk)

    elif req.mode == "allocation_pct":
        if req.allocation_pct is None:
            raise ValueError("allocation_pct required for allocation_pct mode")
        alloc_cash = req.equity * (req.allocation_pct / 100.0)
        qty = math.floor(alloc_cash / req.price)
        details.update(allocation_cash=alloc_cash)

    elif req.mode == "fixed_cash":
        if req.cash_to_use is None:
            raise ValueError("cash_to_use required for fixed_cash mode")
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

def _slipped_price(price: float, side: str, slippage_bps: float) -> float:
    # Slippage moves against you: BUY pays more; SELL receives less.
    sign = 1 if side == "BUY" else -1
    return round(price * (1 + sign * (slippage_bps / 10_000.0)), 6)

@router.post("/execute", response_model=ExecutionRecord)
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

    return ExecutionRecord(id=exec_id, fill_price=fill_price, **req.dict())

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

@router.get("/executions", response_model=List[ExecQueryOut])
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
