"""Tests for the testnet engine.

The gateway is faked out — we never hit the network. Focus is on
the engine's bookkeeping: orders placed and cancelled in the right
order, trade polling drives inventory/PnL correctly, risk halts stop
the loop.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.live.risk_guard import RiskConfig, RiskGuard
from src.live.testnet_engine import TestnetEngine
from src.strategy.naive_maker import Quote


# ---------- fakes ----------


class FakeRedisPipeline:
    def __init__(self, store: Dict[str, str]) -> None:
        self._store = store
        self._ops: List[Tuple[str, tuple]] = []

    def set(self, key: str, value: Any) -> "FakeRedisPipeline":
        self._ops.append(("set", (key, value)))
        return self

    def execute(self) -> None:
        for op, args in self._ops:
            if op == "set":
                key, value = args
                self._store[key] = str(value)
        self._ops.clear()


class FakeRedis:
    def __init__(self) -> None:
        self.store: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = str(value)

    def pipeline(self) -> FakeRedisPipeline:
        return FakeRedisPipeline(self.store)


class FakeGateway:
    """Records the calls the engine makes; returns scripted responses."""

    def __init__(self) -> None:
        self.posts: List[Dict[str, Any]] = []
        self.cancels: List[int] = []
        self.trades_to_return: List[List[Dict[str, Any]]] = []
        self._next_order_id = 1000

    async def post_order(self, **params: Any) -> Dict[str, Any]:
        self.posts.append(params)
        order_id = self._next_order_id
        self._next_order_id += 1
        return {"orderId": order_id, "status": "NEW"}

    async def cancel_order(self, symbol: str, *,
                           order_id: int) -> Dict[str, Any]:
        self.cancels.append(order_id)
        return {"orderId": order_id, "status": "CANCELED"}

    async def get_account_trades(self, symbol: str, *,
                                 from_id: Optional[int] = None,
                                 limit: int = 100,
                                 ) -> List[Dict[str, Any]]:
        if not self.trades_to_return:
            return []
        return self.trades_to_return.pop(0)


class StubStrategy:
    def __init__(self, bid_price: float, ask_price: float,
                 size: float = 0.001) -> None:
        self._bid = Quote(price=bid_price, size=size)
        self._ask = Quote(price=ask_price, size=size)

    def quote_prices(self, **_: Any) -> Tuple[Quote, Quote]:
        return self._bid, self._ask


def _seed_book(r: FakeRedis, best_bid: float, best_ask: float) -> None:
    r.set("lob:best_bid", best_bid)
    r.set("lob:best_ask", best_ask)
    r.set("lob:levels:bids", json.dumps([[best_bid, 1.0]]))
    r.set("lob:levels:asks", json.dumps([[best_ask, 1.0]]))


# ---------- helpers ----------


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _make_engine(redis: FakeRedis, gateway: FakeGateway,
                 strategy: StubStrategy,
                 risk: Optional[RiskGuard] = None,
                 tmp_path: Optional[Path] = None) -> TestnetEngine:
    if risk is None:
        risk = RiskGuard(RiskConfig(max_position=1.0, max_drawdown=1e9))
    log = (tmp_path / "fills.log") if tmp_path else Path("/tmp/_t_fills.log")
    return TestnetEngine(
        redis_client=redis, gateway=gateway, strategy=strategy,
        risk_guard=risk, fills_log_path=log,
    )


# ---------- tests ----------


def test_step_places_both_sides_when_book_ready(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, 100.0, 101.0)
    gw = FakeGateway()
    eng = _make_engine(r, gw, StubStrategy(100.5, 101.5),
                       tmp_path=tmp_path)
    alive = _run(eng.step())
    assert alive is True
    assert len(gw.posts) == 2
    assert {p["side"] for p in gw.posts} == {"BUY", "SELL"}


def test_step_skips_when_book_empty(tmp_path: Path) -> None:
    r = FakeRedis()
    gw = FakeGateway()
    eng = _make_engine(r, gw, StubStrategy(100.0, 101.0), tmp_path=tmp_path)
    alive = _run(eng.step())
    assert alive is True
    assert gw.posts == []


def test_second_step_cancels_previous_orders(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, 100.0, 101.0)
    gw = FakeGateway()
    eng = _make_engine(r, gw, StubStrategy(100.5, 101.5),
                       tmp_path=tmp_path)
    _run(eng.step())
    _run(eng.step())
    assert len(gw.cancels) == 2     # one bid, one ask cancelled
    assert len(gw.posts) == 4       # two ticks × two sides


def test_buy_fill_updates_inventory(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, 100.0, 101.0)
    gw = FakeGateway()
    gw.trades_to_return.append([{
        "id": 1, "price": "100.5", "qty": "0.001",
        "commission": "0.0001", "isBuyer": True,
    }])
    eng = _make_engine(r, gw, StubStrategy(100.5, 101.5),
                       tmp_path=tmp_path)
    _run(eng.step())
    assert eng.state.inventory == pytest.approx(0.001)
    assert eng.state.last_trade_id == 1
    assert any(f["side"] == "buy" for f in eng.state.fills)


def test_sell_fill_decrements_inventory(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, 100.0, 101.0)
    gw = FakeGateway()
    gw.trades_to_return.append([{
        "id": 2, "price": "101.5", "qty": "0.001",
        "commission": "0.0", "isBuyer": False,
    }])
    eng = _make_engine(r, gw, StubStrategy(100.5, 101.5),
                       tmp_path=tmp_path)
    _run(eng.step())
    assert eng.state.inventory == pytest.approx(-0.001)


def test_dedup_trades_by_last_trade_id(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, 100.0, 101.0)
    gw = FakeGateway()
    gw.trades_to_return.extend([
        [{"id": 5, "price": "100.0", "qty": "0.001",
          "commission": "0.0", "isBuyer": True}],
        [{"id": 5, "price": "100.0", "qty": "0.001",
          "commission": "0.0", "isBuyer": True}],
    ])
    eng = _make_engine(r, gw, StubStrategy(100.5, 101.5),
                       tmp_path=tmp_path)
    _run(eng.step())
    _run(eng.step())
    assert len(eng.state.fills) == 1


def test_risk_halt_stops_loop(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, 100.0, 101.0)
    gw = FakeGateway()
    gw.trades_to_return.append([{
        "id": 1, "price": "100.0", "qty": "0.05",
        "commission": "0.0", "isBuyer": True,
    }])
    risk = RiskGuard(RiskConfig(max_position=0.01, max_drawdown=1e9))
    eng = _make_engine(r, gw, StubStrategy(100.0, 101.0),
                       risk=risk, tmp_path=tmp_path)
    alive = _run(eng.step())
    assert alive is False
    assert risk.halted


def test_strategy_exception_trips_risk_guard(tmp_path: Path) -> None:
    class BoomStrategy:
        def quote_prices(self, **_: Any) -> Tuple[Quote, Quote]:
            raise RuntimeError("kaboom")

    r = FakeRedis()
    _seed_book(r, 100.0, 101.0)
    gw = FakeGateway()
    risk = RiskGuard()
    eng = _make_engine(r, gw, BoomStrategy(), risk=risk, tmp_path=tmp_path)
    alive = _run(eng.step())
    assert alive is False
    assert risk.halted


def test_fill_logged_to_disk(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, 100.0, 101.0)
    gw = FakeGateway()
    gw.trades_to_return.append([{
        "id": 9, "price": "100.5", "qty": "0.001",
        "commission": "0.0", "isBuyer": True,
    }])
    log = tmp_path / "fills.log"
    eng = TestnetEngine(
        redis_client=r, gateway=gw,
        strategy=StubStrategy(100.5, 101.5),
        risk_guard=RiskGuard(RiskConfig(max_position=1.0, max_drawdown=1e9)),
        fills_log_path=log,
    )
    _run(eng.step())
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["trade_id"] == 9


def test_target_quotes_published(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, 100.0, 101.0)
    gw = FakeGateway()
    eng = _make_engine(r, gw, StubStrategy(100.5, 101.5),
                       tmp_path=tmp_path)
    _run(eng.step())
    assert "strategy:target_bid" in r.store
    obj = json.loads(r.store["strategy:target_bid"])
    assert obj["price"] == pytest.approx(100.5)
