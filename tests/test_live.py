"""Tests for the paper-mode order manager and risk guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.live.order_manager import OrderManager
from src.live.risk_guard import RiskConfig, RiskGuard
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
    """Just enough Redis surface for the order manager."""

    def __init__(self) -> None:
        self.store: Dict[str, str] = {}
        # Stream: stream_key -> list of (id, fields_dict)
        self.streams: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}
        self._seq = 0

    def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = str(value)

    def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                n += 1
        return n

    def pipeline(self) -> FakeRedisPipeline:
        return FakeRedisPipeline(self.store)

    def xadd(self, stream: str, fields: Dict[str, str],
             maxlen: Optional[int] = None,
             approximate: bool = False) -> str:
        self._seq += 1
        entry_id = f"{self._seq}-0"
        s = self.streams.setdefault(stream, [])
        s.append((entry_id, {k: str(v) for k, v in fields.items()}))
        if maxlen is not None and len(s) > maxlen:
            del s[: len(s) - maxlen]
        return entry_id

    def xread(self, streams: Dict[str, str],
              block: Optional[int] = None,
              count: int = 100) -> List[Tuple[str, List[Tuple[str, Dict[str, str]]]]]:
        def _seq(entry_id: str) -> int:
            try:
                return int(entry_id.split("-", 1)[0])
            except (ValueError, AttributeError):
                return 0

        results: List[Tuple[str, List[Tuple[str, Dict[str, str]]]]] = []
        for stream, after_id in streams.items():
            entries = self.streams.get(stream, [])
            if after_id == "$":
                continue
            after_seq = _seq(after_id)
            new = [(eid, f) for eid, f in entries if _seq(eid) > after_seq]
            if new:
                results.append((stream, new[:count]))
        return results


def _push_trade(redis: FakeRedis, price: float, qty: float, side: str) -> None:
    redis.xadd(
        "trades:stream",
        {"price": str(price), "qty": str(qty),
         "side": side, "timestamp": "0"},
    )


class StubStrategy:
    """Returns whatever quotes the test wires in."""

    def __init__(self, bid_price: float, ask_price: float,
                 size: float = 0.001) -> None:
        self._bid = Quote(price=bid_price, size=size)
        self._ask = Quote(price=ask_price, size=size)

    def quote_prices(self, **_kwargs: Any) -> Tuple[Quote, Quote]:
        return self._bid, self._ask


class SequenceStrategy:
    def __init__(self, quotes: List[Tuple[float, float]],
                 size: float = 0.001) -> None:
        self._quotes = quotes
        self._size = size
        self._idx = 0

    def quote_prices(self, **_kwargs: Any) -> Tuple[Quote, Quote]:
        i = min(self._idx, len(self._quotes) - 1)
        self._idx += 1
        bid, ask = self._quotes[i]
        return Quote(bid, self._size), Quote(ask, self._size)


def _seed_book(redis: FakeRedis, best_bid: float, best_ask: float) -> None:
    redis.set("lob:best_bid", best_bid)
    redis.set("lob:best_ask", best_ask)
    redis.set("lob:levels:bids", json.dumps([[best_bid, 1.0]]))
    redis.set("lob:levels:asks", json.dumps([[best_ask, 1.0]]))


# ---------- risk guard ----------


def test_risk_config_rejects_nonpositive_caps() -> None:
    with pytest.raises(ValueError):
        RiskConfig(max_position=0.0)
    with pytest.raises(ValueError):
        RiskConfig(max_drawdown=-1.0)


def test_risk_guard_passes_under_caps() -> None:
    g = RiskGuard(RiskConfig(max_position=0.1, max_drawdown=100.0))
    assert g.check(inventory=0.05, pnl=10.0) is None
    assert not g.halted


def test_risk_guard_halts_on_max_position() -> None:
    g = RiskGuard(RiskConfig(max_position=0.01, max_drawdown=100.0))
    reason = g.check(inventory=0.02, pnl=0.0)
    assert reason is not None and "max_position" in reason
    assert g.halted


def test_risk_guard_halts_on_max_drawdown() -> None:
    g = RiskGuard(RiskConfig(max_position=1.0, max_drawdown=10.0))
    g.check(inventory=0.0, pnl=50.0)   # peak
    reason = g.check(inventory=0.0, pnl=30.0)  # dd = 20 > 10
    assert reason is not None and "max_drawdown" in reason


def test_risk_guard_short_inventory_also_triggers() -> None:
    g = RiskGuard(RiskConfig(max_position=0.01, max_drawdown=100.0))
    assert g.check(inventory=-0.02, pnl=0.0) is not None


def test_risk_guard_sticky_after_first_halt() -> None:
    g = RiskGuard(RiskConfig(max_position=0.01, max_drawdown=100.0))
    g.check(inventory=0.02, pnl=0.0)
    first_reason = g.halt_reason
    g.check(inventory=0.0, pnl=0.0)
    assert g.halt_reason == first_reason


def test_risk_guard_trip_forces_halt() -> None:
    g = RiskGuard()
    g.trip("manual")
    assert g.halted and g.halt_reason == "manual"


# ---------- paper fills ----------


def test_paper_bid_fills_when_taker_sells_consume_queue(tmp_path: Path) -> None:
    """Bid at empty level (queue=0) → any sell trade at our price fills us."""
    r = FakeRedis()
    _seed_book(r, best_bid=100.0, best_ask=101.0)
    mgr = OrderManager(
        redis_client=r,
        strategy=StubStrategy(bid_price=100.5, ask_price=101.5),
        risk_guard=RiskGuard(RiskConfig(max_position=1.0, max_drawdown=1e9)),
        fills_log_path=tmp_path / "fills.log",
    )
    mgr.step()  # post quotes
    # Our bid is at 100.5 — empty level → queue_ahead_bid = 0.
    # Any taker sell at >= 100.5 should fill us immediately.
    _push_trade(r, price=100.5, qty=0.01, side="sell")
    mgr.step()
    assert any(f["side"] == "buy" for f in mgr.state.fills)
    assert mgr.state.inventory == pytest.approx(0.001)


def test_paper_ask_fills_when_taker_buys_consume_queue(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, best_bid=100.0, best_ask=101.0)
    mgr = OrderManager(
        redis_client=r,
        # Quote inside the spread on both sides — empty levels, queue=0.
        strategy=StubStrategy(bid_price=100.5, ask_price=100.7),
        risk_guard=RiskGuard(RiskConfig(max_position=1.0, max_drawdown=1e9)),
        fills_log_path=tmp_path / "fills.log",
    )
    mgr.step()
    _push_trade(r, price=100.7, qty=0.01, side="buy")
    mgr.step()
    assert any(f["side"] == "sell" for f in mgr.state.fills)
    assert mgr.state.inventory == pytest.approx(-0.001)


def test_queue_ahead_blocks_fill_until_consumed(tmp_path: Path) -> None:
    """Bid at a level with 1.0 of existing volume → queue_ahead=1.0.
    Small taker trades shouldn't fill us until they cumulatively eat the queue.
    """
    r = FakeRedis()
    # Bid at 100.0 has 1.0 size sitting in front of us when we join it.
    r.set("lob:best_bid", 100.0)
    r.set("lob:best_ask", 101.0)
    r.set("lob:levels:bids", json.dumps([[100.0, 1.0]]))
    r.set("lob:levels:asks", json.dumps([[101.0, 1.0]]))
    mgr = OrderManager(
        redis_client=r,
        strategy=StubStrategy(bid_price=100.0, ask_price=101.0),
        risk_guard=RiskGuard(RiskConfig(max_position=1.0, max_drawdown=1e9)),
        fills_log_path=tmp_path / "fills.log",
    )
    mgr.step()
    assert mgr.state.queue_ahead_bid == pytest.approx(1.0)

    _push_trade(r, price=100.0, qty=0.4, side="sell")
    _push_trade(r, price=100.0, qty=0.4, side="sell")
    mgr.step()
    assert mgr.state.fills == []
    assert mgr.state.queue_ahead_bid == pytest.approx(0.2)

    _push_trade(r, price=100.0, qty=0.5, side="sell")
    mgr.step()
    assert any(f["side"] == "buy" for f in mgr.state.fills)


def test_paper_fill_applies_signed_fee_bps(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, best_bid=100.0, best_ask=101.0)
    mgr = OrderManager(
        redis_client=r,
        strategy=StubStrategy(bid_price=100.5, ask_price=101.5, size=0.01),
        risk_guard=RiskGuard(RiskConfig(max_position=1.0, max_drawdown=1e9)),
        fills_log_path=tmp_path / "fills.log",
        fee_bps=1.0,
    )
    mgr.step()
    _push_trade(r, price=100.5, qty=0.05, side="sell")
    mgr.step()

    notional = 100.5 * 0.01
    assert mgr.state.cash == pytest.approx(-notional - 0.0001 * notional)
    assert mgr.state.fills[0]["fee"] == pytest.approx(0.0001 * notional)


def test_paper_quotes_persist_inside_requote_threshold(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, best_bid=100.0, best_ask=101.0)
    mgr = OrderManager(
        redis_client=r,
        strategy=SequenceStrategy([(100.50, 101.50), (100.504, 101.496)]),
        risk_guard=RiskGuard(),
        fills_log_path=tmp_path / "fills.log",
        requote_threshold_bps=1.0,
    )
    mgr.step()
    mgr.step()

    bid_obj = json.loads(r.store["strategy:target_bid"])
    ask_obj = json.loads(r.store["strategy:target_ask"])
    assert bid_obj["price"] == pytest.approx(100.50)
    assert ask_obj["price"] == pytest.approx(101.50)


def test_paper_quotes_replace_outside_requote_threshold(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, best_bid=100.0, best_ask=101.0)
    mgr = OrderManager(
        redis_client=r,
        strategy=SequenceStrategy([(100.50, 101.50), (100.60, 101.40)]),
        risk_guard=RiskGuard(),
        fills_log_path=tmp_path / "fills.log",
        requote_threshold_bps=1.0,
    )
    mgr.step()
    mgr.step()

    bid_obj = json.loads(r.store["strategy:target_bid"])
    ask_obj = json.loads(r.store["strategy:target_ask"])
    assert bid_obj["price"] == pytest.approx(100.60)
    assert ask_obj["price"] == pytest.approx(101.40)


def test_risk_halt_stops_loop(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, best_bid=100.0, best_ask=101.0)
    mgr = OrderManager(
        redis_client=r,
        strategy=StubStrategy(bid_price=100.5, ask_price=101.5, size=0.05),
        risk_guard=RiskGuard(RiskConfig(max_position=0.01, max_drawdown=1e9)),
        fills_log_path=tmp_path / "fills.log",
    )
    assert mgr.step() is True                        # post quotes, no fills
    _push_trade(r, price=100.5, qty=0.1, side="sell")  # buy fill triggered
    assert mgr.step() is False                       # 0.05 > 0.01 cap, halt
    assert mgr._risk.halted


def test_no_fill_when_no_taker_trades(tmp_path: Path) -> None:
    """With no trades in the stream, no fills regardless of book moves."""
    r = FakeRedis()
    _seed_book(r, best_bid=100.0, best_ask=101.0)
    mgr = OrderManager(
        redis_client=r,
        strategy=StubStrategy(bid_price=100.5, ask_price=101.5),
        risk_guard=RiskGuard(),
        fills_log_path=tmp_path / "fills.log",
    )
    mgr.step()
    _seed_book(r, best_bid=100.1, best_ask=100.9)    # book tightens, but no trades
    mgr.step()
    assert mgr.state.fills == []


def test_step_returns_true_when_feed_not_ready(tmp_path: Path) -> None:
    """Empty Redis → manager waits, doesn't crash."""
    r = FakeRedis()
    mgr = OrderManager(
        redis_client=r,
        strategy=StubStrategy(bid_price=100.0, ask_price=101.0),
        risk_guard=RiskGuard(),
        fills_log_path=tmp_path / "fills.log",
    )
    assert mgr.step() is True
    assert mgr.state.fills == []


def test_strategy_exception_trips_risk_guard(tmp_path: Path) -> None:
    class BoomStrategy:
        def quote_prices(self, **_: Any) -> Tuple[Quote, Quote]:
            raise RuntimeError("kaboom")

    r = FakeRedis()
    _seed_book(r, best_bid=100.0, best_ask=101.0)
    mgr = OrderManager(
        redis_client=r,
        strategy=BoomStrategy(),
        risk_guard=RiskGuard(),
        fills_log_path=tmp_path / "fills.log",
    )
    assert mgr.step() is False
    assert mgr._risk.halted
    assert "kaboom" in (mgr._risk.halt_reason or "")


def test_fill_is_written_to_log(tmp_path: Path) -> None:
    log = tmp_path / "fills.log"
    r = FakeRedis()
    _seed_book(r, best_bid=100.0, best_ask=101.0)
    mgr = OrderManager(
        redis_client=r,
        strategy=StubStrategy(bid_price=100.5, ask_price=101.5),
        risk_guard=RiskGuard(RiskConfig(max_position=1.0, max_drawdown=1e9)),
        fills_log_path=log,
    )
    mgr.step()
    _push_trade(r, price=100.5, qty=0.01, side="sell")
    mgr.step()
    contents = log.read_text().strip().splitlines()
    assert len(contents) == 1
    record = json.loads(contents[0])
    assert record["side"] == "buy"
    assert record["price"] == pytest.approx(100.5)


def test_target_quotes_published_to_redis(tmp_path: Path) -> None:
    r = FakeRedis()
    _seed_book(r, best_bid=100.0, best_ask=101.0)
    mgr = OrderManager(
        redis_client=r,
        strategy=StubStrategy(bid_price=100.5, ask_price=101.5),
        risk_guard=RiskGuard(),
        fills_log_path=tmp_path / "fills.log",
    )
    mgr.step()
    assert "strategy:target_bid" in r.store
    bid_obj = json.loads(r.store["strategy:target_bid"])
    assert bid_obj["price"] == pytest.approx(100.5)
