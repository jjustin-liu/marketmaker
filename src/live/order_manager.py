"""Paper-mode order manager.

Polls the live order book in Redis (written by the data feed),
asks the strategy for fresh quotes, publishes them to
strategy:target_bid / strategy:target_ask, and simulates fills
whenever the live opposite-side best price strictly crosses our
quote. Tracks cash + inventory, writes position:inventory and
position:pnl to Redis, appends each fill to data/fills.log.

No real orders are sent. No exchange API keys are used.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple

from src.live.metrics import record_fill, record_quote_refresh, set_position
from src.live.risk_guard import RiskGuard
from src.strategy.naive_maker import Quote

logger = logging.getLogger("order_manager")

DEFAULT_POLL_SECONDS = 0.1
DEFAULT_FILLS_LOG = Path("data/fills.log")
BPS = 1e-4
DEFAULT_REQUOTE_THRESHOLD_BPS = 0.5
TRADE_STREAM_KEY = "trades:stream"


class StrategyProtocol(Protocol):
    """Anything with an EVMaker-shaped quote_prices signature."""

    def quote_prices(self, *args: Any, **kwargs: Any) -> Tuple[Quote, Quote]:
        ...


def _decode(value: Any) -> str:
    """Redis returns bytes by default; decode_responses=True returns str."""
    if isinstance(value, bytes):
        return value.decode()
    return str(value) if value is not None else ""


def _queue_ahead_for_bid(
    our_price: float, bids: List[Tuple[float, float]],
) -> float:
    """Volume that must trade before our resting bid gets hit.

    Includes (a) all visible qty at bid prices strictly higher than
    ours (those have price priority and fill first) and (b) the qty
    already at our exact price level (we joined the back of the queue).
    """
    total = 0.0
    for price, qty in bids:
        if price > our_price:
            total += qty
        elif price == our_price:
            total += qty
    return total


def _queue_ahead_for_ask(
    our_price: float, asks: List[Tuple[float, float]],
) -> float:
    """Volume that must trade before our resting ask gets hit."""
    total = 0.0
    for price, qty in asks:
        if price < our_price:
            total += qty
        elif price == our_price:
            total += qty
    return total


@dataclass
class ManagerState:
    """In-memory book of paper-trading state."""

    inventory: float = 0.0
    cash: float = 0.0
    pnl: float = 0.0
    open_bid: Optional[Quote] = None
    open_ask: Optional[Quote] = None
    # Queue-ahead size: total volume that must trade through our level
    # before we get filled. Snapshotted at post time as the visible
    # bid/ask qty at our price (plus higher-priority levels). Decremented
    # by each observed taker trade at or through our price.
    queue_ahead_bid: float = 0.0
    queue_ahead_ask: float = 0.0
    fills: List[dict] = field(default_factory=list)


class OrderManager:
    """Drives the paper-trading loop. One instance per live run."""

    def __init__(
        self,
        redis_client: Any,
        strategy: StrategyProtocol,
        risk_guard: RiskGuard,
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        fills_log_path: Path = DEFAULT_FILLS_LOG,
        fee_bps: float = 0.0,
        requote_threshold_bps: float = DEFAULT_REQUOTE_THRESHOLD_BPS,
    ) -> None:
        self._redis = redis_client
        self._strategy = strategy
        self._risk = risk_guard
        self._poll_seconds = poll_seconds
        self._fills_log_path = fills_log_path
        if requote_threshold_bps < 0:
            raise ValueError("requote_threshold_bps must be nonnegative")
        self._fee_bps = fee_bps
        self._requote_threshold_bps = requote_threshold_bps
        self.state = ManagerState()
        # Watermark into the trades:stream Redis stream. Initialized to
        # "$" the first time we read so we only see trades posted after
        # we start; otherwise we'd replay history on startup.
        self._last_trade_id: Optional[str] = None
        self._fills_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- public loop control ----

    def step(self) -> bool:
        """Run exactly one tick. Returns False if the loop should stop."""
        book = self._read_book()
        if book is None:
            return True  # feed not ready yet, keep waiting
        best_bid, best_ask, bids, asks = book
        mid = (best_bid + best_ask) / 2.0

        self._simulate_fills_queue_aware(mid)
        self._mark_to_market(mid)

        halt = self._risk.check(self.state.inventory, self.state.pnl)
        if halt is not None:
            logger.warning("risk halt: %s", halt)
            self._cancel_all()
            self._publish_position()
            return False

        try:
            bid_q, ask_q = self._strategy.quote_prices(
                mid_price=mid,
                inventory=self.state.inventory,
                bids=bids,
                asks=asks,
            )
        except Exception as exc:  # strategy bug shouldn't kill the loop silently
            logger.exception("strategy raised; halting: %s", exc)
            self._risk.trip(f"strategy exception: {exc}")
            self._cancel_all()
            return False

        self._refresh_quotes_if_needed(bid_q, ask_q, mid, bids, asks)
        self._publish_open_quotes()
        self._publish_position()
        record_quote_refresh()
        set_position(self.state.inventory, self.state.pnl)
        return True

    def run(self, max_seconds: Optional[float] = None) -> None:
        """Poll loop. Stops on risk halt or after max_seconds."""
        start = time.monotonic()
        while True:
            try:
                alive = self.step()
            except Exception as exc:
                logger.exception("unhandled error in step")
                self._risk.trip(f"unhandled: {exc}")
                self._cancel_all()
                return
            if not alive:
                return
            if max_seconds is not None and (time.monotonic() - start) >= max_seconds:
                logger.info("run window elapsed")
                self._cancel_all()
                return
            time.sleep(self._poll_seconds)

    # ---- internals ----

    def _read_book(self) -> Optional[Tuple[float, float,
                                           List[Tuple[float, float]],
                                           List[Tuple[float, float]]]]:
        best_bid_raw = self._redis.get("lob:best_bid")
        best_ask_raw = self._redis.get("lob:best_ask")
        bids_raw = self._redis.get("lob:levels:bids")
        asks_raw = self._redis.get("lob:levels:asks")
        if not (best_bid_raw and best_ask_raw and bids_raw and asks_raw):
            return None
        best_bid = float(best_bid_raw)
        best_ask = float(best_ask_raw)
        bids = [(float(p), float(q)) for p, q in json.loads(bids_raw)]
        asks = [(float(p), float(q)) for p, q in json.loads(asks_raw)]
        return best_bid, best_ask, bids, asks

    def _simulate_fills_queue_aware(self, mid: float) -> None:
        """Queue-aware fill rule.

        Reads taker trades from the trades:stream Redis stream since the
        last seen ID. For each trade whose price reaches our resting
        quote, decrements queue_ahead. When queue_ahead <= 0, books a
        fill at our quote price.

        A market sell (aggressor='sell') trade reports the bid price
        it executed against. If that price is >= our_bid_price, the
        trade consumed bids at our level or higher-priority levels,
        which sit ahead of us in priority — so we decrement queue_ahead.
        Symmetric for asks: market-buy trades at price <= our_ask_price
        consume our queue-ahead.
        """
        s = self.state
        events = self._read_new_trades()
        for ev in events:
            price = float(ev["price"])
            qty = float(ev["qty"])
            side = ev["side"]
            if (side == "sell" and s.open_bid is not None
                    and price >= s.open_bid.price):
                s.queue_ahead_bid -= qty
                if s.queue_ahead_bid <= 0:
                    self._book_fill("buy", s.open_bid.price,
                                    s.open_bid.size, mid)
                    s.open_bid = None
                    s.queue_ahead_bid = 0.0
            elif (side == "buy" and s.open_ask is not None
                    and price <= s.open_ask.price):
                s.queue_ahead_ask -= qty
                if s.queue_ahead_ask <= 0:
                    self._book_fill("sell", s.open_ask.price,
                                    s.open_ask.size, mid)
                    s.open_ask = None
                    s.queue_ahead_ask = 0.0

    def _read_new_trades(self) -> List[dict]:
        """Return list of trade events posted since our last watermark.

        On the first call, advances the watermark past every entry
        currently in the stream and returns empty — we only care about
        trades that occur after the manager starts running.
        """
        try:
            if self._last_trade_id is None:
                self._last_trade_id = "0-0"
                resp = self._redis.xread(
                    {TRADE_STREAM_KEY: "0-0"}, block=None, count=10000,
                )
                if resp:
                    for _stream, entries in resp:
                        for entry_id, _fields in entries:
                            self._last_trade_id = _decode(entry_id)
                return []
            resp = self._redis.xread(
                {TRADE_STREAM_KEY: self._last_trade_id}, block=None,
                count=500,
            )
        except Exception as exc:  # redis hiccup must not kill the loop
            logger.warning("xread failed: %s", exc)
            return []
        if not resp:
            return []
        events: List[dict] = []
        for _stream, entries in resp:
            for entry_id, fields in entries:
                events.append({
                    "id": _decode(entry_id),
                    "price": _decode(fields.get(b"price") or fields.get("price")),
                    "qty": _decode(fields.get(b"qty") or fields.get("qty")),
                    "side": _decode(fields.get(b"side") or fields.get("side")),
                })
                self._last_trade_id = _decode(entry_id)
        return events

    def _book_fill(self, side: str, price: float, size: float,
                   mid_at_fill: float) -> None:
        s = self.state
        notional = price * size
        if side == "buy":
            s.inventory += size
            s.cash -= notional
        else:
            s.inventory -= size
            s.cash += notional
        fee = self._fee_bps * BPS * notional
        s.cash -= fee
        fill = {
            "ts": time.time(),
            "side": side,
            "price": price,
            "size": size,
            "fee": fee,
            "mid_at_fill": mid_at_fill,
            "inventory_after": s.inventory,
        }
        s.fills.append(fill)
        with self._fills_log_path.open("a") as fh:
            fh.write(json.dumps(fill) + "\n")
        record_fill(side)
        logger.info("fill %s %.6f @ %.2f -> inv %.6f",
                    side, size, price, s.inventory)

    def _mark_to_market(self, mid: float) -> None:
        self.state.pnl = self.state.cash + self.state.inventory * mid

    def _refresh_quotes_if_needed(
        self, target_bid: Quote, target_ask: Quote, mid: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> None:
        threshold = self._requote_threshold_bps * BPS * mid
        if self._should_replace(self.state.open_bid, target_bid, threshold):
            self.state.open_bid = target_bid
            self.state.queue_ahead_bid = _queue_ahead_for_bid(
                target_bid.price, bids,
            )
        if self._should_replace(self.state.open_ask, target_ask, threshold):
            self.state.open_ask = target_ask
            self.state.queue_ahead_ask = _queue_ahead_for_ask(
                target_ask.price, asks,
            )

    @staticmethod
    def _should_replace(
        current: Optional[Quote], target: Quote, threshold: float,
    ) -> bool:
        if current is None:
            return True
        if current.size != target.size:
            return True
        return abs(current.price - target.price) > threshold

    def _publish_open_quotes(self) -> None:
        pipe = self._redis.pipeline()
        if self.state.open_bid is not None:
            pipe.set("strategy:target_bid",
                     json.dumps({"price": self.state.open_bid.price,
                                 "size": self.state.open_bid.size}))
        if self.state.open_ask is not None:
            pipe.set("strategy:target_ask",
                     json.dumps({"price": self.state.open_ask.price,
                                 "size": self.state.open_ask.size}))
        pipe.execute()

    def _publish_position(self) -> None:
        pipe = self._redis.pipeline()
        pipe.set("position:inventory", self.state.inventory)
        pipe.set("position:pnl", self.state.pnl)
        pipe.execute()

    def _cancel_all(self) -> None:
        self.state.open_bid = None
        self.state.open_ask = None
        self._redis.delete("strategy:target_bid", "strategy:target_ask")
