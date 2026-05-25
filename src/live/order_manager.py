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


class StrategyProtocol(Protocol):
    """Anything with an EVMaker-shaped quote_prices signature."""

    def quote_prices(self, *args: Any, **kwargs: Any) -> Tuple[Quote, Quote]:
        ...


@dataclass
class ManagerState:
    """In-memory book of paper-trading state."""

    inventory: float = 0.0
    cash: float = 0.0
    pnl: float = 0.0
    open_bid: Optional[Quote] = None
    open_ask: Optional[Quote] = None
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
    ) -> None:
        self._redis = redis_client
        self._strategy = strategy
        self._risk = risk_guard
        self._poll_seconds = poll_seconds
        self._fills_log_path = fills_log_path
        self.state = ManagerState()
        self._fills_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- public loop control ----

    def step(self) -> bool:
        """Run exactly one tick. Returns False if the loop should stop."""
        book = self._read_book()
        if book is None:
            return True  # feed not ready yet, keep waiting
        best_bid, best_ask, bids, asks = book
        mid = (best_bid + best_ask) / 2.0

        self._simulate_fills(best_bid, best_ask, mid)
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

        self.state.open_bid = bid_q
        self.state.open_ask = ask_q
        self._publish_quotes(bid_q, ask_q)
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

    def _simulate_fills(self, best_bid: float, best_ask: float,
                        mid: float) -> None:
        """Strict-cross fill rule, same model as the backtest."""
        s = self.state
        if s.open_bid is not None and best_ask < s.open_bid.price:
            self._book_fill("buy", s.open_bid.price, s.open_bid.size, mid)
            s.open_bid = None
        if s.open_ask is not None and best_bid > s.open_ask.price:
            self._book_fill("sell", s.open_ask.price, s.open_ask.size, mid)
            s.open_ask = None

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
        fill = {
            "ts": time.time(),
            "side": side,
            "price": price,
            "size": size,
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

    def _publish_quotes(self, bid: Quote, ask: Quote) -> None:
        pipe = self._redis.pipeline()
        pipe.set("strategy:target_bid",
                 json.dumps({"price": bid.price, "size": bid.size}))
        pipe.set("strategy:target_ask",
                 json.dumps({"price": ask.price, "size": ask.size}))
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
