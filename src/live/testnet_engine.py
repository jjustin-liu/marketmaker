"""Testnet engine — posts real orders to Binance Spot testnet.

Same role as src/live/order_manager.OrderManager, but instead of
simulating fills against the live book in Redis, this submits actual
LIMIT orders to testnet.binance.vision and learns about fills by
polling /api/v3/myTrades. Quotes are still computed by the local EV
strategy from the Redis book the data feed publishes.

Leaves quotes resting until the strategy target moves far enough to
justify a cancel/replace.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.live.binance_gateway import BinanceGateway, BinanceGatewayError
from src.live.metrics import record_fill, record_quote_refresh, set_position
from src.live.order_manager import StrategyProtocol
from src.live.risk_guard import RiskGuard
from src.strategy.naive_maker import Quote

logger = logging.getLogger("testnet_engine")

DEFAULT_POLL_SECONDS = 0.5
DEFAULT_FILLS_LOG = Path("data/fills.log")
BPS = 1e-4
DEFAULT_REQUOTE_THRESHOLD_BPS = 0.5


@dataclass
class EngineState:
    inventory: float = 0.0
    cash: float = 0.0
    pnl: float = 0.0
    current_bid_order_id: Optional[int] = None
    current_ask_order_id: Optional[int] = None
    current_bid_quote: Optional[Quote] = None
    current_ask_quote: Optional[Quote] = None
    last_trade_id: Optional[int] = None
    fills: List[dict] = field(default_factory=list)


class TestnetEngine:
    """Drives quote → place → cancel → poll-fills loop against testnet."""

    __test__ = False  # tell pytest not to collect this as a test class

    def __init__(
        self,
        redis_client: Any,
        gateway: BinanceGateway,
        strategy: StrategyProtocol,
        risk_guard: RiskGuard,
        *,
        symbol: str = "BTCUSDT",
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        fills_log_path: Path = DEFAULT_FILLS_LOG,
        requote_threshold_bps: float = DEFAULT_REQUOTE_THRESHOLD_BPS,
    ) -> None:
        self._redis = redis_client
        self._gateway = gateway
        self._strategy = strategy
        self._risk = risk_guard
        self._symbol = symbol.upper()
        self._poll_seconds = poll_seconds
        self._fills_log_path = fills_log_path
        if requote_threshold_bps < 0:
            raise ValueError("requote_threshold_bps must be nonnegative")
        self._requote_threshold_bps = requote_threshold_bps
        self.state = EngineState()
        self._fills_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- public loop ----

    async def step(self) -> bool:
        """One tick. Returns False to stop the loop."""
        book = self._read_book()
        if book is None:
            return True
        best_bid, best_ask, bids, asks = book
        mid = (best_bid + best_ask) / 2.0

        await self._poll_fills(mid)
        self._mark_to_market(mid)

        halt = self._risk.check(self.state.inventory, self.state.pnl)
        if halt is not None:
            logger.warning("risk halt: %s", halt)
            await self._cancel_open_orders()
            self._publish_position()
            return False

        try:
            bid_q, ask_q = self._strategy.quote_prices(
                mid_price=mid,
                inventory=self.state.inventory,
                bids=bids,
                asks=asks,
            )
        except Exception as exc:
            logger.exception("strategy raised; halting")
            self._risk.trip(f"strategy exception: {exc}")
            await self._cancel_open_orders()
            return False

        await self._refresh_orders_if_needed(bid_q, ask_q, mid)
        self._publish_current_quotes()
        self._publish_position()
        record_quote_refresh()
        set_position(self.state.inventory, self.state.pnl)
        return True

    async def run(self, max_seconds: Optional[float] = None) -> None:
        start = time.monotonic()
        while True:
            try:
                alive = await self.step()
            except Exception as exc:
                logger.exception("unhandled error in step")
                self._risk.trip(f"unhandled: {exc}")
                await self._cancel_open_orders()
                return
            if not alive:
                return
            if max_seconds is not None and (time.monotonic() - start) >= max_seconds:
                logger.info("run window elapsed")
                await self._cancel_open_orders()
                return
            await asyncio.sleep(self._poll_seconds)

    # ---- redis I/O ----

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

    def _publish_current_quotes(self) -> None:
        pipe = self._redis.pipeline()
        if self.state.current_bid_quote is not None:
            pipe.set(
                "strategy:target_bid",
                json.dumps({
                    "price": self.state.current_bid_quote.price,
                    "size": self.state.current_bid_quote.size,
                }),
            )
        if self.state.current_ask_quote is not None:
            pipe.set(
                "strategy:target_ask",
                json.dumps({
                    "price": self.state.current_ask_quote.price,
                    "size": self.state.current_ask_quote.size,
                }),
            )
        pipe.execute()

    def _publish_position(self) -> None:
        pipe = self._redis.pipeline()
        pipe.set("position:inventory", self.state.inventory)
        pipe.set("position:pnl", self.state.pnl)
        pipe.execute()

    # ---- exchange I/O ----

    async def _refresh_orders_if_needed(
        self, target_bid: Quote, target_ask: Quote, mid: float,
    ) -> None:
        threshold = self._requote_threshold_bps * BPS * mid
        replace_bid = self._should_replace(
            self.state.current_bid_quote, target_bid, threshold,
        )
        replace_ask = self._should_replace(
            self.state.current_ask_quote, target_ask, threshold,
        )
        if replace_bid and self.state.current_bid_order_id is not None:
            await self._cancel_order("current_bid_order_id")
        if replace_ask and self.state.current_ask_order_id is not None:
            await self._cancel_order("current_ask_order_id")
        if replace_bid:
            await self._place_bid(target_bid.price, target_bid.size)
        if replace_ask:
            await self._place_ask(target_ask.price, target_ask.size)

    @staticmethod
    def _should_replace(
        current: Optional[Quote], target: Quote, threshold: float,
    ) -> bool:
        if current is None:
            return True
        if current.size != target.size:
            return True
        return abs(current.price - target.price) > threshold

    async def _place_orders(self, bid_price: float, bid_size: float,
                            ask_price: float, ask_size: float) -> None:
        await self._place_bid(bid_price, bid_size)
        await self._place_ask(ask_price, ask_size)

    async def _place_bid(self, bid_price: float, bid_size: float) -> None:
        try:
            bid_ack = await self._gateway.post_order(
                symbol=self._symbol, side="BUY", order_type="LIMIT",
                quantity=bid_size, price=bid_price,
            )
            self.state.current_bid_order_id = int(bid_ack["orderId"])
            self.state.current_bid_quote = Quote(bid_price, bid_size)
            logger.info("placed bid %s @ %.2f size %.6f",
                        self.state.current_bid_order_id, bid_price, bid_size)
        except BinanceGatewayError as exc:
            logger.warning("bid place failed: %s", exc)

    async def _place_ask(self, ask_price: float, ask_size: float) -> None:
        try:
            ask_ack = await self._gateway.post_order(
                symbol=self._symbol, side="SELL", order_type="LIMIT",
                quantity=ask_size, price=ask_price,
            )
            self.state.current_ask_order_id = int(ask_ack["orderId"])
            self.state.current_ask_quote = Quote(ask_price, ask_size)
            logger.info("placed ask %s @ %.2f size %.6f",
                        self.state.current_ask_order_id, ask_price, ask_size)
        except BinanceGatewayError as exc:
            logger.warning("ask place failed: %s", exc)

    async def _cancel_open_orders(self) -> None:
        for attr in ("current_bid_order_id", "current_ask_order_id"):
            await self._cancel_order(attr)

    async def _cancel_order(self, order_id_attr: str) -> None:
        oid = getattr(self.state, order_id_attr)
        if oid is None:
            return
        try:
            await self._gateway.cancel_order(self._symbol, order_id=oid)
            logger.info("cancelled order %s", oid)
        except BinanceGatewayError as exc:
            logger.info("cancel %s skipped: %s", oid, exc)
        setattr(self.state, order_id_attr, None)
        quote_attr = (
            "current_bid_quote"
            if order_id_attr == "current_bid_order_id"
            else "current_ask_quote"
        )
        setattr(self.state, quote_attr, None)

    async def _poll_fills(self, mid: float) -> None:
        try:
            trades = await self._gateway.get_account_trades(
                self._symbol, from_id=self.state.last_trade_id,
            )
        except BinanceGatewayError as exc:
            logger.warning("trade poll failed: %s", exc)
            return
        for trade in trades:
            trade_id = int(trade["id"])
            if (self.state.last_trade_id is not None
                    and trade_id <= self.state.last_trade_id):
                continue
            self._book_fill(trade, mid)
            self.state.last_trade_id = trade_id

    def _book_fill(self, trade: dict, mid_at_fill: float) -> None:
        price = float(trade["price"])
        qty = float(trade["qty"])
        commission = float(trade.get("commission", 0.0))
        is_buyer = bool(trade.get("isBuyer", False))
        side = "buy" if is_buyer else "sell"
        notional = price * qty
        if is_buyer:
            self.state.inventory += qty
            self.state.cash -= notional
        else:
            self.state.inventory -= qty
            self.state.cash += notional
        self.state.cash -= commission
        fill = {
            "ts": time.time(),
            "trade_id": int(trade["id"]),
            "side": side,
            "price": price,
            "size": qty,
            "commission": commission,
            "mid_at_fill": mid_at_fill,
            "inventory_after": self.state.inventory,
        }
        self.state.fills.append(fill)
        with self._fills_log_path.open("a") as fh:
            fh.write(json.dumps(fill) + "\n")
        record_fill(side)
        logger.info("fill %s %.6f @ %.2f -> inv %.6f",
                    side, qty, price, self.state.inventory)

    def _mark_to_market(self, mid: float) -> None:
        self.state.pnl = self.state.cash + self.state.inventory * mid
