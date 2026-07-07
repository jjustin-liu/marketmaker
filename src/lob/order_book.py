"""Pure-Python limit order book.

Mirrors the C++ MatchEngine interface. Used in backtests and unit tests
where readability matters more than throughput.

Bids stored in a SortedDict keyed by negated price so index-0 = best bid.
Asks stored in a SortedDict keyed by price (ascending) so index-0 = best ask.
Both best_bid() and best_ask() are O(log n) instead of O(n).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from sortedcontainers import SortedDict


class Side(IntEnum):
    BUY = 0
    SELL = 1


@dataclass
class Order:
    order_id: str
    side: Side
    price: float
    size: float
    timestamp: int


@dataclass
class Fill:
    taker_order_id: str
    maker_order_id: str
    price: float
    size: float
    timestamp: int


def _neg(k: float) -> float:
    return -k


class OrderBook:
    """Limit order book with price-time priority matching."""

    def __init__(self) -> None:
        # Bids: SortedDict keyed on -price so peekitem(0) == best bid
        self._bids: SortedDict = SortedDict(_neg)
        # Asks: SortedDict keyed on +price so peekitem(0) == best ask
        self._asks: SortedDict = SortedDict()
        # order_id -> (side, price) for O(1) cancel lookup
        self._order_map: Dict[str, Tuple[Side, float]] = {}

    # ---- best price lookups ----

    def best_bid(self) -> Optional[float]:
        if not self._bids:
            return None
        return self._bids.peekitem(0)[0]

    def best_ask(self) -> Optional[float]:
        if not self._asks:
            return None
        return self._asks.peekitem(0)[0]

    def qty_at(self, side: Side, price: float) -> float:
        """Aggregate resting quantity at a price level (0 if absent)."""
        book = self._bids if side == Side.BUY else self._asks
        level = book.get(price)
        if not level:
            return 0.0
        return sum(o.size for o in level)

    # ---- depth ----

    def depth(self, side: Side, levels: int) -> List[Tuple[float, float]]:
        """Return top `levels` price levels as [(price, total_qty), ...]."""
        book = self._bids if side == Side.BUY else self._asks
        # SortedDict keys are already in best-first order for each side
        prices = list(book.keys())[:levels]
        return [(p, sum(o.size for o in book[p])) for p in prices]

    # ---- core operations ----

    def insert(
        self,
        order_id: str,
        side: Side,
        price: float,
        size: float,
        timestamp: int,
    ) -> List[Fill]:
        """Insert an order. Match against opposite side, rest any remainder."""
        if size <= 0:
            raise ValueError("size must be positive")
        if order_id in self._order_map:
            raise ValueError(f"duplicate order_id: {order_id}")

        fills, remaining = self._match(order_id, side, price, size, timestamp)
        if remaining > 0:
            self._rest(
                Order(order_id, side, price, remaining, timestamp)
            )
        return fills

    def cancel(self, order_id: str) -> bool:
        """Remove an order. Returns True if it existed."""
        entry = self._order_map.pop(order_id, None)
        if entry is None:
            return False
        side, price = entry
        book = self._bids if side == Side.BUY else self._asks
        queue = book[price]
        # linear scan within this price level only — queues are short
        for i, o in enumerate(queue):
            if o.order_id == order_id:
                queue.pop(i)
                break
        if not queue:
            del book[price]
        return True

    def apply_diff(self, side: Side, price: float, qty: float) -> None:
        """Apply a Binance-style L2 diff: replace the qty at `price`.

        qty == 0 means delete the price level entirely. Used by the data feed
        to mirror Binance's snapshot, not by the matching engine.
        """
        book = self._bids if side == Side.BUY else self._asks
        if qty == 0:
            book.pop(price, None)
            return
        # Replace the level with a single synthetic order representing the
        # aggregated visible depth. Matches the L2-only view we get from
        # Binance's depth stream.
        synthetic = Order(
            order_id=f"_diff_{side.name}_{price}",
            side=side,
            price=price,
            size=qty,
            timestamp=0,
        )
        book[price] = [synthetic]

    # ---- internal ----

    def _match(
        self,
        taker_id: str,
        side: Side,
        price: float,
        size: float,
        timestamp: int,
    ) -> Tuple[List[Fill], float]:
        """Walk the opposite book, fill what we can, return (fills, remaining)."""
        fills: List[Fill] = []
        opposite = self._asks if side == Side.BUY else self._bids
        remaining = size

        while remaining > 0 and opposite:
            # peekitem(0) is always the best price on the opposite side
            best_price = opposite.peekitem(0)[0]

            # crossing check
            crosses = (
                best_price <= price if side == Side.BUY else best_price >= price
            )
            if not crosses:
                break

            queue = opposite[best_price]
            # walk the FIFO queue at this price level
            while remaining > 0 and queue:
                maker = queue[0]
                trade_size = min(remaining, maker.size)
                fills.append(
                    Fill(
                        taker_order_id=taker_id,
                        maker_order_id=maker.order_id,
                        price=best_price,
                        size=trade_size,
                        timestamp=timestamp,
                    )
                )
                maker.size -= trade_size
                remaining -= trade_size
                if maker.size == 0:
                    queue.pop(0)
                    self._order_map.pop(maker.order_id, None)

            if not queue:
                del opposite[best_price]

        return fills, remaining

    def _rest(self, order: Order) -> None:
        book = self._bids if order.side == Side.BUY else self._asks
        if order.price not in book:
            book[order.price] = []
        book[order.price].append(order)
        self._order_map[order.order_id] = (order.side, order.price)
