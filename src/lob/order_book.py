"""Pure-Python limit order book.

Mirrors the C++ MatchEngine interface. Used in backtests and unit tests
where readability matters more than throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


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


class OrderBook:
    """Limit order book with price-time priority matching."""

    def __init__(self) -> None:
        # price -> FIFO list of resting orders at that price
        self._bids: Dict[float, List[Order]] = {}
        self._asks: Dict[float, List[Order]] = {}
        # order_id -> (side, price) for O(log n) cancel
        self._order_map: Dict[str, Tuple[Side, float]] = {}

    # ---- best price lookups ----

    def best_bid(self) -> Optional[float]:
        return max(self._bids) if self._bids else None

    def best_ask(self) -> Optional[float]:
        return min(self._asks) if self._asks else None

    # ---- depth ----

    def depth(self, side: Side, levels: int) -> List[Tuple[float, float]]:
        """Return top `levels` price levels as [(price, total_qty), ...]."""
        book = self._bids if side == Side.BUY else self._asks
        prices = sorted(book.keys(), reverse=(side == Side.BUY))[:levels]
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
            # best price on the opposite side
            best_price = min(opposite) if side == Side.BUY else max(opposite)

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
        book.setdefault(order.price, []).append(order)
        self._order_map[order.order_id] = (order.side, order.price)
