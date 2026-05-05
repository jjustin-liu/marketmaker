"""Publish current LOB state and trade history to Redis.

The data feed owns these keys exclusively. Other processes read,
never write. See ARCHITECTURE.md for the full schema.
"""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

DEPTH_LEVELS_PUBLISHED = 20
TRADE_RINGBUFFER_SIZE = 100
BPS_PER_UNIT = 10000.0


class RedisWriter:
    """Serialize engine state into Redis keys after each update."""

    def __init__(self, redis_client: Any, engine: Any, *,
                 bid_side: Any, ask_side: Any) -> None:
        self._redis = redis_client
        self._engine = engine
        self._BID = bid_side
        self._ASK = ask_side
        self._trades: Deque[Dict[str, Any]] = deque(maxlen=TRADE_RINGBUFFER_SIZE)

    def publish_book(self, now: Optional[float] = None) -> None:
        """Write all lob:* keys for the current engine state."""
        ts = now if now is not None else time.time()

        best_bid = self._engine.best_bid()
        best_ask = self._engine.best_ask()
        bids = list(self._engine.depth(self._BID, DEPTH_LEVELS_PUBLISHED))
        asks = list(self._engine.depth(self._ASK, DEPTH_LEVELS_PUBLISHED))

        pipe = self._redis.pipeline()
        if best_bid is not None:
            pipe.set("lob:best_bid", best_bid)
        if best_ask is not None:
            pipe.set("lob:best_ask", best_ask)
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
            spread_bps = (best_ask - best_bid) / mid * BPS_PER_UNIT
            pipe.set("lob:mid", mid)
            pipe.set("lob:spread_bps", spread_bps)
        pipe.set("lob:levels:bids", json.dumps(bids))
        pipe.set("lob:levels:asks", json.dumps(asks))
        pipe.set("lob:last_update", ts)
        pipe.execute()

    def record_trade(self, price: float, qty: float, side: str,
                     timestamp: int) -> None:
        """Add a trade to the ringbuffer and publish trades:recent.

        side is the aggressor side: 'buy' or 'sell'.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        self._trades.append({
            "price": price,
            "qty": qty,
            "side": side,
            "timestamp": timestamp,
        })
        self._redis.set("trades:recent", json.dumps(list(self._trades)))

    @property
    def recent_trades(self) -> List[Dict[str, Any]]:
        return list(self._trades)
