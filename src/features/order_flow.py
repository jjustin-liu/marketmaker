"""Order Flow Imbalance (OFI).

Cont-Kukanov-Stoikov (2014) top-of-book order flow. Each L1 update
contributes a signed increment: bid additions and ask withdrawals are
bullish (+), bid withdrawals and ask additions are bearish (-). The
windowed sum of increments is a short-horizon predictor of the next
mid move — the standard cheap L2-only directional signal.

Unlike a static book snapshot, OFI is stateful: it depends on the
change between consecutive best-quote observations, so it is tracked
incrementally as the book is replayed.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

# One L1 observation: (best_bid_price, bid_qty, best_ask_price, ask_qty).
L1 = Tuple[float, float, float, float]


def ofi_increment(prev: L1, curr: L1) -> float:
    """Single-step OFI contribution from two consecutive L1 snapshots.

    Positive = net buying pressure (price likely to rise), negative =
    net selling pressure. Follows Cont-Kukanov-Stoikov:

      bid: price up -> +q_bid_now; same -> dq_bid; down -> -q_bid_prev
      ask: price up -> +q_ask_prev; same -> -dq_ask; down -> -q_ask_now

    and OFI = bid_term - ask_term.
    """
    pb0, qb0, pa0, qa0 = prev
    pb1, qb1, pa1, qa1 = curr

    if pb1 > pb0:
        e_bid = qb1
    elif pb1 == pb0:
        e_bid = qb1 - qb0
    else:
        e_bid = -qb0

    if pa1 > pa0:
        e_ask = -qa0
    elif pa1 == pa0:
        e_ask = qa1 - qa0
    else:
        e_ask = qa1

    return e_bid - e_ask


class OFITracker:
    """Rolling-window sum of OFI increments over the last `window` updates.

    Feed each new best-quote observation via update(); the return value
    is the windowed OFI at that point. Reset on epoch boundaries — a
    snapshot reload produces a spurious L1 jump that is not real flow.
    """

    def __init__(self, window: int = 50) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._prev: Optional[L1] = None
        self._buf: Deque[float] = deque(maxlen=window)
        self._sum = 0.0

    def update(
        self,
        best_bid: float,
        bid_qty: float,
        best_ask: float,
        ask_qty: float,
    ) -> float:
        """Add one L1 observation, return the current windowed OFI."""
        curr: L1 = (best_bid, bid_qty, best_ask, ask_qty)
        if self._prev is None:
            self._prev = curr
            return 0.0
        inc = ofi_increment(self._prev, curr)
        self._prev = curr
        if len(self._buf) == self.window:
            self._sum -= self._buf[0]
        self._buf.append(inc)
        self._sum += inc
        return self._sum

    @property
    def value(self) -> float:
        """Current windowed OFI without advancing state."""
        return self._sum

    def reset(self) -> None:
        """Clear all state (call on an epoch boundary)."""
        self._prev = None
        self._buf.clear()
        self._sum = 0.0
