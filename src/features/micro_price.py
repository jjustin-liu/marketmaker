"""Volume-weighted micro price.

Pulls the mid toward the thinner side of the book — a better estimate
of where the next trade actually executes than the simple midpoint.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def calculate_microprice(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
) -> Optional[float]:
    """Return the volume-weighted micro price.

    Formula:
      microprice = (best_bid * ask_qty + best_ask * bid_qty)
                   / (bid_qty + ask_qty)

    Returns None if either side is empty (no microprice defined).
    """
    if not bids or not asks:
        return None
    best_bid, bid_qty = bids[0]
    best_ask, ask_qty = asks[0]
    total = bid_qty + ask_qty
    if total == 0.0:
        return None
    return (best_bid * ask_qty + best_ask * bid_qty) / total
