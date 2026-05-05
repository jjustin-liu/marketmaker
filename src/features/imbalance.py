"""Order book imbalance features.

Imbalance measures whether more volume sits on the bid or ask side.
Computed at multiple depths so the model can pick which time horizon
correlates with fill probability.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

DEFAULT_LEVELS = (1, 2, 5)


def calculate_imbalance(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    levels: int,
) -> float:
    """Imbalance over the top `levels` levels on each side.

    Range: -1.0 (all volume on ask) to +1.0 (all volume on bid).
    Returns 0.0 if the book is empty on both sides up to `levels`.
    """
    if levels <= 0:
        raise ValueError("levels must be positive")
    bid_vol = sum(qty for _, qty in bids[:levels])
    ask_vol = sum(qty for _, qty in asks[:levels])
    total = bid_vol + ask_vol
    if total == 0.0:
        return 0.0
    return (bid_vol - ask_vol) / total


def get_imbalance_features(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
) -> Dict[str, float]:
    """Imbalance at standard depths (1, 2, 5)."""
    return {
        f"imbalance_{n}": calculate_imbalance(bids, asks, n)
        for n in DEFAULT_LEVELS
    }
