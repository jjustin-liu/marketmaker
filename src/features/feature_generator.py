"""Assemble the 10-element feature vector for the fill probability model.

Reads current LOB state from Redis, combines static book features with
the per-candidate order features (price, size, side), and emits a
numpy array in the canonical order the model expects.
"""

from __future__ import annotations

import json
from typing import Any, List, Tuple

import numpy as np

from src.features.imbalance import calculate_imbalance
from src.features.volatility import VolatilityCalculator

FEATURE_VECTOR_SIZE = 10
SIDE_BUY = 1.0
SIDE_SELL = 0.0


def _parse_levels(raw: Any) -> List[Tuple[float, float]]:
    """Parse a Redis JSON levels blob into list of (price, qty)."""
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode()
    parsed = json.loads(raw)
    return [(float(p), float(q)) for p, q in parsed]


class FeatureGenerator:
    """Build the model input vector from Redis state and a candidate quote."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._vol = VolatilityCalculator()

    def generate(
        self,
        candidate_price: float,
        candidate_size: float,
        candidate_side: str,
    ) -> np.ndarray:
        """Return a 10-element feature vector for a single candidate quote.

        candidate_side: 'buy' or 'sell'.
        """
        if candidate_side not in ("buy", "sell"):
            raise ValueError(
                f"candidate_side must be 'buy' or 'sell', got {candidate_side!r}"
            )

        bids = _parse_levels(self._redis.get("lob:levels:bids"))
        asks = _parse_levels(self._redis.get("lob:levels:asks"))
        if not bids or not asks:
            raise RuntimeError("LOB state missing from Redis")

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2.0
        if mid == 0.0:
            raise RuntimeError("mid price is zero")

        spread = (best_ask - best_bid) / mid
        bid_volume = sum(q for _, q in bids)
        ask_volume = sum(q for _, q in asks)
        imb_1 = calculate_imbalance(bids, asks, 1)
        imb_2 = calculate_imbalance(bids, asks, 2)
        imb_5 = calculate_imbalance(bids, asks, 5)
        price_distance = abs(candidate_price - mid) / mid
        side = SIDE_BUY if candidate_side == "buy" else SIDE_SELL

        # Maintain volatility series; not in the v1 vector but tracked
        # so it's available to the strategy for adverse-selection scaling.
        self._vol.update(mid)

        return np.array([
            spread,
            mid,
            bid_volume,
            ask_volume,
            imb_1,
            imb_2,
            imb_5,
            price_distance,
            candidate_size,
            side,
        ], dtype=np.float64)

    @property
    def volatility(self) -> float:
        return self._vol.value()
