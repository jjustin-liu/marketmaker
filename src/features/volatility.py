"""Rolling realized volatility from a stream of mid prices.

Stateful — keeps the last N mid prices in a deque and computes
annualized stdev of returns.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque

WINDOW_SIZE = 100
ANNUALIZATION_FACTOR = math.sqrt(252)


class VolatilityCalculator:
    """Rolling-window realized volatility on tick-level mid prices."""

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self._window: Deque[float] = deque(maxlen=window_size)

    def update(self, mid_price: float) -> float:
        """Add a mid price and return current annualized volatility."""
        self._window.append(mid_price)
        return self._compute()

    def value(self) -> float:
        """Current volatility without ingesting a new price."""
        return self._compute()

    def reset(self) -> None:
        self._window.clear()

    def _compute(self) -> float:
        if len(self._window) < 2:
            return 0.0
        prices = list(self._window)
        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
            if prices[i - 1] != 0.0
        ]
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(var) * ANNUALIZATION_FACTOR
