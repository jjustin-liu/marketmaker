"""Fixed-spread market maker — the baseline strategy.

No EV optimization, no inventory awareness. Posts at mid ± half_spread,
tightening inside the current market spread when book quotes are
provided. Used as the benchmark in backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Quote:
    """Single-side quote: price and size."""

    price: float
    size: float


@dataclass
class NaiveMakerConfig:
    """Fixed-spread maker config."""

    spread: float = 0.001  # relative, e.g. 0.001 = 10 bps total spread
    size: float = 0.001    # BTC per side

    def __post_init__(self) -> None:
        if self.spread <= 0:
            raise ValueError("spread must be positive")
        if self.size <= 0:
            raise ValueError("size must be positive")


class NaiveMaker:
    """Baseline maker: post at mid ± half_spread."""

    def __init__(self, config: Optional[NaiveMakerConfig] = None) -> None:
        self.config = config or NaiveMakerConfig()

    def quote_prices(
        self,
        mid_price: float,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
    ) -> Tuple[Quote, Quote]:
        """Return (bid_quote, ask_quote).

        If best_bid/best_ask are given and tighter than mid ± half_spread,
        quote inside the market by one tick (1 bps of mid) for queue.
        """
        if mid_price <= 0:
            raise ValueError("mid_price must be positive")

        half = (mid_price * self.config.spread) / 2.0
        bid_price = mid_price - half
        ask_price = mid_price + half

        if best_bid is not None and bid_price < best_bid:
            bid_price = best_bid + mid_price * 1e-5
        if best_ask is not None and ask_price > best_ask:
            ask_price = best_ask - mid_price * 1e-5

        return (
            Quote(price=bid_price, size=self.config.size),
            Quote(price=ask_price, size=self.config.size),
        )
