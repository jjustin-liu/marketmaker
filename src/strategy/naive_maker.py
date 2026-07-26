"""Fixed-spread market maker — the baseline strategy.

Quotes symmetrically around the mid. When the live book is provided it
anchors the quoted spread to the current market spread (capped at half
of it) and pulls the quotes toward mid by an `aggressiveness` factor.
This keeps the quote strictly inside the touch and symmetric around
mid, so it can never invert or cross the market — no clamping needed.
Used as the benchmark in backtest.
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

    spread: float = 0.001       # relative target spread, e.g. 0.001 = 10 bps
    size: float = 0.001         # BTC per side
    aggressiveness: float = 0.2  # 0..1; higher pulls quotes closer to mid

    def __post_init__(self) -> None:
        if self.spread <= 0:
            raise ValueError("spread must be positive")
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not 0.0 <= self.aggressiveness < 1.0:
            raise ValueError("aggressiveness must be in [0, 1)")


class NaiveMaker:
    """Baseline maker: post symmetrically around mid, inside the market."""

    def __init__(self, config: Optional[NaiveMakerConfig] = None) -> None:
        self.config = config or NaiveMakerConfig()

    def quote_prices(
        self,
        mid_price: float,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
    ) -> Tuple[Quote, Quote]:
        """Return (bid_quote, ask_quote), symmetric around mid.

        Without a book: quote at mid ± (target_spread / 2).
        With a book: cap the spread at half the live market spread and
        pull toward mid by `aggressiveness`, so the quote always rests
        strictly inside the touch (never crosses, never inverts).
        """
        if mid_price <= 0:
            raise ValueError("mid_price must be positive")

        target_spread = mid_price * self.config.spread

        if best_bid is None or best_ask is None:
            half_spread = target_spread / 2.0
        else:
            market_spread = best_ask - best_bid
            effective_spread = min(target_spread, market_spread * 0.5)
            half_spread = (
                effective_spread * (1.0 - self.config.aggressiveness) / 2.0
            )

        bid_price = mid_price - half_spread
        ask_price = mid_price + half_spread

        return (
            Quote(price=bid_price, size=self.config.size),
            Quote(price=ask_price, size=self.config.size),
        )
