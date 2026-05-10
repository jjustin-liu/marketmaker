"""Inventory-aware quote centring and spread widening.

Long position pulls quotes down (easier to sell, harder to buy);
short position pulls quotes up. Spread widens with |inventory|.
Continuity clip limits per-refresh quote movement to preserve queue
position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

BPS = 10_000.0


@dataclass
class InventorySkewConfig:
    """All five parameters from ARCHITECTURE.md."""

    max_position: float = 1.0
    skew_factor: float = 0.5
    min_spread_bps: float = 1.0
    spread_factor: float = 1.0
    continuity_clip: float = 0.1
    float_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.max_position <= 0:
            raise ValueError("max_position must be positive")
        if self.skew_factor < 0:
            raise ValueError("skew_factor must be non-negative")
        if self.min_spread_bps <= 0:
            raise ValueError("min_spread_bps must be positive")
        if self.spread_factor < 0:
            raise ValueError("spread_factor must be non-negative")
        if self.continuity_clip <= 0:
            raise ValueError("continuity_clip must be positive")


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class InventorySkew:
    """Centre-shift + spread-widen + continuity clip."""

    def __init__(self, config: Optional[InventorySkewConfig] = None) -> None:
        self.config = config or InventorySkewConfig()
        self._prev_bid: Optional[float] = None
        self._prev_ask: Optional[float] = None

    def apply_skew(
        self,
        mid_price: float,
        inventory: float,
    ) -> Tuple[float, float]:
        """Return (bid_price, ask_price) adjusted for inventory."""
        if mid_price <= 0:
            raise ValueError("mid_price must be positive")
        cfg = self.config

        inv = _clip(inventory / cfg.max_position, -1.0, 1.0)

        min_spread = mid_price * cfg.min_spread_bps / BPS
        spread = min_spread * (1.0 + abs(inv) * cfg.spread_factor)
        half_spread = spread / 2.0

        centre_shift = -inv * cfg.skew_factor * spread

        bid_price = mid_price + centre_shift - half_spread
        ask_price = mid_price + centre_shift + half_spread

        if self._prev_bid is not None and self._prev_ask is not None:
            clip_limit = cfg.continuity_clip
            bid_move = _clip(bid_price - self._prev_bid, -clip_limit, clip_limit)
            ask_move = _clip(ask_price - self._prev_ask, -clip_limit, clip_limit)
            bid_price = self._prev_bid + bid_move
            ask_price = self._prev_ask + ask_move

        if ask_price - bid_price < cfg.float_tolerance:
            mid_pt = (ask_price + bid_price) / 2.0
            bid_price = mid_pt - half_spread
            ask_price = mid_pt + half_spread

        self._prev_bid = bid_price
        self._prev_ask = ask_price
        return bid_price, ask_price

    def reset(self) -> None:
        self._prev_bid = None
        self._prev_ask = None
