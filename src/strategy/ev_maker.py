"""EV-maximizing market maker.

Quotes symmetrically around an inventory-shifted fair centre. Candidate
half-spreads are anchored to the *live market spread* (not absolute bps
of mid), so the search lives near the touch instead of dollars away.
For each candidate the edge captured is its distance from the centre;
EV = P(fill) * edge. The argmax trades fill probability against edge, so
a tight quote that fills often can beat a wide quote that rarely fills.
A relative floor keeps the final pair from collapsing on a zero-width
book. Because both sides are centre ± h with h > 0, the quote can never
invert or cross.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from src.strategy.inventory_skew import InventorySkew
from src.strategy.naive_maker import Quote
from src.strategy.size_calculator import SizeCalculator

BPS = 10_000.0


class FillModel(Protocol):
    """Protocol for any fill probability source (real model or stub)."""

    def predict(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        price: float,
        size: float,
        side: str,
    ) -> float:
        ...


@dataclass
class EVConfig:
    # Candidate half-spreads as multiples of the live market spread.
    # min < 0.5 lets the tightest candidate sit *inside* the touch.
    min_half_spread_mult: float = 0.25
    max_half_spread_mult: float = 3.0
    num_points: int = 20
    # Anti-collapse floor only. Must stay below the real market spread
    # (~0.0016 bps on BTCUSDT) x min_half_spread_mult, or it overrides
    # the EV decision and pins quotes wider than the optimizer chose.
    min_spread_bps: float = 0.001
    fallback_spread_bps: float = 1.0  # market spread proxy when no book
    # Ceiling on the live-spread anchor, in bps of mid. The observed
    # spread spikes 100-1000x when the touch is transiently consumed
    # mid-event; anchoring the candidate band to those spikes parks
    # quotes dollars away where only toxic sweeps reach them.
    max_anchor_spread_bps: float = 0.05
    # Risk widening: push the candidate band outward (in market-spread
    # units) as inventory and volatility grow, to resist adverse selection.
    inv_risk_factor: float = 2.0    # widen by this × |inventory_norm|
    vol_risk_factor: float = 1.0    # widen by this × volatility

    def __post_init__(self) -> None:
        if self.min_half_spread_mult <= 0:
            raise ValueError("min_half_spread_mult must be positive")
        if self.max_half_spread_mult <= self.min_half_spread_mult:
            raise ValueError("max_half_spread_mult must exceed min")
        if self.num_points < 2:
            raise ValueError("num_points must be >= 2")
        if self.min_spread_bps <= 0:
            raise ValueError("min_spread_bps must be positive")
        if self.fallback_spread_bps <= 0:
            raise ValueError("fallback_spread_bps must be positive")
        if self.max_anchor_spread_bps <= 0:
            raise ValueError("max_anchor_spread_bps must be positive")
        if self.inv_risk_factor < 0:
            raise ValueError("inv_risk_factor must be non-negative")
        if self.vol_risk_factor < 0:
            raise ValueError("vol_risk_factor must be non-negative")


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _linspace(lo: float, hi: float, n: int) -> List[float]:
    if n == 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


class EVMaker:
    """EV-optimizing strategy with inventory skew + size scaling."""

    def __init__(
        self,
        inventory_skew: InventorySkew,
        size_calculator: SizeCalculator,
        fill_model: Optional[FillModel] = None,
        config: Optional[EVConfig] = None,
    ) -> None:
        self.inventory_skew = inventory_skew
        self.size_calculator = size_calculator
        self.fill_model = fill_model
        self.config = config or EVConfig()

    def quote_prices(
        self,
        mid_price: float,
        inventory: float,
        bids: Optional[List[Tuple[float, float]]] = None,
        asks: Optional[List[Tuple[float, float]]] = None,
        bid_probability: Optional[float] = None,
        ask_probability: Optional[float] = None,
        volatility: float = 0.0,
    ) -> Tuple[Quote, Quote]:
        """Pick the highest-EV (bid, ask) pair around the fair centre."""
        if mid_price <= 0:
            raise ValueError("mid_price must be positive")
        cfg = self.config

        # Inventory-shifted fair centre. Discard the skew's own spread —
        # we re-derive spacing from the live market spread below.
        base_bid, base_ask = self.inventory_skew.apply_skew(mid_price, inventory)
        centre = (base_bid + base_ask) / 2.0
        bid_size, ask_size = self.size_calculator.get_sizes(inventory)

        market_spread = self._market_spread(mid_price, bids, asks)

        # Risk widening: shift the whole candidate band outward as inventory
        # and volatility rise, so EV can't quote into informed flow as tightly.
        inv_norm = abs(_clip(inventory / self.inventory_skew.config.max_position,
                             -1.0, 1.0))
        risk_mult = (
            cfg.inv_risk_factor * inv_norm
            + cfg.vol_risk_factor * max(volatility, 0.0)
        )
        h_lo = (cfg.min_half_spread_mult + risk_mult) * market_spread
        h_hi = (cfg.max_half_spread_mult + risk_mult) * market_spread
        candidates = _linspace(h_lo, h_hi, cfg.num_points)

        best_bid_price, best_bid_ev = centre - h_lo, -float("inf")
        best_ask_price, best_ask_ev = centre + h_lo, -float("inf")

        for h in candidates:
            bid_candidate = centre - h
            ask_candidate = centre + h

            if self.fill_model is not None and bids and asks:
                p_bid = self.fill_model.predict(
                    bids, asks, bid_candidate, bid_size, "buy",
                )
                p_ask = self.fill_model.predict(
                    bids, asks, ask_candidate, ask_size, "sell",
                )
            else:
                p_bid = bid_probability if bid_probability is not None else 0.5
                p_ask = ask_probability if ask_probability is not None else 0.5

            # Edge captured if filled = distance from fair centre = h.
            ev_bid = p_bid * h
            ev_ask = p_ask * h
            if ev_bid > best_bid_ev:
                best_bid_ev, best_bid_price = ev_bid, bid_candidate
            if ev_ask > best_ask_ev:
                best_ask_ev, best_ask_price = ev_ask, ask_candidate

        # Relative floor so the pair never collapses on a zero-width book.
        min_abs_spread = cfg.min_spread_bps * mid_price / BPS
        if best_ask_price - best_bid_price < min_abs_spread:
            c = (best_ask_price + best_bid_price) / 2.0
            best_bid_price = c - min_abs_spread / 2.0
            best_ask_price = c + min_abs_spread / 2.0

        # Never quote through the touch: a bid at/above the best ask
        # (or ask at/below the best bid) is a marketable order, not a
        # resting quote. min()/max() only widen a floored pair, so the
        # clamp cannot re-cross bid and ask.
        if bids and asks and asks[0][0] > bids[0][0]:
            guard = min_abs_spread / 2.0
            best_bid_price = min(best_bid_price, asks[0][0] - guard)
            best_ask_price = max(best_ask_price, bids[0][0] + guard)

        return (
            Quote(price=best_bid_price, size=bid_size),
            Quote(price=best_ask_price, size=ask_size),
        )

    def _market_spread(
        self,
        mid_price: float,
        bids: Optional[List[Tuple[float, float]]],
        asks: Optional[List[Tuple[float, float]]],
    ) -> float:
        """Live touch spread from depth (capped), or a relative fallback."""
        cap = mid_price * self.config.max_anchor_spread_bps / BPS
        if bids and asks:
            spread = asks[0][0] - bids[0][0]
            if spread > 0:
                return min(spread, cap)
        return min(mid_price * self.config.fallback_spread_bps / BPS, cap)
