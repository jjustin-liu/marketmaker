"""EV-maximizing market maker.

Generates a grid of candidate offsets around the inventory-adjusted
base quote, scores each by P(fill) * spread, and picks the argmax.
Enforces a minimum spread on the final pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from src.strategy.inventory_skew import InventorySkew
from src.strategy.naive_maker import Quote
from src.strategy.size_calculator import SizeCalculator


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
    min_spread: float = 0.0005
    max_spread: float = 0.005
    num_points: int = 10

    def __post_init__(self) -> None:
        if self.min_spread <= 0:
            raise ValueError("min_spread must be positive")
        if self.max_spread <= self.min_spread:
            raise ValueError("max_spread must exceed min_spread")
        if self.num_points < 2:
            raise ValueError("num_points must be >= 2")


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
    ) -> Tuple[Quote, Quote]:
        """Pick the highest-EV (bid, ask) pair."""
        if mid_price <= 0:
            raise ValueError("mid_price must be positive")
        cfg = self.config

        base_bid, base_ask = self.inventory_skew.apply_skew(mid_price, inventory)
        bid_size, ask_size = self.size_calculator.get_sizes(inventory)

        offsets = _linspace(0.0, cfg.max_spread, cfg.num_points)

        best_bid_price = base_bid
        best_bid_ev = -float("inf")
        best_ask_price = base_ask
        best_ask_ev = -float("inf")

        for offset in offsets:
            bid_candidate = base_bid - offset * mid_price
            ask_candidate = base_ask + offset * mid_price

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

            ev_bid = p_bid * offset
            ev_ask = p_ask * offset
            if ev_bid > best_bid_ev:
                best_bid_ev = ev_bid
                best_bid_price = bid_candidate
            if ev_ask > best_ask_ev:
                best_ask_ev = ev_ask
                best_ask_price = ask_candidate

        min_abs_spread = cfg.min_spread * mid_price
        if best_ask_price - best_bid_price < min_abs_spread:
            centre = (best_ask_price + best_bid_price) / 2.0
            best_bid_price = centre - min_abs_spread / 2.0
            best_ask_price = centre + min_abs_spread / 2.0

        max_move = cfg.max_spread * mid_price
        assert abs(best_bid_price - base_bid) <= max_move + 1e-9, "bid out of bounds"
        assert abs(best_ask_price - base_ask) <= max_move + 1e-9, "ask out of bounds"

        return (
            Quote(price=best_bid_price, size=bid_size),
            Quote(price=best_ask_price, size=ask_size),
        )
