"""Alpha-gated market maker.

Quotes at the touch like NaiveMaker, but uses the short-horizon alpha
signal to *withdraw the toxic side*: when the signal predicts the mid
is about to fall, a resting bid would be adversely selected (we'd buy
just before the drop), so the bid is pulled and only the ask rests —
and symmetrically when the signal predicts a rise. Optionally the quote
centre is nudged by the predicted move (alpha_gain), clamped to stay
inside the touch so it can never cross.

This is the Phase-13 test of the Phase-11 diagnosis: the loss is
adverse selection, and on a 1-tick book the only real lever is whether
to quote a side at all (you cannot meaningfully reprice within one
tick). The alpha signal is stateful (windowed OFI), so the strategy
consumes every book update via observe_l1(), fed by the backtest
engine's per-tick hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.features.order_flow import OFITracker
from src.models.alpha_model import AlphaModel
from src.strategy.naive_maker import Quote

BPS = 10_000.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class AlphaConfig:
    """Alpha-gated maker config."""

    target_spread: float = 0.001    # relative cap on quoted spread (like naive)
    size: float = 0.001             # BTC per side
    aggressiveness: float = 0.2     # 0..1; higher pulls quotes toward centre
    # Centre nudge = alpha_gain * predicted_return * mid, clamped inside the
    # touch. Default 0: on a 1-tick book repricing within the spread is
    # meaningless, so the gate (not the shift) carries the signal.
    alpha_gain: float = 0.0
    # Withdraw a side when the predicted move against it exceeds this, in
    # bps. 0 = gate on sign (quote one-sided whenever the signal is nonzero).
    gate_threshold_bps: float = 0.0
    ofi_window: int = 50

    def __post_init__(self) -> None:
        if self.target_spread <= 0:
            raise ValueError("target_spread must be positive")
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not 0.0 <= self.aggressiveness < 1.0:
            raise ValueError("aggressiveness must be in [0, 1)")
        if self.alpha_gain < 0:
            raise ValueError("alpha_gain must be non-negative")
        if self.gate_threshold_bps < 0:
            raise ValueError("gate_threshold_bps must be non-negative")
        if self.ofi_window < 1:
            raise ValueError("ofi_window must be >= 1")


class AlphaMaker:
    """At-touch maker that withdraws the side the alpha signal marks toxic."""

    def __init__(
        self,
        alpha_model: AlphaModel,
        config: Optional[AlphaConfig] = None,
    ) -> None:
        self.alpha_model = alpha_model
        self.config = config or AlphaConfig()
        self._ofi = OFITracker(window=self.config.ofi_window)

    def observe_l1(
        self,
        best_bid: float,
        bid_qty: float,
        best_ask: float,
        ask_qty: float,
    ) -> None:
        """Feed one L1 update into the OFI tracker (called every tick)."""
        self._ofi.update(best_bid, bid_qty, best_ask, ask_qty)

    def reset(self) -> None:
        """Clear signal state (epoch boundary)."""
        self._ofi.reset()

    def quote_prices(
        self,
        mid_price: float,
        inventory: float = 0.0,
        bids: Optional[List[Tuple[float, float]]] = None,
        asks: Optional[List[Tuple[float, float]]] = None,
        volatility: float = 0.0,
    ) -> Tuple[Optional[Quote], Optional[Quote]]:
        """Return (bid, ask); either may be None when its side is withdrawn."""
        if mid_price <= 0:
            raise ValueError("mid_price must be positive")
        cfg = self.config

        # No book: fall back to a symmetric two-sided quote at target spread.
        if not bids or not asks:
            half = mid_price * cfg.target_spread / 2.0
            return (
                Quote(mid_price - half, cfg.size),
                Quote(mid_price + half, cfg.size),
            )

        best_bid, best_ask = bids[0][0], asks[0][0]
        market_spread = best_ask - best_bid

        alpha = self.alpha_model.predict_from_book(bids, asks, self._ofi.value)

        # Spread geometry identical to NaiveMaker: rest strictly inside the
        # touch, symmetric around the (alpha-nudged) centre.
        target_spread = mid_price * cfg.target_spread
        effective_spread = min(target_spread, market_spread * 0.5)
        half = effective_spread * (1.0 - cfg.aggressiveness) / 2.0

        # Centre nudge, clamped so both quotes stay inside the touch.
        max_shift = max(0.0, market_spread * 0.5 - half)
        shift = _clip(cfg.alpha_gain * alpha * mid_price, -max_shift, max_shift)
        centre = mid_price + shift

        bid: Optional[Quote] = Quote(centre - half, cfg.size)
        ask: Optional[Quote] = Quote(centre + half, cfg.size)

        # Gate: withdraw the side the signal predicts will be adversely
        # selected. Bearish signal -> a bid fill buys into the drop -> pull
        # the bid; bullish -> an ask fill sells into the rise -> pull the ask.
        threshold = cfg.gate_threshold_bps / BPS
        if alpha < -threshold:
            bid = None
        elif alpha > threshold:
            ask = None

        return bid, ask
