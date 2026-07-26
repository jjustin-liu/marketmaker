"""Tests for the alpha-gated market maker."""

from __future__ import annotations

from typing import List, Tuple

import pytest

from src.models.alpha_model import AlphaModel
from src.strategy.alpha_maker import AlphaConfig, AlphaMaker


class _StubAlpha(AlphaModel):
    """Returns a fixed predicted forward return, ignoring the book."""

    def __init__(self, value: float) -> None:
        super().__init__(model=None, scaler=None, ic=None)
        self._value = value

    def predict_from_book(self, bids, asks, ofi) -> float:  # type: ignore[override]
        return self._value


BOOK: Tuple[List[Tuple[float, float]], List[Tuple[float, float]]] = (
    [(100.0, 5.0), (99.0, 5.0)],
    [(101.0, 5.0), (102.0, 5.0)],
)


def test_bearish_signal_withdraws_bid() -> None:
    m = AlphaMaker(_StubAlpha(-0.001), AlphaConfig(gate_threshold_bps=0.0))
    bid, ask = m.quote_prices(100.5, 0.0, BOOK[0], BOOK[1])
    assert bid is None
    assert ask is not None


def test_bullish_signal_withdraws_ask() -> None:
    m = AlphaMaker(_StubAlpha(0.001), AlphaConfig(gate_threshold_bps=0.0))
    bid, ask = m.quote_prices(100.5, 0.0, BOOK[0], BOOK[1])
    assert ask is None
    assert bid is not None


def test_signal_below_threshold_quotes_both_sides() -> None:
    # predicted move 0.05 bps < 0.3 bps threshold -> no gate
    m = AlphaMaker(_StubAlpha(5e-6), AlphaConfig(gate_threshold_bps=0.3))
    bid, ask = m.quote_prices(100.5, 0.0, BOOK[0], BOOK[1])
    assert bid is not None and ask is not None


def test_quotes_rest_inside_the_touch() -> None:
    m = AlphaMaker(_StubAlpha(0.0), AlphaConfig())
    bid, ask = m.quote_prices(100.5, 0.0, BOOK[0], BOOK[1])
    assert bid is not None and ask is not None
    # Strictly inside best bid (100) / best ask (101), never crossing.
    assert 100.0 < bid.price < ask.price < 101.0


def test_alpha_shift_clamped_inside_touch() -> None:
    # Huge gain + strong signal must not push a quote through the touch.
    m = AlphaMaker(
        _StubAlpha(0.01),
        AlphaConfig(alpha_gain=100.0, gate_threshold_bps=100.0),
    )
    bid, ask = m.quote_prices(100.5, 0.0, BOOK[0], BOOK[1])
    assert bid is not None and ask is not None
    # Clamp lets a quote reach the touch (joining the queue) but never
    # cross it: bid at/above best bid, ask at/below best ask, bid < ask.
    assert 100.0 <= bid.price < ask.price <= 101.0


def test_empty_book_falls_back_to_two_sided() -> None:
    m = AlphaMaker(_StubAlpha(-0.001), AlphaConfig())
    bid, ask = m.quote_prices(100.5, 0.0, None, None)
    assert bid is not None and ask is not None
    assert bid.price < 100.5 < ask.price


def test_observe_l1_updates_ofi_state() -> None:
    m = AlphaMaker(_StubAlpha(0.0), AlphaConfig())
    assert m._ofi.value == 0.0
    m.observe_l1(100.0, 5.0, 101.0, 5.0)      # seed
    m.observe_l1(100.0, 9.0, 101.0, 5.0)      # +4 bid growth
    assert m._ofi.value == pytest.approx(4.0)
    m.reset()
    assert m._ofi.value == 0.0


def test_bad_mid_raises() -> None:
    m = AlphaMaker(_StubAlpha(0.0), AlphaConfig())
    with pytest.raises(ValueError):
        m.quote_prices(0.0, 0.0, BOOK[0], BOOK[1])
