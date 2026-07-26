"""Tests for Order Flow Imbalance."""

from __future__ import annotations

import pytest

from src.features.order_flow import OFITracker, ofi_increment


def test_first_update_returns_zero() -> None:
    t = OFITracker(window=10)
    # No prior L1 to diff against.
    assert t.update(100.0, 5.0, 101.0, 5.0) == 0.0


def test_bid_size_growth_is_bullish() -> None:
    prev = (100.0, 5.0, 101.0, 5.0)
    curr = (100.0, 8.0, 101.0, 5.0)  # more bid volume at same price
    assert ofi_increment(prev, curr) == pytest.approx(3.0)


def test_bid_price_up_is_bullish() -> None:
    prev = (100.0, 5.0, 101.0, 5.0)
    curr = (100.5, 4.0, 101.0, 5.0)  # bid stepped up
    assert ofi_increment(prev, curr) == pytest.approx(4.0)


def test_ask_price_up_is_bullish() -> None:
    prev = (100.0, 5.0, 101.0, 6.0)
    curr = (100.0, 5.0, 101.5, 2.0)  # ask retreated up -> supply withdrawn
    assert ofi_increment(prev, curr) == pytest.approx(6.0)


def test_ask_size_growth_is_bearish() -> None:
    prev = (100.0, 5.0, 101.0, 5.0)
    curr = (100.0, 5.0, 101.0, 9.0)  # more ask volume
    assert ofi_increment(prev, curr) == pytest.approx(-4.0)


def test_bid_price_down_is_bearish() -> None:
    prev = (100.0, 5.0, 101.0, 5.0)
    curr = (99.5, 7.0, 101.0, 5.0)  # bid fell -> demand pulled
    assert ofi_increment(prev, curr) == pytest.approx(-5.0)


def test_window_sum_accumulates_and_evicts() -> None:
    t = OFITracker(window=2)
    t.update(100.0, 5.0, 101.0, 5.0)          # seed, returns 0
    v1 = t.update(100.0, 8.0, 101.0, 5.0)     # +3
    assert v1 == pytest.approx(3.0)
    v2 = t.update(100.0, 10.0, 101.0, 5.0)    # +2 -> window [3, 2]
    assert v2 == pytest.approx(5.0)
    v3 = t.update(100.0, 12.0, 101.0, 5.0)    # +2, evict the first +3
    assert v3 == pytest.approx(4.0)
    assert t.value == pytest.approx(4.0)


def test_reset_clears_state() -> None:
    t = OFITracker(window=5)
    t.update(100.0, 5.0, 101.0, 5.0)
    t.update(100.0, 9.0, 101.0, 5.0)
    assert t.value != 0.0
    t.reset()
    assert t.value == 0.0
    # After reset the next update is treated as a fresh seed.
    assert t.update(200.0, 1.0, 201.0, 1.0) == 0.0


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError):
        OFITracker(window=0)
