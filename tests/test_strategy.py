"""Tests for the strategy layer."""

from typing import List, Tuple

import pytest

from src.strategy.ev_maker import EVConfig, EVMaker
from src.strategy.inventory_skew import InventorySkew, InventorySkewConfig
from src.strategy.naive_maker import NaiveMaker, NaiveMakerConfig
from src.strategy.size_calculator import ScalingType, SizeCalculator, SizeConfig


# ---------- NaiveMaker ----------


def test_naive_ask_above_bid() -> None:
    m = NaiveMaker()
    bid, ask = m.quote_prices(100.0)
    assert ask.price > bid.price


def test_naive_spread_matches_config() -> None:
    m = NaiveMaker(NaiveMakerConfig(spread=0.002, size=0.01))
    bid, ask = m.quote_prices(100.0)
    assert (ask.price - bid.price) == pytest.approx(0.2)
    assert bid.size == 0.01 and ask.size == 0.01


def test_naive_tightens_inside_market() -> None:
    m = NaiveMaker(NaiveMakerConfig(spread=0.01))
    bid, ask = m.quote_prices(100.0, best_bid=99.9, best_ask=100.1)
    # original would be 99.5 / 100.5 — both outside the inside market
    assert bid.price > 99.9 and ask.price < 100.1


def test_naive_rejects_bad_mid() -> None:
    m = NaiveMaker()
    with pytest.raises(ValueError):
        m.quote_prices(0.0)


# ---------- InventorySkew ----------


def test_skew_neutral_symmetric() -> None:
    s = InventorySkew()
    bid, ask = s.apply_skew(100.0, 0.0)
    mid = (bid + ask) / 2.0
    assert mid == pytest.approx(100.0)


def test_skew_long_centre_shifts_down() -> None:
    s = InventorySkew()
    _, _ = s.apply_skew(100.0, 0.0)
    s.reset()
    bid, ask = s.apply_skew(100.0, 0.5)
    mid = (bid + ask) / 2.0
    assert mid < 100.0  # long → quotes shift down


def test_skew_short_centre_shifts_up() -> None:
    s = InventorySkew()
    bid, ask = s.apply_skew(100.0, -0.5)
    assert (bid + ask) / 2.0 > 100.0


def test_skew_spread_widens_with_inventory() -> None:
    s_flat = InventorySkew()
    b0, a0 = s_flat.apply_skew(100.0, 0.0)
    s_long = InventorySkew()
    b1, a1 = s_long.apply_skew(100.0, 1.0)
    assert (a1 - b1) > (a0 - b0)


def test_skew_continuity_clip_limits_movement() -> None:
    cfg = InventorySkewConfig(continuity_clip=0.01)
    s = InventorySkew(cfg)
    b0, a0 = s.apply_skew(100.0, 0.0)
    # Big mid jump
    b1, a1 = s.apply_skew(110.0, 0.0)
    assert abs(b1 - b0) <= cfg.continuity_clip + 1e-9
    assert abs(a1 - a0) <= cfg.continuity_clip + 1e-9


def test_skew_clips_extreme_inventory() -> None:
    s1 = InventorySkew()
    bid1, ask1 = s1.apply_skew(100.0, 1.0)
    s2 = InventorySkew()
    bid2, ask2 = s2.apply_skew(100.0, 100.0)  # way over max
    assert bid1 == pytest.approx(bid2)
    assert ask1 == pytest.approx(ask2)


def test_skew_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        InventorySkewConfig(max_position=-1.0)
    with pytest.raises(ValueError):
        InventorySkewConfig(continuity_clip=0.0)


# ---------- SizeCalculator ----------


def test_size_neutral_symmetric() -> None:
    sc = SizeCalculator()
    bid, ask = sc.get_sizes(0.0)
    assert bid == pytest.approx(ask)


def test_size_long_ask_bigger() -> None:
    sc = SizeCalculator()
    bid, ask = sc.get_sizes(0.5)
    assert ask > bid


def test_size_short_bid_bigger() -> None:
    sc = SizeCalculator()
    bid, ask = sc.get_sizes(-0.5)
    assert bid > ask


def test_size_max_long_zeroes_bid() -> None:
    sc = SizeCalculator()
    bid, ask = sc.get_sizes(1.0)
    assert bid == 0.0
    assert ask > 0.0


def test_size_max_short_zeroes_ask() -> None:
    sc = SizeCalculator()
    bid, ask = sc.get_sizes(-1.0)
    assert ask == 0.0
    assert bid > 0.0


def test_size_sigmoid_mode_runs() -> None:
    sc = SizeCalculator(SizeConfig(scaling_type=ScalingType.SIGMOID))
    bid, ask = sc.get_sizes(0.3)
    assert bid > 0 and ask > 0
    assert ask > bid  # still long-skewed


def test_size_invalid_config() -> None:
    with pytest.raises(ValueError):
        SizeConfig(base_size=-1.0)
    with pytest.raises(ValueError):
        SizeConfig(max_size_mult=0.5)


# ---------- EVMaker ----------


def _make_ev_maker(fill_model=None) -> EVMaker:
    return EVMaker(
        inventory_skew=InventorySkew(),
        size_calculator=SizeCalculator(),
        fill_model=fill_model,
        config=EVConfig(min_spread=0.0005, max_spread=0.005, num_points=10),
    )


def test_ev_ask_above_bid() -> None:
    ev = _make_ev_maker()
    bid, ask = ev.quote_prices(
        mid_price=100.0,
        inventory=0.0,
        bid_probability=0.5,
        ask_probability=0.5,
    )
    assert ask.price > bid.price


def test_ev_min_spread_enforced() -> None:
    ev = _make_ev_maker()
    bid, ask = ev.quote_prices(
        mid_price=100.0,
        inventory=0.0,
        bid_probability=1.0,
        ask_probability=1.0,
    )
    # With p=1.0 everywhere, EV is maximized at the widest offset, but min
    # spread must still hold.
    assert (ask.price - bid.price) >= 0.0005 * 100.0 - 1e-9


def test_ev_picks_highest_ev_offset() -> None:
    class _Model:
        def predict(self, bids, asks, price, size, side):
            # Prefer wide offsets — fill_prob constant at 1.0
            return 1.0

    fake_bids: List[Tuple[float, float]] = [(99.95, 1.0)]
    fake_asks: List[Tuple[float, float]] = [(100.05, 1.0)]
    ev = _make_ev_maker(fill_model=_Model())
    bid, ask = ev.quote_prices(
        mid_price=100.0,
        inventory=0.0,
        bids=fake_bids,
        asks=fake_asks,
    )
    # With constant P=1, EV = offset * 1 is maximized at max offset.
    # So quotes should sit near the widest point of the search range.
    max_off = 0.005 * 100.0
    assert (100.0 - bid.price) > max_off * 0.5
    assert (ask.price - 100.0) > max_off * 0.5


def test_ev_uses_fallback_probabilities() -> None:
    ev = _make_ev_maker()
    bid_high, ask_high = ev.quote_prices(
        mid_price=100.0, inventory=0.0,
        bid_probability=1.0, ask_probability=1.0,
    )
    ev2 = _make_ev_maker()
    bid_low, ask_low = ev2.quote_prices(
        mid_price=100.0, inventory=0.0,
        bid_probability=0.001, ask_probability=0.001,
    )
    # High-prob → wider quotes (EV grows with offset). Low-prob → tighter.
    assert (ask_high.price - bid_high.price) >= (ask_low.price - bid_low.price)


def test_ev_invalid_config() -> None:
    with pytest.raises(ValueError):
        EVConfig(min_spread=0.0)
    with pytest.raises(ValueError):
        EVConfig(min_spread=0.01, max_spread=0.005)
    with pytest.raises(ValueError):
        EVConfig(num_points=1)


def test_ev_inventory_skew_affects_quotes() -> None:
    ev_flat = _make_ev_maker()
    b0, a0 = ev_flat.quote_prices(
        mid_price=100.0, inventory=0.0,
        bid_probability=0.5, ask_probability=0.5,
    )
    ev_long = _make_ev_maker()
    b1, a1 = ev_long.quote_prices(
        mid_price=100.0, inventory=0.5,
        bid_probability=0.5, ask_probability=0.5,
    )
    # Long → quote centre below mid → midpoint of (b1, a1) < midpoint of (b0, a0)
    assert (b1.price + a1.price) / 2 < (b0.price + a0.price) / 2
