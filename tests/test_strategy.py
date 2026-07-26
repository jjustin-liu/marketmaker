"""Tests for the strategy layer."""

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


def test_naive_never_crosses_on_tight_book() -> None:
    # Regression: a 1-cent-wide book must not produce an inverted,
    # self-filling quote. The market-anchored spread keeps it inside.
    m = NaiveMaker(NaiveMakerConfig(spread=0.001))
    bid, ask = m.quote_prices(76832.505, best_bid=76832.50, best_ask=76832.51)
    assert bid.price < ask.price
    # quote rests strictly inside the touch — never marketable
    assert 76832.50 < bid.price < ask.price < 76832.51


def test_naive_quotes_symmetric_inside_market() -> None:
    # market_spread=2, capped at 1.0, aggressiveness 0.2 → half=0.4
    m = NaiveMaker(NaiveMakerConfig(spread=0.05, aggressiveness=0.2))
    bid, ask = m.quote_prices(100.0, best_bid=99.0, best_ask=101.0)
    assert bid.price == pytest.approx(99.6)
    assert ask.price == pytest.approx(100.4)


def test_naive_aggressiveness_pulls_toward_mid() -> None:
    cfg_tight = NaiveMakerConfig(spread=0.05, aggressiveness=0.8)
    cfg_wide = NaiveMakerConfig(spread=0.05, aggressiveness=0.2)
    bid_t, _ = NaiveMaker(cfg_tight).quote_prices(100.0, 99.0, 101.0)
    bid_w, _ = NaiveMaker(cfg_wide).quote_prices(100.0, 99.0, 101.0)
    # higher aggressiveness → bid closer to mid (higher)
    assert bid_t.price > bid_w.price


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
    cfg = InventorySkewConfig(continuity_clip_bps=1.0)
    s = InventorySkew(cfg)
    b0, a0 = s.apply_skew(100.0, 0.0)
    # Big mid jump; per-refresh movement capped at 1 bps of mid
    b1, a1 = s.apply_skew(110.0, 0.0)
    limit = 110.0 * cfg.continuity_clip_bps / 10_000.0
    assert abs(b1 - b0) <= limit + 1e-9
    assert abs(a1 - a0) <= limit + 1e-9


def test_skew_clip_is_relative_tracks_price_scale() -> None:
    # Regression: an absolute $0.10 clip made quotes lag mid by dollars
    # on BTC-scale prices. At 5 bps of a 63k mid, one refresh may move
    # ~$31 — a $10 repricing must complete in a single refresh.
    cfg = InventorySkewConfig(continuity_clip_bps=5.0)
    s = InventorySkew(cfg)
    b0, _ = s.apply_skew(63_400.0, 0.0)
    b1, a1 = s.apply_skew(63_410.0, 0.0)
    mid_spread = a1 - b1
    assert b1 == pytest.approx(63_410.0 - mid_spread / 2.0, abs=1e-6)


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
        InventorySkewConfig(continuity_clip_bps=0.0)


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
        config=EVConfig(
            min_half_spread_mult=0.25, max_half_spread_mult=3.0, num_points=10,
        ),
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


def test_ev_min_spread_floor_enforced() -> None:
    # Even a degenerate near-zero market spread can't collapse the pair.
    ev = _make_ev_maker()
    bid, ask = ev.quote_prices(
        mid_price=100.0,
        inventory=0.0,
        bids=[(100.0, 1.0)],
        asks=[(100.0, 1.0)],  # zero market spread → fallback + floor
        bid_probability=1.0,
        ask_probability=1.0,
    )
    floor = EVConfig().min_spread_bps * 100.0 / 10_000.0
    assert (ask.price - bid.price) >= floor - 1e-12


def test_ev_anchors_to_market_spread() -> None:
    # A wider live book → proportionally wider EV quotes (scale-aware).
    # Spreads sit under the anchor cap (0.05 bps of mid = 0.50 here)
    # so the anchor tracks the live book, not the ceiling.
    class _Flat:
        def predict(self, bids, asks, price, size, side):
            return 1.0

    mid = 100_000.0
    ev = _make_ev_maker(fill_model=_Flat())
    b_tight, a_tight = ev.quote_prices(
        mid_price=mid, inventory=0.0,
        bids=[(mid - 0.05, 1.0)], asks=[(mid + 0.05, 1.0)],  # 0.10 spread
    )
    ev2 = _make_ev_maker(fill_model=_Flat())
    b_wide, a_wide = ev2.quote_prices(
        mid_price=mid, inventory=0.0,
        bids=[(mid - 0.20, 1.0)], asks=[(mid + 0.20, 1.0)],  # 0.40 spread
    )
    assert (a_wide.price - b_wide.price) > (a_tight.price - b_tight.price)


def test_ev_constant_prob_picks_widest() -> None:
    # Flat fill prob → EV = P*h monotonic in h → argmax at the widest.
    class _Flat:
        def predict(self, bids, asks, price, size, side):
            return 1.0

    mid = 100_000.0
    ev = _make_ev_maker(fill_model=_Flat())
    bid, ask = ev.quote_prices(
        mid_price=mid, inventory=0.0,
        bids=[(mid - 0.05, 1.0)], asks=[(mid + 0.05, 1.0)],
    )
    market_spread = 0.10
    # widest half-spread = max_half_spread_mult (3.0) * market_spread
    assert (mid - bid.price) == pytest.approx(3.0 * market_spread)
    assert (ask.price - mid) == pytest.approx(3.0 * market_spread)


def test_ev_decaying_prob_quotes_tight() -> None:
    # Fill prob that decays fast with distance from mid → EV argmax at a
    # tight half-spread, not the widest. This is the corrected tradeoff.
    class _Decay:
        def predict(self, bids, asks, price, size, side):
            mid = (bids[0][0] + asks[0][0]) / 2.0
            dist = abs(price - mid)
            return max(0.0, 1.0 - 50.0 * dist)  # sharp decay

    ev = _make_ev_maker(fill_model=_Decay())
    bid, ask = ev.quote_prices(
        mid_price=100.0, inventory=0.0,
        bids=[(99.95, 1.0)], asks=[(100.05, 1.0)],
    )
    market_spread = 0.10
    # chosen half-spread well below the widest (3 * market_spread)
    assert (100.0 - bid.price) < 3.0 * market_spread
    assert (ask.price - bid.price) > 0.0


def test_ev_high_inventory_widens_quotes() -> None:
    # Risk term: heavier inventory → wider band → wider final spread.
    class _Flat:
        def predict(self, bids, asks, price, size, side):
            return 1.0

    book_bids = [(99.95, 1.0)]
    book_asks = [(100.05, 1.0)]
    ev = _make_ev_maker(fill_model=_Flat())
    b0, a0 = ev.quote_prices(100.0, inventory=0.0,
                             bids=book_bids, asks=book_asks)
    ev2 = _make_ev_maker(fill_model=_Flat())
    b1, a1 = ev2.quote_prices(100.0, inventory=0.5,
                              bids=book_bids, asks=book_asks)
    assert (a1.price - b1.price) > (a0.price - b0.price)


def test_ev_high_volatility_widens_quotes() -> None:
    class _Flat:
        def predict(self, bids, asks, price, size, side):
            return 1.0

    book_bids = [(99.95, 1.0)]
    book_asks = [(100.05, 1.0)]
    ev = _make_ev_maker(fill_model=_Flat())
    b0, a0 = ev.quote_prices(100.0, inventory=0.0,
                             bids=book_bids, asks=book_asks, volatility=0.0)
    ev2 = _make_ev_maker(fill_model=_Flat())
    b1, a1 = ev2.quote_prices(100.0, inventory=0.0,
                              bids=book_bids, asks=book_asks, volatility=0.5)
    assert (a1.price - b1.price) > (a0.price - b0.price)


def test_ev_invalid_config() -> None:
    with pytest.raises(ValueError):
        EVConfig(min_half_spread_mult=0.0)
    with pytest.raises(ValueError):
        EVConfig(inv_risk_factor=-1.0)
    with pytest.raises(ValueError):
        EVConfig(vol_risk_factor=-1.0)
    with pytest.raises(ValueError):
        EVConfig(min_half_spread_mult=3.0, max_half_spread_mult=1.0)
    with pytest.raises(ValueError):
        EVConfig(num_points=1)
    with pytest.raises(ValueError):
        EVConfig(min_spread_bps=0.0)


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


def test_ev_never_quotes_through_the_touch() -> None:
    # A hard inventory skew pushes the centre above the best ask; the
    # final bid must still rest below it (a bid at/above the ask would
    # be a marketable order, not a quote).
    ev = EVMaker(
        inventory_skew=InventorySkew(InventorySkewConfig(
            max_position=0.01, skew_factor=100.0,
        )),
        size_calculator=SizeCalculator(),
        fill_model=None,
        config=EVConfig(inv_risk_factor=0.0, vol_risk_factor=0.0),
    )
    best_bid, best_ask = 100.00, 100.05
    bid, ask = ev.quote_prices(
        mid_price=(best_bid + best_ask) / 2.0,
        inventory=-0.01,  # pinned short → centre shifted far up
        bids=[(best_bid, 1.0)],
        asks=[(best_ask, 1.0)],
    )
    assert bid.price < best_ask
    assert ask.price > best_bid
    assert ask.price > bid.price


def test_ev_anchor_capped_on_transient_wide_spread() -> None:
    # A mid-event book with its touch consumed can show a $7+ "spread".
    # The candidate band must anchor to the capped value, not the spike,
    # or quotes park dollars away where only toxic sweeps fill them.
    ev = _make_ev_maker()
    mid = 63_400.0
    cap = mid * EVConfig().max_anchor_spread_bps / 10_000.0
    bid, ask = ev.quote_prices(
        mid_price=mid,
        inventory=0.0,
        bids=[(mid - 25.0, 1.0)],
        asks=[(mid + 25.0, 1.0)],  # transient $50 spread
        bid_probability=1.0,
        ask_probability=1.0,
    )
    max_half = 3.0 * cap  # band top at max_half_spread_mult x capped anchor
    assert (ask.price - bid.price) / 2.0 <= max_half + 1e-9


def test_ev_edge_model_overrides_geometric_edge() -> None:
    # Flat P with geometric edge picks the widest candidate (see
    # test_ev_constant_prob_picks_widest). A conditional-edge model that
    # says distance is toxic must flip the argmax to the tightest.
    class _FlatP:
        def predict(self, bids, asks, price, size, side):
            return 0.5

    class _TightEdge:
        def predict(self, bids, asks, price, size, side):
            mid = (bids[0][0] + asks[0][0]) / 2.0
            return 0.01 - abs(price - mid)

    mid = 100_000.0
    ev = EVMaker(
        inventory_skew=InventorySkew(),
        size_calculator=SizeCalculator(),
        fill_model=_FlatP(),
        edge_model=_TightEdge(),
        config=EVConfig(
            min_half_spread_mult=0.25, max_half_spread_mult=3.0,
            num_points=10, inv_risk_factor=0.0, vol_risk_factor=0.0,
        ),
    )
    bid, ask = ev.quote_prices(
        mid_price=mid, inventory=0.0,
        bids=[(mid - 0.05, 1.0)], asks=[(mid + 0.05, 1.0)],
    )
    # tightest candidate = min_half_spread_mult (0.25) x 0.10 spread
    assert (mid - bid.price) == pytest.approx(0.25 * 0.10)
    assert (ask.price - mid) == pytest.approx(0.25 * 0.10)
