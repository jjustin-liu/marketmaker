"""Tests for the feature generation modules."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.features.feature_generator import FEATURE_VECTOR_SIZE, FeatureGenerator
from src.features.imbalance import calculate_imbalance, get_imbalance_features
from src.features.micro_price import calculate_microprice
from src.features.volatility import VolatilityCalculator


# ---------- imbalance ----------


def test_imbalance_zero_when_balanced() -> None:
    bids = [(99.0, 1.0)]
    asks = [(100.0, 1.0)]
    assert calculate_imbalance(bids, asks, 1) == 0.0


def test_imbalance_plus_one_when_no_asks() -> None:
    bids = [(99.0, 5.0)]
    asks = []
    assert calculate_imbalance(bids, asks, 1) == 1.0


def test_imbalance_minus_one_when_no_bids() -> None:
    bids = []
    asks = [(100.0, 5.0)]
    assert calculate_imbalance(bids, asks, 1) == -1.0


def test_imbalance_empty_returns_zero() -> None:
    assert calculate_imbalance([], [], 1) == 0.0


def test_imbalance_aggregates_levels() -> None:
    bids = [(99.0, 1.0), (98.0, 4.0)]  # top-2 vol = 5.0
    asks = [(100.0, 2.0), (101.0, 1.0)]  # top-2 vol = 3.0
    # (5 - 3) / (5 + 3) = 0.25
    assert calculate_imbalance(bids, asks, 2) == pytest.approx(0.25)


def test_get_imbalance_features_returns_three_depths() -> None:
    bids = [(99.0, 1.0)] * 5
    asks = [(100.0, 1.0)] * 5
    feats = get_imbalance_features(bids, asks)
    assert set(feats) == {"imbalance_1", "imbalance_2", "imbalance_5"}
    assert all(v == 0.0 for v in feats.values())


def test_imbalance_invalid_levels_raises() -> None:
    with pytest.raises(ValueError):
        calculate_imbalance([(99.0, 1.0)], [(100.0, 1.0)], 0)


# ---------- microprice ----------


def test_microprice_pulls_toward_thin_ask() -> None:
    bids = [(99.0, 10.0)]   # fat bid
    asks = [(100.0, 1.0)]   # thin ask
    mp = calculate_microprice(bids, asks)
    # microprice = (99*1 + 100*10) / 11 = 1099/11 ≈ 99.909
    # Note: pulls toward 100 (thin ask) — closer to 100 than simple mid 99.5
    assert mp == pytest.approx(99.909, abs=0.01)
    assert mp > 99.5  # past the simple mid, toward thin side


def test_microprice_pulls_toward_thin_bid() -> None:
    bids = [(99.0, 1.0)]    # thin bid
    asks = [(100.0, 10.0)]  # fat ask
    mp = calculate_microprice(bids, asks)
    # (99*10 + 100*1) / 11 = 1090/11 ≈ 99.09
    assert mp == pytest.approx(99.0909, abs=0.01)
    assert mp < 99.5  # below simple mid, toward thin bid side


def test_microprice_balanced_equals_simple_mid() -> None:
    bids = [(99.0, 5.0)]
    asks = [(100.0, 5.0)]
    assert calculate_microprice(bids, asks) == pytest.approx(99.5)


def test_microprice_empty_side_returns_none() -> None:
    assert calculate_microprice([], [(100.0, 1.0)]) is None
    assert calculate_microprice([(99.0, 1.0)], []) is None


# ---------- volatility ----------


def test_volatility_zero_with_fewer_than_two_prices() -> None:
    vc = VolatilityCalculator()
    assert vc.update(100.0) == 0.0


def test_volatility_zero_for_constant_prices() -> None:
    vc = VolatilityCalculator()
    for _ in range(20):
        vc.update(100.0)
    assert vc.value() == 0.0


def test_volatility_grows_with_noisier_series() -> None:
    calm = VolatilityCalculator()
    noisy = VolatilityCalculator()
    base = 100.0
    for i in range(50):
        calm.update(base + i * 0.0001)        # near-flat
        noisy.update(base + (i % 2) * 5.0)    # oscillating ±5
    assert noisy.value() > calm.value()


def test_volatility_window_size_validation() -> None:
    with pytest.raises(ValueError):
        VolatilityCalculator(window_size=1)


def test_volatility_reset() -> None:
    vc = VolatilityCalculator()
    for p in [100.0, 101.0, 99.0, 102.0]:
        vc.update(p)
    assert vc.value() > 0.0
    vc.reset()
    assert vc.value() == 0.0


# ---------- feature generator ----------


def make_redis(bids, asks) -> MagicMock:
    r = MagicMock()
    r.get.side_effect = lambda key: {
        "lob:levels:bids": json.dumps(bids),
        "lob:levels:asks": json.dumps(asks),
    }.get(key)
    return r


def test_feature_vector_length_and_order() -> None:
    bids = [[99.0, 1.0], [98.5, 2.0]]
    asks = [[100.0, 1.5], [100.5, 0.5]]
    fg = FeatureGenerator(make_redis(bids, asks))

    vec = fg.generate(candidate_price=99.0, candidate_size=0.1,
                      candidate_side="buy")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (FEATURE_VECTOR_SIZE,)
    # canonical ordering:
    # [spread, mid, bid_vol, ask_vol, imb_1, imb_2, imb_5,
    #  price_distance, size, side]
    spread, mid, bv, av, i1, i2, i5, pd, sz, side = vec
    assert mid == pytest.approx(99.5)
    assert bv == pytest.approx(3.0)
    assert av == pytest.approx(2.0)
    assert spread == pytest.approx((100.0 - 99.0) / 99.5)
    assert sz == 0.1
    assert side == 1.0  # buy


def test_feature_vector_side_sell_is_zero() -> None:
    bids = [[99.0, 1.0]]
    asks = [[100.0, 1.0]]
    fg = FeatureGenerator(make_redis(bids, asks))
    vec = fg.generate(candidate_price=100.0, candidate_size=0.1,
                      candidate_side="sell")
    assert vec[-1] == 0.0


def test_feature_generator_invalid_side() -> None:
    fg = FeatureGenerator(make_redis([[99.0, 1.0]], [[100.0, 1.0]]))
    with pytest.raises(ValueError):
        fg.generate(99.0, 0.1, "neither")


def test_feature_generator_missing_lob_raises() -> None:
    r = MagicMock()
    r.get.return_value = None
    fg = FeatureGenerator(r)
    with pytest.raises(RuntimeError):
        fg.generate(99.0, 0.1, "buy")


def test_feature_generator_updates_volatility() -> None:
    bids = [[99.0, 1.0]]
    asks = [[100.0, 1.0]]
    fg = FeatureGenerator(make_redis(bids, asks))
    for _ in range(5):
        fg.generate(99.0, 0.1, "buy")
    # Mid is constant across calls — vol stays 0
    assert fg.volatility == 0.0
