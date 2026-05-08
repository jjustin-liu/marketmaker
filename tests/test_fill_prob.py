"""Tests for the fill probability model."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.fill_prob import (
    FEATURE_VECTOR_SIZE,
    FillFeatures,
    FillProbabilityModel,
)


# ---------- FillFeatures ----------


def test_fill_features_to_array_length_and_order() -> None:
    f = FillFeatures(
        bid_ask_spread=0.001,
        mid_price=100.0,
        bid_volume=10.0,
        ask_volume=12.0,
        imbalance_1=0.1,
        imbalance_2=0.2,
        imbalance_5=0.3,
        price_distance=0.0005,
        size=0.1,
        side=1.0,
    )
    arr = f.to_array()
    assert arr.shape == (FEATURE_VECTOR_SIZE,)
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(
        arr,
        np.array([0.001, 100.0, 10.0, 12.0, 0.1, 0.2, 0.3, 0.0005, 0.1, 1.0]),
    )


def test_extract_features_basic() -> None:
    bids = [(99.0, 1.0), (98.5, 2.0)]
    asks = [(100.0, 1.5), (100.5, 0.5)]
    f = FillProbabilityModel.extract_features(bids, asks, 99.0, 0.1, "buy")
    assert f.mid_price == pytest.approx(99.5)
    assert f.bid_volume == pytest.approx(3.0)
    assert f.ask_volume == pytest.approx(2.0)
    assert f.side == 1.0


def test_extract_features_rejects_bad_side() -> None:
    bids = [(99.0, 1.0)]
    asks = [(100.0, 1.0)]
    with pytest.raises(ValueError):
        FillProbabilityModel.extract_features(bids, asks, 99.0, 0.1, "neither")


def test_extract_features_empty_book_raises() -> None:
    with pytest.raises(ValueError):
        FillProbabilityModel.extract_features([], [(100.0, 1.0)], 99.0, 0.1, "buy")


# ---------- training fixtures ----------


def _synthetic_fills(n: int = 200) -> pd.DataFrame:
    """Build a synthetic fills dataframe.

    Each row is a real fill at a price near mid (positive label after
    train() generates negatives at worse prices).
    """
    rng = np.random.default_rng(123)
    rows = []
    for _ in range(n):
        mid = 100.0 + rng.normal(0.0, 0.1)
        spread = 0.02 + rng.uniform(0.0, 0.02)
        best_bid = mid - spread / 2
        best_ask = mid + spread / 2
        bids = [(best_bid - i * 0.01, 1.0 + rng.uniform(0, 1)) for i in range(5)]
        asks = [(best_ask + i * 0.01, 1.0 + rng.uniform(0, 1)) for i in range(5)]
        side = "buy" if rng.random() < 0.5 else "sell"
        # Real fills happen near mid — within half the spread.
        if side == "buy":
            price = best_bid + rng.uniform(0.0, spread * 0.4)
        else:
            price = best_ask - rng.uniform(0.0, spread * 0.4)
        rows.append({
            "bids": bids,
            "asks": asks,
            "price": price,
            "size": 0.01,
            "side": side,
        })
    return pd.DataFrame(rows)


# ---------- predict / train ----------


def test_predict_before_train_raises() -> None:
    model = FillProbabilityModel()
    with pytest.raises(RuntimeError):
        model.predict(
            [(99.0, 1.0)], [(100.0, 1.0)], 99.0, 0.1, "buy",
        )


def test_train_returns_auc_above_floor() -> None:
    model = FillProbabilityModel()
    auc = model.train(_synthetic_fills(300))
    assert 0.0 <= auc <= 1.0
    assert auc > 0.6


def test_predict_returns_probability() -> None:
    model = FillProbabilityModel()
    model.train(_synthetic_fills(300))
    p = model.predict(
        [(99.99, 1.0)], [(100.01, 1.0)], 99.99, 0.01, "buy",
    )
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_closer_to_mid_higher_fill_prob() -> None:
    model = FillProbabilityModel()
    model.train(_synthetic_fills(400))
    bids = [(99.99, 1.0), (99.98, 1.0)]
    asks = [(100.01, 1.0), (100.02, 1.0)]
    # buy near mid vs buy far below mid
    p_close = model.predict(bids, asks, 99.99, 0.01, "buy")
    p_far = model.predict(bids, asks, 99.50, 0.01, "buy")
    assert p_close > p_far


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model = FillProbabilityModel()
    model.train(_synthetic_fills(300))
    out = tmp_path / "model.joblib"
    model.save(out)
    assert out.exists()

    loaded = FillProbabilityModel.load(out)
    bids = [(99.99, 1.0)]
    asks = [(100.01, 1.0)]
    p1 = model.predict(bids, asks, 99.99, 0.01, "buy")
    p2 = loaded.predict(bids, asks, 99.99, 0.01, "buy")
    assert p1 == pytest.approx(p2)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FillProbabilityModel.load(tmp_path / "nope.joblib")


def test_predict_under_one_millisecond() -> None:
    model = FillProbabilityModel()
    model.train(_synthetic_fills(300))
    bids = [(99.99, 1.0)]
    asks = [(100.01, 1.0)]
    # warm up
    for _ in range(10):
        model.predict(bids, asks, 99.99, 0.01, "buy")
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(bids, asks, 99.99, 0.01, "buy")
    avg_ms = (time.perf_counter() - t0) * 1000.0 / n
    assert avg_ms < 1.0, f"predict avg {avg_ms:.3f}ms exceeds 1ms"


def test_train_rejects_missing_columns() -> None:
    df = pd.DataFrame({"bids": [[(99.0, 1.0)]], "asks": [[(100.0, 1.0)]]})
    model = FillProbabilityModel()
    with pytest.raises(ValueError):
        model.train(df)


def test_train_rejects_empty_df() -> None:
    df = pd.DataFrame(columns=["bids", "asks", "price", "size", "side"])
    model = FillProbabilityModel()
    with pytest.raises(ValueError):
        model.train(df)
