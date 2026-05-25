"""Tests for the fill probability model."""

import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


# ---------- test fixture ----------


def _fit_test_model() -> FillProbabilityModel:
    """Fit a tiny logistic regression on a labeled feature array.

    Not training data for production — just enough signal for the
    predict/save/load tests to exercise a real fitted model. Labels
    are constructed so price_distance (column 7) is the dominant
    signal: closer to mid → higher fill probability.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0.0, 1.0, size=(n, FEATURE_VECTOR_SIZE))
    # Make column 7 (price_distance) monotonic with label inversion.
    X[:, 7] = rng.uniform(0.0, 1.0, size=n)
    y = (X[:, 7] < 0.5).astype(np.int64)

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000, random_state=42).fit(
        scaler.transform(X), y,
    )
    return FillProbabilityModel(model=model, scaler=scaler, auc=None)


# ---------- predict ----------


def test_predict_before_train_raises() -> None:
    model = FillProbabilityModel()
    with pytest.raises(RuntimeError):
        model.predict(
            [(99.0, 1.0)], [(100.0, 1.0)], 99.0, 0.1, "buy",
        )


def test_predict_returns_probability() -> None:
    model = _fit_test_model()
    p = model.predict(
        [(99.99, 1.0)], [(100.01, 1.0)], 99.99, 0.01, "buy",
    )
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_closer_to_mid_higher_fill_prob() -> None:
    model = _fit_test_model()
    bids = [(99.99, 1.0), (99.98, 1.0)]
    asks = [(100.01, 1.0), (100.02, 1.0)]
    p_close = model.predict(bids, asks, 99.99, 0.01, "buy")
    p_far = model.predict(bids, asks, 99.50, 0.01, "buy")
    assert p_close > p_far


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model = _fit_test_model()
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
    model = _fit_test_model()
    bids = [(99.99, 1.0)]
    asks = [(100.01, 1.0)]
    for _ in range(10):
        model.predict(bids, asks, 99.99, 0.01, "buy")
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(bids, asks, 99.99, 0.01, "buy")
    avg_ms = (time.perf_counter() - t0) * 1000.0 / n
    assert avg_ms < 1.0, f"predict avg {avg_ms:.3f}ms exceeds 1ms"
