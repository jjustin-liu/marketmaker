"""Tests for the short-horizon alpha model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.alpha_model import (
    AlphaFeatures,
    AlphaModel,
    extract_alpha_features,
)


def test_features_to_array_order() -> None:
    f = AlphaFeatures(
        ofi=1.0, microprice_dev=2.0, imbalance_1=3.0,
        imbalance_2=4.0, imbalance_5=5.0,
    )
    assert list(f.to_array()) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_extract_features_bid_heavy_book() -> None:
    bids = [(100.0, 10.0), (99.0, 5.0)]
    asks = [(101.0, 1.0), (102.0, 1.0)]
    f = extract_alpha_features(bids, asks, ofi=7.0)
    assert f.ofi == 7.0
    # Heavier bid -> microprice above mid -> positive deviation.
    assert f.microprice_dev > 0.0
    assert f.imbalance_1 > 0.0


def test_extract_features_empty_book_raises() -> None:
    with pytest.raises(ValueError):
        extract_alpha_features([], [(101.0, 1.0)], ofi=0.0)


def test_predict_before_train_raises() -> None:
    with pytest.raises(RuntimeError):
        AlphaModel().predict(
            AlphaFeatures(0.0, 0.0, 0.0, 0.0, 0.0)
        )


def _fit_synthetic() -> AlphaModel:
    # Label perfectly correlated with the OFI feature.
    rng = np.random.default_rng(0)
    ofi = rng.normal(size=2000)
    x = np.column_stack([ofi, rng.normal(size=2000) * 0.01,
                         np.zeros(2000), np.zeros(2000), np.zeros(2000)])
    y = ofi * 1e-4
    scaler = StandardScaler()
    model = Ridge(alpha=1.0).fit(scaler.fit_transform(x), y)
    return AlphaModel(model=model, scaler=scaler, ic=0.99)


def test_predict_monotone_in_ofi() -> None:
    m = _fit_synthetic()
    low = m.predict(AlphaFeatures(-2.0, 0.0, 0.0, 0.0, 0.0))
    high = m.predict(AlphaFeatures(2.0, 0.0, 0.0, 0.0, 0.0))
    assert high > low


def test_predict_from_book_matches_predict() -> None:
    m = _fit_synthetic()
    bids = [(100.0, 4.0)]
    asks = [(101.0, 4.0)]
    direct = m.predict(extract_alpha_features(bids, asks, ofi=1.5))
    via_book = m.predict_from_book(bids, asks, ofi=1.5)
    assert direct == pytest.approx(via_book)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    m = _fit_synthetic()
    p = m.save(tmp_path / "alpha.joblib")
    loaded = AlphaModel.load(p)
    feats = AlphaFeatures(1.0, 0.0, 0.0, 0.0, 0.0)
    assert loaded.predict(feats) == pytest.approx(m.predict(feats))
    assert loaded.ic == m.ic


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AlphaModel.load(tmp_path / "nope.joblib")
