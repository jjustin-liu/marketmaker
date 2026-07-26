"""Tests for the conditional-edge model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_model_lookahead import train_edge_from_labeled
from src.models.edge_model import EdgeModel

BIDS = [[100.0, 1.0], [99.9, 2.0]]
ASKS = [[100.1, 1.0], [100.2, 2.0]]


def _labeled_df(n: int = 400) -> pd.DataFrame:
    """Synthetic filled candidates where realized edge decays with
    distance from mid: tight fills keep value, far fills are toxic."""
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(n):
        offset = float(rng.uniform(0.01, 2.0))
        side = "buy" if rng.random() < 0.5 else "sell"
        mid = 100.05
        price = mid - offset if side == "buy" else mid + offset
        edge = 0.05 - offset + float(rng.normal(0, 0.005))
        rows.append({
            "bids": BIDS, "asks": ASKS, "price": price, "size": 0.001,
            "side": side, "_label": 1, "_edge": edge,
        })
    return pd.DataFrame(rows)


def test_predict_before_train_raises() -> None:
    with pytest.raises(RuntimeError):
        EdgeModel().predict(
            [(100.0, 1.0)], [(100.1, 1.0)], 100.0, 0.001, "buy",
        )


def test_edge_decays_with_distance_after_fit() -> None:
    model, r2 = train_edge_from_labeled(_labeled_df())
    assert r2 > 0.8  # near-deterministic synthetic relationship
    tight = model.predict(
        [tuple(x) for x in BIDS], [tuple(x) for x in ASKS],
        100.0, 0.001, "buy",
    )
    wide = model.predict(
        [tuple(x) for x in BIDS], [tuple(x) for x in ASKS],
        99.0, 0.001, "buy",
    )
    assert tight > wide
    assert wide < 0  # far fills are toxic in the synthetic data


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model, _ = train_edge_from_labeled(_labeled_df())
    path = model.save(tmp_path / "edge.joblib")
    loaded = EdgeModel.load(path)
    args = ([tuple(x) for x in BIDS], [tuple(x) for x in ASKS],
            99.8, 0.001, "buy")
    assert loaded.predict(*args) == pytest.approx(model.predict(*args))
    assert loaded.r2 == pytest.approx(model.r2)


def test_too_few_filled_rows_raises() -> None:
    with pytest.raises(RuntimeError):
        train_edge_from_labeled(_labeled_df(n=50))


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EdgeModel.load(tmp_path / "nope.joblib")
