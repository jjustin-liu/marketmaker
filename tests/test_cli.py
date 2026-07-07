"""Tests for src.cli strategy construction."""

from __future__ import annotations

import logging
from pathlib import Path

from src.cli import _build_strategy
from src.strategy.ev_maker import EVMaker


def test_build_strategy_falls_back_when_model_missing(
    tmp_path: Path, caplog,
) -> None:
    missing = tmp_path / "nope.joblib"
    with caplog.at_level(logging.WARNING, logger="cli"):
        strategy = _build_strategy(missing)
    assert isinstance(strategy, EVMaker)
    assert strategy.fill_model is None
    assert any(
        "uniform P(fill)=0.5" in rec.getMessage() for rec in caplog.records
    )


def test_build_strategy_falls_back_on_corrupt_model(tmp_path: Path) -> None:
    corrupt = tmp_path / "fill_prob.joblib"
    corrupt.write_bytes(b"not a joblib bundle")
    strategy = _build_strategy(corrupt)
    assert isinstance(strategy, EVMaker)
    assert strategy.fill_model is None
