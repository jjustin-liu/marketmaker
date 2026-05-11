"""Backtest engine and metrics."""

from src.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    Diff,
    load_diffs_from_parquet,
)
from src.backtest.metrics import (
    Fill,
    calculate_adverse_selection,
    calculate_hit_ratio,
    calculate_max_drawdown,
    calculate_sharpe,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "Diff",
    "Fill",
    "calculate_adverse_selection",
    "calculate_hit_ratio",
    "calculate_max_drawdown",
    "calculate_sharpe",
    "load_diffs_from_parquet",
]
