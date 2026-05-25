"""Tests for the backtest engine and metrics."""

from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestEngine,
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
from src.lob.order_book import Side
from src.strategy.ev_maker import EVMaker
from src.strategy.inventory_skew import InventorySkew
from src.strategy.naive_maker import NaiveMaker, NaiveMakerConfig
from src.strategy.size_calculator import SizeCalculator


# ---------- metrics ----------


def test_sharpe_flat_series_is_zero() -> None:
    assert calculate_sharpe([1.0, 1.0, 1.0, 1.0]) == 0.0


def test_sharpe_short_series_is_zero() -> None:
    assert calculate_sharpe([1.0]) == 0.0
    assert calculate_sharpe([]) == 0.0


def test_sharpe_positive_for_rising_pnl() -> None:
    s = calculate_sharpe([0.0, 1.0, 2.0, 3.0, 4.0])
    # All-positive returns with zero variance → sharpe is 0 (div-by-zero
    # guard). Mix in a single small wobble to verify positivity.
    s2 = calculate_sharpe([0.0, 1.0, 2.05, 3.0, 4.0])
    assert s == 0.0
    assert s2 > 0.0


def test_hit_ratio_basic() -> None:
    assert calculate_hit_ratio(5, 10) == 0.5
    assert calculate_hit_ratio(0, 0) == 0.0
    assert calculate_hit_ratio(0, 10) == 0.0


def test_max_drawdown_basic() -> None:
    assert calculate_max_drawdown([0.0, 1.0, 2.0, 3.0]) == 0.0
    assert calculate_max_drawdown([1.0, 5.0, 2.0, 3.0]) == 3.0
    assert calculate_max_drawdown([]) == 0.0


def test_adverse_selection_no_fills() -> None:
    assert calculate_adverse_selection([], []) == 0.0


def test_adverse_selection_buy_loses_when_mid_falls() -> None:
    f = Fill(timestamp=1, side="buy", price=100.0, size=1.0, mid_at_fill=100.0)
    adv = calculate_adverse_selection([f], [99.0])
    assert adv == pytest.approx(-1.0)


def test_adverse_selection_sell_loses_when_mid_rises() -> None:
    f = Fill(timestamp=1, side="sell", price=100.0, size=1.0, mid_at_fill=100.0)
    adv = calculate_adverse_selection([f], [101.0])
    assert adv == pytest.approx(-1.0)


def test_adverse_selection_length_mismatch_raises() -> None:
    f = Fill(timestamp=1, side="buy", price=100.0, size=1.0, mid_at_fill=100.0)
    with pytest.raises(ValueError):
        calculate_adverse_selection([f], [])


# ---------- engine fill simulation ----------


def _seed_book_diffs() -> List[Diff]:
    """Establish a book with bids around 99 and asks around 101."""
    return [
        Diff(0, Side.BUY, 99.0, 5.0),
        Diff(0, Side.BUY, 98.5, 5.0),
        Diff(0, Side.SELL, 101.0, 5.0),
        Diff(0, Side.SELL, 101.5, 5.0),
    ]


def test_engine_bid_fills_when_ask_crosses() -> None:
    naive = NaiveMaker(NaiveMakerConfig(spread=0.04, size=0.1))
    eng = BacktestEngine(strategy=naive, refresh_every=1, use_inventory=False)
    diffs = _seed_book_diffs() + [
        # Crash the ask side: best ask drops below mid → below our bid
        Diff(1, Side.SELL, 101.0, 0.0),
        Diff(1, Side.SELL, 101.5, 0.0),
        Diff(1, Side.SELL, 95.0, 1.0),
    ]
    eng.run(diffs)
    assert any(f.side == "buy" for f in eng.fills)


def test_engine_ask_fills_when_bid_crosses() -> None:
    naive = NaiveMaker(NaiveMakerConfig(spread=0.04, size=0.1))
    eng = BacktestEngine(strategy=naive, refresh_every=1, use_inventory=False)
    diffs = _seed_book_diffs() + [
        Diff(1, Side.BUY, 99.0, 0.0),
        Diff(1, Side.BUY, 98.5, 0.0),
        Diff(1, Side.BUY, 105.0, 1.0),
    ]
    eng.run(diffs)
    assert any(f.side == "sell" for f in eng.fills)


def test_engine_on_tick_callback_fires_once_per_diff_and_flags_fills() -> None:
    naive = NaiveMaker(NaiveMakerConfig(spread=0.04, size=0.1))
    events: List[tuple] = []

    def hook(engine, new_fill: bool) -> None:
        events.append((engine.inventory, len(engine.fills), new_fill))

    eng = BacktestEngine(strategy=naive, refresh_every=1,
                         use_inventory=False, on_tick=hook)
    diffs = _seed_book_diffs() + [
        Diff(1, Side.SELL, 101.0, 0.0),
        Diff(1, Side.SELL, 101.5, 0.0),
        Diff(1, Side.SELL, 95.0, 1.0),
    ]
    eng.run(diffs)
    # at least a few events fire (only diffs producing a valid mid count)
    assert len(events) >= 3
    # exactly one event flagged new_fill=True when the ask crashed
    assert sum(1 for _, _, nf in events if nf) == 1


def test_engine_no_fill_when_price_doesnt_reach() -> None:
    naive = NaiveMaker(NaiveMakerConfig(spread=0.001, size=0.1))
    eng = BacktestEngine(strategy=naive, refresh_every=1, use_inventory=False)
    # Stable book: never crosses Naive's tight quotes
    diffs = _seed_book_diffs() * 5
    eng.run(diffs)
    assert len(eng.fills) == 0


def test_engine_records_pnl_history_and_quotes() -> None:
    naive = NaiveMaker()
    eng = BacktestEngine(strategy=naive, refresh_every=2, use_inventory=False)
    eng.run(_seed_book_diffs() * 4)
    assert eng.quote_count > 0
    assert len(eng.pnl_history) > 0


def test_engine_runs_ev_maker_through() -> None:
    ev = EVMaker(
        inventory_skew=InventorySkew(),
        size_calculator=SizeCalculator(),
    )
    eng = BacktestEngine(strategy=ev, refresh_every=2, use_inventory=True)
    result = eng.run(_seed_book_diffs() * 3)
    assert result.num_quotes > 0
    assert result.num_fills >= 0


# ---------- parquet loader ----------


def test_load_diffs_from_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "timestamp": [1, 2, 3],
        "side": ["buy", "sell", "buy"],
        "price": [100.0, 101.0, 99.5],
        "qty": [1.0, 2.0, 0.0],
    })
    p = tmp_path / "ticks.parquet"
    df.to_parquet(p)
    diffs = load_diffs_from_parquet(p)
    assert len(diffs) == 3
    assert diffs[0].side == Side.BUY
    assert diffs[1].side == Side.SELL


def test_load_diffs_missing_columns_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"timestamp": [1], "side": ["buy"]})
    p = tmp_path / "bad.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError):
        load_diffs_from_parquet(p)
