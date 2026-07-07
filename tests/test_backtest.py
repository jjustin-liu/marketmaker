"""Tests for the backtest engine and metrics."""

from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestEngine,
    Diff,
    load_diffs_from_parquet,
    load_trades_from_parquet,
)
from src.backtest.metrics import (
    Fill,
    calculate_adverse_selection,
    calculate_hit_ratio,
    calculate_max_drawdown,
    calculate_sharpe,
)
from src.lob.order_book import OrderBook, Side
from src.strategy.ev_maker import EVMaker
from src.strategy.inventory_skew import InventorySkew
from src.strategy.naive_maker import NaiveMaker, NaiveMakerConfig, Quote
from src.strategy.size_calculator import SizeCalculator


class _FixedStrategy:
    """Quotes a fixed bid/ask regardless of context — for fill-model tests."""

    def __init__(self, bid: float, ask: float, size: float = 1.0) -> None:
        self._bid, self._ask, self._size = bid, ask, size

    def quote_prices(self, mid_price, best_bid=None, best_ask=None, **kw):
        return Quote(self._bid, self._size), Quote(self._ask, self._size)


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


def _physical_down_move() -> List[Diff]:
    """Clean down-move: clear the book, then rebuild far below at bid 94 /
    ask 95. The book stays uncrossed throughout; the market's best ask ends
    up below a stale bid quote so the strict-cross fill model fires.
    """
    return [
        Diff(1, Side.BUY, 99.0, 0.0),
        Diff(1, Side.BUY, 98.5, 0.0),
        Diff(1, Side.SELL, 101.0, 0.0),
        Diff(1, Side.SELL, 101.5, 0.0),
        Diff(1, Side.BUY, 94.0, 1.0),
        Diff(1, Side.SELL, 95.0, 1.0),
    ]


def _physical_up_move() -> List[Diff]:
    """Clean up-move: clear the book, rebuild far above at bid 105 / ask 106.
    Market best bid ends up above a stale ask quote → ask fills.
    """
    return [
        Diff(1, Side.SELL, 101.0, 0.0),
        Diff(1, Side.SELL, 101.5, 0.0),
        Diff(1, Side.BUY, 99.0, 0.0),
        Diff(1, Side.BUY, 98.5, 0.0),
        Diff(1, Side.SELL, 106.0, 1.0),
        Diff(1, Side.BUY, 105.0, 1.0),
    ]


def test_engine_bid_fills_when_market_drops_through_stale_quote() -> None:
    naive = NaiveMaker(NaiveMakerConfig(spread=0.04, size=0.1))
    # refresh_every=3 sets the quote once at the seed (mid 100, bid ~98)
    # and keeps it stale through the down-move so the cross fires.
    eng = BacktestEngine(strategy=naive, refresh_every=3, use_inventory=False)
    eng.run(_seed_book_diffs() + _physical_down_move())
    assert any(f.side == "buy" for f in eng.fills)
    # the book was never crossed during the fill
    assert eng.crossed_skips == 0


def test_engine_ask_fills_when_market_rises_through_stale_quote() -> None:
    naive = NaiveMaker(NaiveMakerConfig(spread=0.04, size=0.1))
    eng = BacktestEngine(strategy=naive, refresh_every=3, use_inventory=False)
    eng.run(_seed_book_diffs() + _physical_up_move())
    assert any(f.side == "sell" for f in eng.fills)
    assert eng.crossed_skips == 0


def test_engine_skips_crossed_book_ticks() -> None:
    # Leaving a stale bid at 99 while the ask crashes to 95 is a non-physical
    # crossed state — the engine must skip it, not fill or quote on it.
    naive = NaiveMaker(NaiveMakerConfig(spread=0.04, size=0.1))
    eng = BacktestEngine(strategy=naive, refresh_every=1, use_inventory=False)
    eng.run(_seed_book_diffs() + [
        Diff(1, Side.SELL, 101.0, 0.0),
        Diff(1, Side.SELL, 101.5, 0.0),
        Diff(1, Side.SELL, 95.0, 1.0),  # ask below stale bid 99 → crossed
    ])
    assert eng.crossed_skips >= 1


def test_queue_mode_touch_fills_bid_when_strict_would_not() -> None:
    # Quote a bid at 100 (inside the 99/101 spread). An ask arriving at
    # exactly 100 touches it: queue mode fills (best_ask <= price), strict
    # mode does not (needs best_ask < price).
    strat = _FixedStrategy(bid=100.0, ask=100.5)
    diffs = _seed_book_diffs() + [Diff(1, Side.SELL, 100.0, 1.0)]

    q = BacktestEngine(strategy=strat, refresh_every=1,
                       use_inventory=False, fill_mode="queue")
    q.run(diffs)
    s = BacktestEngine(strategy=strat, refresh_every=1,
                       use_inventory=False, fill_mode="strict_cross")
    s.run(diffs)

    assert any(f.side == "buy" for f in q.fills)
    assert not any(f.side == "buy" for f in s.fills)


def test_queue_mode_fills_as_resting_level_depletes() -> None:
    # A bid resting on a real level (99, depth 5) fills as that depth is
    # consumed past our queue position — no price move through us required.
    eng = BacktestEngine(strategy=_FixedStrategy(99.0, 101.0),
                         use_inventory=False, fill_mode="queue")
    book = OrderBook()
    book.apply_diff(Side.BUY, 99.0, 5.0)
    book.apply_diff(Side.SELL, 101.0, 5.0)
    eng.book = book
    eng.open_bid = Quote(99.0, 1.0)
    eng.open_ask = None
    eng._bid_resting = True
    eng._bid_queue = 5.0
    eng._bid_level_qty = 5.0
    fills_idx: List[int] = []

    book.apply_diff(Side.BUY, 99.0, 1.0)  # 5 → 1, consumed 4 (queue 5→1)
    eng._maybe_fill_queue(1, 99.0, 101.0, 100.0, fills_idx, 0)
    assert not eng.fills  # queue not yet cleared

    book.apply_diff(Side.BUY, 99.0, 0.0)  # 1 → 0, consumed 1 (queue → 0)
    eng._maybe_fill_queue(2, 98.5, 101.0, 99.75, fills_idx, 1)
    assert any(f.side == "buy" for f in eng.fills)


def test_trades_mode_bid_fills_on_sell_print_at_price() -> None:
    # A sell print at/through our resting bid fills it, with no book sweep.
    from src.backtest.engine import Trade
    strat = _FixedStrategy(bid=100.0, ask=100.5)
    diffs = _seed_book_diffs() + [Diff(5, Side.BUY, 99.0, 5.0)]
    trades = [Trade(timestamp=5, side="sell", price=99.9, qty=1.0)]  # <= 100
    eng = BacktestEngine(strategy=strat, refresh_every=1,
                         use_inventory=False, fill_mode="trades")
    eng.run(diffs, trades=trades)
    assert any(f.side == "buy" for f in eng.fills)


def test_trades_mode_ask_fills_on_buy_print_and_buy_does_not_fill_bid() -> None:
    from src.backtest.engine import Trade
    strat = _FixedStrategy(bid=99.5, ask=100.0)
    diffs = _seed_book_diffs() + [Diff(5, Side.SELL, 101.0, 5.0)]
    # buy print at 100.1 >= our ask 100.0 → ask fills; bid (99.5) untouched
    trades = [Trade(timestamp=5, side="buy", price=100.1, qty=1.0)]
    eng = BacktestEngine(strategy=strat, refresh_every=1,
                         use_inventory=False, fill_mode="trades")
    eng.run(diffs, trades=trades)
    assert any(f.side == "sell" for f in eng.fills)
    assert not any(f.side == "buy" for f in eng.fills)


def test_trades_mode_print_away_from_quote_does_not_fill() -> None:
    from src.backtest.engine import Trade
    strat = _FixedStrategy(bid=99.0, ask=101.0)
    diffs = _seed_book_diffs() + [Diff(5, Side.BUY, 99.0, 5.0)]
    # sell print at 100.5 is above our bid 99.0 → no fill
    trades = [Trade(timestamp=5, side="sell", price=100.5, qty=1.0)]
    eng = BacktestEngine(strategy=strat, refresh_every=1,
                         use_inventory=False, fill_mode="trades")
    eng.run(diffs, trades=trades)
    assert eng.fills == []


def test_engine_on_tick_callback_fires_once_per_diff_and_flags_fills() -> None:
    naive = NaiveMaker(NaiveMakerConfig(spread=0.04, size=0.1))
    events: List[tuple] = []

    def hook(engine, new_fill: bool) -> None:
        events.append((engine.inventory, len(engine.fills), new_fill))

    eng = BacktestEngine(strategy=naive, refresh_every=3,
                         use_inventory=False, on_tick=hook)
    eng.run(_seed_book_diffs() + _physical_down_move())
    # at least a few events fire (only uncrossed diffs with a valid mid count)
    assert len(events) >= 3
    # exactly one event flagged new_fill=True when the market dropped through
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


def test_engine_passes_depth_to_inventory_strategy() -> None:
    class DepthSpyStrategy:
        def __init__(self) -> None:
            self.calls: List[dict] = []

        def quote_prices(self, **kwargs):
            self.calls.append(kwargs)
            return (
                NaiveMaker(NaiveMakerConfig(spread=0.001)).quote_prices(
                    kwargs["mid_price"],
                )
            )

    strategy = DepthSpyStrategy()
    eng = BacktestEngine(strategy=strategy, refresh_every=1, use_inventory=True)
    eng.run(_seed_book_diffs() * 2)

    assert strategy.calls
    call = strategy.calls[0]
    assert "bid_probability" not in call
    assert "ask_probability" not in call
    assert call["bids"][0] == (99.0, 5.0)
    assert call["asks"][0] == (101.0, 5.0)


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


def test_load_trades_from_parquet_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "timestamp": [1, 2, 3],
        "side": ["sell", "buy", "sell"],
        "price": [100.0, 100.5, 99.5],
        "qty": [0.1, 0.2, 0.3],
    })
    p = tmp_path / "btcusdt_trades_2026-05-26.parquet"
    df.to_parquet(p)
    trades = load_trades_from_parquet(p)
    assert len(trades) == 3
    assert [t.side for t in trades] == ["sell", "buy", "sell"]
    assert trades[1].price == 100.5 and trades[1].qty == 0.2


def test_load_trades_missing_columns_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"timestamp": [1], "side": ["buy"]})
    p = tmp_path / "bad_trades.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError):
        load_trades_from_parquet(p)


def test_load_diffs_missing_columns_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"timestamp": [1], "side": ["buy"]})
    p = tmp_path / "bad.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError):
        load_diffs_from_parquet(p)


def test_load_diffs_preserves_file_order_with_is_snapshot(tmp_path: Path) -> None:
    # File order is preserved so epoch boundaries are correctly detected
    # by position (snapshot block after diffs = reconnect epoch).
    df = pd.DataFrame({
        "timestamp": [10, 1, 11, 1, 12],
        "side":      ["buy", "buy", "sell", "sell", "buy"],
        "price":     [100.0, 99.0, 101.0, 102.0, 100.0],
        "qty":       [0.5, 1.0, 0.3, 2.0, 0.0],
        "is_snapshot": [False, True, False, True, False],
    })
    p = tmp_path / "rotated.parquet"
    df.to_parquet(p)
    diffs = load_diffs_from_parquet(p)
    assert len(diffs) == 5
    # Order matches file exactly
    assert [d.price for d in diffs] == [100.0, 99.0, 101.0, 102.0, 100.0]
    # is_snapshot flag propagated
    assert [d.is_snapshot for d in diffs] == [False, True, False, True, False]


def test_load_diffs_back_compat_without_is_snapshot(tmp_path: Path) -> None:
    # Old parquets have no is_snapshot column. Behavior unchanged:
    # rows returned in file order.
    df = pd.DataFrame({
        "timestamp": [1, 2, 3],
        "side": ["buy", "sell", "buy"],
        "price": [100.0, 101.0, 99.5],
        "qty": [1.0, 2.0, 0.0],
    })
    p = tmp_path / "legacy.parquet"
    df.to_parquet(p)
    diffs = load_diffs_from_parquet(p)
    assert [d.timestamp for d in diffs] == [1, 2, 3]
