"""Unit tests for scripts.run_recorder bootstrap helpers."""

from __future__ import annotations

import pytest

from scripts.run_recorder import (
    SNAPSHOT_LIMIT,
    GapDetected,
    ParquetRotator,
    apply_rows_to_book,
    book_to_snapshot_rows,
    filter_buffered_diffs,
    parse_depth_message,
    parse_trade_message,
    resnapshot_due,
    snapshot_to_rows,
)


def _diff(U: int, u: int) -> tuple:
    return (U, u, [{"timestamp": 0, "side": "buy", "price": 1.0,
                    "qty": 1.0, "is_snapshot": False}])


def test_filter_drops_stale_buffered_diffs() -> None:
    buf = [_diff(10, 20), _diff(21, 30), _diff(31, 40)]
    out = filter_buffered_diffs(buf, snapshot_last_update_id=20)
    # snap=20, so diffs with u<=20 dropped; (21,30) straddles 21; ok
    assert [(U, u) for U, u, _ in out] == [(21, 30), (31, 40)]


def test_filter_first_diff_must_bridge_snap_plus_one() -> None:
    # snap=20, but smallest usable diff starts at U=25 — gap.
    buf = [_diff(10, 20), _diff(25, 30)]
    with pytest.raises(GapDetected):
        filter_buffered_diffs(buf, snapshot_last_update_id=20)


def test_filter_subsequent_diffs_must_be_contiguous() -> None:
    # First straddles fine; second skips u=30 -> U=32. Gap.
    buf = [_diff(10, 30), _diff(32, 40)]
    with pytest.raises(GapDetected):
        filter_buffered_diffs(buf, snapshot_last_update_id=25)


def test_filter_returns_empty_when_all_stale() -> None:
    buf = [_diff(1, 5), _diff(6, 10)]
    out = filter_buffered_diffs(buf, snapshot_last_update_id=20)
    assert out == []


def test_parse_depth_message_extracts_U_u_and_rows() -> None:
    payload = {
        "E": 1234,
        "U": 100, "u": 105,
        "b": [["77000.0", "1.5"], ["76999.0", "0.0"]],
        "a": [["77001.0", "2.0"]],
    }
    U, u, rows = parse_depth_message(payload)
    assert U == 100 and u == 105
    assert rows == [
        {"timestamp": 1234, "side": "buy", "price": 77000.0, "qty": 1.5,
         "is_snapshot": False},
        {"timestamp": 1234, "side": "buy", "price": 76999.0, "qty": 0.0,
         "is_snapshot": False},
        {"timestamp": 1234, "side": "sell", "price": 77001.0, "qty": 2.0,
         "is_snapshot": False},
    ]


def test_snapshot_to_rows_emits_one_row_per_level() -> None:
    snap = {
        "lastUpdateId": 999,
        "bids": [["100", "1"], ["99", "2"]],
        "asks": [["101", "3"]],
    }
    rows = snapshot_to_rows(snap, ts_ms=1700_000_000_000)
    assert len(rows) == 3
    assert rows[0] == {"timestamp": 1700_000_000_000, "side": "buy",
                       "price": 100.0, "qty": 1.0, "is_snapshot": True}
    assert rows[-1]["side"] == "sell"
    assert all(r["is_snapshot"] is True for r in rows)


def test_parse_trade_buyer_maker_is_sell_aggressor() -> None:
    # m=True → buyer is maker → a seller crossed → aggressor "sell".
    payload = {"e": "trade", "E": 9, "T": 1234, "p": "77000.5",
               "q": "0.25", "m": True}
    row = parse_trade_message(payload)
    assert row == {"timestamp": 1234, "side": "sell",
                   "price": 77000.5, "qty": 0.25}


def test_parse_trade_taker_buy_is_buy_aggressor() -> None:
    payload = {"e": "trade", "T": 5678, "p": "76999.0", "q": "1.0", "m": False}
    row = parse_trade_message(payload)
    assert row["side"] == "buy"
    assert row["timestamp"] == 5678


def test_trade_rotator_writes_separate_kind_file(tmp_path) -> None:
    rot = ParquetRotator(tmp_path, "btcusdt", kind="trades")
    rot.add({"timestamp": 1, "side": "buy", "price": 100.0, "qty": 0.5})
    n = rot.flush()
    assert n == 1
    files = list(tmp_path.glob("btcusdt_trades_*.parquet"))
    assert len(files) == 1


def test_book_to_snapshot_rows_drops_zero_qty_levels() -> None:
    book = {
        "buy": {77000.0: 1.5, 76999.0: 0.0},
        "sell": {77001.0: 2.0},
    }
    rows = book_to_snapshot_rows(book, ts_ms=42)
    # the qty=0 buy is dropped (snapshots list only live levels)
    assert len(rows) == 2
    assert all(r["is_snapshot"] is True for r in rows)
    assert all(r["qty"] > 0 for r in rows)
    sides = {r["side"] for r in rows}
    assert sides == {"buy", "sell"}


def test_snapshot_limit_is_5000() -> None:
    # REST /api/v3/depth max. A shallower snapshot leaves stale deep
    # levels in replay that never receive a qty=0 delete.
    assert SNAPSHOT_LIMIT == 5000


def test_resnapshot_due_threshold() -> None:
    assert resnapshot_due(now=6 * 3600.0, last_bootstrap=0.0,
                          resnapshot_hours=6.0) is True
    assert resnapshot_due(now=6 * 3600.0 - 1, last_bootstrap=0.0,
                          resnapshot_hours=6.0) is False


def test_resnapshot_due_disabled_when_zero() -> None:
    assert resnapshot_due(now=1e12, last_bootstrap=0.0,
                          resnapshot_hours=0.0) is False


def test_apply_rows_to_book_applies_snapshot_rows_and_deletes() -> None:
    book: dict = {"buy": {}, "sell": {}}
    rows = [
        {"timestamp": 1, "side": "buy", "price": 100.0, "qty": 2.0,
         "is_snapshot": True},
        {"timestamp": 1, "side": "sell", "price": 101.0, "qty": 1.0,
         "is_snapshot": True},
        {"timestamp": 2, "side": "buy", "price": 100.0, "qty": 0.0,
         "is_snapshot": False},
        {"timestamp": 2, "side": "buy", "price": 99.0, "qty": 3.0,
         "is_snapshot": False},
    ]
    n_diffs = apply_rows_to_book(book, rows)
    # snapshot rows seed the book, diff qty=0 deletes the seeded level
    assert book == {"buy": {99.0: 3.0}, "sell": {101.0: 1.0}}
    # only the two diff rows count toward tick statistics
    assert n_diffs == 2
