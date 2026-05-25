"""Unit tests for scripts.run_recorder bootstrap helpers."""

from __future__ import annotations

import pytest

from scripts.run_recorder import (
    GapDetected,
    filter_buffered_diffs,
    parse_depth_message,
    snapshot_to_rows,
)


def _diff(U: int, u: int) -> tuple:
    return (U, u, [{"timestamp": 0, "side": "buy", "price": 1.0, "qty": 1.0}])


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
        {"timestamp": 1234, "side": "buy", "price": 77000.0, "qty": 1.5},
        {"timestamp": 1234, "side": "buy", "price": 76999.0, "qty": 0.0},
        {"timestamp": 1234, "side": "sell", "price": 77001.0, "qty": 2.0},
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
                       "price": 100.0, "qty": 1.0}
    assert rows[-1]["side"] == "sell"
