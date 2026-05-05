"""Tests for the data feed: bootstrap, redis writer, message parsers."""

import json
from unittest.mock import MagicMock

import pytest

from src.data_feed.binance_ws import (
    parse_depth_diff,
    parse_snapshot,
    parse_trade,
)
from src.data_feed.bootstrap import (
    BookSynchronizer,
    DepthDiff,
    GapDetectedError,
    Snapshot,
)
from src.data_feed.redis_writer import RedisWriter
from src.lob import OrderBook, Side


# ---------- bootstrap ----------


def make_sync() -> tuple[BookSynchronizer, OrderBook]:
    book = OrderBook()
    sync = BookSynchronizer(book, bid_side=Side.BUY, ask_side=Side.SELL)
    return sync, book


def test_snapshot_only_no_buffered_diffs() -> None:
    sync, book = make_sync()
    snap = Snapshot(
        last_update_id=100,
        bids=[(99.0, 1.0), (98.0, 2.0)],
        asks=[(101.0, 1.5)],
    )
    sync.apply_snapshot(snap)
    assert sync.synced
    assert sync.last_update_id == 100
    assert book.best_bid() == 99.0
    assert book.best_ask() == 101.0


def test_buffered_diffs_replayed_after_snapshot() -> None:
    sync, book = make_sync()
    # Buffer arrives first; snapshot lastUpdateId=100.
    sync.buffer(DepthDiff(95, 99, [], []))   # entirely pre-snapshot, discard
    sync.buffer(DepthDiff(99, 101, [(99.0, 5.0)], []))  # straddles 101
    sync.buffer(DepthDiff(102, 103, [], [(101.0, 0.0)]))  # delete ask

    snap = Snapshot(
        last_update_id=100,
        bids=[(99.0, 1.0)],
        asks=[(101.0, 1.5)],
    )
    sync.apply_snapshot(snap)
    assert sync.synced
    assert sync.last_update_id == 103
    # First buffered diff bumped 99 to 5; second wiped the ask.
    assert book.depth(Side.BUY, 1) == [(99.0, 5.0)]
    assert book.best_ask() is None


def test_apply_snapshot_raises_when_no_bridge_diff() -> None:
    sync, _ = make_sync()
    # Buffered diff starts at 105 — too far ahead of snapshot at 100.
    sync.buffer(DepthDiff(105, 110, [], []))
    snap = Snapshot(last_update_id=100, bids=[], asks=[])
    with pytest.raises(GapDetectedError):
        sync.apply_snapshot(snap)


def test_apply_snapshot_raises_on_gap_within_buffer() -> None:
    sync, _ = make_sync()
    sync.buffer(DepthDiff(99, 101, [], []))   # bridges
    sync.buffer(DepthDiff(105, 110, [], []))  # gap from 101 to 105
    snap = Snapshot(last_update_id=100, bids=[], asks=[])
    with pytest.raises(GapDetectedError):
        sync.apply_snapshot(snap)


def test_live_apply_in_sequence() -> None:
    sync, book = make_sync()
    sync.apply_snapshot(
        Snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(101.0, 1.0)])
    )
    sync.apply(DepthDiff(101, 102, [(99.0, 3.0)], []))
    assert book.depth(Side.BUY, 1) == [(99.0, 3.0)]
    assert sync.last_update_id == 102


def test_live_apply_raises_on_gap() -> None:
    sync, _ = make_sync()
    sync.apply_snapshot(Snapshot(last_update_id=100, bids=[], asks=[]))
    with pytest.raises(GapDetectedError):
        sync.apply(DepthDiff(105, 110, [], []))


def test_qty_zero_diff_deletes_level() -> None:
    sync, book = make_sync()
    sync.apply_snapshot(
        Snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[])
    )
    sync.apply(DepthDiff(101, 101, [(99.0, 0.0)], []))
    assert book.best_bid() is None


def test_buffer_after_synced_raises() -> None:
    sync, _ = make_sync()
    sync.apply_snapshot(Snapshot(last_update_id=100, bids=[], asks=[]))
    with pytest.raises(RuntimeError):
        sync.buffer(DepthDiff(101, 102, [], []))


def test_apply_before_snapshot_raises() -> None:
    sync, _ = make_sync()
    with pytest.raises(RuntimeError):
        sync.apply(DepthDiff(1, 2, [], []))


# ---------- parsers ----------


def test_parse_depth_diff() -> None:
    payload = {
        "U": 10, "u": 12,
        "b": [["99.5", "1.0"], ["99.0", "0"]],
        "a": [["100.5", "2.5"]],
    }
    diff = parse_depth_diff(payload)
    assert diff.first_update_id == 10
    assert diff.last_update_id == 12
    assert diff.bid_updates == [(99.5, 1.0), (99.0, 0.0)]
    assert diff.ask_updates == [(100.5, 2.5)]


def test_parse_trade_buy_aggressor() -> None:
    # m=False means buyer was aggressor.
    payload = {"p": "100.0", "q": "0.5", "m": False, "T": 1700000000000}
    price, qty, side, ts = parse_trade(payload)
    assert (price, qty, side, ts) == (100.0, 0.5, "buy", 1700000000000)


def test_parse_trade_sell_aggressor() -> None:
    payload = {"p": "100.0", "q": "0.5", "m": True, "T": 1700000000000}
    _, _, side, _ = parse_trade(payload)
    assert side == "sell"


def test_parse_snapshot() -> None:
    payload = {
        "lastUpdateId": 42,
        "bids": [["99.0", "1.0"]],
        "asks": [["101.0", "2.0"]],
    }
    snap = parse_snapshot(payload)
    assert snap.last_update_id == 42
    assert snap.bids == [(99.0, 1.0)]
    assert snap.asks == [(101.0, 2.0)]


# ---------- redis writer ----------


def test_publish_book_writes_expected_keys() -> None:
    book = OrderBook()
    book.insert("a1", Side.SELL, 101.0, 2.0, 1)
    book.insert("b1", Side.BUY, 99.0, 1.0, 2)

    redis = MagicMock()
    pipe = MagicMock()
    redis.pipeline.return_value = pipe

    writer = RedisWriter(redis, book, bid_side=Side.BUY, ask_side=Side.SELL)
    writer.publish_book(now=1700000000.0)

    set_calls = {call.args[0]: call.args[1] for call in pipe.set.call_args_list}
    assert set_calls["lob:best_bid"] == 99.0
    assert set_calls["lob:best_ask"] == 101.0
    assert set_calls["lob:mid"] == 100.0
    # spread = (101-99)/100 * 10000 = 200 bps
    assert set_calls["lob:spread_bps"] == pytest.approx(200.0)
    assert json.loads(set_calls["lob:levels:bids"]) == [[99.0, 1.0]]
    assert json.loads(set_calls["lob:levels:asks"]) == [[101.0, 2.0]]
    assert set_calls["lob:last_update"] == 1700000000.0
    pipe.execute.assert_called_once()


def test_publish_book_skips_best_when_empty() -> None:
    book = OrderBook()
    redis = MagicMock()
    pipe = MagicMock()
    redis.pipeline.return_value = pipe

    writer = RedisWriter(redis, book, bid_side=Side.BUY, ask_side=Side.SELL)
    writer.publish_book(now=1.0)

    keys = [call.args[0] for call in pipe.set.call_args_list]
    assert "lob:best_bid" not in keys
    assert "lob:best_ask" not in keys
    assert "lob:mid" not in keys
    assert "lob:levels:bids" in keys


def test_record_trade_appends_and_publishes() -> None:
    redis = MagicMock()
    book = OrderBook()
    writer = RedisWriter(redis, book, bid_side=Side.BUY, ask_side=Side.SELL)

    writer.record_trade(100.0, 0.5, "buy", 1700000000000)
    writer.record_trade(100.5, 0.3, "sell", 1700000000001)

    assert len(writer.recent_trades) == 2
    last_call = redis.set.call_args
    assert last_call.args[0] == "trades:recent"
    trades = json.loads(last_call.args[1])
    assert trades[-1]["price"] == 100.5
    assert trades[-1]["side"] == "sell"


def test_record_trade_rejects_invalid_side() -> None:
    redis = MagicMock()
    book = OrderBook()
    writer = RedisWriter(redis, book, bid_side=Side.BUY, ask_side=Side.SELL)
    with pytest.raises(ValueError):
        writer.record_trade(1.0, 1.0, "neither", 0)


def test_trade_ringbuffer_caps_at_100() -> None:
    redis = MagicMock()
    book = OrderBook()
    writer = RedisWriter(redis, book, bid_side=Side.BUY, ask_side=Side.SELL)
    for i in range(150):
        writer.record_trade(100.0 + i, 1.0, "buy", i)
    assert len(writer.recent_trades) == 100
    assert writer.recent_trades[0]["timestamp"] == 50  # oldest 50 dropped
