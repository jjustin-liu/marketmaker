"""Tests for the order book implementations.

Each test runs against both the pure-Python OrderBook and the C++
MatchEngine. Both must pass — they implement the same contract.
"""

import pytest

from src.lob import MatchEngine, OrderBook
from src.lob import CppSide, Side


PY = ("python", OrderBook, Side)
CPP = ("cpp", MatchEngine, CppSide)


@pytest.fixture(params=[PY, CPP], ids=["python", "cpp"])
def book_and_side(request):
    """Yield a fresh book instance and the matching Side enum."""
    _, factory, side_enum = request.param
    return factory(), side_enum


def test_insert_buy_matches_resting_ask(book_and_side) -> None:
    book, S = book_and_side
    book.insert("a1", S.SELL, 100.0, 1.0, 1)

    fills = book.insert("b1", S.BUY, 100.0, 1.0, 2)

    assert len(fills) == 1
    f = fills[0]
    assert f.taker_order_id == "b1"
    assert f.maker_order_id == "a1"
    assert f.price == 100.0
    assert f.size == 1.0
    assert book.best_ask() is None
    assert book.best_bid() is None


def test_insert_sell_matches_resting_bid(book_and_side) -> None:
    book, S = book_and_side
    book.insert("b1", S.BUY, 100.0, 2.0, 1)

    fills = book.insert("a1", S.SELL, 100.0, 2.0, 2)

    assert len(fills) == 1
    assert fills[0].price == 100.0
    assert fills[0].size == 2.0
    assert book.best_bid() is None


def test_non_crossing_order_rests(book_and_side) -> None:
    book, S = book_and_side
    fills = book.insert("b1", S.BUY, 99.0, 1.0, 1)
    assert list(fills) == []
    assert book.best_bid() == 99.0

    fills = book.insert("a1", S.SELL, 101.0, 1.0, 2)
    assert list(fills) == []
    assert book.best_ask() == 101.0


def test_cancel_existing_order(book_and_side) -> None:
    book, S = book_and_side
    book.insert("b1", S.BUY, 99.0, 1.0, 1)

    assert book.cancel("b1") is True
    assert book.best_bid() is None


def test_cancel_nonexistent_order(book_and_side) -> None:
    book, _ = book_and_side
    assert book.cancel("ghost") is False


def test_price_priority_best_price_first(book_and_side) -> None:
    book, S = book_and_side
    book.insert("a_high", S.SELL, 101.0, 1.0, 1)
    book.insert("a_low", S.SELL, 100.0, 1.0, 2)

    fills = book.insert("b1", S.BUY, 101.0, 1.0, 3)

    assert len(fills) == 1
    assert fills[0].maker_order_id == "a_low"
    assert fills[0].price == 100.0
    assert book.best_ask() == 101.0


def test_time_priority_fifo_at_same_price(book_and_side) -> None:
    book, S = book_and_side
    book.insert("a_first", S.SELL, 100.0, 1.0, 1)
    book.insert("a_second", S.SELL, 100.0, 1.0, 2)

    fills = book.insert("b1", S.BUY, 100.0, 1.0, 3)

    assert len(fills) == 1
    assert fills[0].maker_order_id == "a_first"


def test_partial_fill_leaves_remainder_resting(book_and_side) -> None:
    book, S = book_and_side
    book.insert("a1", S.SELL, 100.0, 1.0, 1)

    fills = book.insert("b1", S.BUY, 100.0, 3.0, 2)

    assert len(fills) == 1
    assert fills[0].size == 1.0
    assert book.best_bid() == 100.0
    assert list(book.depth(S.BUY, 1)) == [(100.0, 2.0)]


def test_partial_maker_fill_keeps_rest(book_and_side) -> None:
    book, S = book_and_side
    book.insert("a1", S.SELL, 100.0, 5.0, 1)

    fills = book.insert("b1", S.BUY, 100.0, 2.0, 2)

    assert len(fills) == 1
    assert fills[0].size == 2.0
    assert list(book.depth(S.SELL, 1)) == [(100.0, 3.0)]


def test_walk_multiple_levels(book_and_side) -> None:
    book, S = book_and_side
    book.insert("a1", S.SELL, 100.0, 1.0, 1)
    book.insert("a2", S.SELL, 101.0, 1.0, 2)

    fills = book.insert("b1", S.BUY, 101.0, 2.0, 3)

    assert len(fills) == 2
    assert fills[0].price == 100.0
    assert fills[1].price == 101.0
    assert book.best_ask() is None


def test_apply_diff_zero_qty_deletes_level(book_and_side) -> None:
    book, S = book_and_side
    book.apply_diff(S.BUY, 99.0, 2.0)
    assert book.best_bid() == 99.0

    book.apply_diff(S.BUY, 99.0, 0)
    assert book.best_bid() is None


def test_apply_diff_replaces_level(book_and_side) -> None:
    book, S = book_and_side
    book.apply_diff(S.SELL, 100.0, 1.0)
    book.apply_diff(S.SELL, 100.0, 5.0)
    assert list(book.depth(S.SELL, 1)) == [(100.0, 5.0)]


def test_depth_returns_sorted_levels(book_and_side) -> None:
    book, S = book_and_side
    book.insert("a1", S.SELL, 101.0, 1.0, 1)
    book.insert("a2", S.SELL, 100.0, 2.0, 2)
    book.insert("a3", S.SELL, 102.0, 3.0, 3)
    assert list(book.depth(S.SELL, 3)) == [
        (100.0, 2.0),
        (101.0, 1.0),
        (102.0, 3.0),
    ]

    book.insert("b1", S.BUY, 98.0, 1.0, 4)
    book.insert("b2", S.BUY, 99.0, 2.0, 5)
    assert list(book.depth(S.BUY, 2)) == [(99.0, 2.0), (98.0, 1.0)]


def test_insert_zero_size_raises(book_and_side) -> None:
    book, S = book_and_side
    with pytest.raises((ValueError, RuntimeError)):
        book.insert("b1", S.BUY, 100.0, 0.0, 1)


def test_insert_duplicate_id_raises(book_and_side) -> None:
    book, S = book_and_side
    book.insert("b1", S.BUY, 99.0, 1.0, 1)
    with pytest.raises((ValueError, RuntimeError)):
        book.insert("b1", S.BUY, 99.0, 1.0, 2)


def test_cancel_one_of_many_at_same_price(book_and_side) -> None:
    book, S = book_and_side
    book.insert("a1", S.SELL, 100.0, 1.0, 1)
    book.insert("a2", S.SELL, 100.0, 2.0, 2)
    book.insert("a3", S.SELL, 100.0, 3.0, 3)

    assert book.cancel("a2") is True
    assert list(book.depth(S.SELL, 1)) == [(100.0, 4.0)]

    fills = book.insert("b1", S.BUY, 100.0, 1.0, 4)
    assert fills[0].maker_order_id == "a1"
