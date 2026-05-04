"""Limit order book package.

Two implementations with the same interface:
- OrderBook: pure-Python, used in backtests and tests.
- MatchEngine: C++ via pybind11, used in live trading.
"""

from src.lob._match_engine import Fill as CppFill
from src.lob._match_engine import MatchEngine
from src.lob._match_engine import Side as CppSide
from src.lob.order_book import Fill, Order, OrderBook, Side

__all__ = [
    "Fill",
    "Order",
    "OrderBook",
    "Side",
    "MatchEngine",
    "CppFill",
    "CppSide",
]
