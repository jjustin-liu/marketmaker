"""Backtest engine — replay historical L2 diffs through the strategy.

Loads parquet files containing one diff per row
(timestamp, side, price, qty), applies them to a Python OrderBook,
periodically asks the strategy for fresh quotes, and simulates fills
when the opposite-side best price strictly crosses the resting quote
(worst-case queue: only fill on full level sweep).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Tuple

import pandas as pd

from src.backtest.metrics import (
    Fill,
    calculate_adverse_selection,
    calculate_hit_ratio,
    calculate_max_drawdown,
    calculate_sharpe,
)
from src.lob.order_book import OrderBook, Side
from src.strategy.naive_maker import Quote


class QuoteStrategy(Protocol):
    """Anything that returns a (bid_quote, ask_quote) given context."""

    def quote_prices(self, *args, **kwargs) -> Tuple[Quote, Quote]:
        ...


@dataclass
class Diff:
    """One row of the historical L2 stream."""

    timestamp: int
    side: Side
    price: float
    qty: float


@dataclass
class BacktestResult:
    """Final metrics from a single strategy run."""

    pnl: float
    sharpe: float
    hit_ratio: float
    adverse_selection: float
    max_drawdown: float
    num_fills: int
    num_quotes: int


@dataclass
class BacktestEngine:
    """Replays diffs and tracks PnL for one strategy."""

    strategy: QuoteStrategy
    refresh_every: int = 10
    markout_lookahead: int = 50  # diffs after fill for adverse selection
    use_inventory: bool = True   # pass inventory to strategy if it accepts

    book: OrderBook = field(default_factory=OrderBook)
    inventory: float = 0.0
    cash: float = 0.0
    fills: List[Fill] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)
    quote_count: int = 0
    open_bid: Optional[Quote] = None
    open_ask: Optional[Quote] = None

    def run(self, diffs: Iterable[Diff]) -> BacktestResult:
        mids_at_fill_index: List[int] = []
        all_mids: List[float] = []

        for i, diff in enumerate(diffs):
            self.book.apply_diff(diff.side, diff.price, diff.qty)

            best_bid = self.book.best_bid()
            best_ask = self.book.best_ask()
            if best_bid is None or best_ask is None:
                continue
            mid = (best_bid + best_ask) / 2.0
            all_mids.append(mid)

            self._maybe_fill(diff.timestamp, best_bid, best_ask, mid,
                             mids_at_fill_index, i)

            if i % self.refresh_every == 0:
                self._refresh_quotes(mid, best_bid, best_ask)

            self.pnl_history.append(self.cash + self.inventory * mid)

        mid_after = [
            all_mids[idx + self.markout_lookahead]
            if idx + self.markout_lookahead < len(all_mids) else None
            for idx in mids_at_fill_index
        ]

        return BacktestResult(
            pnl=self.pnl_history[-1] if self.pnl_history else 0.0,
            sharpe=calculate_sharpe(self.pnl_history),
            hit_ratio=calculate_hit_ratio(len(self.fills), self.quote_count),
            adverse_selection=calculate_adverse_selection(self.fills, mid_after),
            max_drawdown=calculate_max_drawdown(self.pnl_history),
            num_fills=len(self.fills),
            num_quotes=self.quote_count,
        )

    def _maybe_fill(
        self,
        ts: int,
        best_bid: float,
        best_ask: float,
        mid: float,
        fill_indices: List[int],
        i: int,
    ) -> None:
        """Strict cross fill model — worst-case queue position."""
        if self.open_bid is not None and best_ask < self.open_bid.price:
            f = Fill(timestamp=ts, side="buy",
                     price=self.open_bid.price, size=self.open_bid.size,
                     mid_at_fill=mid)
            self.fills.append(f)
            fill_indices.append(i)
            self.inventory += self.open_bid.size
            self.cash -= self.open_bid.price * self.open_bid.size
            self.open_bid = None
        if self.open_ask is not None and best_bid > self.open_ask.price:
            f = Fill(timestamp=ts, side="sell",
                     price=self.open_ask.price, size=self.open_ask.size,
                     mid_at_fill=mid)
            self.fills.append(f)
            fill_indices.append(i)
            self.inventory -= self.open_ask.size
            self.cash += self.open_ask.price * self.open_ask.size
            self.open_ask = None

    def _refresh_quotes(
        self,
        mid: float,
        best_bid: float,
        best_ask: float,
    ) -> None:
        try:
            if self.use_inventory:
                bid_q, ask_q = self.strategy.quote_prices(
                    mid_price=mid,
                    inventory=self.inventory,
                    bid_probability=0.5,
                    ask_probability=0.5,
                )
            else:
                bid_q, ask_q = self.strategy.quote_prices(
                    mid_price=mid,
                    best_bid=best_bid,
                    best_ask=best_ask,
                )
        except TypeError:
            bid_q, ask_q = self.strategy.quote_prices(mid_price=mid)

        self.open_bid = bid_q
        self.open_ask = ask_q
        self.quote_count += 1


def load_diffs_from_parquet(path: Path) -> List[Diff]:
    """Read a parquet file with columns: timestamp, side, price, qty.

    side column is either int (0/1) or str ('buy'/'sell'/'bid'/'ask').
    """
    df = pd.read_parquet(path)
    required = {"timestamp", "side", "price", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parquet missing columns: {missing}")

    diffs: List[Diff] = []
    for _, row in df.iterrows():
        raw_side = row["side"]
        if isinstance(raw_side, str):
            s = raw_side.lower()
            side = Side.BUY if s in ("buy", "bid", "b") else Side.SELL
        else:
            side = Side(int(raw_side))
        diffs.append(Diff(
            timestamp=int(row["timestamp"]),
            side=side,
            price=float(row["price"]),
            qty=float(row["qty"]),
        ))
    return diffs
