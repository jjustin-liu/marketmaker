"""Backtest engine — replay historical L2 diffs through the strategy.

Loads parquet files containing one diff per row
(timestamp, side, price, qty), applies them to a Python OrderBook,
periodically asks the strategy for fresh quotes, and simulates fills
when the opposite-side best price strictly crosses the resting quote
(worst-case queue: only fill on full level sweep).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Protocol, Tuple

import pandas as pd

from src.backtest.metrics import (
    Fill,
    calculate_adverse_selection,
    calculate_hit_ratio,
    calculate_max_drawdown,
    calculate_sharpe_resampled,
)
from src.features.volatility import VolatilityCalculator
from src.lob.order_book import OrderBook, Side
from src.strategy.naive_maker import Quote

BPS = 1e-4
SIDES_PER_REFRESH = 2
DEPTH_LEVELS_FOR_STRATEGY = 5


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
    is_snapshot: bool = False


@dataclass
class Trade:
    """One executed trade, tagged with aggressor side.

    side is the aggressor: "buy" means a buyer crossed the spread (lifts
    asks), "sell" means a seller hit bids. A resting bid fills on a sell
    print at/through its price; a resting ask on a buy print.
    """

    timestamp: int
    side: str
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
    crossed_skips: int = 0


@dataclass
class BacktestEngine:
    """Replays diffs and tracks PnL for one strategy."""

    strategy: QuoteStrategy
    refresh_every: int = 10
    markout_lookahead: int = 50  # diffs after fill for adverse selection
    use_inventory: bool = True   # pass inventory to strategy if it accepts
    fee_bps: float = 0.0         # signed maker fee in bps. Negative = rebate.
    sharpe_bucket_seconds: int = 60
    # "strict_cross": fill only when the market sweeps through our price
    # (worst-case queue). "queue": fill as the depth ahead of us at our
    # price is consumed (depth-only proxy for trade flow) or on a touch.
    # "trades": fill resting quotes from executed trade prints passed via
    # run(trades=...), merged by timestamp — most realistic.
    fill_mode: str = "strict_cross"
    on_tick: Optional[Callable[["BacktestEngine", bool], None]] = None
    tick_sleep_seconds: float = 0.0

    book: OrderBook = field(default_factory=OrderBook)
    vol_calc: VolatilityCalculator = field(default_factory=VolatilityCalculator)
    _volatility: float = 0.0
    inventory: float = 0.0
    cash: float = 0.0
    fills: List[Fill] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)
    pnl_with_ts: List[Tuple[int, float]] = field(default_factory=list)
    quote_count: int = 0
    crossed_skips: int = 0
    open_bid: Optional[Quote] = None
    open_ask: Optional[Quote] = None
    # Queue-position state (fill_mode="queue"): liquidity still ahead of us
    # at our price, and the last-seen aggregate qty at that level.
    _bid_queue: float = 0.0
    _bid_level_qty: float = 0.0
    _bid_resting: bool = False
    _ask_queue: float = 0.0
    _ask_level_qty: float = 0.0
    _ask_resting: bool = False

    def run(
        self,
        diffs: Iterable[Diff],
        trades: Optional[List[Trade]] = None,
    ) -> BacktestResult:
        mids_at_fill_index: List[int] = []
        all_mids: List[float] = []
        _in_snapshot = False

        # fill_mode="trades": fill resting quotes from trade prints, merged
        # by timestamp. Trades are flushed up to each diff's timestamp.
        trade_list = sorted(trades, key=lambda t: t.timestamp) if trades else []
        trade_idx = 0

        for i, diff in enumerate(diffs):
            # Epoch boundary: first snapshot row after live diffs → reset book
            if diff.is_snapshot and not _in_snapshot:
                self.book = OrderBook()
                self.vol_calc.reset()
                self.open_bid = None
                self.open_ask = None
                _in_snapshot = True
            elif not diff.is_snapshot:
                _in_snapshot = False

            self.book.apply_diff(diff.side, diff.price, diff.qty)

            # Skip mid/fill/quote logic while seeding snapshot rows
            if diff.is_snapshot:
                continue

            best_bid = self.book.best_bid()
            best_ask = self.book.best_ask()
            if best_bid is None or best_ask is None:
                continue
            # Skip crossed/locked books — a non-physical reconstruction state
            # (stale level never deleted). Quoting or filling here is garbage.
            # Both strategies skip identically, so the comparison stays fair.
            if best_bid >= best_ask:
                self.crossed_skips += 1
                continue
            mid = (best_bid + best_ask) / 2.0
            all_mids.append(mid)
            self._volatility = self.vol_calc.update(mid)

            fills_before = len(self.fills)
            if self.fill_mode == "trades":
                while (trade_idx < len(trade_list)
                       and trade_list[trade_idx].timestamp <= diff.timestamp):
                    self._maybe_fill_trade(trade_list[trade_idx], mid,
                                           mids_at_fill_index, i)
                    trade_idx += 1
            else:
                self._maybe_fill(diff.timestamp, best_bid, best_ask, mid,
                                 mids_at_fill_index, i)
            new_fill = len(self.fills) > fills_before

            if i % self.refresh_every == 0:
                self._refresh_quotes(mid, best_bid, best_ask)

            pnl_now = self.cash + self.inventory * mid
            self.pnl_history.append(pnl_now)
            self.pnl_with_ts.append((diff.timestamp, pnl_now))

            if self.on_tick is not None:
                self.on_tick(self, new_fill)
            if self.tick_sleep_seconds > 0.0:
                time.sleep(self.tick_sleep_seconds)

        mid_after = [
            all_mids[idx + self.markout_lookahead]
            if idx + self.markout_lookahead < len(all_mids) else None
            for idx in mids_at_fill_index
        ]

        return BacktestResult(
            pnl=self.pnl_history[-1] if self.pnl_history else 0.0,
            sharpe=calculate_sharpe_resampled(
                self.pnl_with_ts, self.sharpe_bucket_seconds,
            ),
            hit_ratio=calculate_hit_ratio(
                len(self.fills), self.quote_count * SIDES_PER_REFRESH,
            ),
            adverse_selection=calculate_adverse_selection(self.fills, mid_after),
            max_drawdown=calculate_max_drawdown(self.pnl_history),
            num_fills=len(self.fills),
            num_quotes=self.quote_count,
            crossed_skips=self.crossed_skips,
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
        if self.fill_mode == "queue":
            self._maybe_fill_queue(ts, best_bid, best_ask, mid, fill_indices, i)
        else:
            self._maybe_fill_strict(ts, best_bid, best_ask, mid, fill_indices, i)

    def _maybe_fill_strict(
        self, ts: int, best_bid: float, best_ask: float, mid: float,
        fill_indices: List[int], i: int,
    ) -> None:
        """Worst-case queue: fill only when the market sweeps past our price."""
        if self.open_bid is not None and best_ask < self.open_bid.price:
            self._book_fill("buy", ts, mid, self.open_bid.price,
                            self.open_bid.size, fill_indices, i)
            self.open_bid = None
        if self.open_ask is not None and best_bid > self.open_ask.price:
            self._book_fill("sell", ts, mid, self.open_ask.price,
                            self.open_ask.size, fill_indices, i)
            self.open_ask = None

    def _maybe_fill_queue(
        self, ts: int, best_bid: float, best_ask: float, mid: float,
        fill_indices: List[int], i: int,
    ) -> None:
        """Depth-only flow proxy: fill on a touch (opposite best reaches our
        price), or — when we rest on a real level — as the queue ahead of us
        is consumed. Touch fills capture the moment of trade rather than
        waiting for a full adverse sweep below/above our price."""
        if self.open_bid is not None:
            pb = self.open_bid.price
            filled = best_ask <= pb  # a sell reaches our bid
            if not filled and self._bid_resting:
                cur = self.book.qty_at(Side.BUY, pb)
                consumed = self._bid_level_qty - cur
                if consumed > 0:
                    self._bid_queue = max(0.0, self._bid_queue - consumed)
                self._bid_level_qty = cur
                filled = self._bid_queue <= 1e-12
            if filled:
                self._book_fill("buy", ts, mid, pb, self.open_bid.size,
                                fill_indices, i)
                self.open_bid = None
        if self.open_ask is not None:
            pa = self.open_ask.price
            filled = best_bid >= pa  # a buy reaches our ask
            if not filled and self._ask_resting:
                cur = self.book.qty_at(Side.SELL, pa)
                consumed = self._ask_level_qty - cur
                if consumed > 0:
                    self._ask_queue = max(0.0, self._ask_queue - consumed)
                self._ask_level_qty = cur
                filled = self._ask_queue <= 1e-12
            if filled:
                self._book_fill("sell", ts, mid, pa, self.open_ask.size,
                                fill_indices, i)
                self.open_ask = None

    def _maybe_fill_trade(
        self, trade: Trade, mid: float, fill_indices: List[int], i: int,
    ) -> None:
        """Fill a resting quote when a trade prints at/through its price.

        A sell-aggressor print at <= our bid hits our bid (we buy); a
        buy-aggressor print at >= our ask lifts our ask (we sell). These
        fills are decoupled from where the book moves next, so they can be
        followed by favorable reversions — the realism depth-only can't give.
        """
        if (self.open_bid is not None and trade.side == "sell"
                and trade.price <= self.open_bid.price):
            self._book_fill("buy", trade.timestamp, mid, self.open_bid.price,
                            self.open_bid.size, fill_indices, i)
            self.open_bid = None
        if (self.open_ask is not None and trade.side == "buy"
                and trade.price >= self.open_ask.price):
            self._book_fill("sell", trade.timestamp, mid, self.open_ask.price,
                            self.open_ask.size, fill_indices, i)
            self.open_ask = None

    def _book_fill(
        self, side: str, ts: int, mid: float, price: float, size: float,
        fill_indices: List[int], i: int,
    ) -> None:
        """Record a fill and update inventory, cash, and fees."""
        notional = price * size
        self.fills.append(Fill(timestamp=ts, side=side, price=price,
                               size=size, mid_at_fill=mid))
        fill_indices.append(i)
        if side == "buy":
            self.inventory += size
            self.cash -= notional
        else:
            self.inventory -= size
            self.cash += notional
        self.cash -= self.fee_bps * BPS * notional

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
                    bids=self.book.depth(Side.BUY, DEPTH_LEVELS_FOR_STRATEGY),
                    asks=self.book.depth(Side.SELL, DEPTH_LEVELS_FOR_STRATEGY),
                    volatility=self._volatility,
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

        if self.fill_mode == "queue":
            self._init_queue_state(best_bid, best_ask)

    def _init_queue_state(self, best_bid: float, best_ask: float) -> None:
        """Snapshot the queue ahead of each new quote. A quote that lands on
        an existing level rests behind that depth; a quote between levels
        (the usual case) has no queue and fills purely on a touch."""
        pb = self.open_bid.price if self.open_bid else None
        if pb is not None:
            q = self.book.qty_at(Side.BUY, pb)
            self._bid_resting = q > 0.0 and pb <= best_bid
            self._bid_queue = q
            self._bid_level_qty = q
        pa = self.open_ask.price if self.open_ask else None
        if pa is not None:
            q = self.book.qty_at(Side.SELL, pa)
            self._ask_resting = q > 0.0 and pa >= best_ask
            self._ask_queue = q
            self._ask_level_qty = q


def load_diffs_from_parquet(path: Path, skip_rows: int = 0) -> List[Diff]:
    """Read a parquet file with columns: timestamp, side, price, qty.

    side column is either int (0/1) or str ('buy'/'sell'/'bid'/'ask').

    If an is_snapshot column is present, file order is preserved. Real
    daily files hold many snapshot blocks (one per reconnect or
    scheduled re-snapshot) interleaved with diffs; the engine treats
    each False→True is_snapshot transition as an epoch boundary that
    resets the book. Sorting snapshot rows to the front would merge
    different point-in-time books and cross the replay.

    skip_rows: legacy workaround for pre-is_snapshot parquets. Ignored
    when is_snapshot is present.
    """
    df = pd.read_parquet(path)
    required = {"timestamp", "side", "price", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parquet missing columns: {missing}")

    # Legacy fallback for pre-is_snapshot files: skip header rows
    if "is_snapshot" not in df.columns and skip_rows > 0:
        df = df.iloc[skip_rows:].reset_index(drop=True)

    ts_arr = df["timestamp"].to_numpy(dtype="int64")
    price_arr = df["price"].to_numpy(dtype="float64")
    qty_arr = df["qty"].to_numpy(dtype="float64")
    side_raw = df["side"].to_numpy()
    if side_raw.dtype.kind in ("U", "O"):
        side_arr = [
            Side.BUY if str(s).lower() in ("buy", "bid", "b") else Side.SELL
            for s in side_raw
        ]
    else:
        side_arr = [Side(int(s)) for s in side_raw]

    has_snap = "is_snapshot" in df.columns
    snap_arr = df["is_snapshot"].astype(bool).to_numpy() if has_snap else None

    return [
        Diff(
            timestamp=int(ts_arr[i]),
            side=side_arr[i],
            price=float(price_arr[i]),
            qty=float(qty_arr[i]),
            is_snapshot=bool(snap_arr[i]) if snap_arr is not None else False,
        )
        for i in range(len(df))
    ]


def load_trades_from_parquet(path: Path) -> List[Trade]:
    """Read a trades parquet (timestamp, side, price, qty) into Trades.

    side is the aggressor ("buy"/"sell"), as written by the recorder's
    parse_trade_message. Rows are returned in file order (chronological).
    """
    df = pd.read_parquet(path)
    required = {"timestamp", "side", "price", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trades parquet missing columns: {missing}")

    ts_arr = df["timestamp"].to_numpy(dtype="int64")
    price_arr = df["price"].to_numpy(dtype="float64")
    qty_arr = df["qty"].to_numpy(dtype="float64")
    side_arr = df["side"].to_numpy()
    return [
        Trade(
            timestamp=int(ts_arr[i]),
            side=str(side_arr[i]).lower(),
            price=float(price_arr[i]),
            qty=float(qty_arr[i]),
        )
        for i in range(len(df))
    ]
