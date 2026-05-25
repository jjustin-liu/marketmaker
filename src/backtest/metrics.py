"""Backtest performance metrics.

Sharpe, hit ratio, adverse selection, max drawdown. Pure functions
operating on sequences of fills and PnL samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

ANNUALIZATION_FACTOR = math.sqrt(252)
# Crypto trades 24/7. Minutes in a year = 60 * 24 * 365.
MINUTES_PER_YEAR = 60 * 24 * 365


@dataclass
class Fill:
    """A simulated fill in the backtest."""

    timestamp: int
    side: str         # 'buy' (our bid filled) or 'sell' (our ask filled)
    price: float
    size: float
    mid_at_fill: float


def calculate_sharpe(pnl_series: List[float]) -> float:
    """Annualized Sharpe of a PnL time series.

    Returns 0.0 for fewer than two samples or zero variance.
    """
    if len(pnl_series) < 2:
        return 0.0
    rets = [pnl_series[i] - pnl_series[i - 1] for i in range(1, len(pnl_series))]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    if var <= 0.0:
        return 0.0
    return (mean / math.sqrt(var)) * ANNUALIZATION_FACTOR


def calculate_sharpe_resampled(
    pnl_with_ts: List[Tuple[int, float]],
    bucket_seconds: int = 60,
) -> float:
    """Annualized Sharpe from a timestamped PnL stream.

    Buckets the raw per-event samples into fixed-width time buckets
    (default 1 minute), takes the last PnL in each bucket, computes
    per-bucket returns, and annualizes with sqrt(buckets_per_year).
    This avoids the silent bug of treating per-event samples as
    daily returns.

    pnl_with_ts: list of (timestamp_ms, pnl) ordered by timestamp.
    Returns 0.0 for fewer than two buckets or zero variance.
    """
    if len(pnl_with_ts) < 2:
        return 0.0
    bucket_ms = bucket_seconds * 1000
    bucketed: List[float] = []
    cur_bucket = pnl_with_ts[0][0] // bucket_ms
    last_pnl = pnl_with_ts[0][1]
    for ts, pnl in pnl_with_ts[1:]:
        b = ts // bucket_ms
        if b != cur_bucket:
            bucketed.append(last_pnl)
            cur_bucket = b
        last_pnl = pnl
    bucketed.append(last_pnl)

    if len(bucketed) < 2:
        return 0.0
    rets = [bucketed[i] - bucketed[i - 1] for i in range(1, len(bucketed))]
    mean = sum(rets) / len(rets)
    if len(rets) < 2:
        return 0.0
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    if var <= 0.0:
        return 0.0
    buckets_per_year = MINUTES_PER_YEAR * 60 / bucket_seconds
    return (mean / math.sqrt(var)) * math.sqrt(buckets_per_year)


def calculate_hit_ratio(num_fills: int, num_quotes: int) -> float:
    """Fills per quote refresh. Returns 0.0 if no quotes were posted."""
    if num_quotes <= 0:
        return 0.0
    return num_fills / num_quotes


def calculate_adverse_selection(
    fills: List[Fill],
    mid_after: List[Optional[float]],
) -> float:
    """Average per-fill markout: signed PnL `lookahead` mid moves after fill.

    For a bid fill (we bought): adverse if mid falls afterwards.
    For an ask fill (we sold):  adverse if mid rises afterwards.
    Returns 0.0 if no fills.
    """
    if not fills:
        return 0.0
    if len(fills) != len(mid_after):
        raise ValueError("fills and mid_after must be same length")
    contribs: List[float] = []
    for fill, mid_late in zip(fills, mid_after):
        if mid_late is None:
            continue
        if fill.side == "buy":
            contribs.append(mid_late - fill.mid_at_fill)
        else:
            contribs.append(fill.mid_at_fill - mid_late)
    if not contribs:
        return 0.0
    return sum(contribs) / len(contribs)


def calculate_max_drawdown(pnl_series: List[float]) -> float:
    """Max peak-to-trough drawdown. Returns 0.0 for empty/flat series."""
    if not pnl_series:
        return 0.0
    peak = pnl_series[0]
    max_dd = 0.0
    for x in pnl_series:
        if x > peak:
            peak = x
        dd = peak - x
        if dd > max_dd:
            max_dd = dd
    return max_dd
