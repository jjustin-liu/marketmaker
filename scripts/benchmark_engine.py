"""Benchmark the order-book engines: throughput and per-event latency.

Streams a real recorded L2 event sequence through each engine and
measures sustained throughput (events/sec) and per-event latency
(p50/p99). The C++ MatchEngine is the low-latency hot path; the
pure-Python OrderBook is reported for comparison. Redis is NOT in this
path by design — it is an async state bus off the hot loop, so engine
latency is measured in isolation.

Two workloads:
  apply_diff        — the raw book-mutation op (one L2 event)
  tick (diff+top)   — apply_diff + best_bid + best_ask, the work the
                      quoting loop actually does per event

Usage:
  python -m scripts.benchmark_engine \\
      --input data/raw/btcusdt_depth_2026-07-08.parquet --events 1000000
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from src.backtest.engine import load_diffs_from_parquet
from src.lob import CppSide, MatchEngine, OrderBook, Side

logger = logging.getLogger("benchmark_engine")

WARMUP = 50_000


def _timer_overhead_ns() -> float:
    """Median cost of a back-to-back perf_counter_ns pair (subtracted out)."""
    samples = np.empty(10_000, dtype=np.int64)
    for i in range(10_000):
        t0 = time.perf_counter_ns()
        t1 = time.perf_counter_ns()
        samples[i] = t1 - t0
    return float(np.median(samples))


def _load_events(
    path: Path, n: int,
) -> List[Tuple[str, float, float]]:
    """Return up to n (side_str, price, qty) events from a depth parquet."""
    diffs = load_diffs_from_parquet(path)
    out: List[Tuple[str, float, float]] = []
    for d in diffs:
        if d.is_snapshot:
            continue
        side = "buy" if d.side == Side.BUY else "sell"
        out.append((side, d.price, d.qty))
        if len(out) >= n:
            break
    return out


def _bench(
    name: str,
    op: Callable[[int], None],
    n: int,
    overhead_ns: float,
) -> None:
    """Time n calls of op(i): report throughput and latency percentiles."""
    # Warmup (JIT-free, but warms caches / branch predictors / allocator).
    for i in range(min(WARMUP, n)):
        op(i)

    # Throughput: bulk timing over the whole run.
    t0 = time.perf_counter_ns()
    for i in range(n):
        op(i)
    elapsed_ns = time.perf_counter_ns() - t0
    thru = n / (elapsed_ns / 1e9)
    mean_us = (elapsed_ns / n) / 1e3

    # Latency distribution: per-call timing on a sample (timer overhead out).
    sample = min(n, 200_000)
    lat = np.empty(sample, dtype=np.float64)
    for i in range(sample):
        s = time.perf_counter_ns()
        op(i)
        lat[i] = max(0.0, (time.perf_counter_ns() - s) - overhead_ns)
    p50 = np.percentile(lat, 50) / 1e3
    p99 = np.percentile(lat, 99) / 1e3

    logger.info(
        "%-26s %12s ev/s  mean %6.3f us  p50 %6.3f us  p99 %6.3f us",
        name, f"{thru:,.0f}", mean_us, p50, p99,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Order-book engine benchmark.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--events", type=int, default=1_000_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("loading up to %d events from %s ...", args.events, args.input.name)
    events = _load_events(args.input, args.events)
    n = len(events)
    logger.info("loaded %d events", n)

    overhead = _timer_overhead_ns()
    logger.info("timer overhead ~%.0f ns (subtracted from latencies)\n", overhead)

    # Pre-map sides to each engine's enum so the hot loop does no lookups.
    cpp_sides = [CppSide.BUY if s == "buy" else CppSide.SELL for s, _, _ in events]
    py_sides = [Side.BUY if s == "buy" else Side.SELL for s, _, _ in events]
    prices = [p for _, p, _ in events]
    qtys = [q for _, _, q in events]

    logger.info("=== C++ MatchEngine (hot path) ===")
    cpp = MatchEngine()

    def cpp_diff(i: int) -> None:
        cpp.apply_diff(cpp_sides[i], prices[i], qtys[i])

    def cpp_tick(i: int) -> None:
        cpp.apply_diff(cpp_sides[i], prices[i], qtys[i])
        cpp.best_bid()
        cpp.best_ask()

    _bench("C++ apply_diff", cpp_diff, n, overhead)
    cpp2 = MatchEngine()

    def cpp_tick2(i: int) -> None:
        cpp2.apply_diff(cpp_sides[i], prices[i], qtys[i])
        cpp2.best_bid()
        cpp2.best_ask()

    _bench("C++ tick (diff+top)", cpp_tick2, n, overhead)

    logger.info("\n=== Python OrderBook (comparison) ===")
    py = OrderBook()

    def py_diff(i: int) -> None:
        py.apply_diff(py_sides[i], prices[i], qtys[i])

    _bench("Python apply_diff", py_diff, n, overhead)


if __name__ == "__main__":
    main()
