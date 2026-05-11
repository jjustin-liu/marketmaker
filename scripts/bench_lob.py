"""Throughput benchmark for the C++ MatchEngine and Python OrderBook.

Synthetic insert/cancel workload. Reports ops/sec for each engine.
Designed for resume-defensible throughput numbers — not a perf
regression test; rerun manually when you want a fresh measurement.

Usage:
  python -m scripts.bench_lob [--n 200000] [--mid 60000]
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Tuple

from src.lob import CppSide, MatchEngine, OrderBook, Side


def _bench_engine(
    insert_fn,
    cancel_fn,
    n: int,
    mid: float,
    seed: int,
) -> Tuple[float, float]:
    """Return (insert_ops_per_sec, cancel_ops_per_sec)."""
    rng = random.Random(seed)
    order_ids = [f"o{i}" for i in range(n)]
    sides = [rng.choice([0, 1]) for _ in range(n)]
    # Tight cluster around mid: 5 bps either side
    prices = [
        mid * (1.0 - 0.0005) if sides[i] == 0
        else mid * (1.0 + 0.0005)
        for i in range(n)
    ]
    sizes = [0.001 for _ in range(n)]
    ts = list(range(n))

    t0 = time.perf_counter()
    for i in range(n):
        insert_fn(order_ids[i], sides[i], prices[i], sizes[i], ts[i])
    insert_dt = time.perf_counter() - t0

    rng.shuffle(order_ids)
    t0 = time.perf_counter()
    for oid in order_ids:
        cancel_fn(oid)
    cancel_dt = time.perf_counter() - t0

    return n / insert_dt, n / cancel_dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--mid", type=float, default=60_000.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"workload: {args.n:,} inserts + {args.n:,} cancels, mid=${args.mid:,.2f}")
    print()

    # C++ MatchEngine
    cpp = MatchEngine()
    cpp_ins, cpp_cnl = _bench_engine(
        insert_fn=lambda oid, s, p, sz, t:
            cpp.insert(oid, CppSide.BUY if s == 0 else CppSide.SELL, p, sz, t),
        cancel_fn=cpp.cancel,
        n=args.n, mid=args.mid, seed=args.seed,
    )
    print(f"C++ MatchEngine:")
    print(f"  insert:  {cpp_ins:>12,.0f} ops/sec  ({1e6 / cpp_ins:.2f} µs/op)")
    print(f"  cancel:  {cpp_cnl:>12,.0f} ops/sec  ({1e6 / cpp_cnl:.2f} µs/op)")
    print()

    # Python OrderBook
    py = OrderBook()
    py_ins, py_cnl = _bench_engine(
        insert_fn=lambda oid, s, p, sz, t:
            py.insert(oid, Side.BUY if s == 0 else Side.SELL, p, sz, t),
        cancel_fn=py.cancel,
        n=args.n, mid=args.mid, seed=args.seed,
    )
    print(f"Python OrderBook:")
    print(f"  insert:  {py_ins:>12,.0f} ops/sec  ({1e6 / py_ins:.2f} µs/op)")
    print(f"  cancel:  {py_cnl:>12,.0f} ops/sec  ({1e6 / py_cnl:.2f} µs/op)")
    print()

    print(f"C++ speedup: insert {cpp_ins / py_ins:.1f}x  "
          f"cancel {cpp_cnl / py_cnl:.1f}x")


if __name__ == "__main__":
    main()
