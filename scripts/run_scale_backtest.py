"""Shard-parallel backtest over a large partitioned L2 corpus.

Demonstrates data engineering at scale: each parquet shard is an
independent replay (bounded memory — one shard per worker), fanned out
across CPU cores, then aggregated. Reports the receipt that backs a
"backtested N events / X GB of L2" claim: total shards, rows, on-disk
bytes, wall-clock, and sustained events/sec.

Each shard is replayed in full (its own epochs), so per-shard results
are independent and safe to parallelize; aggregate PnL/fills/quotes are
summed across shards.

Usage:
  python -m scripts.run_scale_backtest --dir data/l2_shards --workers 8
  python -m scripts.run_scale_backtest --glob "data/raw/*_depth_*.parquet"
"""

from __future__ import annotations

import argparse
import glob as globmod
import logging
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

from src.backtest.engine import BacktestEngine, load_diffs_from_parquet
from src.strategy.naive_maker import NaiveMaker

logger = logging.getLogger("run_scale_backtest")

GB = 1024 ** 3


@dataclass
class ShardResult:
    name: str
    rows: int
    bytes_on_disk: int
    pnl: float
    fills: int
    quotes: int
    seconds: float


def _run_shard(args: tuple) -> Optional[ShardResult]:
    path_str, fee_bps = args
    path = Path(path_str)
    t0 = time.perf_counter()
    diffs = load_diffs_from_parquet(path)
    eng = BacktestEngine(strategy=NaiveMaker(), use_inventory=False,
                         fee_bps=fee_bps, fill_mode="strict_cross")
    res = eng.run(diffs)
    return ShardResult(
        name=path.name, rows=len(diffs), bytes_on_disk=path.stat().st_size,
        pnl=res.pnl, fills=res.num_fills, quotes=res.num_quotes,
        seconds=time.perf_counter() - t0,
    )


def _discover(dir_: Optional[Path], patterns: Optional[List[str]]) -> List[Path]:
    paths: List[Path] = []
    if dir_ is not None:
        paths.extend(sorted(dir_.glob("*.parquet")))
    for pat in patterns or []:
        paths.extend(Path(p) for p in sorted(globmod.glob(pat)))
    # de-dup, preserve order
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard-parallel scale backtest.")
    parser.add_argument("--dir", type=Path, default=None,
                        help="directory of *.parquet shards")
    parser.add_argument("--glob", dest="globs", action="append", default=None,
                        help="glob for shards (repeatable)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    parser.add_argument("--fee-bps", type=float, default=0.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    shards = _discover(args.dir, args.globs)
    if not shards:
        raise SystemExit("no shards found (use --dir or --glob)")
    logger.info("discovered %d shards, %d workers", len(shards), args.workers)

    wall0 = time.perf_counter()
    tasks = [(str(p), args.fee_bps) for p in shards]
    with Pool(processes=args.workers) as pool:
        results = [r for r in pool.map(_run_shard, tasks) if r is not None]
    wall = time.perf_counter() - wall0

    rows = sum(r.rows for r in results)
    disk = sum(r.bytes_on_disk for r in results)
    fills = sum(r.fills for r in results)
    quotes = sum(r.quotes for r in results)
    pnl = sum(r.pnl for r in results)
    cpu_seconds = sum(r.seconds for r in results)

    logger.info("\n=== scale backtest receipt ===")
    logger.info("shards processed:  %d", len(results))
    logger.info("L2 events:         %s", f"{rows:,}")
    logger.info("on-disk volume:    %.2f GB", disk / GB)
    logger.info("wall-clock:        %.1f s   (CPU %.1f s, %.1fx parallel)",
                wall, cpu_seconds, cpu_seconds / wall if wall else 0.0)
    logger.info("throughput:        %s events/sec (wall)", f"{rows / wall:,.0f}")
    logger.info("aggregate fills:   %s   quotes: %s",
                f"{fills:,}", f"{quotes:,}")
    logger.info("aggregate PnL:     %+.2f", pnl)


if __name__ == "__main__":
    main()
