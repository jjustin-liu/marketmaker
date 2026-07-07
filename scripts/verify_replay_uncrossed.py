"""Acceptance check: replay a parquet through OrderBook, assert 0 crossed.

Usage:
  python scripts/verify_replay_uncrossed.py data/raw/btcusdt_depth_YYYY-MM-DD.parquet

Exit code:
  0  every tick had best_bid < best_ask (or one side empty)
  1  at least one crossed tick — script prints up to 10 samples

A "tick" is the state of the book after applying one diff row. Snapshot
rows are applied first as a block and not counted as ticks — they're
the starting state.

This is the script that should be in CI / pre-resume-ship checklist for
any recording you intend to point a reviewer at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.backtest.engine import load_diffs_from_parquet  # noqa: E402
from src.lob.order_book import OrderBook  # noqa: E402


def _count_snapshot_rows(path: Path) -> int:
    df = pd.read_parquet(path)
    if "is_snapshot" not in df.columns:
        return 0
    return int(df["is_snapshot"].astype(bool).sum())


def verify(path: Path) -> int:
    diffs = load_diffs_from_parquet(path)
    n_snapshot = _count_snapshot_rows(path)
    print(f"file:           {path.name}")
    print(f"total rows:     {len(diffs):,}")
    print(f"snapshot rows:  {n_snapshot:,}  (applied as seed, not counted as ticks)")

    book = OrderBook()
    crossed = 0
    samples: list[tuple[int, float, float]] = []
    n_ticks = 0

    for i, d in enumerate(diffs):
        book.apply_diff(d.side, d.price, d.qty)
        if i < n_snapshot:
            continue
        n_ticks += 1
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None:
            continue
        if bb >= ba:
            crossed += 1
            if len(samples) < 10:
                samples.append((i, bb, ba))

    pct = 100 * crossed / n_ticks if n_ticks else 0.0
    print(f"ticks checked:  {n_ticks:,}")
    print(f"crossed:        {crossed:,} ({pct:.4f}%)")

    if crossed:
        print("\nfirst crossed samples (row_idx, best_bid, best_ask):")
        for s in samples:
            print(f"  {s[0]:>10}  bid={s[1]:.2f}  ask={s[2]:.2f}")
        return 1
    print("\nPASS: 0 crossed ticks")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    if not args.path.exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        sys.exit(2)
    sys.exit(verify(args.path))


if __name__ == "__main__":
    main()
