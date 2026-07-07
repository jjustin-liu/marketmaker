"""Acceptance check: replay a parquet through OrderBook, assert 0 crossed.

Usage:
  python scripts/verify_replay_uncrossed.py data/raw/btcusdt_depth_YYYY-MM-DD.parquet

Exit code:
  0  book uncrossed at every event boundary (or one side empty)
  1  at least one crossed event — script prints up to 10 samples

Rows sharing a timestamp form one exchange event. The book is only
well-defined at event boundaries: while an event's rows apply one at a
time, one side lands before the other and the intermediate state
routinely "crosses". Those transients are expected and not counted.
Snapshot blocks (is_snapshot=True) reset the book — one per bootstrap,
reconnect, or scheduled re-snapshot.

This is the script that should be in CI / pre-resume-ship checklist for
any recording you intend to point a reviewer at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.backtest.engine import load_diffs_from_parquet  # noqa: E402
from src.lob.order_book import OrderBook  # noqa: E402

MAX_SAMPLES = 10


def verify(path: Path) -> int:
    diffs = load_diffs_from_parquet(path)
    n_snapshot = sum(1 for d in diffs if d.is_snapshot)
    print(f"file:           {path.name}")
    print(f"total rows:     {len(diffs):,}")
    print(f"snapshot rows:  {n_snapshot:,}  (seed blocks, reset the book)")

    book = OrderBook()
    crossed = 0
    events = 0
    samples: List[Tuple[int, float, float]] = []
    prev_snap = False
    prev_ts: Optional[int] = None
    row_idx_of_event_end = 0

    def _check(idx: int) -> None:
        nonlocal crossed, events
        events += 1
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is not None and ba is not None and bb >= ba:
            crossed += 1
            if len(samples) < MAX_SAMPLES:
                samples.append((idx, bb, ba))

    for i, d in enumerate(diffs):
        if d.is_snapshot and not prev_snap:
            if prev_ts is not None:
                _check(row_idx_of_event_end)
                prev_ts = None
            book = OrderBook()
        prev_snap = d.is_snapshot

        if not d.is_snapshot and prev_ts is not None and d.timestamp != prev_ts:
            _check(row_idx_of_event_end)

        book.apply_diff(d.side, d.price, d.qty)

        if not d.is_snapshot:
            prev_ts = d.timestamp
            row_idx_of_event_end = i

    if prev_ts is not None:
        _check(row_idx_of_event_end)

    pct = 100 * crossed / events if events else 0.0
    print(f"events checked: {events:,}")
    print(f"crossed:        {crossed:,} ({pct:.4f}%)")

    if crossed:
        print("\nfirst crossed samples (row_idx, best_bid, best_ask):")
        for s in samples:
            print(f"  {s[0]:>10}  bid={s[1]:.2f}  ask={s[2]:.2f}")
        return 1
    print("\nPASS: 0 crossed events")
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
