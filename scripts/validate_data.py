"""Data quality validator for recorded L2 depth parquets.

Replays each parquet through a Python OrderBook and reports:
  - schema completeness
  - snapshot seed presence
  - crossed-book ticks (post-warmup)
  - timestamp monotonicity gaps
  - duplicate rows

With --clean, writes a filtered parquet that drops the warmup region
and any rows that would cause a duplicate (keeping last occurrence).

Usage:
  python -m scripts.validate_data data/raw/*.parquet
  python -m scripts.validate_data data/raw/btcusdt_depth_2026-05-25.parquet --clean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.lob.order_book import OrderBook, Side

REQUIRED_COLS = {"timestamp", "side", "price", "qty"}
WARMUP_TICKS = 500  # diffs to apply before evaluating crossed-book rate


def _parse_side(s: str) -> Side:
    return Side.BUY if str(s).lower() in ("buy", "bid", "b") else Side.SELL


def validate_file(path: Path, clean: bool = False) -> bool:
    """Validate one parquet. Returns True if the file is CLEAN."""
    print(f"\nFile: {path.name}")

    df = pd.read_parquet(path)

    # 1. Schema check
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"  [ERROR] Missing columns: {missing}")
        return False

    has_snapshot_col = "is_snapshot" in df.columns
    total_rows = len(df)

    # 2. Snapshot seed presence
    if has_snapshot_col:
        snap_rows = int(df["is_snapshot"].astype(bool).sum())
        snap_ok = snap_rows > 0
        snap_str = f"{snap_rows:,} ({'seed present ✓' if snap_ok else 'MISSING ✗'})"
    else:
        snap_rows = 0
        snap_ok = False
        snap_str = "0 (no is_snapshot column ✗)"

    # 3. Duplicate rows
    dup_count = int(
        df.duplicated(subset=["timestamp", "side", "price", "qty"]).sum()
    )

    # 4. Timestamp monotonicity (diffs only, after snapshot rows)
    if has_snapshot_col:
        diff_df = df[~df["is_snapshot"].astype(bool)]
    else:
        diff_df = df
    ts_arr = diff_df["timestamp"].to_numpy()
    backwards_gaps = int((ts_arr[1:] < ts_arr[:-1]).sum()) if len(ts_arr) > 1 else 0

    # 5. Replay to detect crossed-book ticks (epoch-aware)
    # Each is_snapshot=True block after live diffs marks a reconnect — reset book.
    book = OrderBook()
    crossed = 0
    total_live_ticks = 0
    warmup_end: Optional[int] = None
    _in_snapshot = False

    for row in df.itertuples(index=False):
        is_snap = (
            bool(getattr(row, "is_snapshot", False))
            if has_snapshot_col else False
        )

        if is_snap and not _in_snapshot:
            # Epoch boundary: wipe old book state
            book = OrderBook()
            _in_snapshot = True
        elif not is_snap:
            _in_snapshot = False

        side = _parse_side(row.side)
        book.apply_diff(side, float(row.price), float(row.qty))

        if is_snap:
            continue

        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None:
            continue

        total_live_ticks += 1

        if total_live_ticks <= WARMUP_TICKS:
            if warmup_end is None and bb < ba:
                warmup_end = total_live_ticks
            continue

        if bb >= ba:
            crossed += 1

    post_warmup_ticks = max(0, total_live_ticks - WARMUP_TICKS)
    crossed_pct = (crossed / post_warmup_ticks * 100) if post_warmup_ticks > 0 else 0.0

    # Determine overall status
    issues: List[str] = []
    if not snap_ok:
        issues.append("no snapshot seed")
    if dup_count > 0:
        issues.append(f"{dup_count} duplicates")
    if backwards_gaps > 0:
        issues.append(f"{backwards_gaps} time gaps")
    if crossed_pct > 1.0:
        issues.append(f"{crossed_pct:.1f}% crossed ticks")

    status = "CLEAN" if not issues else "WARN: " + ", ".join(issues)
    status_sym = "✓" if not issues else "✗"

    print(f"  Rows:            {total_rows:,}")
    print(f"  Snapshot rows:   {snap_str}")
    print(f"  Live ticks:      {total_live_ticks:,}")
    print(f"  Crossed ticks:   {crossed_pct:.2f}% (post-warmup {WARMUP_TICKS})")
    print(f"  Time gaps:       {backwards_gaps}")
    print(f"  Duplicates:      {dup_count}")
    print(f"  Status:          {status_sym} {status}")

    if clean and path.suffix == ".parquet":
        _write_clean(df, path, has_snapshot_col, warmup_end)

    return not issues


def _write_clean(
    df: pd.DataFrame,
    source_path: Path,
    has_snapshot_col: bool,
    warmup_end: Optional[int],
) -> None:
    """Write a cleaned parquet next to the original with a _clean suffix."""
    out_path = source_path.with_name(
        source_path.stem + "_clean" + source_path.suffix
    )

    # Drop duplicate rows (keep last occurrence — latest write wins)
    before = len(df)
    df = df.drop_duplicates(
        subset=["timestamp", "side", "price", "qty"], keep="last"
    ).reset_index(drop=True)

    # File order is preserved — epoch boundaries are positional, not column-sorted

    df.to_parquet(out_path, index=False)
    print(f"  Cleaned:         {out_path.name} ({before - len(df):,} rows dropped)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate L2 depth parquets.")
    parser.add_argument("files", nargs="+", type=Path,
                        help="Parquet file(s) to validate")
    parser.add_argument("--clean", action="store_true",
                        help="Write a cleaned copy of each file alongside original")
    args = parser.parse_args()

    all_clean = True
    for p in args.files:
        if not p.exists():
            print(f"\n[ERROR] File not found: {p}")
            all_clean = False
            continue
        ok = validate_file(p, clean=args.clean)
        if not ok:
            all_clean = False

    print()
    sys.exit(0 if all_clean else 1)


if __name__ == "__main__":
    main()
