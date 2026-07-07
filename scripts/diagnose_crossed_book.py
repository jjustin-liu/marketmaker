"""Diagnose the residual crossed-book ticks in post-bootstrap parquets.

Two hypotheses for why ~8% of replayed ticks are crossed, concentrated
in the snapshot-warmup region:

    H1  float-precision key mismatch on book.pop(price, None) in
        OrderBook.apply_diff — a delete diff arrives with a price that
        round-trips through JSON/Decimal and no longer equals (bit-exact)
        the float key inserted by the snapshot, so the level survives.

    H2  snapshot/diff interleaving — a delete diff for level X is
        replayed before the snapshot row that inserts level X. The
        delete no-ops, then the snapshot insert creates a stale level.

This script runs both checks against one parquet and reports.
"""

from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

TICK_SIZE = 0.01  # BTCUSDT spot price increment
NEAR_TICK_TOL = TICK_SIZE / 2  # half a tick — anything closer is float fuzz

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.lob.order_book import OrderBook, Side  # noqa: E402


def _row_side(s: object) -> Side:
    if isinstance(s, str):
        return Side.BUY if s.lower() in ("buy", "bid", "b") else Side.SELL
    return Side(int(s))


def h1_float_precision(df: pd.DataFrame) -> dict:
    """For every qty=0 delete, classify it:
      - bit_exact_hit:  price exists as a bit-exact key (delete will succeed)
      - near_seen:      no bit-exact hit, but a seen price within half a tick
                        — i.e. float-precision drift; OrderBook.pop misses
      - far_unseen:     nearest seen price > half a tick away — likely a
                        below-depth-1000 delete the snapshot never seeded;
                        harmless, the level wasn't in our book anyway
    """
    seen: dict[Side, set[float]] = defaultdict(set)
    sorted_seen: dict[Side, list[float]] = defaultdict(list)

    bit_exact_hit = 0
    near_seen = 0
    far_unseen = 0
    deletes_total = 0

    for _, row in df.iterrows():
        side = _row_side(row["side"])
        price = float(row["price"])
        qty = float(row["qty"])
        if qty == 0.0:
            deletes_total += 1
            if price in seen[side]:
                bit_exact_hit += 1
            else:
                arr = sorted_seen[side]
                if not arr:
                    far_unseen += 1
                else:
                    i = bisect.bisect_left(arr, price)
                    candidates = []
                    if i < len(arr):
                        candidates.append(arr[i])
                    if i > 0:
                        candidates.append(arr[i - 1])
                    nearest_gap = min(abs(price - c) for c in candidates)
                    if nearest_gap <= NEAR_TICK_TOL:
                        near_seen += 1
                    else:
                        far_unseen += 1
        else:
            if price not in seen[side]:
                seen[side].add(price)
                bisect.insort(sorted_seen[side], price)
    return {
        "deletes_total": deletes_total,
        "bit_exact_hit": bit_exact_hit,
        "near_seen": near_seen,
        "far_unseen": far_unseen,
    }


def replay_naive(df: pd.DataFrame, n_check: int) -> int:
    """Apply rows in file order. Count crossed ticks in first n_check."""
    book = OrderBook()
    crossed = 0
    for i, row in enumerate(df.itertuples(index=False)):
        if i >= n_check:
            break
        book.apply_diff(_row_side(row.side), float(row.price), float(row.qty))
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is not None and ba is not None and bb >= ba:
            crossed += 1
    return crossed


def replay_snapshot_first(
    df: pd.DataFrame, snapshot_rows: int, n_check: int
) -> int:
    """Apply the first `snapshot_rows` as a snapshot block, then diffs.
    Within the snapshot block, sort by side/price so the book is
    consistent before any delete-diff fires.
    """
    snap = df.iloc[:snapshot_rows]
    rest = df.iloc[snapshot_rows:]
    book = OrderBook()
    # apply snapshot rows ignoring any qty=0 rows that snuck in
    for row in snap.itertuples(index=False):
        if float(row.qty) == 0.0:
            continue
        book.apply_diff(
            _row_side(row.side), float(row.price), float(row.qty)
        )
    crossed = 0
    for i, row in enumerate(rest.itertuples(index=False)):
        if i >= n_check:
            break
        book.apply_diff(_row_side(row.side), float(row.price), float(row.qty))
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is not None and ba is not None and bb >= ba:
            crossed += 1
    return crossed


def main() -> None:
    parquet = REPO_ROOT / "data" / "raw" / "btcusdt_depth_2026-05-24.parquet"
    df = pd.read_parquet(parquet)
    print(f"file: {parquet.name}  rows: {len(df)}")

    print("\n--- H1: float-precision key mismatch (tightened) ---")
    r = h1_float_precision(df)
    tot = r["deletes_total"] or 1
    print(f"qty=0 deletes total: {r['deletes_total']:,}")
    print(
        f"  bit-exact hit (pop succeeds):     "
        f"{r['bit_exact_hit']:,} ({100 * r['bit_exact_hit'] / tot:.2f}%)"
    )
    print(
        f"  near-seen (<= 0.5 tick away):     "
        f"{r['near_seen']:,} ({100 * r['near_seen'] / tot:.2f}%)  "
        f"<-- float-precision mismatches"
    )
    print(
        f"  far-unseen (> 0.5 tick, no hit):  "
        f"{r['far_unseen']:,} ({100 * r['far_unseen'] / tot:.2f}%)  "
        f"<-- below-depth-1000, harmless"
    )

    print("\n--- H2: snapshot/diff interleaving ---")
    n = min(5000, len(df))
    snap_rows = 2000  # recorder writes ~2000 snapshot rows up front
    naive = replay_naive(df, n)
    fixed = replay_snapshot_first(df, snap_rows, n - snap_rows)
    print(f"  naive (file order) crossed:        {naive}/{n}")
    print(f"  snapshot-first then diffs crossed: {fixed}/{n - snap_rows}")
    print(
        "  if naive >> snapshot-first: H2 is the cause "
        "(ordering bug, not OrderBook bug)."
    )


if __name__ == "__main__":
    main()
