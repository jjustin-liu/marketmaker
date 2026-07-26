"""Catalog a directory of L2 parquet shards: rows and bytes processed.

Gives the "N TB backtest" claim a receipt. Scans shards, sums on-disk
bytes and row counts, and (with --uncompressed) estimates the logical
uncompressed volume using the parquet metadata compression ratio — the
figure a data vendor quotes and the honest number for "X TB of L2 data."

Usage:
  python -m scripts.data_catalog --dir data/l2_shards
  python -m scripts.data_catalog --dir data/l2_shards --uncompressed
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import pyarrow.parquet as pq

logger = logging.getLogger("data_catalog")

TB = 1024 ** 4
GB = 1024 ** 3


def scan(shard_dir: Path) -> List[Tuple[str, int, int, int]]:
    """Return (name, rows, on_disk_bytes, uncompressed_bytes) per shard."""
    out: List[Tuple[str, int, int, int]] = []
    for path in sorted(shard_dir.glob("*.parquet")):
        md = pq.ParquetFile(path).metadata
        uncompressed = sum(
            md.row_group(i).column(j).total_uncompressed_size
            for i in range(md.num_row_groups)
            for j in range(md.num_columns)
        )
        out.append((path.name, md.num_rows, path.stat().st_size, uncompressed))
    return out


def _fmt(nbytes: int) -> str:
    if nbytes >= TB:
        return f"{nbytes / TB:.2f} TB"
    return f"{nbytes / GB:.2f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 shard catalog.")
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--uncompressed", action="store_true",
                        help="report logical uncompressed volume too")
    parser.add_argument("--top", type=int, default=10,
                        help="list this many shards (0 = summary only)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.dir.exists():
        raise SystemExit(f"dir not found: {args.dir}")
    rows = scan(args.dir)
    if not rows:
        raise SystemExit(f"no parquet shards in {args.dir}")

    n_rows = sum(r for _, r, _, _ in rows)
    n_disk = sum(d for _, _, d, _ in rows)
    n_uncomp = sum(u for _, _, _, u in rows)

    for name, r, d, u in rows[:args.top]:
        line = f"  {name:<40} {r:>14,} rows  {_fmt(d):>10} on-disk"
        if args.uncompressed:
            line += f"  {_fmt(u):>10} uncompressed"
        logger.info(line)
    if args.top and len(rows) > args.top:
        logger.info("  ... and %d more shards", len(rows) - args.top)

    logger.info("-" * 60)
    logger.info("shards:            %d", len(rows))
    logger.info("total rows:        %s", f"{n_rows:,}")
    logger.info("on-disk (parquet): %s", _fmt(n_disk))
    if args.uncompressed:
        ratio = n_uncomp / n_disk if n_disk else 0.0
        logger.info("uncompressed L2:   %s  (%.1fx)", _fmt(n_uncomp), ratio)


if __name__ == "__main__":
    main()
