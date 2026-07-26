"""Ingest Tardis.dev incremental_book_L2 CSVs into our depth parquet schema.

Tardis is the canonical source of TB-scale historical crypto L2. Its
`incremental_book_L2` CSV columns are:

  exchange, symbol, timestamp, local_timestamp, is_snapshot, side,
  price, amount

which map directly onto the depth schema this project replays
(timestamp, side, price, qty, is_snapshot). Files are read in bounded
chunks and written as parquet row groups, so an arbitrarily large CSV
(or CSV.gz) converts with flat memory — one shard out per file in,
partitioned by whatever the source file granularity is (typically a
symbol-day). Point `stream_diffs_from_parquets` at the output dir to
replay the whole corpus.

Usage:
  python -m scripts.ingest_tardis --inputs data/tardis/*.csv.gz \\
      --out-dir data/l2_shards --chunksize 1000000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("ingest_tardis")

SRC_COLS = ["timestamp", "is_snapshot", "side", "price", "amount"]
OUT_SCHEMA = pa.schema([
    ("timestamp", pa.int64()),
    ("side", pa.string()),
    ("price", pa.float64()),
    ("qty", pa.float64()),
    ("is_snapshot", pa.bool_()),
])


def _to_table(chunk: pd.DataFrame) -> pa.Table:
    """Map a Tardis chunk to our depth schema."""
    is_snap = chunk["is_snapshot"].astype(str).str.lower().isin(("true", "1"))
    out = pd.DataFrame({
        "timestamp": chunk["timestamp"].astype("int64"),
        # loader accepts 'bid'/'ask'; keep them verbatim
        "side": chunk["side"].astype(str).str.lower(),
        "price": chunk["price"].astype("float64"),
        "qty": chunk["amount"].astype("float64"),
        "is_snapshot": is_snap.astype(bool),
    })
    return pa.Table.from_pandas(out, schema=OUT_SCHEMA, preserve_index=False)


def convert_file(src: Path, out_dir: Path, chunksize: int) -> int:
    """Convert one Tardis CSV(.gz) to a parquet shard. Returns rows written."""
    out_path = out_dir / (src.name.split(".")[0] + ".parquet")
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for chunk in pd.read_csv(src, usecols=SRC_COLS, chunksize=chunksize):
            table = _to_table(chunk)
            if writer is None:
                writer = pq.ParquetWriter(out_path, OUT_SCHEMA,
                                          compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    logger.info("  %s -> %s (%d rows)", src.name, out_path.name, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Tardis L2 -> depth parquet.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="Tardis incremental_book_L2 CSV(.gz) files")
    parser.add_argument("--out-dir", type=Path, default=Path("data/l2_shards"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs: List[Path] = [p for p in args.inputs if p.exists()]
    if not inputs:
        raise SystemExit("no input files found")

    total = 0
    for src in inputs:
        total += convert_file(src, args.out_dir, args.chunksize)
    logger.info("converted %d files, %d total rows -> %s",
                len(inputs), total, args.out_dir)


if __name__ == "__main__":
    main()
