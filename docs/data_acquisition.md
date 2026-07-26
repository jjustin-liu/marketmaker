# Acquiring TB-scale L2 data

The pipeline ingests, streams, and accounts for real L2 at scale. What's
left to hit a literal "5 TB L2 backtest" is *obtaining* the bytes. Three
real sources, with the exact steps.

## Volume arithmetic

- Throttled top-of-book (`@depth@100ms`, what we self-record today):
  ~14M rows/day, ~33 MB/day compressed for BTCUSDT. Too small to reach
  TBs from one symbol.
- **Full-depth incremental L2** (`@depth`, every update, all levels):
  ~5–15 GB/day **uncompressed** per liquid symbol. Hence:
  - BTCUSDT full L2 × ~1 year ≈ **2–5 TB uncompressed**, or
  - ~30–50 symbols × 1 month, or a few symbols × a quarter.

Parquet+zstd compresses this ~3–6×; the catalog reports both the on-disk
and the logical uncompressed figure (the number a vendor quotes).

## Path A — Tardis.dev (canonical, paid)

Full historical incremental L2 for Binance, years deep.

```bash
# 1. Download incremental_book_L2 CSV.gz (via Tardis API/client), e.g. a
#    year of BTCUSDT, or many symbol-days.
# 2. Convert to our schema (bounded memory, one shard per file):
python -m scripts.ingest_tardis --inputs data/tardis/*.csv.gz \
    --out-dir data/l2_shards --chunksize 1000000
# 3. Confirm the volume (the receipt for the claim):
python -m scripts.data_catalog --dir data/l2_shards --uncompressed
# 4. Backtest the whole corpus, one shard resident at a time:
python -c "from pathlib import Path; \
from src.backtest.engine import BacktestEngine, stream_diffs_from_parquets; \
from src.strategy.naive_maker import NaiveMaker; \
print(BacktestEngine(strategy=NaiveMaker(), use_inventory=False).run( \
  stream_diffs_from_parquets(sorted(Path('data/l2_shards').glob('*.parquet')))).pnl)"
```

`scripts/ingest_tardis.py` maps Tardis columns
(`timestamp,is_snapshot,side,price,amount`) onto the depth schema and is
verified to replay uncrossed through the engine.

## Path B — Self-record multi-symbol full-depth (free, slow)

Run the recorder per symbol on the unthrottled `@depth` stream across the
top-N pairs, partitioned by symbol-date into `data/l2_shards/`. Weeks–
months across dozens of symbols reaches TB scale. Needs a always-on host
(a VPS in a Binance-supported region) and disk. This is a recorder
extension (multi-symbol + full-depth) — the ingest/stream/catalog side is
already done.

## Path C — Binance Data Vision (free, different granularity)

`data.binance.vision` publishes free historical `bookTicker` (best-bid/
ask updates) and `bookDepth` snapshots for many symbols/years. Across the
full symbol universe this is multi-TB and free. Note it is BBO updates +
periodic snapshots, not full incremental L2 — state the granularity
honestly ("bookTicker/bookDepth" rather than "full L2") if you use it.

## Cost / storage reality

- 5 TB uncompressed is ~1 TB on disk after zstd — fits one drive; keep it
  in S3 and stream with pyarrow/fsspec if you don't want it local.
- Tardis is a paid subscription; Data Vision is free; self-recording
  costs a VPS + uptime.
- The engineering (ingest, bounded-memory stream, catalog, backtest) is
  done and validated; only the data purchase/recording remains.
