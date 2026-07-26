# Data engineering at scale

The backtest pipeline processes large, partitioned L2 corpora with
bounded memory and CPU-parallel fan-out. Two runners:

- `stream_diffs_from_parquets` — replay a whole corpus as one iterator,
  one shard resident at a time (bounded memory; see `docs/data_scale.md`).
- `scripts/run_scale_backtest.py` — replay each shard independently,
  fanned out across cores, then aggregate. Prints a receipt: shards,
  events, on-disk volume, wall-clock, and sustained events/sec.

Each parquet shard is a self-contained replay (its own epochs), so
per-shard runs are independent and safe to parallelize; aggregate
PnL/fills/quotes sum across shards.

## Receipt — real captured data

All self-recorded BTCUSDT depth shards, 6 workers:

```
python -m scripts.run_scale_backtest --glob "data/raw/*_depth_*.parquet" --workers 6
```

| Metric | Value |
|---|---|
| Shards | 5 (multi-day) |
| **L2 events** | **36,561,491** |
| On-disk volume | 0.15 GB (parquet+zstd) |
| Wall-clock | 292.9 s (2.9× parallel) |
| Throughput | ~125k events/sec (full backtest: quoting + fills + markout) |
| Aggregate fills / quotes | 32,517 / 2,974,517 |

This is **real** market data: ~36.5M L2 events across multiple trading
days, replayed end-to-end.

## Receipt — scaled corpus (real + synthetic)

Synthetic shards (`scripts/gen_synthetic_l2.py`, deterministic and
replayable) bulk the corpus up to demonstrate the pipeline scales beyond
what was captured:

```
python -m scripts.run_scale_backtest --dir data/l2_shards_synth \
    --glob "data/raw/*_depth_*.parquet" --workers 8
```

| Metric | Value |
|---|---|
| Shards | 25 (5 real + 20 synthetic) |
| **L2 events** | **96,561,898** |
| On-disk volume | 0.84 GB (parquet+zstd); ~1.2 GB uncompressed |
| Wall-clock | 529.6 s |
| CPU-time | 2,812 s → **5.3× parallel** speedup on 8 workers |
| Throughput | **~182k events/sec** (full backtest loop) |
| Quotes evaluated | 8,973,324 |
| Peak memory | one shard per worker (bounded, ~independent of corpus size) |

Nearly **100 million L2 events** replayed through the full quoting +
fill + markout loop in under 9 minutes on a laptop, memory bounded by a
single shard. Doubling the corpus doubles wall-clock, not memory —
that's the property that reaches TB scale. (Synthetic shards use the
strict-cross fill mode with no trade file, so they add events and quotes
but few fills; the aggregate PnL/fills come from the real shards.)

## Why this scales to TB

- **Bounded memory:** one shard per worker, released before the next —
  independent of total corpus size.
- **Linear fan-out:** wall-clock ≈ (total events / throughput / workers);
  add cores or workers to cut it.
- **Throughput headroom:** the raw C++ engine sustains >1.2M events/sec
  (`docs/benchmarks.md`); the ~125k/s here is the *full* backtest loop
  (strategy + fills + markout), not the book op.
- The only thing between this and a literal multi-TB run is the data
  itself — see `docs/data_acquisition.md` (Tardis / multi-symbol
  recording / Binance Data Vision).

Reproduce:
```
python -m scripts.gen_synthetic_l2 --rows 3000000 --epochs 30 \
    --out data/l2_shards_synth/depth_00.parquet --trades /tmp/t.parquet
python -m scripts.run_scale_backtest --dir data/l2_shards_synth \
    --glob "data/raw/*_depth_*.parquet" --workers 8
python -m scripts.data_catalog --dir data/l2_shards_synth --uncompressed
```
