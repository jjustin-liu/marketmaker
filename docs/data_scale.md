# Backtesting at data-set scale

The backtest engine consumes an **iterator** of diffs, not a list:

```python
def run(self, diffs: Iterable[Diff], trades=None) -> BacktestResult: ...
```

`stream_diffs_from_parquets(paths)` (in `src/backtest/engine.py`) yields
diffs across many parquet shards while holding **one shard in memory at
a time**, releasing it before opening the next. So the total data set is
bounded by disk and time, not RAM — the standard partitioned-shard
pattern for TB-scale L2 replay.

```python
from pathlib import Path
from src.backtest.engine import BacktestEngine, stream_diffs_from_parquets
from src.strategy.naive_maker import NaiveMaker

shards = sorted(Path("data/raw").glob("btcusdt_depth_*.parquet"))
eng = BacktestEngine(strategy=NaiveMaker(), use_inventory=False)
result = eng.run(stream_diffs_from_parquets(shards))
```

## Why this reaches TB scale

- **Throughput:** the C++ engine sustains **>1.2M events/sec** end-to-end
  (see `docs/benchmarks.md`), so throughput is not the bottleneck.
- **Memory:** streaming caps resident memory at one shard regardless of
  total size. Partition the corpus into day/hour shards and the same run
  loop replays 5 GB or 5 TB.
- **Arithmetic:** an L2 diff is ~30–50 B in parquet, so ~5 TB ≈ 10¹¹
  events; at ~1M events/sec that is ~30 CPU-hours — an offline batch job,
  not a memory problem. Shard-parallelize across cores/workers to cut
  wall-clock.

## Honest scope

- Validated end-to-end on GB-scale real captures (~28 h, 20 M rows) and
  on synthetic data (`scripts/gen_synthetic_l2.py`, arbitrary volume).
  We have not personally stored 5 TB; the architecture supports it
  (bounded memory + benchmarked throughput), and a vendor L2 archive
  (e.g. Tardis.dev) or months of `run_recorder` output would supply it.
- Two engine-side caveats for very long single runs: `trades` are sorted
  once up front (chunk per shard for true streaming), and markout keeps
  processed-mid history in memory (bounded by *ticks*, far below raw row
  count). Per-shard runs sidestep both.
