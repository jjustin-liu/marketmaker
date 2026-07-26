# Engine benchmarks

Throughput and per-event latency of the order-book engines, measured by
`scripts/benchmark_engine.py` over a real recorded BTCUSDT L2 event
stream (1,000,000 events from `data/raw/btcusdt_depth_2026-07-08.parquet`).

## Method

- Events are the raw L2 diff stream (side, price, qty), replayed in order.
- **Throughput** is bulk-timed: total wall time over N events, N / elapsed.
- **Latency** is per-call `perf_counter_ns` around each op on a 200k
  sample; the median back-to-back timer cost (~41 ns) is subtracted.
- 50k-event warmup before timing (caches, branch predictors, allocator).
- Redis is **not** in this path. It is an async state bus off the hot
  loop, so the engine latency is measured in isolation.
- Numbers include the Python→C++ pybind11 boundary crossing, so they
  reflect the real cost the Python-orchestrated system pays per event.

Hardware: Apple Silicon (darwin), Python 3.11, release build of the
pybind11 extension. Re-run with:

```
python -m scripts.benchmark_engine \
    --input data/raw/btcusdt_depth_2026-07-08.parquet --events 1000000
```

## Results

| Workload | Engine | Throughput | mean | p50 | p99 |
|----------|--------|-----------:|-----:|----:|----:|
| `apply_diff` | **C++ MatchEngine** | **1,745,334 ev/s** | 0.573 µs | 0.626 µs | 2.917 µs |
| `apply_diff` + top-of-book | **C++ MatchEngine** | **1,274,258 ev/s** | 0.785 µs | 0.751 µs | 3.167 µs |
| `apply_diff` | Python OrderBook | 660,713 ev/s | 1.514 µs | 1.209 µs | 3.709 µs |

`apply_diff` is one book mutation (one L2 event). The `+ top-of-book`
row adds `best_bid()`/`best_ask()` — the work the quoting loop does per
event.

## Interpretation

- The C++ hot path sustains **>1.2M events/sec** with **sub-microsecond
  median** and **~3 µs p99** per event, including the pybind11 crossing.
- That is ~8–11× the 150k events/sec target and well inside a 20 µs
  latency budget — the budget is dominated not by the book op but by
  network/exchange round-trips, which sit outside this measurement.
- The C++ engine is ~2.4× the Python book on throughput and ~2× on
  median latency; the pure-Python book is retained for backtests and as
  the contract oracle in tests.

## Honest scope

These measure the **engine**, not end-to-end order latency (which would
include exchange network RTT, feed decode, and strategy compute).
"Sub-20 µs" refers to the C++ book/event-handling path only. Redis
round-trips (~30–100 µs) are deliberately kept off this path.
