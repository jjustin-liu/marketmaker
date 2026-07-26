# Hit-ratio lift from imbalance + volatility

Measured by `scripts/eval_hit_ratio.py` on 1.5M diffs of BTCUSDT L2
(`data/raw/btcusdt_depth_2026-07-08.parquet`), ~500k labeled candidate
quotes.

## What is measured

A maker with a fixed order budget must choose *which* equidistant quotes
to place. We hold the quote distance constant (a multiple of the live
spread) so distance carries no information, label each candidate by
lookahead (did the market reach it), train a logistic model on the
earlier **time-ordered** 80% and evaluate on the later 20%.

**Hit-ratio lift = precision-at-budget:** give the maker the top-20% of
candidates by predicted P(fill) vs a random 20% (base rate).

```
lift = model_fill_rate / base_fill_rate - 1
```

Feature sets: `imbalance_1/2/5 + volatility + side` (the claim), and the
same `+ distance` as a control.

## Results (lookahead 200 ≈ a few seconds; realistic maker horizon)

| Offset | Features | AUC | base hit | model hit (top 20%) | **lift** |
|-------:|----------|----:|---------:|--------------------:|---------:|
| 0.5×spread | imbalance+volatility | 0.853 | 7.6% | 29.1% | **+280%** |
| 0.5×spread | + distance | 0.879 | 7.6% | 30.2% | +295% |
| 1.0×spread | imbalance+volatility | 0.860 | 7.5% | 28.9% | **+284%** |
| 1.0×spread | + distance | 0.886 | 7.5% | 30.1% | +300% |

Imbalance + volatility alone lift realized hit ratio ~3.8× (≈+280%),
far above the 35% target, on a held-out split with distance controlled
out.

## Honest scope / regime dependence

- The signal is strong when a fill requires a **directional move** (a
  quote a fraction of a spread or more from mid, resolved over a few
  seconds): imbalance and volatility genuinely predict that move.
- At a **very short horizon** (lookahead 50) and **at-touch** placement,
  fills are dominated by microstructure noise the features don't capture,
  and imbalance+volatility alone give little or negative lift. The +280%
  figure is the realistic multi-second regime, not a universal constant.
- This is a **fill-rate** result, not a P&L result. A higher hit ratio
  does not imply profit — filled quotes can still be adversely selected
  (see `progress.md`, Phase 11). It demonstrates the features carry
  genuine predictive signal for quote execution.

Re-run:
```
python -m scripts.eval_hit_ratio \
    --input data/raw/btcusdt_depth_2026-07-08.parquet \
    --max-diffs 1500000 --lookahead 200 --offset-mult 1.0
```
