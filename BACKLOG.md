# Backlog

Deferred work, known gaps, and follow-ups discovered along the way.
Items here are intentional — they were skipped to keep the current
phase scoped. Revisit before going live or when the phase that owns
them comes up.

## Phase 11 follow-ups

### Round-trip-aware quoting objective
- Finding (2026-07-10): per-leg conditional edge is negative at every
  distance (edge model: mean −$4.87 over 5s), yet at-touch symmetric
  naive quoting profits (+$21/2.5h, fee 0) — value is created by the
  round trip (both legs filling within seconds), not by any single
  fill. EV = P·h and EV = P·edge both optimize the leg, so both lose;
  every inventory-protection mechanism (skew centre, size asymmetry,
  risk widening) suppresses churn and costs money on calm tape.
- Idea: optimize expected round-trip value — quote both sides at the
  touch by default; use the fill/imbalance models not for distance
  but for *when to pull one side* (toxicity gate: widen/withdraw the
  side facing momentum). Research-grade; needs careful backtest.
- Reality check to keep in the writeup: at Binance VIP0 (10 bps
  maker) all of this is structurally unprofitable on BTCUSDT's
  0.0016 bps spread; churn strategies die by fees first.

## Phase 8 follow-ups

### Quote persistence / hysteresis in the paper simulator
- Where: `src/live/order_manager.py`, `OrderManager.step`
- Symptom: 1-hour live paper run on 2026-05-19 produced 0 fills
  because every tick unconditionally overwrites `open_bid` /
  `open_ask`. The "open quote" never rests long enough for the
  market to cross it.
- Fix: only requote when the new strategy target differs from the
  currently-open quote by more than a threshold (e.g., > 0.5 bps of
  mid, or > 1 tick). Hysteresis dead zone so small noise doesn't
  trigger a requote.
- Real makers: do exactly this — quotes rest at the same price as
  long as fair value is consistent. Cancel-and-replace only fires
  on meaningful fair-value movement.
- Note: the testnet engine (`src/live/testnet_engine.py`) has the
  same design and same gap. On testnet it papers over the issue
  because the real matching engine has other participants. Add
  hysteresis there too for production hygiene.

### Fees in paper mode
- Where: `src/live/order_manager.py`, `OrderManager._book_fill`
- Symptom: paper fills go in at the quoted price with no maker fee
  or rebate. The backtest engine has `fee_bps`; the live paper
  manager does not. Paper PnL will look better than reality by
  ~1 bp / round trip on Binance VIP0.
- Fix: thread `fee_bps` through `OrderManager` the same way
  `BacktestEngine` does. Default 0 keeps existing behavior; live
  CLI flag wires it through.

### Queue position simulation
- Where: `src/backtest/engine.py` and `src/live/order_manager.py`
- Current fill rule is strict-cross: assumes our quote is at the
  *back* of the queue at its price level and only fills when the
  level is fully swept. Conservative.
- Realistic upgrade: track queue position from time-of-post,
  estimate ahead-of-us volume from trades + cancels, partially
  fill as the queue moves. Big change — separate phase.
- **UPDATE 2026-06-03: partially done.** `BacktestEngine.fill_mode`
  now supports "queue" alongside "strict_cross" (default). Queue mode
  fills on a *touch* (opposite best reaches our price, `<=` not `<`)
  and, when a quote rests on a real level, as that level's depth is
  consumed past our queue position (`OrderBook.qty_at`). run_backtest
  uses queue mode for both strategies.
- **Known limitation — no trade stream in recorded data.** The
  recorder only subscribes to `@depth`; parquets have no trades, so we
  cannot model true trade-flow fills (a fill followed by a favorable
  reversion). Our strategies also quote *between* book levels, so the
  queue-consumption path rarely engages and queue mode reduces mostly
  to touch fills. Full fidelity needs the recorder to also capture
  `@trade`/`@aggTrade` and re-record — blocked on live data access
  (testnet geo-block). Until then, depth-only touch fills are the
  ceiling on realism.

## Phase 11 tuning inputs (from joshleemarketmaker review, 2026-07-07)

Reviewed ~/Desktop/joshleemarketmaker (github joshhlee614/Market-Maker)
before EV tuning. Their EV maker is inert (grid in absolute dollars vs
bps base spread; P(fill) hardcoded 0.5 in every runnable path → argmax
always widest; vol computed but unused; no fees anywhere; crossing-only
one-fill-per-tick backtest). Confirms our architecture; do not copy
their sim. Worth adopting / verifying during tuning:
1. At retrain, verify the fitted P(fill) actually decays with distance
   fast enough that EV = P·h has an interior argmax — check the curve,
   not just AUC. Flat P ⇒ EV runs to the widest candidate (their exact
   failure, and our old symptom).
2. Fee-aware edge: EV = P(fill)·(h − fee_abs) instead of P·h. They
   never subtract fees; we apply fees to PnL but not to the objective.
3. Consider microprice (bid·ask_vol + ask·bid_vol)/(bid_vol+ask_vol)
   as the quoting centre before inventory shift — built in our Phase 4
   features, unused in quoting. Cheap directional signal.
4. Their reservation-shift formula for reference: centre_shift =
   −clip(inv/max_pos,−1,1)·skew_factor·spread, skew_factor=0.5.
   Confirm our InventorySkew centre shift is proportional like this.
5. Already ahead of them (keep): spread ceiling via
   max_half_spread_mult, live vol term, trade-print fills, queue
   awareness, fees in PnL, requote hysteresis.

## Phase 10 follow-ups

### Backtest engine evaluates per-row, not per-event
- Where: `src/backtest/engine.py`, `BacktestEngine.run`
- Finding (2026-07-07, Phase 10): rows sharing a timestamp form one
  exchange event; applying them one at a time makes the book look
  crossed mid-event (~15% of rows on a clean capture) while the
  book is 0.00% crossed at every event boundary. The old "12–19%
  residual crossing from stale deep levels" was mostly this artifact.
- The engine's per-row `crossed_skips` guard therefore skips mostly
  intra-event transients — fair (both strategies skip identically)
  but the counter overstates data problems, queue-mode touch fills
  can trigger on transient states that never existed at an event
  boundary, and refresh cadence is counted in rows not events.
- Refinement: batch diffs by timestamp in `run()` and evaluate
  fills/quotes only at event boundaries. Validators already measure
  this way (`validate_data.py`, `verify_replay_uncrossed.py`).
- Note: VISION-mirror depth events arrive at ~1s granularity
  (~100 rows/event), so event-boundary evaluation also sets the
  honest quote-refresh resolution for replays.

## Phase 7 follow-ups

### OrderBook perf — sortedcontainers swap
- Where: `src/lob/order_book.py`
- Symptom: full-day backtest (8.3M diffs × 2 strategies) takes
  ~1 hour because `best_bid()` does `max(self._bids)` and
  `best_ask()` does `min(self._asks)` on plain `dict` — O(n) over
  every price level per call, ~1000 levels on BTCUSDT.
- Fix: swap `_bids` / `_asks` from `Dict` to
  `sortedcontainers.SortedDict`. `best_bid()`/`best_ask()` drop to
  O(1) peek; `depth(k)` drops from O(n log n) sort to O(log n + k).
  Expected 5–10× speedup on the engine loop.
- New dep: `sortedcontainers` — pure-Python, no compile, used by
  many production trading libs.

### Sharpe / hit_ratio empirical re-verification
- Where: `src/backtest/engine.py`, `src/backtest/metrics.py`
- The metric fixes (resampled Sharpe, hit_ratio normalized to
  [0,1], maker fees applied) are unit-tested for correctness but
  the post-fix full-day backtest run was killed for time. Run
  once on real recorded data and snapshot the numbers.

## Phase 3 follow-ups

### Replay warmup crosses — recorder snapshot persistence
- Where: `scripts/run_recorder.py` (write side),
  `src/backtest/engine.py::load_diffs_from_parquet` (replay side)
- Symptom: replaying a post-bootstrap parquet through `OrderBook`
  shows ~8% crossed ticks, concentrated entirely in the first 20%
  of each file. The book self-heals to 0% crossed after ~1000 diffs
  (≈ the depth of a normal REST snapshot).
- **Originally hypothesized as an OrderBook `apply_diff(qty=0)`
  bug. That was wrong** — diagnostic in
  `scripts/diagnose_crossed_book.py` shows 94% of qty=0 deletes
  hit bit-exact and pop successfully, 0 float-precision mismatches,
  and the remaining 6% target levels below the snapshot's
  depth=1000 (genuinely never in our book, no-op is correct).
- **Real root cause: the parquet schema has no way to mark snapshot
  rows vs diff rows.** Two consequences:
    1. `_bootstrap_session` does call
       `rotator.extend(snapshot_to_rows(...))`, but snapshot rows
       and same-timestamp diff rows are indistinguishable under the
       current `(timestamp, side, price, qty)` schema. The replay
       loader can't tell them apart and applies them in file order.
    2. The recorder bootstraps **once at connection time**. The
       daily file rotator opens a new file at UTC midnight without
       re-writing a snapshot. So every rotated parquet (e.g.
       `2026-05-25.parquet`, day 2 of a recording that bootstrapped
       on the 24th) has *no* seeding rows at all. Replaying day 2
       in isolation starts from an empty book.
- Fix:
    1. Add `is_snapshot: bool` column to the parquet schema.
       `snapshot_to_rows` writes `True`; diff writes write `False`.
    2. On daily rotation, take an in-memory snapshot of the current
       book state and write it as the first rows of the new file
       with `is_snapshot=True`. Each file then replays
       independently.
    3. `load_diffs_from_parquet`: apply all `is_snapshot=True` rows
       first (order irrelevant — point-in-time levels), then apply
       `is_snapshot=False` rows in timestamp order. Drop the
       `skip_rows` workaround.
- Acceptance: replay any single post-fix parquet through
  `OrderBook` and assert 0 crossed ticks across the whole file (no
  warmup region, no `skip_rows`).
- **UPDATE 2026-06-03: the "sort all is_snapshot rows to the front"
  fix was wrong for real daily files.** A 24h recording reconnects
  ~34 times; each reconnect writes a fresh REST snapshot block
  *inline*, so one file holds ~35 snapshot epochs interleaved with
  diffs. Sorting all snapshot rows to the front merges 35 different
  point-in-time books into one → 100% crossed on replay.
  Corrected approach (now in `engine.py` and `validate_data.py`):
  preserve file order, and treat each `False→True` transition in
  `is_snapshot` as an epoch boundary that *resets* the book. After
  this, residual crossing is ~12–19% per file — these are stale
  top-of-book levels that never receive a qty=0 delete because they
  sit below the snapshot's depth=1000 window. The backtest engine
  now skips any tick where `best_bid >= best_ask` (non-physical
  reconstruction state); both strategies skip identically so the
  comparison stays fair. Tracked count surfaced as
  `BacktestResult.crossed_skips`.
- Remaining cleanup (not blocking): widen the recorder snapshot
  depth or periodically re-snapshot to prune stale deep levels, so
  the raw crossed-tick rate drops without the engine-side skip.

### EV strategy fill-model calibration
- Where: `src/models/fill_prob.py`,
  `scripts/train_model_lookahead.py`
- Symptom: EVMaker quotes at mid ± ~$390 on BTC, never gets filled
  in any realistic replay. The fill model was trained with
  candidate offsets sampled at `spread × uniform(0.1, 2.0)` — i.e.
  $0.001–$0.02 from mid. When EV optimizer queries P(fill at $400
  away), the model extrapolates wildly because that distance was
  never in the training distribution.
- Fix: widen the offset sampling in the lookahead trainer to a
  log-uniform range covering 0.1× spread → 50 bps of mid. Retrain,
  verify AUC holds, re-run EV backtest. Should produce realistically
  tight EV quotes that actually fill.
- **UPDATE 2026-06-03: the EVMaker quoting logic was rewritten** so it
  no longer depends on the model behaving outside its training range.
  Candidate half-spreads are now multiples of the *live market spread*
  (EVConfig.min/max_half_spread_mult), so the fill model is only ever
  queried near the touch — inside the offsets it was trained on. Edge
  is measured from the inventory-shifted centre and EV=P(fill)×edge, so
  the tightest candidate can win. The model-retraining fix above is now
  optional polish, not a prerequisite for EV to fill.

### EV inventory-skew continuity clip is absolute, not relative
- Where: `src/strategy/inventory_skew.py`, `InventorySkewConfig.continuity_clip`
- The clip caps per-refresh quote movement to an absolute $0.10. On
  BTC at ~$77k that lets the inventory-shifted centre move only $0.10
  per refresh while mid moves dollars, so the centre can lag/drift.
  Minor while inventory stays near 0 (centre ≈ mid), but should be
  re-denominated in bps of mid. Low priority.

## Other / cross-cutting

### Mainnet live trading
- Hard gate: don't even consider until paper trading (in either
  simulator or testnet form) has run cleanly for many hours with
  fills behaving as expected and PnL within 2σ of backtest
  predictions on overlapping date ranges.
