# Phases

## How to use this file
One phase at a time. Never start the next phase until I say "proceed".
After each phase output a verification table showing every item below.
All items must be green before we move on.

---

## Phase 1 — Project scaffold and environment

### Goal
Get a working Python 3.11 project with correct structure,
dependencies installed, linting passing, and tests running.
This phase is pure setup — no real code yet.

### Tasks
- Create pyproject.toml with all dependencies listed
- Create Makefile with install, test, lint, format commands
- Create .flake8 config (max line length 88)
- Create .gitignore (venv, __pycache__, *.pyc, .env, data/*.parquet)
- Create pytest.ini
- Create placeholder test so make test runs without error
- Create .github/workflows/ci.yml so tests run on every push
- Confirm make lint passes on empty scaffold
- Confirm make test passes
- Push to GitHub and confirm CI passes

### Done when
| Item | Must be true |
|------|-------------|
| Python version | python --version shows 3.11.x |
| Dependencies | pip install -e ".[dev]" completes without error |
| make lint | passes with no errors |
| make test | runs and shows at least 1 passed test |
| CI | GitHub Actions workflow exists and passes on push |
| Structure | all src/ subdirs exist with __init__.py files |

---

## Phase 2 — C++ match engine and Python order book

### Goal
Build the core data structure that maintains the order book.
Two implementations: C++ for live performance, Python for testing.
Everything downstream depends on this being correct.

### Tasks
- Write match_engine.cpp with BidBook, AskBook, MatchEngine class
- Implement insert(order_id, side, price, size, timestamp) → fills
- Implement cancel(order_id) → bool
- Write pybind11 bindings exposing MatchEngine, Side, Fill to Python
- Update pyproject.toml to compile C++ extension on install
- Write order_book.py — Python implementation, same interface
- Write tests/test_lob.py covering:
    insert buy order that matches existing ask → fill returned
    insert sell order that matches existing bid → fill returned
    insert order with no match → rests in book
    cancel existing order → returns true, order gone
    cancel nonexistent order → returns false
    price priority: best price matched first
    time priority: earlier order matched first at same price
    qty = 0 diff applied → price level deleted
- Confirm from src.lob import MatchEngine works in Python
- make test passes

### Done when
| Item | Must be true |
|------|-------------|
| C++ compiles | pip install -e . builds extension without error |
| Python import | from src.lob import MatchEngine, Side works |
| Match logic | buy crosses ask → fill returned with correct price/size |
| Match logic | sell crosses bid → fill returned with correct price/size |
| Rest logic | non-crossing order sits in book |
| Cancel | order removed correctly |
| Price priority | best price always matched first |
| Time priority | FIFO at same price level |
| Python LOB | order_book.py passes same test suite |
| make test | all tests pass |

---

## Phase 3 — Data feed

### Goal
Connect to Binance WebSocket, maintain a correctly synchronized
local order book, and write live state to Redis.
After this phase Redis keys are updating in real time.

### Tasks
- Write src/data_feed/binance_ws.py
    connect to btcusdt@depth WebSocket stream
    connect to btcusdt@trade stream
    handle reconnection with exponential backoff
- Write src/data_feed/bootstrap.py
    fetch REST snapshot GET /api/v3/depth?symbol=BTCUSDT&limit=1000
    implement full bootstrap sequence (buffer → snapshot → apply)
    detect sequence gaps and restart
- Write src/data_feed/redis_writer.py
    write all Redis keys defined in ARCHITECTURE.md after each update
    write trade events to trades:recent ringbuffer (maxlen 100)
    tag each trade with aggressor side
- Write tests/test_data_feed.py covering:
    bootstrap sequence applies diffs in correct order
    sequence gap triggers restart
    qty=0 diff deletes price level
    redis keys written correctly after update
- Run feed for 60 seconds against Binance testnet
- Confirm all Redis keys update in real time

### Done when
| Item | Must be true |
|------|-------------|
| WebSocket | connects to both depth and trade streams |
| Bootstrap | correct sequence: buffer → snapshot → apply |
| Gap detection | sequence gap triggers full restart |
| Diffs applied | local book matches Binance book after 60s |
| Redis | all lob: and trades: keys updating every 100ms |
| Aggressor side | each trade tagged buy or sell correctly |
| Reconnect | feed reconnects automatically after disconnect |
| make test | all tests pass |

---

## Phase 4 — Feature generation

### Goal
Read LOB state from Redis and compute the full feature vector.
Every feature must be tested against known inputs with expected outputs.
These features are the inputs to the model — they must be correct.

### Tasks
- Write src/features/imbalance.py
    calculate_imbalance(bids, asks, levels) → float
    get_imbalance_features(bids, asks) → dict with 1, 2, 5 levels
    formula: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    range: -1.0 to +1.0
- Write src/features/micro_price.py
    calculate_microprice(bids, asks) → Decimal
    formula: (best_bid × best_ask_vol + best_ask × best_bid_vol)
             / (best_bid_vol + best_ask_vol)
- Write src/features/volatility.py
    VolatilityCalculator class with rolling window deque
    update(mid_price) → float
    window_size: 100 mid prices
    formula: std(returns) × sqrt(252) annualized
- Write src/features/feature_generator.py
    reads all lob: and trades: keys from Redis
    assembles full 10-element feature vector
    returns as numpy array in correct order:
    [bid_ask_spread, mid_price, bid_volume, ask_volume,
     imbalance_1, imbalance_2, imbalance_5,
     price_distance, size, side]
- Write tests/test_features.py covering:
    imbalance = 0 when bid_vol == ask_vol
    imbalance = +1 when ask_vol == 0
    imbalance = -1 when bid_vol == 0
    microprice pulls toward thinner side
    volatility = 0 with fewer than 2 prices
    volatility increases with more volatile price series
    feature vector has correct length and order

### Done when
| Item | Must be true |
|------|-------------|
| imbalance_1 | correct on known inputs |
| imbalance_2 | correct on known inputs |
| imbalance_5 | correct on known inputs |
| microprice | pulls toward thinner side correctly |
| volatility | increases monotonically with volatile prices |
| feature vector | 10 elements in correct order |
| Redis read | feature_generator reads live Redis correctly |
| make test | all tests pass |

---

## Phase 5 — Fill probability model

### Goal
Build, train, evaluate, and persist the fill probability model.
At the end of this phase the model loads from disk and runs
inference in under 1ms.

### Tasks
- Write src/models/fill_prob.py
    FillFeatures dataclass with to_array() method
    FillProbabilityModel class with:
      extract_features(bids, asks, price, size, side) → FillFeatures
      train(fills_df) → auc score
        for each fill: 1 positive example
        for each fill: 5 negative examples at worse prices
        StandardScaler fit on train set only
        LogisticRegression fit
        evaluate with roc_auc_score
      predict(bids, asks, price, size, side) → float
      save() → joblib.dump to data/models/fill_prob.joblib
      load() → joblib.load from data/models/fill_prob.joblib
- Write scripts/train_model.py
    loads historical fill data from data/
    trains model
    prints AUC score
    saves model
- Write tests/test_fill_prob.py covering:
    predict raises RuntimeError if model not trained
    predict returns float between 0 and 1
    model closer to mid price → higher fill probability
    model further from mid price → lower fill probability
    save then load → same predictions
    AUC on test set > 0.6

### Done when
| Item | Must be true |
|------|-------------|
| FillFeatures | to_array() returns 10-element array in correct order |
| Training | runs without error on synthetic fill data |
| AUC | test set AUC > 0.6 |
| Probability direction | closer to mid = higher P(fill) |
| Inference speed | predict() runs in under 1ms |
| Persistence | save → load → same predictions |
| make test | all tests pass |

---

## Phase 6 — Strategy

### Goal
Build all three strategy components:
NaiveMaker as baseline, InventorySkew for risk management,
SizeCalculator for sizing, EVMaker as the full strategy.

### Tasks
- Write src/strategy/naive_maker.py
    NaiveMakerConfig dataclass
    NaiveMaker.quote_prices() → (Quote, Quote)
    posts at mid ± half_spread
    tightens inside market spread when best_bid/ask provided
- Write src/strategy/inventory_skew.py
    InventorySkewConfig dataclass with validation
    InventorySkew.apply_skew(mid, inventory) → (bid, ask)
    implement all 5 steps from ARCHITECTURE.md exactly
- Write src/strategy/size_calculator.py
    SizeConfig dataclass
    SizeCalculator.get_sizes(inventory) → (bid_size, ask_size)
    LINEAR and SIGMOID scaling modes
- Write src/strategy/ev_maker.py
    EVConfig dataclass
    EVMaker.quote_prices() → (Quote, Quote)
    implement all 7 steps from ARCHITECTURE.md exactly
- Write tests/test_strategy.py covering:
    naive maker: ask > bid always
    naive maker: spread matches config
    inventory skew: long inventory → ask tighter than bid
    inventory skew: short inventory → bid tighter than ask
    inventory skew: neutral → symmetric quotes
    inventory skew: continuity clip limits movement
    size calculator: long → ask_size > bid_size
    size calculator: short → bid_size > ask_size
    size calculator: max long → bid_size = 0
    ev maker: ask > bid always
    ev maker: higher fill prob offset wins
    ev maker: min spread enforced

### Done when
| Item | Must be true |
|------|-------------|
| NaiveMaker | ask always > bid |
| NaiveMaker | spread matches config exactly |
| InventorySkew | long position skews ask tighter |
| InventorySkew | short position skews bid tighter |
| InventorySkew | continuity clip works correctly |
| SizeCalculator | sizes move in correct direction with inventory |
| SizeCalculator | max inventory zeroes out correct side |
| EVMaker | selects highest EV offset |
| EVMaker | min spread enforced on output |
| make test | all tests pass |

---

## Phase 7 — Backtest engine

### Goal
Replay historical L2 data through the full pipeline.
Compare NaiveMaker vs EVMaker.
Output real performance metrics.
Target: EVMaker shows 35%+ better hit ratio than NaiveMaker.

### Tasks
- Write src/backtest/engine.py
    load parquet files from data/ in chronological order
    replay each diff through Python OrderBook
    compute feature vector at each step
    run both NaiveMaker and EVMaker
    simulate fills (worst-case queue position)
    track inventory and PnL for both
- Write src/backtest/metrics.py
    calculate_sharpe(pnl_series) → float annualized
    calculate_hit_ratio(fills, quotes) → float
    calculate_adverse_selection(fills, mid_prices) → float
    calculate_max_drawdown(pnl_series) → float
- Write scripts/run_backtest.py
    runs backtest over date range
    prints comparison table: NaiveMaker vs EVMaker
    saves results to data/backtest_results.csv
- Write tests/test_backtest.py covering:
    fill simulation: bid fills when ask crosses our level
    fill simulation: ask fills when bid crosses our level
    fill simulation: no fill when price doesnt reach level
    metrics: sharpe of flat PnL = 0
    metrics: hit ratio calculation correct

### Done when
| Item | Must be true |
|------|-------------|
| Data loading | parquet files load without error |
| LOB replay | book state correct after replaying diffs |
| Fill simulation | fills trigger at correct price levels |
| NaiveMaker metrics | PnL, Sharpe, hit ratio all output |
| EVMaker metrics | PnL, Sharpe, hit ratio all output |
| Hit ratio | EVMaker hit ratio >= 35% better than NaiveMaker |
| Adverse selection | cost calculated per fill |
| make test | all tests pass |

---

## Phase 8 — Paper trading

### Goal
Run the full live pipeline end-to-end without real money.
Connects to real Binance data, runs real strategy,
simulates fills locally instead of sending real orders.
Must run for 1 hour without errors before going live.

### Tasks
- Write src/live/order_manager.py
    track open bid and ask order state
    read strategy:target_bid and strategy:target_ask from Redis
    paper mode: simulate fills when price crosses quote level
    update position:inventory and position:pnl in Redis
    log all fills to data/fills.log
- Write src/live/risk_guard.py
    halt trading if inventory exceeds max_position
    halt trading if PnL drawdown exceeds max_drawdown
    cancel all orders and stop on any unhandled exception
- Write src/cli.py
    backtest command → src/backtest/engine.py
    live command → src/live/order_manager.py
    load config from environment variables
    set up logging
    handle graceful shutdown on Ctrl+C
- Write tests/test_live.py covering:
    paper fill triggers when price crosses bid level
    paper fill triggers when price crosses ask level
    inventory updates correctly after fill
    risk guard halts at max inventory
    risk guard halts at max drawdown
- Run paper trading for 1 hour
- Confirm no errors in logs
- Confirm inventory stays within bounds
- Confirm Redis keys updating throughout

### Done when
| Item | Must be true |
|------|-------------|
| CLI | both backtest and live commands work |
| Paper fills | simulate correctly when price crosses level |
| Inventory tracking | updates correctly after every fill |
| PnL tracking | updates correctly after every fill |
| Risk guard | halts at max_position |
| Risk guard | halts at max_drawdown |
| 1 hour run | no errors in logs |
| Redis | all keys updating throughout run |
| make test | all tests pass |

---

## Phase 9 — Monitoring (optional)

### Goal
Visibility into the running system.
A dashboard showing live PnL, fill rate, inventory, and latency.

### Tasks
- Write docker/docker-compose.yml
    Prometheus service
    Grafana service
    volumes for persistence
- Write docker/prometheus.yml
    scrape config for the market maker process
- Add metrics endpoint to src/live/order_manager.py
    expose: current PnL, inventory, fill rate, quote refresh rate
    expose: feed latency (time since lob:last_update)
- Import Grafana dashboard JSON with panels for all metrics
- Write docs/monitoring.md explaining how to run the stack

### Done when
| Item | Must be true |
|------|-------------|
| Docker stack | docker-compose up starts without error |
| Prometheus | scrapes market maker metrics |
| Grafana | dashboard shows PnL, inventory, fill rate |
| Feed latency | visible on dashboard |
| docs | monitoring.md explains setup |

---

## Phase 10 — Live-path and recorder fixes

### Goal
Close the gaps between "code exists" and "system runs credibly":
recorder produces clean full-day captures, live mode degrades
gracefully, EV risk terms work identically in backtest and live,
and the backtest fill mode is user-controllable.

### Tasks
- Recorder: SNAPSHOT_LIMIT 1000 → 5000
- Recorder: apply is_snapshot rows to the internal live book,
  clear the live book on every re-bootstrap
- Recorder: periodic re-snapshot (--resnapshot-hours, default 6.0)
  to prune stale deep levels mid-file
- CLI live mode: missing/corrupt fill model → warn + uniform
  P(fill)=0.5 fallback (mirror run_backtest), not a crash
- Thread VolatilityCalculator into OrderManager and TestnetEngine
  so EVMaker's vol_risk_factor is live
- scripts/run_feed.py --forever flag; Makefile feed/paper/record targets
- scripts/run_backtest.py --fill-mode {queue,trades,strict_cross}
  override; fix stale fill_mode and loader docstrings

### Done when
| Item | Must be true |
|------|-------------|
| Recorder snapshot | depth 5000, live book seeded from snapshot rows |
| Re-snapshot | fresh is_snapshot block written every N hours |
| Model fallback | live mode runs with model file missing, warns loudly |
| Volatility | live strategies receive per-tick volatility |
| Feed | make feed runs indefinitely, Ctrl-C exits clean |
| Fill mode | --fill-mode override works, bad combos exit clearly |
| make test | all tests pass, new behaviors covered |
| make lint | clean |

---

## Phase 11 — Clean 24h capture, retrain, credible backtest

### Goal
One full day of depth+trades recorded with the fixed recorder,
fill model retrained on it, and a Naive-vs-EV backtest with real
trade-driven fills. Honest numbers either way.

### Tasks
- Run recorder ~24h (depth@100ms + trades, 6h re-snapshots)
- Validate capture: validate_data.py + verify_replay_uncrossed.py
- Retrain fill model on the new depth data (train_model_lookahead)
- Backtest both strategies with --fill-mode trades --fee-bps 10
- If EV loses: tune EVConfig band/risk factors, consider fee-aware
  edge (h − fee), vary size in trainer rows. Time-boxed.

### Done when
| Item | Must be true |
|------|-------------|
| Capture | both depth and trades parquets for the same day |
| Crossed ticks | well below the old 12–19% residual |
| Model | retrained, AUC >= ~0.75, old joblib kept until validated |
| Backtest | trades fill mode engaged, fees applied |
| Comparison | EV vs Naive table reported with honest numbers |

---

## Phase 12 — Adverse-selection alpha signal (OFI + microprice)

### Goal
Build and *offline-validate* a short-horizon directional signal that
predicts the next-few-seconds mid move, so a later phase can quote
only when expected edge beats adverse selection. This phase writes NO
strategy code and changes no quoting — it only proves (or disproves)
that a usable signal exists on our tape. If the signal has no edge,
we stop here and write up the negative result; there is no point
gating on a signal that doesn't separate toxic fills from benign ones.

### Why (grounded in research)
Phase 11 established the loss is adverse selection, not fees: filled
passive quotes have mean realized edge ~ −$4.87 at 5s; markout is
negative at every distance. The literature (Market Maker's Dilemma,
arXiv 2502.18625; Fodra–Labadie 1511.04116) says the fix is not a
better P(fill) estimate but a *directional* signal — order-flow
imbalance (OFI) and the microprice are the standard, cheap, L2-only
candidates. Avellaneda–Stoikov does NOT supply this: its optimal
spread has no adverse-selection term (mid is an independent Brownian
motion), so pure AS cannot help. The only transferable AS piece is
inventory skew, which we already have in InventorySkew.

### Tasks
- src/features/order_flow.py — OFI from consecutive L1/L2 updates
  (Cont–Kukanov–Stoikov definition) and the Stoikov microprice
  (imbalance-weighted mid). Both computed from the book the backtest
  engine already maintains. Type hints + docstrings, no magic numbers.
- src/models/alpha_model.py — fit signal(s) → signed forward mid
  return over a lookahead horizon (Ridge, reuse edge_model plumbing).
  Distinct from edge_model: this is unconditional forward-return
  prediction, not conditional-on-fill edge.
- scripts/train_alpha.py — CLI: build features from a depth parquet,
  fit, report information coefficient (IC) at 1s/5s horizons.
- Offline validation harness: markout of hypothetical at-touch fills
  bucketed by signal decile — must show toxic vs benign separation.
- tests/test_order_flow.py, tests/test_alpha_model.py.

### Done when
| Item | Must be true |
|------|-------------|
| OFI + microprice | computed, unit-tested, sane on real book |
| Alpha model | trains, positive out-of-sample IC at 1s and/or 5s |
| Separation | markout of "signal-agrees" fills materially less negative than "signal-disagrees" |
| Kill check | if no separation, STOP and record the negative result — do not build Phase 13 |

---

## Phase 13 — Alpha-gated / one-sided quoting

### Goal
Fold the validated signal into the reservation price and *gate*
quoting: skip or go one-sided when expected edge (spread capture ±
alpha) does not beat estimated adverse-selection cost. Measure against
Naive and EV under honest trade-driven fills at fee 0 and fee 10 bps.
Honest numbers either way — the research finds NO published maker that
is reliably net-positive after realistic crypto costs, so the target
is "markout materially improved and loss closed or eliminated," not a
guaranteed profit.

### Tasks
- Extend EVMaker (or a new AlphaMaker sharing its scaffold) to:
  fair = mid + c1·alpha; reservation = InventorySkew(fair, inv);
  gate = quote a side only if edge_side > adverse_cost_est, else
  widen/skip that side (one-sided quoting).
- Requote policy: hold queue position on noise (extend the existing
  continuity clip), only cancel/reprice when the signal or mid moves
  beyond a threshold — cancel/repost churn is itself adverse-selection
  cost (Law–Viens 1903.07222).
- scripts/run_backtest.py: add the new strategy to the comparison
  table; sweep fee 0 and fee 10 bps.
- tests/test_alpha_maker.py: gating logic, one-sided output, no
  through-the-touch quoting, inventory bounds.

### Done when
| Item | Must be true |
|------|-------------|
| Gating | verified: quotes suppressed/one-sided when signal is adverse |
| Markout | AlphaMaker markout materially better than EV=P·h |
| Fee 0 | AlphaMaker beats or matches Naive |
| Fee 10 bps | result reported honestly (pass or documented structural loss) |
| No regressions | lint + mypy clean, full suite green |

---

## Phase 15 — Resume-hardening: benchmarks, scale, CI/CD, deployment

### Goal
Make the project's headline capability claims true and reproducible
(engineering/scale/signal — independent of strategy PnL), each backed by
a script and a doc. Honest wording where a claim can't be fully met.

### Tasks
- Engine benchmark (throughput + p50/p99 latency), C++ vs Python;
  docs/benchmarks.md. Redis kept off the measured hot path.
- Hit-ratio-lift experiment on imbalance+volatility with distance
  controlled out, time-ordered split; docs/hit_ratio.md.
- Bounded-memory shard streaming for TB-scale replay; synthetic L2
  generator; docs/data_scale.md.
- CI job: generate data -> backtest -> P&L report artifact (nightly +
  per-PR); pnl_report.py.
- 24/7 paper deployment: Dockerfile + compose overlay (auto-restart) +
  docs/live_deployment.md.
- docs/resume_claims.md mapping each claim to its evidence + honest
  wording.

### Done when
| Item | Must be true |
|------|-------------|
| Benchmark | measured >150k ev/s and latency reported, reproducible |
| Hit ratio | imbalance+volatility lift >= 35% on held-out split |
| Scale | streaming iterator + engine runs on multi-shard input |
| CI/CD | workflow runs a backtest and uploads a P&L artifact |
| 24/7 | containerized auto-restart paper loop documented |
| Honesty | overstated claims reworded; no PnL/live-money overclaim |
| Quality | lint + mypy clean, full suite green |

---

## Phase 14 — Paper acceptance and README

### Goal
The live paper loop demonstrably works: real Binance data in,
realistic simulated fills out, dashboard shows activity. Repo is
presentable.

### Tasks
- make feed + MM_FEE_BPS=10 make paper for 1 hour
- Acceptance: fills occur, inventory bounded, PnL sane vs backtest,
  no errors, Redis keys updating
- README.md: architecture, run guide, results, known limitations
- Fix stale CLI flags in ARCHITECTURE.md / spec.md
- Update progress.md; final commits (with approval)

### Done when
| Item | Must be true |
|------|-------------|
| 1-hour paper run | fills > 0, no errors, inventory within bounds |
| Dashboard | watch_paper.py / Grafana show live activity |
| README | exists, honest, covers record→train→backtest→paper |
| Docs | ARCHITECTURE.md/spec.md match the real CLI |
| progress.md | updated with Phase 10–14 outcomes |