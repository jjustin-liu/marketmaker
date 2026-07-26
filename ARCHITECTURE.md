# Architecture

## What this document is
The technical blueprint for the market maker.
Every implementation decision should reference this document.
Terms are defined in plain English in SPEC.md.

---

## Process model

The system runs as three separate processes.
They do not share memory. Redis is the only way they communicate.

Process 1 — data_feed
  Connects to Binance WebSocket
  Maintains local order book via LOB engine
  Writes current state to Redis after every update
  Never reads strategy output
  Never touches Binance order API

Process 2 — strategy
  Reads LOB state and trade history from Redis
  Computes feature vector
  Runs fill probability model
  Computes EV and outputs quote targets to Redis
  Never connects to Binance directly

Process 3 — live (or backtest)
  Live mode:    reads targets from Redis, sends orders to Binance
  Backtest mode: replays historical data, simulates fills locally
  Writes fills and position updates back to Redis

Why this separation matters:
  Feed crashes → strategy waits for fresh data, does not crash
  Strategy crashes → open orders stay on Binance, live handles cancel
  Each process can be tested independently with mocked Redis state

---

## Redis key schema

All keys use colon-separated namespacing.
Values are JSON strings unless marked as float or string.

LOB state — written by data_feed, read by strategy
  lob:best_bid          float    current best bid price
  lob:best_ask          float    current best ask price
  lob:mid               float    (best_bid + best_ask) / 2
  lob:spread_bps        float    spread in basis points
  lob:levels:bids       JSON     list of [price, qty] top 20 bids
                                 sorted descending
  lob:levels:asks       JSON     list of [price, qty] top 20 asks
                                 sorted ascending
  lob:last_update       float    unix timestamp of last update

Trade history — written by data_feed, read by strategy
  trades:recent         JSON     list of last 100 trades
                                 each: {price, qty, side, timestamp}
                                 side: "buy" or "sell" = aggressor side

Strategy output — written by strategy, read by live
  strategy:target_bid   float    target bid price
  strategy:target_ask   float    target ask price
  strategy:bid_qty      float    target bid quantity in BTC
  strategy:ask_qty      float    target ask quantity in BTC
  strategy:last_update  float    unix timestamp

Position state — written by live, read by strategy
  position:inventory    float    current BTC held (positive=long)
  position:pnl          float    running realized PnL in USDT
  position:open_bid_id  string   Binance order ID of open bid
  position:open_ask_id  string   Binance order ID of open ask

---

## LOB engine (src/lob/)

### Two implementations
match_engine.cpp — C++ implementation for live trading
  Microsecond performance
  Used when processing live Binance stream
  Compiled and wrapped with pybind11
  Imported as: from src.lob import MatchEngine, Side

order_book.py — Python implementation for backtesting
  Slower but readable and easily testable
  Used in src/backtest/ and unit tests
  Same interface as the C++ version

### C++ data structures
BidBook: std::map<double, PriceLevel, std::greater<double>>
  Sorted descending — best bid (highest price) always at index 0

AskBook: std::map<double, PriceLevel, std::less<double>>
  Sorted ascending — best ask (lowest price) always at index 0

PriceLevel: vector of shared_ptr<Order>
  Orders at each price in arrival order (queue priority)

order_map: std::map<string, pair<Side, double>>
  order_id → (side, price) for O(1) cancel lookup

### Operations exposed to Python via pybind11
insert(order_id, side, price, size, timestamp) → list of Fill
  Tries to match against opposite side first
  Any unfilled remainder rests in the book
  Returns list of Fill objects for any matches

cancel(order_id) → bool
  Removes order from book
  Cleans up empty price levels
  Returns true if found and cancelled

Fill object fields:
  taker_order_id   string   the aggressive order
  maker_order_id   string   the passive order that was resting
  price            float    fill price
  size             float    fill quantity in BTC
  timestamp        int64    unix timestamp in ms

Side enum:
  Side.BUY
  Side.SELL

### Binance WebSocket diff format
Each message contains:
  U: first update ID in this event
  u: last update ID in this event
  b: list of [price, qty] bid updates
  a: list of [price, qty] ask updates
  qty = "0" means delete that price level entirely

### Bootstrap sequence — must follow exactly
1. Open WebSocket and start buffering all incoming messages
2. Fetch REST snapshot:
   GET /api/v3/depth?symbol=BTCUSDT&limit=1000
   Note the lastUpdateId field from the response
3. Discard any buffered messages where u <= lastUpdateId
4. Find first buffered message where U <= lastUpdateId+1 <= u
   Apply it
5. Apply all subsequent messages in strict sequence order
6. If gap detected (U != previous_u + 1): restart from step 1

If you skip this and connect cold your local book
will silently diverge from reality.

---

## Feature vector (src/features/)

12 features computed on every LOB update.
Passed as a numpy array to the fill probability model.

### Imbalance features (src/features/imbalance.py)

Formula for all three:
  imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
  Range: -1.0 to +1.0
  +1 = all volume on bid side (strong buy pressure)
   0 = perfectly balanced
  -1 = all volume on ask side (strong sell pressure)

imbalance_1
  Computed using only the best price level on each side
  Most reactive, most noisy
  Strong signal for immediate next-tick direction

imbalance_2
  Computed using top 2 levels on each side
  Slightly broader, slightly smoother

imbalance_5
  Computed using top 5 levels on each side
  Broader market pressure signal
  Slower to react, less noisy

### Microprice (src/features/micro_price.py)
  Formula:
    microprice = (best_bid × best_ask_volume
                + best_ask × best_bid_volume)
                / (best_bid_volume + best_ask_volume)

  Why better than simple mid:
    Simple mid = (best_bid + best_ask) / 2  — ignores volume
    Microprice pulls toward the thinner side of the book
    If 10 BTC on bid, 1 BTC on ask:
      microprice pulls toward ask — thin ask likely to get lifted
    Better estimate of where the next trade will actually happen

### Volatility (src/features/volatility.py)
  Rolling window of last 100 mid price updates (deque, maxlen=100)
  Computes returns: ret_i = (price_i - price_{i-1}) / price_{i-1}
  Volatility = std(returns) × sqrt(252)  — annualized
  High volatility → widen quotes to cover adverse selection risk

### Order book features (src/models/fill_prob.py)
  bid_ask_spread    (best_ask - best_bid) / mid_price
  mid_price         absolute mid price in USDT
  bid_volume        sum of all qty across all bid levels
  ask_volume        sum of all qty across all ask levels

### Order features (computed per candidate quote)
  price_distance    abs(order_price - mid_price) / mid_price
  size              order size in BTC
  side              1.0 for buy, 0.0 for sell

---

## Fill probability model (src/models/fill_prob.py)

### What it answers
Given current market conditions and a candidate order price,
what is the probability this order gets filled?

### Model type
sklearn.linear_model.LogisticRegression
Features normalized with sklearn.preprocessing.StandardScaler
Saved and loaded together via joblib
Path: data/models/fill_prob.joblib

### Feature vector — 10 inputs in this order
[bid_ask_spread, mid_price, bid_volume, ask_volume,
 imbalance_1, imbalance_2, imbalance_5,
 price_distance, size, side]

### Training data construction
Source: historical fill data from backtest runs
For each real fill → 1 positive example (label = 1)
For each real fill → 5 synthetic negative examples (label = 0)
  Negative examples use same market state, worse price:
    Buy order: price = mid - spread × uniform(0.5, 1.5)
    Sell order: price = mid + spread × uniform(0.5, 1.5)
This handles class imbalance —
most limit orders in reality do not fill

### Training pipeline
1. Load fills dataframe
2. Extract 10-element feature vector per row
3. Generate 5 negatives per positive
4. train_test_split 80/20, random_state=42
5. StandardScaler.fit_transform on train set
6. StandardScaler.transform on test set (same scale, no leakage)
7. LogisticRegression.fit on scaled train set
8. Evaluate: roc_auc_score on test set
9. Log AUC score
10. joblib.dump({model, scaler}) to data/models/fill_prob.joblib

### Inference
Input:  bids, asks, order_price, order_size, order_side
Output: float probability between 0.0 and 1.0
Speed:  sub-millisecond (dot product + sigmoid)
Called: once per candidate offset per quote refresh cycle

---

## Inventory skew (src/strategy/inventory_skew.py)

### Config (InventorySkewConfig)
max_position:     1.0    BTC  — clips inventory to [-1, 1]
skew_factor:      0.5         — aggressiveness of centre shift
min_spread_bps:   1.0    bps  — minimum spread at neutral inventory
spread_factor:    1.0         — spread widening rate with inventory
continuity_clip:  0.1         — max quote move per refresh call
float_tolerance:  1e-10       — floating point comparison tolerance

### Logic — applied in this exact order

Step 1 — normalize inventory
  inv = clip(inventory / max_position, -1.0, 1.0)
  Prevents extreme positions from exploding quote prices

Step 2 — compute spread (widens with inventory magnitude)
  min_spread = mid_price × min_spread_bps / 10000
  spread     = min_spread × (1.0 + abs(inv) × spread_factor)
  half_spread = spread / 2
  Further from flat = wider spread on both sides

Step 3 — compute centre shift (directional)
  centre_shift = -inv × skew_factor × spread
  Long  (+inv): centre_shift is negative → quotes shift down
                makes selling easier, buying harder
  Short (-inv): centre_shift is positive → quotes shift up
                makes buying easier, selling harder

Step 4 — raw bid and ask
  bid_price = mid + centre_shift - half_spread
  ask_price = mid + centre_shift + half_spread
  ask > bid always guaranteed by half_spread symmetry

Step 5 — continuity clip (applied if previous quotes exist)
  bid_move = new_bid - prev_bid
  ask_move = new_ask - prev_ask
  bid_price = prev_bid + clip(bid_move, -clip_limit, +clip_limit)
  ask_price = prev_ask + clip(ask_move, -clip_limit, +clip_limit)
  Prevents violent jumps that lose queue position

---

## Size calculator (src/strategy/size_calculator.py)

### Config (SizeConfig)
base_size:          0.001 BTC  — size at neutral inventory
max_size_mult:      3.0        — max multiplier at extreme inventory
max_position:       1.0 BTC    — inventory that triggers max scaling
scaling_type:       LINEAR or SIGMOID
sigmoid_steepness:  4.0        — steepness of sigmoid curve
min_size_mult:      0.1        — never fully zero except at maximum

### Logic

Step 1 — normalize inventory
  norm_inv = clip(inventory / max_position, -1.0, 1.0)

Step 2 — apply scaling function
  LINEAR:  scaled_inv = norm_inv
  SIGMOID: scaled_inv = 2 × (1 / (1 + e^(-inv × steepness))) - 1
    Sigmoid gives gentler adjustments at moderate inventory
    Falls back to linear at extreme positions (abs >= 1)
    Prevents overreacting to small inventory imbalances

Step 3 — compute multipliers
  base_mult = (max_size_mult - 1.0) / 2.0

  Long (+scaled_inv):
    bid_mult = max(min_size_mult, 1.0 - scaled_inv × base_mult × 2)
    ask_mult = 1.0 + scaled_inv × base_mult × 2
    At maximum long: bid_mult = 0 (stop posting bids entirely)

  Short (-scaled_inv):
    bid_mult = 1.0 - scaled_inv × base_mult × 2
    ask_mult = max(min_size_mult, 1.0 + scaled_inv × base_mult × 2)
    At maximum short: ask_mult = 0 (stop posting asks entirely)

Step 4 — apply to base size
  bid_size = base_size × bid_mult
  ask_size = base_size × ask_mult

---

## EV strategy (src/strategy/ev_maker.py)

### Config (EVConfig)
min_spread:   0.0005  — minimum spread enforced on final quotes
max_spread:   0.005   — upper bound of spread search space
num_points:   10      — number of candidate spreads to evaluate

### Two strategies
NaiveMaker — fixed spread, no EV optimization
  Posts at mid ± half_spread
  Tightens inside current market spread if best_bid/ask provided
  Used as benchmark in backtest

EVMaker — full EV optimization
  The real strategy, described below

### EVMaker quote generation — step by step

Step 1 — get inventory-adjusted base quotes
  base_bid, base_ask = inventory_skew.apply_skew(mid, inventory)
  This is the centre point accounting for position risk

Step 2 — generate 10 candidate spreads
  spreads = evenly spaced from 0 to max_spread (10 points)
  bid candidates: price = base_bid - spread
  ask candidates: price = base_ask + spread

Step 3 — compute EV at each candidate
  For each candidate price:
    if fill_model available:
      fill_prob = fill_model.predict(bids, asks, price, size, side)
    else:
      fill_prob = passed-in bid_probability or ask_probability

    ev = fill_prob × spread

  Intuition: gross profit if filled = spread distance
             expected profit = gross profit × probability of filling

Step 4 — select best price
  best_bid_price = candidate with highest ev on bid side
  best_ask_price = candidate with highest ev on ask side

Step 5 — enforce minimum spread
  if (best_ask - best_bid) < min_spread:
    centre = (best_ask + best_bid) / 2
    best_bid = centre - min_spread / 2
    best_ask = centre + min_spread / 2

Step 6 — get sizes from size calculator
  bid_size, ask_size = size_calculator.get_sizes(inventory)

Step 7 — assert bounds (safety check)
  bid must not have moved more than max_spread from base_bid
  ask must not have moved more than max_spread from base_ask
  Raises AssertionError if violated

### Output
Quote(price=best_bid_price, size=bid_size)
Quote(price=best_ask_price, size=ask_size)

---

## Backtest engine (src/backtest/)

### Data format
Historical L2 order book snapshots
Stored as parquet files in data/
Each row: timestamp, side, price, qty (one diff per row)

### Replay loop
For each historical diff in chronological order:
  Apply diff to local OrderBook (Python implementation)
  Compute feature vector from current book state
  Run fill_model inference
  Run EVMaker to get quote targets
  Simulate fills:
    Bid fills when ask price trades through our bid level
    Ask fills when bid price trades through our ask level
    Queue position: assumed worst case (last in queue)
    Conservative — real performance likely better
  Update simulated inventory and PnL

### Comparison
Runs both NaiveMaker and EVMaker on same historical data
Hit ratio improvement = (EV fill rate - Naive fill rate) / Naive fill rate
Target: 35% improvement — the resume bullet point

### Output metrics
Total PnL in USDT
Sharpe ratio (annualized)
Fill rate (fills per hour)
Adverse selection cost
  Average PnL in first 5 seconds after each fill
  Negative = getting picked off by informed traders
Max drawdown
Inventory path over time (chart)

---

## Live engine (src/live/)

### Order lifecycle
PENDING   → order sent to Binance, awaiting acknowledgment
OPEN      → acknowledged, resting in Binance book
FILLED    → fully filled
CANCELLED → cancelled by us or expired

### Quote refresh logic
Read strategy:target_bid and strategy:target_ask from Redis
Compare to current open order prices
If difference > 1 bps: amend order
If difference <= 1 bps: do nothing (preserve queue position)
Amending costs: API latency + potential queue position loss

### On fill
Update position:inventory in Redis
Update position:pnl in Redis
Log fill to data/fills.log

### On disconnect from Binance
Cancel all open orders immediately
Do not re-enter market until feed is confirmed live
Never leave orphaned orders on the exchange

### Paper trading mode
Full pipeline runs as normal
Instead of sending API calls, simulates fills locally
Verify end-to-end for minimum 1 hour before going live

---

## Entry point (src/cli.py)

python -m src.cli backtest --start-date 2024-01-01 --symbol BTCUSDT
python -m src.cli live --api-key KEY --api-secret SECRET

Routes to src/backtest/ or src/live/
Handles: config loading, logging setup, graceful shutdown

---

## Monitoring (docker/)

Docker Compose stack:
  Prometheus — scrapes metrics from running system
  Grafana    — visualizes metrics as dashboards

Key metrics on dashboard:
  Running PnL
  Fill rate per hour
  Current inventory
  Feed latency
  Quote refresh rate
  Adverse selection cost (rolling)

Run with: docker-compose up