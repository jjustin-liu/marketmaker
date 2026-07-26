# Market Maker — Project Spec

## What this is
A market making system for Binance BTCUSDT.
Posts bid and ask limit orders simultaneously, capturing the spread
between them. Makes money by buying slightly below mid price and
selling slightly above it, repeatedly, using expected value to find
the best offset rather than a fixed spread.

## How it makes money
Edge comes from the spread — the difference between bid and ask.
For every candidate quote offset δ from mid price, the system
computes EV = P(fill | δ, features) × edge − adverse_selection_cost
and posts at the offset where EV is highest.

## The two risks it manages

### Adverse selection
Getting filled by an informed trader who knows the price is about
to move against you. Managed by modeling fill probability using
market microstructure features — when conditions suggest informed
flow, quotes widen or pull back.

### Inventory risk
Accumulating too much BTC on one side, creating directional
exposure. Managed by inventory skew — if long, the ask tightens
and the bid widens to nudge the position back toward flat.

## Data flow
Binance WebSocket
  → src/data_feed
    Ingests live order book and trade stream
    Writes current state to Redis after every update

  → src/lob
    C++ limit order book engine (pybind11 bindings)
    Maintains accurate local copy of Binance order book
    Operations: apply_diff, best_bid, best_ask, depth

  → src/features
    Reads LOB state from Redis
    Computes: imbalance, spread, mid drift,
              trade flow toxicity, queue depth

  → src/models
    Fill probability model trained on historical LOB data
    Input: feature vector + candidate offset δ
    Output: P(fill) between 0 and 1

  → src/strategy
    Computes EV for each candidate offset
    Applies inventory skew
    Outputs: target bid price, target ask price, quantities

  → src/live        (live mode)
    Sends real limit orders to Binance
    Manages order lifecycle: place, amend, cancel
    Tracks fills and updates inventory in Redis

  → src/backtest    (backtest mode)
    Replays historical LOB data through full pipeline
    Simulates fills accounting for queue position
    Reports: PnL, Sharpe ratio, fill rate, adverse selection cost

## Entry point
src/cli.py routes to either backtest or live mode:
  python -m src.cli backtest --start-date 2024-01-01
  python -m src.cli live --api-key <key> --api-secret <secret>

## Shared state
Redis acts as the state bus between processes:
  lob:best_bid          current best bid price
  lob:best_ask          current best ask price
  lob:levels:bids       top 20 bid levels [price, qty]
  lob:levels:asks       top 20 ask levels [price, qty]
  trades:recent         last 100 trades (ringbuffer)
  strategy:target_bid   current target bid from strategy
  strategy:target_ask   current target ask from strategy
  position:inventory    current BTC position
  position:pnl          running PnL

## What done looks like
Phase by phase — see phases.md for full criteria.
Overall done when:
- Backtest runs on 1 week of historical data and reports
  PnL, Sharpe, fill rate, and adverse selection cost
- Paper trading runs end-to-end for 1 hour without errors
- Every module has unit tests that pass
- make lint and make test both pass clean