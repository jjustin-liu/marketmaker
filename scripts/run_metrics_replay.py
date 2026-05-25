"""Replay a parquet of historical L2 diffs and publish live metrics.

Drives the same BacktestEngine that powers `make backtest`, but adds a
per-tick callback that pushes inventory / PnL / fills / quote refreshes
into the prometheus_client globals exposed by src.live.metrics. The
upshot: Grafana's "Market Maker" dashboard shows realistic numbers
moving in near-real-time, instead of the flat zeros the paper sim
produces against current behavior.

Usage:
  python -m scripts.run_metrics_replay --input data/raw/<file>.parquet \
      --speed 200 --port 8000

--speed is a replay multiplier: 1.0 = wall-clock, 200 = 200x realtime.
Implemented via a fixed sleep per diff calibrated off the file's
median inter-arrival time.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

import redis

from src.backtest.engine import BacktestEngine, load_diffs_from_parquet
from src.live.metrics import (
    DEFAULT_METRICS_PORT,
    install_feed_latency_collector,
    record_fill,
    record_quote_refresh,
    reset_for_tests,
    set_position,
    start_metrics_server,
)
from src.models.fill_prob import FillProbabilityModel
from src.strategy.ev_maker import EVMaker
from src.strategy.inventory_skew import InventorySkew
from src.strategy.naive_maker import NaiveMaker, NaiveMakerConfig
from src.strategy.size_calculator import SizeCalculator

logger = logging.getLogger("metrics_replay")


def _mean_interarrival_seconds(timestamps: list[int]) -> float:
    """Average gap per diff. Timestamps are millis.

    Uses total span / n_diffs rather than median-of-non-zero, because
    L2 updates come in tight bursts that all share a millisecond
    timestamp — median(diff > 0) over-counts the inter-burst gap and
    makes replay run far slower than intended.
    """
    if len(timestamps) < 2:
        return 0.05
    arr = np.array(timestamps, dtype="int64")
    span_ms = int(arr[-1]) - int(arr[0])
    if span_ms <= 0:
        return 0.05
    return (span_ms / 1000.0) / (len(timestamps) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="parquet file with timestamp, side, price, qty")
    parser.add_argument("--port", type=int, default=DEFAULT_METRICS_PORT)
    parser.add_argument("--speed", type=float, default=200.0,
                        help="replay multiplier; 1.0 = realtime")
    parser.add_argument("--refresh-every", type=int, default=10,
                        help="diffs between strategy quote refreshes")
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--model", type=Path,
                        default=Path("data/models/fill_prob.joblib"))
    parser.add_argument("--max-diffs", type=int, default=None,
                        help="cap on diffs replayed; useful for demos")
    parser.add_argument("--strategy", choices=["naive", "ev"], default="naive",
                        help="naive = tight quotes (visible fills); "
                             "ev = the real strategy (wide quotes, few fills "
                             "on short replays)")
    parser.add_argument("--naive-spread", type=float, default=2.0,
                        help="full spread in USD for the naive strategy")
    parser.add_argument("--naive-size", type=float, default=0.001,
                        help="quote size in BTC for the naive strategy")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("loading diffs from %s", args.input)
    diffs = load_diffs_from_parquet(args.input)
    if args.max_diffs is not None:
        diffs = diffs[: args.max_diffs]
    logger.info("loaded %d diffs", len(diffs))

    mean_dt = _mean_interarrival_seconds([d.timestamp for d in diffs])
    sleep_per_diff = mean_dt / max(args.speed, 1e-9)
    logger.info("mean inter-arrival=%.5fs, speed=%.1fx -> sleep %.6fs/diff",
                mean_dt, args.speed, sleep_per_diff)

    if args.strategy == "ev":
        fill_model = FillProbabilityModel.load(args.model)
        strategy: object = EVMaker(
            inventory_skew=InventorySkew(),
            size_calculator=SizeCalculator(),
            fill_model=fill_model,
        )
        use_inventory = True
    else:
        strategy = NaiveMaker(NaiveMakerConfig(
            spread=args.naive_spread, size=args.naive_size,
        ))
        use_inventory = False

    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    r.ping()
    install_feed_latency_collector(r)
    reset_for_tests()
    start_metrics_server(args.port)
    logger.info("metrics server on :%d/metrics", args.port)

    last_quote_count = 0
    last_latency_write = [0.0]

    def on_tick(engine: BacktestEngine, new_fill: bool) -> None:
        nonlocal last_quote_count
        set_position(engine.inventory, engine.pnl_history[-1])
        if engine.quote_count > last_quote_count:
            record_quote_refresh()
            last_quote_count = engine.quote_count
        if new_fill:
            record_fill(engine.fills[-1].side)
        # publish a synthetic lob:last_update so the feed-latency panel
        # shows ~0 instead of NaN. Throttle to once per 200ms.
        now = time.time()
        if now - last_latency_write[0] > 0.2:
            r.set("lob:last_update", now)
            last_latency_write[0] = now

    engine = BacktestEngine(
        strategy=strategy,  # type: ignore[arg-type]
        refresh_every=args.refresh_every,
        fee_bps=args.fee_bps,
        use_inventory=use_inventory,
        on_tick=on_tick,
        tick_sleep_seconds=sleep_per_diff,
    )

    start = time.time()
    result = engine.run(diffs)
    elapsed = time.time() - start
    logger.info(
        "replay done in %.1fs: pnl=%.4f fills=%d quotes=%d hit_ratio=%.4f",
        elapsed, result.pnl, result.num_fills, result.num_quotes,
        result.hit_ratio,
    )
    # keep the process alive a moment so Grafana can scrape the final state
    time.sleep(10.0)


if __name__ == "__main__":
    main()
