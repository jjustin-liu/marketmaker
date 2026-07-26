"""Generate synthetic but replayable BTCUSDT L2 depth + trades parquets.

Deterministic (seeded) data for CI backtests and for demonstrating that
the pipeline scales to arbitrary data volume. The generator maintains
its own book model and only emits diffs consistent with it, so the
output always replays uncrossed and with valid epochs — the same
contract as recorded data.

Output columns:
  depth:  timestamp, side, price, qty, is_snapshot
  trades: timestamp, side, price, qty

Usage:
  python -m scripts.gen_synthetic_l2 --rows 200000 \\
      --out data/synthetic/depth.parquet --trades data/synthetic/trades.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("gen_synthetic_l2")

TICK = 0.1
MID0 = 60_000.0
LEVELS = 10
BASE_QTY = 2.0


def generate(
    rows: int,
    seed: int,
    epochs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (depth_df, trades_df) with ~`rows` depth rows across `epochs`."""
    rng = np.random.default_rng(seed)
    depth: List[dict] = []
    trades: List[dict] = []
    bids: Dict[float, float] = {}
    asks: Dict[float, float] = {}
    state = {"ts": 1_600_000_000_000, "mid": MID0}
    snap_every = max(1, rows // max(1, epochs))

    def _q() -> float:
        return float(BASE_QTY * (1.0 + rng.random()))

    def snapshot() -> None:
        bids.clear()
        asks.clear()
        best_bid = round(state["mid"] - TICK / 2, 1)
        best_ask = round(state["mid"] + TICK / 2, 1)
        for i in range(LEVELS):
            pb = round(best_bid - i * TICK, 1)
            pa = round(best_ask + i * TICK, 1)
            bids[pb] = _q()
            asks[pa] = _q()
            for side, price, qty in (("buy", pb, bids[pb]),
                                     ("sell", pa, asks[pa])):
                depth.append({"timestamp": state["ts"], "side": side,
                              "price": price, "qty": qty, "is_snapshot": True})
                state["ts"] += 1

    def emit(side: str, price: float, qty: float) -> None:
        book = bids if side == "buy" else asks
        if qty <= 0.0:
            book.pop(price, None)
        else:
            book[price] = qty
        depth.append({"timestamp": state["ts"], "side": side,
                      "price": round(price, 1), "qty": float(qty),
                      "is_snapshot": False})
        state["ts"] += 1

    snapshot()
    next_snap = snap_every
    while len(depth) < rows:
        best_bid, best_ask = max(bids), min(asks)
        r = rng.random()
        if r < 0.35:                                   # refresh a level's qty
            side = "buy" if rng.random() < 0.5 else "sell"
            book = bids if side == "buy" else asks
            emit(side, float(rng.choice(list(book))), _q())
        elif r < 0.60 and len(asks) > 2:               # touch up: drop best ask
            emit("sell", best_ask, 0.0)
            emit("sell", round(max(asks) + TICK, 1), _q())
        elif r < 0.85 and len(bids) > 2:               # touch down: drop best bid
            emit("buy", best_bid, 0.0)
            emit("buy", round(min(bids) - TICK, 1), _q())
        else:                                          # trade print at the touch
            if rng.random() < 0.5:
                trades.append({"timestamp": state["ts"], "side": "sell",
                               "price": best_bid, "qty": float(0.5 * rng.random()
                                                               + 0.01)})
            else:
                trades.append({"timestamp": state["ts"], "side": "buy",
                               "price": best_ask, "qty": float(0.5 * rng.random()
                                                               + 0.01)})
            state["ts"] += 1

        if len(depth) >= next_snap:                    # epoch reseed + mid drift
            state["mid"] = max(1000.0, state["mid"] + float(rng.normal(0, TICK * 3)))
            snapshot()
            next_snap += snap_every

    return pd.DataFrame(depth), pd.DataFrame(trades)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic L2 generator.")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path,
                        default=Path("data/synthetic/depth.parquet"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/synthetic/trades.parquet"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    depth_df, trades_df = generate(args.rows, args.seed, args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.trades.parent.mkdir(parents=True, exist_ok=True)
    depth_df.to_parquet(args.out, index=False)
    trades_df.to_parquet(args.trades, index=False)
    logger.info("wrote %d depth rows -> %s", len(depth_df), args.out)
    logger.info("wrote %d trades   -> %s", len(trades_df), args.trades)


if __name__ == "__main__":
    main()
