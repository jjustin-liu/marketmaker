"""Continuously retrain the fill-probability model from live parquets.

Loop:
  1. Find the most-recently-modified parquet in --input-dir.
  2. If it has grown by at least --min-new-rows since the last training
     run, retrain on it. (Skips noop work when the recorder hasn't
     flushed new data yet.)
  3. Atomic-swap data/models/fill_prob.joblib so the strategy can
     hot-reload without ever seeing a partially-written file.
  4. Sleep --interval-seconds.

Designed to run alongside scripts/run_recorder.py without coordination.
The recorder appends rows; this script consumes them whenever it wakes.

Usage:
  python -m scripts.continuous_train
  python -m scripts.continuous_train --input-dir data/raw/verified_candidates
                                     --interval-seconds 300
                                     --max-offset-bps 50
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from scripts.train_model_lookahead import (
    LOOKAHEAD_DEFAULT,
    MAX_OFFSET_BPS_DEFAULT,
    build_fills_dataset,
    train_from_labeled,
)
from src.models.fill_prob import DEFAULT_MODEL_PATH

DEFAULT_INPUT_DIR = Path("data/raw/verified_candidates")
DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_MIN_NEW_ROWS = 10_000
DEFAULT_MAX_TICKS = 200_000

logger = logging.getLogger("continuous_train")


def _latest_parquet(input_dir: Path) -> Optional[Path]:
    candidates = list(input_dir.glob("*.parquet"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _atomic_save_model(model: object, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(tmp)  # type: ignore[attr-defined]
    os.replace(tmp, dest)


def train_once(
    parquet: Path,
    lookahead: int,
    max_ticks: int,
    max_offset_bps: float,
    out_path: Path,
) -> float:
    df = build_fills_dataset(parquet, lookahead, max_ticks, max_offset_bps)
    model, auc = train_from_labeled(df)
    _atomic_save_model(model, out_path)
    return auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--interval-seconds", type=int,
                    default=DEFAULT_INTERVAL_SECONDS)
    ap.add_argument("--min-new-rows", type=int, default=DEFAULT_MIN_NEW_ROWS,
                    help="skip retrain if parquet grew by less than this")
    ap.add_argument("--lookahead", type=int, default=LOOKAHEAD_DEFAULT)
    ap.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS)
    ap.add_argument("--max-offset-bps", type=float,
                    default=MAX_OFFSET_BPS_DEFAULT)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    stop = False

    def _handle(*_: object) -> None:
        nonlocal stop
        logger.info("shutdown signal received")
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle)

    last_trained_path: Optional[Path] = None
    last_trained_rows: int = 0

    while not stop:
        latest = _latest_parquet(args.input_dir)
        if latest is None:
            logger.info("no parquet in %s; sleeping", args.input_dir)
            time.sleep(args.interval_seconds)
            continue

        try:
            current_rows = len(pd.read_parquet(latest, columns=["timestamp"]))
        except Exception as e:
            logger.warning("could not read %s: %s; sleeping", latest, e)
            time.sleep(args.interval_seconds)
            continue

        is_new_file = latest != last_trained_path
        grew_enough = current_rows - last_trained_rows >= args.min_new_rows

        if not is_new_file and not grew_enough:
            logger.info(
                "%s rows=%d unchanged enough (last=%d, need +%d); skipping",
                latest.name, current_rows, last_trained_rows,
                args.min_new_rows,
            )
            time.sleep(args.interval_seconds)
            continue

        logger.info("training on %s (rows=%d, was=%d)",
                    latest.name, current_rows, last_trained_rows)
        t0 = time.perf_counter()
        try:
            auc = train_once(
                latest, args.lookahead, args.max_ticks,
                args.max_offset_bps, args.out,
            )
        except Exception as e:
            logger.exception("training failed: %s", e)
            time.sleep(args.interval_seconds)
            continue

        elapsed = time.perf_counter() - t0
        logger.info(
            "saved %s  AUC=%.4f  trained_in=%.1fs", args.out, auc, elapsed,
        )
        last_trained_path = latest
        last_trained_rows = current_rows

        for _ in range(args.interval_seconds):
            if stop:
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
