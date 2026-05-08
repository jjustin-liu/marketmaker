"""Train the fill probability model from a fills parquet file.

Usage:
  python -m scripts.train_model [--input PATH] [--output PATH]

Defaults:
  input  = data/fills.parquet
  output = data/models/fill_prob.joblib
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.models.fill_prob import DEFAULT_MODEL_PATH, FillProbabilityModel

DEFAULT_INPUT = Path("data/fills.parquet")

logger = logging.getLogger("train_model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fill probability model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.input.exists():
        raise SystemExit(f"input file not found: {args.input}")

    fills = pd.read_parquet(args.input)
    logger.info("loaded %d fills from %s", len(fills), args.input)

    model = FillProbabilityModel()
    auc = model.train(fills)
    logger.info("test AUC = %.4f", auc)

    out = model.save(args.output)
    logger.info("saved model to %s", out)


if __name__ == "__main__":
    main()
