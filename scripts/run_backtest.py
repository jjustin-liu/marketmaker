"""Run a backtest comparing NaiveMaker and EVMaker on historical L2 data.

Usage:
  python -m scripts.run_backtest --input data/ticks.parquet [--out FILE]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestEngine, load_diffs_from_parquet
from src.strategy.ev_maker import EVMaker
from src.strategy.inventory_skew import InventorySkew
from src.strategy.naive_maker import NaiveMaker
from src.strategy.size_calculator import SizeCalculator

logger = logging.getLogger("run_backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest comparison.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("data/backtest_results.csv"))
    parser.add_argument("--fee-bps", type=float, default=0.0,
                        help="signed maker fee in bps. Negative = rebate.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    diffs = load_diffs_from_parquet(args.input)
    logger.info("loaded %d diffs", len(diffs))

    naive_engine = BacktestEngine(
        strategy=NaiveMaker(), use_inventory=False, fee_bps=args.fee_bps,
    )
    naive = naive_engine.run(diffs)

    ev_engine = BacktestEngine(
        strategy=EVMaker(
            inventory_skew=InventorySkew(),
            size_calculator=SizeCalculator(),
        ),
        use_inventory=True,
        fee_bps=args.fee_bps,
    )
    ev = ev_engine.run(diffs)

    rows = [
        {"strategy": "naive", "pnl": naive.pnl, "sharpe": naive.sharpe,
         "hit_ratio": naive.hit_ratio, "adverse": naive.adverse_selection,
         "max_dd": naive.max_drawdown, "fills": naive.num_fills,
         "quotes": naive.num_quotes},
        {"strategy": "ev", "pnl": ev.pnl, "sharpe": ev.sharpe,
         "hit_ratio": ev.hit_ratio, "adverse": ev.adverse_selection,
         "max_dd": ev.max_drawdown, "fills": ev.num_fills,
         "quotes": ev.num_quotes},
    ]
    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    logger.info("\n%s", out_df.to_string(index=False))
    if naive.hit_ratio > 0:
        lift = (ev.hit_ratio - naive.hit_ratio) / naive.hit_ratio
        logger.info("EV hit-ratio lift over Naive: %.2f%%", lift * 100)


if __name__ == "__main__":
    main()
