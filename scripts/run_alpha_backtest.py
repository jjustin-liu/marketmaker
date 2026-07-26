"""Phase-13 comparison: Naive vs EV vs AlphaMaker at fee 0 and 10 bps.

Runs each strategy through the same honest trade-driven backtest and
prints one table per fee level. AlphaMaker variants sweep the gate
threshold. Honest by construction: the same tape, fills, and markout
horizon as the Phase-11 numbers.

Usage:
  python -m scripts.run_alpha_backtest \\
      --input data/raw/btcusdt_depth_2026-07-08.parquet \\
      --alpha-model data/models/alpha_lh50.joblib \\
      --fill-model data/models/fill_prob_lh1000.joblib \\
      --max-diffs 1500000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from src.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    load_diffs_from_parquet,
    load_trades_from_parquet,
)
from src.models.alpha_model import AlphaModel
from src.strategy.alpha_maker import AlphaConfig, AlphaMaker
from src.strategy.ev_maker import EVMaker
from src.strategy.inventory_skew import InventorySkew
from src.strategy.naive_maker import NaiveMaker
from src.strategy.size_calculator import SizeCalculator

logger = logging.getLogger("run_alpha_backtest")

GATE_SWEEP_BPS = (0.0, 0.1, 0.3)


def _run(
    strategy, use_inventory: bool, fee_bps: float, diffs, trades,
) -> BacktestResult:
    engine = BacktestEngine(
        strategy=strategy,
        use_inventory=use_inventory,
        fee_bps=fee_bps,
        fill_mode="trades",
    )
    return engine.run(diffs, trades=trades)


def _row(name: str, r: BacktestResult) -> str:
    return (
        f"{name:<26}{r.pnl:>10.2f}{r.sharpe:>11.1f}"
        f"{r.adverse_selection:>11.3f}{r.num_fills:>8}{r.num_quotes:>9}"
    )


def _print_table(fee_bps: float, results: List[Tuple[str, BacktestResult]]) -> None:
    hdr = (f"{'Strategy':<26}{'PnL':>10}{'Sharpe':>11}"
           f"{'Markout':>11}{'Fills':>8}{'Quotes':>9}")
    print(f"\n=== fee {fee_bps:.0f} bps — {len(results)} strategies ===")
    print(hdr)
    print("-" * len(hdr))
    for name, r in results:
        print(_row(name, r))


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive vs EV vs AlphaMaker.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--alpha-model", type=Path,
                        default=Path("data/models/alpha_lh50.joblib"))
    parser.add_argument("--fill-model", type=Path,
                        default=Path("data/models/fill_prob_lh1000.joblib"))
    parser.add_argument("--max-diffs", type=int, default=1_500_000)
    parser.add_argument("--fees", type=float, nargs="+", default=[0.0, 10.0])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    diffs = load_diffs_from_parquet(args.input)[:args.max_diffs]
    trades = load_trades_from_parquet(
        args.input.with_name(args.input.name.replace("_depth_", "_trades_"))
    )
    logger.info("%d diffs, %d trades", len(diffs), len(trades))

    alpha_model = AlphaModel.load(args.alpha_model)
    logger.info("alpha model IC %.4f", alpha_model.ic or float("nan"))

    fill_model: Optional[object] = None
    if args.fill_model.exists():
        from src.models.fill_prob import FillProbabilityModel
        fill_model = FillProbabilityModel.load(args.fill_model)
        logger.info("fill model AUC %.3f", fill_model.auc)

    for fee in args.fees:
        results: List[Tuple[str, BacktestResult]] = []
        results.append(("Naive", _run(NaiveMaker(), False, fee, diffs, trades)))
        results.append(("EV (P*h)", _run(
            EVMaker(InventorySkew(), SizeCalculator(), fill_model=fill_model),
            True, fee, diffs, trades)))
        for gate in GATE_SWEEP_BPS:
            cfg = AlphaConfig(gate_threshold_bps=gate)
            results.append((f"Alpha gate={gate:g}bps", _run(
                AlphaMaker(alpha_model, cfg), True, fee, diffs, trades)))
        _print_table(fee, results)


if __name__ == "__main__":
    main()
