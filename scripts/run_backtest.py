"""Run a backtest comparing NaiveMaker and EVMaker on historical L2 data.

Usage:
  python -m scripts.run_backtest --input data/raw/btcusdt_depth_2026-05-25.parquet
  python -m scripts.run_backtest --input data/raw/btcusdt_depth_2026-05-25.parquet \\
      --fee-bps -0.5 --model data/models/fill_prob.joblib
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    load_diffs_from_parquet,
    load_trades_from_parquet,
)
from src.strategy.ev_maker import EVMaker
from src.strategy.inventory_skew import InventorySkew
from src.strategy.naive_maker import NaiveMaker
from src.strategy.size_calculator import SizeCalculator

logger = logging.getLogger("run_backtest")

_COL_W = 16  # column width for the result table


def _pct_delta(a: float, b: float) -> str:
    """Return signed % change from a to b, or '--' if a is zero."""
    if abs(a) < 1e-12:
        return "--"
    delta = (b - a) / abs(a) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.0f}%"


def _fmt(val: float, decimals: int = 2, prefix: str = "") -> str:
    sign = "+" if val > 0 else ""
    return f"{prefix}{sign}{val:.{decimals}f}"


def print_table(
    path: Path,
    naive: BacktestResult,
    ev: BacktestResult,
    model_loaded: bool,
) -> None:
    w = _COL_W
    sep = "═" * (20 + w * 3)
    thin = "─" * (20 + w * 3)
    label_w = 20

    header = f"{'Metric':<{label_w}}{'NaiveMaker':>{w}}{'EVMaker':>{w}}{'Δ':>{w}}"

    rows = [
        ("PnL (USDT)", _fmt(naive.pnl), _fmt(ev.pnl),
         _pct_delta(naive.pnl, ev.pnl)),
        ("Sharpe (ann.)", _fmt(naive.sharpe), _fmt(ev.sharpe),
         _pct_delta(naive.sharpe, ev.sharpe)),
        ("Hit ratio", f"{naive.hit_ratio*100:.1f}%", f"{ev.hit_ratio*100:.1f}%",
         _pct_delta(naive.hit_ratio, ev.hit_ratio)),
        ("Markout ($/fill)",
         _fmt(naive.adverse_selection, 3),
         _fmt(ev.adverse_selection, 3),
         _pct_delta(naive.adverse_selection, ev.adverse_selection)),
        ("Max drawdown", _fmt(naive.max_drawdown), _fmt(ev.max_drawdown),
         _pct_delta(naive.max_drawdown, ev.max_drawdown)),
        ("Fills", str(naive.num_fills), str(ev.num_fills), ""),
        ("Quotes", str(naive.num_quotes), str(ev.num_quotes), ""),
    ]

    print(f"\nStrategy Comparison — {path.name}")
    if not model_loaded:
        print("  (fill model not found — EVMaker using uniform P=0.5)")
    print(sep)
    print(header)
    print(thin)
    for label, n_val, e_val, delta in rows:
        print(f"{label:<{label_w}}{n_val:>{w}}{e_val:>{w}}{delta:>{w}}")
    print(sep)

    if naive.hit_ratio > 0:
        lift = (ev.hit_ratio - naive.hit_ratio) / naive.hit_ratio * 100
        target_met = "✓" if lift >= 35 else "✗ (target ≥35%)"
        sign = "+" if lift >= 0 else ""
        print(f"EVMaker hit-ratio lift: {sign}{lift:.1f}% {target_met}")
    if ev.crossed_skips:
        print(f"Crossed-book ticks skipped: {ev.crossed_skips:,} "
              "(non-physical reconstruction states, skipped by both)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NaiveMaker vs EVMaker backtest.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to a depth parquet file.")
    parser.add_argument("--out", type=Path,
                        default=Path("data/backtest_results.csv"))
    parser.add_argument("--fee-bps", type=float, default=0.0,
                        help="Signed maker fee in bps (negative = rebate).")
    parser.add_argument("--model", type=Path,
                        default=Path("data/models/fill_prob.joblib"),
                        help="Path to fill_prob.joblib for EVMaker.")
    parser.add_argument("--max-diffs", type=int, default=None,
                        help="Cap diffs replayed (fast iteration on big files).")
    parser.add_argument("--trades", type=Path, default=None,
                        help="Trades parquet. Default: sibling _trades_ file "
                             "if present. Enables trade-driven fills.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.input.exists():
        logger.error("input file not found: %s", args.input)
        raise SystemExit(1)

    logger.info("Loading diffs from %s ...", args.input)
    diffs = load_diffs_from_parquet(args.input)
    if args.max_diffs is not None:
        diffs = diffs[:args.max_diffs]
        logger.info("  capped to first %d diffs", len(diffs))
    else:
        logger.info("  %d diffs loaded", len(diffs))

    # Trade-driven fills if a trades file is given or sits beside the input.
    trades_path = args.trades
    if trades_path is None:
        sibling = args.input.with_name(
            args.input.name.replace("_depth_", "_trades_"))
        if "_depth_" in args.input.name and sibling.exists():
            trades_path = sibling
    trades = None
    fill_mode = "queue"
    if trades_path is not None and trades_path.exists():
        trades = load_trades_from_parquet(trades_path)
        fill_mode = "trades"
        logger.info("  %d trades loaded from %s → trade-driven fills",
                    len(trades), trades_path.name)
    else:
        logger.info("  no trades file → queue-mode (depth-only) fills")

    # NaiveMaker
    logger.info("Running NaiveMaker ...")
    naive_engine = BacktestEngine(
        strategy=NaiveMaker(),
        use_inventory=False,
        fee_bps=args.fee_bps,
        fill_mode=fill_mode,
    )
    naive = naive_engine.run(diffs, trades=trades)

    # EVMaker — optionally load fill model
    model_loaded = False
    fill_model = None
    if args.model.exists():
        try:
            from src.models.fill_prob import FillProbabilityModel
            fill_model = FillProbabilityModel.load(args.model)
            model_loaded = True
            logger.info("Fill model loaded from %s (AUC %.3f)",
                        args.model, fill_model.auc)
        except Exception as exc:
            logger.warning("Could not load fill model: %s", exc)

    logger.info("Running EVMaker ...")
    ev_engine = BacktestEngine(
        strategy=EVMaker(
            inventory_skew=InventorySkew(),
            size_calculator=SizeCalculator(),
            fill_model=fill_model,
        ),
        use_inventory=True,
        fee_bps=args.fee_bps,
        fill_mode=fill_mode,
    )
    ev = ev_engine.run(diffs, trades=trades)

    print_table(args.input, naive, ev, model_loaded)

    # Save CSV
    import pandas as pd
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    logger.info("Results saved to %s", args.out)


if __name__ == "__main__":
    main()
