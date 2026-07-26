"""Render a backtest results CSV into a Markdown P&L report.

Consumes the CSV written by scripts/run_backtest.py (one row per
strategy: pnl, sharpe, hit_ratio, adverse, max_dd, fills, quotes) and
emits a Markdown summary suitable for a CI artifact or PR comment.

Usage:
  python -m scripts.pnl_report --input data/backtest_results.csv \\
      --out reports/pnl.md [--title "Nightly backtest"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def render(df: pd.DataFrame, title: str, source: str) -> str:
    lines = [f"# {title}", "", f"Source: `{source}`", ""]
    lines.append("| Strategy | PnL (USDT) | Sharpe | Hit ratio | "
                 "Markout ($/fill) | Max DD | Fills | Quotes |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['pnl']:+.2f} | {r['sharpe']:.1f} | "
            f"{r['hit_ratio'] * 100:.1f}% | {r['adverse']:+.3f} | "
            f"{r['max_dd']:.2f} | {int(r['fills'])} | {int(r['quotes'])} |"
        )
    lines.append("")

    if {"naive", "ev"}.issubset(set(df["strategy"])):
        naive = df[df["strategy"] == "naive"].iloc[0]
        ev = df[df["strategy"] == "ev"].iloc[0]
        winner = "Naive" if naive["pnl"] >= ev["pnl"] else "EV"
        lines.append(f"**Higher PnL:** {winner} "
                     f"(Naive {naive['pnl']:+.2f} vs EV {ev['pnl']:+.2f}).")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest CSV -> Markdown P&L.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/pnl.md"))
    parser.add_argument("--title", default="Backtest P&L report")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"results CSV not found: {args.input}")
    df = pd.read_csv(args.input)
    report = render(df, args.title, args.input.name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
