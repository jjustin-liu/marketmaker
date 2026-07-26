"""Measure the hit-ratio lift from imbalance + volatility features.

A market maker with a fixed order budget must choose *which* quotes to
place. This experiment holds the quote distance constant (so distance
carries no information) and asks: using only imbalance and volatility
features, can a model pick the equidistant quotes that actually fill?

At each sampled tick we place one buy and one sell candidate a fixed
multiple of the spread from mid, and label each by lookahead (did the
market reach it within `lookahead` positions). A logistic model is
trained on a TIME-ORDERED earlier split and evaluated on the later one.

Hit-ratio lift is precision-at-budget: give the maker a budget of the
top-`select_frac` candidates.
  baseline  = place a random budget  -> fill rate = base rate
  model     = place the top-budget by predicted P(fill)
  lift      = model_fill_rate / base_rate - 1

Two feature sets are reported:
  imb+vol   = imbalance_1/2/5 + volatility + side   (the claim)
  +distance = the above plus the (here-constant) distance, as a control

Usage:
  python -m scripts.eval_hit_ratio \\
      --input data/raw/btcusdt_depth_2026-07-08.parquet \\
      --max-diffs 1500000 --lookahead 50 --offset-mult 1.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.backtest.engine import load_diffs_from_parquet
from src.features.imbalance import calculate_imbalance
from src.features.volatility import VolatilityCalculator
from src.lob.order_book import OrderBook, Side

logger = logging.getLogger("eval_hit_ratio")

DEPTH_LEVELS = 5
TEST_FRACTION = 0.2
SELECT_FRAC_DEFAULT = 0.2


def build_dataset(
    input_path: Path,
    max_diffs: int,
    lookahead: int,
    stride: int,
    offset_mult: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay the tape; return (features, fill labels) for fixed-distance quotes.

    Feature columns: imbalance_1, imbalance_2, imbalance_5, volatility,
    distance_bps, side (0 buy / 1 sell). Distance is held ~constant by
    construction so imbalance+volatility carry the signal.
    """
    diffs = load_diffs_from_parquet(input_path)
    if max_diffs > 0:
        diffs = diffs[:max_diffs]
    logger.info("replaying %d diffs", len(diffs))

    book = OrderBook()
    vol = VolatilityCalculator()
    in_snapshot = False

    best_bids: List[float] = []
    best_asks: List[float] = []
    # Pending candidates: (mid_idx, side, price, feature_row).
    pending: List[Tuple[int, str, float, List[float]]] = []
    rows: List[List[float]] = []
    labels: List[int] = []

    def _resolve(upto_idx: int) -> None:
        # Label any pending candidate whose lookahead window has closed.
        keep: List[Tuple[int, str, float, List[float]]] = []
        for (i0, side, price, feat) in pending:
            end = i0 + lookahead
            if end >= len(best_bids):
                if end < upto_idx:  # window fully elapsed but ran off the data
                    continue
                keep.append((i0, side, price, feat))
                continue
            window = range(i0 + 1, end + 1)
            if side == "buy":
                filled = any(best_asks[j] <= price for j in window)
            else:
                filled = any(best_bids[j] >= price for j in window)
            rows.append(feat)
            labels.append(1 if filled else 0)
        pending[:] = keep

    idx = 0
    for diff in diffs:
        if diff.is_snapshot and not in_snapshot:
            book = OrderBook()
            vol.reset()
            in_snapshot = True
        elif not diff.is_snapshot:
            in_snapshot = False
        book.apply_diff(diff.side, diff.price, diff.qty)
        if diff.is_snapshot:
            continue
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None or bb >= ba:
            continue

        mid = (bb + ba) / 2.0
        best_bids.append(bb)
        best_asks.append(ba)
        v = vol.update(mid)

        if idx % stride == 0:
            bids = book.depth(Side.BUY, DEPTH_LEVELS)
            asks = book.depth(Side.SELL, DEPTH_LEVELS)
            spread = ba - bb
            offset = spread * offset_mult
            dist_bps = offset / mid * 1e4
            imb1 = calculate_imbalance(bids, asks, 1)
            imb2 = calculate_imbalance(bids, asks, 2)
            imb5 = calculate_imbalance(bids, asks, 5)
            for side, price in (("buy", mid - offset), ("sell", mid + offset)):
                feat = [imb1, imb2, imb5, v, dist_bps,
                        0.0 if side == "buy" else 1.0]
                pending.append((idx, side, price, feat))
        idx += 1
        _resolve(idx)

    # Flush any candidates whose window closed at the tail.
    for (i0, side, price, feat) in pending:
        end = i0 + lookahead
        if end >= len(best_bids):
            continue
        window = range(i0 + 1, end + 1)
        if side == "buy":
            filled = any(best_asks[j] <= price for j in window)
        else:
            filled = any(best_bids[j] >= price for j in window)
        rows.append(feat)
        labels.append(1 if filled else 0)

    if len(rows) < 500:
        raise RuntimeError(f"only {len(rows)} rows — raise --max-diffs")
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int64)


def _lift(
    x: np.ndarray,
    y: np.ndarray,
    cols: List[int],
    select_frac: float,
    label: str,
) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    split = int(len(x) * (1.0 - TEST_FRACTION))
    x_tr, x_te = x[:split][:, cols], x[split:][:, cols]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(scaler.fit_transform(x_tr), y_tr)
    p = model.predict_proba(scaler.transform(x_te))[:, 1]

    auc = roc_auc_score(y_te, p) if len(np.unique(y_te)) > 1 else float("nan")
    base_rate = float(y_te.mean())
    budget = max(1, int(len(y_te) * select_frac))
    top = np.argsort(p)[::-1][:budget]
    model_rate = float(y_te[top].mean())
    lift = (model_rate / base_rate - 1.0) * 100 if base_rate > 0 else float("nan")

    logger.info(
        "%-22s AUC %.3f | base hit %.1f%% -> model hit %.1f%% "
        "(top %.0f%%) | LIFT %+.1f%%",
        label, auc, base_rate * 100, model_rate * 100,
        select_frac * 100, lift,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hit-ratio lift experiment.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-diffs", type=int, default=1_500_000)
    parser.add_argument("--lookahead", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--offset-mult", type=float, default=1.0,
                        help="quote distance as a multiple of the spread")
    parser.add_argument("--select-frac", type=float, default=SELECT_FRAC_DEFAULT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    x, y = build_dataset(
        args.input, args.max_diffs, args.lookahead, args.stride,
        args.offset_mult,
    )
    logger.info("%d candidates, base fill rate %.1f%%\n",
                len(y), y.mean() * 100)

    # cols: 0-2 imbalance, 3 volatility, 4 distance, 5 side
    _lift(x, y, [0, 1, 2, 3, 5], args.select_frac, "imbalance+volatility")
    _lift(x, y, [0, 1, 2, 3, 4, 5], args.select_frac, "  + distance (control)")


if __name__ == "__main__":
    main()
