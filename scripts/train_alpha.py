"""Train and offline-validate the short-horizon alpha model.

Walks a depth parquet with the same epoch handling as the backtest
engine (reset book + OFI on each snapshot boundary), builds alpha
features at each processed tick, and labels each with the signed
forward mid return over `horizon` processed positions. Fits a Ridge
regressor with a TIME-ORDERED train/test split (train on the earlier
part, test on the later part) so the reported information coefficient
(IC) has no lookahead leakage.

This is Phase 12: validation only. It writes no strategy code and
changes no quoting. The decision it informs: does a usable directional
signal exist on this tape? Reported as:
  - test IC (Pearson corr of predicted vs realized forward return)
  - a decile table of mean realized forward return by predicted decile
  - the toxic-vs-benign split: mean forward return when the signal is
    bullish vs bearish (a bid fill is toxic exactly when the signal is
    bearish and the fill happens anyway)

Usage:
  python -m scripts.train_alpha --input data/raw/btcusdt_depth_DAY.parquet
      [--out data/models/alpha.joblib] [--horizon 50] [--ofi-window 50]
      [--max-diffs 1500000] [--stride 5]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.backtest.engine import load_diffs_from_parquet
from src.features.order_flow import OFITracker
from src.lob.order_book import OrderBook, Side
from src.models.alpha_model import (
    DEFAULT_ALPHA_MODEL_PATH,
    AlphaModel,
    extract_alpha_features,
)

logger = logging.getLogger("train_alpha")

HORIZON_DEFAULT = 50       # processed-mid positions ahead for the label
OFI_WINDOW_DEFAULT = 50
STRIDE_DEFAULT = 5         # sample 1 in N ticks to decorrelate rows
DEPTH_LEVELS = 5
TEST_FRACTION = 0.2
DECILES = 10


def build_alpha_dataset(
    input_path: Path,
    horizon: int,
    ofi_window: int,
    max_diffs: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay the tape and return (X features, y forward returns).

    Handles epochs exactly like BacktestEngine.run: a False->True
    is_snapshot transition resets the book and the OFI tracker; snapshot
    rows and crossed/empty books are skipped and never emit a row.
    """
    diffs = load_diffs_from_parquet(input_path)
    if max_diffs > 0:
        diffs = diffs[:max_diffs]
    logger.info("replaying %d diffs", len(diffs))

    book = OrderBook()
    ofi = OFITracker(window=ofi_window)
    in_snapshot = False

    mids: List[float] = []
    # feats_at[i] is the AlphaFeatures.to_array() at processed index i,
    # or None if that tick was not sampled (stride).
    feats_at: List[np.ndarray] = []
    sampled_idx: List[int] = []

    for diff in diffs:
        if diff.is_snapshot and not in_snapshot:
            book = OrderBook()
            ofi.reset()
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

        bids = book.depth(Side.BUY, DEPTH_LEVELS)
        asks = book.depth(Side.SELL, DEPTH_LEVELS)
        ofi_val = ofi.update(bids[0][0], bids[0][1], asks[0][0], asks[0][1])

        idx = len(mids)
        mids.append((bb + ba) / 2.0)
        feats_at.append(None)  # type: ignore[arg-type]
        if idx % stride == 0:
            feats_at[idx] = extract_alpha_features(bids, asks, ofi_val).to_array()
            sampled_idx.append(idx)

    x_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    for idx in sampled_idx:
        j = idx + horizon
        if j >= len(mids):
            break
        fwd_ret = (mids[j] - mids[idx]) / mids[idx]
        x_rows.append(feats_at[idx])
        y_rows.append(fwd_ret)

    if len(x_rows) < 200:
        raise RuntimeError(
            f"only {len(x_rows)} usable rows — tape too short; raise --max-diffs"
        )
    logger.info("built %d labeled rows", len(x_rows))
    return np.vstack(x_rows), np.array(y_rows, dtype=np.float64)


def _report_separation(y_pred: np.ndarray, y_true: np.ndarray) -> None:
    """Print decile table and bullish/bearish split of realized returns."""
    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    n = len(y_sorted)
    logger.info("  predicted-alpha decile -> mean realized forward return (bps)")
    for d in range(DECILES):
        lo = d * n // DECILES
        hi = (d + 1) * n // DECILES
        bucket = y_sorted[lo:hi]
        if len(bucket):
            logger.info("    D%-2d  %+8.4f bps  (n=%d)",
                        d + 1, bucket.mean() * 1e4, len(bucket))

    bull = y_true[y_pred > 0]
    bear = y_true[y_pred < 0]
    logger.info("  toxic-vs-benign split:")
    if len(bull):
        logger.info("    signal bullish (pred>0): mean fwd return %+8.4f bps "
                    "(n=%d) -> bid fills here are benign",
                    bull.mean() * 1e4, len(bull))
    if len(bear):
        logger.info("    signal bearish (pred<0): mean fwd return %+8.4f bps "
                    "(n=%d) -> bid fills here are toxic",
                    bear.mean() * 1e4, len(bear))


def train_alpha(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[AlphaModel, float]:
    """Fit Ridge with a time-ordered split. Returns (model, test IC)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    split = int(len(x) * (1.0 - TEST_FRACTION))
    x_tr, x_te = x[:split], x[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)
    model = Ridge(alpha=1.0)
    model.fit(x_tr_s, y_tr)

    y_pred = model.predict(x_te_s)
    if np.std(y_pred) < 1e-15 or np.std(y_te) < 1e-15:
        ic = 0.0
    else:
        ic = float(np.corrcoef(y_pred, y_te)[0, 1])
    logger.info("test IC (Pearson) = %+.4f  on %d test rows", ic, len(y_te))
    _report_separation(y_pred, y_te)
    return AlphaModel(model=model, scaler=scaler, ic=ic), ic


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + validate alpha model.")
    parser.add_argument("--input", type=Path, required=True,
                        help="depth parquet")
    parser.add_argument("--out", type=Path, default=DEFAULT_ALPHA_MODEL_PATH)
    parser.add_argument("--horizon", type=int, default=HORIZON_DEFAULT,
                        help="forward mid positions for the label")
    parser.add_argument("--ofi-window", type=int, default=OFI_WINDOW_DEFAULT)
    parser.add_argument("--max-diffs", type=int, default=1_500_000)
    parser.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    parser.add_argument("--no-save", action="store_true",
                        help="validate only; do not write the model file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    x, y = build_alpha_dataset(
        args.input, args.horizon, args.ofi_window, args.max_diffs, args.stride,
    )
    model, ic = train_alpha(x, y)
    if args.no_save:
        logger.info("--no-save: skipping model write")
        return
    out = model.save(args.out)
    logger.info("saved alpha model to %s (IC %.4f)", out, ic)


if __name__ == "__main__":
    main()
