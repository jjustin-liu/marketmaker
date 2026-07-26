"""Train the fill probability model using lookahead labeling on recorded data.

Better than synthetic negatives: walks forward through the recorded
diff stream and labels each candidate quote based on whether the
market actually crossed it within `lookahead_ticks`.

For each sampled tick, generate K candidate buy and ask prices around
mid, then:
  - buy candidate at price P labeled 1 iff min(best_ask) over next
    LOOKAHEAD ticks <= P
  - sell candidate at price P labeled 1 iff max(best_bid) over next
    LOOKAHEAD ticks >= P

Usage:
  python -m scripts.train_model_lookahead --input data/raw/btcusdt.parquet
                                          [--out data/models/fill_prob.joblib]
                                          [--lookahead 50] [--max-ticks 50000]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.lob.order_book import OrderBook, Side
from src.models.fill_prob import DEFAULT_MODEL_PATH, FillProbabilityModel

BPS = 1e-4
LOOKAHEAD_DEFAULT = 50
SAMPLES_PER_TICK = 4  # 2 buy + 2 sell candidates per sampled tick
TICK_SAMPLE_STRIDE = 5  # only sample 1 in N ticks for training (decorrelate)
MAX_OFFSET_BPS_DEFAULT = 50.0

logger = logging.getLogger("train_lookahead")


def _build_book_snapshots(
    df: pd.DataFrame,
    max_ticks: int,
) -> Tuple[List[List[Tuple[float, float]]],
           List[List[Tuple[float, float]]],
           List[float]]:
    """Replay diffs and snapshot top-5 bids/asks + mid at each step."""
    book = OrderBook()
    bids_seq: List[List[Tuple[float, float]]] = []
    asks_seq: List[List[Tuple[float, float]]] = []
    mids: List[float] = []

    for i, row in enumerate(df.itertuples(index=False)):
        if i >= max_ticks:
            break
        side = Side.BUY if str(row.side).lower() in ("buy", "bid", "b") else Side.SELL
        book.apply_diff(side, float(row.price), float(row.qty))
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None:
            bids_seq.append([])
            asks_seq.append([])
            mids.append(float("nan"))
            continue
        bids_seq.append(book.depth(Side.BUY, 5))
        asks_seq.append(book.depth(Side.SELL, 5))
        mids.append((bb + ba) / 2.0)
    return bids_seq, asks_seq, mids


def _label(
    mid: float,
    candidate_price: float,
    side: str,
    future_best_asks: List[float],
    future_best_bids: List[float],
) -> int:
    if side == "buy":
        for ba in future_best_asks:
            if ba <= candidate_price:
                return 1
        return 0
    else:
        for bb in future_best_bids:
            if bb >= candidate_price:
                return 1
        return 0


def build_fills_dataset(
    parquet_path: Path,
    lookahead: int,
    max_ticks: int,
    max_offset_bps: float = MAX_OFFSET_BPS_DEFAULT,
) -> pd.DataFrame:
    if max_offset_bps <= 0:
        raise ValueError("max_offset_bps must be positive")
    raw = pd.read_parquet(parquet_path)
    logger.info("loaded %d raw rows", len(raw))

    bids_seq, asks_seq, mids = _build_book_snapshots(raw, max_ticks)
    logger.info("replayed %d ticks", len(mids))

    rows: List[dict] = []
    rng = np.random.default_rng(42)
    n = len(mids)

    for i in range(0, n - lookahead, TICK_SAMPLE_STRIDE):
        mid_i = mids[i]
        if not np.isfinite(mid_i):
            continue
        bids = bids_seq[i]
        asks = asks_seq[i]
        if not bids or not asks:
            continue

        spread = asks[0][0] - bids[0][0]
        if spread <= 0:
            continue
        min_offset = spread * 0.1
        max_offset = max(min_offset, mid_i * max_offset_bps * BPS)

        future_best_asks = [
            asks_seq[j][0][0] for j in range(i + 1, i + 1 + lookahead)
            if asks_seq[j]
        ]
        future_best_bids = [
            bids_seq[j][0][0] for j in range(i + 1, i + 1 + lookahead)
            if bids_seq[j]
        ]
        if not future_best_asks or not future_best_bids:
            continue

        # Realized conditional edge: what a fill at this price is worth
        # by the end of the lookahead window. Signed per side.
        mid_horizon = mids[i + lookahead]

        for _ in range(SAMPLES_PER_TICK // 2):
            offset = float(
                np.exp(rng.uniform(np.log(min_offset), np.log(max_offset)))
            )
            for side, candidate in (
                ("buy", mid_i - offset),
                ("sell", mid_i + offset),
            ):
                if np.isfinite(mid_horizon):
                    edge = (mid_horizon - candidate if side == "buy"
                            else candidate - mid_horizon)
                else:
                    edge = float("nan")
                rows.append({
                    "bids": [list(x) for x in bids],
                    "asks": [list(x) for x in asks],
                    "price": candidate,
                    "size": 0.001,
                    "side": side,
                    "_label": _label(
                        mid_i, candidate, side,
                        future_best_asks, future_best_bids,
                    ),
                    "_edge": edge,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("dataset is empty — input too short or all NaN mids")
    pos = int(df["_label"].sum())
    logger.info("built %d rows (%d positives, %.1f%% fill rate)",
                len(df), pos, 100.0 * pos / len(df))
    return df


def train_from_labeled(df: pd.DataFrame) -> Tuple[FillProbabilityModel, float]:
    """Train with externally-labeled rows (skips synthetic negative step).

    Hijacks the FillProbabilityModel internals: extract features per
    row, use the row's _label, fit the same scaler+logreg pipeline.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    for _, row in df.iterrows():
        bids = [tuple(x) for x in row["bids"]]
        asks = [tuple(x) for x in row["asks"]]
        feats = FillProbabilityModel.extract_features(
            bids, asks, float(row["price"]), float(row["size"]),
            str(row["side"]),
        )
        X_rows.append(feats.to_array())
        y_rows.append(int(row["_label"]))

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=np.int64)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr_s, y_tr)
    probs = model.predict_proba(X_te_s)[:, 1]
    auc = float(roc_auc_score(y_te, probs))
    return FillProbabilityModel(model=model, scaler=scaler, auc=auc), auc


def train_edge_from_labeled(df: pd.DataFrame) -> Tuple["EdgeModel", float]:
    """Fit the conditional-edge regressor on FILLED candidates only.

    Label is the realized signed edge (_edge) of rows with _label == 1:
    the value of the fill after the adverse move that tends to follow
    it. Returns (EdgeModel, test R^2).
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from src.models.edge_model import EdgeModel

    filled = df[(df["_label"] == 1) & np.isfinite(df["_edge"])]
    if len(filled) < 100:
        raise RuntimeError(
            f"only {len(filled)} filled candidates — too few to fit the "
            "edge model; increase --max-ticks"
        )

    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    for _, row in filled.iterrows():
        bids = [tuple(x) for x in row["bids"]]
        asks = [tuple(x) for x in row["asks"]]
        feats = FillProbabilityModel.extract_features(
            bids, asks, float(row["price"]), float(row["size"]),
            str(row["side"]),
        )
        X_rows.append(feats.to_array())
        y_rows.append(float(row["_edge"]))

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=np.float64)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = Ridge(alpha=1.0)
    model.fit(X_tr_s, y_tr)
    r2 = float(r2_score(y_te, model.predict(X_te_s)))
    logger.info("edge model: %d filled rows, mean edge %.4f, test R2 %.4f",
                len(filled), float(y.mean()), r2)
    return EdgeModel(model=model, scaler=scaler, r2=r2), r2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--edge-out", type=Path, default=None,
                        help="also fit the conditional-edge model on filled "
                             "candidates and save it here")
    parser.add_argument("--lookahead", type=int, default=LOOKAHEAD_DEFAULT)
    parser.add_argument("--max-ticks", type=int, default=50_000)
    parser.add_argument(
        "--max-offset-bps",
        type=float,
        default=MAX_OFFSET_BPS_DEFAULT,
        help="widest candidate distance from mid, in bps; default matches EV",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    df = build_fills_dataset(
        args.input, args.lookahead, args.max_ticks, args.max_offset_bps,
    )
    model, auc = train_from_labeled(df)
    logger.info("test AUC = %.4f", auc)
    out = model.save(args.out)
    logger.info("saved model to %s", out)

    if args.edge_out is not None:
        edge_model, _ = train_edge_from_labeled(df)
        edge_path = edge_model.save(args.edge_out)
        logger.info("saved edge model to %s", edge_path)


if __name__ == "__main__":
    main()
