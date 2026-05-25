"""Fill probability model.

Logistic regression over the 10-element feature vector defined in
ARCHITECTURE.md. Training is done by scripts/train_model_lookahead.py,
which builds labels by walking forward through recorded L2 diffs and
checking whether each candidate quote was actually crossed within a
fixed lookahead window. This class owns the inference path
(extract_features / predict) and persistence (save / load).

Input feature order (must match src/features/feature_generator.py):
  [bid_ask_spread, mid_price, bid_volume, ask_volume,
   imbalance_1, imbalance_2, imbalance_5,
   price_distance, size, side]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.imbalance import calculate_imbalance

FEATURE_VECTOR_SIZE = 10
SIDE_BUY = 1.0
SIDE_SELL = 0.0
DEFAULT_MODEL_PATH = Path("data/models/fill_prob.joblib")


@dataclass
class FillFeatures:
    """Single row of model input.

    Field order is the canonical model input order. Do not reorder.
    """

    bid_ask_spread: float
    mid_price: float
    bid_volume: float
    ask_volume: float
    imbalance_1: float
    imbalance_2: float
    imbalance_5: float
    price_distance: float
    size: float
    side: float

    def to_array(self) -> np.ndarray:
        """Return as a length-10 float64 numpy array in canonical order."""
        return np.array(
            [
                self.bid_ask_spread,
                self.mid_price,
                self.bid_volume,
                self.ask_volume,
                self.imbalance_1,
                self.imbalance_2,
                self.imbalance_5,
                self.price_distance,
                self.size,
                self.side,
            ],
            dtype=np.float64,
        )


Levels = List[Tuple[float, float]]


@dataclass
class FillProbabilityModel:
    """Logistic regression fill probability model with persistence."""

    model: Optional[LogisticRegression] = None
    scaler: Optional[StandardScaler] = None
    auc: Optional[float] = field(default=None)

    @staticmethod
    def extract_features(
        bids: Levels,
        asks: Levels,
        price: float,
        size: float,
        side: str,
    ) -> FillFeatures:
        """Build a FillFeatures row from book state and a candidate quote."""
        if not bids or not asks:
            raise ValueError("bids and asks must both be non-empty")
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2.0
        if mid == 0.0:
            raise ValueError("mid price is zero")

        spread = (best_ask - best_bid) / mid
        bid_volume = sum(q for _, q in bids)
        ask_volume = sum(q for _, q in asks)
        side_val = SIDE_BUY if side == "buy" else SIDE_SELL

        return FillFeatures(
            bid_ask_spread=spread,
            mid_price=mid,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance_1=calculate_imbalance(bids, asks, 1),
            imbalance_2=calculate_imbalance(bids, asks, 2),
            imbalance_5=calculate_imbalance(bids, asks, 5),
            price_distance=abs(price - mid) / mid,
            size=size,
            side=side_val,
        )

    def predict(
        self,
        bids: Levels,
        asks: Levels,
        price: float,
        size: float,
        side: str,
    ) -> float:
        """Return P(fill) in [0, 1]. Raises if model is not trained."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("model is not trained; call train() or load()")
        feats = self.extract_features(bids, asks, price, size, side)
        x = feats.to_array().reshape(1, -1)
        x_s = self.scaler.transform(x)
        return float(self.model.predict_proba(x_s)[0, 1])

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist model + scaler with joblib. Returns path written."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("nothing to save; model is not trained")
        target = Path(path) if path is not None else DEFAULT_MODEL_PATH
        os.makedirs(target.parent, exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "auc": self.auc},
            target,
        )
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FillProbabilityModel":
        """Load a previously-saved model bundle."""
        target = Path(path) if path is not None else DEFAULT_MODEL_PATH
        if not target.exists():
            raise FileNotFoundError(f"no model file at {target}")
        bundle = joblib.load(target)
        return cls(
            model=bundle["model"],
            scaler=bundle["scaler"],
            auc=bundle.get("auc"),
        )
