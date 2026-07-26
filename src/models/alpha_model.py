"""Short-horizon directional alpha model.

Predicts the signed forward mid return over a lookahead horizon from
cheap L2-only signals: windowed OFI, microprice deviation, and book
imbalance at depths 1/2/5. This is what lets a later phase quote only
when expected edge beats adverse selection — it is NOT a fill model
and NOT a conditional-edge model.

Distinct from EdgeModel: EdgeModel predicts the value of a fill
*conditional on filling* (per-quote, price-dependent). AlphaModel
predicts the *unconditional* forward move of the mid (per-tick,
price-independent) — the direction the market is about to go, which
tells you whether to quote a side at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.features.imbalance import calculate_imbalance
from src.features.micro_price import calculate_microprice

DEFAULT_ALPHA_MODEL_PATH = Path("data/models/alpha.joblib")
IMBALANCE_DEPTHS = (1, 2, 5)

Levels = List[Tuple[float, float]]


@dataclass
class AlphaFeatures:
    """Directional signal vector at one tick.

    ofi: windowed order-flow imbalance (size units).
    microprice_dev: (microprice - mid) / mid, relative so it is
      comparable across price levels.
    imbalance_1/2/5: book imbalance over the top 1/2/5 levels.
    """

    ofi: float
    microprice_dev: float
    imbalance_1: float
    imbalance_2: float
    imbalance_5: float

    def to_array(self) -> np.ndarray:
        """Canonical float64 feature order. Must match training."""
        return np.array(
            [
                self.ofi,
                self.microprice_dev,
                self.imbalance_1,
                self.imbalance_2,
                self.imbalance_5,
            ],
            dtype=np.float64,
        )


def extract_alpha_features(
    bids: Levels,
    asks: Levels,
    ofi: float,
) -> AlphaFeatures:
    """Build the alpha feature vector from a book snapshot + windowed OFI.

    OFI is passed in because it is stateful (computed by OFITracker over
    the replay), not derivable from a single snapshot. Raises ValueError
    if the book is empty on either side (no mid/microprice defined).
    """
    if not bids or not asks:
        raise ValueError("cannot extract alpha features from an empty book")
    mid = (bids[0][0] + asks[0][0]) / 2.0
    if mid <= 0:
        raise ValueError("non-positive mid price")
    micro = calculate_microprice(bids, asks)
    micro_dev = 0.0 if micro is None else (micro - mid) / mid
    return AlphaFeatures(
        ofi=ofi,
        microprice_dev=micro_dev,
        imbalance_1=calculate_imbalance(bids, asks, 1),
        imbalance_2=calculate_imbalance(bids, asks, 2),
        imbalance_5=calculate_imbalance(bids, asks, 5),
    )


@dataclass
class AlphaModel:
    """Ridge over AlphaFeatures -> expected signed forward mid return."""

    model: Optional[Ridge] = None
    scaler: Optional[StandardScaler] = None
    ic: Optional[float] = field(default=None)

    def predict(self, features: AlphaFeatures) -> float:
        """Predicted signed forward mid return (relative units).

        Positive = mid expected to rise. Raises RuntimeError if untrained.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("model is not trained; load() a bundle first")
        x = features.to_array().reshape(1, -1)
        x_s = self.scaler.transform(x)
        return float(self.model.predict(x_s)[0])

    def predict_from_book(
        self,
        bids: Levels,
        asks: Levels,
        ofi: float,
    ) -> float:
        """Convenience: extract features from a book snapshot, then predict."""
        return self.predict(extract_alpha_features(bids, asks, ofi))

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist model + scaler with joblib. Returns path written."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("nothing to save; model is not trained")
        target = Path(path) if path is not None else DEFAULT_ALPHA_MODEL_PATH
        os.makedirs(target.parent, exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "ic": self.ic},
            target,
        )
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AlphaModel":
        """Load a previously-saved bundle."""
        target = Path(path) if path is not None else DEFAULT_ALPHA_MODEL_PATH
        if not target.exists():
            raise FileNotFoundError(f"no alpha model file at {target}")
        bundle = joblib.load(target)
        return cls(
            model=bundle["model"],
            scaler=bundle["scaler"],
            ic=bundle.get("ic"),
        )
