"""Expected-edge model for filled quotes.

Ridge regression predicting the signed per-fill edge of a candidate
quote, conditional on it filling: for a buy at price p, the label is
mid(t + horizon) - p; for a sell, p - mid(t + horizon). Positive means
the fill captured more spread than it lost to adverse selection over
the horizon.

This is the second half of an adverse-selection-aware EV objective.
P(fill) alone rewards distance (EV = P*h grows with h wherever P is
flat), but fills at distance are exactly the toxic ones — P(fill) and
adverse selection are positively correlated. Pairing P(fill) with a
conditional-edge estimate (EV = P * edge) prices that correlation in.

Trained by scripts/train_model_lookahead.py on the filled candidates of
the same lookahead walk that labels the fill model. Reuses
FillFeatures so both models see identical inputs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.fill_prob import FillProbabilityModel, Levels

DEFAULT_EDGE_MODEL_PATH = Path("data/models/edge.joblib")


@dataclass
class EdgeModel:
    """Ridge regression over FillFeatures -> expected signed edge ($)."""

    model: Optional[Ridge] = None
    scaler: Optional[StandardScaler] = None
    r2: Optional[float] = field(default=None)

    def predict(
        self,
        bids: Levels,
        asks: Levels,
        price: float,
        size: float,
        side: str,
    ) -> float:
        """Expected signed edge in dollars, conditional on a fill.

        Raises RuntimeError if the model is not trained.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("model is not trained; load() a bundle first")
        feats = FillProbabilityModel.extract_features(
            bids, asks, price, size, side,
        )
        x = feats.to_array().reshape(1, -1)
        x_s = self.scaler.transform(x)
        return float(self.model.predict(x_s)[0])

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist model + scaler with joblib. Returns path written."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("nothing to save; model is not trained")
        target = Path(path) if path is not None else DEFAULT_EDGE_MODEL_PATH
        os.makedirs(target.parent, exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "r2": self.r2},
            target,
        )
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EdgeModel":
        """Load a previously-saved bundle."""
        target = Path(path) if path is not None else DEFAULT_EDGE_MODEL_PATH
        if not target.exists():
            raise FileNotFoundError(f"no edge model file at {target}")
        bundle = joblib.load(target)
        return cls(
            model=bundle["model"],
            scaler=bundle["scaler"],
            r2=bundle.get("r2"),
        )
