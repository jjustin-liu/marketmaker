"""Inventory-aware order sizing.

Bid/ask sizes scale with inventory: long position grows ask size and
shrinks bid size (push to flatten); symmetric for short. At max
position the unwanted side zeros out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ScalingType(str, Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"


@dataclass
class SizeConfig:
    base_size: float = 0.001
    max_size_mult: float = 3.0
    max_position: float = 1.0
    scaling_type: ScalingType = ScalingType.LINEAR
    sigmoid_steepness: float = 4.0
    min_size_mult: float = 0.1

    def __post_init__(self) -> None:
        if self.base_size <= 0:
            raise ValueError("base_size must be positive")
        if self.max_size_mult < 1.0:
            raise ValueError("max_size_mult must be >= 1.0")
        if self.max_position <= 0:
            raise ValueError("max_position must be positive")
        if self.min_size_mult < 0:
            raise ValueError("min_size_mult must be non-negative")


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class SizeCalculator:
    """Compute per-side sizes from current inventory."""

    def __init__(self, config: Optional[SizeConfig] = None) -> None:
        self.config = config or SizeConfig()

    def get_sizes(self, inventory: float) -> Tuple[float, float]:
        """Return (bid_size, ask_size) given current inventory."""
        cfg = self.config
        norm = _clip(inventory / cfg.max_position, -1.0, 1.0)

        if cfg.scaling_type is ScalingType.SIGMOID and abs(norm) < 1.0:
            scaled = 2.0 / (1.0 + math.exp(-norm * cfg.sigmoid_steepness)) - 1.0
        else:
            scaled = norm

        base_mult = (cfg.max_size_mult - 1.0) / 2.0

        if scaled >= 0:
            bid_mult = max(cfg.min_size_mult, 1.0 - scaled * base_mult * 2.0)
            ask_mult = 1.0 + scaled * base_mult * 2.0
            if scaled >= 1.0:
                bid_mult = 0.0
        else:
            bid_mult = 1.0 - scaled * base_mult * 2.0
            ask_mult = max(cfg.min_size_mult, 1.0 + scaled * base_mult * 2.0)
            if scaled <= -1.0:
                ask_mult = 0.0

        return cfg.base_size * bid_mult, cfg.base_size * ask_mult
