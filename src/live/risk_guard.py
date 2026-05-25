"""Risk guard — hard halts on inventory or drawdown breach.

Pure-logic, no I/O. The order manager calls check() before each
quote refresh and on each fill; if it returns a non-empty halt
reason, the manager cancels open quotes and stops the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskConfig:
    """Hard caps. Both are absolute, not relative."""

    max_position: float = 0.01    # BTC
    max_drawdown: float = 50.0    # USD from peak PnL

    def __post_init__(self) -> None:
        if self.max_position <= 0:
            raise ValueError("max_position must be positive")
        if self.max_drawdown <= 0:
            raise ValueError("max_drawdown must be positive")


class RiskGuard:
    """Tracks peak PnL and reports the first violation it sees."""

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()
        self._peak_pnl: float = 0.0
        self._halted_reason: Optional[str] = None

    @property
    def halted(self) -> bool:
        return self._halted_reason is not None

    @property
    def halt_reason(self) -> Optional[str]:
        return self._halted_reason

    def check(self, inventory: float, pnl: float) -> Optional[str]:
        """Return halt reason if a cap is breached, else None.

        Once halted, stays halted (sticky). Tracks peak PnL across
        every call to compute drawdown from the high-water mark.
        """
        if self._halted_reason is not None:
            return self._halted_reason

        if pnl > self._peak_pnl:
            self._peak_pnl = pnl

        if abs(inventory) > self.config.max_position:
            self._halted_reason = (
                f"max_position breach: |inventory|={abs(inventory):.6f} "
                f"> {self.config.max_position}"
            )
            return self._halted_reason

        drawdown = self._peak_pnl - pnl
        if drawdown > self.config.max_drawdown:
            self._halted_reason = (
                f"max_drawdown breach: dd={drawdown:.2f} "
                f"> {self.config.max_drawdown}"
            )
            return self._halted_reason

        return None

    def trip(self, reason: str) -> None:
        """Force a halt with a custom reason (used on unhandled exceptions)."""
        if self._halted_reason is None:
            self._halted_reason = reason
