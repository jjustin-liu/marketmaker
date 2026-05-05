"""Order book bootstrap and sync against Binance depth stream.

Implements the buffer-snapshot-apply algorithm from Binance's
documentation. Pure logic — no I/O. Caller fetches the REST snapshot
and pipes WebSocket messages in. Detects sequence gaps and signals
the caller to restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple


class GapDetectedError(Exception):
    """Raised when sequence numbers indicate dropped messages."""


@dataclass(frozen=True)
class DepthDiff:
    """A single Binance depth update event.

    first_update_id (U) and last_update_id (u) define the sequence
    range covered by this event. Updates are [(price, qty), ...].
    qty == 0 means delete that price level.
    """

    first_update_id: int
    last_update_id: int
    bid_updates: List[Tuple[float, float]]
    ask_updates: List[Tuple[float, float]]


@dataclass(frozen=True)
class Snapshot:
    """REST depth snapshot."""

    last_update_id: int
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]


class BookSynchronizer:
    """Drives the local order book through bootstrap and ongoing sync.

    Lifecycle:
      1. Caller creates a synchronizer and opens the WebSocket.
      2. Every incoming WS diff is fed to buffer() until the REST
         snapshot is fetched.
      3. Caller calls apply_snapshot(); the synchronizer discards
         stale buffered diffs, applies the snapshot, then replays
         any buffered diffs that come after it.
      4. After that, every live diff goes through apply().
      5. If apply() raises GapDetectedError, caller must drop the
         engine state, create a new synchronizer, and restart.
    """

    def __init__(self, engine: Any, *, bid_side: Any, ask_side: Any) -> None:
        self._engine = engine
        self._BID = bid_side
        self._ASK = ask_side
        self._buffer: List[DepthDiff] = []
        self._last_update_id: int = -1
        self._synced: bool = False

    @property
    def synced(self) -> bool:
        return self._synced

    @property
    def last_update_id(self) -> int:
        return self._last_update_id

    def buffer(self, diff: DepthDiff) -> None:
        """Hold a diff received before the snapshot arrives."""
        if self._synced:
            raise RuntimeError("already synced; call apply() instead")
        self._buffer.append(diff)

    def apply_snapshot(self, snapshot: Snapshot) -> None:
        """Load the snapshot and replay buffered diffs that follow it."""
        if self._synced:
            raise RuntimeError("apply_snapshot called twice")

        for price, qty in snapshot.bids:
            self._engine.apply_diff(self._BID, price, qty)
        for price, qty in snapshot.asks:
            self._engine.apply_diff(self._ASK, price, qty)

        snap_id = snapshot.last_update_id
        # Discard buffered diffs entirely older than the snapshot.
        usable = [d for d in self._buffer if d.last_update_id > snap_id]

        if not usable:
            # No buffered diffs cover post-snapshot territory yet.
            # That's fine — live diffs from now on will pick up the thread.
            self._last_update_id = snap_id
            self._buffer.clear()
            self._synced = True
            return

        # The first usable diff must straddle snap_id+1, per Binance's rule.
        first = usable[0]
        if not (first.first_update_id <= snap_id + 1 <= first.last_update_id):
            raise GapDetectedError(
                f"first buffered diff U={first.first_update_id}, "
                f"u={first.last_update_id} does not bridge snapshot "
                f"lastUpdateId={snap_id}"
            )

        prev_u = snap_id
        for d in usable:
            # First diff straddles; subsequent diffs must be strictly contiguous.
            if d is not first and d.first_update_id != prev_u + 1:
                raise GapDetectedError(
                    f"sequence gap: expected U={prev_u + 1}, "
                    f"got U={d.first_update_id}"
                )
            self._apply_updates(d)
            prev_u = d.last_update_id

        self._last_update_id = prev_u
        self._buffer.clear()
        self._synced = True

    def apply(self, diff: DepthDiff) -> None:
        """Apply a live diff. Raises GapDetectedError on sequence break."""
        if not self._synced:
            raise RuntimeError("apply() before apply_snapshot()")
        expected = self._last_update_id + 1
        if diff.first_update_id != expected:
            raise GapDetectedError(
                f"sequence gap: expected U={expected}, "
                f"got U={diff.first_update_id}"
            )
        self._apply_updates(diff)
        self._last_update_id = diff.last_update_id

    def _apply_updates(self, diff: DepthDiff) -> None:
        for price, qty in diff.bid_updates:
            self._engine.apply_diff(self._BID, price, qty)
        for price, qty in diff.ask_updates:
            self._engine.apply_diff(self._ASK, price, qty)
