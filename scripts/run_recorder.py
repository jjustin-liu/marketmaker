"""L2 depth recorder — independent process, writes parquet to data/raw/.

Subscribes to Binance btcusdt@depth, bootstraps from a REST snapshot
per Binance's documented sequence (buffer → snapshot → filter →
replay → stream), and writes the full reconstructable diff sequence
to a daily-rotated parquet file (UTC). Each parquet starts with the
snapshot encoded as one row per price level, then contains live
diffs from there forward. Downstream replayers can apply rows in
order and the book will always be uncrossed.

Run as a separate process from the live feed. Disk slowness here
cannot affect live trading.

Usage:
  python -m scripts.run_recorder [--hours 24] [--symbol btcusdt]
                                 [--out data/raw] [--flush-rows 10000]
                                 [--no-bootstrap]   # legacy mode
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp
import pandas as pd
import websockets

WS_BASES = {
    "GLOBAL": "wss://stream.binance.com:9443/stream",
    "VISION": "wss://data-stream.binance.vision/stream",
    "US": "wss://stream.binance.us:9443/stream",
    "TEST": "wss://stream.testnet.binance.vision/stream",
}
# REST mirrors. VISION is the US-accessible public market-data mirror.
REST_BASES = {
    "GLOBAL": "https://api.binance.com",
    "VISION": "https://data-api.binance.vision",
    "US": "https://api.binance.us",
    "TEST": "https://testnet.binance.vision",
}
DEFAULT_REGION = "VISION"
SNAPSHOT_LIMIT = 1000

logger = logging.getLogger("recorder")


class GapDetected(Exception):
    """Sequence gap. Caller must drop state and restart."""


class ParquetRotator:
    """Daily-rotated parquet writer (UTC). One file per day."""

    def __init__(self, out_dir: Path, symbol: str) -> None:
        self.out_dir = out_dir
        self.symbol = symbol
        self.buffer: List[dict] = []
        self.current_date: dt.date | None = None
        out_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, date: dt.date) -> Path:
        return self.out_dir / f"{self.symbol}_depth_{date.isoformat()}.parquet"

    def add(self, row: dict) -> None:
        self.buffer.append(row)

    def extend(self, rows: List[dict]) -> None:
        self.buffer.extend(rows)

    def flush(self) -> int:
        if not self.buffer:
            return 0
        now = dt.datetime.now(dt.timezone.utc).date()
        df = pd.DataFrame(self.buffer)
        path = self._path_for(now)
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old, df], ignore_index=True)
        df.to_parquet(path, index=False)
        n = len(self.buffer)
        self.buffer.clear()
        self.current_date = now
        return n


def parse_depth_message(payload: dict) -> Tuple[int, int, List[dict]]:
    """Flatten a depth event. Returns (U, u, rows).

    U = first_update_id, u = last_update_id. rows are the parquet
    rows (one per price level update), one per (bid, ask) entry.
    """
    ts = int(payload.get("E", 0))
    U = int(payload.get("U", -1))
    u = int(payload.get("u", -1))
    rows: List[dict] = []
    for p, q in payload.get("b", []):
        rows.append({"timestamp": ts, "side": "buy",
                     "price": float(p), "qty": float(q)})
    for p, q in payload.get("a", []):
        rows.append({"timestamp": ts, "side": "sell",
                     "price": float(p), "qty": float(q)})
    return U, u, rows


def snapshot_to_rows(snapshot: dict, ts_ms: int) -> List[dict]:
    """Encode a REST depth snapshot as parquet rows (one per level)."""
    rows: List[dict] = []
    for p, q in snapshot.get("bids", []):
        rows.append({"timestamp": ts_ms, "side": "buy",
                     "price": float(p), "qty": float(q)})
    for p, q in snapshot.get("asks", []):
        rows.append({"timestamp": ts_ms, "side": "sell",
                     "price": float(p), "qty": float(q)})
    return rows


async def fetch_snapshot(
    session: aiohttp.ClientSession,
    symbol: str,
    rest_base: str,
    limit: int = SNAPSHOT_LIMIT,
) -> dict:
    """GET /api/v3/depth. Returns parsed JSON dict."""
    url = (f"{rest_base}/api/v3/depth"
           f"?symbol={symbol.upper()}&limit={limit}")
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
        r.raise_for_status()
        return await r.json()


def filter_buffered_diffs(
    buffered: List[Tuple[int, int, List[dict]]],
    snapshot_last_update_id: int,
) -> List[Tuple[int, int, List[dict]]]:
    """Apply Binance's bootstrap filter to buffered events.

    Per docs:
      1. Drop events where u <= snapshot.lastUpdateId.
      2. First remaining event must satisfy U <= snap+1 <= u (it
         straddles the boundary). If not, the buffer is too late
         or has a gap; raise GapDetected so the caller restarts.
      3. Subsequent events must be strictly contiguous (each U =
         previous u + 1). A gap here also restarts.
    """
    usable = [e for e in buffered if e[1] > snapshot_last_update_id]
    if not usable:
        return []
    U0, u0, _ = usable[0]
    if not (U0 <= snapshot_last_update_id + 1 <= u0):
        raise GapDetected(
            f"first usable diff (U={U0}, u={u0}) does not bridge "
            f"snapshot lastUpdateId={snapshot_last_update_id}"
        )
    prev_u = u0
    for i in range(1, len(usable)):
        Ui, ui, _ = usable[i]
        if Ui != prev_u + 1:
            raise GapDetected(
                f"sequence gap in buffered diffs: expected U={prev_u + 1}, "
                f"got U={Ui}"
            )
        prev_u = ui
    return usable


async def _bootstrap_session(
    ws: websockets.WebSocketClientProtocol,
    rest_session: aiohttp.ClientSession,
    symbol: str,
    rest_base: str,
    rotator: ParquetRotator,
) -> int:
    """Run buffer→snapshot→filter→write. Returns last_update_id we hold."""
    buffered: List[Tuple[int, int, List[dict]]] = []
    snapshot: Optional[dict] = None
    snap_ts_ms = 0

    async for raw in ws:
        env = json.loads(raw)
        payload = env.get("data", {})
        if not payload.get("U"):
            continue
        buffered.append(parse_depth_message(payload))

        if snapshot is None:
            snapshot = await fetch_snapshot(rest_session, symbol, rest_base)
            snap_ts_ms = int(
                dt.datetime.now(dt.timezone.utc).timestamp() * 1000
            )
            logger.info(
                "snapshot fetched: lastUpdateId=%s, bids=%d, asks=%d",
                snapshot["lastUpdateId"],
                len(snapshot["bids"]), len(snapshot["asks"]),
            )

        snap_id = int(snapshot["lastUpdateId"])
        if buffered[-1][1] <= snap_id:
            # newest event is still stale; keep buffering
            continue

        usable = filter_buffered_diffs(buffered, snap_id)
        rotator.extend(snapshot_to_rows(snapshot, snap_ts_ms))
        for _, _, rows in usable:
            rotator.extend(rows)
        last_u = usable[-1][1] if usable else snap_id
        logger.info(
            "bootstrap complete: snap=%d, %d buffered diffs applied, "
            "now at u=%d",
            snap_id, len(usable), last_u,
        )
        return last_u

    raise GapDetected("ws closed before bootstrap completed")


async def record(
    symbol: str,
    out_dir: Path,
    duration_sec: float | None,
    flush_rows: int,
    region: str,
    use_bootstrap: bool = True,
) -> None:
    if region not in WS_BASES:
        raise ValueError(f"unknown region {region!r}; pick from {list(WS_BASES)}")
    rest_base = REST_BASES[region]
    rotator = ParquetRotator(out_dir, symbol)
    stream_url = f"{WS_BASES[region]}?streams={symbol}@depth"

    stop = asyncio.Event()

    def handle_signal(*_: object) -> None:
        logger.info("shutdown signal received")
        stop.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    deadline = (
        loop.time() + duration_sec if duration_sec is not None else None
    )
    total_rows = 0

    async with aiohttp.ClientSession() as http:
        while not stop.is_set():
            try:
                async with websockets.connect(stream_url) as ws:
                    logger.info("connected to %s", stream_url)
                    if use_bootstrap:
                        last_u = await _bootstrap_session(
                            ws, http, symbol, rest_base, rotator,
                        )
                    else:
                        last_u = -1

                    async for raw in ws:
                        if stop.is_set():
                            break
                        if deadline is not None and loop.time() >= deadline:
                            stop.set()
                            break
                        env = json.loads(raw)
                        payload = env.get("data", {})
                        U, u, rows = parse_depth_message(payload)
                        if use_bootstrap and U != -1 and last_u != -1:
                            if U != last_u + 1:
                                raise GapDetected(
                                    f"live sequence gap: expected U="
                                    f"{last_u + 1}, got U={U}"
                                )
                            last_u = u
                        rotator.extend(rows)
                        if len(rotator.buffer) >= flush_rows:
                            n = rotator.flush()
                            total_rows += n
                            logger.info(
                                "flushed %d rows (total %d)", n, total_rows,
                            )
            except GapDetected as e:
                logger.warning("gap: %s; re-bootstrapping in 2s", e)
                await asyncio.sleep(2)
            except (websockets.ConnectionClosed, OSError) as e:
                if stop.is_set():
                    break
                logger.warning("connection dropped: %s; reconnecting in 5s", e)
                await asyncio.sleep(5)

    n = rotator.flush()
    total_rows += n
    logger.info("final flush %d rows; total recorded %d", n, total_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 depth recorder.")
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--hours", type=float, default=None,
                        help="record for this many hours; default = forever")
    parser.add_argument("--flush-rows", type=int, default=10_000)
    parser.add_argument("--region", default=DEFAULT_REGION,
                        choices=list(WS_BASES),
                        help="endpoint set; VISION is US-accessible")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="skip the REST snapshot bootstrap (legacy mode)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    duration = args.hours * 3600 if args.hours else None

    try:
        asyncio.run(record(
            args.symbol, args.out, duration, args.flush_rows, args.region,
            use_bootstrap=not args.no_bootstrap,
        ))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
