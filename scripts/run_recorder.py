"""L2 depth recorder — independent process, writes parquet to data/raw/.

Subscribes to Binance btcusdt@depth, buffers diffs in memory, and
flushes to a daily-rotated parquet file (UTC). No bootstrap, no
local-book maintenance, no Redis — pure archival.

Run as a separate process from the live feed. Disk slowness here
cannot affect live trading.

Usage:
  python -m scripts.run_recorder [--hours 24] [--symbol btcusdt]
                                 [--out data/raw] [--flush-rows 10000]
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
from typing import List

import pandas as pd
import websockets

WS_BASE = "wss://stream.binance.com:9443/stream"

logger = logging.getLogger("recorder")


class ParquetRotator:
    """Daily-rotated parquet writer using pandas.

    Buffers rows in memory and writes one file per UTC day. Reopens a
    fresh file at midnight UTC. On flush, appends rows to that day's
    file (rewriting the whole file each time — fine for our row rates
    and simpler than streaming ParquetWriter).
    """

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


def parse_depth_message(payload: dict) -> List[dict]:
    """Flatten one Binance depth event into one row per price update."""
    ts = int(payload.get("E", 0))
    rows: List[dict] = []
    for p, q in payload.get("b", []):
        rows.append({"timestamp": ts, "side": "buy",
                     "price": float(p), "qty": float(q)})
    for p, q in payload.get("a", []):
        rows.append({"timestamp": ts, "side": "sell",
                     "price": float(p), "qty": float(q)})
    return rows


async def record(
    symbol: str,
    out_dir: Path,
    duration_sec: float | None,
    flush_rows: int,
) -> None:
    rotator = ParquetRotator(out_dir, symbol)
    stream_url = f"{WS_BASE}?streams={symbol}@depth"

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

    while not stop.is_set():
        try:
            async with websockets.connect(stream_url) as ws:
                logger.info("connected to %s", stream_url)
                async for raw in ws:
                    if stop.is_set():
                        break
                    if deadline is not None and loop.time() >= deadline:
                        stop.set()
                        break
                    env = json.loads(raw)
                    payload = env.get("data", {})
                    rows = parse_depth_message(payload)
                    for r in rows:
                        rotator.add(r)
                    if len(rotator.buffer) >= flush_rows:
                        n = rotator.flush()
                        total_rows += n
                        logger.info("flushed %d rows (total %d)", n, total_rows)
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    duration = args.hours * 3600 if args.hours else None

    try:
        asyncio.run(record(args.symbol, args.out, duration, args.flush_rows))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
