"""L2 depth recorder — independent process, writes parquet to data/raw/.

Subscribes to Binance btcusdt@depth, bootstraps from a REST snapshot
per Binance's documented sequence (buffer → snapshot → filter →
replay → stream), and writes the full reconstructable diff sequence
to a daily-rotated parquet file (UTC). Each parquet starts with the
snapshot encoded as one row per price level, then contains live
diffs from there forward. Downstream replayers can apply rows in
order and the book will always be uncrossed.

Also subscribes to btcusdt@trade (unless --no-trades) and writes
executed trades, tagged with aggressor side, to a separate daily
file ({symbol}_trades_{date}.parquet). Trades let a backtest fill a
resting quote when a print crosses its price — independent of where
the book moves next — which depth-only replay can't capture.

Run as a separate process from the live feed. Disk slowness here
cannot affect live trading.

Usage:
  python -m scripts.run_recorder [--hours 24] [--symbol btcusdt]
                                 [--out data/raw] [--flush-rows 10000]
                                 [--no-bootstrap]   # legacy mode
                                 [--no-trades]      # depth only
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

STATUS_INTERVAL_SEC = 60  # how often to print a status line

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
    """Daily-rotated parquet writer (UTC). One file per day.

    `kind` selects the filename stream tag ("depth" or "trades") so depth
    and trade streams write to separate daily files.
    """

    def __init__(self, out_dir: Path, symbol: str, kind: str = "depth") -> None:
        self.out_dir = out_dir
        self.symbol = symbol
        self.kind = kind
        self.buffer: List[dict] = []
        self.current_date: dt.date | None = None
        out_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, date: dt.date) -> Path:
        return (self.out_dir
                / f"{self.symbol}_{self.kind}_{date.isoformat()}.parquet")

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
    Diff rows carry is_snapshot=False so replay can distinguish them
    from bootstrap/seed rows.
    """
    ts = int(payload.get("E", 0))
    U = int(payload.get("U", -1))
    u = int(payload.get("u", -1))
    rows: List[dict] = []
    for p, q in payload.get("b", []):
        rows.append({"timestamp": ts, "side": "buy",
                     "price": float(p), "qty": float(q),
                     "is_snapshot": False})
    for p, q in payload.get("a", []):
        rows.append({"timestamp": ts, "side": "sell",
                     "price": float(p), "qty": float(q),
                     "is_snapshot": False})
    return U, u, rows


def parse_trade_message(payload: dict) -> dict:
    """Flatten a Binance @trade event into a parquet row.

    Fields: T = trade time (ms), p = price, q = quantity, m = isBuyerMaker.
    Aggressor side: if the buyer is the maker the trade was initiated by a
    seller → aggressor = "sell"; otherwise "buy". A resting bid fills on a
    "sell" print at/through its price; a resting ask on a "buy" print.
    """
    ts = int(payload.get("T", payload.get("E", 0)))
    is_buyer_maker = bool(payload.get("m", False))
    return {
        "timestamp": ts,
        "side": "sell" if is_buyer_maker else "buy",
        "price": float(payload.get("p", 0.0)),
        "qty": float(payload.get("q", 0.0)),
    }


def snapshot_to_rows(snapshot: dict, ts_ms: int) -> List[dict]:
    """Encode a REST depth snapshot as parquet rows (one per level).

    Snapshot rows carry is_snapshot=True. The replay loader applies
    all snapshot rows before any diffs, so order within the snapshot
    block is irrelevant.
    """
    rows: List[dict] = []
    for p, q in snapshot.get("bids", []):
        rows.append({"timestamp": ts_ms, "side": "buy",
                     "price": float(p), "qty": float(q),
                     "is_snapshot": True})
    for p, q in snapshot.get("asks", []):
        rows.append({"timestamp": ts_ms, "side": "sell",
                     "price": float(p), "qty": float(q),
                     "is_snapshot": True})
    return rows


def book_to_snapshot_rows(
    book: dict, ts_ms: int,
) -> List[dict]:
    """Encode the recorder's in-memory live book as snapshot rows.

    Used at daily file rotation: the new parquet starts with a
    point-in-time picture of the book so it can be replayed
    independently of the previous day's file.

    book layout: {"buy": {price: qty, ...}, "sell": {price: qty, ...}}.
    Zero-qty levels are skipped — a snapshot lists live levels only.
    """
    rows: List[dict] = []
    for side, levels in book.items():
        for price, qty in levels.items():
            if qty <= 0:
                continue
            rows.append({"timestamp": ts_ms, "side": side,
                         "price": float(price), "qty": float(qty),
                         "is_snapshot": True})
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
    use_trades: bool = True,
) -> None:
    if region not in WS_BASES:
        raise ValueError(f"unknown region {region!r}; pick from {list(WS_BASES)}")
    rest_base = REST_BASES[region]
    rotator = ParquetRotator(out_dir, symbol)
    trade_rotator = ParquetRotator(out_dir, symbol, kind="trades")
    streams = f"{symbol}@depth"
    if use_trades:
        streams += f"/{symbol}@trade"
    stream_url = f"{WS_BASES[region]}?streams={streams}"

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
    tick_count = 0
    crossed_count = 0
    live_book: dict = {"buy": {}, "sell": {}}
    last_known_date = dt.datetime.now(dt.timezone.utc).date()
    last_status_time = loop.time()

    def _apply_to_live_book(rows_to_apply: List[dict]) -> None:
        nonlocal tick_count, crossed_count
        for r in rows_to_apply:
            if r.get("is_snapshot"):
                continue
            side = r["side"]
            price = r["price"]
            qty = r["qty"]
            if qty == 0:
                live_book[side].pop(price, None)
            else:
                live_book[side][price] = qty
            tick_count += 1

        # Crossed-book check after batch
        bids = live_book["buy"]
        asks = live_book["sell"]
        if bids and asks:
            best_bid = max(bids)
            best_ask = min(asks)
            if best_bid >= best_ask:
                crossed_count += 1
                logger.debug(
                    "crossed book: best_bid=%.2f >= best_ask=%.2f",
                    best_bid, best_ask,
                )

    async with aiohttp.ClientSession() as http:
        while not stop.is_set():
            try:
                async with websockets.connect(stream_url) as ws:
                    logger.info("connected to %s", stream_url)
                    if use_bootstrap:
                        last_u = await _bootstrap_session(
                            ws, http, symbol, rest_base, rotator,
                        )
                        _apply_to_live_book(rotator.buffer)
                    else:
                        last_u = -1

                    async for raw in ws:
                        if stop.is_set():
                            break
                        if deadline is not None and loop.time() >= deadline:
                            stop.set()
                            break
                        env = json.loads(raw)
                        stream = env.get("stream", "")
                        payload = env.get("data", {})

                        # Trade stream → separate daily trades parquet.
                        if use_trades and "@trade" in stream:
                            trade_rotator.add(parse_trade_message(payload))
                            if len(trade_rotator.buffer) >= flush_rows:
                                trade_rotator.flush()
                            continue

                        U, u, rows = parse_depth_message(payload)
                        if use_bootstrap and U != -1 and last_u != -1:
                            if U != last_u + 1:
                                raise GapDetected(
                                    f"live sequence gap: expected U="
                                    f"{last_u + 1}, got U={U}"
                                )
                            last_u = u

                        now_loop = loop.time()
                        if now_loop - last_status_time >= STATUS_INTERVAL_SEC:
                            bids = live_book["buy"]
                            asks = live_book["sell"]
                            now_str = dt.datetime.now(
                                dt.timezone.utc
                            ).strftime("%H:%M:%S")
                            crossed_pct = (
                                crossed_count / tick_count * 100
                                if tick_count else 0.0
                            )
                            logger.info(
                                "[%s] seq=%d bids=%d asks=%d "
                                "ticks=%d crossed=%.2f%%",
                                now_str, last_u, len(bids), len(asks),
                                tick_count, crossed_pct,
                            )
                            last_status_time = now_loop

                        today = dt.datetime.now(dt.timezone.utc).date()
                        if today != last_known_date:
                            n = rotator.flush()
                            trade_rotator.flush()
                            total_rows += n
                            now_ms = int(
                                dt.datetime.now(dt.timezone.utc).timestamp()
                                * 1000
                            )
                            seed = book_to_snapshot_rows(live_book, now_ms)
                            rotator.extend(seed)
                            logger.info(
                                "day rollover %s -> %s; seeded %d snapshot "
                                "rows into new file",
                                last_known_date, today, len(seed),
                            )
                            last_known_date = today

                        rotator.extend(rows)
                        _apply_to_live_book(rows)
                        if len(rotator.buffer) >= flush_rows:
                            n = rotator.flush()
                            total_rows += n
                            logger.info(
                                "flushed %d rows (total %d)", n, total_rows,
                            )
            except GapDetected as e:
                logger.warning("sequence gap — %s; flushing and re-bootstrapping in 2s", e)
                rotator.flush()
                await asyncio.sleep(2)
            except (websockets.ConnectionClosed, OSError) as e:
                if stop.is_set():
                    break
                logger.warning("connection dropped: %s; reconnecting in 5s", e)
                await asyncio.sleep(5)

    n = rotator.flush()
    total_rows += n
    n_trades = trade_rotator.flush()
    logger.info("final flush %d depth rows (total %d), %d trade rows",
                n, total_rows, n_trades)


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
    parser.add_argument("--no-trades", action="store_true",
                        help="record depth only; skip the @trade stream")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    duration = args.hours * 3600 if args.hours else None

    try:
        asyncio.run(record(
            args.symbol, args.out, duration, args.flush_rows, args.region,
            use_bootstrap=not args.no_bootstrap,
            use_trades=not args.no_trades,
        ))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
