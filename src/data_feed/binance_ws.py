"""Binance WebSocket data feed.

Connects to combined depth + trade streams for one symbol, drives the
local order book through bootstrap and ongoing sync, publishes state
to Redis. Reconnects with exponential backoff on any failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Tuple

import aiohttp
import websockets

from src.data_feed.bootstrap import (
    BookSynchronizer,
    DepthDiff,
    GapDetectedError,
    Snapshot,
)
from src.data_feed.redis_writer import RedisWriter

SNAPSHOT_DEPTH = 1000
INITIAL_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 60.0
BACKOFF_MULTIPLIER = 2.0

# (ws_base, rest_url) per region. VISION is the public market-data
# mirror — same streams as GLOBAL but not geo-blocked from the US.
ENDPOINTS: Dict[str, Tuple[str, str]] = {
    "GLOBAL": (
        "wss://stream.binance.com:9443/stream",
        "https://api.binance.com/api/v3/depth",
    ),
    "VISION": (
        "wss://data-stream.binance.vision/stream",
        "https://data-api.binance.vision/api/v3/depth",
    ),
    "US": (
        "wss://stream.binance.us:9443/stream",
        "https://api.binance.us/api/v3/depth",
    ),
    "TEST": (
        "wss://stream.testnet.binance.vision/stream",
        "https://testnet.binance.vision/api/v3/depth",
    ),
}
DEFAULT_REGION = "VISION"

logger = logging.getLogger(__name__)


def parse_depth_diff(payload: Dict[str, Any]) -> DepthDiff:
    """Parse a Binance depthUpdate event into a DepthDiff."""
    return DepthDiff(
        first_update_id=int(payload["U"]),
        last_update_id=int(payload["u"]),
        bid_updates=[(float(p), float(q)) for p, q in payload["b"]],
        ask_updates=[(float(p), float(q)) for p, q in payload["a"]],
    )


def parse_trade(payload: Dict[str, Any]) -> Tuple[float, float, str, int]:
    """Parse a Binance trade event.

    Returns (price, qty, aggressor_side, timestamp_ms).
    Binance's 'm' field is True when the buyer was the maker — meaning
    the aggressor was the seller.
    """
    aggressor = "sell" if payload["m"] else "buy"
    return (
        float(payload["p"]),
        float(payload["q"]),
        aggressor,
        int(payload["T"]),
    )


def parse_snapshot(payload: Dict[str, Any]) -> Snapshot:
    """Parse a REST depth snapshot response."""
    return Snapshot(
        last_update_id=int(payload["lastUpdateId"]),
        bids=[(float(p), float(q)) for p, q in payload["bids"]],
        asks=[(float(p), float(q)) for p, q in payload["asks"]],
    )


class BinanceFeed:
    """One symbol, depth + trade streams, full bootstrap and reconnect."""

    def __init__(
        self,
        symbol: str,
        engine_factory: Callable[[], Any],
        redis_client: Any,
        *,
        bid_side: Any,
        ask_side: Any,
        region: str = DEFAULT_REGION,
    ) -> None:
        if region not in ENDPOINTS:
            raise ValueError(
                f"unknown region {region!r}; pick from {list(ENDPOINTS)}"
            )
        self._symbol = symbol.lower()
        self._engine_factory = engine_factory
        self._redis = redis_client
        self._BID = bid_side
        self._ASK = ask_side
        self._ws_base, self._rest_url = ENDPOINTS[region]
        # Latest engine + writer; replaced on each restart.
        self.engine: Any = None
        self.writer: RedisWriter | None = None

    async def run(self, stop_after_sec: float | None = None) -> None:
        """Run the feed until cancelled or stop_after_sec elapses."""
        backoff = INITIAL_BACKOFF_SEC
        deadline = (
            asyncio.get_event_loop().time() + stop_after_sec
            if stop_after_sec
            else None
        )
        while True:
            if deadline and asyncio.get_event_loop().time() >= deadline:
                return
            try:
                await self._run_once(deadline)
                return
            except (GapDetectedError, ConnectionError, asyncio.TimeoutError) as e:
                logger.warning("feed restart: %s; backoff=%.1fs", e, backoff)
            except Exception as e:
                logger.exception("feed crashed: %s; backoff=%.1fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SEC)

    async def _run_once(self, deadline: float | None) -> None:
        """One full bootstrap-and-consume cycle. Returns on deadline."""
        self.engine = self._engine_factory()
        self.writer = RedisWriter(
            self._redis, self.engine,
            bid_side=self._BID, ask_side=self._ASK,
        )
        sync = BookSynchronizer(
            self.engine, bid_side=self._BID, ask_side=self._ASK,
        )

        stream_url = (
            f"{self._ws_base}?streams="
            f"{self._symbol}@depth/{self._symbol}@trade"
        )
        snapshot_fetch: asyncio.Task[Snapshot] | None = None

        async with websockets.connect(stream_url) as ws:
            snapshot_fetch = asyncio.create_task(self._fetch_snapshot())

            async for raw in ws:
                if deadline and asyncio.get_event_loop().time() >= deadline:
                    return
                envelope = json.loads(raw)
                stream = envelope.get("stream", "")
                payload = envelope.get("data", {})

                if stream.endswith("@depth"):
                    diff = parse_depth_diff(payload)
                    if not sync.synced:
                        sync.buffer(diff)
                        if snapshot_fetch.done():
                            snapshot = snapshot_fetch.result()
                            sync.apply_snapshot(snapshot)  # may raise GapDetected
                            self.writer.publish_book()
                    else:
                        sync.apply(diff)  # may raise GapDetected
                        self.writer.publish_book()
                elif stream.endswith("@trade"):
                    price, qty, side, ts = parse_trade(payload)
                    self.writer.record_trade(price, qty, side, ts)

    async def _fetch_snapshot(self) -> Snapshot:
        params = {"symbol": self._symbol.upper(), "limit": str(SNAPSHOT_DEPTH)}
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(self._rest_url, params=params) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        return parse_snapshot(payload)
