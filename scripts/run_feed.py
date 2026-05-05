"""Run the Binance feed for a fixed duration and dump Redis state.

Smoke test for Phase 3. Requires a running local Redis on :6379.
"""

import argparse
import asyncio
import json
import logging

import redis

from src.data_feed.binance_ws import BinanceFeed
from src.lob import CppSide, MatchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def main(symbol: str, duration: float, region: str) -> None:
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    feed = BinanceFeed(
        symbol=symbol,
        engine_factory=MatchEngine,
        redis_client=r,
        bid_side=CppSide.BUY,
        ask_side=CppSide.SELL,
        region=region,
    )
    print(f"running feed for {duration}s on {symbol} via {region}...")
    await feed.run(stop_after_sec=duration)

    print("\n--- redis state ---")
    for key in [
        "lob:best_bid", "lob:best_ask", "lob:mid", "lob:spread_bps",
        "lob:last_update",
    ]:
        print(f"{key}: {r.get(key)}")
    bids_raw = r.get("lob:levels:bids")
    asks_raw = r.get("lob:levels:asks")
    if bids_raw:
        bids = json.loads(bids_raw)
        print(f"top 3 bids: {bids[:3]}")
    if asks_raw:
        asks = json.loads(asks_raw)
        print(f"top 3 asks: {asks[:3]}")
    trades_raw = r.get("trades:recent")
    if trades_raw:
        trades = json.loads(trades_raw)
        print(f"trade count: {len(trades)}")
        if trades:
            print(f"last trade: {trades[-1]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="btcusdt")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument(
        "--region",
        default="VISION",
        choices=["GLOBAL", "VISION", "US", "TEST"],
    )
    args = p.parse_args()
    asyncio.run(main(args.symbol, args.duration, args.region))
