"""Run the Binance feed: WebSocket -> local book -> Redis.

Two modes:
  --duration N   run N seconds, then dump Redis state (smoke test)
  --forever      run until Ctrl-C; feeds the live paper trader

Requires a running local Redis on :6379.
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Optional

import redis

from src.data_feed.binance_ws import BinanceFeed
from src.lob import CppSide, MatchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def main(symbol: str, duration: Optional[float], region: str) -> None:
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    feed = BinanceFeed(
        symbol=symbol,
        engine_factory=MatchEngine,
        redis_client=r,
        bid_side=CppSide.BUY,
        ask_side=CppSide.SELL,
        region=region,
    )
    if duration is None:
        print(f"running feed on {symbol} via {region} until Ctrl-C...")
    else:
        print(f"running feed for {duration}s on {symbol} via {region}...")
    await feed.run(stop_after_sec=duration)

    if duration is None:
        return

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
    p.add_argument("--forever", action="store_true",
                   help="run until Ctrl-C (ignores --duration)")
    p.add_argument(
        "--region",
        default="VISION",
        choices=["GLOBAL", "VISION", "US", "TEST"],
    )
    args = p.parse_args()
    run_duration = None if args.forever else args.duration
    try:
        asyncio.run(main(args.symbol, run_duration, args.region))
    except KeyboardInterrupt:
        sys.exit(0)
