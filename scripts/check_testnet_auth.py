"""One-shot sanity check that BINANCE_API_KEY/SECRET work against testnet.

Usage:
  source .env.local
  python -m scripts.check_testnet_auth

Prints 'open orders: []' on success. Any other output is a problem.
"""

from __future__ import annotations

import asyncio
import os
import sys

from src.live.binance_gateway import BinanceGateway, BinanceGatewayError


async def _main() -> int:
    key = os.environ.get("BINANCE_API_KEY")
    secret = os.environ.get("BINANCE_API_SECRET")
    testnet = os.environ.get("BINANCE_TESTNET", "1") not in ("0", "false", "")
    if not key or not secret:
        print("BINANCE_API_KEY / BINANCE_API_SECRET not set. "
              "Did you `source .env.local`?", file=sys.stderr)
        return 2
    try:
        async with BinanceGateway(key, secret, testnet=testnet) as g:
            orders = await g.get_open_orders("BTCUSDT")
            print("auth ok. open orders:", orders)
            return 0
    except BinanceGatewayError as exc:
        print(f"auth failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
