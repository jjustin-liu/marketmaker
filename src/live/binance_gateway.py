"""Binance Spot REST gateway for live order management.

Async aiohttp client targeting testnet.binance.vision by default.
HMAC-SHA256 signed endpoints for order placement, cancellation,
open-order lookup, and account-trade polling.

Constructed via async context manager so the underlying ClientSession
is closed on exit.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

TESTNET_BASE_URL = "https://testnet.binance.vision"
MAINNET_BASE_URL = "https://api.binance.com"
DEFAULT_RECV_WINDOW_MS = 5000


class BinanceGatewayError(RuntimeError):
    """Raised when the API returns a non-2xx status."""


class BinanceGateway:
    """Minimal Binance Spot REST client. Async, signed."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        testnet: bool = True,
        recv_window_ms: int = DEFAULT_RECV_WINDOW_MS,
    ) -> None:
        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret are required")
        self._api_key = api_key
        self._api_secret = api_secret.encode("utf-8")
        self._base_url = TESTNET_BASE_URL if testnet else MAINNET_BASE_URL
        self._recv_window_ms = recv_window_ms
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "BinanceGateway":
        self._session = aiohttp.ClientSession(
            headers={"X-MBX-APIKEY": self._api_key},
        )
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ---- signing ----

    def _sign(self, query: str) -> str:
        return hmac.new(
            self._api_secret, query.encode("utf-8"), hashlib.sha256,
        ).hexdigest()

    def _signed_query(self, params: Dict[str, Any]) -> str:
        params = {**params, "timestamp": int(time.time() * 1000),
                  "recvWindow": self._recv_window_ms}
        query = urllib.parse.urlencode(params)
        return f"{query}&signature={self._sign(query)}"

    # ---- request helper ----

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        if self._session is None:
            raise RuntimeError("gateway used outside async context manager")
        params = params or {}
        if signed:
            query = self._signed_query(params)
            url = f"{self._base_url}{path}?{query}"
            req_params: Optional[Dict[str, Any]] = None
        else:
            url = f"{self._base_url}{path}"
            req_params = params
        async with self._session.request(method, url, params=req_params) as resp:
            body = await resp.json(content_type=None)
            if resp.status >= 400:
                raise BinanceGatewayError(
                    f"{method} {path} -> {resp.status}: {body}"
                )
            return body

    # ---- endpoints ----

    async def post_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
    ) -> Dict[str, Any]:
        """Place a new order. Returns Binance's order ack."""
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": f"{quantity:.6f}",
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("price required for LIMIT order")
            params["price"] = f"{price:.2f}"
            params["timeInForce"] = time_in_force
        return await self._request("POST", "/api/v3/order",
                                   params=params, signed=True)

    async def cancel_order(
        self,
        symbol: str,
        *,
        order_id: int,
    ) -> Dict[str, Any]:
        """Cancel an order by exchange order id."""
        return await self._request(
            "DELETE", "/api/v3/order",
            params={"symbol": symbol.upper(), "orderId": order_id},
            signed=True,
        )

    async def get_open_orders(
        self, symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params = {"symbol": symbol.upper()} if symbol else {}
        return await self._request("GET", "/api/v3/openOrders",
                                   params=params, signed=True)

    async def get_account_trades(
        self,
        symbol: str,
        *,
        from_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch this account's recent fills for a symbol.

        Pass from_id to page forward; Binance returns trades with id >= from_id.
        """
        params: Dict[str, Any] = {"symbol": symbol.upper(), "limit": limit}
        if from_id is not None:
            params["fromId"] = from_id
        return await self._request("GET", "/api/v3/myTrades",
                                   params=params, signed=True)
