"""Tests for the Binance Spot REST gateway.

Unit-level only: signing, query construction, error mapping.
No live HTTP calls. The async transport is exercised indirectly
in tests/test_testnet_engine.py via a fake gateway.
"""

from __future__ import annotations

import hashlib
import hmac
import urllib.parse

import pytest

from src.live.binance_gateway import (
    BinanceGateway,
    BinanceGatewayError,
    MAINNET_BASE_URL,
    TESTNET_BASE_URL,
)


def test_constructor_requires_credentials() -> None:
    with pytest.raises(ValueError):
        BinanceGateway("", "secret")
    with pytest.raises(ValueError):
        BinanceGateway("key", "")


def test_default_base_url_is_testnet() -> None:
    g = BinanceGateway("k", "s")
    assert g._base_url == TESTNET_BASE_URL


def test_mainnet_base_url_when_testnet_false() -> None:
    g = BinanceGateway("k", "s", testnet=False)
    assert g._base_url == MAINNET_BASE_URL


def test_sign_matches_hmac_sha256() -> None:
    g = BinanceGateway("k", "secret_value")
    payload = "symbol=BTCUSDT&side=BUY"
    expected = hmac.new(
        b"secret_value", payload.encode(), hashlib.sha256,
    ).hexdigest()
    assert g._sign(payload) == expected


def test_signed_query_includes_timestamp_and_signature() -> None:
    g = BinanceGateway("k", "s")
    query = g._signed_query({"symbol": "BTCUSDT"})
    parts = dict(urllib.parse.parse_qsl(query))
    assert "timestamp" in parts
    assert "recvWindow" in parts
    assert "signature" in parts
    assert parts["symbol"] == "BTCUSDT"


def test_post_order_limit_requires_price() -> None:
    import asyncio
    g = BinanceGateway("k", "s")
    with pytest.raises(ValueError):
        asyncio.run(g.post_order(
            symbol="BTCUSDT", side="BUY", order_type="LIMIT",
            quantity=0.001,
        ))


def test_request_without_session_raises() -> None:
    import asyncio
    g = BinanceGateway("k", "s")
    with pytest.raises(RuntimeError):
        asyncio.run(g._request("GET", "/api/v3/time"))


def test_gateway_error_type() -> None:
    assert issubclass(BinanceGatewayError, RuntimeError)
