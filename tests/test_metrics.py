"""Unit tests for src.live.metrics."""

from __future__ import annotations

import time
from typing import Any, Optional

from prometheus_client import CollectorRegistry, generate_latest

from src.live import metrics as mm


class _FakeRedis:
    def __init__(self, value: Optional[str] = None) -> None:
        self.value = value

    def get(self, key: str) -> Optional[str]:  # noqa: ARG002
        return self.value


def _scrape(registry: CollectorRegistry) -> str:
    return generate_latest(registry).decode()


def test_record_fill_and_refresh_increment_counters() -> None:
    mm.reset_for_tests()
    mm.record_fill("buy")
    mm.record_fill("buy")
    mm.record_fill("sell")
    mm.record_quote_refresh()
    mm.record_quote_refresh()
    mm.record_quote_refresh()

    assert mm.FILLS.labels(side="buy")._value.get() == 2.0
    assert mm.FILLS.labels(side="sell")._value.get() == 1.0
    assert mm.QUOTE_REFRESHES._value.get() == 3.0


def test_set_position_updates_gauges() -> None:
    mm.reset_for_tests()
    mm.set_position(inventory=0.005, pnl=12.34)
    assert mm.INVENTORY._value.get() == 0.005
    assert mm.PNL._value.get() == 12.34


def test_feed_latency_collector_emits_seconds_since_last_update() -> None:
    reg = CollectorRegistry()
    past = time.time() - 2.5
    fake = _FakeRedis(str(past))
    reg.register(mm._FeedLatencyCollector(fake))

    out = _scrape(reg)
    line = [ln for ln in out.splitlines()
            if ln.startswith("mm_feed_latency_seconds ")][0]
    value = float(line.split()[1])
    assert 2.0 <= value <= 5.0


def test_feed_latency_collector_handles_missing_key() -> None:
    reg = CollectorRegistry()
    reg.register(mm._FeedLatencyCollector(_FakeRedis(None)))
    out = _scrape(reg)
    assert "mm_feed_latency_seconds NaN" in out


def test_feed_latency_collector_handles_bad_value() -> None:
    reg = CollectorRegistry()
    reg.register(mm._FeedLatencyCollector(_FakeRedis("not-a-number")))
    out = _scrape(reg)
    assert "mm_feed_latency_seconds NaN" in out


def test_feed_latency_collector_swallows_redis_errors() -> None:
    class _Boom:
        def get(self, key: str) -> Any:  # noqa: ARG002
            raise RuntimeError("redis down")

    reg = CollectorRegistry()
    reg.register(mm._FeedLatencyCollector(_Boom()))
    out = _scrape(reg)
    assert "mm_feed_latency_seconds NaN" in out
