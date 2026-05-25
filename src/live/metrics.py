"""Prometheus metrics for the live market maker.

Exposes a /metrics HTTP endpoint scraped by Prometheus. Metrics:
  mm_pnl                          gauge, USD, mark-to-market PnL
  mm_inventory                    gauge, BTC, signed position
  mm_fills_total{side=...}        counter, lifetime fills by side
  mm_quote_refreshes_total        counter, lifetime quote refresh cycles
  mm_feed_latency_seconds         gauge, now - lob:last_update (lazy)

The feed-latency gauge is published by a custom collector that reads
lob:last_update from Redis at scrape time, so the order manager loop
doesn't pay any extra cost per tick.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Optional

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    start_http_server,
)
from prometheus_client.core import GaugeMetricFamily

logger = logging.getLogger("metrics")

DEFAULT_METRICS_PORT = 8000

PNL = Gauge("mm_pnl", "Mark-to-market PnL in USD")
INVENTORY = Gauge("mm_inventory", "Signed inventory in base asset units")
FILLS = Counter("mm_fills_total", "Fills booked, by side", ["side"])
QUOTE_REFRESHES = Counter(
    "mm_quote_refreshes_total", "Strategy quote refresh cycles",
)


class _FeedLatencyCollector:
    """Reads lob:last_update from Redis on each scrape.

    Returns NaN when the key is missing so Grafana can render a gap
    rather than a fake zero.
    """

    def __init__(self, redis_client: Any, key: str = "lob:last_update") -> None:
        self._redis = redis_client
        self._key = key

    def collect(self) -> Iterable[GaugeMetricFamily]:
        g = GaugeMetricFamily(
            "mm_feed_latency_seconds",
            "Seconds since the data feed last wrote lob:last_update.",
        )
        try:
            raw = self._redis.get(self._key)
        except Exception:  # noqa: BLE001 - scrape must never raise
            raw = None
        if raw is None:
            g.add_metric([], float("nan"))
        else:
            try:
                last = float(raw)
                g.add_metric([], max(0.0, time.time() - last))
            except (TypeError, ValueError):
                g.add_metric([], float("nan"))
        yield g


def install_feed_latency_collector(
    redis_client: Any,
    registry: CollectorRegistry = REGISTRY,
) -> _FeedLatencyCollector:
    """Register the feed-latency collector. Returns the instance for tests."""
    collector = _FeedLatencyCollector(redis_client)
    registry.register(collector)
    return collector


def start_metrics_server(port: int = DEFAULT_METRICS_PORT) -> None:
    """Start the Prometheus HTTP endpoint on the given port."""
    start_http_server(port)
    logger.info("metrics server listening on :%d/metrics", port)


def record_fill(side: str) -> None:
    """Increment the per-side fill counter. side must be 'buy' or 'sell'."""
    FILLS.labels(side=side).inc()


def record_quote_refresh() -> None:
    QUOTE_REFRESHES.inc()


def set_position(inventory: float, pnl: float) -> None:
    INVENTORY.set(inventory)
    PNL.set(pnl)


def reset_for_tests(registry: Optional[CollectorRegistry] = None) -> None:
    """Zero the global metrics. Test-only helper."""
    PNL.set(0.0)
    INVENTORY.set(0.0)
    FILLS.clear()
    QUOTE_REFRESHES._value.set(0)  # type: ignore[attr-defined]
