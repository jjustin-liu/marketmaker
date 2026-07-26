# Monitoring stack

A docker-compose stack with Redis + Prometheus + Grafana for live
visibility into PnL, inventory, fill rate, quote refresh rate, and
data-feed latency.

## What it monitors

The live process (`python -m src.cli live`) exposes a Prometheus
endpoint on `:8000/metrics`:

| Metric                       | Type    | Description                                    |
|------------------------------|---------|------------------------------------------------|
| `mm_pnl`                     | gauge   | Mark-to-market PnL (USD).                      |
| `mm_inventory`               | gauge   | Signed inventory (BTC).                        |
| `mm_fills_total{side}`       | counter | Fills booked, labeled `buy` / `sell`.          |
| `mm_quote_refreshes_total`   | counter | Strategy quote-refresh cycles.                 |
| `mm_feed_latency_seconds`    | gauge   | `now() - lob:last_update`, read on each scrape.|

Feed latency is published by a custom collector that reads
`lob:last_update` from Redis at scrape time, so the order-manager
loop pays no extra cost per tick.

## Running it

```bash
cd docker
docker-compose up -d
```

Services come up on:

- Redis      `localhost:6379`
- Prometheus `http://localhost:9090`
- Grafana    `http://localhost:3000` (login `admin` / `admin`)

The Grafana datasource and the "Market Maker" dashboard are
auto-provisioned on first boot from `docker/grafana/provisioning/`
and `docker/grafana/dashboards/`.

Start the live process from the host (not inside docker) so it can
scrape its own metrics endpoint:

```bash
python -m src.cli live
```

Prometheus scrapes `host.docker.internal:8000` every 5s, defined in
`docker/prometheus.yml`. On Linux, the `extra_hosts` mapping in
`docker-compose.yml` resolves that name to the host gateway; on
Mac/Windows Docker Desktop resolves it natively.

## Configuration

| Env var            | Default | Purpose                          |
|--------------------|---------|----------------------------------|
| `MM_METRICS_PORT`  | 8000    | Port the `/metrics` endpoint binds. |

To use a different port, set `MM_METRICS_PORT` and update the
target in `docker/prometheus.yml` to match.

## Tearing down

```bash
cd docker
docker-compose down          # keeps volumes
docker-compose down -v       # also wipes prometheus + grafana state
```
