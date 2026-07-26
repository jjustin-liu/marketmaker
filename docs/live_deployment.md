# 24/7 live / paper deployment

Continuous operation is run as three auto-restarting containers: Redis,
the feed (Binance WebSocket → Redis book), and the paper trader (quoting
loop + Prometheus metrics). The full monitoring stack (Prometheus +
Grafana) layers on top.

## Run it

```bash
cd docker
# monitoring stack + live stack together
docker compose -f docker-compose.yml -f docker-compose.live.yml up -d --build
```

- `feed` runs `run_feed --forever`; `paper` runs `src.cli live` at
  `MM_FEE_BPS=10`, serving metrics on `:8000`.
- Both services set `restart: unless-stopped`, so a crash or host reboot
  brings them back automatically — the basis of unattended 24/7 running.
- Grafana at `:3000` shows PnL, inventory, fills, and feed latency; the
  latency collector scrapes `lob:last_update` from Redis every 5 s and
  goes `NaN` if the feed stalls, so a dead feed is visible on the board.

## Health and recovery

- **Auto-restart:** container-level (`unless-stopped`). For a bare-metal
  deploy, an equivalent `systemd` unit with `Restart=always` works.
- **Risk guard:** the trading loop trips a halt on repeated strategy
  exceptions or breached risk limits (see `test_testnet_engine.py`),
  rather than looping on a bad state.
- **Feed staleness:** surfaced as a metric; alert on the latency gauge.

## Honest scope

- **Paper mode is the validated 24/7 target.** Phase 10 confirmed the
  paper loop produces realistic queue-aware fills against live data
  (4 fills in 90 s). Real-money `testnet`/live is gated behind an API
  key and is **not** claimed as validated here.
- **Geo-block caveat:** Binance blocks some regions; the 1-hour live
  acceptance in Phase 8 was blocked for this reason. A VPS in a
  supported region is required for an actual continuous live run — the
  deployment mechanics above are region-agnostic.
- Claim scope: "24/7 live trading" should be stated as **"24/7
  paper-trading loop against live market data, containerized with
  auto-restart and monitoring."** That is what is built and validated.
