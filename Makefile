.PHONY: install test lint format feed paper record bench backtest-ci scale

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	flake8 src tests
	mypy src

format:
	black src tests

# Engine throughput/latency benchmark (needs a depth parquet).
bench:
	python -m scripts.benchmark_engine \
	  --input data/raw/btcusdt_depth_2026-07-08.parquet --events 1000000

# CI backtest: generate synthetic L2, run the sim, render a P&L report.
backtest-ci:
	python -m scripts.gen_synthetic_l2 --rows 200000 --epochs 20 \
	  --out data/synthetic/depth.parquet --trades data/synthetic/trades.parquet
	python -m scripts.run_backtest --input data/synthetic/depth.parquet \
	  --trades data/synthetic/trades.parquet --fee-bps 10 \
	  --out reports/backtest_results.csv
	python -m scripts.pnl_report --input reports/backtest_results.csv \
	  --out reports/pnl.md --title "CI backtest P&L (synthetic L2, 10bps)"

# Shard-parallel scale backtest over all captured shards.
scale:
	python -m scripts.run_scale_backtest --glob "data/raw/*_depth_*.parquet"

# Live data feed: WebSocket -> local book -> Redis. Run in its own terminal.
feed:
	python -m scripts.run_feed --forever

# Paper trader (or testnet engine if BINANCE_API_KEY is set). Needs feed.
paper:
	python -m src.cli live

# L2 depth + trade recorder -> data/raw/. Run in its own terminal.
record:
	python -m scripts.run_recorder
