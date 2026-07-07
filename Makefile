.PHONY: install test lint format feed paper record

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	flake8 src tests
	mypy src

format:
	black src tests

# Live data feed: WebSocket -> local book -> Redis. Run in its own terminal.
feed:
	python -m scripts.run_feed --forever

# Paper trader (or testnet engine if BINANCE_API_KEY is set). Needs feed.
paper:
	python -m src.cli live

# L2 depth + trade recorder -> data/raw/. Run in its own terminal.
record:
	python -m scripts.run_recorder
