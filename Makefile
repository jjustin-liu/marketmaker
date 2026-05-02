.PHONY: install test lint format

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	flake8 src tests
	mypy src

format:
	black src tests
