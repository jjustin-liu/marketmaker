# App image for the market maker: builds the C++ engine and installs the
# Python package. Used by the feed and paper-trading services for
# continuous (24/7) operation under docker-compose with auto-restart.
FROM python:3.11-slim

# Build toolchain for the pybind11 C++ extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml setup.py ./
COPY src ./src
COPY scripts ./scripts

RUN python -m pip install --upgrade pip && pip install -e .

# Default: paper loop. Override `command:` per service in compose.
ENV MM_METRICS_PORT=8000
CMD ["python", "-m", "src.cli", "live"]
