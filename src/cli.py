"""Unified entry point: `python -m src.cli {backtest,live}`.

Environment variables (live mode):
  REDIS_HOST           default 'localhost'
  REDIS_PORT           default 6379
  MM_MAX_POSITION      default 0.01 (BTC)
  MM_MAX_DRAWDOWN      default 50   (USD)
  MM_POLL_SECONDS      default 0.1
  MM_FILL_MODEL_PATH   default data/models/fill_prob.joblib
  MM_SYMBOL            default 'BTCUSDT' (used for testnet orders)
  MM_METRICS_PORT      default 8000. Prometheus /metrics endpoint port
  BINANCE_API_KEY      optional; if set with BINANCE_API_SECRET,
                       runs against testnet.binance.vision instead
                       of the paper simulator
  BINANCE_API_SECRET   pair with BINANCE_API_KEY
  BINANCE_TESTNET      default '1'. Set to '0' to hit mainnet
                       (do not do this without understanding the
                       consequences)
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("cli")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marketmaker")
    sub = parser.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="run backtest on recorded parquet")
    bt.add_argument("--input", type=Path, required=True)
    bt.add_argument("--out", type=Path,
                    default=Path("data/backtest_results.csv"))
    bt.add_argument("--fee-bps", type=float, default=0.0)

    live = sub.add_parser("live",
                          help="run paper-mode order manager against Redis")
    live.add_argument("--duration", type=float, default=None,
                      help="stop after N seconds (default: run forever)")
    return parser


def _run_backtest(args: argparse.Namespace) -> int:
    from scripts.run_backtest import main as backtest_main
    sys.argv = [
        "run_backtest",
        "--input", str(args.input),
        "--out", str(args.out),
        "--fee-bps", str(args.fee_bps),
    ]
    backtest_main()
    return 0


def _build_strategy(model_path: Path) -> Any:
    from src.models.fill_prob import FillProbabilityModel
    from src.strategy.ev_maker import EVMaker
    from src.strategy.inventory_skew import InventorySkew
    from src.strategy.size_calculator import SizeCalculator

    fill_model = FillProbabilityModel.load(model_path)
    logger.info("loaded fill model from %s", model_path)
    return EVMaker(
        inventory_skew=InventorySkew(),
        size_calculator=SizeCalculator(),
        fill_model=fill_model,
    )


def _install_signal_trip(risk: Any) -> None:
    def _on_signal(signum: int, _frame: Optional[object]) -> None:
        logger.info("received signal %d, shutting down", signum)
        risk.trip("ctrl-c")
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)


def _run_live(args: argparse.Namespace) -> int:
    import redis

    from src.live.metrics import (
        DEFAULT_METRICS_PORT,
        install_feed_latency_collector,
        start_metrics_server,
    )
    from src.live.order_manager import OrderManager
    from src.live.risk_guard import RiskConfig, RiskGuard

    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    max_position = float(os.environ.get("MM_MAX_POSITION", "0.01"))
    max_drawdown = float(os.environ.get("MM_MAX_DRAWDOWN", "50"))
    poll = float(os.environ.get("MM_POLL_SECONDS", "0.1"))
    model_path = Path(os.environ.get(
        "MM_FILL_MODEL_PATH", "data/models/fill_prob.joblib",
    ))
    symbol = os.environ.get("MM_SYMBOL", "BTCUSDT")
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    testnet = os.environ.get("BINANCE_TESTNET", "1") not in ("0", "false", "")
    metrics_port = int(os.environ.get("MM_METRICS_PORT",
                                      str(DEFAULT_METRICS_PORT)))

    r = redis.Redis(host=host, port=port, decode_responses=True)
    r.ping()
    logger.info("connected to redis at %s:%d", host, port)

    install_feed_latency_collector(r)
    start_metrics_server(metrics_port)

    strategy = _build_strategy(model_path)
    risk = RiskGuard(RiskConfig(
        max_position=max_position, max_drawdown=max_drawdown,
    ))
    _install_signal_trip(risk)

    if api_key and api_secret:
        logger.info("BINANCE_API_KEY present -> testnet=%s engine", testnet)
        return _run_testnet(args, r, strategy, risk, symbol,
                            api_key, api_secret, testnet)

    logger.info("no Binance credentials -> paper simulator")
    manager = OrderManager(
        redis_client=r, strategy=strategy, risk_guard=risk,
        poll_seconds=poll,
    )
    manager.run(max_seconds=args.duration)
    logger.info("manager exited. final inventory=%.6f pnl=%.4f fills=%d",
                manager.state.inventory, manager.state.pnl,
                len(manager.state.fills))
    return 0 if not risk.halted or risk.halt_reason == "ctrl-c" else 1


def _run_testnet(args: argparse.Namespace, redis_client: Any,
                 strategy: Any, risk: Any, symbol: str,
                 api_key: str, api_secret: str, testnet: bool) -> int:
    import asyncio

    from src.live.binance_gateway import BinanceGateway
    from src.live.testnet_engine import TestnetEngine

    async def _amain() -> int:
        async with BinanceGateway(api_key, api_secret,
                                  testnet=testnet) as gateway:
            engine = TestnetEngine(
                redis_client=redis_client, gateway=gateway,
                strategy=strategy, risk_guard=risk, symbol=symbol,
            )
            await engine.run(max_seconds=args.duration)
            logger.info("engine exited. final inventory=%.6f pnl=%.4f fills=%d",
                        engine.state.inventory, engine.state.pnl,
                        len(engine.state.fills))
            return 0 if not risk.halted or risk.halt_reason == "ctrl-c" else 1

    return asyncio.run(_amain())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.cmd == "backtest":
        sys.exit(_run_backtest(args))
    elif args.cmd == "live":
        sys.exit(_run_live(args))


if __name__ == "__main__":
    main()
