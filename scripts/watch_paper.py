"""Live paper-trading watcher — quotes, position, P&L, recent fills.

Reads from Redis (the paper trader publishes there) and tails
data/fills.log. Refresh every 0.5s. Terminal-only, no Grafana needed.

Usage:
  python scripts/watch_paper.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import redis

REFRESH_SECONDS = 0.5
FILLS_LOG = Path("data/fills.log")


def _fmt_quote(raw: str | None) -> str:
    if not raw:
        return "(none)"
    try:
        q = json.loads(raw)
        return f"${q['price']:.2f} @ {q['size']:.6f}"
    except (json.JSONDecodeError, KeyError, TypeError):
        return raw


def _color(val: float) -> str:
    if val > 0:
        return f"\033[32m{val:+.4f}\033[0m"
    if val < 0:
        return f"\033[31m{val:+.4f}\033[0m"
    return f" {val:+.4f}"


def main() -> None:
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    last_fill_size = 0
    recent: deque = deque(maxlen=8)

    while True:
        sys.stdout.write("\033[2J\033[H")  # clear + home cursor

        bb = r.get("lob:best_bid")
        ba = r.get("lob:best_ask")
        mid = r.get("lob:mid")
        last_update = r.get("lob:last_update")
        tgt_bid = r.get("strategy:target_bid")
        tgt_ask = r.get("strategy:target_ask")
        inv_raw = r.get("position:inventory")
        pnl_raw = r.get("position:pnl")

        inv = float(inv_raw) if inv_raw else 0.0
        pnl = float(pnl_raw) if pnl_raw else 0.0

        if last_update:
            feed_age = time.time() - float(last_update)
            feed_status = (
                f"\033[32m{feed_age:.2f}s\033[0m" if feed_age < 2.0
                else f"\033[33m{feed_age:.2f}s\033[0m"
                if feed_age < 10.0
                else f"\033[31mSTALE {feed_age:.0f}s\033[0m"
            )
        else:
            feed_status = "\033[31mNO DATA\033[0m"

        print("\033[1m=== marketmaker paper trader ===\033[0m")
        print()
        print(f"  feed:        {feed_status}")
        print(f"  market mid:  ${float(mid):.2f}" if mid else "  market mid:  -")
        print(f"  best bid:    ${float(bb):.2f}" if bb else "  best bid:    -")
        print(f"  best ask:    ${float(ba):.2f}" if ba else "  best ask:    -")
        print()
        print(f"  target bid:  {_fmt_quote(tgt_bid)}")
        print(f"  target ask:  {_fmt_quote(tgt_ask)}")
        if bb and ba and tgt_bid and tgt_ask:
            try:
                tb = json.loads(tgt_bid)["price"]
                ta = json.loads(tgt_ask)["price"]
                mid_f = (float(bb) + float(ba)) / 2.0
                print(f"  bid distance from mid: ${mid_f - tb:.2f}")
                print(f"  ask distance from mid: ${ta - mid_f:.2f}")
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        print()
        print(f"  inventory:   {_color(inv)} BTC")
        print(f"  pnl:         {_color(pnl)} USD")

        if FILLS_LOG.exists():
            current_size = FILLS_LOG.stat().st_size
            if current_size != last_fill_size:
                with FILLS_LOG.open() as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                            tag = f"{rec['side']:>4} {rec['size']:.6f} @ ${rec['price']:.2f}"
                            if rec not in recent:
                                recent.append(rec)
                        except (json.JSONDecodeError, KeyError):
                            continue
                last_fill_size = current_size

        print()
        print(f"\033[1m  recent fills ({len(recent)}):\033[0m")
        for rec in list(recent)[-8:]:
            side_color = "\033[32m" if rec["side"] == "buy" else "\033[31m"
            ts = time.strftime("%H:%M:%S", time.localtime(rec["ts"]))
            print(f"    {ts}  {side_color}{rec['side']:>4}\033[0m "
                  f"{rec['size']:.6f} @ ${rec['price']:.2f}  "
                  f"inv→{rec['inventory_after']:+.6f}")
        if not recent:
            print("    (none yet)")

        print()
        print(f"  refresh {REFRESH_SECONDS}s — Ctrl-C to quit")

        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write("\033[2J\033[H")
        print("watcher exited")
