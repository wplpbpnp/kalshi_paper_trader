#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data.kalshi_http import KalshiHttpClient
from market_data.selection import pick_active_open_market
from pipeline.runtime_config import cfg_get, cfg_get_path, load_runtime_config
from pipeline.schemas import SnapshotRecord
from pipeline.strategy_runtime import OrderIntent, RuntimeState, build_engine, load_strategy_spec


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _snapshot_from_market(m: Dict[str, Any]) -> Optional[SnapshotRecord]:
    ticker = m.get("ticker")
    if not ticker:
        return None
    yes_bid = _to_int(m.get("yes_bid"))
    yes_ask = _to_int(m.get("yes_ask"))
    no_bid = _to_int(m.get("no_bid"))
    no_ask = _to_int(m.get("no_ask"))
    if no_bid is None and yes_ask is not None:
        no_bid = 100 - yes_ask
    if no_ask is None and yes_bid is not None:
        no_ask = 100 - yes_bid
    yes_mid = None
    if yes_bid is not None and yes_ask is not None:
        yes_mid = 0.5 * (yes_bid + yes_ask)
    spread = None
    if yes_bid is not None and yes_ask is not None:
        spread = yes_ask - yes_bid
    return SnapshotRecord(
        ts_ms=int(time.time() * 1000),
        market_ticker=str(ticker),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        yes_mid=yes_mid,
        spread=spread,
    )


def _seconds_to_close(iso_ts: Optional[str]) -> Optional[float]:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _place_order(
    http: KalshiHttpClient,
    *,
    ticker: str,
    intent: OrderIntent,
    dry_run: bool,
) -> Dict[str, Any]:
    payload = {
        "ticker": ticker,
        "action": "buy",
        "side": intent.side,
        "count": int(intent.count),
        "type": "limit",
        "client_order_id": str(uuid.uuid4()),
    }
    if intent.side == "yes":
        payload["yes_price"] = int(intent.limit_price_cents)
    else:
        payload["no_price"] = int(intent.limit_price_cents)

    if dry_run:
        return {"dry_run": True, "payload": payload}

    return http.post("/trade-api/v2/portfolio/orders", json_body=payload)


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config/runtime.json")
    pre_args, _ = pre.parse_known_args()
    cfg = load_runtime_config(str(pre_args.config))

    ap = argparse.ArgumentParser(description="Generic strategy executor that reads a strategy file and trades it.")
    ap.add_argument("--config", default=str(pre_args.config), help="Runtime config JSON path.")
    ap.add_argument("--strategy", required=True, help="Path to strategy.json")
    ap.add_argument("--env-file", default=cfg_get_path(cfg, "env_file", ROOT))
    ap.add_argument("--pem-file", default=cfg_get_path(cfg, "pem_file", ROOT))
    ap.add_argument("--base-url", default=cfg_get(cfg, "base_url", "https://api.elections.kalshi.com"))
    ap.add_argument("--series", default="", help="Overrides strategy markets.series_ticker")
    ap.add_argument("--poll-seconds", type=float, default=float(cfg_get(cfg, "poll_seconds", 1.0)))
    ap.add_argument("--hours", type=float, default=0.0, help="0 means run until interrupted")
    ap.add_argument("--live", action="store_true", help="Actually submit orders. Default is dry-run.")
    args = ap.parse_args()

    spec = load_strategy_spec(args.strategy)
    series = args.series or str(spec.markets.get("series_ticker", cfg_get(cfg, "series", "KXBTC15M")))
    dry_run = not args.live

    engine = build_engine(args.strategy)
    if not args.pem_file:
        raise RuntimeError("Missing --pem-file (or pem_file in config/runtime.json).")
    http = KalshiHttpClient.from_files(
        base_url=args.base_url,
        env_file=args.env_file if args.env_file else None,
        pem_file=args.pem_file,
        timeout_s=10.0,
    )

    if not dry_run:
        print("LIVE mode enabled. Type CONFIRM to continue:")
        if input().strip() != "CONFIRM":
            print("aborted")
            return 1

    started = time.time()
    max_seconds = max(0.0, float(args.hours) * 3600.0)
    position_side = "flat"
    current_ticker = ""

    print(
        json.dumps(
            {
                "mode": ("dry_run" if dry_run else "live"),
                "series": series,
                "strategy_id": spec.strategy_id,
                "strategy_type": spec.strategy_type,
                "poll_seconds": args.poll_seconds,
            },
            indent=2,
        ),
        flush=True,
    )

    while True:
        if max_seconds > 0 and (time.time() - started) >= max_seconds:
            break

        try:
            resp = http.get(
                "/trade-api/v2/markets",
                params={"series_ticker": series, "status": "open", "limit": 200},
            )
            sel = pick_active_open_market(resp.get("markets", []))
            if sel is None:
                time.sleep(max(0.2, args.poll_seconds))
                continue

            m = sel.market
            ticker = str(sel.ticker)
            if ticker != current_ticker:
                current_ticker = ticker
                position_side = "flat"
                engine.on_new_market(ticker)
                print(f"new market: {ticker}", flush=True)

            snapshot = _snapshot_from_market(m)
            if snapshot is None:
                time.sleep(max(0.2, args.poll_seconds))
                continue

            state = RuntimeState(
                market_ticker=ticker,
                position_side=position_side,
                seconds_to_close=_seconds_to_close(m.get("close_time")),
                now_ts=time.time(),
            )
            intents = engine.on_snapshot(snapshot, state)
            for intent in intents:
                if position_side != "flat":
                    continue
                res = _place_order(http, ticker=ticker, intent=intent, dry_run=dry_run)
                position_side = intent.side
                print(
                    json.dumps(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "ticker": ticker,
                            "intent": {
                                "side": intent.side,
                                "count": intent.count,
                                "limit_price_cents": intent.limit_price_cents,
                                "reason": intent.reason,
                            },
                            "result": res,
                        }
                    ),
                    flush=True,
                )
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"loop_error={e}", flush=True)

        time.sleep(max(0.2, args.poll_seconds))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
