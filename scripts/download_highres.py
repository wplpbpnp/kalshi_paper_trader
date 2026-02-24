#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data.kalshi_http import KalshiHttpClient
from market_data.kalshi_ws import KalshiWsConfig
from market_data.kalshi_ws_scrape import _run_for_market, _select_market
from pipeline.runtime_config import cfg_get, cfg_get_path, load_runtime_config


def _find_market_meta(
    http: KalshiHttpClient, *, series: str, market_ticker: str
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    resp = http.get(
        "/trade-api/v2/markets",
        params={"series_ticker": series, "status": "open", "limit": 200},
    )
    for m in resp.get("markets", []):
        if m.get("ticker") == market_ticker:
            return m.get("close_time"), m.get("floor_strike"), m.get("cap_strike")
    return None, None, None


async def _main_async(args: argparse.Namespace) -> int:
    if not args.pem_file:
        raise RuntimeError("Missing --pem-file (or pem_file in config/runtime.json).")
    os.makedirs(args.out_dir, exist_ok=True)
    http = KalshiHttpClient.from_files(
        base_url=args.base_url,
        env_file=args.env_file if args.env_file and os.path.exists(args.env_file) else None,
        pem_file=args.pem_file,
        timeout_s=10.0,
    )
    ws_cfg = KalshiWsConfig(ws_url=args.ws_url)

    if args.market_ticker:
        ticker = args.market_ticker
        close_time_iso, floor_strike, cap_strike = _find_market_meta(
            http, series=args.series, market_ticker=ticker
        )
    else:
        ticker, close_time_iso, floor_strike, cap_strike = _select_market(http, args.series)

    print(f"collecting market={ticker} snapshot_ms={args.snapshot_ms} out_dir={args.out_dir}", flush=True)

    stop_flag = {"flag": False}

    def _stop(_sig, _frame):
        stop_flag["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    auto_stop_task = None
    if args.duration_min > 0:
        duration_s = float(args.duration_min) * 60.0

        async def _auto_stop():
            await asyncio.sleep(duration_s)
            stop_flag["flag"] = True

        auto_stop_task = asyncio.create_task(_auto_stop())

    try:
        await _run_for_market(
            http=http,
            ws_cfg=ws_cfg,
            market_ticker=ticker,
            close_time_iso=close_time_iso,
            floor_strike=floor_strike,
            cap_strike=cap_strike,
            out_dir=args.out_dir,
            gzip_level=(None if args.gzip == 0 else max(1, min(9, int(args.gzip)))),
            snapshot_ms=max(10, int(args.snapshot_ms)),
            include_raw=bool(args.include_raw),
            auth=(not args.no_auth),
            stop_flag=stop_flag,
            heartbeat_ms=max(0, int(args.heartbeat_ms)),
            book_depth=max(0, int(args.book_depth)),
            trade_lines=max(0, int(args.trade_lines)),
        )
    finally:
        if auto_stop_task is not None:
            auto_stop_task.cancel()
            try:
                await auto_stop_task
            except asyncio.CancelledError:
                pass
    return 0


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config/runtime.json")
    pre_args, _ = pre.parse_known_args()
    cfg = load_runtime_config(str(pre_args.config))

    ap = argparse.ArgumentParser(description="Download high-resolution Kalshi snapshots (unified canonical format).")
    ap.add_argument("--config", default=str(pre_args.config), help="Runtime config JSON path.")
    ap.add_argument("--series", default=cfg_get(cfg, "series", "KXBTC15M"))
    ap.add_argument("--market-ticker", default="", help="Optional fixed market ticker (else auto-picks active open market).")
    ap.add_argument("--base-url", default=cfg_get(cfg, "base_url", "https://api.elections.kalshi.com"))
    ap.add_argument("--ws-url", default=cfg_get(cfg, "ws_url", "wss://api.elections.kalshi.com/trade-api/ws/v2"))
    ap.add_argument("--env-file", default=cfg_get_path(cfg, "env_file", ROOT))
    ap.add_argument("--pem-file", default=cfg_get_path(cfg, "pem_file", ROOT))
    ap.add_argument("--out-dir", default=cfg_get_path(cfg, "highres_dir", ROOT, "pipeline_data/highres"))
    ap.add_argument("--snapshot-ms", type=int, default=100)
    ap.add_argument("--duration-min", type=float, default=0.0, help="0 means run until close/interrupt.")
    ap.add_argument("--gzip", type=int, default=6, help="0 disables gzip")
    ap.add_argument("--include-raw", action="store_true")
    ap.add_argument("--heartbeat-ms", type=int, default=1000)
    ap.add_argument("--book-depth", type=int, default=3)
    ap.add_argument("--trade-lines", type=int, default=5)
    ap.add_argument("--no-auth", action="store_true")
    args = ap.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
