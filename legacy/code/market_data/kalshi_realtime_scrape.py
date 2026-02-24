"""
Real-time Kalshi market snapshot logger (finer-than-1m).

Default behavior: poll the current (nearest-to-expiry) OPEN market in a series
and write JSONL snapshots to disk. For KXBTC15M this gives sub-minute quote
resolution (bounded by polling interval + API limits).
"""

import argparse
import gzip
import json
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TextIO

from .kalshi_http import KalshiHttpClient


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso_z(ts: str) -> datetime:
    # Kalshi uses e.g. "2026-01-24T21:45:00Z"
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _open_jsonl(path: str, gzip_level: Optional[int]) -> TextIO:
    if gzip_level is None:
        return open(path, "a", buffering=1)
    return gzip.open(path, "at", compresslevel=gzip_level)


@dataclass
class MarketSelection:
    ticker: str
    close_time: datetime
    market: Dict[str, Any]


def pick_active_open_market(markets: list[Dict[str, Any]]) -> Optional[MarketSelection]:
    now = _utcnow()
    best_active: Optional[MarketSelection] = None
    best_future: Optional[MarketSelection] = None

    for m in markets:
        ticker = m.get("ticker")
        close_time_s = m.get("close_time")
        if not ticker or not close_time_s:
            continue

        try:
            close_time = _parse_iso_z(close_time_s)
        except Exception:
            continue

        open_time = None
        open_time_s = m.get("open_time")
        if open_time_s:
            try:
                open_time = _parse_iso_z(open_time_s)
            except Exception:
                open_time = None

        # Prefer an actually-active market (open_time <= now < close_time) if open_time is available.
        if open_time is not None and open_time <= now < close_time:
            if best_active is None or close_time < best_active.close_time:
                best_active = MarketSelection(ticker=ticker, close_time=close_time, market=m)
            continue

        # Otherwise fall back to nearest future close_time.
        if close_time > now:
            if best_future is None or close_time < best_future.close_time:
                best_future = MarketSelection(ticker=ticker, close_time=close_time, market=m)

    return best_active or best_future


def extract_compact_snapshot(market: Dict[str, Any]) -> Dict[str, Any]:
    # Keep this conservative: only fields we know we use elsewhere + obvious quote fields.
    out: Dict[str, Any] = {}

    for key in (
        "ticker",
        "status",
        "close_time",
        "open_time",
        "volume",
        "open_interest",
        "floor_strike",
        "cap_strike",
        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",
        "last_price",
        "last_trade_price",
    ):
        if key in market:
            out[key] = market.get(key)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-time Kalshi market snapshot scraper (JSONL).")
    ap.add_argument("--series", default="KXBTC15M", help="Series ticker (default: KXBTC15M)")
    ap.add_argument(
        "--base-url",
        default="https://api.elections.kalshi.com",
        help="Kalshi REST base URL",
    )
    ap.add_argument(
        "--env-file",
        default=os.path.join("kalshi_paper_trader", ".env"),
        help="Path to .env containing KALSHI_API_KEY=... (optional if env var set)",
    )
    ap.add_argument(
        "--pem-file",
        default=os.path.join("kalshi_paper_trader", "kalshi_secret.pem"),
        help="Path to kalshi_secret.pem",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join("kalshi_paper_trader", "market_data", "data"),
        help="Output directory for JSONL files",
    )
    ap.add_argument("--poll-ms", type=int, default=1000, help="Polling interval in milliseconds")
    ap.add_argument(
        "--gzip",
        type=int,
        default=6,
        help="Gzip level (1-9). Set 0 to disable gzip.",
    )
    ap.add_argument(
        "--include-raw",
        action="store_true",
        help="Include full market JSON under record['raw_market'] (bigger files).",
    )
    ap.add_argument(
        "--duration-min",
        type=float,
        default=0.0,
        help="Stop after N minutes (0 = run until Ctrl-C).",
    )
    args = ap.parse_args()

    gzip_level: Optional[int] = None if args.gzip == 0 else max(1, min(9, args.gzip))
    _ensure_dir(args.out_dir)

    client = KalshiHttpClient.from_files(
        base_url=args.base_url,
        env_file=args.env_file if args.env_file and os.path.exists(args.env_file) else None,
        pem_file=args.pem_file,
        timeout_s=10.0,
    )

    stop = {"flag": False}

    def _on_signal(_sig, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    start_time = time.time()
    duration_s = max(0.0, float(args.duration_min)) * 60.0

    current: Optional[MarketSelection] = None
    out_fh: Optional[TextIO] = None
    out_path: Optional[str] = None

    def _close_out():
        nonlocal out_fh
        if out_fh is not None:
            out_fh.close()
            out_fh = None

    try:
        while not stop["flag"]:
            if duration_s and (time.time() - start_time) >= duration_s:
                break

            now = _utcnow()
            if current is None or now >= current.close_time:
                # Refresh open markets and (re)select active market.
                markets_resp = client.get(
                    "/trade-api/v2/markets",
                    params={"series_ticker": args.series, "status": "open", "limit": 200},
                )
                markets = markets_resp.get("markets", [])
                selection = pick_active_open_market(markets)
                if selection is None:
                    time.sleep(min(2.0, args.poll_ms / 1000.0))
                    continue

                if current is None or selection.ticker != current.ticker:
                    current = selection
                    _close_out()
                    suffix = "jsonl.gz" if gzip_level is not None else "jsonl"
                    out_path = os.path.join(args.out_dir, f"{current.ticker}.{suffix}")
                    out_fh = _open_jsonl(out_path, gzip_level)

                    meta_path = os.path.join(args.out_dir, f"{current.ticker}.meta.json")
                    with open(meta_path, "w") as f:
                        json.dump(
                            {
                                "series": args.series,
                                "ticker": current.ticker,
                                "selected_at": now.isoformat(),
                                "close_time": current.close_time.isoformat(),
                                "market": current.market,
                            },
                            f,
                            indent=2,
                            sort_keys=True,
                        )

            assert current is not None
            assert out_fh is not None
            assert out_path is not None

            t0 = time.time()
            error: Optional[str] = None
            market_detail: Optional[Dict[str, Any]] = None

            try:
                resp = client.get(f"/trade-api/v2/markets/{current.ticker}")
                market_detail = resp.get("market", resp)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

            t1 = time.time()
            record: Dict[str, Any] = {
                "ts_ms": int(t1 * 1000),
                "ticker": current.ticker,
                "poll_ms": args.poll_ms,
                "request_latency_ms": int((t1 - t0) * 1000),
            }

            if error is not None:
                record["error"] = error
            else:
                record["snapshot"] = extract_compact_snapshot(market_detail or {})
                if args.include_raw:
                    record["raw_market"] = market_detail

            out_fh.write(json.dumps(record, separators=(",", ":")) + "\n")

            # Sleep remaining time, if any.
            elapsed = time.time() - t0
            sleep_s = max(0.0, (args.poll_ms / 1000.0) - elapsed)
            if sleep_s:
                time.sleep(sleep_s)

    finally:
        _close_out()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
