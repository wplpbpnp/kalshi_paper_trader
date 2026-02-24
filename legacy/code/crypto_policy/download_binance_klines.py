from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import requests


BINANCE = "https://api.binance.com"
BINANCE_US = "https://api.binance.us"


def _dt_utc_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _parse_ts(s: str) -> int:
    """
    Parse either:
      - an integer ms timestamp
      - or an ISO-8601 datetime (assumed UTC if no tz offset provided).
    Returns ms.
    """
    s = s.strip()
    if s.isdigit():
        return int(s)
    # datetime.fromisoformat doesn't accept Z; normalize.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclass(frozen=True)
class Kline:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int
    quote_volume: float
    trade_count: int
    taker_buy_base_vol: float
    taker_buy_quote_vol: float


def _fetch_klines(
    *,
    base_url: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int],
    limit: int = 1000,
    timeout_s: int = 30,
) -> list[Kline]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": limit,
    }
    if end_ms is not None:
        params["endTime"] = end_ms
    r = requests.get(f"{base_url}/api/v3/klines", params=params, timeout=timeout_s)
    r.raise_for_status()
    raw = r.json()
    out: list[Kline] = []
    for row in raw:
        # https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
        out.append(
            Kline(
                open_time_ms=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time_ms=int(row[6]),
                quote_volume=float(row[7]),
                trade_count=int(row[8]),
                taker_buy_base_vol=float(row[9]),
                taker_buy_quote_vol=float(row[10]),
            )
        )
    return out


def _write_csv(path: Path, klines: Iterable[Kline]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "open_time_ms",
        "open_time_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time_ms",
        "close_time_utc",
        "quote_volume",
        "trade_count",
        "taker_buy_base_vol",
        "taker_buy_quote_vol",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in klines:
            w.writerow(
                {
                    "open_time_ms": k.open_time_ms,
                    "open_time_utc": _dt_utc_iso(k.open_time_ms),
                    "open": k.open,
                    "high": k.high,
                    "low": k.low,
                    "close": k.close,
                    "volume": k.volume,
                    "close_time_ms": k.close_time_ms,
                    "close_time_utc": _dt_utc_iso(k.close_time_ms),
                    "quote_volume": k.quote_volume,
                    "trade_count": k.trade_count,
                    "taker_buy_base_vol": k.taker_buy_base_vol,
                    "taker_buy_quote_vol": k.taker_buy_quote_vol,
                }
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download Binance/Binance.US OHLCV klines to CSV.")
    ap.add_argument("--exchange", choices=["binance", "binanceus"], default="binanceus")
    ap.add_argument("--symbol", required=True, help="e.g., BTCUSDT or PENGUUSDT")
    ap.add_argument("--interval", default="30m", help="Binance interval string, e.g. 1m,5m,15m,30m,1h,4h,1d")
    ap.add_argument("--start", required=True, help="ISO-8601 (UTC) or ms timestamp")
    ap.add_argument("--end", default="", help="ISO-8601 (UTC) or ms timestamp (optional)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--sleep-ms", type=int, default=250, help="Sleep between paginated requests")
    ap.add_argument("--timeout-s", type=int, default=30)
    ap.add_argument("--assumption-notes", default="", help="Optional note recorded in assumptions.json")
    args = ap.parse_args(argv)

    base_url = BINANCE_US if args.exchange == "binanceus" else BINANCE
    start_ms = _parse_ts(args.start)
    end_ms = _parse_ts(args.end) if args.end.strip() else None

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Assumptions log
    (out_dir / "assumptions.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": "kalshi_paper_trader/crypto_policy/download_binance_klines.py",
                "args": {k: getattr(args, k) for k in vars(args)},
                "data_assumptions": [
                    "Data source is Binance REST /api/v3/klines (spot candles).",
                    "Timestamps are in milliseconds since epoch (UTC).",
                    "This fetcher paginates forward by open_time_ms; it does not attempt to repair missing bars.",
                ],
                "note": args.assumption_notes,
                "source": {"exchange": args.exchange, "base_url": base_url, "endpoint": "/api/v3/klines"},
            },
            indent=2,
        )
    )

    klines_all: list[Kline] = []
    next_start = start_ms
    while True:
        batch = _fetch_klines(
            base_url=base_url,
            symbol=args.symbol,
            interval=args.interval,
            start_ms=next_start,
            end_ms=end_ms,
            limit=1000,
            timeout_s=args.timeout_s,
        )
        if not batch:
            break
        klines_all.extend(batch)

        last_open = batch[-1].open_time_ms
        # Stop if we're not advancing (safety).
        if last_open <= next_start:
            break
        next_start = last_open + 1

        # Stop if we've reached the end bound.
        if end_ms is not None and last_open >= end_ms:
            break
        time.sleep(max(0.0, args.sleep_ms / 1000.0))

        # If the API returned fewer than limit, we're likely at the end.
        if len(batch) < 1000:
            break

    # De-dup and sort by open_time
    by_open: dict[int, Kline] = {k.open_time_ms: k for k in klines_all}
    klines_sorted = [by_open[t] for t in sorted(by_open)]

    _write_csv(out_path, klines_sorted)
    print(f"Wrote {len(klines_sorted)} klines to {out_path}")
    if klines_sorted:
        print(f"First: {_dt_utc_iso(klines_sorted[0].open_time_ms)}  Last: {_dt_utc_iso(klines_sorted[-1].open_time_ms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

