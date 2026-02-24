import argparse
import csv
import gzip
import json
from typing import Any, Dict, Iterable, Optional, TextIO


def _open_text(path: str) -> TextIO:
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _iter_records(path: str) -> Iterable[Dict[str, Any]]:
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Kalshi snapshot JSONL(.gz) to a flat CSV.")
    ap.add_argument("--input", required=True, help="Path to .jsonl or .jsonl.gz")
    ap.add_argument("--output", required=True, help="Path to output .csv")
    args = ap.parse_args()

    fieldnames = [
        "ts_ms",
        "ticker",
        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",
        "yes_mid",
        "no_mid",
        "volume",
        "open_interest",
        "status",
        "close_time",
        "floor_strike",
        "cap_strike",
    ]

    with open(args.output, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=fieldnames)
        w.writeheader()

        for rec in _iter_records(args.input):
            if rec.get("error"):
                continue
            snap = rec.get("snapshot") or {}
            yes_bid = _as_int(snap.get("yes_bid"))
            yes_ask = _as_int(snap.get("yes_ask"))
            no_bid = _as_int(snap.get("no_bid"))
            no_ask = _as_int(snap.get("no_ask"))

            yes_mid = None
            if yes_bid is not None and yes_ask is not None and yes_bid > 0 and yes_ask > 0:
                yes_mid = 0.5 * (yes_bid + yes_ask)

            no_mid = None
            if no_bid is not None and no_ask is not None and no_bid > 0 and no_ask > 0:
                no_mid = 0.5 * (no_bid + no_ask)

            w.writerow(
                {
                    "ts_ms": rec.get("ts_ms"),
                    "ticker": rec.get("ticker"),
                    "yes_bid": yes_bid,
                    "yes_ask": yes_ask,
                    "no_bid": no_bid,
                    "no_ask": no_ask,
                    "yes_mid": yes_mid,
                    "no_mid": no_mid,
                    "volume": snap.get("volume"),
                    "open_interest": snap.get("open_interest"),
                    "status": snap.get("status"),
                    "close_time": snap.get("close_time"),
                    "floor_strike": snap.get("floor_strike"),
                    "cap_strike": snap.get("cap_strike"),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

