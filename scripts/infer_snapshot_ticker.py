#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def infer_ticker(path: Path) -> str:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            ticker = str(row.get("market_ticker") or "")
            if ticker:
                return ticker
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Infer market_ticker from a snapshot JSONL(.gz) file.")
    ap.add_argument("snapshot")
    args = ap.parse_args()
    ticker = infer_ticker(Path(args.snapshot))
    if not ticker:
        return 1
    print(ticker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

