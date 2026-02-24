from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED = ["open_time_ms", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, Any]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def validate_csv(path: Path, *, bar_minutes: int) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    if not path.exists():
        return ValidationResult(False, [f"missing file: {path}"], [], {})

    rows = _read_rows(path)
    if not rows:
        return ValidationResult(False, ["empty csv"], [], {})

    cols = set(rows[0].keys())
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        errors.append(f"missing required columns: {missing}")

    # Parse timestamps and basic OHLC sanity.
    ts: list[int] = []
    n_bad_ts = 0
    n_bad_ohlc = 0
    n_nonpos_close = 0
    for i, row in enumerate(rows):
        try:
            t = int(float(row["open_time_ms"]))
            ts.append(t)
        except Exception:
            n_bad_ts += 1
            continue
        try:
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            if not (l <= min(o, c) <= max(o, c) <= h):
                n_bad_ohlc += 1
            if c <= 0:
                n_nonpos_close += 1
        except Exception:
            n_bad_ohlc += 1

    if n_bad_ts:
        errors.append(f"bad open_time_ms rows: {n_bad_ts}")
    if n_bad_ohlc:
        warnings.append(f"ohlc sanity failures: {n_bad_ohlc}")
    if n_nonpos_close:
        errors.append(f"non-positive close rows: {n_nonpos_close}")

    if ts:
        # Order / duplicates / gaps
        ts_sorted = sorted(ts)
        if ts_sorted != ts:
            warnings.append("rows are not strictly time-ordered; scanner will sort internally, but you should write sorted output")
        dup = sum(1 for a, b in zip(ts_sorted, ts_sorted[1:]) if a == b)
        if dup:
            warnings.append(f"duplicate timestamps: {dup}")
        dt_expected = int(bar_minutes * 60 * 1000)
        gaps = []
        for a, b in zip(ts_sorted, ts_sorted[1:]):
            d = b - a
            if d <= 0:
                continue
            if d != dt_expected:
                gaps.append(d)
        if gaps:
            # Report a few of the most common gap sizes.
            from collections import Counter

            c = Counter(gaps)
            common = c.most_common(5)
            warnings.append(f"non-uniform bar spacing found; expected {dt_expected}ms. common gaps: {common}")

        stats["rows"] = len(rows)
        stats["first_ts_utc"] = datetime.fromtimestamp(ts_sorted[0] / 1000.0, tz=timezone.utc).isoformat()
        stats["last_ts_utc"] = datetime.fromtimestamp(ts_sorted[-1] / 1000.0, tz=timezone.utc).isoformat()
        stats["expected_dt_ms"] = dt_expected

    ok = (len(errors) == 0)
    return ValidationResult(ok, errors, warnings, stats)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a universal OHLCV CSV for scanner ingestion.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--bar-minutes", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    res = validate_csv(Path(args.csv), bar_minutes=args.bar_minutes)
    payload = {
        "ok": res.ok,
        "errors": res.errors,
        "warnings": res.warnings,
        "stats": res.stats,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0 if res.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

