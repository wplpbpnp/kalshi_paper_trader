#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.data import iter_snapshot_files, load_label_map, read_snapshot_file
from pipeline.runtime_config import cfg_get, cfg_get_path, load_runtime_config


def _ticker_from_snapshot_path(path: Path) -> str:
    name = path.name
    i = name.find(".snap_")
    if i > 0:
        return name[:i]
    recs = read_snapshot_file(path)
    if recs:
        return recs[-1].market_ticker
    return ""


def _load_yes_no_labels(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    raw = load_label_map(str(path))
    return {k: ("yes" if int(v) == 1 else "no") for k, v in raw.items() if int(v) in (0, 1)}


def _parse_market_result(resp: Dict) -> Tuple[Optional[str], str]:
    market = resp.get("market")
    if not isinstance(market, dict):
        return None, "missing_market_payload"
    result = str(market.get("result", "")).strip().lower()
    if result in ("yes", "no"):
        return result, str(market.get("status", ""))
    return None, str(market.get("status", ""))


def _fetch_result(http, ticker: str, retries: int, retry_sleep_s: float) -> Tuple[Optional[str], str]:
    last_err = ""
    for i in range(max(1, retries)):
        try:
            resp = http.get(f"/trade-api/v2/markets/{ticker}")
            result, status = _parse_market_result(resp)
            return result, status
        except Exception as e:
            last_err = str(e)
            if i + 1 < retries:
                time.sleep(max(0.0, float(retry_sleep_s)))
    return None, f"error:{last_err}" if last_err else "error:unknown"


def _collect_snapshot_tickers(path_or_glob: str, series_filter: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for p in iter_snapshot_files(path_or_glob):
        ticker = _ticker_from_snapshot_path(p)
        if not ticker:
            continue
        if series_filter and not ticker.startswith(f"{series_filter}-"):
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    out.sort()
    return out


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config/runtime.json")
    pre_args, _ = pre.parse_known_args()
    cfg = load_runtime_config(str(pre_args.config))

    ap = argparse.ArgumentParser(
        description="Sync settlement labels for snapshot tickers from Kalshi API and write an aligned label file."
    )
    ap.add_argument("--config", default=str(pre_args.config), help="Runtime config JSON path.")
    ap.add_argument(
        "--snapshots",
        default=cfg_get_path(cfg, "highres_dir", ROOT, "pipeline_data/highres"),
        help="Directory/file/glob of *.snap_*ms.jsonl(.gz).",
    )
    ap.add_argument("--series", default=cfg_get(cfg, "series", "KXBTC15M"))
    ap.add_argument(
        "--labels-existing",
        default=cfg_get_path(cfg, "labels_file", ROOT, "pipeline_data/labels/kalshi_settlements.json"),
        help="Existing labels JSON to bootstrap from.",
    )
    ap.add_argument(
        "--output",
        default=str((ROOT / "pipeline_data/labels/aligned.json").resolve()),
        help="Output aligned labels JSON path.",
    )
    ap.add_argument(
        "--unresolved-out",
        default="",
        help="Optional JSON path to write unresolved ticker statuses.",
    )
    ap.add_argument("--base-url", default=cfg_get(cfg, "base_url", "https://api.elections.kalshi.com"))
    ap.add_argument("--env-file", default=cfg_get_path(cfg, "env_file", ROOT))
    ap.add_argument("--pem-file", default=cfg_get_path(cfg, "pem_file", ROOT))
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-sleep-s", type=float, default=0.2)
    ap.add_argument("--no-api", action="store_true", help="Only use existing labels file; skip Kalshi API lookups.")
    ap.add_argument(
        "--update-existing",
        action="store_true",
        help="Merge newly fetched settled labels back into --labels-existing.",
    )
    args = ap.parse_args()

    tickers = _collect_snapshot_tickers(str(args.snapshots), str(args.series).strip())
    if not tickers:
        raise RuntimeError("No snapshot tickers found.")

    existing_path = Path(str(args.labels_existing))
    existing = _load_yes_no_labels(existing_path)
    aligned: Dict[str, str] = {}
    unresolved: Dict[str, str] = {}

    from_existing = 0
    from_api = 0
    api_attempted = 0

    http = None
    if not args.no_api:
        from market_data.kalshi_http import KalshiHttpClient

        if not args.pem_file:
            raise RuntimeError("Missing --pem-file (or pem_file in config/runtime.json).")
        http = KalshiHttpClient.from_files(
            base_url=str(args.base_url),
            env_file=(str(args.env_file) if args.env_file else None),
            pem_file=str(args.pem_file),
            timeout_s=float(args.timeout_s),
        )

    newly_fetched: Dict[str, str] = {}
    for t in tickers:
        if t in existing:
            aligned[t] = existing[t]
            from_existing += 1
            continue

        if http is None:
            unresolved[t] = "missing_in_existing"
            continue

        api_attempted += 1
        result, status = _fetch_result(http, t, retries=int(args.retries), retry_sleep_s=float(args.retry_sleep_s))
        if result in ("yes", "no"):
            aligned[t] = result
            from_api += 1
            newly_fetched[t] = result
        else:
            unresolved[t] = status or "unresolved"

    out_path = Path(str(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(sorted(aligned.items())), indent=2))

    if args.unresolved_out:
        u_path = Path(str(args.unresolved_out))
        u_path.parent.mkdir(parents=True, exist_ok=True)
        u_path.write_text(json.dumps(dict(sorted(unresolved.items())), indent=2))

    if args.update_existing and newly_fetched:
        merged = dict(existing)
        merged.update(newly_fetched)
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text(json.dumps(dict(sorted(merged.items())), indent=2))

    summary = {
        "snapshots_tickers": len(tickers),
        "aligned_labels": len(aligned),
        "from_existing": from_existing,
        "from_api": from_api,
        "api_attempted": api_attempted,
        "unresolved": len(unresolved),
        "output": str(out_path),
        "labels_existing": str(existing_path),
        "updated_existing": bool(args.update_existing and newly_fetched),
    }
    print(json.dumps(summary, indent=2))

    if unresolved:
        sample = sorted(unresolved.items())[:10]
        print("unresolved_examples:")
        for ticker, status in sample:
            print(f"- {ticker}: {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
