#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.runtime_config import cfg_get_path, load_runtime_config


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config/runtime.json")
    pre_args, _ = pre.parse_known_args()
    cfg = load_runtime_config(str(pre_args.config))

    ap = argparse.ArgumentParser(description="Build ticker->settlement label map from historical backtest JSON.")
    ap.add_argument("--config", default=str(pre_args.config), help="Runtime config JSON path.")
    ap.add_argument(
        "--input",
        default=str((ROOT / "legacy/artifacts/kalshi_backtest_data.json").resolve()),
        help="Path to historical market JSON.",
    )
    ap.add_argument(
        "--output",
        default=cfg_get_path(cfg, "labels_file", ROOT, "pipeline_data/labels/kalshi_settlements.json"),
        help="Output label map JSON path.",
    )
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise RuntimeError(f"Missing input file: {src}")

    raw = json.loads(src.read_text())
    labels: dict[str, str] = {}
    for m in raw:
        if not isinstance(m, dict):
            continue
        ticker = m.get("ticker")
        result = str(m.get("result", "")).lower()
        if not ticker or result not in ("yes", "no"):
            continue
        labels[str(ticker)] = result

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels, indent=2, sort_keys=True))
    print(f"wrote {len(labels)} labels to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

