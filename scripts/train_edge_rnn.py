#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.model import save_checkpoint
from pipeline.runtime_config import cfg_get, cfg_get_path, load_runtime_config
from pipeline.schemas import StrategySpec, utc_now_iso
from pipeline.train_rnn import TrainArgs, train


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config/runtime.json")
    pre_args, _ = pre.parse_known_args()
    cfg = load_runtime_config(str(pre_args.config))

    ap = argparse.ArgumentParser(description="Train an RNN edge model on high-resolution snapshot files.")
    ap.add_argument("--config", default=str(pre_args.config), help="Runtime config JSON path.")
    ap.add_argument(
        "--snapshots",
        default=cfg_get_path(cfg, "highres_dir", ROOT, "pipeline_data/highres"),
        help="Directory/file/glob of *.snap_*ms.jsonl(.gz)",
    )
    ap.add_argument(
        "--labels",
        default=cfg_get_path(cfg, "labels_file", ROOT, "pipeline_data/labels/kalshi_settlements.json"),
        help="JSON map/list containing ticker settlement labels.",
    )
    ap.add_argument(
        "--outdir",
        default=cfg_get_path(cfg, "strategy_dir", ROOT, "pipeline_data/strategies/rnn_edge_latest"),
    )
    ap.add_argument("--series", default=cfg_get(cfg, "series", "KXBTC15M"))
    ap.add_argument("--seq-len", type=int, default=240)
    ap.add_argument("--min-records", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--long-threshold", type=float, default=0.58)
    ap.add_argument("--short-threshold", type=float, default=0.42)
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--allow-no", action="store_true")
    ap.add_argument("--price-offset-cents", type=int, default=0)
    ap.add_argument("--min-seconds-between-trades", type=float, default=30.0)
    ap.add_argument("--max-entries-per-market", type=int, default=1)
    args = ap.parse_args()
    if not Path(args.labels).exists():
        raise RuntimeError(
            f"Labels file not found: {args.labels}. "
            "Set --labels or update config/runtime.json labels_file."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model.pt"
    strategy_path = outdir / "strategy.json"
    metrics_path = outdir / "metrics.json"

    targs = TrainArgs(
        snapshot_glob=args.snapshots,
        labels_json=args.labels,
        seq_len=args.seq_len,
        min_records=args.min_records,
        val_frac=args.val_frac,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    try:
        res = train(targs)
    except RuntimeError as e:
        raise RuntimeError(
            f"{e}\n"
            "Next steps:\n"
            "1) Use snapshots whose market tickers exist in your labels file.\n"
            "2) If you only want a pipeline sanity check, run `make smoke-train`.\n"
            "3) To inspect current overlap quickly, compare tickers in --snapshots vs --labels."
        ) from e
    save_checkpoint(str(model_path), res.model)

    strategy = StrategySpec(
        schema_version="strategy/v1",
        strategy_id=f"rnn-edge-{utc_now_iso()}",
        created_at_utc=utc_now_iso(),
        strategy_type="rnn_edge_v1",
        markets={"series_ticker": args.series},
        model={
            "model_path": "model.pt",
            "seq_len": args.seq_len,
            "device": args.device,
            "feature_names": ["yes_bid", "yes_ask", "no_bid", "no_ask", "yes_mid", "spread"],
        },
        signals={
            "long_threshold": args.long_threshold,
            "short_threshold": args.short_threshold,
        },
        execution={
            "contracts": max(1, int(args.contracts)),
            "allow_no": bool(args.allow_no),
            "price_offset_cents": int(args.price_offset_cents),
        },
        risk={
            "min_seconds_between_trades": float(args.min_seconds_between_trades),
            "max_entries_per_market": max(1, int(args.max_entries_per_market)),
            "dry_run_default": True,
        },
    )
    strategy_path.write_text(json.dumps(strategy.to_dict(), indent=2))

    metrics = {
        "created_at_utc": utc_now_iso(),
        "train_count": res.train_count,
        "val_count": res.val_count,
        "val_loss": res.val_loss,
        "val_accuracy": res.val_accuracy,
        "tickers_used": len(res.tickers_used),
        "model_path": str(model_path),
        "strategy_path": str(strategy_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
