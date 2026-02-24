#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.data import iter_snapshot_files, load_label_map, read_snapshot_file
from pipeline.model import save_checkpoint
from pipeline.runtime_config import cfg_get, cfg_get_path, load_runtime_config
from pipeline.strategy_runtime import RuntimeState, build_engine, load_strategy_spec
from pipeline.train_rnn import TrainArgs, train


MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass(frozen=True)
class MarketSample:
    path: Path
    ticker: str
    sort_key: Tuple[int, int, int, int, int, str]


@dataclass(frozen=True)
class MarketEval:
    ticker: str
    had_trade: bool
    side: str
    contracts: int
    entry_price_cents: int
    won: bool
    gross_pnl_cents: int
    fee_cents: int
    pnl_cents: int
    signal_reason: str


def _ticker_from_snapshot_path(path: Path) -> str:
    name = path.name
    i = name.find(".snap_")
    if i > 0:
        return name[:i]
    recs = read_snapshot_file(path)
    if recs:
        return recs[-1].market_ticker
    return ""


def _ticker_sort_key(ticker: str) -> Tuple[int, int, int, int, int, str]:
    # Expected: KXBTC15M-25DEC101715-15
    parts = ticker.split("-")
    if len(parts) >= 3:
        token = parts[1].upper()
        if len(token) == 11 and token[:2].isdigit() and token[5:7].isdigit() and token[7:9].isdigit() and token[9:11].isdigit():
            yy = int(token[:2])
            mm = MONTHS.get(token[2:5], 0)
            dd = int(token[5:7])
            hh = int(token[7:9])
            mi = int(token[9:11])
            if mm > 0:
                return (2000 + yy, mm, dd, hh, mi, ticker)
    return (0, 0, 0, 0, 0, ticker)


def _kalshi_fee_cents(price_cents: int, contracts: int, fee_rate: float) -> int:
    p = max(0.0, min(1.0, float(price_cents) / 100.0))
    raw_fee_cents = 100.0 * float(fee_rate) * float(contracts) * p * (1.0 - p)
    return int(math.ceil(raw_fee_cents - 1e-12))


def _settle_trade(side: str, contracts: int, entry_price_cents: int, label_yes: int, fee_rate: float) -> Tuple[bool, int, int, int]:
    won = (side == "yes" and label_yes == 1) or (side == "no" and label_yes == 0)
    payout_cents = int(contracts) * (100 if won else 0)
    cost_cents = int(contracts) * int(entry_price_cents)
    gross_pnl_cents = payout_cents - cost_cents
    fee_cents = _kalshi_fee_cents(entry_price_cents, contracts, fee_rate=fee_rate)
    pnl_cents = gross_pnl_cents - fee_cents
    return won, gross_pnl_cents, fee_cents, pnl_cents


def _max_drawdown_cents(pnl_cents_series: Sequence[int]) -> int:
    peak = 0
    eq = 0
    max_dd = 0
    for x in pnl_cents_series:
        eq += int(x)
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe_like(pnl_cents_series: Sequence[int]) -> float:
    xs = [float(x) for x in pnl_cents_series]
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / float(n)
    var = sum((x - mean) ** 2 for x in xs) / float(n - 1)
    if var <= 0:
        return 0.0
    std = math.sqrt(var)
    return mean / std * math.sqrt(float(n))


def _build_samples(snapshot_glob: str, labels: Dict[str, int], series_filter: str) -> List[MarketSample]:
    out: List[MarketSample] = []
    for p in iter_snapshot_files(snapshot_glob):
        ticker = _ticker_from_snapshot_path(p)
        if not ticker:
            continue
        if series_filter and not ticker.startswith(f"{series_filter}-"):
            continue
        if ticker not in labels:
            continue
        out.append(MarketSample(path=p, ticker=ticker, sort_key=_ticker_sort_key(ticker)))
    out.sort(key=lambda m: m.sort_key)
    return out


def _simulate_market(
    *,
    sample: MarketSample,
    label_yes: int,
    engine: Any,
    slippage_cents: int,
    fee_rate: float,
) -> Optional[MarketEval]:
    records = read_snapshot_file(sample.path)
    if not records:
        return None
    ticker = records[-1].market_ticker
    if ticker != sample.ticker:
        ticker = sample.ticker

    engine.on_new_market(ticker)
    position_side = "flat"
    entry_side = ""
    entry_px = 0
    entry_count = 0
    entry_reason = ""
    close_ts_ms = records[-1].ts_ms

    for rec in records:
        now_ts = float(rec.ts_ms) / 1000.0
        state = RuntimeState(
            market_ticker=ticker,
            position_side=position_side,
            seconds_to_close=max(0.0, float(close_ts_ms - rec.ts_ms) / 1000.0),
            now_ts=now_ts,
        )
        intents = engine.on_snapshot(rec, state)
        for intent in intents:
            if position_side != "flat":
                continue
            px = int(intent.limit_price_cents) + int(slippage_cents)
            px = max(1, min(99, px))
            position_side = str(intent.side)
            entry_side = str(intent.side)
            entry_px = px
            entry_count = max(1, int(intent.count))
            entry_reason = str(intent.reason)
            break

    if not entry_side:
        return MarketEval(
            ticker=ticker,
            had_trade=False,
            side="",
            contracts=0,
            entry_price_cents=0,
            won=False,
            gross_pnl_cents=0,
            fee_cents=0,
            pnl_cents=0,
            signal_reason="",
        )

    won, gross, fee, pnl = _settle_trade(entry_side, entry_count, entry_px, label_yes, fee_rate=fee_rate)
    return MarketEval(
        ticker=ticker,
        had_trade=True,
        side=entry_side,
        contracts=entry_count,
        entry_price_cents=entry_px,
        won=won,
        gross_pnl_cents=gross,
        fee_cents=fee,
        pnl_cents=pnl,
        signal_reason=entry_reason,
    )


def _aggregate(evals: Sequence[MarketEval]) -> Dict[str, Any]:
    markets = len(evals)
    trades = [e for e in evals if e.had_trade]
    pnl_series = [e.pnl_cents for e in evals]
    trade_pnls = [e.pnl_cents for e in trades]
    gross_series = [e.gross_pnl_cents for e in trades]
    fee_series = [e.fee_cents for e in trades]
    total_contracts = sum(e.contracts for e in trades)
    wins = sum(1 for e in trades if e.won)
    yes_trades = [e for e in trades if e.side == "yes"]
    no_trades = [e for e in trades if e.side == "no"]
    yes_pnl = sum(e.pnl_cents for e in yes_trades)
    no_pnl = sum(e.pnl_cents for e in no_trades)
    gross_profit = sum(x for x in trade_pnls if x > 0)
    gross_loss = -sum(x for x in trade_pnls if x < 0)
    profit_factor = (float(gross_profit) / float(gross_loss)) if gross_loss > 0 else 0.0

    out = {
        "markets_evaluated": markets,
        "trades": len(trades),
        "trade_rate": (float(len(trades)) / float(markets)) if markets > 0 else 0.0,
        "wins": wins,
        "win_rate": (float(wins) / float(len(trades))) if trades else 0.0,
        "gross_pnl_dollars": sum(gross_series) / 100.0,
        "fees_dollars": sum(fee_series) / 100.0,
        "net_pnl_dollars": sum(trade_pnls) / 100.0,
        "ev_per_trade_dollars": (sum(trade_pnls) / 100.0 / float(len(trades))) if trades else 0.0,
        "ev_per_contract_cents": (float(sum(trade_pnls)) / float(total_contracts)) if total_contracts > 0 else 0.0,
        "avg_entry_price_cents": (float(sum(e.entry_price_cents for e in trades)) / float(len(trades))) if trades else 0.0,
        "max_drawdown_dollars": _max_drawdown_cents(pnl_series) / 100.0,
        "sharpe_like_per_trade": _sharpe_like(trade_pnls),
        "profit_factor": profit_factor,
        "yes_trades": len(yes_trades),
        "no_trades": len(no_trades),
        "yes_pnl_dollars": yes_pnl / 100.0,
        "no_pnl_dollars": no_pnl / 100.0,
    }
    return out


def _link_files(files: Sequence[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        dst = out_dir / f"{i:06d}_{src.name}"
        try:
            os.symlink(src.resolve(), dst)
        except Exception:
            shutil.copy2(src, dst)


def _run_fixed(
    *,
    samples: Sequence[MarketSample],
    labels: Dict[str, int],
    strategy_path: str,
    slippage_cents: int,
    fee_rate: float,
) -> Dict[str, Any]:
    engine = build_engine(strategy_path)
    evals: List[MarketEval] = []
    for sample in samples:
        label_yes = labels.get(sample.ticker, -1)
        if label_yes not in (0, 1):
            continue
        res = _simulate_market(
            sample=sample,
            label_yes=label_yes,
            engine=engine,
            slippage_cents=slippage_cents,
            fee_rate=fee_rate,
        )
        if res is not None:
            evals.append(res)
    summary = _aggregate(evals)
    return {
        "mode": "fixed",
        "summary": summary,
    }


def _run_walkforward(
    *,
    base_strategy_path: str,
    base_strategy_raw: Dict[str, Any],
    samples: Sequence[MarketSample],
    labels_path: str,
    labels: Dict[str, int],
    slippage_cents: int,
    fee_rate: float,
    train_markets: int,
    test_markets: int,
    step_markets: int,
    max_windows: int,
    seq_len: int,
    min_records: int,
    val_frac: float,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    if str(base_strategy_raw.get("strategy_type", "")) != "rnn_edge_v1":
        raise RuntimeError("walkforward mode currently supports strategy_type=rnn_edge_v1 only")

    windows: List[Dict[str, Any]] = []
    all_evals: List[MarketEval] = []

    with tempfile.TemporaryDirectory(prefix="kalshi_eval_") as td:
        tmp_root = Path(td)
        start = 0
        w = 0
        n = len(samples)
        while start + train_markets + test_markets <= n:
            if max_windows > 0 and w >= max_windows:
                break

            tr = list(samples[start : start + train_markets])
            te = list(samples[start + train_markets : start + train_markets + test_markets])
            if not tr or not te:
                break

            win_dir = tmp_root / f"w{w:03d}"
            tr_dir = win_dir / "train_snapshots"
            _link_files([m.path for m in tr], tr_dir)

            targs = TrainArgs(
                snapshot_glob=str(tr_dir),
                labels_json=labels_path,
                seq_len=seq_len,
                min_records=min_records,
                val_frac=val_frac,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                seed=seed + w,
                device=device,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            train_res = train(targs)
            model_path = win_dir / "model.pt"
            save_checkpoint(str(model_path), train_res.model)

            strategy_raw = dict(base_strategy_raw)
            model_cfg = dict(strategy_raw.get("model") or {})
            model_cfg["model_path"] = str(model_path.resolve())
            model_cfg["seq_len"] = int(seq_len)
            model_cfg["device"] = str(device)
            strategy_raw["model"] = model_cfg
            strategy_path = win_dir / "strategy.json"
            strategy_path.write_text(json.dumps(strategy_raw, indent=2))

            engine = build_engine(str(strategy_path))
            evals_window: List[MarketEval] = []
            for sample in te:
                label_yes = labels.get(sample.ticker, -1)
                if label_yes not in (0, 1):
                    continue
                res = _simulate_market(
                    sample=sample,
                    label_yes=label_yes,
                    engine=engine,
                    slippage_cents=slippage_cents,
                    fee_rate=fee_rate,
                )
                if res is not None:
                    evals_window.append(res)
                    all_evals.append(res)

            win_summary = _aggregate(evals_window)
            windows.append(
                {
                    "window_index": w,
                    "train_markets": len(tr),
                    "test_markets": len(te),
                    "train_start_ticker": tr[0].ticker,
                    "train_end_ticker": tr[-1].ticker,
                    "test_start_ticker": te[0].ticker,
                    "test_end_ticker": te[-1].ticker,
                    "train_val_loss": train_res.val_loss,
                    "train_val_accuracy": train_res.val_accuracy,
                    "summary": win_summary,
                }
            )

            w += 1
            start += step_markets

    if not windows:
        raise RuntimeError(
            "No walk-forward windows evaluated. "
            "Try smaller --train-markets/--test-markets or provide more snapshot files."
        )

    overall = _aggregate(all_evals)
    return {
        "mode": "walkforward",
        "base_strategy": str(Path(base_strategy_path).resolve()),
        "windows": windows,
        "overall": overall,
    }


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config/runtime.json")
    pre_args, _ = pre.parse_known_args()
    cfg = load_runtime_config(str(pre_args.config))

    ap = argparse.ArgumentParser(
        description="Evaluate strategy edge by replaying snapshots and settling against labels."
    )
    ap.add_argument("--config", default=str(pre_args.config), help="Runtime config JSON path.")
    ap.add_argument("--strategy", required=True, help="Path to strategy.json")
    ap.add_argument(
        "--snapshots",
        default=cfg_get_path(cfg, "highres_dir", ROOT, "pipeline_data/highres"),
        help="Directory/file/glob of *.snap_*ms.jsonl(.gz)",
    )
    ap.add_argument(
        "--labels",
        default=cfg_get_path(cfg, "labels_file", ROOT, "pipeline_data/labels/kalshi_settlements.json"),
        help="JSON ticker settlement labels.",
    )
    ap.add_argument("--series", default="", help="Optional series filter (defaults to strategy markets.series_ticker).")
    ap.add_argument("--mode", choices=["walkforward", "fixed"], default="walkforward")
    ap.add_argument("--test-frac", type=float, default=1.0, help="Fixed mode only: evaluate most recent fraction.")
    ap.add_argument("--eval-last", type=int, default=0, help="Fixed mode only: evaluate only the N most recent markets.")
    ap.add_argument("--slippage-cents", type=int, default=0, help="Adverse fill slippage in cents.")
    ap.add_argument("--fee-rate", type=float, default=0.07, help="Fee coefficient in Kalshi fee formula.")
    ap.add_argument("--out", default="", help="Optional JSON output path.")

    # Walk-forward / retrain args.
    ap.add_argument("--train-markets", type=int, default=2)
    ap.add_argument("--test-markets", type=int, default=1)
    ap.add_argument("--step-markets", type=int, default=1)
    ap.add_argument("--max-windows", type=int, default=0, help="0 means no cap.")
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
    args = ap.parse_args()

    labels = load_label_map(str(args.labels))
    base_spec = load_strategy_spec(str(args.strategy))
    base_raw = json.loads(Path(args.strategy).read_text())

    series = str(args.series or base_spec.markets.get("series_ticker", cfg_get(cfg, "series", ""))).strip()
    samples = _build_samples(str(args.snapshots), labels=labels, series_filter=series)
    if not samples:
        raise RuntimeError("No labeled snapshot files found for evaluation.")

    if args.mode == "fixed":
        subset = list(samples)
        if args.eval_last > 0:
            subset = subset[-max(1, int(args.eval_last)) :]
        elif 0.0 < float(args.test_frac) < 1.0:
            n = max(1, int(round(len(subset) * float(args.test_frac))))
            subset = subset[-n:]
        result = _run_fixed(
            samples=subset,
            labels=labels,
            strategy_path=str(args.strategy),
            slippage_cents=int(args.slippage_cents),
            fee_rate=float(args.fee_rate),
        )
    else:
        result = _run_walkforward(
            base_strategy_path=str(args.strategy),
            base_strategy_raw=base_raw,
            samples=samples,
            labels_path=str(args.labels),
            labels=labels,
            slippage_cents=int(args.slippage_cents),
            fee_rate=float(args.fee_rate),
            train_markets=max(1, int(args.train_markets)),
            test_markets=max(1, int(args.test_markets)),
            step_markets=max(1, int(args.step_markets)),
            max_windows=max(0, int(args.max_windows)),
            seq_len=int(args.seq_len),
            min_records=int(args.min_records),
            val_frac=float(args.val_frac),
            epochs=max(1, int(args.epochs)),
            batch_size=max(1, int(args.batch_size)),
            lr=float(args.lr),
            hidden_size=max(1, int(args.hidden_size)),
            num_layers=max(1, int(args.num_layers)),
            dropout=float(args.dropout),
            seed=int(args.seed),
            device=str(args.device),
        )

    result["inputs"] = {
        "strategy": str(Path(args.strategy).resolve()),
        "snapshots": str(args.snapshots),
        "labels": str(args.labels),
        "series_filter": series,
        "mode": args.mode,
        "slippage_cents": int(args.slippage_cents),
        "fee_rate": float(args.fee_rate),
    }
    if args.mode == "fixed":
        result["inputs"]["test_frac"] = float(args.test_frac)
        result["inputs"]["eval_last"] = int(args.eval_last)
    else:
        result["inputs"]["train_markets"] = int(args.train_markets)
        result["inputs"]["test_markets"] = int(args.test_markets)
        result["inputs"]["step_markets"] = int(args.step_markets)
        result["inputs"]["max_windows"] = int(args.max_windows)
        result["inputs"]["epochs"] = int(args.epochs)
        result["inputs"]["batch_size"] = int(args.batch_size)
        result["inputs"]["seq_len"] = int(args.seq_len)

    txt = json.dumps(result, indent=2)
    print(txt)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

