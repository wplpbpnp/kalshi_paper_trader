from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from collections import Counter, deque, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Optional

from data import load_markets, Market15m
from sim import _taker_fee_cents


def _bucket_floor(x: float, step: int) -> int:
    return int((x // step) * step)


@dataclass(frozen=True)
class Bucket:
    minute: int
    ask_bin_cents: int


@dataclass
class BucketStats:
    bucket: Bucket
    n: int
    mean_pnl: float
    win_rate: float
    avg_ask: float
    avg_spread: float


def _iter_days(markets: list[Market15m]) -> list[str]:
    return sorted({m.close_date for m in markets})


def _pnl_buy_yes_hold_to_settle(ask_cents: int, y_yes: bool) -> int:
    fee = _taker_fee_cents(ask_cents)
    payoff = 100 if y_yes else 0
    return payoff - ask_cents - fee


def _get_snapshot_at_minute(m: Market15m, minute: int) -> Optional[tuple[int, int]]:
    """
    Return (ask_cents, spread_cents) for YES at candle open of the given minute.
    """
    if minute < 0 or minute >= len(m.candles):
        return None
    c = m.candles[minute]
    yb = c.yes_bid.open
    ya = c.yes_ask.open
    if yb is None or ya is None or ya <= yb:
        return None
    ask = int(ya)
    spread = int(ya - yb)
    return ask, spread


def _bucketize_market(
    m: Market15m,
    minute: int,
    ask_bin_size: int,
    max_spread_cents: int,
) -> Optional[tuple[Bucket, int, int, bool]]:
    snap = _get_snapshot_at_minute(m, minute)
    if snap is None:
        return None
    ask, spread = snap
    if spread > max_spread_cents:
        return None
    b = Bucket(minute=minute, ask_bin_cents=_bucket_floor(ask, ask_bin_size))
    pnl = _pnl_buy_yes_hold_to_settle(ask, m.settles_yes())
    return b, pnl, ask, spread


def _compute_bucket_stats(
    markets: list[Market15m],
    days: set[str],
    *,
    minute_range: range,
    ask_bin_size: int,
    max_spread_cents: int,
    min_n: int,
) -> list[BucketStats]:
    # bucket -> lists
    pnl_by_bucket: dict[Bucket, list[int]] = {}
    wins_by_bucket: dict[Bucket, list[int]] = {}
    asks_by_bucket: dict[Bucket, list[int]] = {}
    spreads_by_bucket: dict[Bucket, list[int]] = {}

    for m in markets:
        if m.close_date not in days:
            continue
        for minute in minute_range:
            tup = _bucketize_market(m, minute, ask_bin_size, max_spread_cents)
            if tup is None:
                continue
            b, pnl, ask, spread = tup
            pnl_by_bucket.setdefault(b, []).append(pnl)
            wins_by_bucket.setdefault(b, []).append(1 if m.settles_yes() else 0)
            asks_by_bucket.setdefault(b, []).append(ask)
            spreads_by_bucket.setdefault(b, []).append(spread)

    out: list[BucketStats] = []
    for b, pnls in pnl_by_bucket.items():
        n = len(pnls)
        if n < min_n:
            continue
        wins = wins_by_bucket.get(b, [])
        asks = asks_by_bucket.get(b, [])
        spreads = spreads_by_bucket.get(b, [])
        out.append(
            BucketStats(
                bucket=b,
                n=n,
                mean_pnl=float(mean(pnls)),
                win_rate=float(mean(wins)) if wins else 0.0,
                avg_ask=float(mean(asks)) if asks else 0.0,
                avg_spread=float(mean(spreads)) if spreads else 0.0,
            )
        )

    out.sort(key=lambda s: s.mean_pnl, reverse=True)
    return out


def _apply_bucket_policy(
    markets: list[Market15m],
    days: set[str],
    bucket: Bucket,
    *,
    ask_bin_size: int,
    max_spread_cents: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    """
    For each market in `days`, if it matches the chosen bucket at that minute,
    execute 1 taker buy YES and hold to settlement.
    Returns (per_trade_rows, pnls).
    """
    trades: list[dict[str, Any]] = []
    pnls: list[int] = []

    for m in markets:
        if m.close_date not in days:
            continue
        tup = _bucketize_market(m, bucket.minute, ask_bin_size, max_spread_cents)
        if tup is None:
            continue
        b, pnl, ask, spread = tup
        if b != bucket:
            continue
        trades.append(
            {
                "close_date": m.close_date,
                "ticker": m.ticker,
                "minute": bucket.minute,
                "ask_bin_cents": bucket.ask_bin_cents,
                "ask_cents": ask,
                "spread_cents": spread,
                "result": "yes" if m.settles_yes() else "no",
                "pnl_cents": pnl,
            }
        )
        pnls.append(pnl)

    return trades, pnls


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        # Rows can be written from multiple branches (skips/gates), so keys vary.
        # Build a stable union of fieldnames in first-seen order.
        fieldnames: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _write_assumptions(outdir: Path, args: argparse.Namespace) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "assumptions.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": "kalshi_paper_trader/maker-strat/forecast_backtest.py",
                "args": {k: getattr(args, k) for k in vars(args)},
                "data_assumptions": [
                    "Input data is Kalshi candlesticks (top-of-book bid/ask OHLC), typically 1-minute resolution.",
                    "Bucket snapshots use candle-open YES bid/ask values.",
                ],
                "execution_assumptions": [
                    "Each trade is 1 taker buy YES at candle-open ask for the chosen bucket, held to settlement.",
                    "No slippage model beyond spread embedded in bid/ask and taker fee model.",
                    "Bucket selection is trained on past days only (walk-forward).",
                    "Stability filter, train gate, and recent-performance gate can skip windows; skipped windows place no trades.",
                ],
                "fee_model": {
                    "description": "Approx Kalshi taker fee per contract: fee_dollars = ceil(0.07 * P * (1-P)), where P is contract price in dollars.",
                    "implementation": "kalshi_paper_trader/maker-strat/sim.py::_taker_fee_cents",
                },
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward backtest for YES forecasting buckets on KXBTC15M.")
    ap.add_argument("--data", required=True, help="Path to KXBTC15M_candles.json")
    ap.add_argument("--outdir", default="kalshi_paper_trader/maker-strat/forecast_backtest_out")

    ap.add_argument("--ask-bin-size", type=int, default=10)
    ap.add_argument("--max-spread-cents", type=int, default=40)
    ap.add_argument("--minute-start", type=int, default=0)
    ap.add_argument("--minute-end", type=int, default=14, help="Inclusive")
    ap.add_argument("--min-n", type=int, default=100, help="Min samples per bucket in train")

    ap.add_argument("--train-days", type=int, default=14)
    ap.add_argument("--test-days", type=int, default=3)
    ap.add_argument("--step-days", type=int, default=3)

    # Stability filter: only trade buckets that repeatedly rank in top-K.
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--stability-y", type=int, default=8, help="Lookback windows for stability counting.")
    ap.add_argument("--stability-x", type=int, default=3, help="Require bucket appears in >=X of last Y top-K sets.")
    ap.add_argument("--require-stability-after", type=int, default=8,
                    help="After this many windows, require chosen bucket be stable; before that, allow best-in-window.")
    ap.add_argument("--min-chosen-train-mean-pnl-cents", type=float, default=2.0,
                    help="Skip the window unless chosen bucket's train mean pnl >= this.")
    ap.add_argument("--min-chosen-train-n", type=int, default=150,
                    help="Skip the window unless chosen bucket has >= this many train samples.")
    ap.add_argument("--recent-perf-y", type=int, default=5,
                    help="Keep this many recent test-window results per bucket (for regime gate).")
    ap.add_argument("--recent-perf-min-obs", type=int, default=2,
                    help="Only apply the recent performance gate once we have at least this many past results.")
    ap.add_argument("--recent-perf-min-mean-pnl-cents", type=float, default=0.0,
                    help="Skip if the bucket's recent mean test pnl/trade is below this.")
    ap.add_argument("--switch-margin-cents", type=float, default=0.0,
                    help="If >0 and the previously chosen bucket is still eligible, keep it unless the new choice beats it by this much (in train mean pnl).")

    ap.add_argument("--require-positive-train", action="store_true",
                    help="Only trade buckets with positive train mean pnl; otherwise skip window.")
    args = ap.parse_args(argv)

    markets = load_markets(args.data)
    days_all = _iter_days(markets)
    if len(days_all) < (args.train_days + args.test_days):
        raise SystemExit("Not enough days in dataset for the chosen window sizes.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_assumptions(outdir, args)

    minute_range = range(args.minute_start, args.minute_end + 1)

    window_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []

    # Walk-forward
    start = 0
    window_id = 0
    all_test_pnls: list[int] = []
    all_test_trade_count = 0

    topk_history: deque[set[Bucket]] = deque(maxlen=max(1, args.stability_y))
    bucket_recent_perf: dict[Bucket, deque[float]] = defaultdict(lambda: deque(maxlen=max(1, args.recent_perf_y)))
    prev_bucket: Bucket | None = None

    while start + args.train_days + args.test_days <= len(days_all):
        train_days = set(days_all[start : start + args.train_days])
        test_days = set(days_all[start + args.train_days : start + args.train_days + args.test_days])
        window_id += 1

        stats = _compute_bucket_stats(
            markets,
            train_days,
            minute_range=minute_range,
            ask_bin_size=args.ask_bin_size,
            max_spread_cents=args.max_spread_cents,
            min_n=args.min_n,
        )

        if not stats:
            window_rows.append(
                {
                    "window_id": window_id,
                    "train_start": min(train_days),
                    "train_end": max(train_days),
                    "test_start": min(test_days),
                    "test_end": max(test_days),
                    "top_k": args.top_k,
                    "stability_x": args.stability_x,
                    "stability_y": args.stability_y,
                    "eligible_buckets": 0,
                    "chosen_bucket_count": 0,
                    "test_trades": 0,
                    "test_mean_pnl_cents": "",
                    "test_total_pnl_cents": 0,
                    "skipped": 1,
                }
            )
            start += args.step_days
            continue

        # Current top-K by train mean pnl
        topk = stats[: max(1, args.top_k)]
        topk_set = {s.bucket for s in topk}
        topk_history.append(topk_set)

        # Count how often buckets appeared in the last Y windows.
        cnt = Counter()
        for sset in topk_history:
            cnt.update(sset)
        eligible = {b for b, c in cnt.items() if c >= max(1, args.stability_x)}

        # Only consider buckets that are both in current top-K and stable over history.
        stable_candidates = eligible.intersection(topk_set)

        # Optionally require the best bucket in train be positive; if not, skip the window.
        best = topk[0]
        if args.require_positive_train and best.mean_pnl <= 0:
            window_rows.append(
                {
                    "window_id": window_id,
                    "train_start": min(train_days),
                    "train_end": max(train_days),
                    "test_start": min(test_days),
                    "test_end": max(test_days),
                    "top_k": args.top_k,
                    "stability_x": args.stability_x,
                    "stability_y": args.stability_y,
                    "eligible_buckets": len(eligible),
                    "chosen_bucket_count": 0,
                    "test_trades": 0,
                    "test_mean_pnl_cents": "",
                    "test_total_pnl_cents": 0,
                    "skipped": 1,
                }
            )
            start += args.step_days
            continue

        # Choose one bucket to trade this window.
        chosen_bucket = None
        if stable_candidates:
            # Pick the stable candidate with highest train mean PnL.
            mean_by_bucket = {s.bucket: s.mean_pnl for s in topk}
            chosen_bucket = max(stable_candidates, key=lambda b: mean_by_bucket.get(b, float("-inf")))
        else:
            # Before we have enough history, allow best-in-window; afterwards skip.
            if window_id < args.require_stability_after:
                chosen_bucket = best.bucket

        # If the previously chosen bucket is still a valid candidate, avoid switching
        # unless the new pick materially improves train mean pnl.
        stats_by_bucket = {s.bucket: s for s in stats}
        if args.switch_margin_cents > 0 and prev_bucket is not None and chosen_bucket is not None:
            prev_stats = stats_by_bucket.get(prev_bucket)
            chosen_stats = stats_by_bucket.get(chosen_bucket)
            prev_is_candidate = prev_bucket in stable_candidates or (window_id < args.require_stability_after and prev_bucket in topk_set)
            if prev_stats is not None and chosen_stats is not None and prev_is_candidate:
                if (chosen_stats.mean_pnl - prev_stats.mean_pnl) < args.switch_margin_cents:
                    chosen_bucket = prev_bucket

        if chosen_bucket is None:
            window_rows.append(
                {
                    "window_id": window_id,
                    "train_start": min(train_days),
                    "train_end": max(train_days),
                    "test_start": min(test_days),
                    "test_end": max(test_days),
                    "top_k": args.top_k,
                    "stability_x": args.stability_x,
                    "stability_y": args.stability_y,
                    "eligible_buckets": len(eligible),
                    "chosen_bucket_count": 0,
                    "test_trades": 0,
                    "test_mean_pnl_cents": "",
                    "test_total_pnl_cents": 0,
                    "skipped": 1,
                }
            )
            start += args.step_days
            continue

        # Train gate for chosen bucket.
        chosen_stats = stats_by_bucket.get(chosen_bucket)
        chosen_train_mean = chosen_stats.mean_pnl if chosen_stats else float("nan")
        chosen_train_n = chosen_stats.n if chosen_stats else 0

        if chosen_stats is None or chosen_train_n < args.min_chosen_train_n or chosen_train_mean < args.min_chosen_train_mean_pnl_cents:
            window_rows.append(
                {
                    "window_id": window_id,
                    "train_start": min(train_days),
                    "train_end": max(train_days),
                    "test_start": min(test_days),
                    "test_end": max(test_days),
                    "top_k": args.top_k,
                    "stability_x": args.stability_x,
                    "stability_y": args.stability_y,
                    "eligible_buckets": len(eligible),
                    "chosen_bucket_count": 0,
                    "chosen_minute": chosen_bucket.minute,
                    "chosen_ask_bin": chosen_bucket.ask_bin_cents,
                    "chosen_train_mean_pnl_cents": chosen_train_mean,
                    "chosen_train_n": chosen_train_n,
                    "recent_perf_mean_pnl_cents": "",
                    "recent_perf_obs": 0,
                    "gate": "train_threshold",
                    "test_trades": 0,
                    "test_mean_pnl_cents": "",
                    "test_total_pnl_cents": 0,
                    "skipped": 1,
                }
            )
            start += args.step_days
            continue

        # Recent performance gate for this bucket (regime check).
        hist = bucket_recent_perf.get(chosen_bucket)
        if hist is not None and len(hist) >= args.recent_perf_min_obs:
            recent_mean = float(mean(hist))
            if recent_mean < args.recent_perf_min_mean_pnl_cents:
                window_rows.append(
                    {
                        "window_id": window_id,
                        "train_start": min(train_days),
                        "train_end": max(train_days),
                        "test_start": min(test_days),
                        "test_end": max(test_days),
                        "top_k": args.top_k,
                        "stability_x": args.stability_x,
                        "stability_y": args.stability_y,
                        "eligible_buckets": len(eligible),
                        "chosen_bucket_count": 0,
                        "chosen_minute": chosen_bucket.minute,
                        "chosen_ask_bin": chosen_bucket.ask_bin_cents,
                        "chosen_train_mean_pnl_cents": chosen_train_mean,
                        "chosen_train_n": chosen_train_n,
                        "recent_perf_mean_pnl_cents": recent_mean,
                        "recent_perf_obs": len(hist),
                        "gate": "recent_perf",
                        "test_trades": 0,
                        "test_mean_pnl_cents": "",
                        "test_total_pnl_cents": 0,
                        "skipped": 1,
                    }
                )
                start += args.step_days
                continue

        trades, pnls = _apply_bucket_policy(
            markets,
            test_days,
            chosen_bucket,
            ask_bin_size=args.ask_bin_size,
            max_spread_cents=args.max_spread_cents,
        )

        # tag trades w/ window metadata
        for tr in trades:
            tr["window_id"] = window_id
            tr["train_start"] = min(train_days)
            tr["train_end"] = max(train_days)
            tr["test_start"] = min(test_days)
            tr["test_end"] = max(test_days)
            tr["top_k"] = args.top_k
            tr["stability_x"] = args.stability_x
            tr["stability_y"] = args.stability_y
            tr["eligible_buckets"] = len(eligible)
            tr["chosen_bucket_count"] = 1
            tr["chosen_minute"] = chosen_bucket.minute
            tr["chosen_ask_bin"] = chosen_bucket.ask_bin_cents
            tr["best_train_mean_pnl_cents"] = best.mean_pnl
            tr["best_train_n"] = best.n
        trade_rows.extend(trades)

        test_mean = float(mean(pnls)) if pnls else 0.0
        test_total = int(sum(pnls)) if pnls else 0

        window_rows.append(
            {
                "window_id": window_id,
                "train_start": min(train_days),
                "train_end": max(train_days),
                "test_start": min(test_days),
                "test_end": max(test_days),
                "top_k": args.top_k,
                "stability_x": args.stability_x,
                "stability_y": args.stability_y,
                "eligible_buckets": len(eligible),
                "chosen_bucket_count": 1,
                "chosen_minute": chosen_bucket.minute,
                "chosen_ask_bin": chosen_bucket.ask_bin_cents,
                "chosen_train_mean_pnl_cents": chosen_train_mean,
                "chosen_train_n": chosen_train_n,
                "recent_perf_mean_pnl_cents": float(mean(bucket_recent_perf[chosen_bucket])) if len(bucket_recent_perf[chosen_bucket]) else "",
                "recent_perf_obs": len(bucket_recent_perf[chosen_bucket]),
                "gate": "",
                "best_train_bucket_minute": best.bucket.minute,
                "best_train_bucket_ask_bin": best.bucket.ask_bin_cents,
                "best_train_mean_pnl_cents": best.mean_pnl,
                "best_train_n": best.n,
                "test_trades": len(pnls),
                "test_mean_pnl_cents": test_mean,
                "test_total_pnl_cents": test_total,
                "skipped": 0,
            }
        )

        # Update recent performance after observing test results for the chosen bucket.
        if pnls:
            bucket_recent_perf[chosen_bucket].append(test_mean)
            prev_bucket = chosen_bucket

        # daily aggregation inside test days
        by_day = {}
        for tr in trades:
            d = tr["close_date"]
            by_day.setdefault(d, {"close_date": d, "window_id": window_id, "trades": 0, "pnl_cents": 0})
            by_day[d]["trades"] += 1
            by_day[d]["pnl_cents"] += int(tr["pnl_cents"])
        for d in sorted(by_day):
            daily_rows.append(by_day[d])

        all_test_pnls.extend(pnls)
        all_test_trade_count += len(pnls)

        start += args.step_days

    # Overall summary (test trades only)
    overall = {
        "total_windows": window_id,
        "total_test_trades": all_test_trade_count,
        "mean_test_pnl_cents_per_trade": float(mean(all_test_pnls)) if all_test_pnls else 0.0,
        "std_test_pnl_cents_per_trade": float(pstdev(all_test_pnls)) if len(all_test_pnls) > 1 else 0.0,
        "total_test_pnl_cents": int(sum(all_test_pnls)) if all_test_pnls else 0,
        "mean_test_pnl_cents_per_hour_per_contract": float(4.0 * mean(all_test_pnls)) if all_test_pnls else 0.0,
    }
    (outdir / "summary.json").write_text(json.dumps(overall, indent=2))

    _write_csv(outdir / "windows.csv", window_rows)
    _write_csv(outdir / "trades.csv", trade_rows)
    _write_csv(outdir / "daily.csv", daily_rows)

    print(f"Wrote backtest to {outdir}")
    print("Overall:", overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
