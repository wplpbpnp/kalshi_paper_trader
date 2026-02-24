from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from . import walkforward


def _max_drawdown_bp(pnls_bp: list[float]) -> float:
    """
    Max peak-to-trough drawdown of the compounded equity curve, in bp (positive number).

    pnls_bp are assumed to be *log-wealth* changes in bp, so equity compounds as:
      equity *= exp(pnl_bp / 1e4)
    """
    peak = 1.0
    equity = 1.0
    max_dd = 0.0
    for p in pnls_bp:
        equity *= float(pow(2.718281828459045, float(p) / 1e4))
        if equity > peak:
            peak = equity
        dd_frac = (peak - equity) / peak if peak > 0 else 0.0
        if dd_frac > max_dd:
            max_dd = dd_frac
    return float(max_dd * 1e4)


def _sharpe_annualized(pnls_bp: list[float], *, periods_per_year: float) -> float:
    if len(pnls_bp) < 2:
        return 0.0
    m = mean(pnls_bp)
    s = pstdev(pnls_bp)
    if s <= 0:
        return 0.0
    return float((m / s) * (periods_per_year**0.5))


def _series_stats(prefix: str, pnls_bp: list[float], *, periods_per_year: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out[f"{prefix}_total_bp"] = float(sum(pnls_bp)) if pnls_bp else 0.0
    out[f"{prefix}_mean_bp_per_window"] = float(mean(pnls_bp)) if pnls_bp else 0.0
    out[f"{prefix}_std_bp_per_window"] = float(pstdev(pnls_bp)) if len(pnls_bp) > 1 else 0.0
    out[f"{prefix}_sharpe"] = _sharpe_annualized(pnls_bp, periods_per_year=periods_per_year)
    out[f"{prefix}_max_drawdown_bp"] = _max_drawdown_bp(pnls_bp)
    # Extra diagnostics for "risk discount" comparisons.
    # Since pnls_bp are log-wealth deltas, sum(pnls_bp)/1e4 is ln(final_wealth).
    years = (len(pnls_bp) / periods_per_year) if periods_per_year > 0 else 0.0
    total_ln = float(sum(pnls_bp)) / 1e4 if pnls_bp else 0.0
    if years > 0:
        out[f"{prefix}_cagr"] = float(pow(2.718281828459045, total_ln / years) - 1.0)
    else:
        out[f"{prefix}_cagr"] = 0.0
    # Annualized volatility in bp-of-log-wealth.
    out[f"{prefix}_ann_vol_bp"] = float(out[f"{prefix}_std_bp_per_window"] * (periods_per_year**0.5))
    # Calmar ratio: CAGR / max drawdown fraction.
    dd_frac = float(out[f"{prefix}_max_drawdown_bp"]) / 1e4
    out[f"{prefix}_calmar"] = float(out[f"{prefix}_cagr"] / dd_frac) if dd_frac > 0 else 0.0
    return out


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds.strip():
        out: list[int] = []
        for part in args.seeds.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                lo = int(a.strip())
                hi = int(b.strip())
                out.extend(list(range(lo, hi + 1)))
            else:
                out.append(int(part))
        return sorted(set(out))
    return list(range(int(args.seed_start), int(args.seed_end) + 1))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _read_windows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a seed sweep for crypto_policy.walkforward and aggregate results.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--seeds", default="", help="Comma list of ints and/or ranges like 1-10,12,20-25")
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-end", type=int, default=20)

    # Mirror walkforward flags we care about. Anything not exposed here can still be added later.
    ap.add_argument("--bar-minutes", type=int, default=30)
    ap.add_argument("--features", default="basic", help="Feature set: basic|prices")
    ap.add_argument("--bars-per-day", type=int, default=0,
                    help="Override inferred bars/day (passed through to walkforward). 0 = infer from data.")
    ap.add_argument("--train-days", type=int, default=21)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--episode-days", type=int, default=7)
    ap.add_argument("--lookback", type=int, default=16)
    ap.add_argument("--fee-bps", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=1.0)

    ap.add_argument("--one-decision-per-episode", action="store_true")
    ap.add_argument("--allow-one-exit", action="store_true")
    ap.add_argument("--force-flat-at-end", action="store_true")
    ap.add_argument("--require-position", action="store_true")
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--carry-position", action="store_true")
    ap.add_argument("--liquidate-end", action="store_true")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-episodes", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, action="append", default=[64], help="Repeatable")

    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Write sweep assumptions.
    (outdir / "assumptions.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": "kalshi_paper_trader.crypto_policy.seed_sweep",
                "args": {k: getattr(args, k) for k in vars(args)},
                "notes": [
                    "This runs multiple walk-forward backtests with different RNG seeds and aggregates totals.",
                    "This is primarily for stability diagnostics, not for selecting the best seed.",
                ],
            },
            indent=2,
        )
    )

    seeds = _parse_seeds(args)
    rows: list[dict[str, Any]] = []

    # Each window is a test of length args.test_days, with one PnL observation per window.
    # For sharpe, treat each window as one "period".
    periods_per_year = float(365.0 / max(1.0, float(args.test_days)))

    t0 = time.time()
    for s in seeds:
        t_seed0 = time.time()
        run_out = outdir / f"seed_{s:04d}"
        argv_wf = [
            "--data",
            args.data,
            "--outdir",
            str(run_out),
            "--bar-minutes",
            str(args.bar_minutes),
            "--features",
            str(args.features),
            "--bars-per-day",
            str(args.bars_per_day),
            "--train-days",
            str(args.train_days),
            "--test-days",
            str(args.test_days),
            "--step-days",
            str(args.step_days),
            "--episode-days",
            str(args.episode_days),
            "--lookback",
            str(args.lookback),
            "--fee-bps",
            str(args.fee_bps),
            "--slippage-bps",
            str(args.slippage_bps),
            "--epochs",
            str(args.epochs),
            "--batch-episodes",
            str(args.batch_episodes),
            "--lr",
            str(args.lr),
            "--gamma",
            str(args.gamma),
            "--entropy-coef",
            str(args.entropy_coef),
            "--value-coef",
            str(args.value_coef),
            "--grad-clip",
            str(args.grad_clip),
            "--seed",
            str(s),
            "--device",
            str(args.device),
        ]
        for h in args.hidden:
            argv_wf.extend(["--hidden", str(h)])
        if args.one_decision_per_episode:
            argv_wf.append("--one-decision-per-episode")
        if args.allow_one_exit:
            argv_wf.append("--allow-one-exit")
        if args.force_flat_at_end:
            argv_wf.append("--force-flat-at-end")
        if args.require_position:
            argv_wf.append("--require-position")
        if args.long_only:
            argv_wf.append("--long-only")
        if args.carry_position:
            argv_wf.append("--carry-position")
        if args.liquidate_end:
            argv_wf.append("--liquidate-end")

        rc = walkforward.main(argv_wf)
        if rc != 0:
            raise SystemExit(f"walkforward failed for seed {s} with rc={rc}")

        summary = json.loads((run_out / "summary.json").read_text())
        win_rows = _read_windows(run_out / "windows.csv")
        pnl = [float(r["test_pnl_bp"]) for r in win_rows]
        bL = [float(r["baseline_long_bp"]) for r in win_rows]
        bS = [float(r["baseline_short_bp"]) for r in win_rows]
        alphaL = [p - b for p, b in zip(pnl, bL)]
        alphaS = [p - b for p, b in zip(pnl, bS)]
        oracle = [max(l, s) for l, s in zip(bL, bS)]
        regret = [p - o for p, o in zip(pnl, oracle)]  # <= 0 when policy underperforms best-of-week direction
        correct = sum(1 for p, o in zip(pnl, oracle) if abs(p - o) < 1e-9)

        # Oracle constrained to the policy's *allowed* action set.
        if args.long_only and args.require_position:
            # Only +1 is feasible.
            oracle_allowed = list(bL)
        elif args.long_only:
            # Only {0,+1} feasible.
            oracle_allowed = [max(0.0, l) for l in bL]
        elif args.require_position:
            # Only {-1,+1} feasible.
            oracle_allowed = list(oracle)
        else:
            # {-1,0,+1} feasible.
            oracle_allowed = [max(0.0, l, s) for l, s in zip(bL, bS)]

        oracle_allowed_regret = [p - o for p, o in zip(pnl, oracle_allowed)]
        oracle_allowed_correct = sum(1 for p, o in zip(pnl, oracle_allowed) if abs(p - o) < 1e-9)

        # Exposure: fraction of windows where we held a non-flat position during the test.
        # Use the logged time_in_mkt (robust to carry mode, where turnover can be 0 even if long).
        tis = [float(r.get("test_time_in_mkt", 0.0) or 0.0) for r in win_rows]
        exposure_rate = float(mean(tis)) if tis else 0.0

        row: dict[str, Any] = {
            "seed": s,
            "windows": int(summary.get("total_windows", len(win_rows))),
            # Keep the old column names for compatibility with earlier analyses.
            "policy_total_bp": float(summary.get("total_test_pnl_bp", sum(pnl))),
            "policy_mean_bp_per_window": float(summary.get("mean_test_pnl_bp_per_window", mean(pnl) if pnl else 0.0)),
            "policy_std_bp_per_window": float(summary.get("std_test_pnl_bp_per_window", pstdev(pnl) if len(pnl) > 1 else 0.0)),
            "baseline_long_total_bp": float(sum(bL)),
            "baseline_short_total_bp": float(sum(bS)),
            "alpha_vs_long_total_bp": float(sum(alphaL)),
            "alpha_vs_short_total_bp": float(sum(alphaS)),
            "alpha_vs_long_mean_bp_per_window": float(mean(alphaL) if alphaL else 0.0),
            "alpha_vs_short_mean_bp_per_window": float(mean(alphaS) if alphaS else 0.0),
            "oracle_best_total_bp": float(sum(oracle)),
            "oracle_regret_total_bp": float(sum(regret)),
            "oracle_regret_mean_bp_per_window": float(mean(regret) if regret else 0.0),
            "oracle_correct_pick_rate": float(correct / len(oracle)) if oracle else 0.0,
            "exposure_rate": float(exposure_rate),
            "oracle_allowed_total_bp": float(sum(oracle_allowed)),
            "oracle_allowed_regret_total_bp": float(sum(oracle_allowed_regret)),
            "oracle_allowed_regret_mean_bp_per_window": float(mean(oracle_allowed_regret) if oracle_allowed_regret else 0.0),
            "oracle_allowed_correct_pick_rate": float(oracle_allowed_correct / len(oracle_allowed)) if oracle_allowed else 0.0,
        }

        # Risk metrics (annualized sharpe, max drawdown on cumulative PnL curve).
        row.update(_series_stats("policy", pnl, periods_per_year=periods_per_year))
        row.update(_series_stats("baseline_long", bL, periods_per_year=periods_per_year))
        row.update(_series_stats("baseline_short", bS, periods_per_year=periods_per_year))
        row.update(_series_stats("oracle_best", oracle, periods_per_year=periods_per_year))
        # Regret is always <= 0; max drawdown is still meaningful but usually less interesting.
        row.update(_series_stats("oracle_regret", regret, periods_per_year=periods_per_year))
        row.update(_series_stats("oracle_allowed", oracle_allowed, periods_per_year=periods_per_year))
        row.update(_series_stats("oracle_allowed_regret", oracle_allowed_regret, periods_per_year=periods_per_year))

        rows.append(row)
        dt_seed = time.time() - t_seed0
        done = len(rows)
        dt = time.time() - t0
        avg = dt / max(1, done)
        eta = avg * max(0, len(seeds) - done)
        print(f"[{done}/{len(seeds)}] seed={s} done in {dt_seed:.1f}s (avg {avg:.1f}s/seed, ETA {eta/60:.1f}m)")

    _write_csv(outdir / "results.csv", rows)
    # Lightweight console summary.
    # In long-only mode, alpha vs short is not a meaningful selection key.
    if args.long_only:
        key = "alpha_vs_long_total_bp"
    else:
        key = "alpha_vs_short_total_bp"
    best = max(rows, key=lambda r: float(r.get(key, -1e18)))
    print(f"Wrote sweep to {outdir}")
    print(f"Best by {key}:", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
