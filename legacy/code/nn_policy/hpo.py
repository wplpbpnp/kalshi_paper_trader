from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    """Sample from log-uniform on [lo, hi]."""
    assert lo > 0 and hi > 0 and hi >= lo
    a = math.log(lo)
    b = math.log(hi)
    return math.exp(rng.uniform(a, b))


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


@dataclass(frozen=True)
class TrialCfg:
    lr: float
    entropy_coef: float
    value_coef: float
    grad_clip: float
    lookback: int
    epochs: int
    batch_episodes: int
    max_contracts: int
    hidden: tuple[int, ...]


def _sample_trial(rng: random.Random) -> TrialCfg:
    hidden_choices = [
        # Tiny models: high bias, low variance. Often best when signal is extremely weak/noisy.
        (12,),
        (16,),
        (32,),
        (64, 64),
        (64,),
        (128, 128),
        (128,),
        (256, 256),
        (128, 64),
        (256, 128),
    ]
    lookback_choices = [3, 5, 8]
    epochs_choices = [2, 4, 6, 8]
    batch_choices = [32, 64, 128, 256]
    max_contracts_choices = [1, 2, 3, 5]

    return TrialCfg(
        lr=_log_uniform(rng, 1e-5, 3e-3),
        entropy_coef=_log_uniform(rng, 1e-4, 5e-2),
        value_coef=rng.uniform(0.1, 1.0),
        grad_clip=rng.choice([0.5, 1.0, 2.0]),
        lookback=rng.choice(lookback_choices),
        epochs=rng.choice(epochs_choices),
        batch_episodes=rng.choice(batch_choices),
        max_contracts=rng.choice(max_contracts_choices),
        hidden=rng.choice(hidden_choices),
    )


def _run_one(
    *,
    py: str,
    data: str,
    outdir: Path,
    seed: int,
    train_days: int,
    test_days: int,
    step_days: int,
    max_windows: int,
    epochs: int,
    cfg: TrialCfg,
    device: str,
    settle_only: bool,
    features: str,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        py,
        "-m",
        "kalshi_paper_trader.nn_policy.walkforward",
        "--data",
        data,
        "--outdir",
        str(outdir),
        "--train-days",
        str(train_days),
        "--test-days",
        str(test_days),
        "--step-days",
        str(step_days),
        "--max-windows",
        str(max_windows),
        "--epochs",
        str(epochs),
        "--batch-episodes",
        str(cfg.batch_episodes),
        "--lr",
        f"{cfg.lr:.8g}",
        "--entropy-coef",
        f"{cfg.entropy_coef:.8g}",
        "--value-coef",
        f"{cfg.value_coef:.8g}",
        "--grad-clip",
        f"{cfg.grad_clip:.8g}",
        "--lookback",
        str(cfg.lookback),
        "--max-contracts",
        str(cfg.max_contracts),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if settle_only:
        cmd.append("--settle-only")
    if features:
        cmd.extend(["--features", features])
    for h in cfg.hidden:
        cmd.extend(["--hidden", str(h)])

    # Silence stdout except on failure; summary.json is the interface.
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        return {
            "ok": 0,
            "seed": seed,
            "error": (r.stderr or r.stdout)[-2000:],
        }

    summ = json.loads((outdir / "summary.json").read_text())
    summ["ok"] = 1
    summ["seed"] = seed
    return summ


def _fmt_secs(s: float) -> str:
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        m = int(s // 60)
        ss = int(s % 60)
        return f"{m}m{ss:02d}s"
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h}h{m:02d}m"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Random-search HPO for nn_policy.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="kalshi_paper_trader/nn_policy/hpo_out")

    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seeds", default="1,2,3", help="Comma-separated seeds. Median across seeds is used for ranking.")

    # Fast-eval settings (use smaller windows to triage quickly).
    ap.add_argument("--train-days", type=int, default=14)
    ap.add_argument("--test-days", type=int, default=3)
    ap.add_argument("--step-days", type=int, default=3)
    ap.add_argument("--max-windows", type=int, default=4, help="Fast HPO uses only the first N windows.")

    ap.add_argument("--device", default="", help="cpu|mps|cuda. Default: mps if available else cpu.")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed for sampling hyperparams.")
    ap.add_argument("--asha", action="store_true", help="Use successive-halving/ASHA-style budgeting.")
    ap.add_argument("--eta", type=int, default=3, help="Promotion factor for ASHA (keep top 1/eta each rung).")
    ap.add_argument("--rungs", default="1:1,2:2,4:4",
                    help="Comma-separated rungs as max_windows:epochs. Example: '1:1,2:2,4:4,0:8' (0 max_windows means full).")

    ap.add_argument("--min-mean-pnl-cents", type=float, default=-5.0,
                    help="Early stop: if first seed mean pnl is below this, skip remaining seeds for the trial.")
    ap.add_argument("--progress-every", type=int, default=5,
                    help="Print progress every N completed (trial,seed) runs.")
    ap.add_argument("--settle-only", action="store_true",
                    help="Pass --settle-only to walkforward (disable close action).")
    ap.add_argument("--features", default="raw", help="Feature set passed to walkforward: raw|deltas|deltas+vol")
    ap.add_argument("--assumption-notes", default="",
                    help="Optional freeform note recorded in assumptions.json for this HPO run.")

    args = ap.parse_args(argv)

    # Default device selection (mirrors walkforward.py).
    if not args.device:
        try:
            import torch

            args.device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    py = os.path.join("kalshi_paper_trader", "venv", "bin", "python")
    if not Path(py).exists():
        raise SystemExit(f"Expected venv python at {py}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Top-level assumptions log for this HPO run.
    (outdir / "assumptions.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": "kalshi_paper_trader.nn_policy.hpo",
                "args": {k: getattr(args, k) for k in vars(args)},
                "hpo_assumptions": [
                    "Hyperparameter optimization is black-box; it cannot validate whether a discovered edge is real vs overfit.",
                    "Trials are evaluated by walk-forward backtests (see each trial's subdirectory for its own assumptions.json).",
                    "ASHA/successive-halving uses smaller budgets early (max_windows/epochs) and promotes top performers.",
                    "ETA printed during the run is an upper bound (assumes no early stops).",
                ],
                "scoring": {
                    "objective": "maximize median of mean_test_pnl_cents_per_episode across seeds",
                    "early_stop": "if first-seed mean_test_pnl_cents_per_episode < --min-mean-pnl-cents, skip remaining seeds for that trial/rung",
                },
                "note": args.assumption_notes,
            },
            indent=2,
        )
    )

    progress_path = outdir / "progress.json"
    t0 = time.time()
    completed_runs = 0
    recent_run_times: list[float] = []

    def _avg_run_s() -> float:
        return (sum(recent_run_times) / len(recent_run_times)) if recent_run_times else 0.0

    def _record_progress(payload: dict[str, Any]) -> None:
        payload = {
            "elapsed_s": time.time() - t0,
            "completed_runs": completed_runs,
            "avg_run_s_recent": _avg_run_s(),
            **payload,
        }
        progress_path.write_text(json.dumps(payload, indent=2))

    def _on_run_done(start_s: float) -> None:
        nonlocal completed_runs
        dt = time.time() - start_s
        completed_runs += 1
        recent_run_times.append(dt)
        if len(recent_run_times) > 50:
            recent_run_times.pop(0)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rng = random.Random(args.seed)

    rows: list[dict[str, Any]] = []

    # Parse rungs.
    rungs: list[tuple[int, int]] = []
    for part in args.rungs.split(","):
        part = part.strip()
        if not part:
            continue
        mw_s, ep_s = part.split(":")
        rungs.append((int(mw_s), int(ep_s)))
    if not rungs:
        rungs = [(args.max_windows, 2)]

    # Sample trial configs upfront so ASHA can promote among a fixed population.
    trials_cfg: dict[int, TrialCfg] = {tid: _sample_trial(rng) for tid in range(1, args.trials + 1)}

    def run_trial_seed(trial_id: int, rung_id: int, seed: int, max_windows: int, epochs: int) -> dict[str, Any]:
        cfg = trials_cfg[trial_id]
        mw = max_windows if max_windows != 0 else 0
        run_out = outdir / f"trial_{trial_id:04d}" / f"rung_{rung_id:02d}_mw{mw}_ep{epochs}" / f"seed_{seed}"
        return _run_one(
            py=py,
            data=args.data,
            outdir=run_out,
            seed=seed,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            max_windows=mw,
            epochs=epochs,
            cfg=cfg,
            device=args.device,
            settle_only=args.settle_only,
            features=args.features,
        )

    if args.asha:
        candidates = list(trials_cfg.keys())
        rung_scores: dict[tuple[int, int], float] = {}

        for rung_id, (mw, ep) in enumerate(rungs, start=1):
            use_seeds = [seeds[0]] if rung_id < len(rungs) else seeds
            trials_in_rung = len(candidates)
            runs_in_rung_upper = trials_in_rung * len(use_seeds)

            # Upper-bound remaining runs: assume no early-stops and standard promotions.
            future_runs_upper = 0
            future_candidates = trials_in_rung
            for next_rung_id in range(rung_id + 1, len(rungs) + 1):
                future_candidates = max(1, int(math.ceil(future_candidates / max(1, args.eta))))
                next_use_seeds = [seeds[0]] if next_rung_id < len(rungs) else seeds
                future_runs_upper += future_candidates * len(next_use_seeds)
            total_remaining_upper = runs_in_rung_upper + future_runs_upper

            _record_progress(
                {
                    "phase": "asha_rung_start",
                    "rung_id": rung_id,
                    "rung_total": len(rungs),
                    "trials_in_rung": trials_in_rung,
                    "max_windows": mw,
                    "epochs": ep,
                    "runs_in_rung_upper": runs_in_rung_upper,
                    "runs_remaining_upper": total_remaining_upper,
                }
            )

            rung_rows: list[dict[str, Any]] = []
            for trial_idx, trial_id in enumerate(candidates, start=1):
                cfg = trials_cfg[trial_id]
                per_seed = []
                for i, seed in enumerate(use_seeds):
                    run_start = time.time()
                    summ = run_trial_seed(trial_id, rung_id, seed, mw, ep)
                    per_seed.append(summ)
                    _on_run_done(run_start)

                    if args.progress_every > 0 and (completed_runs % args.progress_every) == 0:
                        avg = _avg_run_s()
                        # Upper bound ETA; early-stops will only improve it.
                        eta_upper = avg * max(0, total_remaining_upper - (trial_idx - 1) * len(use_seeds) - (i + 1))
                        print(
                            f"[ASHA] rung {rung_id}/{len(rungs)} trial {trial_idx}/{trials_in_rung} seed {i+1}/{len(use_seeds)} "
                            f"elapsed={_fmt_secs(time.time()-t0)} eta<={_fmt_secs(eta_upper)} avg_run={avg:.2f}s"
                        )
                        _record_progress(
                            {
                                "phase": "asha_running",
                                "rung_id": rung_id,
                                "rung_total": len(rungs),
                                "trial_index": trial_idx,
                                "trial_total": trials_in_rung,
                                "seed_index": i + 1,
                                "seed_total": len(use_seeds),
                                "max_windows": mw,
                                "epochs": ep,
                                "eta_upper_s": eta_upper,
                                "runs_remaining_upper": total_remaining_upper,
                            }
                        )
                    if i == 0 and summ.get("ok") == 1:
                        if float(summ.get("mean_test_pnl_cents_per_episode", 0.0)) < args.min_mean_pnl_cents:
                            break

                ok = [s for s in per_seed if s.get("ok") == 1]
                mean_pnls = [float(s.get("mean_test_pnl_cents_per_episode", 0.0)) for s in ok]
                score = float(median(mean_pnls)) if mean_pnls else float("-inf")
                rung_scores[(trial_id, rung_id)] = score

                rung_rows.append(
                    {
                        "trial_id": trial_id,
                        "rung": rung_id,
                        "max_windows": mw,
                        "epochs": ep,
                        "ok_seeds": len(ok),
                        "seed_means": ",".join(f"{x:.4f}" for x in mean_pnls),
                        "median_mean_pnl_cents_per_episode": score,
                        "lr": cfg.lr,
                        "entropy_coef": cfg.entropy_coef,
                        "value_coef": cfg.value_coef,
                        "grad_clip": cfg.grad_clip,
                        "lookback": cfg.lookback,
                        "batch_episodes": cfg.batch_episodes,
                        "max_contracts": cfg.max_contracts,
                        "hidden": "-".join(str(h) for h in cfg.hidden),
                    }
                )

            # Write per-rung snapshot.
            rung_rows_sorted = sorted(rung_rows, key=lambda r: float(r["median_mean_pnl_cents_per_episode"]), reverse=True)
            _write_csv(outdir / f"rung_{rung_id:02d}_results.csv", rung_rows_sorted)

            # Promote top fraction unless final rung.
            if rung_id < len(rungs):
                keep = max(1, int(math.ceil(len(candidates) / max(1, args.eta))))
                candidates = [r["trial_id"] for r in rung_rows_sorted[:keep]]

        # Final results = last rung only.
        final_rung = len(rungs)
        rows = []
        for trial_id in trials_cfg:
            cfg = trials_cfg[trial_id]
            score = rung_scores.get((trial_id, final_rung), float("-inf"))
            rows.append(
                {
                    "trial_id": trial_id,
                    "median_mean_pnl_cents_per_episode": score,
                    "lr": cfg.lr,
                    "entropy_coef": cfg.entropy_coef,
                    "value_coef": cfg.value_coef,
                    "grad_clip": cfg.grad_clip,
                    "lookback": cfg.lookback,
                    "batch_episodes": cfg.batch_episodes,
                    "max_contracts": cfg.max_contracts,
                    "hidden": "-".join(str(h) for h in cfg.hidden),
                }
            )
        rows_sorted = sorted(rows, key=lambda r: float(r["median_mean_pnl_cents_per_episode"]), reverse=True)
        _write_csv(outdir / "results.csv", rows_sorted)
    else:
        total_runs_upper = args.trials * len(seeds)
        _record_progress({"phase": "random_start", "total_runs_upper": total_runs_upper})
        for trial_id in range(1, args.trials + 1):
            cfg = trials_cfg[trial_id]

            per_seed: list[dict[str, Any]] = []
            for i, seed in enumerate(seeds):
                run_out = outdir / f"trial_{trial_id:04d}" / f"seed_{seed}"
                run_start = time.time()
                summ = _run_one(
                    py=py,
                    data=args.data,
                    outdir=run_out,
                    seed=seed,
                    train_days=args.train_days,
                    test_days=args.test_days,
                    step_days=args.step_days,
                    max_windows=args.max_windows,
                    epochs=cfg.epochs,
                    cfg=cfg,
                    device=args.device,
                    settle_only=args.settle_only,
                    features=args.features,
                )
                _on_run_done(run_start)
                if args.progress_every > 0 and (completed_runs % args.progress_every) == 0:
                    avg = _avg_run_s()
                    eta_upper = avg * max(0, total_runs_upper - completed_runs)
                    print(
                        f"[HPO] trial {trial_id}/{args.trials} seed {i+1}/{len(seeds)} "
                        f"elapsed={_fmt_secs(time.time()-t0)} eta<={_fmt_secs(eta_upper)} avg_run={avg:.2f}s"
                    )
                    _record_progress(
                        {
                            "phase": "random_running",
                            "trial_id": trial_id,
                            "trial_total": args.trials,
                            "seed_index": i + 1,
                            "seed_total": len(seeds),
                            "eta_upper_s": eta_upper,
                            "total_runs_upper": total_runs_upper,
                        }
                    )
                per_seed.append(summ)

                # Early stop after the first seed if it's clearly bad.
                if i == 0 and summ.get("ok") == 1:
                    if float(summ.get("mean_test_pnl_cents_per_episode", 0.0)) < args.min_mean_pnl_cents:
                        break

            ok_seeds = [s for s in per_seed if s.get("ok") == 1]
            mean_pnls = [float(s.get("mean_test_pnl_cents_per_episode", 0.0)) for s in ok_seeds]
            total_pnls = [int(s.get("total_test_pnl_cents", 0)) for s in ok_seeds]
            buys = [int(s.get("total_test_buys", 0)) for s in ok_seeds]

            row: dict[str, Any] = {
                "trial_id": trial_id,
                "ok_seeds": len(ok_seeds),
                "seed_means": ",".join(f"{x:.4f}" for x in mean_pnls),
                "seed_totals": ",".join(str(x) for x in total_pnls),
                "seed_buys": ",".join(str(x) for x in buys),
                "median_mean_pnl_cents_per_episode": float(median(mean_pnls)) if mean_pnls else float("-inf"),
                "mean_mean_pnl_cents_per_episode": float(mean(mean_pnls)) if mean_pnls else float("-inf"),
                "lr": cfg.lr,
                "entropy_coef": cfg.entropy_coef,
                "value_coef": cfg.value_coef,
                "grad_clip": cfg.grad_clip,
                "lookback": cfg.lookback,
                "epochs": cfg.epochs,
                "batch_episodes": cfg.batch_episodes,
                "max_contracts": cfg.max_contracts,
                "hidden": "-".join(str(h) for h in cfg.hidden),
            }
            rows.append(row)

            # Keep a running leaderboard.
            rows_sorted = sorted(rows, key=lambda r: float(r["median_mean_pnl_cents_per_episode"]), reverse=True)
            _write_csv(outdir / "results.csv", rows_sorted)

    print(f"Wrote HPO results to {outdir / 'results.csv'}")
    if rows:
        best = max(rows, key=lambda r: float(r["median_mean_pnl_cents_per_episode"]))
        print("Best trial (by median mean pnl):", {k: best[k] for k in best if k not in ("seed_means", "seed_totals", "seed_buys")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
