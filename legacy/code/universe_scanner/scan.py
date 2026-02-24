from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .validate import validate_csv


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


def _summarize_seed_sweep(results_csv: Path) -> dict[str, Any]:
    """
    Summarize key columns as mean/median/p05/p95 over seeds.
    """
    if not results_csv.exists():
        return {"ok": False, "error": f"missing {results_csv}"}
    rows = list(csv.DictReader(results_csv.open("r", newline="")))
    if not rows:
        return {"ok": False, "error": f"empty {results_csv}"}

    def col(name: str) -> list[float]:
        out: list[float] = []
        for r in rows:
            if name not in r or r[name] == "":
                continue
            out.append(float(r[name]))
        return out

    def pct(xs: list[float], p: float) -> float:
        xs = sorted(xs)
        if not xs:
            return 0.0
        k = (len(xs) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(len(xs) - 1, lo + 1)
        w = k - lo
        return float(xs[lo] * (1.0 - w) + xs[hi] * w)

    keys = [
        "alpha_vs_long_total_bp",
        "policy_total_bp",
        "policy_sharpe",
        "policy_max_drawdown_bp",
        "policy_cagr",
        "policy_calmar",
        "exposure_rate",
    ]
    out: dict[str, Any] = {"ok": True, "n_seeds": len(rows)}
    for k in keys:
        xs = col(k)
        out[k] = {
            "mean": sum(xs) / max(1, len(xs)),
            "p50": pct(xs, 50),
            "p05": pct(xs, 5),
            "p95": pct(xs, 95),
            "min": min(xs) if xs else 0.0,
            "max": max(xs) if xs else 0.0,
        }
    # Count beats_long by alpha > 0
    alpha = col("alpha_vs_long_total_bp")
    out["beats_long_rate"] = float(sum(1 for a in alpha if a > 0.0) / max(1, len(alpha)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan a universe of assets using the crypto_policy harness.")
    ap.add_argument("--config", default="kalshi_paper_trader/universe_scanner/config.json")
    ap.add_argument("--data-dir", default="kalshi_paper_trader/crypto_policy/data",
                    help="Directory containing universal OHLCV CSVs.")
    ap.add_argument("--outdir", default="kalshi_paper_trader/universe_scanner/out")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    bar_minutes = int(cfg.get("bar_minutes", 30))
    assets = list(cfg.get("assets") or [])
    defaults = dict(cfg.get("scanner_defaults") or {})

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "assumptions.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": "kalshi_paper_trader.universe_scanner.scan",
                "config_path": str(args.config),
                "data_dir": str(args.data_dir),
                "config": cfg,
            },
            indent=2,
        )
    )

    rows: list[dict[str, Any]] = []

    # Import late to keep scan.py lightweight if user hasn't set up torch deps elsewhere.
    from kalshi_paper_trader.crypto_policy import seed_sweep  # noqa: WPS433

    for a in assets:
        aid = str(a.get("id") or "")
        rel = str(a.get("csv") or "")
        features = str(a.get("features") or "prices")
        kind = str(a.get("kind") or "")
        csv_path = Path(args.data_dir) / rel

        asset_out = outdir / aid
        asset_out.mkdir(parents=True, exist_ok=True)

        v = validate_csv(csv_path, bar_minutes=bar_minutes)
        (asset_out / "validate.json").write_text(
            json.dumps(
                {
                    "ok": v.ok,
                    "errors": v.errors,
                    "warnings": v.warnings,
                    "stats": v.stats,
                },
                indent=2,
            )
        )

        row: dict[str, Any] = {
            "asset": aid,
            "kind": kind,
            "csv": str(csv_path),
            "features": features,
            "data_ok": bool(v.ok),
            "data_errors": " | ".join(v.errors),
            "data_warnings": " | ".join(v.warnings),
        }

        if not v.ok:
            rows.append(row)
            continue

        # Run configured cost scenarios.
        cost_scenarios = list((defaults.get("cost_scenarios") or []))
        for cs in cost_scenarios:
            name = str(cs.get("name") or "cost")
            fee_bps = float(cs.get("fee_bps", 1.0))
            slippage_bps = float(cs.get("slippage_bps", 1.0))

            sweep_out = asset_out / f"sweep_{name}"
            sweep_args = [
                "--data",
                str(csv_path),
                "--outdir",
                str(sweep_out),
                "--seeds",
                str(defaults.get("seeds", "1-20")),
                "--bar-minutes",
                str(bar_minutes),
                "--features",
                str(features),
                "--bars-per-day",
                str(int(defaults.get("bars_per_day", 0) or 0)),
                "--train-days",
                str(int(defaults.get("train_days", 21))),
                "--test-days",
                str(int(defaults.get("test_days", 7))),
                "--step-days",
                str(int(defaults.get("step_days", 7))),
                "--episode-days",
                str(int(defaults.get("episode_days", 7))),
                "--lookback",
                str(int(defaults.get("lookback", 144))),
                "--fee-bps",
                str(fee_bps),
                "--slippage-bps",
                str(slippage_bps),
                "--epochs",
                str(int(defaults.get("epochs", 5))),
                "--batch-episodes",
                str(int(defaults.get("batch_episodes", 128))),
                "--lr",
                str(float(defaults.get("lr", 3e-4))),
                "--gamma",
                str(float(defaults.get("gamma", 1.0))),
                "--entropy-coef",
                str(float(defaults.get("entropy_coef", 0.01))),
                "--value-coef",
                str(float(defaults.get("value_coef", 0.5))),
                "--grad-clip",
                str(float(defaults.get("grad_clip", 1.0))),
                "--device",
                str(args.device),
            ]

            for h in list(defaults.get("hidden") or [64]):
                sweep_args.extend(["--hidden", str(int(h))])

            mode = dict(defaults.get("mode") or {})
            if mode.get("one_decision_per_episode", True):
                sweep_args.append("--one-decision-per-episode")
            if mode.get("allow_one_exit", False):
                sweep_args.append("--allow-one-exit")
            if mode.get("force_flat_at_end", False):
                sweep_args.append("--force-flat-at-end")
            if mode.get("require_position", False):
                sweep_args.append("--require-position")
            if mode.get("long_only", True):
                sweep_args.append("--long-only")
            if mode.get("carry_position", True):
                sweep_args.append("--carry-position")
            if mode.get("liquidate_end", True):
                sweep_args.append("--liquidate-end")

            rc = seed_sweep.main(sweep_args)
            row[f"{name}_rc"] = int(rc)
            if rc != 0:
                row[f"{name}_ok"] = False
                continue

            summ = _summarize_seed_sweep(sweep_out / "results.csv")
            row[f"{name}_ok"] = bool(summ.get("ok"))
            # Flatten summary fields we care about for ranking.
            for k in ["alpha_vs_long_total_bp", "policy_sharpe", "policy_max_drawdown_bp", "policy_total_bp", "policy_cagr", "policy_calmar", "exposure_rate"]:
                s = summ.get(k) or {}
                row[f"{name}_{k}_p50"] = float(s.get("p50", 0.0))
                row[f"{name}_{k}_p05"] = float(s.get("p05", 0.0))
                row[f"{name}_{k}_p95"] = float(s.get("p95", 0.0))
            row[f"{name}_beats_long_rate"] = float(summ.get("beats_long_rate", 0.0))

        rows.append(row)

    _write_csv(outdir / "universe_results.csv", rows)
    print(f"Wrote {len(rows)} rows to {outdir / 'universe_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

