from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from .imports import add_maker_strat_to_path

add_maker_strat_to_path()

from data import load_markets, Market15m  # type: ignore  # noqa: E402

from .env import EpisodeEnv
from .model import PolicyConfig, PolicyNet
from .train import TrainConfig, train_epoch_batch, rollout_batch_eval, rollout_episode


def _iter_days(markets: list[Market15m]) -> list[str]:
    return sorted({m.close_date for m in markets})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Union of keys.
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

def _write_assumptions(outdir: Path, args: argparse.Namespace) -> None:
    """
    Write a machine-readable assumptions log so results can be audited/reproduced.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    assumptions = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "kalshi_paper_trader.nn_policy.walkforward",
        "args": {k: getattr(args, k) for k in vars(args)},
        "data_assumptions": [
            "Input data is Kalshi candlesticks (top-of-book bid/ask OHLC), typically 1-minute resolution.",
            "No full orderbook snapshots, queue position, or trade prints are used.",
        ],
        "execution_assumptions": [
            "Taker entries buy at candle-open ask; taker exits sell at candle-open bid.",
            "No slippage model beyond spread embedded in bid/ask and taker fee model.",
            "At most 1 entry per episode (once flat after entering, further entries are disallowed).",
            "Position can be held to settlement if not closed early (Kalshi binary payoff 0/100).",
            "If --settle-only is set, close action is disabled; only enter/hold-to-settle is possible.",
        ],
        "fee_model": {
            "description": "Approx Kalshi taker fee per contract: fee_dollars = ceil(0.07 * P * (1-P)), where P is contract price in dollars.",
            "implementation": "kalshi_paper_trader/maker-strat/sim.py::_taker_fee_cents",
        },
        "label_assumptions": [
            "Episode outcome uses Kalshi market 'result' field (yes/no) from the downloaded series data.",
        ],
        "notes": [
            "This is an offline RL-style policy learner; it is highly susceptible to overfitting and regime drift.",
            "If the best policy converges to abstaining (0 trades), that indicates no detectable edge under these assumptions.",
        ],
    }
    (outdir / "assumptions.json").write_text(json.dumps(assumptions, indent=2))


def _sample_env_batch(
    markets: list[Market15m],
    *,
    batch: int,
    lookback: int,
    max_contracts: int,
    allow_close: bool,
    features: str,
    rng: np.random.Generator,
) -> list[EpisodeEnv]:
    if not markets:
        return []
    n = min(batch, len(markets))
    # Sample without replacement if possible.
    idx = rng.choice(len(markets), size=n, replace=(len(markets) < batch))
    out = [
        EpisodeEnv(markets[int(i)], lookback=lookback, max_contracts=max_contracts, allow_close=allow_close, features=features)
        for i in idx
    ]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward backtest for a black-box NN policy on KXBTC15M candle data.")
    ap.add_argument("--data", required=True, help="Path to KXBTC15M_candles.json")
    ap.add_argument("--outdir", default="kalshi_paper_trader/nn_policy/out")

    ap.add_argument("--train-days", type=int, default=14)
    ap.add_argument("--test-days", type=int, default=3)
    ap.add_argument("--step-days", type=int, default=3)
    ap.add_argument("--max-windows", type=int, default=0,
                    help="If >0, stop after this many walk-forward windows (for fast HPO).")

    ap.add_argument("--lookback", type=int, default=5)
    ap.add_argument("--max-contracts", type=int, default=5)
    ap.add_argument("--settle-only", action="store_true",
                    help="Disable the close action. Policy can enter once and then must hold to settlement.")
    ap.add_argument("--features", default="raw", help="Feature set: raw|deltas|deltas+vol")

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-episodes", type=int, default=128)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--hidden", type=int, action="append", default=[128, 128], help="Repeatable. Example: --hidden 128 --hidden 128")

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="", help="cpu|mps|cuda. Default: mps if available else cpu.")
    args = ap.parse_args(argv)

    # Pick default device if not provided.
    if not args.device:
        args.device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    markets = load_markets(args.data)
    days_all = _iter_days(markets)
    if len(days_all) < (args.train_days + args.test_days):
        raise SystemExit("Not enough days in dataset for the chosen window sizes.")

    # Infer obs_dim from env.
    dummy_env = EpisodeEnv(
        markets[0],
        lookback=args.lookback,
        max_contracts=args.max_contracts,
        allow_close=(not args.settle_only),
        features=args.features,
    )
    obs_dim = int(dummy_env.reset().shape[0])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_assumptions(outdir, args)

    window_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    all_test_pnls: list[int] = []
    all_test_buys = 0
    all_test_closes = 0

    start = 0
    window_id = 0
    while start + args.train_days + args.test_days <= len(days_all):
        train_days = set(days_all[start : start + args.train_days])
        test_days = set(days_all[start + args.train_days : start + args.train_days + args.test_days])
        window_id += 1

        train_markets = [m for m in markets if m.close_date in train_days]
        test_markets = [m for m in markets if m.close_date in test_days]

        model = PolicyNet(
            PolicyConfig(
                obs_dim=obs_dim,
                hidden=tuple(args.hidden),
                n_actions=4,
                size_buckets=tuple(sorted({0, 1, min(2, args.max_contracts), min(3, args.max_contracts), args.max_contracts})),
            )
        )
        try:
            model = model.to(args.device)
        except Exception as e:
            # Some setups report MPS as available but fail at runtime (e.g., older macOS).
            if str(args.device).lower() == "mps":
                print(f"WARNING: failed to use MPS ({e}); falling back to CPU.")
                args.device = "cpu"
                model = model.to(args.device)
            else:
                raise
        opt = optim.Adam(model.parameters(), lr=args.lr)
        tcfg = TrainConfig(
            lr=args.lr,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            grad_clip=args.grad_clip,
            device=args.device,
        )

        # Train from scratch per window to avoid lookahead leakage.
        train_stats: list[dict[str, float]] = []
        for _ in range(max(1, args.epochs)):
            env_batch = _sample_env_batch(
                train_markets,
                batch=args.batch_episodes,
                lookback=args.lookback,
                max_contracts=args.max_contracts,
                allow_close=(not args.settle_only),
                features=args.features,
                rng=rng,
            )
            if not env_batch:
                break
            train_stats.append(train_epoch_batch(env_batch, model, opt, tcfg))

        train_mean = float(mean([s["mean_pnl_cents"] for s in train_stats])) if train_stats else 0.0

        # Deterministic eval on all test markets (batched).
        test_envs = [
            EpisodeEnv(
                m,
                lookback=args.lookback,
                max_contracts=args.max_contracts,
                allow_close=(not args.settle_only),
                features=args.features,
            )
            for m in test_markets
        ]
        test_pnls, buys, closes = rollout_batch_eval(test_envs, model, device=args.device)
        all_test_buys += buys
        all_test_closes += closes
        for env, pnl in zip(test_envs, test_pnls):
            trade_rows.append(
                {
                    "window_id": window_id,
                    "close_date": env.m.close_date,
                    "ticker": env.m.ticker,
                    "result": env.m.result,
                    "episode_pnl_cents": int(pnl),
                    "buys": int(env.buy_count),
                    "closes": int(env.close_count),
                }
            )

        test_mean = float(np.mean(test_pnls)) if test_pnls else 0.0
        test_total = int(np.sum(test_pnls)) if test_pnls else 0

        window_rows.append(
            {
                "window_id": window_id,
                "train_start": min(train_days),
                "train_end": max(train_days),
                "test_start": min(test_days),
                "test_end": max(test_days),
                "train_markets": len(train_markets),
                "test_markets": len(test_markets),
                "train_epochs": len(train_stats),
                "train_mean_pnl_cents_per_episode": train_mean,
                "test_mean_pnl_cents_per_episode": test_mean,
                "test_total_pnl_cents": test_total,
            }
        )

        all_test_pnls.extend(test_pnls)
        start += args.step_days
        if args.max_windows > 0 and window_id >= args.max_windows:
            break

    overall = {
        "total_windows": window_id,
        "total_test_episodes": len(all_test_pnls),
        "mean_test_pnl_cents_per_episode": float(mean(all_test_pnls)) if all_test_pnls else 0.0,
        "std_test_pnl_cents_per_episode": float(pstdev(all_test_pnls)) if len(all_test_pnls) > 1 else 0.0,
        "total_test_pnl_cents": int(sum(all_test_pnls)) if all_test_pnls else 0,
        "total_test_buys": int(all_test_buys),
        "total_test_closes": int(all_test_closes),
    }

    (outdir / "summary.json").write_text(json.dumps(overall, indent=2))
    _write_csv(outdir / "windows.csv", window_rows)
    _write_csv(outdir / "trades.csv", trade_rows)

    print(f"Wrote backtest to {outdir}")
    print("Overall:", overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
