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
import math

from .data import load_bars_csv, iter_days
from .env import SeriesEnv, build_series_cache
from .model import PolicyConfig, PolicyNet
from .train import TrainConfig, train_epoch_batch, eval_policy


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


def _day_to_indices(bars) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, b in enumerate(bars):
        out.setdefault(b.date_utc, []).append(i)
    return out


def _sample_episode_starts(day_to_idx: dict[str, list[int]], days: list[str], episode_bars: int) -> list[int]:
    starts: list[int] = []
    for d in days:
        idxs = day_to_idx.get(d, [])
        if not idxs:
            continue
        # allow starting anywhere within this day, as long as we have episode_bars ahead in the global index space.
        for i in idxs:
            starts.append(i)
    # We'll filter invalid starts at env creation time.
    return starts


def _write_assumptions(outdir: Path, args: argparse.Namespace) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "assumptions.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": "kalshi_paper_trader.crypto_policy.walkforward",
                "args": {k: getattr(args, k) for k in vars(args)},
                "data_assumptions": [
                    "Input data is exchange OHLCV (Binance /api/v3/klines) at a fixed interval.",
                    "Bars are assumed to be time-ordered; missing bars are not repaired.",
                    "bars_per_day is inferred as the median daily bar count unless overridden with --bars-per-day.",
                ],
                "execution_assumptions": [
                    "Trades are modeled at bar closes using close-to-close *log returns* (no intrabar fills).",
                    "Position is 1x notional in {-1,0,+1}; no leverage or margin model.",
                    "Transaction costs are linear in turnover: (fee_bps + slippage_bps) * abs(delta_pos), applied multiplicatively via log(1 - cost_frac).",
                    "Rewards and PnL are reported in basis points (bp) of log-wealth change (additive under compounding).",
                    "If --one-decision-per-episode is set, the policy can choose the position only once at the start of each episode, then must hold; optionally it may exit to flat once if --allow-one-exit is set.",
                    "If --force-flat-at-end is set, the environment liquidates any open position at the episode end and charges one additional turnover cost.",
                    "If --require-position is set (and --one-decision-per-episode), the initial decision cannot stay flat; it must choose long or short.",
                    "If --long-only is set, the policy cannot go short (pos in {0,+1}).",
                    "If --carry-position is set, evaluation carries positions across consecutive test windows and charges costs only when the desired position changes (plus an optional final liquidation at the end of the full walk-forward backtest).",
                ],
                "notes": [
                    "This is a toy RL backtest harness, meant to test whether longer-horizon OHLCV contains learnable signal after costs.",
                ],
            },
            indent=2,
        )
    )


def _baseline_pnl_bp(
    bars,
    *,
    start_idx: int,
    n_bars: int,
    pos: int,
    fee_bps: float,
    slippage_bps: float,
    force_flat_at_end: bool,
) -> float:
    """
    Baseline: enter to target pos at start, hold through window, never close.
    PnL in bp of log-wealth using close-to-close log returns (compounding-consistent).
    """
    if n_bars <= 1:
        return 0.0
    end = min(len(bars), start_idx + n_bars)
    close = np.asarray([b.close for b in bars], dtype=np.float32)
    # Geometric (buy-and-hold) log return over the slice.
    start_px = float(close[start_idx])
    end_px = float(close[end - 1])
    if start_px <= 0 or end_px <= 0:
        return 0.0
    pnl = float(pos) * float(np.log(end_px / start_px)) * 1e4

    # Costs are applied multiplicatively; convert linear bp to log-wealth bp.
    def _cost_log_bp(turns: int) -> float:
        c_bp = float(fee_bps + slippage_bps) * float(turns)
        c_frac = min(0.999999, max(0.0, c_bp / 1e4))
        return float(np.log1p(-c_frac) * 1e4)

    # Enter once from flat; optionally liquidate at the end.
    pnl += _cost_log_bp(abs(pos))
    if force_flat_at_end and pos != 0:
        pnl += _cost_log_bp(abs(pos))
    return float(pnl)


def _cost_log_bp_from_bps(cost_bps: float) -> float:
    """
    Convert a linear cost in bp to a log-wealth change in bp using log(1 - cost_frac).
    """
    c_frac = min(0.999999, max(0.0, float(cost_bps) / 1e4))
    return float(math.log1p(-c_frac) * 1e4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward NN policy backtest on exchange OHLCV bars.")
    ap.add_argument("--data", required=True, help="CSV path from download_binance_klines.py")
    ap.add_argument("--outdir", default="kalshi_paper_trader/crypto_policy/out")
    ap.add_argument("--bar-minutes", type=int, default=30)
    ap.add_argument("--features", default="basic", help="Feature set: basic|prices")
    ap.add_argument("--bars-per-day", type=int, default=0,
                    help="Override inferred bars/day (useful for non-24/7 venues). 0 = infer from data (median daily bar count).")

    ap.add_argument("--train-days", type=int, default=21)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--step-days", type=int, default=7)

    ap.add_argument("--episode-days", type=int, default=7, help="Episode length for training/eval")
    ap.add_argument("--lookback", type=int, default=16)

    ap.add_argument("--fee-bps", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=1.0)

    ap.add_argument("--one-decision-per-episode", action="store_true",
                    help="Constrain policy to choose position once at episode start, then hold (optionally allow one exit).")
    ap.add_argument("--allow-one-exit", action="store_true",
                    help="Only relevant with --one-decision-per-episode. Allow a single exit to flat after the initial decision.")
    ap.add_argument("--force-flat-at-end", action="store_true",
                    help="Liquidate any open position at the episode end and charge an additional turnover cost.")
    ap.add_argument("--require-position", action="store_true",
                    help="Only relevant with --one-decision-per-episode. Disallow staying flat on the initial decision (must choose long or short).")
    ap.add_argument("--long-only", action="store_true",
                    help="Disallow shorts (pos in {0,+1}). Useful for testing risk-reduction policies vs buy-and-hold.")
    ap.add_argument("--carry-position", action="store_true",
                    help="Carry the position across consecutive test windows (week-to-week) instead of forcing a close each window. Costs apply only on position changes and (optionally) final liquidation.")
    ap.add_argument("--liquidate-end", action="store_true",
                    help="When --carry-position is set, liquidate any open position at the end of the full walk-forward backtest (charged once).")

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-episodes", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, action="append", default=[64], help="Repeatable")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    if args.carry_position and not args.one_decision_per_episode:
        raise SystemExit("--carry-position requires --one-decision-per-episode (weekly decisions).")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    bars = load_bars_csv(args.data)
    days = iter_days(bars)
    if len(days) < (args.train_days + args.test_days):
        raise SystemExit("Not enough days in dataset for chosen window sizes.")

    # For non-24/7 venues, bar_minutes->24h bars/day is wrong. Infer from data unless overridden.
    if args.bars_per_day and int(args.bars_per_day) > 0:
        bars_per_day = int(args.bars_per_day)
    else:
        by_day: dict[str, int] = {}
        for b in bars:
            by_day[b.date_utc] = by_day.get(b.date_utc, 0) + 1
        counts = sorted(c for c in by_day.values() if c > 0)
        bars_per_day = int(counts[len(counts) // 2]) if counts else int(round((24 * 60) / max(1, args.bar_minutes)))

    cache = build_series_cache(bars, bar_minutes=args.bar_minutes, bars_per_day=bars_per_day)

    episode_bars = int(args.episode_days * bars_per_day)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_assumptions(outdir, args)

    day_to_idx = _day_to_indices(bars)

    # Infer obs_dim
    dummy = SeriesEnv(
        bars,
        start=0,
        length=min(len(bars), episode_bars),
        bar_minutes=args.bar_minutes,
        lookback=args.lookback,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        features=args.features,
        one_decision_per_episode=args.one_decision_per_episode,
        allow_one_exit=args.allow_one_exit,
        force_flat_at_end=args.force_flat_at_end,
        require_position=args.require_position,
        long_only=args.long_only,
        cache=cache,
    )
    obs_dim = int(dummy.reset().shape[0])

    window_rows: list[dict[str, Any]] = []
    all_test_pnls: list[float] = []

    # For carry-mode evaluation.
    carry_pos = 0
    carry_cost_bps = float(args.fee_bps + args.slippage_bps)
    carry_last_row_idx: int | None = None

    start_day = 0
    window_id = 0
    while start_day + args.train_days + args.test_days <= len(days):
        train_days = days[start_day : start_day + args.train_days]
        test_days = days[start_day + args.train_days : start_day + args.train_days + args.test_days]
        window_id += 1

        # Build model fresh per window to avoid lookahead.
        model = PolicyNet(PolicyConfig(obs_dim=obs_dim, hidden=tuple(args.hidden), n_actions=4)).to(args.device)
        opt = optim.Adam(model.parameters(), lr=args.lr)
        tcfg = TrainConfig(
            lr=args.lr,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            grad_clip=args.grad_clip,
            device=args.device,
        )

        # Train by sampling random episode starts within train days.
        starts = _sample_episode_starts(day_to_idx, train_days, episode_bars)
        # Prevent lookahead leakage: every training episode must be fully contained
        # within the train period, not crossing into the test days.
        train_end_idx = max(day_to_idx[train_days[-1]])
        starts = [s for s in starts if (int(s) + episode_bars - 1) <= train_end_idx]
        for _ in range(max(1, args.epochs)):
            if not starts:
                break
            batch = rng.choice(starts, size=min(args.batch_episodes, len(starts)), replace=(len(starts) < args.batch_episodes))
            envs: list[SeriesEnv] = []
            for s in batch:
                if s + episode_bars >= len(bars):
                    continue
                envs.append(
                    SeriesEnv(
                        bars,
                        start=int(s),
                        length=episode_bars,
                        bar_minutes=args.bar_minutes,
                        lookback=args.lookback,
                        fee_bps=args.fee_bps,
                        slippage_bps=args.slippage_bps,
                        features=args.features,
                        one_decision_per_episode=args.one_decision_per_episode,
                        allow_one_exit=args.allow_one_exit,
                        force_flat_at_end=args.force_flat_at_end,
                        require_position=args.require_position,
                        long_only=args.long_only,
                        cache=cache,
                    )
                )
            if not envs:
                break
            train_epoch_batch(envs, model, opt, tcfg)

        # Evaluate.
        test_start_idx = min(day_to_idx[test_days[0]])
        test_len_bars = len(test_days) * bars_per_day
        test_len_bars = min(test_len_bars, len(bars) - test_start_idx)
        if args.carry_position:
            # Decision is made from a flat starting state (pos=0), then applied to the
            # carried position across weeks. This avoids forcing weekly liquidation.
            env_dec = SeriesEnv(
                bars,
                start=int(test_start_idx),
                length=int(test_len_bars),
                bar_minutes=args.bar_minutes,
                lookback=args.lookback,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                features=args.features,
                one_decision_per_episode=True,
                allow_one_exit=False,
                force_flat_at_end=False,
                require_position=args.require_position,
                long_only=args.long_only,
                cache=cache,
            )
            obs = env_dec.reset()
            with torch.no_grad():
                o = torch.from_numpy(obs).to(args.device)
                logits, _v = model(o.unsqueeze(0))
                logits = logits.squeeze(0)
                mask = torch.from_numpy(env_dec.valid_action_mask()).to(args.device)
                logits = logits.masked_fill(~mask, -1e9)
                a = int(torch.argmax(logits).item())

            # Desired position from a flat baseline.
            desired = 0
            if a == 1:
                desired = 1
            elif a == 2:
                desired = -1
            elif a in (0, 3):
                desired = 0
            if args.long_only and desired == -1:
                desired = 0

            # Log return over the full test window.
            end_idx = min(cache.close.shape[0] - 1, int(test_start_idx) + int(test_len_bars) - 1)
            start_px = float(cache.close[int(test_start_idx)])
            end_px = float(cache.close[int(end_idx)])
            if start_px > 0 and end_px > 0:
                window_log_bp = float(desired) * float(math.log(end_px / start_px)) * 1e4
            else:
                window_log_bp = 0.0

            # Transaction cost only if we change position vs carried pos.
            turnover = abs(int(desired) - int(carry_pos))
            cost_log_bp = _cost_log_bp_from_bps(carry_cost_bps * float(turnover)) if turnover else 0.0
            pnl = float(window_log_bp + cost_log_bp)
            carry_pos = int(desired)

            res = {
                "steps": float(test_len_bars),
                "turns": float(1 if turnover else 0),
                "turn_rate": float((1 if turnover else 0) / max(1, int(test_len_bars))),
                "time_in_mkt": float(1.0 if desired != 0 else 0.0),
                "time_long": float(1.0 if desired == 1 else 0.0),
                "time_short": float(1.0 if desired == -1 else 0.0),
                "turnover": float(turnover),
                "cost_log_bp": float(cost_log_bp),
                "desired_pos": float(desired),
                "pos_after": float(carry_pos),
            }
            all_test_pnls.append(pnl)
        else:
            # One episode over the entire test window (starting at first bar of test_start day).
            env_test = SeriesEnv(
                bars,
                start=int(test_start_idx),
                length=int(test_len_bars),
                bar_minutes=args.bar_minutes,
                lookback=args.lookback,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                features=args.features,
                one_decision_per_episode=args.one_decision_per_episode,
                allow_one_exit=args.allow_one_exit,
                force_flat_at_end=args.force_flat_at_end,
                require_position=args.require_position,
                long_only=args.long_only,
                cache=cache,
            )
            res = eval_policy(env_test, model, device=args.device)
            pnl = float(res["episode_pnl_bp"])
            all_test_pnls.append(pnl)

        # Baselines for context.
        b_flat = 0.0
        if args.carry_position:
            # Carry-style baselines: pay costs only when the baseline position changes.
            # For these baselines, desired position is constant per window.
            end_idx = min(cache.close.shape[0] - 1, int(test_start_idx) + int(test_len_bars) - 1)
            start_px = float(cache.close[int(test_start_idx)])
            end_px = float(cache.close[int(end_idx)])
            if start_px > 0 and end_px > 0:
                base_log_bp = float(math.log(end_px / start_px)) * 1e4
            else:
                base_log_bp = 0.0

            # Long baseline (always +1)
            if window_id == 1:
                base_long_pos = 0
                base_short_pos = 0
            else:
                base_long_pos = int(window_rows[-1].get("_base_long_pos_after", 1))
                base_short_pos = int(window_rows[-1].get("_base_short_pos_after", -1))

            long_turn = abs(1 - base_long_pos)
            long_cost = _cost_log_bp_from_bps(carry_cost_bps * float(long_turn)) if long_turn else 0.0
            b_long = float(base_log_bp + long_cost)

            # Short baseline (always -1), unless long_only.
            if args.long_only:
                b_short = 0.0
            else:
                short_turn = abs(-1 - base_short_pos)
                short_cost = _cost_log_bp_from_bps(carry_cost_bps * float(short_turn)) if short_turn else 0.0
                b_short = float(-base_log_bp + short_cost)
        else:
            b_long = _baseline_pnl_bp(
                bars,
                start_idx=test_start_idx,
                n_bars=test_len_bars,
                pos=+1,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                force_flat_at_end=args.force_flat_at_end,
            )
            b_short = _baseline_pnl_bp(
                bars,
                start_idx=test_start_idx,
                n_bars=test_len_bars,
                pos=-1,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                force_flat_at_end=args.force_flat_at_end,
            )

        window_rows.append(
            {
                "window_id": window_id,
                "train_start": train_days[0],
                "train_end": train_days[-1],
                "test_start": test_days[0],
                "test_end": test_days[-1],
                "test_pnl_bp": pnl,
                "test_steps": int(res.get("steps", 0.0)),
                "test_turns": int(res.get("turns", 0.0)),
                "test_turn_rate": float(res.get("turn_rate", 0.0)),
                "test_time_in_mkt": float(res.get("time_in_mkt", 0.0)),
                "test_time_long": float(res.get("time_long", 0.0)),
                "test_time_short": float(res.get("time_short", 0.0)),
                "test_desired_pos": float(res.get("desired_pos", 0.0)),
                "test_pos_after": float(res.get("pos_after", 0.0)),
                "test_turnover": float(res.get("turnover", 0.0)),
                "test_cost_log_bp": float(res.get("cost_log_bp", 0.0)),
                "test_days": len(test_days),
                "bars_per_day": bars_per_day,
                "test_len_bars": test_len_bars,
                "baseline_flat_bp": b_flat,
                "baseline_long_bp": b_long,
                "baseline_short_bp": b_short,
                "_base_long_pos_after": 1 if args.carry_position else 0,
                "_base_short_pos_after": (-1 if (args.carry_position and not args.long_only) else 0),
            }
        )
        carry_last_row_idx = len(window_rows) - 1

        start_day += args.step_days

    # Final liquidation in carry mode (charged once).
    if args.carry_position and args.liquidate_end and window_rows:
        # Apply liquidation costs to the last window so totals are realized and comparable.
        if carry_pos != 0:
            liq_turn = abs(int(carry_pos))
            liq_cost = _cost_log_bp_from_bps(carry_cost_bps * float(liq_turn))
            window_rows[carry_last_row_idx]["test_pnl_bp"] = float(window_rows[carry_last_row_idx]["test_pnl_bp"]) + liq_cost
            window_rows[carry_last_row_idx]["test_turnover"] = float(window_rows[carry_last_row_idx].get("test_turnover", 0.0)) + float(liq_turn)
            window_rows[carry_last_row_idx]["test_cost_log_bp"] = float(window_rows[carry_last_row_idx].get("test_cost_log_bp", 0.0)) + liq_cost
            all_test_pnls[-1] = float(all_test_pnls[-1]) + liq_cost
            carry_pos = 0

        # Baseline long liquidation (always +1 in carry mode).
        liq_cost = _cost_log_bp_from_bps(carry_cost_bps * 1.0)
        window_rows[carry_last_row_idx]["baseline_long_bp"] = float(window_rows[carry_last_row_idx]["baseline_long_bp"]) + liq_cost

        # Baseline short liquidation if enabled.
        if not args.long_only:
            window_rows[carry_last_row_idx]["baseline_short_bp"] = float(window_rows[carry_last_row_idx]["baseline_short_bp"]) + liq_cost

    overall = {
        "total_windows": window_id,
        "mean_test_pnl_bp_per_window": float(mean(all_test_pnls)) if all_test_pnls else 0.0,
        "std_test_pnl_bp_per_window": float(pstdev(all_test_pnls)) if len(all_test_pnls) > 1 else 0.0,
        "total_test_pnl_bp": float(sum(all_test_pnls)) if all_test_pnls else 0.0,
    }
    (outdir / "summary.json").write_text(json.dumps(overall, indent=2))
    _write_csv(outdir / "windows.csv", window_rows)

    print(f"Wrote backtest to {outdir}")
    print("Overall:", overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
