from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from data import load_markets
from sim import Strategy, simulate_market


def _parse_range_list(s: str) -> list[int]:
    """
    Accept:
      - "0-10" (inclusive)
      - "0,1,2,5"
    """
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _iter_ints(expr: str) -> list[int]:
    return _parse_range_list(expr)


def _block_bootstrap_mean(trades_by_day: dict[str, list[int]], n_boot: int, *, rng: random.Random) -> dict[str, float]:
    """
    Block bootstrap on day (close_date): sample days w/ replacement, recompute mean pnl.
    PnL inputs are cents per *attempt* (including 0 for "no fill").
    """
    days = list(trades_by_day.keys())
    if not days:
        return {"p05": float("nan"), "p50": float("nan"), "p95": float("nan")}

    boot_means: list[float] = []
    for _ in range(n_boot):
        sample_days = [rng.choice(days) for _ in range(len(days))]
        pnl: list[int] = []
        for d in sample_days:
            pnl.extend(trades_by_day[d])
        boot_means.append(mean(pnl) if pnl else 0.0)

    boot_means.sort()
    def q(p: float) -> float:
        if not boot_means:
            return float("nan")
        idx = int(round(p * (len(boot_means) - 1)))
        return float(boot_means[idx])

    return {"p05": q(0.05), "p50": q(0.50), "p95": q(0.95)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Maker-first parameter scan on KXBTC15M candlestick data.")
    ap.add_argument("--data", required=True, help="Path to KXBTC15M_candles.json")
    ap.add_argument("--side", choices=["YES", "NO"], default="NO")
    ap.add_argument("--entry-minutes", default="0-10")
    ap.add_argument("--entry-improve", default="0,1,2", help="Cents added to best bid (capped below ask).")
    ap.add_argument("--ttl-minutes", default="1,2,3")
    ap.add_argument("--tp-cents", default="0,1,2,3,4,5", help="Take-profit in cents. 0 = no TP, hold to fallback.")
    ap.add_argument("--stop-minute", default="12", help="Minute index to begin stop/flatten attempts (0-14).")
    ap.add_argument("--stop-exit-ttl-minutes", default="1,2", help="Minutes to attempt maker-flatten after stop-minute.")
    ap.add_argument("--stop-exit-improve", default="0,1,2", help="Cents to improve the stop exit ask (0=join ask).")
    ap.add_argument("--max-entry-spread-cents", default="100", help="Skip entry if spread > this (cents).")
    ap.add_argument("--max-stop-spread-cents", default="100", help="Skip maker-stop attempt if spread > this (cents).")
    ap.add_argument("--fallback", choices=["settle", "taker_exit"], default="taker_exit",
                    help="If maker stop doesn't fill: settle (hold to expiry) or taker_exit (force flatten w/ fees).")
    ap.add_argument("--bootstrap", type=int, default=0, help="If >0, block bootstrap mean pnl by close-date with N resamples.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--market-sample", type=int, default=0,
                    help="If >0, randomly sample this many markets (faster iteration).")
    ap.add_argument("--random-combos", type=int, default=0,
                    help="If >0, randomly sample this many parameter combos instead of full grid.")
    ap.add_argument("--progress-every", type=int, default=0,
                    help="If >0, print progress every N parameter combos.")
    ap.add_argument("--out", default="kalshi_paper_trader/maker-strat/results.csv")
    args = ap.parse_args(argv)

    markets = load_markets(args.data)
    rng = random.Random(args.seed)
    if args.market_sample and args.market_sample > 0 and args.market_sample < len(markets):
        markets = rng.sample(markets, args.market_sample)
        print(f"Using market sample: {len(markets)} markets")

    entry_minutes = _iter_ints(args.entry_minutes)
    entry_improve = _iter_ints(args.entry_improve)
    ttl_minutes = _iter_ints(args.ttl_minutes)
    tp_cents = _iter_ints(args.tp_cents)
    stop_minutes = _iter_ints(args.stop_minute)
    stop_exit_ttl = _iter_ints(args.stop_exit_ttl_minutes)
    stop_exit_improve = _iter_ints(args.stop_exit_improve)
    max_entry_spread = _iter_ints(args.max_entry_spread_cents)
    max_stop_spread = _iter_ints(args.max_stop_spread_cents)

    combos = []
    for em in entry_minutes:
        for imp in entry_improve:
            for ttl in ttl_minutes:
                for tp in tp_cents:
                    for sm in stop_minutes:
                        for sttl in stop_exit_ttl:
                            for simp in stop_exit_improve:
                                for mes in max_entry_spread:
                                    for mss in max_stop_spread:
                                        combos.append((em, imp, ttl, tp, sm, sttl, simp, mes, mss))

    print(f"Grid combos: {len(combos)}")
    if args.random_combos and args.random_combos > 0 and args.random_combos < len(combos):
        combos = rng.sample(combos, args.random_combos)
        print(f"Using random combo sample: {len(combos)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (em, imp, ttl, tp, sm, sttl, simp, mes, mss) in enumerate(combos, start=1):
        if args.progress_every and (idx % args.progress_every == 0):
            print(f"Progress: {idx}/{len(combos)} combos")

        strat = Strategy(
            side=args.side,
            entry_minute=em,
            entry_improve_cents=imp,
            ttl_minutes=ttl,
            tp_cents=tp,
            stop_minute=sm,
            stop_exit_ttl_minutes=sttl,
            stop_exit_improve_cents=simp,
            max_entry_spread_cents=mes,
            max_stop_spread_cents=mss,
            fallback=args.fallback,
        )

        pnl_attempts: list[int] = []
        pnl_filled: list[int] = []
        entry_px_filled: list[int] = []
        fill_minute_filled: list[int] = []
        settle_pnl_filled: list[int] = []
        filled_n = 0
        exited_n = 0
        win_filled_n = 0

        trades_by_day: dict[str, list[int]] = defaultdict(list)

        for m in markets:
            tr = simulate_market(m, strat)
            if tr is None:
                continue
            pnl_attempts.append(tr.pnl_cents)
            trades_by_day[m.close_date].append(tr.pnl_cents)
            if tr.filled:
                filled_n += 1
                pnl_filled.append(tr.pnl_cents)
                entry_px_filled.append(tr.entry_px)
                fill_minute_filled.append(tr.fill_minute)
                settles_yes = (m.result == "yes")
                win = settles_yes if args.side == "YES" else (not settles_yes)
                win_filled_n += 1 if win else 0
                settle_pnl_filled.append((100 if win else 0) - tr.entry_px)
            if tr.exited:
                exited_n += 1

        if not pnl_attempts:
            continue

        r = {
            "side": args.side,
            "entry_minute": em,
            "entry_improve_cents": imp,
            "ttl_minutes": ttl,
            "tp_cents": tp,
            "stop_minute": sm,
            "stop_exit_ttl_minutes": sttl,
            "stop_exit_improve_cents": simp,
            "max_entry_spread_cents": mes,
            "max_stop_spread_cents": mss,
            "fallback": args.fallback,
            "attempts": len(pnl_attempts),
            "fills": filled_n,
            "fill_rate": filled_n / len(pnl_attempts),
            "exits": exited_n,
            "exit_rate_given_fill": (exited_n / filled_n) if filled_n else 0.0,
            "mean_entry_px_filled": mean(entry_px_filled) if entry_px_filled else 0.0,
            "mean_fill_minute": mean(fill_minute_filled) if fill_minute_filled else 0.0,
            "win_rate_given_fill": (win_filled_n / filled_n) if filled_n else 0.0,
            "mean_settle_pnl_cents_per_fill": mean(settle_pnl_filled) if settle_pnl_filled else 0.0,
            "mean_pnl_cents_per_attempt": mean(pnl_attempts),
            "mean_pnl_cents_per_fill": mean(pnl_filled) if pnl_filled else 0.0,
            "ev_cents_per_hour_per_contract": 4.0 * mean(pnl_attempts),
        }

        if args.bootstrap and args.bootstrap > 0:
            ci = _block_bootstrap_mean(trades_by_day, args.bootstrap, rng=rng)
            r["boot_p05_mean_pnl_attempt_cents"] = ci["p05"]
            r["boot_p50_mean_pnl_attempt_cents"] = ci["p50"]
            r["boot_p95_mean_pnl_attempt_cents"] = ci["p95"]

        rows.append(r)

    # Sort by best mean pnl per attempt
    rows.sort(key=lambda x: x["mean_pnl_cents_per_attempt"], reverse=True)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_path}")
    if rows:
        top = rows[0]
        print("Top row:", top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
