from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from statistics import mean
from typing import Any, Literal, Optional

import numpy as np

from data import load_markets
from sim import _taker_fee_cents


Side = Literal["YES", "NO"]


def _bucket_floor(x: float, step: int) -> int:
    return int(np.floor(x / step) * step)


def _split_days(markets, train_frac: float):
    # Sort by close_time (string is ISO Z; safe lexicographic order).
    ms = sorted(markets, key=lambda m: m.close_time)
    days = sorted({m.close_date for m in ms})
    if not days:
        return set(), set()
    cut = int(round(train_frac * len(days)))
    cut = max(1, min(len(days) - 1, cut))
    train_days = set(days[:cut])
    test_days = set(days[cut:])
    return train_days, test_days


@dataclass
class Obs:
    close_date: str
    ticker: str
    minute: int
    yes_bid: int
    yes_ask: int
    mid: float
    spread: int
    y: int  # 1 if YES wins else 0


def _collect_observations(markets) -> list[Obs]:
    obs: list[Obs] = []
    for m in markets:
        y = 1 if m.settles_yes() else 0
        for minute, c in enumerate(m.candles):
            yb = c.yes_bid.open
            ya = c.yes_ask.open
            if yb is None or ya is None:
                continue
            if ya <= yb:
                continue
            sp = int(ya - yb)
            obs.append(
                Obs(
                    close_date=m.close_date,
                    ticker=m.ticker,
                    minute=minute,
                    yes_bid=int(yb),
                    yes_ask=int(ya),
                    mid=(yb + ya) / 2.0,
                    spread=sp,
                    y=y,
                )
            )
    return obs


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _spread_stats_by_minute(obs: list[Obs]) -> list[dict[str, Any]]:
    by = defaultdict(list)
    for o in obs:
        by[o.minute].append(o.spread)
    out = []
    for minute in sorted(by):
        sp = np.array(by[minute], dtype=float)
        out.append(
            {
                "minute": minute,
                "n": int(len(sp)),
                "p50": float(np.quantile(sp, 0.50)),
                "p75": float(np.quantile(sp, 0.75)),
                "p90": float(np.quantile(sp, 0.90)),
                "p95": float(np.quantile(sp, 0.95)),
                "p99": float(np.quantile(sp, 0.99)),
                "max": float(sp.max()),
            }
        )
    return out


def _calibration_surface(
    obs: list[Obs],
    *,
    mid_bin_cents: int,
    minute_bin: int,
    max_spread_cents: int,
    days_filter: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """
    For each (minute_bin, mid_bin), compute empirical win-rate and mispricing vs mid.
    Mispricing = win_rate - avg_mid/100.
    """
    buckets = defaultdict(list)
    for o in obs:
        if days_filter is not None and o.close_date not in days_filter:
            continue
        if o.spread > max_spread_cents:
            continue
        mb = _bucket_floor(o.minute, minute_bin)
        pb = _bucket_floor(o.mid, mid_bin_cents)
        buckets[(mb, pb)].append(o)

    out = []
    for (mb, pb), xs in buckets.items():
        ys = [x.y for x in xs]
        avg_mid = mean([x.mid for x in xs]) / 100.0
        avg_spread = mean([x.spread for x in xs])
        wr = mean(ys)
        out.append(
            {
                "minute_bin": int(mb),
                "mid_bin_cents": int(pb),
                "n": int(len(xs)),
                "win_rate_yes": float(wr),
                "avg_mid": float(avg_mid),
                "avg_spread_cents": float(avg_spread),
                "mispricing_yes_minus_mid": float(wr - avg_mid),
            }
        )

    out.sort(key=lambda r: (r["minute_bin"], r["mid_bin_cents"]))
    return out


def _taker_yes_ev_surface(
    markets,
    *,
    mid_bin_cents: int,
    minute_bin: int,
    max_spread_cents: int,
    days_filter: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """
    For each (minute_bin, ask_price_bin), compute taker EV if you buy YES at yes_ask.open.
    EV per contract (cents):
      pnl = 100*y - ask - taker_fee(ask)
    """
    buckets: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    # store tuples: (pnl_cents, y, ask, spread)

    for m in markets:
        if days_filter is not None and m.close_date not in days_filter:
            continue
        y = 1 if m.settles_yes() else 0
        for minute, c in enumerate(m.candles):
            yb = c.yes_bid.open
            ya = c.yes_ask.open
            if yb is None or ya is None or ya <= yb:
                continue
            sp = int(ya - yb)
            if sp > max_spread_cents:
                continue
            ask = int(ya)
            fee = _taker_fee_cents(ask)
            pnl = 100 * y - ask - fee
            mb = _bucket_floor(minute, minute_bin)
            pb = _bucket_floor(ask, mid_bin_cents)
            buckets[(mb, pb)].append((pnl, y, ask, sp))

    out = []
    for (mb, pb), xs in buckets.items():
        pnls = [p for (p, _, _, _) in xs]
        ys = [yy for (_, yy, _, _) in xs]
        asks = [a for (_, _, a, _) in xs]
        sps = [s for (_, _, _, s) in xs]
        out.append(
            {
                "minute_bin": int(mb),
                "ask_bin_cents": int(pb),
                "n": int(len(xs)),
                "win_rate_yes": float(mean(ys)),
                "avg_ask_cents": float(mean(asks)),
                "avg_spread_cents": float(mean(sps)),
                "mean_taker_pnl_cents": float(mean(pnls)),
            }
        )
    out.sort(key=lambda r: (r["minute_bin"], r["ask_bin_cents"]))
    return out


def _maker_yes_ev_surface(
    markets,
    *,
    mid_bin_cents: int,
    minute_bin: int,
    max_spread_cents: int,
    entry_improve_cents: int,
    ttl_minutes: int,
    days_filter: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """
    For each (minute_bin, entry_price_bin), compute maker-entry EV if you:
      - post YES bid at (best_bid_open + improve), capped below ask
      - get filled if ask_low <= entry_px within TTL window
      - if filled, hold to settlement
      - if not filled, pnl=0

    Outputs both:
      - mean_pnl_per_attempt (includes 0 for no-fill)
      - mean_pnl_per_fill
      - win_rate_given_fill
      - fill_rate
    """
    if ttl_minutes <= 0:
        return []

    buckets = defaultdict(list)
    # store tuples: (pnl_attempt, filled, pnl_fill, win, entry_px, spread)
    for m in markets:
        if days_filter is not None and m.close_date not in days_filter:
            continue
        y = 1 if m.settles_yes() else 0
        n = len(m.candles)
        for minute in range(n):
            c0 = m.candles[minute]
            bid0 = c0.yes_bid.open
            ask0 = c0.yes_ask.open
            if bid0 is None or ask0 is None or ask0 <= bid0:
                continue
            sp = int(ask0 - bid0)
            if sp > max_spread_cents:
                continue
            entry_px = int(min(bid0 + entry_improve_cents, ask0 - 1))
            entry_px = max(1, min(99, entry_px))
            end_fill = min(n - 1, minute + ttl_minutes - 1)
            filled = False
            for t in range(minute, end_fill + 1):
                ask_low = m.candles[t].yes_ask.low
                if ask_low is None:
                    continue
                if ask_low <= entry_px:
                    filled = True
                    break
            if filled:
                pnl_fill = 100 * y - entry_px
                pnl_attempt = pnl_fill
                win = y
            else:
                pnl_fill = 0
                pnl_attempt = 0
                win = 0

            mb = _bucket_floor(minute, minute_bin)
            pb = _bucket_floor(entry_px, mid_bin_cents)
            buckets[(mb, pb)].append((pnl_attempt, filled, pnl_fill, win, entry_px, sp))

    out = []
    for (mb, pb), xs in buckets.items():
        pnls_attempt = [p for (p, _, _, _, _, _) in xs]
        fills = [f for (_, f, _, _, _, _) in xs]
        pnls_fill = [pf for (_, f, pf, _, _, _) in xs if f]
        wins = [w for (_, f, _, w, _, _) in xs if f]
        entry_pxs = [ep for (_, _, _, _, ep, _) in xs if ep is not None]
        sps = [s for (_, _, _, _, _, s) in xs]
        fill_n = sum(1 for f in fills if f)
        out.append(
            {
                "minute_bin": int(mb),
                "entry_px_bin_cents": int(pb),
                "n": int(len(xs)),
                "fill_rate": float(fill_n / len(xs)) if xs else 0.0,
                "win_rate_given_fill": float(mean(wins)) if wins else 0.0,
                "avg_entry_px_cents": float(mean(entry_pxs)) if entry_pxs else 0.0,
                "avg_spread_cents": float(mean(sps)) if sps else 0.0,
                "mean_maker_settle_pnl_cents_per_attempt": float(mean(pnls_attempt)) if pnls_attempt else 0.0,
                "mean_maker_settle_pnl_cents_per_fill": float(mean(pnls_fill)) if pnls_fill else 0.0,
            }
        )
    out.sort(key=lambda r: (r["minute_bin"], r["entry_px_bin_cents"]))
    return out


def _maker_entry_toxicity(
    markets,
    *,
    side: Side,
    entry_minutes: list[int],
    entry_improve_cents: list[int],
    ttl_minutes: list[int],
    max_entry_spread_cents: int,
) -> list[dict[str, Any]]:
    """
    Evaluate maker-entry-only (hold to settle) to quantify fill toxicity:
      - fill_rate
      - win_rate_given_fill
      - mean_settle_pnl_cents_per_fill
      - mean_settle_pnl_cents_per_attempt (EV per market "attempt")

    Fill model: you post a bid at entry_px (best_bid+improve, capped below ask),
    and you're filled if ask_low <= entry_px within TTL window.
    """
    out = []
    for em in entry_minutes:
        for imp in entry_improve_cents:
            for ttl in ttl_minutes:
                pnls_attempt: list[int] = []
                pnls_fill: list[int] = []
                wins: int = 0
                fills: int = 0
                entry_pxs: list[int] = []
                for m in markets:
                    if em < 0 or em >= len(m.candles):
                        continue
                    c0 = m.candles[em]
                    if side == "YES":
                        bid0 = c0.yes_bid.open
                        ask0 = c0.yes_ask.open
                    else:
                        bid0 = c0.no_bid.open
                        ask0 = c0.no_ask.open
                    if bid0 is None or ask0 is None or ask0 <= bid0:
                        continue
                    sp = int(ask0 - bid0)
                    if sp > max_entry_spread_cents:
                        continue
                    entry_px = int(min(bid0 + imp, ask0 - 1))
                    entry_px = max(1, min(99, entry_px))
                    end_fill = min(len(m.candles) - 1, em + ttl - 1)
                    filled = False
                    for t in range(em, end_fill + 1):
                        ct = m.candles[t]
                        ask_low = (ct.yes_ask.low if side == "YES" else ct.no_ask.low)
                        if ask_low is None:
                            continue
                        if ask_low <= entry_px:
                            filled = True
                            break
                    if not filled:
                        pnls_attempt.append(0)
                        continue
                    fills += 1
                    entry_pxs.append(entry_px)
                    win = (m.settles_yes() if side == "YES" else (not m.settles_yes()))
                    wins += 1 if win else 0
                    payoff = 100 if win else 0
                    pnl = payoff - entry_px
                    pnls_attempt.append(pnl)
                    pnls_fill.append(pnl)

                if not pnls_attempt:
                    continue
                out.append(
                    {
                        "side": side,
                        "entry_minute": em,
                        "entry_improve_cents": imp,
                        "ttl_minutes": ttl,
                        "max_entry_spread_cents": max_entry_spread_cents,
                        "attempts": int(len(pnls_attempt)),
                        "fills": int(fills),
                        "fill_rate": float(fills / len(pnls_attempt)),
                        "win_rate_given_fill": float(wins / fills) if fills else 0.0,
                        "mean_entry_px_filled": float(mean(entry_pxs)) if entry_pxs else 0.0,
                        "mean_settle_pnl_cents_per_fill": float(mean(pnls_fill)) if pnls_fill else 0.0,
                        "mean_settle_pnl_cents_per_attempt": float(mean(pnls_attempt)),
                        "ev_cents_per_hour_per_contract": float(4.0 * mean(pnls_attempt)),
                    }
                )

    out.sort(key=lambda r: r["mean_settle_pnl_cents_per_attempt"], reverse=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnostics to identify promising strategy families from KXBTC15M candles.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="kalshi_paper_trader/maker-strat/diagnostics_out")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--mid-bin-cents", type=int, default=10)
    ap.add_argument("--minute-bin", type=int, default=1)
    ap.add_argument("--max-spread-cents", type=int, default=40, help="Spread filter for calibration surface.")
    ap.add_argument("--emit-taker-ev", action="store_true", help="Write taker EV surface (buy YES at ask, incl taker fees).")
    ap.add_argument("--emit-maker-yes-surface", action="store_true",
                    help="Write maker-entry YES surface (fill->settle) for a fixed improve+TTL.")
    ap.add_argument("--maker-yes-surface-entry-improve", type=int, default=0)
    ap.add_argument("--maker-yes-surface-ttl-minutes", type=int, default=1)
    ap.add_argument("--maker-side", choices=["YES", "NO", "BOTH"], default="BOTH")
    ap.add_argument("--maker-entry-minutes", default="0-14")
    ap.add_argument("--maker-entry-improve", default="0,1,2")
    ap.add_argument("--maker-ttl-minutes", default="1,2,3")
    ap.add_argument("--maker-max-entry-spread-cents", type=int, default=40)
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Machine-readable assumptions log.
    (outdir / "assumptions.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": "kalshi_paper_trader/maker-strat/diagnose.py",
                "args": {k: getattr(args, k) for k in vars(args)},
                "data_assumptions": [
                    "Input data is Kalshi candlesticks (top-of-book bid/ask OHLC), typically 1-minute resolution.",
                    "No full orderbook snapshots, queue position, or trade prints are used.",
                ],
                "execution_assumptions": [
                    "Calibration surfaces use YES mid=(bid+ask)/2 from candle open values.",
                    "Taker EV surface (if enabled) assumes buying YES at candle-open ask and holding to settlement, including taker fees.",
                    "Maker-entry toxicity uses OHLC heuristic: filled if ask_low <= entry_px within TTL window (queue position not modeled).",
                    "NO quotes are derived from YES via parity mapping; NO-side results are indicative, not definitive.",
                ],
                "fee_model": {
                    "description": "Approx Kalshi taker fee per contract: fee_dollars = ceil(0.07 * P * (1-P)), where P is contract price in dollars.",
                    "implementation": "kalshi_paper_trader/maker-strat/sim.py::_taker_fee_cents",
                },
            },
            indent=2,
        )
    )

    markets = load_markets(args.data)
    obs = _collect_observations(markets)
    train_days, test_days = _split_days(markets, args.train_frac)

    summary = {
        "n_markets": len(markets),
        "n_obs": len(obs),
        "uncond_yes_rate": float(sum(1 for m in markets if m.settles_yes()) / len(markets)) if markets else 0.0,
        "train_days": len(train_days),
        "test_days": len(test_days),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "notes": [
            "Calibration surface uses YES mid=(bid+ask)/2 from candle open values (top-of-book only).",
            "Maker toxicity uses OHLC fill heuristic: filled if ask_low <= entry_px within TTL window.",
            "NO quotes are derived from YES via parity mapping; results for NO are indicative, not definitive.",
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    spread_rows = _spread_stats_by_minute(obs)
    _write_csv(outdir / "spread_by_minute.csv", spread_rows)

    cal_train = _calibration_surface(
        obs,
        mid_bin_cents=args.mid_bin_cents,
        minute_bin=args.minute_bin,
        max_spread_cents=args.max_spread_cents,
        days_filter=train_days,
    )
    cal_test = _calibration_surface(
        obs,
        mid_bin_cents=args.mid_bin_cents,
        minute_bin=args.minute_bin,
        max_spread_cents=args.max_spread_cents,
        days_filter=test_days,
    )
    _write_csv(outdir / "calibration_train.csv", cal_train)
    _write_csv(outdir / "calibration_test.csv", cal_test)

    if args.emit_taker_ev:
        taker_train = _taker_yes_ev_surface(
            markets,
            mid_bin_cents=args.mid_bin_cents,
            minute_bin=args.minute_bin,
            max_spread_cents=args.max_spread_cents,
            days_filter=train_days,
        )
        taker_test = _taker_yes_ev_surface(
            markets,
            mid_bin_cents=args.mid_bin_cents,
            minute_bin=args.minute_bin,
            max_spread_cents=args.max_spread_cents,
            days_filter=test_days,
        )
        _write_csv(outdir / "taker_yes_ev_train.csv", taker_train)
        _write_csv(outdir / "taker_yes_ev_test.csv", taker_test)

    if args.emit_maker_yes_surface:
        maker_yes_train = _maker_yes_ev_surface(
            markets,
            mid_bin_cents=args.mid_bin_cents,
            minute_bin=args.minute_bin,
            max_spread_cents=args.max_spread_cents,
            entry_improve_cents=args.maker_yes_surface_entry_improve,
            ttl_minutes=args.maker_yes_surface_ttl_minutes,
            days_filter=train_days,
        )
        maker_yes_test = _maker_yes_ev_surface(
            markets,
            mid_bin_cents=args.mid_bin_cents,
            minute_bin=args.minute_bin,
            max_spread_cents=args.max_spread_cents,
            entry_improve_cents=args.maker_yes_surface_entry_improve,
            ttl_minutes=args.maker_yes_surface_ttl_minutes,
            days_filter=test_days,
        )
        _write_csv(outdir / "maker_yes_surface_train.csv", maker_yes_train)
        _write_csv(outdir / "maker_yes_surface_test.csv", maker_yes_test)

    # Maker toxicity (entry only, hold to settle) for quick viability detection.
    def _parse_int_list(s: str) -> list[int]:
        s = s.strip()
        if "-" in s and "," not in s:
            a, b = s.split("-", 1)
            return list(range(int(a), int(b) + 1))
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    entry_minutes = _parse_int_list(args.maker_entry_minutes)
    entry_improve = _parse_int_list(args.maker_entry_improve)
    ttl_minutes = _parse_int_list(args.maker_ttl_minutes)

    tox_rows: list[dict[str, Any]] = []
    if args.maker_side in ("YES", "BOTH"):
        tox_rows.extend(
            _maker_entry_toxicity(
                markets,
                side="YES",
                entry_minutes=entry_minutes,
                entry_improve_cents=entry_improve,
                ttl_minutes=ttl_minutes,
                max_entry_spread_cents=args.maker_max_entry_spread_cents,
            )
        )
    if args.maker_side in ("NO", "BOTH"):
        tox_rows.extend(
            _maker_entry_toxicity(
                markets,
                side="NO",
                entry_minutes=entry_minutes,
                entry_improve_cents=entry_improve,
                ttl_minutes=ttl_minutes,
                max_entry_spread_cents=args.maker_max_entry_spread_cents,
            )
        )
    if tox_rows:
        tox_rows.sort(key=lambda r: r["mean_settle_pnl_cents_per_attempt"], reverse=True)
        _write_csv(outdir / "maker_entry_toxicity.csv", tox_rows)

    print(f"Wrote diagnostics to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
