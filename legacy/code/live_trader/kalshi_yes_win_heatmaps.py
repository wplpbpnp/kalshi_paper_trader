#!/usr/bin/env python3
"""
Generate Kalshi YES win-rate heatmaps by price (1-cent buckets) and time (1-minute buckets),
including conservative (Wilson lower-bound) estimates for YES and NO.
"""

import json
import math
import argparse
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


PRICE_CENTS = 100  # discrete cent prices: 0..99 (we skip 0 and 100 in data filter)
Z_SCORE = 1.96  # 95% confidence


def load_kalshi_data(path):
    with open(path, "r") as f:
        return json.load(f)

def parse_market_time(market):
    close_time_str = market.get("close_time")
    if not close_time_str:
        return None
    try:
        return datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
    except Exception:
        return None


def get_yes_price_cents(candle, max_spread_cents=None):
    yes_bid = candle.get("yes_bid", {}).get("close", 0)
    yes_ask = candle.get("yes_ask", {}).get("close", 100)
    if yes_bid > 0 and yes_ask < 100:
        spread = yes_ask - yes_bid
        if max_spread_cents is not None and spread > max_spread_cents:
            return None
        return (yes_bid + yes_ask) / 2
    return None


def wilson_lower_bound(wins, total, z=Z_SCORE):
    if total == 0:
        return 0.0
    phat = wins / total
    denom = 1 + (z * z) / total
    center = phat + (z * z) / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * total)) / total)
    return max(0.0, (center - margin) / denom)


def build_counts(markets, max_spread_cents=None):
    # Determine max minutes across markets to size time axis
    max_minutes = 0
    for m in markets:
        candles = m.get("candlesticks", [])
        if len(candles) > max_minutes:
            max_minutes = len(candles)

    yes_counts = np.zeros((PRICE_CENTS, max_minutes), dtype=int)
    total_counts = np.zeros((PRICE_CENTS, max_minutes), dtype=int)

    for m in markets:
        result = m.get("result", "").lower()
        if result not in ("yes", "no"):
            continue
        candles = m.get("candlesticks", [])
        if len(candles) < 2:
            continue

        for idx, candle in enumerate(candles):
            yes_price = get_yes_price_cents(candle, max_spread_cents=max_spread_cents)
            if yes_price is None or yes_price <= 0 or yes_price >= 100:
                continue

            price_cent = int(round(yes_price))
            if price_cent < 0 or price_cent >= PRICE_CENTS:
                continue

            total_counts[price_cent, idx] += 1
            if result == "yes":
                yes_counts[price_cent, idx] += 1

    return yes_counts, total_counts


def build_counts_weighted(markets, decay_half_life_days=7.0, max_spread_cents=None):
    max_minutes = 0
    latest_time = None
    for m in markets:
        candles = m.get("candlesticks", [])
        if len(candles) > max_minutes:
            max_minutes = len(candles)
        t = parse_market_time(m)
        if t and (latest_time is None or t > latest_time):
            latest_time = t

    yes_counts = np.zeros((PRICE_CENTS, max_minutes), dtype=float)
    total_counts = np.zeros((PRICE_CENTS, max_minutes), dtype=float)

    decay_constant = math.log(2) / decay_half_life_days if decay_half_life_days > 0 else 0.0

    for m in markets:
        result = m.get("result", "").lower()
        if result not in ("yes", "no"):
            continue
        candles = m.get("candlesticks", [])
        if len(candles) < 2:
            continue

        t = parse_market_time(m)
        if latest_time and t:
            age_days = (latest_time - t).total_seconds() / (24 * 3600)
            if age_days < 0:
                age_days = 0
            weight = math.exp(-decay_constant * age_days) if decay_constant > 0 else 1.0
        else:
            weight = 1.0

        for idx, candle in enumerate(candles):
            yes_price = get_yes_price_cents(candle, max_spread_cents=max_spread_cents)
            if yes_price is None or yes_price <= 0 or yes_price >= 100:
                continue

            price_cent = int(round(yes_price))
            if price_cent < 0 or price_cent >= PRICE_CENTS:
                continue

            total_counts[price_cent, idx] += weight
            if result == "yes":
                yes_counts[price_cent, idx] += weight

    return yes_counts, total_counts


def compute_matrices(yes_counts, total_counts):
    with np.errstate(divide="ignore", invalid="ignore"):
        p_yes = np.where(total_counts > 0, yes_counts / total_counts, np.nan)

    # Conservative lower bounds for YES and NO
    p_yes_lb = np.full_like(p_yes, np.nan, dtype=float)
    p_no_lb = np.full_like(p_yes, np.nan, dtype=float)

    rows, cols = total_counts.shape
    for i in range(rows):
        for j in range(cols):
            total = total_counts[i, j]
            if total == 0:
                continue
            yes_wins = yes_counts[i, j]
            no_wins = total - yes_wins
            p_yes_lb[i, j] = wilson_lower_bound(yes_wins, total)
            p_no_lb[i, j] = wilson_lower_bound(no_wins, total)

    return p_yes, p_yes_lb, p_no_lb


def plot_heatmap(matrix, title, output_path, vmin=0.0, vmax=1.0, cbar_label="Probability", annotate=False, fmt="{:.0f}", label_mask=None, yticklabels=None):
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required to plot heatmaps")

    plt.figure(figsize=(14, 8))
    im = plt.imshow(matrix, aspect="auto", origin="lower", cmap="RdYlGn", vmin=vmin, vmax=vmax, interpolation="none")
    plt.colorbar(im, label=cbar_label)
    plt.xlabel("Minutes Elapsed")
    plt.ylabel("YES Price Bucket (cents)")
    if yticklabels is not None:
        plt.yticks(range(len(yticklabels)), yticklabels)
    plt.title(title)
    if annotate:
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(cols):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                if label_mask is not None and not label_mask[i, j]:
                    continue
                plt.text(j, i, fmt.format(val), ha="center", va="center", fontsize=6, color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_sample_heatmap_with_ev_labels(sample_counts, ev_yes_pct, ev_no_pct, kelly_yes_frac, kelly_no_frac, label_mask, output_path, yticklabels=None):
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required to plot heatmaps")

    plt.figure(figsize=(14, 8))
    im = plt.imshow(sample_counts, aspect="auto", origin="lower", cmap="Blues", interpolation="none")
    plt.colorbar(im, label="Samples")
    plt.xlabel("Minutes Elapsed")
    plt.ylabel("YES Price Bucket (cents)")
    if yticklabels is not None:
        plt.yticks(range(len(yticklabels)), yticklabels)
    plt.title("Sample Count with Positive Conservative EV Labels (YES/NO)")

    rows, cols = sample_counts.shape
    for i in range(rows):
        for j in range(cols):
            if not label_mask[i, j]:
                continue
            y = ev_yes_pct[i, j]
            n = ev_no_pct[i, j]
            ky = kelly_yes_frac[i, j]
            kn = kelly_no_frac[i, j]
            if not np.isnan(y) and y > 0:
                label = f"{y:+.1f}%, K {ky*100:.1f}%"
                plt.text(j, i, label, ha="center", va="center", fontsize=6, color="green")
            if not np.isnan(n) and n > 0:
                label = f"{n:+.1f}%, K {kn*100:.1f}%"
                plt.text(j, i, label, ha="center", va="center", fontsize=6, color="red")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def strip_zero_cent_row(*matrices):
    # Drop 0-cent row (100-cent row isn't present in data arrays).
    return [m[1:, :] for m in matrices]

def aggregate_counts_by_price(yes_counts, total_counts, bucket_size):
    rows, cols = yes_counts.shape
    n_buckets = int(math.ceil(rows / bucket_size))
    agg_yes = np.zeros((n_buckets, cols), dtype=int)
    agg_total = np.zeros((n_buckets, cols), dtype=int)
    ranges = []
    for b in range(n_buckets):
        start = b * bucket_size
        end = min(rows, (b + 1) * bucket_size)
        agg_yes[b, :] = yes_counts[start:end, :].sum(axis=0)
        agg_total[b, :] = total_counts[start:end, :].sum(axis=0)
        ranges.append((start, end - 1))  # inclusive indices
    return agg_yes, agg_total, ranges

def summarize_bucket_sweep(yes_counts, total_counts, bucket_sizes, min_samples=30, worst_price=False):
    print("\n" + "=" * 80)
    print("BUCKET SWEEP (Conservative EV vs Raw EV)")
    if worst_price:
        print("Note: price for each bucket uses worst-case price in the cent range.")
    else:
        print("Note: price for each bucket uses midpoint of its cent range.")
    print("=" * 80)
    for size in bucket_sizes:
        agg_yes, agg_total, ranges = aggregate_counts_by_price(yes_counts, total_counts, size)
        p_yes, p_yes_lb, p_no_lb = compute_matrices(agg_yes, agg_total)
        p_no = 1 - p_yes

        # Price midpoint per bucket (rows correspond to cents starting at 1)
        prices = np.array([((r[0] + r[1]) / 2 + 1) / 100.0 for r in ranges])

        price_range = [((r[0] + 1) / 100.0, (r[1] + 1) / 100.0) for r in ranges]
        ev_yes_cons, ev_no_cons = compute_ev_matrices(p_yes_lb, p_no_lb, prices=prices, price_range=price_range, worst_price=worst_price)
        ev_yes_raw, ev_no_raw = compute_ev_matrices(p_yes, p_no, prices=prices, price_range=price_range, worst_price=worst_price)

        mask = agg_total >= min_samples
        yes_pos = mask & (ev_yes_cons > 0)
        no_pos = mask & (ev_no_cons > 0)

        yes_hit = (np.nanmean(ev_yes_raw[yes_pos] > 0) if np.any(yes_pos) else float('nan'))
        no_hit = (np.nanmean(ev_no_raw[no_pos] > 0) if np.any(no_pos) else float('nan'))

        yes_cons_avg = (np.nanmean(ev_yes_cons[yes_pos]) if np.any(yes_pos) else float('nan'))
        no_cons_avg = (np.nanmean(ev_no_cons[no_pos]) if np.any(no_pos) else float('nan'))
        yes_raw_avg = (np.nanmean(ev_yes_raw[yes_pos]) if np.any(yes_pos) else float('nan'))
        no_raw_avg = (np.nanmean(ev_no_raw[no_pos]) if np.any(no_pos) else float('nan'))

        print(f"\nBucket size: {size}¢")
        print(f"  YES+ cells: {int(np.sum(yes_pos))} | hit rate: {yes_hit:.2%} | cons EV avg: {yes_cons_avg*100:.3f}% | raw EV avg: {yes_raw_avg*100:.3f}%")
        print(f"  NO+  cells: {int(np.sum(no_pos))} | hit rate: {no_hit:.2%} | cons EV avg: {no_cons_avg*100:.3f}% | raw EV avg: {no_raw_avg*100:.3f}%")

def compute_ev_matrices(p_yes, p_no, prices=None, price_offset_cents=0, price_range=None, worst_price=False):
    rows, cols = p_yes.shape
    ev_yes = np.full_like(p_yes, np.nan, dtype=float)
    ev_no = np.full_like(p_yes, np.nan, dtype=float)

    for price_idx in range(rows):
        if prices is None:
            price_mid = (price_idx + price_offset_cents) / 100.0
            if price_range is None:
                price_min = price_mid
                price_max = price_mid
            else:
                price_min, price_max = price_range[price_idx]
        else:
            price_mid = prices[price_idx]
            if price_range is None:
                price_min = price_mid
                price_max = price_mid
            else:
                price_min, price_max = price_range[price_idx]

        if worst_price:
            # Worst price for EV: YES uses max price, NO uses max NO price (= min YES price)
            price_yes = price_max
            price_no = 1.0 - price_min
        else:
            price_yes = price_mid
            price_no = 1.0 - price_mid

        fee_yes = math.ceil(0.07 * price_yes * (1 - price_yes) * 100) / 100.0
        fee_no = math.ceil(0.07 * price_no * (1 - price_no) * 100) / 100.0
        for t in range(cols):
            p_yes_cell = p_yes[price_idx, t]
            p_no_cell = p_no[price_idx, t]
            if not np.isnan(p_yes_cell):
                # EV for buying YES at discrete cent price
                ev_yes[price_idx, t] = p_yes_cell * (1 - price_yes) - (1 - p_yes_cell) * price_yes - fee_yes
            if not np.isnan(p_no_cell):
                # EV for buying NO at no_price
                ev_no[price_idx, t] = p_no_cell * (1 - price_no) - (1 - p_no_cell) * price_no - fee_no

    return ev_yes, ev_no

def compute_kelly_matrices(p_yes, p_no, prices=None, price_offset_cents=0, price_range=None, worst_price=False):
    rows, cols = p_yes.shape
    k_yes = np.full_like(p_yes, np.nan, dtype=float)
    k_no = np.full_like(p_yes, np.nan, dtype=float)

    for price_idx in range(rows):
        if prices is None:
            price_mid = (price_idx + price_offset_cents) / 100.0
            if price_range is None:
                price_min = price_mid
                price_max = price_mid
            else:
                price_min, price_max = price_range[price_idx]
        else:
            price_mid = prices[price_idx]
            if price_range is None:
                price_min = price_mid
                price_max = price_mid
            else:
                price_min, price_max = price_range[price_idx]

        if worst_price:
            price_yes = price_max
            price_no = 1.0 - price_min
        else:
            price_yes = price_mid
            price_no = 1.0 - price_mid

        fee_yes = math.ceil(0.07 * price_yes * (1 - price_yes) * 100) / 100.0
        fee_no = math.ceil(0.07 * price_no * (1 - price_no) * 100) / 100.0

        win_yes = 1.0 - price_yes - fee_yes
        loss_yes = price_yes + fee_yes
        win_no = 1.0 - price_no - fee_no
        loss_no = price_no + fee_no

        for t in range(cols):
            p_yes_cell = p_yes[price_idx, t]
            p_no_cell = p_no[price_idx, t]

            if not np.isnan(p_yes_cell) and win_yes > 0 and loss_yes > 0:
                b = win_yes / loss_yes
                k = (p_yes_cell * b - (1 - p_yes_cell)) / b
                k_yes[price_idx, t] = max(0.0, k)

            if not np.isnan(p_no_cell) and win_no > 0 and loss_no > 0:
                b = win_no / loss_no
                k = (p_no_cell * b - (1 - p_no_cell)) / b
                k_no[price_idx, t] = max(0.0, k)

    return k_yes, k_no

def plot_rolling_ev(top_yes_cell, top_no_cell, markets, bucket_size, min_samples, worst_price, output_path, window_weeks=1, step_weeks=1, decay_half_life_days=7.0, max_spread_cents=None):
    if not HAS_MATPLOTLIB:
        return

    timed_markets = []
    for m in markets:
        t = parse_market_time(m)
        if t is None:
            continue
        m["_close_time"] = t
        timed_markets.append(m)

    timed_markets.sort(key=lambda x: x["_close_time"])
    if not timed_markets:
        return

    start = timed_markets[0]["_close_time"]
    end = timed_markets[-1]["_close_time"]
    window_delta = timedelta(weeks=window_weeks)
    step_delta = timedelta(weeks=step_weeks)

    max_minutes = max((len(m.get("candlesticks", [])) for m in timed_markets), default=0)
    if max_minutes == 0:
        return

    rows = 99
    if bucket_size > 1:
        ranges = []
        n_buckets = int(math.ceil(rows / bucket_size))
        for b in range(n_buckets):
            start_c = b * bucket_size + 1
            end_c = min(rows, (b + 1) * bucket_size)
            ranges.append((start_c, end_c))
    else:
        ranges = [(i + 1, i + 1) for i in range(rows)]

    range_to_idx = {r: i for i, r in enumerate(ranges)}

    def cell_ev(cell, window_markets, reference_time):
        # Time-decayed weighted counts
        decay_constant = math.log(2) / decay_half_life_days if decay_half_life_days > 0 else 0.0
        yes_counts = np.zeros((PRICE_CENTS, max_minutes), dtype=float)
        total_counts = np.zeros((PRICE_CENTS, max_minutes), dtype=float)

        for wm in window_markets:
            close_time = wm.get("_close_time")
            if close_time is None:
                continue
            age_days = (reference_time - close_time).total_seconds() / (24 * 3600)
            if age_days < 0:
                age_days = 0
            weight = math.exp(-decay_constant * age_days) if decay_constant > 0 else 1.0

            result = wm.get("result", "").lower()
            if result not in ("yes", "no"):
                continue
            candles = wm.get("candlesticks", [])
            if len(candles) < 2:
                continue
            for idx, candle in enumerate(candles):
                yes_price = get_yes_price_cents(candle, max_spread_cents=max_spread_cents)
                if yes_price is None or yes_price <= 0 or yes_price >= 100:
                    continue
                price_cent = int(round(yes_price))
                if price_cent < 0 or price_cent >= PRICE_CENTS:
                    continue
                total_counts[price_cent, idx] += weight
                if result == "yes":
                    yes_counts[price_cent, idx] += weight

        yes_counts = strip_zero_cent_row(yes_counts)[0]
        total_counts = strip_zero_cent_row(total_counts)[0]
        if bucket_size > 1:
            yes_counts, total_counts, _ = aggregate_counts_by_price(yes_counts, total_counts, bucket_size)

        # Weighted Wilson using weighted totals
        with np.errstate(divide="ignore", invalid="ignore"):
            p_yes = np.where(total_counts > 0, yes_counts / total_counts, np.nan)
        p_yes_lb = np.full_like(p_yes, np.nan, dtype=float)
        p_no_lb = np.full_like(p_yes, np.nan, dtype=float)
        rows, cols = total_counts.shape
        for i in range(rows):
            for j in range(cols):
                total = total_counts[i, j]
                if total == 0:
                    continue
                yes_wins = yes_counts[i, j]
                no_wins = total - yes_wins
                p_yes_lb[i, j] = wilson_lower_bound(yes_wins, total)
                p_no_lb[i, j] = wilson_lower_bound(no_wins, total)

        price_low, price_high = cell["price_range"]
        price_range_cents = (int(round(price_low * 100)), int(round(price_high * 100)))
        idx = range_to_idx.get(price_range_cents)
        if idx is None:
            return None
        time_idx = cell["time_range_minutes"][0]
        if time_idx < 0 or time_idx >= p_yes.shape[1]:
            return None
        if total_counts[idx, time_idx] == 0:
            return None

        if worst_price:
            price_yes = price_high
            price_no = 1.0 - price_low
        else:
            price_yes = (price_low + price_high) / 2
            price_no = 1.0 - ((price_low + price_high) / 2)

        if cell["side"] == "YES":
            p = p_yes_lb[idx, time_idx]
            if p is None or np.isnan(p):
                return None
            fee = math.ceil(0.07 * price_yes * (1 - price_yes) * 100) / 100.0
            ev = p * (1 - price_yes) - (1 - p) * price_yes - fee
            return ev * 100
        else:
            p = p_no_lb[idx, time_idx]
            if p is None or np.isnan(p):
                return None
            fee = math.ceil(0.07 * price_no * (1 - price_no) * 100) / 100.0
            ev = p * (1 - price_no) - (1 - p) * price_no - fee
            return ev * 100

    times = []
    ev_yes_series = []
    ev_no_series = []

    window_start = start
    while window_start + window_delta <= end:
        window_end = window_start + window_delta
        window_markets = [m for m in timed_markets if window_start <= m["_close_time"] < window_end]
        if window_markets:
            times.append(window_start + (window_delta / 2))
            ev_yes_series.append(cell_ev(top_yes_cell, window_markets, window_end) if top_yes_cell else None)
            ev_no_series.append(cell_ev(top_no_cell, window_markets, window_end) if top_no_cell else None)
        window_start = window_start + step_delta

    if not times:
        return

    if not top_yes_cell and not top_no_cell:
        return

    plt.figure(figsize=(10, 5))
    has_series = False
    if top_yes_cell:
        plt.plot(times, ev_yes_series, label="Top YES cell", color="green")
        has_series = True
    if top_no_cell:
        plt.plot(times, ev_no_series, label="Top NO cell", color="red")
        has_series = True
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Rolling Conservative EV (Top Cells)")
    plt.xlabel("Time")
    plt.ylabel("EV (%)")
    if has_series:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def run_for_bucket_size(yes_counts, total_counts, markets, bucket_size, label_min_samples=30, detailed_charts=False, worst_price=False, kelly_frac=0.25, output_table=None, rolling_decay_hl=7.0, max_spread_cents=None):
    # Aggregate if needed (counts are already zero-cent stripped)
    if bucket_size > 1:
        yes_counts_b, total_counts_b, ranges = aggregate_counts_by_price(yes_counts, total_counts, bucket_size)
        prices = np.array([((r[0] + r[1]) / 2 + 1) / 100.0 for r in ranges])
        price_range = [((r[0] + 1) / 100.0, (r[1] + 1) / 100.0) for r in ranges]
        yticklabels = [f"{r[0]+1}-{r[1]+1}" for r in ranges]
    else:
        yes_counts_b, total_counts_b = yes_counts, total_counts
        prices = None  # use price_idx + offset
        price_range = None
        ranges = [(i, i) for i in range(yes_counts_b.shape[0])]
        yticklabels = None

    p_yes, p_yes_lb, p_no_lb = compute_matrices(yes_counts_b, total_counts_b)
    ev_yes_cons, ev_no_cons = compute_ev_matrices(p_yes_lb, p_no_lb, prices=prices, price_offset_cents=1, price_range=price_range, worst_price=worst_price)
    p_no = 1 - p_yes
    ev_yes_raw, ev_no_raw = compute_ev_matrices(p_yes, p_no, prices=prices, price_offset_cents=1, price_range=price_range, worst_price=worst_price)
    k_yes, k_no = compute_kelly_matrices(p_yes_lb, p_no_lb, prices=prices, price_offset_cents=1, price_range=price_range, worst_price=worst_price)

    label_mask = total_counts_b >= label_min_samples

    # Identify top YES/NO cells for rolling EV chart
    top_yes_cell = None
    top_no_cell = None
    if np.any(label_mask & (ev_yes_cons > 0)):
        idx = np.nanargmax(np.where(label_mask, ev_yes_cons, np.nan))
        i, j = np.unravel_index(idx, ev_yes_cons.shape)
        top_yes_cell = {
            "price_range": [ (ranges[i][0] + 1) / 100.0, (ranges[i][1] + 1) / 100.0 ],
            "time_range_minutes": [j, j + 1],
            "side": "YES",
        }
    if np.any(label_mask & (ev_no_cons > 0)):
        idx = np.nanargmax(np.where(label_mask, ev_no_cons, np.nan))
        i, j = np.unravel_index(idx, ev_no_cons.shape)
        top_no_cell = {
            "price_range": [ (ranges[i][0] + 1) / 100.0, (ranges[i][1] + 1) / 100.0 ],
            "time_range_minutes": [j, j + 1],
            "side": "NO",
        }

    # Convert EV to percent for display
    ev_yes_pct = ev_yes_cons * 100
    ev_no_pct = ev_no_cons * 100
    ev_yes_raw_pct = ev_yes_raw * 100
    ev_no_raw_pct = ev_no_raw * 100

    max_abs_ev = np.nanmax(np.abs(np.concatenate([
        ev_yes_cons[~np.isnan(ev_yes_cons)],
        ev_no_cons[~np.isnan(ev_no_cons)],
        ev_yes_raw[~np.isnan(ev_yes_raw)],
        ev_no_raw[~np.isnan(ev_no_raw)],
    ]))) if np.any(~np.isnan(ev_yes_cons)) or np.any(~np.isnan(ev_no_cons)) else 0.01
    if max_abs_ev == 0:
        max_abs_ev = 0.01
    max_abs_ev_pct = max_abs_ev * 100

    # Chart titles include bucket size
    bucket_label = f"{bucket_size}¢" if bucket_size > 1 else "1¢"
    plot_sample_heatmap_with_ev_labels(
        total_counts_b,
        ev_yes_pct,
        ev_no_pct,
        k_yes * kelly_frac,
        k_no * kelly_frac,
        label_mask,
        "kalshi_samples_with_ev_labels.png",
        yticklabels=yticklabels,
    )
    plot_rolling_ev(
        top_yes_cell,
        top_no_cell,
        markets,
        bucket_size,
        label_min_samples,
        worst_price,
        "rolling_ev_top_cells.png",
        decay_half_life_days=rolling_decay_hl,
        max_spread_cents=max_spread_cents,
    )

    if output_table:
        if not output_table.endswith(".json"):
            output_table = output_table + ".json"
        cells = []
        rows, cols = total_counts_b.shape
        for i in range(rows):
            price_low = (ranges[i][0] + 1) / 100.0
            price_high = (ranges[i][1] + 1) / 100.0
            for j in range(cols):
                if not label_mask[i, j]:
                    continue
                if not np.isnan(ev_yes_pct[i, j]) and ev_yes_pct[i, j] > 0:
                    cells.append({
                        "price_range": [price_low, price_high],
                        "time_range_minutes": [j, j + 1],
                        "side": "YES",
                        "expected_ev_pct": round(float(ev_yes_pct[i, j]), 4),
                        "kelly_fraction": round(float(k_yes[i, j] * kelly_frac), 6),
                    })
                if not np.isnan(ev_no_pct[i, j]) and ev_no_pct[i, j] > 0:
                    cells.append({
                        "price_range": [price_low, price_high],
                        "time_range_minutes": [j, j + 1],
                        "side": "NO",
                        "expected_ev_pct": round(float(ev_no_pct[i, j]), 4),
                        "kelly_fraction": round(float(k_no[i, j] * kelly_frac), 6),
                    })

        output = {
            "bucket_size_cents": bucket_size,
            "min_samples": label_min_samples,
            "worst_price": worst_price,
            "kelly_fractional_multiplier": kelly_frac,
            "cells": cells,
        }
        with open(output_table, "w") as f:
            json.dump(output, f, indent=2)
    if detailed_charts:
        plot_heatmap(
            p_yes,
            f"Historical YES Win % by Price ({bucket_label}) and Time (1m)",
            "kalshi_yes_winrate_heatmap.png",
            annotate=True,
            fmt="{:.0%}",
            label_mask=label_mask,
            yticklabels=yticklabels,
        )
        plot_heatmap(
            p_yes_lb,
            f"Conservative YES Win % (Wilson Lower Bound) ({bucket_label})",
            "kalshi_yes_winrate_conservative.png",
            annotate=True,
            fmt="{:.0%}",
            label_mask=label_mask,
            yticklabels=yticklabels,
        )
        plot_heatmap(
            p_no_lb,
            f"Conservative NO Win % (Wilson Lower Bound) ({bucket_label})",
            "kalshi_no_winrate_conservative.png",
            annotate=True,
            fmt="{:.0%}",
            label_mask=label_mask,
            yticklabels=yticklabels,
        )
        plot_heatmap(
            ev_yes_pct,
            f"Conservative EV for YES (Wilson Lower Bound) ({bucket_label})",
            "kalshi_yes_ev_conservative.png",
            vmin=-max_abs_ev_pct,
            vmax=max_abs_ev_pct,
            cbar_label="% Return",
            annotate=True,
            fmt="{:+.1f}%",
            label_mask=(label_mask & (ev_yes_pct > 0)),
            yticklabels=yticklabels,
        )
        plot_heatmap(
            ev_no_pct,
            f"Conservative EV for NO (Wilson Lower Bound) ({bucket_label})",
            "kalshi_no_ev_conservative.png",
            vmin=-max_abs_ev_pct,
            vmax=max_abs_ev_pct,
            cbar_label="% Return",
            annotate=True,
            fmt="{:+.1f}%",
            label_mask=(label_mask & (ev_no_pct > 0)),
            yticklabels=yticklabels,
        )
        plot_heatmap(
            ev_yes_raw_pct,
            f"Raw EV for YES (Historical Winrate) ({bucket_label})",
            "kalshi_yes_ev_raw.png",
            vmin=-max_abs_ev_pct,
            vmax=max_abs_ev_pct,
            cbar_label="% Return",
            annotate=True,
            fmt="{:+.1f}%",
            label_mask=label_mask,
            yticklabels=yticklabels,
        )

    # Print summary for selected bucket size
    yes_pos = label_mask & (ev_yes_cons > 0)
    no_pos = label_mask & (ev_no_cons > 0)
    yes_hit = (np.nanmean(ev_yes_raw[yes_pos] > 0) if np.any(yes_pos) else float('nan'))
    no_hit = (np.nanmean(ev_no_raw[no_pos] > 0) if np.any(no_pos) else float('nan'))
    yes_cons_avg = (np.nanmean(ev_yes_cons[yes_pos]) if np.any(yes_pos) else float('nan'))
    no_cons_avg = (np.nanmean(ev_no_cons[no_pos]) if np.any(no_pos) else float('nan'))
    yes_raw_avg = (np.nanmean(ev_yes_raw[yes_pos]) if np.any(yes_pos) else float('nan'))
    no_raw_avg = (np.nanmean(ev_no_raw[no_pos]) if np.any(no_pos) else float('nan'))

    print("\n" + "=" * 80)
    print(f"SELECTED BUCKET SIZE SUMMARY ({bucket_label})")
    print("=" * 80)
    print(f"  YES+ cells: {int(np.sum(yes_pos))} | hit rate: {yes_hit:.2%} | cons EV avg: {yes_cons_avg*100:.3f}% | raw EV avg: {yes_raw_avg*100:.3f}%")
    print(f"  NO+  cells: {int(np.sum(no_pos))} | hit rate: {no_hit:.2%} | cons EV avg: {no_cons_avg*100:.3f}% | raw EV avg: {no_raw_avg*100:.3f}%")


def main():
    parser = argparse.ArgumentParser(description="Kalshi YES win-rate heatmaps with conservative EV.")
    parser.add_argument("--bucket-size", type=int, default=1, help="Price bucket size in cents (default: 1)")
    parser.add_argument("--min-samples", type=int, default=30, help="Minimum samples to label cells (default: 30)")
    parser.add_argument("--detailed-charts", action="store_true", help="Generate full set of winrate/EV charts (default: only samples+EV overlay)")
    parser.add_argument("--worst-price", action="store_true", help="Use worst-case price within each bucket for EV")
    parser.add_argument("--kelly-frac", type=float, default=0.25, help="Fractional Kelly multiplier (default: 0.25)")
    parser.add_argument("--output-table", help="Write labeled cells to JSON file")
    parser.add_argument("--decay-hl", type=float, default=7.0, help="Half-life in days for main heatmaps decay (default: 7.0, set 0 to disable)")
    parser.add_argument("--rolling-decay-hl", type=float, default=7.0, help="Half-life in days for rolling EV decay (default: 7.0)")
    parser.add_argument("--max-spread", type=float, default=5.0, help="Max bid/ask spread in cents (default: 5). Candles without bid/ask are skipped.")
    args = parser.parse_args()

    data_path = "kalshi_backtest_data.json"
    markets = load_kalshi_data(data_path)

    if args.decay_hl and args.decay_hl > 0:
        yes_counts, total_counts = build_counts_weighted(markets, decay_half_life_days=args.decay_hl, max_spread_cents=args.max_spread)
    else:
        yes_counts, total_counts = build_counts(markets, max_spread_cents=args.max_spread)
    # Remove 0-cent row from counts before any bucketing
    total_counts = strip_zero_cent_row(total_counts)[0]
    yes_counts = strip_zero_cent_row(yes_counts)[0]

    run_for_bucket_size(
        yes_counts,
        total_counts,
        markets,
        args.bucket_size,
        label_min_samples=args.min_samples,
        detailed_charts=args.detailed_charts,
        worst_price=args.worst_price,
        kelly_frac=args.kelly_frac,
        output_table=args.output_table,
        rolling_decay_hl=args.rolling_decay_hl,
        max_spread_cents=args.max_spread,
    )
    summarize_bucket_sweep(yes_counts, total_counts, [2, 5, 10], min_samples=args.min_samples, worst_price=args.worst_price)

if __name__ == "__main__":
    main()
