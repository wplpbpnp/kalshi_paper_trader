#!/usr/bin/env python3
"""Rolling-window backtest for Kalshi decision tables.

Builds a table on a training window, then applies it to the following test window.
Uses fixed settings from a provided table JSON.
"""

import argparse
import json
import math
from datetime import datetime, timedelta

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_json(path):
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


def get_yes_price(candle, max_spread_cents=None):
    yes_bid = candle.get("yes_bid", {}).get("close", 0)
    yes_ask = candle.get("yes_ask", {}).get("close", 100)
    if yes_bid > 0 and yes_ask < 100:
        spread = yes_ask - yes_bid
        if max_spread_cents is not None and spread > max_spread_cents:
            return None
        return (yes_bid + yes_ask) / 2 / 100.0
    price = candle.get("price", {}).get("close", 50)
    if price is None:
        return None
    return price / 100.0


def fee_for_trade(price, contracts):
    raw = 0.07 * contracts * price * (1 - price)
    return math.ceil(raw * 100) / 100.0


def wilson_lower_bound(wins, total, z=1.96):
    if total == 0:
        return 0.0
    phat = wins / total
    denom = 1 + (z * z) / total
    center = phat + (z * z) / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * total)) / total)
    return max(0.0, (center - margin) / denom)


def build_counts(markets, max_minutes, max_spread_cents=None):
    yes_counts = np.zeros((99, max_minutes), dtype=int)  # cents 1..99 -> index 0..98
    total_counts = np.zeros((99, max_minutes), dtype=int)

    for m in markets:
        result = m.get("result", "").lower()
        if result not in ("yes", "no"):
            continue
        candles = m.get("candlesticks", [])
        if len(candles) < 2:
            continue
        for idx, candle in enumerate(candles):
            if idx >= max_minutes:
                break
            yes_price = get_yes_price(candle, max_spread_cents=max_spread_cents)
            if yes_price is None or yes_price <= 0 or yes_price >= 1:
                continue
            price_cent = int(round(yes_price * 100))
            if price_cent <= 0 or price_cent >= 100:
                continue
            row = price_cent - 1
            total_counts[row, idx] += 1
            if result == "yes":
                yes_counts[row, idx] += 1

    return yes_counts, total_counts


def aggregate_counts(yes_counts, total_counts, bucket_size):
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
        ranges.append((start + 1, end))  # cents inclusive
    return agg_yes, agg_total, ranges


def compute_matrices(yes_counts, total_counts):
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

    return p_yes, p_yes_lb, p_no_lb


def compute_kelly(p_yes, price_yes):
    fee_yes = math.ceil(0.07 * price_yes * (1 - price_yes) * 100) / 100.0
    win_yes = 1.0 - price_yes - fee_yes
    loss_yes = price_yes + fee_yes
    if win_yes <= 0 or loss_yes <= 0:
        return 0.0
    b = win_yes / loss_yes
    k = (p_yes * b - (1 - p_yes)) / b
    return max(0.0, k)


def build_table(markets, bucket_size, min_samples, worst_price, kelly_frac, max_spread_cents=None):
    max_minutes = max((len(m.get("candlesticks", [])) for m in markets), default=0)
    if max_minutes == 0:
        return {"cells": [], "worst_price": worst_price}

    yes_counts, total_counts = build_counts(markets, max_minutes, max_spread_cents=max_spread_cents)

    if bucket_size > 1:
        yes_counts, total_counts, ranges = aggregate_counts(yes_counts, total_counts, bucket_size)
    else:
        rows = yes_counts.shape[0]
        ranges = [(i + 1, i + 1) for i in range(rows)]

    p_yes, p_yes_lb, p_no_lb = compute_matrices(yes_counts, total_counts)

    cells = []
    rows, cols = total_counts.shape
    for i in range(rows):
        price_low = ranges[i][0] / 100.0
        price_high = ranges[i][1] / 100.0
        price_mid = (price_low + price_high) / 2
        for j in range(cols):
            if total_counts[i, j] < min_samples:
                continue

            p_yes_c = p_yes_lb[i, j]
            p_no_c = p_no_lb[i, j]
            if np.isnan(p_yes_c) or np.isnan(p_no_c):
                continue

            if worst_price:
                price_yes = price_high
                price_no = 1.0 - price_low
            else:
                price_yes = price_mid
                price_no = 1.0 - price_mid

            # EV for YES
            fee_yes = math.ceil(0.07 * price_yes * (1 - price_yes) * 100) / 100.0
            ev_yes = p_yes_c * (1 - price_yes) - (1 - p_yes_c) * price_yes - fee_yes
            if ev_yes > 0:
                k_yes = compute_kelly(p_yes_c, price_yes) * kelly_frac
                cells.append({
                    "price_range": [price_low, price_high],
                    "time_range_minutes": [j, j + 1],
                    "side": "YES",
                    "expected_ev_pct": round(float(ev_yes * 100), 4),
                    "kelly_fraction": round(float(k_yes), 6),
                })

            # EV for NO
            fee_no = math.ceil(0.07 * price_no * (1 - price_no) * 100) / 100.0
            ev_no = p_no_c * (1 - price_no) - (1 - p_no_c) * price_no - fee_no
            if ev_no > 0:
                k_no = compute_kelly(p_no_c, price_no) * kelly_frac
                cells.append({
                    "price_range": [price_low, price_high],
                    "time_range_minutes": [j, j + 1],
                    "side": "NO",
                    "expected_ev_pct": round(float(ev_no * 100), 4),
                    "kelly_fraction": round(float(k_no), 6),
                })

    return {"cells": cells, "worst_price": worst_price}


def build_index(table):
    by_time = {}
    for cell in table["cells"]:
        t0, t1 = cell["time_range_minutes"]
        for t in range(t0, t1):
            by_time.setdefault(t, []).append(cell)
    return by_time


def in_price_range(price, price_range):
    low, high = price_range
    return low <= price <= high


def pick_trade_yes_price(price, cell, worst_price):
    if not worst_price:
        return price
    low, high = cell["price_range"]
    if cell["side"] == "YES":
        return high
    return low


def simulate_table(table, markets, bankroll, max_market_exposure_pct=0.0, max_spread_cents=None):
    worst_price = bool(table.get("worst_price"))
    by_time = build_index(table)

    cash = bankroll
    stats = {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "gross_pnl": 0.0,
        "fees": 0.0,
        "markets_traded": 0,
        "total_markets": 0,
    }
    equity = []

    for market in markets:
        result = market.get("result", "").lower()
        if result not in ("yes", "no"):
            continue
        stats["total_markets"] += 1

        open_positions = []
        market_spent = 0.0
        market_had_trade = False

        candles = market.get("candlesticks", [])
        for t_idx, candle in enumerate(candles):
            if t_idx not in by_time:
                continue
            yes_price = get_yes_price(candle, max_spread_cents=max_spread_cents)
            if yes_price is None or yes_price <= 0 or yes_price >= 1:
                continue

            for cell in by_time[t_idx]:
                if not in_price_range(yes_price, cell["price_range"]):
                    continue

                side = cell["side"]
                trade_yes_price = pick_trade_yes_price(yes_price, cell, worst_price)
                if side == "NO":
                    price = 1.0 - trade_yes_price
                else:
                    price = trade_yes_price

                kelly = cell["kelly_fraction"]
                if kelly <= 0:
                    continue

                cost_per_contract = price
                if cost_per_contract <= 0:
                    continue
                contracts = math.floor((kelly * cash) / cost_per_contract)
                if contracts <= 0:
                    continue

                fee = fee_for_trade(price, contracts)
                total_cost = contracts * cost_per_contract + fee
                if max_market_exposure_pct > 0:
                    cap = max_market_exposure_pct * cash
                    if market_spent + total_cost > cap:
                        continue
                if total_cost > cash:
                    continue

                cash -= total_cost
                market_spent += total_cost
                open_positions.append({
                    "side": side,
                    "price": price,
                    "contracts": contracts,
                    "fee": fee,
                })
                market_had_trade = True

        for pos in open_positions:
            side = pos["side"]
            price = pos["price"]
            contracts = pos["contracts"]
            fee = pos["fee"]
            won = (result == "yes" and side == "YES") or (result == "no" and side == "NO")
            payout = contracts * (1.0 if won else 0.0)
            pnl = payout - (contracts * price) - fee

            cash += payout
            stats["trades"] += 1
            if won:
                stats["wins"] += 1
            else:
                stats["losses"] += 1
            stats["gross_pnl"] += pnl
            stats["fees"] += fee

        equity.append(cash)
        if market_had_trade:
            stats["markets_traded"] += 1

    return stats, equity, cash


def plot_equity_curve(equity, output_path):
    if not HAS_MATPLOTLIB:
        return
    if not equity:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(equity)), equity, color="black", linewidth=1.5)
    plt.title("Equity Curve (rolling test markets)")
    plt.xlabel("Test Market Index")
    plt.ylabel("Bankroll ($)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_drawdown(equity):
    max_dd = 0.0
    peak = equity[0] if equity else 0.0
    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_sharpe(equity):
    returns = []
    for i in range(1, len(equity)):
        if equity[i - 1] > 0:
            returns.append((equity[i] - equity[i - 1]) / equity[i - 1])
    if len(returns) < 2:
        return 0.0
    avg_r = sum(returns) / len(returns)
    var = sum((r - avg_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    return (avg_r / std) * math.sqrt(len(returns)) if std > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Rolling-window backtest for Kalshi decision tables")
    parser.add_argument("--settings-table", required=True, help="Path to a decision table JSON to read settings")
    parser.add_argument("--data", default="kalshi_backtest_data.json", help="Path to Kalshi data JSON")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Starting bankroll (default: 100)")
    parser.add_argument("--train-weeks", type=int, default=4, help="Training window length in weeks (default: 4)")
    parser.add_argument("--test-weeks", type=int, default=1, help="Test window length in weeks (default: 1)")
    parser.add_argument("--step-weeks", type=int, default=1, help="Step size in weeks (default: 1)")
    parser.add_argument("--max-market-exposure", type=float, default=0.0, help="Max fraction of bankroll per market (e.g., 0.05 for 5%%)")
    parser.add_argument("--equity-chart", default="rolling_equity_curve.png", help="Output path for equity curve PNG")
    parser.add_argument("--override-worst-price", action="store_true", help="Override table worst_price to True")
    parser.add_argument("--override-mid-price", action="store_true", help="Override table worst_price to False")
    parser.add_argument("--max-spread", type=float, default=5.0, help="Max bid/ask spread in cents (default: 5). Candles without bid/ask are skipped.")
    parser.add_argument("--fixed-table", action="store_true", help="Use settings table as-is for all windows (no retraining)")
    args = parser.parse_args()

    settings = load_json(args.settings_table)
    bucket_size = settings.get("bucket_size_cents", 1)
    min_samples = settings.get("min_samples", 30)
    worst_price = settings.get("worst_price", False)
    if args.override_worst_price:
        worst_price = True
    if args.override_mid_price:
        worst_price = False
    kelly_frac = settings.get("kelly_fractional_multiplier", 0.25)

    data = load_json(args.data)
    markets = []
    for m in data:
        t = parse_market_time(m)
        if t is None:
            continue
        m["_close_time"] = t
        markets.append(m)

    markets.sort(key=lambda x: x["_close_time"])
    if not markets:
        print("No markets with close_time available.")
        return

    start_time = markets[0]["_close_time"]
    end_time = markets[-1]["_close_time"]

    train_delta = timedelta(weeks=args.train_weeks)
    test_delta = timedelta(weeks=args.test_weeks)
    step_delta = timedelta(weeks=args.step_weeks)

    bankroll = args.bankroll
    overall_equity = []
    overall_stats = {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "gross_pnl": 0.0,
        "fees": 0.0,
        "markets_traded": 0,
        "total_markets": 0,
    }

    window_start = start_time
    window_idx = 0
    while window_start + train_delta + test_delta <= end_time:
        train_start = window_start
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta

        train_markets = [m for m in markets if train_start <= m["_close_time"] < train_end]
        test_markets = [m for m in markets if test_start <= m["_close_time"] < test_end]

        if args.fixed_table:
            table = settings
        else:
            table = build_table(train_markets, bucket_size, min_samples, worst_price, kelly_frac, max_spread_cents=args.max_spread)
        stats, equity, bankroll = simulate_table(table, test_markets, bankroll, max_market_exposure_pct=args.max_market_exposure, max_spread_cents=args.max_spread)

        overall_equity.extend(equity)
        for k in overall_stats:
            overall_stats[k] += stats[k]

        window_idx += 1
        window_start = window_start + step_delta

    max_dd = compute_drawdown(overall_equity)
    sharpe = compute_sharpe(overall_equity)
    total_return = ((bankroll / args.bankroll) - 1) * 100
    market_pct = (overall_stats["markets_traded"] / overall_stats["total_markets"] * 100) if overall_stats["total_markets"] > 0 else 0.0

    print("Rolling backtest results")
    print("=" * 60)
    print(f"Windows: {window_idx}")
    print(f"Trades: {overall_stats['trades']}")
    print(f"Wins:   {overall_stats['wins']}")
    print(f"Losses: {overall_stats['losses']}")
    print(f"Fees:   ${overall_stats['fees']:.2f}")
    print(f"PnL:    ${overall_stats['gross_pnl']:.2f}")
    print(f"Final bankroll: ${bankroll:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Markets traded: {overall_stats['markets_traded']} / {overall_stats['total_markets']} ({market_pct:.1f}%)")
    print(f"Timespan: {(end_time - start_time).total_seconds() / (7 * 24 * 3600):.2f} weeks")
    print(f"Max drawdown: {max_dd*100:.2f}%")
    print(f"Sharpe (per-market): {sharpe:.2f}")

    plot_equity_curve(overall_equity, args.equity_chart)


if __name__ == "__main__":
    main()
