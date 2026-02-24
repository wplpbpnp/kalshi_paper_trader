#!/usr/bin/env python3
"""Backtest a decision table against Kalshi historical data.

Policy:
- When price and time are inside a table cell, place a trade on the specified side
  sized by Kelly fraction (from the table) of current bankroll.
- Multiple signals per market are allowed.
- Uses mid price from data by default, or worst price from table buckets if
  table's worst_price is true.
"""

import argparse
import json
import math
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


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
    # fee = ceil(0.07 * C * P * (1-P)) to next cent
    raw = 0.07 * contracts * price * (1 - price)
    return math.ceil(raw * 100) / 100.0


def build_index(table):
    # Index by time bucket and side for quick lookup
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
    # NO side worst price is highest NO price => lowest YES price
    return low


def parse_market_time(market):
    close_time_str = market.get("close_time")
    if not close_time_str:
        return None
    try:
        return datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
    except Exception:
        return None


def simulate(table_path, data_path, bankroll, max_market_exposure_pct=0.0, max_spread_cents=None):
    table = load_json(table_path)
    data = load_json(data_path)

    worst_price = bool(table.get("worst_price"))
    by_time = build_index(table)

    cash = bankroll
    equity_curve = []
    stats = {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "gross_pnl": 0.0,
        "fees": 0.0,
        "final_bankroll": bankroll,
        "markets_traded": 0,
        "total_markets": 0,
        "start_time": None,
        "end_time": None,
    }

    for market in data:
        result = market.get("result", "").lower()
        if result not in ("yes", "no"):
            continue
        stats["total_markets"] += 1

        m_time = parse_market_time(market)
        if m_time:
            if stats["start_time"] is None or m_time < stats["start_time"]:
                stats["start_time"] = m_time
            if stats["end_time"] is None or m_time > stats["end_time"]:
                stats["end_time"] = m_time

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
                    price = 1.0 - trade_yes_price  # NO price
                else:
                    price = trade_yes_price

                kelly = cell["kelly_fraction"]
                if kelly <= 0:
                    continue

                # Option A sizing: cost-based
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

        # Resolve positions for this market
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
        equity_curve.append(cash)
        if market_had_trade:
            stats["markets_traded"] += 1

    stats["final_bankroll"] = cash
    return stats, equity_curve


def plot_equity_curve(equity, output_path):
    if not HAS_MATPLOTLIB:
        return
    if not equity:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(equity)), equity, color="black", linewidth=1.5)
    plt.title("Equity Curve (per market)")
    plt.xlabel("Market Index")
    plt.ylabel("Bankroll ($)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Backtest decision table against Kalshi data")
    parser.add_argument("--table", required=True, help="Path to decision table JSON")
    parser.add_argument("--data", default="kalshi_backtest_data.json", help="Path to Kalshi data JSON")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Starting bankroll (default: 100)")
    parser.add_argument("--max-market-exposure", type=float, default=0.0, help="Max fraction of bankroll per market (e.g., 0.05 for 5%%)")
    parser.add_argument("--max-spread", type=float, default=5.0, help="Max bid/ask spread in cents (default: 5). Candles without bid/ask are skipped.")
    parser.add_argument("--equity-chart", default="equity_curve.png", help="Output path for equity curve PNG")
    args = parser.parse_args()

    stats, equity = simulate(args.table, args.data, args.bankroll, max_market_exposure_pct=args.max_market_exposure, max_spread_cents=args.max_spread)
    max_drawdown = 0.0
    peak = equity[0] if equity else args.bankroll
    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    returns = []
    for i in range(1, len(equity)):
        if equity[i-1] > 0:
            returns.append((equity[i] - equity[i-1]) / equity[i-1])
    avg_return = sum(returns) / len(returns) if returns else 0.0
    variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1) if len(returns) > 1 else 0.0
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    sharpe = (avg_return / std_dev) * math.sqrt(len(returns)) if std_dev > 0 else 0.0

    market_pct = (stats["markets_traded"] / stats["total_markets"] * 100) if stats["total_markets"] > 0 else 0.0
    timespan_weeks = 0.0
    if stats["start_time"] and stats["end_time"]:
        timespan_weeks = (stats["end_time"] - stats["start_time"]).total_seconds() / (7 * 24 * 3600)
    print("Backtest results")
    print("=" * 60)
    print(f"Trades: {stats['trades']}")
    print(f"Wins:   {stats['wins']}")
    print(f"Losses: {stats['losses']}")
    print(f"Fees:   ${stats['fees']:.2f}")
    print(f"PnL:    ${stats['gross_pnl']:.2f}")
    print(f"Final bankroll: ${stats['final_bankroll']:.2f}")
    print(f"Total return: {((stats['final_bankroll'] / args.bankroll) - 1) * 100:.2f}%")
    print(f"Markets traded: {stats['markets_traded']} / {stats['total_markets']} ({market_pct:.1f}%)")
    print(f"Timespan: {timespan_weeks:.2f} weeks")
    print(f"Max drawdown: {max_drawdown*100:.2f}%")
    print(f"Sharpe (per-market): {sharpe:.2f}")

    plot_equity_curve(equity, args.equity_chart)


if __name__ == "__main__":
    main()
