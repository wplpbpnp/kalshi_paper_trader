# kalshi_edge_finder.py
# Tool for discovering edge in Kalshi recurring market series

import os
import sys
import requests
import base64
import time
import json
import argparse
import math
from datetime import datetime, timezone
from collections import defaultdict

# Add parent dir for shared auth (or copy it)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving files
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

# ============== CONFIG ==============
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")

# ============== AUTH ==============
def load_auth():
    """Load API credentials from parent directory"""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    api_key = os.getenv("KALSHI_API_KEY")
    if not api_key:
        env_path = os.path.join(parent_dir, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("KALSHI_API_KEY=") or line.startswith("API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break

    pem_path = os.path.join(parent_dir, "kalshi_secret.pem")
    with open(pem_path, "r") as f:
        secret_key_str = f.read().strip()

    if not secret_key_str.startswith("-----"):
        secret_key_str = f"-----BEGIN RSA PRIVATE KEY-----\n{secret_key_str}\n-----END RSA PRIVATE KEY-----"

    private_key = serialization.load_pem_private_key(
        secret_key_str.encode(),
        password=None
    )

    return api_key, private_key

API_KEY, PRIVATE_KEY = load_auth()
BASE_URL = "https://api.elections.kalshi.com"

def sign_pss_text(text: str) -> str:
    message = text.encode('utf-8')
    signature = PRIVATE_KEY.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')

def get_headers(method: str, path: str):
    timestamp = str(int(time.time() * 1000))
    path_for_signing = path.split('?')[0]
    msg = timestamp + method + path_for_signing
    signature = sign_pss_text(msg)
    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": API_KEY,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }

def api_get(path: str, params: dict = None):
    params = params or {}
    headers = get_headers("GET", path)
    resp = requests.get(BASE_URL + path, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()


# ============== SERIES DISCOVERY ==============
def list_series():
    """List all available series on Kalshi"""
    print("Fetching available series...")

    # Get all events to find series patterns
    all_series = set()
    cursor = None

    while True:
        params = {"limit": 200, "status": "settled"}
        if cursor:
            params["cursor"] = cursor

        data = api_get("/trade-api/v2/markets", params)
        markets = data.get("markets", [])

        for market in markets:
            ticker = market.get("ticker", "")
            # Extract series prefix (e.g., KXBTC15M from KXBTC15M-26JAN240215-15)
            if "-" in ticker:
                series = ticker.split("-")[0]
                all_series.add(series)

        cursor = data.get("cursor")
        if not cursor or not markets:
            break

        print(f"  Scanned {len(all_series)} series so far...")
        time.sleep(0.1)

    return sorted(all_series)


def get_series_info(series_ticker: str):
    """Get info about a specific series"""
    params = {
        "series_ticker": series_ticker,
        "limit": 1
    }
    try:
        data = api_get("/trade-api/v2/markets", params)
        markets = data.get("markets", [])
        if markets:
            m = markets[0]
            return {
                "series": series_ticker,
                "sample_ticker": m.get("ticker"),
                "title": m.get("title"),
                "market_type": m.get("market_type"),
            }
    except:
        pass
    return None


# ============== DATA DOWNLOAD ==============
def download_series_data(series_ticker: str, update_only: bool = False):
    """Download all settled markets for a series"""
    os.makedirs(DATA_DIR, exist_ok=True)
    data_file = os.path.join(DATA_DIR, f"{series_ticker}.json")

    existing_data = []
    existing_tickers = set()

    if update_only and os.path.exists(data_file):
        with open(data_file, "r") as f:
            existing_data = json.load(f)
            existing_tickers = {m['ticker'] for m in existing_data}
        print(f"Loaded {len(existing_data)} existing markets")

    print(f"Fetching settled markets for {series_ticker}...")
    all_markets = []
    cursor = None

    while True:
        params = {
            "series_ticker": series_ticker,
            "status": "settled",
            "limit": 200
        }
        if cursor:
            params["cursor"] = cursor

        data = api_get("/trade-api/v2/markets", params)
        markets = data.get("markets", [])

        for m in markets:
            if m['ticker'] not in existing_tickers:
                # Only keep fields we need
                all_markets.append({
                    'ticker': m['ticker'],
                    'result': m.get('result'),
                    'close_time': m.get('close_time'),
                    'yes_bid': m.get('yes_bid'),
                    'yes_ask': m.get('yes_ask'),
                    'no_bid': m.get('no_bid'),
                    'no_ask': m.get('no_ask'),
                    'volume': m.get('volume'),
                    'floor_strike': m.get('floor_strike'),
                    'cap_strike': m.get('cap_strike'),
                })

        cursor = data.get("cursor")
        print(f"  Fetched {len(all_markets)} new markets...")

        if not cursor or not markets:
            break

        time.sleep(0.1)

    # Combine and save
    combined = existing_data + all_markets
    with open(data_file, "w") as f:
        json.dump(combined, f)

    print(f"Saved {len(combined)} total markets to {data_file}")
    return combined


def load_series_data(series_ticker: str):
    """Load cached series data"""
    data_file = os.path.join(DATA_DIR, f"{series_ticker}.json")
    if not os.path.exists(data_file):
        print(f"No data found for {series_ticker}. Run --download {series_ticker} first.")
        return None

    with open(data_file, "r") as f:
        return json.load(f)


# ============== CANDLESTICK DATA ==============
def get_candlesticks(ticker: str, start_ts: int, end_ts: int, series_ticker=None):
    """
    Get 1-minute candlesticks for a market.

    Kalshi's API has had multiple candlestick endpoints over time. We try:
      1) /trade-api/v2/series/{SERIES}/markets/{TICKER}/candlesticks (newer)
      2) /trade-api/v2/markets/{TICKER}/candlesticks (older)

    Also, some endpoints expect seconds and some milliseconds; we attempt seconds first,
    then retry with ms if we got an empty response.
    """
    def _fetch(path: str, params: dict):
        data = api_get(path, params)
        return data.get("candlesticks", [])

    base_params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": 1,  # 1 minute candles
    }

    paths = []
    if series_ticker:
        paths.append(f"/trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticks")
    paths.append(f"/trade-api/v2/markets/{ticker}/candlesticks")

    for path in paths:
        try:
            candles = _fetch(path, dict(base_params))
            if candles:
                return candles
            # Retry with milliseconds if empty.
            ms_params = dict(base_params)
            ms_params["start_ts"] = int(start_ts * 1000)
            ms_params["end_ts"] = int(end_ts * 1000)
            candles = _fetch(path, ms_params)
            if candles:
                return candles
        except Exception:
            continue

    return []


def download_series_candles(series_ticker: str, update_only: bool = False):
    """Download candlestick data for all markets in a series"""
    os.makedirs(DATA_DIR, exist_ok=True)
    candles_file = os.path.join(DATA_DIR, f"{series_ticker}_candles.json")

    existing_data = []
    existing_tickers = set()

    if update_only and os.path.exists(candles_file):
        with open(candles_file, "r") as f:
            existing_data = json.load(f)
            existing_tickers = {m['ticker'] for m in existing_data}
        print(f"Loaded {len(existing_data)} existing markets with candles")

    # First load market data
    markets = load_series_data(series_ticker)
    if not markets:
        print(f"No market data found. Run --download {series_ticker} first.")
        return None

    # Filter to markets we don't have candles for
    markets_to_fetch = [m for m in markets if m['ticker'] not in existing_tickers]

    if not markets_to_fetch:
        print("No new markets to fetch candles for")
        return existing_data

    print(f"Fetching candlesticks for {len(markets_to_fetch)} markets...")

    new_data = []
    for i, market in enumerate(markets_to_fetch):
        ticker = market['ticker']
        result = market.get('result')

        # Skip markets without results
        if result not in ['yes', 'no']:
            continue

        try:
            close_time = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))
            end_ts = int(close_time.timestamp())
            # KXBTC15M markets are 15 minutes long; request within the market window.
            start_ts = end_ts - (15 * 60)

            candles = get_candlesticks(ticker, start_ts, end_ts, series_ticker=series_ticker)

            if candles:
                new_data.append({
                    'ticker': ticker,
                    'result': result,
                    'close_time': market['close_time'],
                    # Keep schema compatible with older datasets in this repo.
                    # For KXBTC15M, floor_strike is the 15m "open" reference.
                    'strike': market.get('floor_strike'),
                    'floor_strike': market.get('floor_strike'),
                    'yes_bid': market.get('yes_bid'),
                    'yes_ask': market.get('yes_ask'),
                    'no_bid': market.get('no_bid'),
                    'no_ask': market.get('no_ask'),
                    'volume': market.get('volume'),
                    'candlesticks': candles
                })

            time.sleep(0.1)  # Rate limit

        except Exception as e:
            print(f"Error on {ticker}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(markets_to_fetch)} markets, collected {len(new_data)} with candles")

    # Combine and save
    combined = existing_data + new_data
    with open(candles_file, "w") as f:
        json.dump(combined, f)

    print(f"Saved {len(combined)} total markets with candles to {candles_file}")
    return combined


def load_series_candles(series_ticker: str):
    """Load cached candlestick data"""
    candles_file = os.path.join(DATA_DIR, f"{series_ticker}_candles.json")
    if not os.path.exists(candles_file):
        print(f"No candle data found for {series_ticker}. Run --download-candles {series_ticker} first.")
        return None

    with open(candles_file, "r") as f:
        return json.load(f)


# ============== FAIR VALUE SURFACE ANALYSIS ==============
def get_btc_price_at_time(market, minutes_before_close):
    """
    Estimate BTC price at a specific time before close.
    Uses the strike price and YES price to infer where BTC is relative to strike.

    Returns: (btc_estimate, distance_from_strike_pct, yes_mid)
    """
    candles = market.get('candlesticks', [])
    if not candles:
        return None, None, None

    strike = market.get('floor_strike')
    if not strike:
        return None, None, None

    close_time = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))

    # Find candle at target time
    best_candle = None
    best_diff = float('inf')

    for candle in candles:
        end_ts = candle.get('end_period_ts')
        if not end_ts:
            continue

        if isinstance(end_ts, int):
            candle_time = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        else:
            candle_time = datetime.fromisoformat(str(end_ts).replace('Z', '+00:00'))

        time_remaining = (close_time - candle_time).total_seconds() / 60
        diff = abs(time_remaining - minutes_before_close)

        if diff < best_diff and diff < 0.5:
            best_diff = diff
            best_candle = candle

    if not best_candle:
        return None, None, None

    # Extract YES price
    yes_bid_data = best_candle.get('yes_bid') or {}
    yes_ask_data = best_candle.get('yes_ask') or {}

    if isinstance(yes_bid_data, dict):
        yes_bid = yes_bid_data.get('close') or 0
    else:
        yes_bid = yes_bid_data or 0

    if isinstance(yes_ask_data, dict):
        yes_ask = yes_ask_data.get('close') or 0
    else:
        yes_ask = yes_ask_data or 0

    if not yes_bid or not yes_ask:
        return None, None, None

    if yes_bid > 1:
        yes_bid = yes_bid / 100
    if yes_ask > 1:
        yes_ask = yes_ask / 100

    yes_mid = (yes_bid + yes_ask) / 2

    # For KXBTC15M markets, YES = "BTC will be >= strike"
    # If YES is priced high (>0.5), BTC is likely above strike
    # If YES is priced low (<0.5), BTC is likely below strike
    #
    # We can use the "open" price data or estimate from the YES price itself
    # For now, we'll track relative to strike using the candle's price data

    # Get the actual BTC price from the candle if available
    price_data = best_candle.get('price') or {}
    if isinstance(price_data, dict):
        btc_ref = price_data.get('close')
    else:
        btc_ref = price_data

    # If we have price data, calculate distance from strike
    # Otherwise, estimate from YES price
    if btc_ref and btc_ref > 100:  # Looks like a BTC price
        distance_pct = (btc_ref - strike) / strike * 100
        return btc_ref, distance_pct, yes_mid

    # Fallback: estimate based on YES price
    # This is less accurate but still useful
    # If YES = 0.7, BTC is probably ~0.7 "probability units" above strike
    # We'll use a rough mapping based on typical volatility

    return None, None, yes_mid


def extract_distance_and_price(market, minutes_before_close):
    """
    Extract distance proxy and YES mid price at a specific time.

    Distance proxy = |YES_mid - 0.5| (how confident market is in one side)
    This correlates with actual distance from strike.
    """
    candles = market.get('candlesticks', [])
    if not candles:
        return None, None, None

    close_time = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))

    # Find candle at target time
    best_candle = None
    best_diff = float('inf')

    for candle in candles:
        end_ts = candle.get('end_period_ts')
        if not end_ts:
            continue

        if isinstance(end_ts, int):
            candle_time = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        else:
            candle_time = datetime.fromisoformat(str(end_ts).replace('Z', '+00:00'))

        time_remaining = (close_time - candle_time).total_seconds() / 60
        diff = abs(time_remaining - minutes_before_close)

        if diff < best_diff and diff < 1.0:  # Allow 1 min tolerance
            best_diff = diff
            best_candle = candle

    if not best_candle:
        return None, None, None

    # Extract YES price - try yes_bid/yes_ask first, fall back to price
    yes_bid_data = best_candle.get('yes_bid') or {}
    yes_ask_data = best_candle.get('yes_ask') or {}

    if isinstance(yes_bid_data, dict):
        yes_bid = yes_bid_data.get('close') or 0
    else:
        yes_bid = yes_bid_data or 0

    if isinstance(yes_ask_data, dict):
        yes_ask = yes_ask_data.get('close') or 0
    else:
        yes_ask = yes_ask_data or 0

    # Fallback to price field
    if not yes_bid or not yes_ask:
        price_data = best_candle.get('price') or {}
        if isinstance(price_data, dict):
            price_val = price_data.get('close') or 0
        else:
            price_val = price_data or 0
        if price_val:
            yes_bid = yes_bid or price_val
            yes_ask = yes_ask or price_val

    if not yes_bid and not yes_ask:
        return None, None, None

    # Handle cents vs dollars
    if yes_bid > 1:
        yes_bid = yes_bid / 100
    if yes_ask > 1:
        yes_ask = yes_ask / 100

    yes_mid = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else (yes_bid or yes_ask)

    if yes_mid <= 0 or yes_mid >= 1:
        return None, None, None

    # Proxy for distance: how far is YES from 0.50?
    # This correlates with BTC's distance from strike
    proxy_distance = abs(yes_mid - 0.5)
    leader_side = 'YES' if yes_mid > 0.5 else 'NO'

    return proxy_distance, yes_mid, leader_side


def calculate_surface_calibration(markets, time_slices=[13, 10, 7, 5, 3, 2]):
    """
    Calculate fair value surface: actual win rate by (distance proxy, time remaining).

    Returns dict of {time: {distance_bucket: {'count': N, 'wins': N, 'win_rate': R, 'avg_price': P}}}
    """
    # Distance buckets (proxy distance = |YES - 0.5|)
    distance_buckets = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]

    results = {}

    for t in time_slices:
        buckets = defaultdict(lambda: {'count': 0, 'leader_wins': 0, 'prices': []})

        for market in markets:
            result = market.get('result')
            if result not in ['yes', 'no']:
                continue

            data = extract_distance_and_price(market, t)
            if data[0] is None:
                continue

            proxy_distance, yes_mid, leader_side = data

            # Find distance bucket
            bucket = None
            for b in distance_buckets:
                if proxy_distance <= b:
                    bucket = b
                    break
            if bucket is None:
                bucket = 0.3  # overflow bucket

            # Leader wins if:
            # - leader_side == 'YES' and result == 'yes'
            # - leader_side == 'NO' and result == 'no'
            leader_won = (leader_side == 'YES' and result == 'yes') or \
                        (leader_side == 'NO' and result == 'no')

            buckets[bucket]['count'] += 1
            if leader_won:
                buckets[bucket]['leader_wins'] += 1

            # Track the "leader price" (how confident market was in leader)
            leader_price = yes_mid if leader_side == 'YES' else (1 - yes_mid)
            buckets[bucket]['prices'].append(leader_price)

        # Calculate win rates and average prices
        for bucket in buckets.values():
            if bucket['count'] > 0:
                bucket['win_rate'] = bucket['leader_wins'] / bucket['count']
                bucket['avg_price'] = sum(bucket['prices']) / len(bucket['prices'])
            else:
                bucket['win_rate'] = 0
                bucket['avg_price'] = 0

        results[t] = dict(buckets)

    return results


def plot_fair_value_surface(surface_data, series_ticker, output_dir=None):
    """Plot fair value surface: leader win rate by distance and time"""
    os.makedirs(output_dir or CHARTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Fair value surface (actual win rates)
    time_slices = sorted(surface_data.keys(), reverse=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_slices)))

    for t, color in zip(time_slices, colors):
        buckets = surface_data[t]
        distances = sorted(buckets.keys())
        win_rates = [buckets[d]['win_rate'] for d in distances]
        counts = [buckets[d]['count'] for d in distances]

        # Only plot points with enough data
        valid = [(d, wr) for d, wr, c in zip(distances, win_rates, counts) if c >= 10]
        if valid:
            ds, wrs = zip(*valid)
            ax1.plot(ds, wrs, 'o-', color=color, label=f'T-{t} min', markersize=6)

    ax1.set_xlabel('Distance from Strike (proxy: |YES - 0.5|)')
    ax1.set_ylabel('Actual Leader Win Rate')
    ax1.set_title(f'{series_ticker}: Fair Value Surface')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.35)
    ax1.set_ylim(0, 1.0)  # Full range to show mean reversion at small distances

    # Plot 2: Mispricing (actual - market price)
    for t, color in zip(time_slices, colors):
        buckets = surface_data[t]
        distances = sorted(buckets.keys())

        mispricing = []
        valid_distances = []
        for d in distances:
            if buckets[d]['count'] >= 10:
                actual = buckets[d]['win_rate']
                market = buckets[d]['avg_price']
                mispricing.append((actual - market) * 100)  # As percentage
                valid_distances.append(d)

        if valid_distances:
            ax2.plot(valid_distances, mispricing, 'o-', color=color, label=f'T-{t} min', markersize=6)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Distance from Strike (proxy)')
    ax2.set_ylabel('Mispricing: Actual - Market (%)')
    ax2.set_title(f'{series_ticker}: Leader Mispricing by Distance & Time')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.35)

    plt.tight_layout()

    output_path = os.path.join(output_dir or CHARTS_DIR, f"{series_ticker}_fair_value_surface.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved fair value surface chart to {output_path}")
    return output_path


def plot_surface_heatmap(surface_data, series_ticker, output_dir=None):
    """Plot fair value as a heatmap"""
    os.makedirs(output_dir or CHARTS_DIR, exist_ok=True)

    time_slices = sorted(surface_data.keys())
    all_distances = set()
    for t in time_slices:
        all_distances.update(surface_data[t].keys())
    distances = sorted(all_distances)

    # Build matrix
    matrix = []
    for t in time_slices:
        row = []
        for d in distances:
            if d in surface_data[t] and surface_data[t][d]['count'] >= 5:
                row.append(surface_data[t][d]['win_rate'])
            else:
                row.append(np.nan)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1.0)

    ax.set_xticks(range(len(distances)))
    ax.set_xticklabels([f'{d:.2f}' for d in distances], rotation=45)
    ax.set_yticks(range(len(time_slices)))
    ax.set_yticklabels([f'T-{t}' for t in time_slices])

    ax.set_xlabel('Distance from Strike (proxy)')
    ax.set_ylabel('Time Remaining')
    ax.set_title(f'{series_ticker}: Fair Value Surface Heatmap\n(Leader Win Rate)')

    plt.colorbar(im, ax=ax, label='Leader Win Rate')

    # Add text annotations
    for i, t in enumerate(time_slices):
        for j, d in enumerate(distances):
            if not np.isnan(matrix[i, j]):
                count = surface_data[t][d]['count']
                text = f'{matrix[i, j]:.0%}\nn={count}'
                ax.text(j, i, text, ha='center', va='center', fontsize=7)

    plt.tight_layout()

    output_path = os.path.join(output_dir or CHARTS_DIR, f"{series_ticker}_surface_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved surface heatmap to {output_path}")
    return output_path


def print_surface_summary(surface_data, series_ticker):
    """Print summary of fair value surface findings"""
    print(f"\n{'='*70}")
    print(f"FAIR VALUE SURFACE SUMMARY: {series_ticker}")
    print(f"{'='*70}")

    print("\nKey: When 'leader' (side with >50% price) has distance X from strike at time T,")
    print("     what's the actual probability the leader wins vs what market prices?\n")

    print(f"{'Time':<8} | {'Distance':<10} | {'N':<6} | {'Actual':<8} | {'Market':<8} | {'Edge':<8}")
    print("-" * 70)

    # Find biggest edges
    all_edges = []

    for t in sorted(surface_data.keys()):
        for d in sorted(surface_data[t].keys()):
            data = surface_data[t][d]
            if data['count'] >= 10:
                actual = data['win_rate']
                market = data['avg_price']
                edge = actual - market

                print(f"T-{t:<5} | {d:<10.3f} | {data['count']:<6} | {actual:<8.1%} | {market:<8.1%} | {edge:+8.1%}")

                if abs(edge) > 0.03:
                    all_edges.append((t, d, data['count'], actual, market, edge))

    if all_edges:
        print(f"\n{'='*70}")
        print("SIGNIFICANT EDGES (>3% mispricing):")
        print("-" * 70)

        for t, d, n, actual, market, edge in sorted(all_edges, key=lambda x: -abs(x[5])):
            action = "BUY LEADER" if edge > 0 else "BUY UNDERDOG"
            print(f"  T-{t} min, dist={d:.3f}: {action} ({edge:+.1%} edge, n={n})")

    print(f"\n{'='*70}\n")


# Need numpy for surface analysis
try:
    import numpy as np
except ImportError:
    print("numpy not installed. Run: pip install numpy")
    np = None


# ============== TIME-SLICED ANALYSIS ==============
def extract_price_at_time(market, minutes_before_close):
    """Extract YES mid price at a specific time before close"""
    candles = market.get('candlesticks', [])
    if not candles:
        return None

    close_time = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))

    best_candle = None
    best_diff = float('inf')

    for candle in candles:
        end_ts = candle.get('end_period_ts')
        if not end_ts:
            continue

        if isinstance(end_ts, int):
            candle_time = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        else:
            candle_time = datetime.fromisoformat(str(end_ts).replace('Z', '+00:00'))

        time_remaining = (close_time - candle_time).total_seconds() / 60
        diff = abs(time_remaining - minutes_before_close)

        # Find candle closest to target time (within 0.5 min tolerance)
        if diff < best_diff and diff < 0.5:
            best_diff = diff
            best_candle = candle

    if not best_candle:
        return None

    # Extract bid/ask from candle
    yes_bid_data = best_candle.get('yes_bid') or {}
    yes_ask_data = best_candle.get('yes_ask') or {}

    if isinstance(yes_bid_data, dict):
        yes_bid = yes_bid_data.get('close') or 0
    else:
        yes_bid = yes_bid_data or 0

    if isinstance(yes_ask_data, dict):
        yes_ask = yes_ask_data.get('close') or 0
    else:
        yes_ask = yes_ask_data or 0

    # Fallback to price
    if not yes_bid or not yes_ask:
        price_data = best_candle.get('price') or {}
        if isinstance(price_data, dict):
            price_val = price_data.get('close') or 0
        else:
            price_val = price_data or 0

        if not yes_bid and price_val:
            yes_bid = price_val
        if not yes_ask and price_val:
            yes_ask = price_val

    if not yes_bid or not yes_ask:
        return None

    # Convert from cents to dollars
    if yes_bid > 1:
        yes_bid = yes_bid / 100
    if yes_ask > 1:
        yes_ask = yes_ask / 100

    return (yes_bid + yes_ask) / 2


def calculate_time_calibration(markets, minutes_before_close, price_buckets=20):
    """Calculate calibration at a specific time before close"""
    bucket_size = 1.0 / price_buckets
    buckets = defaultdict(lambda: {'count': 0, 'wins': 0})

    valid_count = 0
    for market in markets:
        result = market.get('result')
        if result not in ['yes', 'no']:
            continue

        yes_mid = extract_price_at_time(market, minutes_before_close)
        if yes_mid is None or yes_mid <= 0 or yes_mid >= 1:
            continue

        valid_count += 1

        # Find bucket
        bucket_idx = min(int(yes_mid / bucket_size), price_buckets - 1)
        bucket_mid = (bucket_idx + 0.5) * bucket_size

        buckets[bucket_mid]['count'] += 1
        if result == 'yes':
            buckets[bucket_mid]['wins'] += 1

    # Calculate win rates
    for bucket in buckets.values():
        if bucket['count'] > 0:
            bucket['win_rate'] = bucket['wins'] / bucket['count']
        else:
            bucket['win_rate'] = 0

    return dict(buckets), valid_count


def analyze_time_slices(markets, time_slices=[5, 3, 2, 1]):
    """Analyze calibration at multiple time slices"""
    results = {}
    for t in time_slices:
        calibration, count = calculate_time_calibration(markets, t)
        edge = calculate_edge(calibration)
        results[t] = {
            'calibration': calibration,
            'edge': edge,
            'sample_count': count
        }
    return results


# ============== TIME-SLICED VISUALIZATION ==============
def plot_time_calibration(time_results, series_ticker, output_dir=None):
    """Plot calibration curves at different times before close (2x2 grid)"""
    os.makedirs(output_dir or CHARTS_DIR, exist_ok=True)

    time_slices = sorted(time_results.keys(), reverse=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, t in enumerate(time_slices[:4]):
        ax = axes[idx]
        calibration = time_results[t]['calibration']
        sample_count = time_results[t]['sample_count']

        prices = sorted(calibration.keys())
        actual_rates = [calibration[p]['win_rate'] for p in prices]
        counts = [calibration[p]['count'] for p in prices]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

        # Actual calibration
        ax.scatter(prices, actual_rates, s=50, alpha=0.7, label='Actual')
        ax.plot(prices, actual_rates, 'b-', alpha=0.5)

        # Highlight mispricing
        for p, actual in zip(prices, actual_rates):
            if calibration[p]['count'] >= 5:
                color = 'green' if actual > p else 'red'
                ax.plot([p, p], [p, actual], color=color, linewidth=2, alpha=0.7)

        ax.set_xlabel('YES Market Price ($)')
        ax.set_ylabel('Actual YES Win Rate')
        ax.set_title(f'T-{t} min: Market Calibration (n={sample_count})')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{series_ticker} Calibration by Time to Settlement', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir or CHARTS_DIR, f"{series_ticker}_time_calibration.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved time calibration chart to {output_path}")
    return output_path


def plot_time_edge(time_results, series_ticker, output_dir=None):
    """Plot edge at different times before close"""
    os.makedirs(output_dir or CHARTS_DIR, exist_ok=True)

    time_slices = sorted(time_results.keys(), reverse=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, t in enumerate(time_slices[:4]):
        ax = axes[idx]
        edge_data = time_results[t]['edge']

        if not edge_data:
            ax.set_title(f'T-{t} min: No data')
            continue

        prices = sorted(edge_data.keys())
        yes_ev = [edge_data[p]['yes_ev_per_dollar'] * 100 for p in prices]
        no_ev = [edge_data[p]['no_ev_per_dollar'] * 100 for p in prices]

        width = 0.02
        ax.bar([p - width/2 for p in prices], yes_ev, width, label='BUY YES EV%', color='green', alpha=0.7)
        ax.bar([p + width/2 for p in prices], no_ev, width, label='BUY NO EV%', color='red', alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('YES Price ($)')
        ax.set_ylabel('EV per Dollar (%)')
        ax.set_title(f'T-{t} min: Edge by Price')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0, 1)

    plt.suptitle(f'{series_ticker} Edge by Time to Settlement', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir or CHARTS_DIR, f"{series_ticker}_time_edge.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved time edge chart to {output_path}")
    return output_path


def print_time_edge_summary(time_results, series_ticker):
    """Print summary of edge at each time slice"""
    print(f"\n{'='*70}")
    print(f"TIME-SLICED EDGE SUMMARY: {series_ticker}")
    print(f"{'='*70}")

    for t in sorted(time_results.keys()):
        edge_data = time_results[t]['edge']
        sample = time_results[t]['sample_count']

        # Find best opportunities
        best_no = [(p, d) for p, d in edge_data.items()
                   if d['no_ev_per_dollar'] > 0.02 and d['count'] >= 10]
        best_yes = [(p, d) for p, d in edge_data.items()
                    if d['yes_ev_per_dollar'] > 0.02 and d['count'] >= 10]

        print(f"\nT-{t} min (n={sample}):")

        if best_no:
            best = max(best_no, key=lambda x: x[1]['no_ev_per_dollar'])
            print(f"  Best NO: YES @ ${best[0]:.2f} → {best[1]['no_ev_per_dollar']*100:+.1f}% EV "
                  f"(actual {best[1]['actual_win_rate']*100:.0f}% vs implied {best[0]*100:.0f}%)")

        if best_yes:
            best = max(best_yes, key=lambda x: x[1]['yes_ev_per_dollar'])
            print(f"  Best YES: @ ${best[0]:.2f} → {best[1]['yes_ev_per_dollar']*100:+.1f}% EV "
                  f"(actual {best[1]['actual_win_rate']*100:.0f}% vs implied {best[0]*100:.0f}%)")

        if not best_no and not best_yes:
            print(f"  No significant edge found")

    print(f"\n{'='*70}\n")


# ============== ANALYSIS ==============
def calculate_calibration(markets, price_buckets=10):
    """
    Calculate calibration: for each price bucket, what's the actual win rate?

    Returns dict of {bucket_mid: {'count': N, 'wins': W, 'win_rate': W/N}}
    """
    # Create buckets
    bucket_size = 1.0 / price_buckets
    buckets = defaultdict(lambda: {'count': 0, 'wins': 0})

    for m in markets:
        result = m.get('result')
        if result not in ['yes', 'no']:
            continue

        # Get YES price (use mid of bid/ask, or just bid if no ask)
        yes_bid = m.get('yes_bid') or 0
        yes_ask = m.get('yes_ask') or yes_bid

        if yes_bid == 0 and yes_ask == 0:
            continue

        # Convert from cents to dollars
        if yes_bid > 1:
            yes_bid = yes_bid / 100
        if yes_ask > 1:
            yes_ask = yes_ask / 100

        yes_mid = (yes_bid + yes_ask) / 2

        # Find bucket
        bucket_idx = min(int(yes_mid / bucket_size), price_buckets - 1)
        bucket_mid = (bucket_idx + 0.5) * bucket_size

        buckets[bucket_mid]['count'] += 1
        if result == 'yes':
            buckets[bucket_mid]['wins'] += 1

    # Calculate win rates
    for bucket in buckets.values():
        if bucket['count'] > 0:
            bucket['win_rate'] = bucket['wins'] / bucket['count']
        else:
            bucket['win_rate'] = 0

    return dict(buckets)


def calculate_edge(calibration):
    """
    Calculate edge for each price bucket.
    Edge = actual_win_rate - price (for YES bets)

    If edge > 0 at price P, buying YES is +EV
    If edge < 0 at price P, buying NO is +EV
    """
    edge = {}
    for price, data in calibration.items():
        if data['count'] < 5:  # Need minimum sample
            continue

        actual = data['win_rate']
        implied = price

        # Edge for YES bet
        yes_edge = actual - implied

        # EV per dollar for YES bet: win_rate * (1-price) - (1-win_rate) * price
        # Simplified: win_rate - price
        yes_ev = actual * (1 - price) - (1 - actual) * price

        # EV per dollar for NO bet
        no_ev = (1 - actual) * price - actual * (1 - price)

        edge[price] = {
            'count': data['count'],
            'actual_win_rate': actual,
            'implied_prob': implied,
            'yes_edge': yes_edge,
            'yes_ev_per_dollar': yes_ev,
            'no_ev_per_dollar': no_ev,
            'best_action': 'BUY_YES' if yes_ev > no_ev and yes_ev > 0 else ('BUY_NO' if no_ev > 0 else 'NONE'),
            'best_ev': max(yes_ev, no_ev, 0)
        }

    return edge


# ============== VISUALIZATION ==============
def plot_calibration(calibration, series_ticker, output_dir=None):
    """Plot calibration curve: price vs actual win rate"""
    os.makedirs(output_dir or CHARTS_DIR, exist_ok=True)

    prices = sorted(calibration.keys())
    actual_rates = [calibration[p]['win_rate'] for p in prices]
    counts = [calibration[p]['count'] for p in prices]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

    # Actual calibration
    ax.scatter(prices, actual_rates, s=[c/2 for c in counts], alpha=0.7, label='Actual')
    ax.plot(prices, actual_rates, 'b-', alpha=0.5)

    # Highlight mispricing
    for p, actual in zip(prices, actual_rates):
        if calibration[p]['count'] >= 10:
            color = 'green' if actual > p else 'red'
            ax.plot([p, p], [p, actual], color=color, linewidth=2, alpha=0.7)

    ax.set_xlabel('YES Price ($)', fontsize=12)
    ax.set_ylabel('Actual YES Win Rate', fontsize=12)
    ax.set_title(f'{series_ticker} Calibration\n(bubble size = sample count)', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir or CHARTS_DIR, f"{series_ticker}_calibration.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved calibration chart to {output_path}")
    return output_path


def plot_edge(edge_data, series_ticker, output_dir=None):
    """Plot edge by price bucket"""
    os.makedirs(output_dir or CHARTS_DIR, exist_ok=True)

    prices = sorted(edge_data.keys())
    yes_ev = [edge_data[p]['yes_ev_per_dollar'] * 100 for p in prices]  # As percentage
    no_ev = [edge_data[p]['no_ev_per_dollar'] * 100 for p in prices]

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.035
    ax.bar([p - width/2 for p in prices], yes_ev, width, label='BUY YES EV%', color='green', alpha=0.7)
    ax.bar([p + width/2 for p in prices], no_ev, width, label='BUY NO EV%', color='red', alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('YES Price ($)', fontsize=12)
    ax.set_ylabel('EV per Dollar (%)', fontsize=12)
    ax.set_title(f'{series_ticker} Edge by Price Bucket', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    output_path = os.path.join(output_dir or CHARTS_DIR, f"{series_ticker}_edge.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved edge chart to {output_path}")
    return output_path


def print_edge_summary(edge_data, series_ticker):
    """Print summary of where edge exists"""
    print(f"\n{'='*60}")
    print(f"EDGE SUMMARY: {series_ticker}")
    print(f"{'='*60}")

    # Find actionable edges
    yes_edges = [(p, d) for p, d in edge_data.items() if d['yes_ev_per_dollar'] > 0.02 and d['count'] >= 10]
    no_edges = [(p, d) for p, d in edge_data.items() if d['no_ev_per_dollar'] > 0.02 and d['count'] >= 10]

    if yes_edges:
        print("\nBUY YES opportunities (>2% EV):")
        for price, data in sorted(yes_edges, key=lambda x: -x[1]['yes_ev_per_dollar']):
            print(f"  YES @ ${price:.2f}: {data['yes_ev_per_dollar']*100:+.1f}% EV "
                  f"(actual {data['actual_win_rate']*100:.0f}% vs implied {price*100:.0f}%, n={data['count']})")

    if no_edges:
        print("\nBUY NO opportunities (>2% EV):")
        for price, data in sorted(no_edges, key=lambda x: -x[1]['no_ev_per_dollar']):
            print(f"  NO @ ${1-price:.2f} (YES @ ${price:.2f}): {data['no_ev_per_dollar']*100:+.1f}% EV "
                  f"(YES wins {data['actual_win_rate']*100:.0f}% vs implied {price*100:.0f}%, n={data['count']})")

    if not yes_edges and not no_edges:
        print("\nNo significant edge found (>2% EV with n>=10)")

    # Total opportunity
    total_ev = sum(d['best_ev'] * d['count'] for d in edge_data.values() if d['best_ev'] > 0)
    total_trades = sum(d['count'] for d in edge_data.values() if d['best_ev'] > 0)

    if total_trades > 0:
        print(f"\nTotal theoretical edge: ${total_ev:.2f} across {total_trades} trades")
        print(f"Average edge per trade: ${total_ev/total_trades:.2f}")

    print(f"{'='*60}\n")


# ============== MAIN ==============
def main():
    parser = argparse.ArgumentParser(
        description='Kalshi Edge Finder - Discover mispricing in recurring markets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kalshi_edge_finder.py --list-series
  python kalshi_edge_finder.py --download KXBTC15M
  python kalshi_edge_finder.py --download-candles KXBTC15M
  python kalshi_edge_finder.py --analyze KXBTC15M
  python kalshi_edge_finder.py --analyze-time KXBTC15M
  python kalshi_edge_finder.py --analyze-surface KXBTC15M

Full workflow:
  1. python kalshi_edge_finder.py --download KXBTC15M
  2. python kalshi_edge_finder.py --download-candles KXBTC15M
  3. python kalshi_edge_finder.py --analyze-time KXBTC15M
  4. python kalshi_edge_finder.py --analyze-surface KXBTC15M
        """
    )

    parser.add_argument('--list-series', action='store_true',
                        help='List all available series on Kalshi')
    parser.add_argument('--download', metavar='SERIES',
                        help='Download historical market data for a series')
    parser.add_argument('--update', metavar='SERIES',
                        help='Update existing market data with new markets')
    parser.add_argument('--download-candles', metavar='SERIES',
                        help='Download candlestick data for a series (requires --download first)')
    parser.add_argument('--update-candles', metavar='SERIES',
                        help='Update existing candlestick data with new markets')
    parser.add_argument('--analyze', metavar='SERIES', nargs='?', const='__FROM_DOWNLOAD__',
                        help='Analyze a series for edge (uses final settlement prices)')
    parser.add_argument('--analyze-time', metavar='SERIES',
                        help='Analyze edge at different times before settlement (requires candles)')
    parser.add_argument('--analyze-surface', metavar='SERIES',
                        help='Analyze fair value surface: win rate by (distance, time) (requires candles)')
    parser.add_argument('--output', metavar='DIR',
                        help='Output directory for charts')

    args = parser.parse_args()

    if not any([args.list_series, args.download, args.update, args.download_candles,
                args.update_candles, args.analyze, args.analyze_time, args.analyze_surface]):
        parser.print_help()
        return

    # List series
    if args.list_series:
        series = list_series()
        print(f"\nFound {len(series)} series:")
        for s in series:
            info = get_series_info(s)
            if info:
                print(f"  {s}: {info.get('title', 'N/A')[:60]}")
            else:
                print(f"  {s}")
        return

    # Download market data
    series_to_analyze = None
    if args.download:
        download_series_data(args.download)
        series_to_analyze = args.download

    if args.update:
        download_series_data(args.update, update_only=True)
        series_to_analyze = args.update

    # Download candlestick data
    if args.download_candles:
        download_series_candles(args.download_candles)
        series_to_analyze = args.download_candles

    if args.update_candles:
        download_series_candles(args.update_candles, update_only=True)
        series_to_analyze = args.update_candles

    # Basic analysis (final prices)
    if args.analyze:
        series = args.analyze if args.analyze != '__FROM_DOWNLOAD__' else series_to_analyze
        if not series:
            print("Error: specify series to analyze")
            return

        markets = load_series_data(series)
        if not markets:
            return

        print(f"\nAnalyzing {len(markets)} markets (final prices)...")

        calibration = calculate_calibration(markets, price_buckets=20)
        edge = calculate_edge(calibration)

        output_dir = args.output or CHARTS_DIR
        plot_calibration(calibration, series, output_dir)
        plot_edge(edge, series, output_dir)
        print_edge_summary(edge, series)

    # Time-sliced analysis (requires candles)
    if args.analyze_time:
        series = args.analyze_time
        markets = load_series_candles(series)
        if not markets:
            return

        print(f"\nAnalyzing {len(markets)} markets with candlestick data...")
        print("Computing calibration at T-5, T-3, T-2, T-1 minutes...")

        time_results = analyze_time_slices(markets, time_slices=[5, 3, 2, 1])

        output_dir = args.output or CHARTS_DIR
        plot_time_calibration(time_results, series, output_dir)
        plot_time_edge(time_results, series, output_dir)
        print_time_edge_summary(time_results, series)

    # Fair value surface analysis (requires candles)
    if args.analyze_surface:
        if np is None:
            print("Error: numpy required for surface analysis. Run: pip install numpy")
            return

        series = args.analyze_surface
        markets = load_series_candles(series)
        if not markets:
            return

        print(f"\nAnalyzing fair value surface for {len(markets)} markets...")
        print("Computing leader win rate by (distance, time)...")

        surface_data = calculate_surface_calibration(markets, time_slices=[13, 10, 7, 5, 3, 2])

        output_dir = args.output or CHARTS_DIR
        plot_fair_value_surface(surface_data, series, output_dir)
        plot_surface_heatmap(surface_data, series, output_dir)
        print_surface_summary(surface_data, series)


if __name__ == "__main__":
    main()
