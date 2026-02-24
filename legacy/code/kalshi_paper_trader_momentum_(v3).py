# kalshi_paper_trader_momentum_(v3).py
# MOMENTUM STRATEGY: Trades when YES price rises from low to ~50% zone
# Based on finding that markets rising TO ~50% have YES win rate of only 7.5%

import os
import requests
import base64
import time
import json
import math
import argparse
from datetime import datetime, timezone, timedelta
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

# Optional: matplotlib for plots (graceful fallback if not installed)
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============== CONFIG ==============
STARTING_BALANCE = 1000
BET_SIZE = 100
RUN_DURATION_HOURS = 8  # How long to run
TRADES_FILE = "paper_trades_momentum.json"  # Separate file for momentum strategy
BACKTEST_DATA_FILE = "kalshi_backtest_data.json"

# ============== AUTH ==============
# Load credentials - adjust paths as needed
API_KEY = os.getenv("KALSHI_API_KEY")
if not API_KEY:
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("KALSHI_API_KEY=") or line.startswith("API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()
                break

with open("kalshi_secret.pem", "r") as f:
    secret_key_str = f.read().strip()

if not secret_key_str.startswith("-----"):
    secret_key_str = f"-----BEGIN RSA PRIVATE KEY-----\n{secret_key_str}\n-----END RSA PRIVATE KEY-----"

PRIVATE_KEY = serialization.load_pem_private_key(
    secret_key_str.encode(),
    password=None
)

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

def api_get(path: str, params: dict = {}):
    headers = get_headers("GET", path)
    resp = requests.get(BASE_URL + path, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()


# ============== DATA DOWNLOAD ==============
def get_all_settled_markets():
    """Get all settled KXBTC15M markets"""
    all_markets = []
    cursor = None

    print("Fetching settled markets...")
    while True:
        params = {
            "series_ticker": "KXBTC15M",
            "status": "settled",
            "limit": 200
        }
        if cursor:
            params["cursor"] = cursor

        data = api_get("/trade-api/v2/markets", params)
        markets = data.get("markets", [])
        all_markets.extend(markets)

        cursor = data.get("cursor")
        print(f"  Fetched {len(all_markets)} markets...")

        if not cursor or not markets:
            break

        time.sleep(0.1)

    print(f"Total settled markets: {len(all_markets)}")
    return all_markets


def get_candlesticks(ticker: str, start_ts: int, end_ts: int):
    """Get 1-minute candlesticks for a market"""
    params = {
        "series_ticker": ticker,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": 1  # 1 minute candles
    }
    try:
        data = api_get(f"/trade-api/v2/series/KXBTC15M/markets/{ticker}/candlesticks", params)
        return data.get("candlesticks", [])
    except:
        # Try alternative endpoint
        try:
            data = api_get(f"/trade-api/v2/markets/{ticker}/candlesticks", params)
            return data.get("candlesticks", [])
        except Exception as e:
            print(f"Error getting candlesticks for {ticker}: {e}")
            return []


def download_backtest_data(update_only=False):
    """Download historical market data for backtesting

    Args:
        update_only: If True, only download new markets not in existing file
    """
    existing_tickers = set()
    existing_data = []

    # Load existing data if updating
    if update_only:
        try:
            with open(BACKTEST_DATA_FILE, "r") as f:
                existing_data = json.load(f)
                existing_tickers = {m['ticker'] for m in existing_data}
                print(f"Loaded {len(existing_data)} existing markets from {BACKTEST_DATA_FILE}")
        except FileNotFoundError:
            print(f"No existing {BACKTEST_DATA_FILE} found, downloading all data")
            update_only = False

    markets = get_all_settled_markets()

    # Filter to new markets only if updating
    if update_only:
        markets = [m for m in markets if m['ticker'] not in existing_tickers]
        print(f"Found {len(markets)} new markets to download")

        if not markets:
            print("No new markets to download!")
            return existing_data

    new_data = []
    for i, market in enumerate(markets):
        ticker = market['ticker']
        result = market.get('result')

        # Skip markets without results
        if result not in ['yes', 'no']:
            continue

        try:
            close_time = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))
            end_ts = int(close_time.timestamp())
            start_ts = end_ts - (20 * 60)  # 20 min before close

            candles = get_candlesticks(ticker, start_ts, end_ts)

            if candles:
                new_data.append({
                    'ticker': ticker,
                    'strike': market.get('floor_strike'),
                    'result': result,
                    'close_time': market.get('close_time'),
                    'yes_bid': market.get('yes_bid'),
                    'yes_ask': market.get('yes_ask'),
                    'no_bid': market.get('no_bid'),
                    'no_ask': market.get('no_ask'),
                    'candlesticks': candles
                })

            time.sleep(0.1)  # Rate limit

        except Exception as e:
            print(f"Error on {ticker}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(markets)} markets, collected {len(new_data)} with data")

    # Combine existing and new data
    all_data = existing_data + new_data

    # Save to file
    with open(BACKTEST_DATA_FILE, "w") as f:
        json.dump(all_data, f)

    if update_only:
        print(f"\nDone! Added {len(new_data)} new markets (total: {len(all_data)}) to {BACKTEST_DATA_FILE}")
    else:
        print(f"\nDone! Saved {len(all_data)} markets to {BACKTEST_DATA_FILE}")

    return all_data


# ============== BOT ==============
class PaperTradingBot:
    def __init__(self, starting_balance=1000, bet_size=100):
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.bet_size = bet_size
        self.trades = []
        self.pending_trades = {}
        self.seen_markets = set()

        # MOMENTUM STRATEGY ZONES
        # Based on discovery: when YES rises TO ~50% from low prices, YES wins only 7.5%
        # So we BUY NO when we detect upward momentum into the 48-55 cent zone
        self.momentum_zone = (0.48, 0.55)       # Zone where momentum edge exists
        self.momentum_start_max = 0.30          # Must have started BELOW this
        self.momentum_rise_min = 0.15           # Must have risen at least 15 cents
        self.lookback_minutes = 4               # How far back to look for starting price

        # Legacy zones (disabled for momentum strategy)
        self.buy_no_zone = (0.0, 0.0)           # DISABLED - using momentum instead
        self.buy_yes_zone = (0.0, 0.0)          # DISABLED

        self.min_time_remaining = 0.5   # 30 seconds - avoid liquidity desert
        self.max_time_remaining = 1.5   # Enter from T-1.5 to T-0.5
        self.max_spread = 0.05          # 5 cent max spread

        # Track price history for momentum detection
        self.price_history = {}  # ticker -> list of (timestamp, yes_mid)

    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def track_price(self, market):
        """Track price history for momentum detection"""
        ticker = market['ticker']
        yes_bid = market.get('yes_bid', 0) / 100
        yes_ask = market.get('yes_ask', 0) / 100

        if yes_bid <= 0 or yes_ask <= 0:
            return

        yes_mid = (yes_bid + yes_ask) / 2
        now = datetime.now(timezone.utc)

        if ticker not in self.price_history:
            self.price_history[ticker] = []

        self.price_history[ticker].append((now, yes_mid))

        # Keep only last 10 minutes of data to avoid memory bloat
        cutoff = now.timestamp() - (10 * 60)
        self.price_history[ticker] = [
            (ts, price) for ts, price in self.price_history[ticker]
            if ts.timestamp() > cutoff
        ]

    def get_price_at_lookback(self, ticker, lookback_minutes):
        """Get the price from lookback_minutes ago (approximately)"""
        if ticker not in self.price_history:
            return None

        history = self.price_history[ticker]
        if not history:
            return None

        now = datetime.now(timezone.utc)
        target_time = now.timestamp() - (lookback_minutes * 60)

        # Find the price closest to our target time
        best = None
        best_diff = float('inf')

        for ts, price in history:
            diff = abs(ts.timestamp() - target_time)
            if diff < best_diff:
                best_diff = diff
                best = (ts, price)

        # Only return if we have a price within 1 minute of our target
        if best and best_diff < 60:
            return best[1]

        return None

    def check_momentum(self, ticker, current_price):
        """Check if market shows upward momentum into the momentum zone

        Returns:
            'BUY_NO' if upward momentum detected (price rose from below 0.30 to ~0.50)
            None if no momentum signal
        """
        # Check if current price is in the momentum zone
        if not (self.momentum_zone[0] <= current_price <= self.momentum_zone[1]):
            return None

        # Get historical price
        start_price = self.get_price_at_lookback(ticker, self.lookback_minutes)
        if start_price is None:
            return None

        # Check conditions for upward momentum:
        # 1. Started below momentum_start_max (e.g., < $0.30)
        # 2. Rose at least momentum_rise_min (e.g., 15 cents)
        if start_price <= self.momentum_start_max:
            price_rise = current_price - start_price
            if price_rise >= self.momentum_rise_min:
                self.log(f"📈 MOMENTUM DETECTED: {ticker} rose from ${start_price:.2f} → ${current_price:.2f} (+${price_rise:.2f})")
                return 'BUY_NO'

        return None

    def get_active_markets(self):
        params = {
            "series_ticker": "KXBTC15M",
            "status": "open",
            "limit": 10
        }
        try:
            data = api_get("/trade-api/v2/markets", params)
            return data.get("markets", [])
        except Exception as e:
            self.log(f"Error fetching markets: {e}")
            return []

    def get_market_details(self, ticker):
        try:
            data = api_get(f"/trade-api/v2/markets/{ticker}")
            return data.get("market", {})
        except Exception as e:
            self.log(f"Error fetching {ticker}: {e}")
            return None

    def calculate_time_remaining(self, market):
        close_time = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        remaining = (close_time - now).total_seconds() / 60
        return remaining

    def check_for_opportunity(self, market):
        ticker = market['ticker']

        # Always track price, even if not in trading window (for momentum detection)
        self.track_price(market)

        if ticker in self.seen_markets:
            return None

        time_remaining = self.calculate_time_remaining(market)

        if not (self.min_time_remaining <= time_remaining <= self.max_time_remaining):
            return None

        yes_bid = market.get('yes_bid', 0) / 100
        yes_ask = market.get('yes_ask', 0) / 100

        if yes_bid == 0 or yes_ask == 0:
            return None

        spread = yes_ask - yes_bid
        yes_mid = (yes_bid + yes_ask) / 2

        # MOMENTUM STRATEGY: Check for upward momentum into the ~50% zone
        momentum_signal = self.check_momentum(ticker, yes_mid)

        if momentum_signal:
            # Check spread before trading
            if spread > self.max_spread:
                self.log(f"⏭️  SKIPPED (momentum): {ticker} - spread ${spread:.2f} > max ${self.max_spread:.2f} (YES mid: ${yes_mid:.2f}, T-{time_remaining:.1f}min)")
                return None

            # Check NO liquidity
            no_ask = market.get('no_ask', 0) / 100
            if no_ask == 0:
                self.log(f"⏭️  SKIPPED (momentum): {ticker} - no NO liquidity (YES mid: ${yes_mid:.2f}, T-{time_remaining:.1f}min)")
                return None

            # Get starting price for logging
            start_price = self.get_price_at_lookback(ticker, self.lookback_minutes)

            return {
                'action': 'BUY_NO',
                'price': no_ask,
                'yes_mid': yes_mid,
                'spread': spread,
                'time_remaining': time_remaining,
                'start_price': start_price,  # For logging
                'signal': 'momentum_up'
            }

        # LEGACY ZONES (disabled by default, but kept for fallback)
        in_no_zone = self.buy_no_zone[0] <= yes_mid <= self.buy_no_zone[1]
        in_yes_zone = self.buy_yes_zone[0] <= yes_mid <= self.buy_yes_zone[1]

        if not (in_no_zone or in_yes_zone):
            return None

        # Now check spread - only log if we would have traded
        if spread > self.max_spread:
            zone = "NO zone" if in_no_zone else "YES zone"
            self.log(f"⏭️  SKIPPED ({zone}): {ticker} - spread ${spread:.2f} > max ${self.max_spread:.2f} (YES mid: ${yes_mid:.2f}, T-{time_remaining:.1f}min)")
            return None

        if in_no_zone:
            no_ask = market.get('no_ask', 0) / 100
            if no_ask == 0:  # No liquidity on NO side
                self.log(f"⏭️  SKIPPED (NO zone): {ticker} - no NO liquidity (YES mid: ${yes_mid:.2f}, T-{time_remaining:.1f}min)")
                return None
            return {
                'action': 'BUY_NO',
                'price': no_ask,
                'yes_mid': yes_mid,
                'spread': spread,
                'time_remaining': time_remaining,
                'signal': 'zone'
            }

        if in_yes_zone:
            return {
                'action': 'BUY_YES',
                'price': yes_ask,
                'yes_mid': yes_mid,
                'spread': spread,
                'time_remaining': time_remaining,
                'signal': 'zone'
            }

        return None

    def place_paper_trade(self, market, opportunity):
        ticker = market['ticker']

        trade = {
            'ticker': ticker,
            'action': opportunity['action'],
            'price': opportunity['price'],
            'yes_mid': opportunity['yes_mid'],
            'spread': opportunity['spread'],
            'bet_size': self.bet_size,
            'time_remaining': opportunity['time_remaining'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'close_time': market['close_time'],
            'strike': market.get('floor_strike'),
            'status': 'pending',
            'signal': opportunity.get('signal', 'unknown'),
            'start_price': opportunity.get('start_price')
        }

        self.pending_trades[ticker] = trade
        self.seen_markets.add(ticker)

        # Log with momentum info if applicable
        if opportunity.get('signal') == 'momentum_up' and opportunity.get('start_price'):
            start_price = opportunity['start_price']
            self.log(f"📝 PAPER TRADE (MOMENTUM): {opportunity['action']} @ ${opportunity['price']:.2f} "
                    f"(YES rose: ${start_price:.2f} → ${opportunity['yes_mid']:.2f}, "
                    f"spread: ${opportunity['spread']:.2f}, T-{opportunity['time_remaining']:.1f}min) "
                    f"on {ticker}")
        else:
            self.log(f"📝 PAPER TRADE: {opportunity['action']} @ ${opportunity['price']:.2f} "
                    f"(YES mid: ${opportunity['yes_mid']:.2f}, spread: ${opportunity['spread']:.2f}, T-{opportunity['time_remaining']:.1f}min) "
                    f"on {ticker}")

        return trade

    def check_settled_trades(self):
        settled_tickers = []

        for ticker, trade in self.pending_trades.items():
            try:
                market = self.get_market_details(ticker)
                if market:
                    result = market.get('result')
                    # Only resolve if result is actually populated
                    if result in ['yes', 'no']:
                        self.resolve_trade(trade, result)
                        settled_tickers.append(ticker)
                    # else: keep waiting - market may be closed but not yet settled
            except Exception as e:
                self.log(f"Error checking {ticker}: {e}")

        for ticker in settled_tickers:
            del self.pending_trades[ticker]

    def calculate_fees(self, contracts, price):
        """Kalshi fee formula: round_up(0.07 * C * P * (1-P))"""
        raw_fee = 0.07 * contracts * price * (1 - price)
        return math.ceil(raw_fee * 100) / 100  # Round up to nearest cent

    def resolve_trade(self, trade, result):
        action = trade['action']
        price = trade['price']
        bet_size = trade['bet_size']

        if action == 'BUY_YES':
            won = (result == 'yes')
        else:
            won = (result == 'no')

        if won:
            pnl = bet_size * (1 - price)
        else:
            pnl = -bet_size * price

        # Kalshi fees: 0.07 * C * P * (1-P), rounded up to nearest cent
        contracts = bet_size  # bet_size is number of contracts
        fees = self.calculate_fees(contracts, price)
        pnl -= fees

        self.balance += pnl

        trade['result'] = result
        trade['won'] = won
        trade['pnl'] = pnl
        trade['fees'] = fees
        trade['status'] = 'settled'
        self.trades.append(trade)

        emoji = "✅" if won else "❌"
        self.log(f"{emoji} SETTLED: {trade['action']} on {trade['ticker']} - "
                f"Result: {result}, PnL: ${pnl:+.2f} (fees: ${fees:.2f}), Balance: ${self.balance:.2f}")

    def print_stats(self):
        completed = [t for t in self.trades if t['status'] == 'settled']
        if not completed:
            self.log("No completed trades yet")
            return

        wins = sum(1 for t in completed if t['won'])
        total_pnl = sum(t['pnl'] for t in completed)

        self.log(f"\n📊 STATS: {len(completed)} trades, {wins} wins ({wins/len(completed)*100:.1f}%), "
                f"Total PnL: ${total_pnl:+.2f}, Balance: ${self.balance:.2f}")

    def save_results(self):
        with open(TRADES_FILE, "w") as f:
            json.dump({
                'starting_balance': self.starting_balance,
                'final_balance': self.balance,
                'trades': self.trades,
                'pending_trades': self.pending_trades  # Full trade objects, not just keys
            }, f, indent=2)
        self.log(f"💾 Saved results to {TRADES_FILE}")

    def load_from_file(self):
        """Load previous trades and balance from paper_trades.json"""
        try:
            with open(TRADES_FILE, "r") as f:
                data = json.load(f)

            self.trades = data.get('trades', [])
            self.balance = data.get('final_balance', self.starting_balance)

            # Add all previous tickers to seen_markets to avoid re-trading
            for trade in self.trades:
                self.seen_markets.add(trade['ticker'])

            # Restore pending trades (full trade objects)
            self.pending_trades = data.get('pending_trades', {})
            # Also add pending tickers to seen_markets
            for ticker in self.pending_trades:
                self.seen_markets.add(ticker)

            self.log(f"📂 Loaded {len(self.trades)} trades from {TRADES_FILE}")
            self.log(f"   Balance: ${self.balance:.2f}")
            self.log(f"   Seen markets: {len(self.seen_markets)}")
            if self.pending_trades:
                self.log(f"   ⏳ {len(self.pending_trades)} pending trades restored (will check for settlement)")

            return True
        except FileNotFoundError:
            self.log(f"⚠️  No {TRADES_FILE} found, starting fresh")
            return False
        except Exception as e:
            self.log(f"❌ Error loading {TRADES_FILE}: {e}")
            return False

    def backtest(self, data_file=BACKTEST_DATA_FILE, period_days=7):
        """Run backtest on historical data using MOMENTUM strategy

        Args:
            data_file: Path to backtest data JSON
            period_days: Size of rolling periods for analysis (default 7 days)
        """
        self.log(f"📊 Running MOMENTUM backtest on {data_file}...")

        try:
            with open(data_file, "r") as f:
                markets = json.load(f)
        except FileNotFoundError:
            self.log(f"❌ No {data_file} found. Run with --download-data first.")
            return

        self.log(f"   Loaded {len(markets)} markets")
        self.log(f"   MOMENTUM STRATEGY:")
        self.log(f"     Zone: YES @ ${self.momentum_zone[0]:.2f}-${self.momentum_zone[1]:.2f}")
        self.log(f"     Start price max: ${self.momentum_start_max:.2f}")
        self.log(f"     Min rise required: ${self.momentum_rise_min:.2f}")
        self.log(f"     Lookback: {self.lookback_minutes} minutes")
        self.log(f"   Time window: T-{self.max_time_remaining} to T-{self.min_time_remaining} minutes")
        self.log(f"   Max spread: ${self.max_spread:.2f}")
        self.log("")

        trades = []
        skipped_spread = 0
        skipped_spread_would_lose = 0
        skipped_no_liquidity = 0
        skipped_no_candles = 0
        skipped_no_momentum = 0

        def get_candle_price(candle):
            """Extract yes_mid price from candle data"""
            yes_bid_data = candle.get('yes_bid') or {}
            yes_ask_data = candle.get('yes_ask') or {}

            if isinstance(yes_bid_data, dict):
                yes_bid = yes_bid_data.get('close') or 0
            else:
                yes_bid = yes_bid_data or 0

            if isinstance(yes_ask_data, dict):
                yes_ask = yes_ask_data.get('close') or 0
            else:
                yes_ask = yes_ask_data or 0

            yes_bid = yes_bid or 0
            yes_ask = yes_ask or 0

            if yes_bid > 1:
                yes_bid = yes_bid / 100
            if yes_ask > 1:
                yes_ask = yes_ask / 100

            if yes_bid <= 0 or yes_ask <= 0:
                return None, None, None

            spread = yes_ask - yes_bid
            yes_mid = (yes_bid + yes_ask) / 2
            return yes_mid, yes_bid, spread

        for market in markets:
            ticker = market['ticker']
            result = market['result']
            candles = market.get('candlesticks', [])

            if not candles:
                skipped_no_candles += 1
                continue

            close_time = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))

            t1_candle = None
            t1_time_remaining = None
            t5_candle = None
            t5_time_remaining = None

            for candle in candles:
                end_ts = candle.get('end_period_ts') or candle.get('end_ts') or candle.get('timestamp')
                if isinstance(end_ts, int):
                    candle_time = datetime.fromtimestamp(end_ts, tz=timezone.utc)
                else:
                    candle_time = datetime.fromisoformat(str(end_ts).replace('Z', '+00:00'))
                time_remaining = (close_time - candle_time).total_seconds() / 60

                if self.min_time_remaining <= time_remaining <= self.max_time_remaining:
                    t1_candle = candle
                    t1_time_remaining = time_remaining

                if self.lookback_minutes <= time_remaining <= self.lookback_minutes + 2:
                    t5_candle = candle
                    t5_time_remaining = time_remaining

            if not t1_candle or not t5_candle:
                skipped_no_candles += 1
                continue

            t1_price, t1_bid, t1_spread = get_candle_price(t1_candle)
            t5_price, _, _ = get_candle_price(t5_candle)

            if t1_price is None or t5_price is None:
                skipped_no_liquidity += 1
                continue

            if not (self.momentum_zone[0] <= t1_price <= self.momentum_zone[1]):
                skipped_no_momentum += 1
                continue

            if t5_price > self.momentum_start_max:
                skipped_no_momentum += 1
                continue

            price_rise = t1_price - t5_price
            if price_rise < self.momentum_rise_min:
                skipped_no_momentum += 1
                continue

            if t1_spread is not None and t1_spread > self.max_spread:
                skipped_spread += 1
                if result == 'yes':
                    skipped_spread_would_lose += 1
                continue

            action = 'BUY_NO'
            price = 1 - t1_bid

            won = (result == 'no')

            if won:
                pnl = self.bet_size * (1 - price)
            else:
                pnl = -self.bet_size * price

            fees = self.calculate_fees(self.bet_size, price)
            pnl -= fees

            trade = {
                'ticker': ticker,
                'action': action,
                'price': price,
                'yes_mid': t1_price,
                'start_price': t5_price,
                'price_rise': price_rise,
                'spread': t1_spread,
                'time_remaining': t1_time_remaining,
                'close_time': close_time,
                'result': result,
                'won': won,
                'pnl': pnl,
                'fees': fees,
                'signal': 'momentum_up'
            }
            trades.append(trade)

        # Sort trades by time
        trades.sort(key=lambda t: t['close_time'])

        # Calculate running balance and drawdown
        balance = self.starting_balance
        peak_balance = balance
        max_drawdown = 0
        max_drawdown_pct = 0

        balance_history = [(trades[0]['close_time'] - timedelta(hours=1), balance)] if trades else []

        for trade in trades:
            balance += trade['pnl']
            trade['balance'] = balance
            balance_history.append((trade['close_time'], balance))

            if balance > peak_balance:
                peak_balance = balance

            drawdown = peak_balance - balance
            drawdown_pct = (drawdown / peak_balance * 100) if peak_balance > 0 else 0

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

        self.balance = balance
        self.trades = trades

        # Calculate overall stats
        wins = sum(1 for t in trades if t['won'])
        losses = len(trades) - wins
        total_pnl = sum(t['pnl'] for t in trades)

        if trades:
            returns = [t['pnl'] for t in trades]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_return = variance ** 0.5 if variance > 0 else 0
            sharpe = mean_return / std_return if std_return > 0 else 0
        else:
            mean_return = std_return = sharpe = 0

        # Print results
        self.log("=" * 60)
        self.log("MOMENTUM BACKTEST RESULTS")
        self.log("=" * 60)
        self.log(f"Total markets analyzed: {len(markets)}")
        self.log(f"  Skipped (no momentum signal): {skipped_no_momentum}")
        self.log(f"  Skipped (spread > max): {skipped_spread} ({skipped_spread_would_lose} would have LOST)")
        self.log(f"  Skipped (no liquidity): {skipped_no_liquidity}")
        self.log(f"  Skipped (no candle data): {skipped_no_candles}")
        self.log(f"  Trades executed: {len(trades)}")
        self.log("")

        if trades:
            date_range = f"{trades[0]['close_time'].strftime('%Y-%m-%d')} to {trades[-1]['close_time'].strftime('%Y-%m-%d')}"
            days_span = (trades[-1]['close_time'] - trades[0]['close_time']).days + 1
            self.log(f"Period: {date_range} ({days_span} days)")
            self.log(f"Trades/day: {len(trades)/days_span:.1f}")
            self.log("")

            self.log(f"OVERALL: {wins}W / {losses}L ({100*wins/len(trades):.1f}% win rate)")
            self.log(f"  Total PnL: ${total_pnl:+.2f}")
            self.log(f"  Mean PnL per trade: ${mean_return:+.2f}")
            self.log(f"  Std dev: ${std_return:.2f}")
            self.log(f"  Sharpe ratio: {sharpe:.3f}")
            self.log(f"  Final balance: ${self.balance:.2f}")
            self.log(f"  Max drawdown: ${max_drawdown:.2f} ({max_drawdown_pct:.1f}%)")

            # Without spread filter stats
            total_with_spread = len(trades) + skipped_spread
            losses_with_spread = losses + skipped_spread_would_lose
            wins_with_spread = total_with_spread - losses_with_spread
            if total_with_spread > 0:
                wr_no_filter = 100 * wins_with_spread / total_with_spread
                self.log("")
                self.log(f"WITHOUT spread filter: {wins_with_spread}W / {losses_with_spread}L ({wr_no_filter:.1f}% win rate)")

            # Rolling period analysis
            self.log("")
            self.log(f"ROLLING {period_days}-DAY PERIODS:")
            self.log("-" * 60)

            if trades:
                start_date = trades[0]['close_time'].date()
                end_date = trades[-1]['close_time'].date()
                current_start = start_date

                while current_start <= end_date:
                    current_end = current_start + timedelta(days=period_days)
                    period_trades = [t for t in trades
                                   if current_start <= t['close_time'].date() < current_end]

                    if period_trades:
                        p_wins = sum(1 for t in period_trades if t['won'])
                        p_losses = len(period_trades) - p_wins
                        p_pnl = sum(t['pnl'] for t in period_trades)
                        p_wr = 100 * p_wins / len(period_trades) if period_trades else 0

                        # Period drawdown
                        p_balance = self.starting_balance
                        p_peak = p_balance
                        p_max_dd = 0
                        for t in period_trades:
                            p_balance += t['pnl']
                            if p_balance > p_peak:
                                p_peak = p_balance
                            dd = p_peak - p_balance
                            if dd > p_max_dd:
                                p_max_dd = dd

                        self.log(f"  {current_start} to {current_end - timedelta(days=1)}: "
                                f"{p_wins}W/{p_losses}L ({p_wr:.0f}%) | "
                                f"PnL: ${p_pnl:+.0f} | MaxDD: ${p_max_dd:.0f}")

                    current_start = current_end

            # Show losing trades
            losing_trades = [t for t in trades if not t['won']]
            if losing_trades:
                self.log("")
                self.log(f"LOSING TRADES ({len(losing_trades)}):")
                for t in losing_trades:
                    self.log(f"  ❌ {t['ticker']}: ${t['start_price']:.2f} → ${t['yes_mid']:.2f} → {t['result']}, PnL: ${t['pnl']:+.2f}")

            # Generate plot
            if HAS_MATPLOTLIB and balance_history:
                self.log("")
                self.log("Generating performance plot...")

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

                # Balance over time
                times = [t[0] for t in balance_history]
                balances = [t[1] for t in balance_history]

                ax1.plot(times, balances, 'b-', linewidth=1.5)
                ax1.axhline(y=self.starting_balance, color='gray', linestyle='--', alpha=0.5)
                ax1.fill_between(times, self.starting_balance, balances,
                               where=[b >= self.starting_balance for b in balances],
                               color='green', alpha=0.3)
                ax1.fill_between(times, self.starting_balance, balances,
                               where=[b < self.starting_balance for b in balances],
                               color='red', alpha=0.3)
                ax1.set_ylabel('Balance ($)')
                ax1.set_title(f'Momentum Strategy Backtest: ${self.starting_balance} → ${balance:.0f} ({total_pnl:+.0f})')
                ax1.grid(True, alpha=0.3)

                # Drawdown over time
                peak = self.starting_balance
                drawdowns = []
                for b in balances:
                    if b > peak:
                        peak = b
                    dd_pct = (peak - b) / peak * 100 if peak > 0 else 0
                    drawdowns.append(dd_pct)

                ax2.fill_between(times, 0, drawdowns, color='red', alpha=0.5)
                ax2.set_ylabel('Drawdown (%)')
                ax2.set_xlabel('Date')
                ax2.set_ylim(max(drawdowns) * 1.1 if drawdowns else 10, 0)
                ax2.grid(True, alpha=0.3)

                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                plt.xticks(rotation=45)

                plt.tight_layout()
                plot_file = 'momentum_backtest_performance.png'
                plt.savefig(plot_file, dpi=150)
                plt.close()
                self.log(f"📈 Saved plot to {plot_file}")
            elif not HAS_MATPLOTLIB:
                self.log("")
                self.log("(Install matplotlib for performance plots: pip install matplotlib)")

        else:
            self.log("No trades matched momentum criteria")

        self.log("=" * 60)

        return trades

    def recheck_trades(self):
        """Re-fetch results for all trades and recalculate balance"""
        self.log("🔄 Rechecking all trades against Kalshi API...")

        try:
            with open(TRADES_FILE, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            self.log(f"❌ No {TRADES_FILE} found")
            return

        trades = data.get('trades', [])
        pending_trades = data.get('pending_trades', {})

        if not trades and not pending_trades:
            self.log("No trades to recheck")
            return

        self.balance = data.get('starting_balance', STARTING_BALANCE)
        self.trades = []
        self.pending_trades = {}

        # First, resolve any pending trades
        if pending_trades:
            self.log(f"   Checking {len(pending_trades)} pending trades...")
            for ticker, trade in pending_trades.items():
                market = self.get_market_details(ticker)
                if not market:
                    self.log(f"⚠️  Could not fetch {ticker}, keeping as pending")
                    self.pending_trades[ticker] = trade
                    continue

                result = market.get('result')
                if result in ['yes', 'no']:
                    # Resolve it
                    action = trade['action']
                    price = trade['price']
                    bet_size = trade.get('bet_size', BET_SIZE)

                    if action == 'BUY_YES':
                        won = (result == 'yes')
                    else:
                        won = (result == 'no')

                    if won:
                        pnl = bet_size * (1 - price)
                    else:
                        pnl = -bet_size * price

                    fees = self.calculate_fees(bet_size, price)
                    pnl -= fees
                    self.balance += pnl

                    trade['result'] = result
                    trade['won'] = won
                    trade['pnl'] = pnl
                    trade['fees'] = fees
                    trade['status'] = 'settled'
                    self.trades.append(trade)

                    emoji = "✅" if won else "❌"
                    self.log(f"{emoji} RESOLVED PENDING: {ticker}: {action} @ ${price:.2f} → {result}, PnL: ${pnl:+.2f}")
                else:
                    self.log(f"⚠️  {ticker} still has no result, keeping as pending")
                    self.pending_trades[ticker] = trade

        # Then recheck settled trades
        for trade in trades:
            ticker = trade['ticker']
            action = trade['action']
            price = trade['price']
            bet_size = trade.get('bet_size', BET_SIZE)

            # Fetch current market data
            market = self.get_market_details(ticker)
            if not market:
                self.log(f"⚠️  Could not fetch {ticker}, skipping")
                continue

            result = market.get('result')
            if result not in ['yes', 'no']:
                self.log(f"⚠️  {ticker} still has no result: {result}")
                continue

            # Recalculate PnL
            if action == 'BUY_YES':
                won = (result == 'yes')
            else:
                won = (result == 'no')

            if won:
                pnl = bet_size * (1 - price)
            else:
                pnl = -bet_size * price

            fees = self.calculate_fees(bet_size, price)
            pnl -= fees

            self.balance += pnl

            # Update trade record
            trade['result'] = result
            trade['won'] = won
            trade['pnl'] = pnl
            trade['fees'] = fees
            trade['status'] = 'settled'
            self.trades.append(trade)

            old_result = data['trades'][trades.index(trade)].get('result', 'None')
            old_won = data['trades'][trades.index(trade)].get('won', False)

            emoji = "✅" if won else "❌"
            changed = " (CHANGED!)" if won != old_won else ""
            self.log(f"{emoji} {ticker}: {action} @ ${price:.2f} → {result}, PnL: ${pnl:+.2f}{changed}")

        self.log(f"\n📊 RECHECKED RESULTS:")
        self.print_stats()
        self.save_results()

    def run(self, duration_minutes=60):
        self.log(f"🚀 Starting MOMENTUM paper trading bot")
        self.log(f"   Balance: ${self.balance:.2f}")
        self.log(f"   Bet size: ${self.bet_size}")
        self.log(f"   MOMENTUM STRATEGY:")
        self.log(f"     Zone: YES @ ${self.momentum_zone[0]:.2f}-${self.momentum_zone[1]:.2f}")
        self.log(f"     Start price max: ${self.momentum_start_max:.2f}")
        self.log(f"     Min rise required: ${self.momentum_rise_min:.2f}")
        self.log(f"     Lookback: {self.lookback_minutes} minutes")
        self.log(f"   Time window: T-{self.max_time_remaining} to T-{self.min_time_remaining} minutes")
        self.log(f"   Max spread: ${self.max_spread:.2f}")
        self.log(f"   Running for {duration_minutes} minutes...\n")

        start_time = time.time()
        check_interval = 10

        try:
            while (time.time() - start_time) < duration_minutes * 60:
                markets = self.get_active_markets()

                for market in markets:
                    opportunity = self.check_for_opportunity(market)
                    if opportunity:
                        self.place_paper_trade(market, opportunity)

                self.check_settled_trades()
                time.sleep(check_interval)

        except KeyboardInterrupt:
            self.log("\n⏹️ Stopped by user")

        self.print_stats()
        self.save_results()


# ============== MAIN ==============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Kalshi Paper Trading Bot for KXBTC15M markets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kalshi_paper_trader.py                    Run paper trading for 8 hours
  python kalshi_paper_trader.py --hours 4          Run for 4 hours
  python kalshi_paper_trader.py --resume           Continue from previous session
  python kalshi_paper_trader.py --recheck          Fix results in paper_trades.json
  python kalshi_paper_trader.py --download-data    Download all historical data
  python kalshi_paper_trader.py --update-data      Download only new markets
  python kalshi_paper_trader.py --backtest         Run backtest on historical data
        """
    )

    # Paper trading args
    parser.add_argument('--resume', action='store_true',
                        help='Resume from paper_trades.json (load previous balance and seen markets)')
    parser.add_argument('--recheck', action='store_true',
                        help='Recheck all trades in paper_trades.json and fix results from Kalshi API')
    parser.add_argument('--hours', type=int, default=RUN_DURATION_HOURS,
                        help=f'Hours to run paper trading (default: {RUN_DURATION_HOURS})')

    # Backtest args
    parser.add_argument('--download-data', action='store_true',
                        help='Download ALL historical market data for backtesting (slow, ~10-15 min)')
    parser.add_argument('--update-data', action='store_true',
                        help='Download only NEW markets since last download (fast)')
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest on historical data in kalshi_backtest_data.json')

    args = parser.parse_args()

    # Download data mode (full)
    if args.download_data:
        download_backtest_data(update_only=False)
        exit(0)

    # Update data mode (diff only)
    if args.update_data:
        download_backtest_data(update_only=True)
        exit(0)

    bot = PaperTradingBot(starting_balance=STARTING_BALANCE, bet_size=BET_SIZE)

    # Backtest mode
    if args.backtest:
        bot.backtest()
        exit(0)

    # Recheck mode: fix results and exit
    if args.recheck:
        bot.recheck_trades()
        exit(0)

    # Resume mode: load previous state
    if args.resume:
        bot.load_from_file()

    # Run in 1-hour chunks, save after each
    hours = args.hours
    for hour in range(hours):
        print(f"\n{'='*50}")
        print(f"HOUR {hour+1}/{hours}")
        print(f"{'='*50}")
        bot.run(duration_minutes=60)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    bot.print_stats()