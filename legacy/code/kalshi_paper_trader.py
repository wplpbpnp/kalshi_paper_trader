# kalshi_paper_trader.py

import os
import requests
import base64
import time
import json
import math
import argparse
from datetime import datetime, timezone
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

# ============== CONFIG ==============
STARTING_BALANCE = 1000
BET_SIZE = 100
RUN_DURATION_HOURS = 8  # How long to run
TRADES_FILE = "paper_trades.json"
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

        self.buy_no_zone = (0.20, 0.45)   # Fat edge (~$25/trade, 95.7% win rate)
        self.buy_yes_zone = (0.70, 0.90)    # testing new ev finding tool, says there's edge here
        self.min_time_remaining = 0   # 30 seconds - avoid liquidity desert
        self.max_time_remaining = 1.5   # Enter from T-1.5 to T-0.5
        self.max_spread = 0.05          # 5 cent max spread

    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

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

        # Check if in a trading zone first
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
                'time_remaining': time_remaining
            }

        if in_yes_zone:
            return {
                'action': 'BUY_YES',
                'price': yes_ask,
                'yes_mid': yes_mid,
                'spread': spread,
                'time_remaining': time_remaining
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
            'status': 'pending'
        }

        self.pending_trades[ticker] = trade
        self.seen_markets.add(ticker)

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

    def backtest(self, data_file=BACKTEST_DATA_FILE):
        """Run backtest on historical data"""
        self.log(f"📊 Running backtest on {data_file}...")

        try:
            with open(data_file, "r") as f:
                markets = json.load(f)
        except FileNotFoundError:
            self.log(f"❌ No {data_file} found. Run with --download-data first.")
            return

        self.log(f"   Loaded {len(markets)} markets")
        self.log(f"   Buy NO zone: YES @ ${self.buy_no_zone[0]:.2f}-${self.buy_no_zone[1]:.2f}")
        self.log(f"   Buy YES zone: YES @ ${self.buy_yes_zone[0]:.2f}-${self.buy_yes_zone[1]:.2f}")
        self.log(f"   Time window: T-{self.max_time_remaining} to T-{self.min_time_remaining} minutes")
        self.log(f"   Max spread: ${self.max_spread:.2f}")
        self.log("")

        trades = []
        skipped_spread = 0
        skipped_no_liquidity = 0
        skipped_no_candles = 0
        skipped_outside_zone = 0

        for market in markets:
            ticker = market['ticker']
            result = market['result']
            candles = market.get('candlesticks', [])

            if not candles:
                skipped_no_candles += 1
                continue

            # Find candle at T-1 minute (or closest to our target window)
            close_time = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))

            # Look for candles in our time window
            best_candle = None
            best_time_remaining = None

            for candle in candles:
                # Handle both int timestamp and ISO string format
                end_ts = candle.get('end_period_ts') or candle.get('end_ts') or candle.get('timestamp')
                if isinstance(end_ts, int):
                    candle_time = datetime.fromtimestamp(end_ts, tz=timezone.utc)
                else:
                    candle_time = datetime.fromisoformat(str(end_ts).replace('Z', '+00:00'))
                time_remaining = (close_time - candle_time).total_seconds() / 60

                if self.min_time_remaining <= time_remaining <= self.max_time_remaining:
                    # Use the candle closest to T-1 (middle of our window)
                    if best_candle is None or abs(time_remaining - 1.0) < abs(best_time_remaining - 1.0):
                        best_candle = candle
                        best_time_remaining = time_remaining

            if not best_candle:
                skipped_no_candles += 1
                continue

            # Extract prices from candle - handle nested dict structure
            yes_bid_data = best_candle.get('yes_bid') or {}
            yes_ask_data = best_candle.get('yes_ask') or {}

            # Get close prices (in cents)
            if isinstance(yes_bid_data, dict):
                yes_bid = yes_bid_data.get('close') or 0
            else:
                yes_bid = yes_bid_data or 0

            if isinstance(yes_ask_data, dict):
                yes_ask = yes_ask_data.get('close') or 0
            else:
                yes_ask = yes_ask_data or 0

            # Fallback to price if no bid/ask
            if not yes_bid or not yes_ask:
                price_data = best_candle.get('price') or {}
                if isinstance(price_data, dict):
                    price_val = price_data.get('close') or 0
                else:
                    price_val = price_data or 0

                if not yes_bid and price_val:
                    yes_bid = price_val - 1 if price_val > 1 else price_val
                if not yes_ask and price_val:
                    yes_ask = price_val + 1

            # Ensure we have numbers
            yes_bid = yes_bid or 0
            yes_ask = yes_ask or 0

            # Convert from cents to dollars
            if yes_bid > 1:
                yes_bid = yes_bid / 100
            if yes_ask > 1:
                yes_ask = yes_ask / 100

            if yes_bid <= 0 or yes_ask <= 0:
                skipped_no_liquidity += 1
                continue

            spread = yes_ask - yes_bid
            yes_mid = (yes_bid + yes_ask) / 2

            # Check zones
            in_no_zone = self.buy_no_zone[0] <= yes_mid <= self.buy_no_zone[1]
            in_yes_zone = self.buy_yes_zone[0] <= yes_mid <= self.buy_yes_zone[1]

            if not (in_no_zone or in_yes_zone):
                skipped_outside_zone += 1
                continue

            # Check spread
            if spread > self.max_spread:
                skipped_spread += 1
                continue

            # Determine trade
            if in_no_zone:
                action = 'BUY_NO'
                # Estimate no_ask from yes_bid
                price = 1 - yes_bid
            else:
                action = 'BUY_YES'
                price = yes_ask

            # Calculate outcome
            if action == 'BUY_YES':
                won = (result == 'yes')
            else:
                won = (result == 'no')

            if won:
                pnl = self.bet_size * (1 - price)
            else:
                pnl = -self.bet_size * price

            fees = self.calculate_fees(self.bet_size, price)
            pnl -= fees

            self.balance += pnl

            trade = {
                'ticker': ticker,
                'action': action,
                'price': price,
                'yes_mid': yes_mid,
                'spread': spread,
                'time_remaining': best_time_remaining,
                'result': result,
                'won': won,
                'pnl': pnl,
                'fees': fees
            }
            trades.append(trade)

        # Calculate stats
        self.trades = trades
        wins = sum(1 for t in trades if t['won'])
        losses = len(trades) - wins
        total_pnl = sum(t['pnl'] for t in trades)

        # Separate stats by zone
        no_trades = [t for t in trades if t['action'] == 'BUY_NO']
        yes_trades = [t for t in trades if t['action'] == 'BUY_YES']

        no_wins = sum(1 for t in no_trades if t['won'])
        yes_wins = sum(1 for t in yes_trades if t['won'])

        no_pnl = sum(t['pnl'] for t in no_trades)
        yes_pnl = sum(t['pnl'] for t in yes_trades)

        # Calculate Sharpe-like ratio (mean/std of returns)
        if trades:
            returns = [t['pnl'] for t in trades]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_return = variance ** 0.5 if variance > 0 else 0
            sharpe = mean_return / std_return if std_return > 0 else 0
        else:
            mean_return = 0
            std_return = 0
            sharpe = 0

        # Print results
        self.log("=" * 60)
        self.log("BACKTEST RESULTS")
        self.log("=" * 60)
        self.log(f"Total markets analyzed: {len(markets)}")
        self.log(f"  Skipped (outside zone): {skipped_outside_zone}")
        self.log(f"  Skipped (spread > max): {skipped_spread}")
        self.log(f"  Skipped (no liquidity): {skipped_no_liquidity}")
        self.log(f"  Skipped (no candle data): {skipped_no_candles}")
        self.log(f"  Trades executed: {len(trades)}")
        self.log("")
        self.log(f"OVERALL: {wins}W / {losses}L ({100*wins/len(trades):.1f}% win rate)" if trades else "No trades")
        self.log(f"  Total PnL: ${total_pnl:+.2f}")
        self.log(f"  Mean PnL per trade: ${mean_return:+.2f}")
        self.log(f"  Std dev: ${std_return:.2f}")
        self.log(f"  Sharpe ratio: {sharpe:.3f}")
        self.log(f"  Final balance: ${self.balance:.2f}")
        self.log("")

        if no_trades:
            self.log(f"NO ZONE ({len(no_trades)} trades):")
            self.log(f"  {no_wins}W / {len(no_trades)-no_wins}L ({100*no_wins/len(no_trades):.1f}% win rate)")
            self.log(f"  PnL: ${no_pnl:+.2f}")
            self.log(f"  Avg PnL: ${no_pnl/len(no_trades):+.2f}")

        if yes_trades:
            self.log(f"YES ZONE ({len(yes_trades)} trades):")
            self.log(f"  {yes_wins}W / {len(yes_trades)-yes_wins}L ({100*yes_wins/len(yes_trades):.1f}% win rate)")
            self.log(f"  PnL: ${yes_pnl:+.2f}")
            self.log(f"  Avg PnL: ${yes_pnl/len(yes_trades):+.2f}")

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
        self.log(f"🚀 Starting paper trading bot")
        self.log(f"   Balance: ${self.balance:.2f}")
        self.log(f"   Bet size: ${self.bet_size}")
        self.log(f"   Buy NO zone: YES @ ${self.buy_no_zone[0]:.2f}-${self.buy_no_zone[1]:.2f}")
        self.log(f"   Buy YES zone: YES @ ${self.buy_yes_zone[0]:.2f}-${self.buy_yes_zone[1]:.2f}")
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