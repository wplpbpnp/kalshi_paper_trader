# kalshi_paper_trader_v2.py
# Table-driven paper trading bot using decision tables from price_to_probability.py

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
RUN_DURATION_HOURS = 8  # How long to run
TRADES_FILE = "paper_trades_v2.json"
DECISION_TABLE_FILE = "decision_table.json"
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
    def __init__(self, starting_balance=1000, decision_table_path=None):
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.trades = []
        self.pending_trades = {}  # ticker -> list of trades
        self.seen_markets = {}    # ticker -> set of (price_bucket, time_bucket) tuples traded

        # Decision table from price_to_probability.py
        self.decision_table = None
        self.fractional_kelly = 0.25  # Default, overridden by table
        self.window_minutes = 15  # Default, overridden by table

        if decision_table_path:
            self.load_decision_table(decision_table_path)

        # Spread filter (still useful even with table)
        self.max_spread = 0.05  # 5 cent max spread

    def load_decision_table(self, path):
        """Load decision table from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.decision_table = data.get('cells', [])
            self.fractional_kelly = data.get('fractional_kelly', 0.25)
            self.window_minutes = data.get('window_minutes', 15)

            print(f"📋 Loaded decision table from {path}")
            print(f"   {len(self.decision_table)} tradeable cells")
            print(f"   Fractional Kelly: {self.fractional_kelly}")
            print(f"   Window: {self.window_minutes} minutes")

            # Summarize by side
            yes_cells = [c for c in self.decision_table if c['side'] == 'YES']
            no_cells = [c for c in self.decision_table if c['side'] == 'NO']
            print(f"   YES cells: {len(yes_cells)}, NO cells: {len(no_cells)}")

        except FileNotFoundError:
            print(f"❌ Decision table not found: {path}")
            self.decision_table = []
        except Exception as e:
            print(f"❌ Error loading decision table: {e}")
            self.decision_table = []

    def find_matching_cell(self, yes_price_cents, time_fraction):
        """
        Find the best matching cell in the decision table.

        Args:
            yes_price_cents: YES price in cents (0-100)
            time_fraction: Fraction of window elapsed (0-1)

        Returns:
            Matching cell dict or None
        """
        if not self.decision_table:
            return None

        for cell in self.decision_table:
            price_low, price_high = cell['price_bucket']
            time_low, time_high = cell['time_bucket']

            if price_low <= yes_price_cents < price_high:
                if time_low <= time_fraction < time_high:
                    return cell

        return None

    def get_available_balance(self):
        """Get balance minus pending exposure."""
        pending_exposure = 0
        for trades in self.pending_trades.values():
            if isinstance(trades, list):
                pending_exposure += sum(t.get('bet_size', 0) for t in trades)
            else:
                pending_exposure += trades.get('bet_size', 0)
        return max(0, self.balance - pending_exposure)

    def get_pending_exposure_for_market(self, ticker):
        """Get total exposure already committed to a specific market."""
        if ticker not in self.pending_trades:
            return 0
        # Could be a single trade or list of trades
        trade = self.pending_trades[ticker]
        if isinstance(trade, list):
            return sum(t.get('bet_size', 0) for t in trade)
        return trade.get('bet_size', 0)

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
        """Calculate time remaining in minutes."""
        close_time = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        remaining = (close_time - now).total_seconds() / 60
        return remaining

    def calculate_time_fraction(self, market):
        """Calculate fraction of window elapsed (0-1)."""
        time_remaining = self.calculate_time_remaining(market)
        # time_remaining is in minutes, window is self.window_minutes
        # fraction elapsed = 1 - (remaining / window)
        fraction = 1 - (time_remaining / self.window_minutes)
        return max(0, min(1, fraction))  # Clamp to [0, 1]

    def check_for_opportunity(self, market):
        """Check if market matches any cell in the decision table."""
        ticker = market['ticker']

        if not self.decision_table:
            return None

        # Get current state
        yes_bid = market.get('yes_bid', 0)  # In cents
        yes_ask = market.get('yes_ask', 0)  # In cents

        if yes_bid == 0 or yes_ask == 0:
            return None

        yes_mid_cents = (yes_bid + yes_ask) / 2
        time_fraction = self.calculate_time_fraction(market)
        time_remaining = self.calculate_time_remaining(market)

        # Look up in decision table
        cell = self.find_matching_cell(yes_mid_cents, time_fraction)

        if not cell:
            return None

        # Check if we've already traded this specific cell for this market
        cell_key = (tuple(cell['price_bucket']), tuple(cell['time_bucket']))
        traded_cells = self.seen_markets.get(ticker, set())
        if cell_key in traded_cells:
            return None

        # Check if we already have a position on the OPPOSITE side for this market
        # (Don't hedge ourselves - commit to one side only)
        current_side = cell['side']  # YES or NO
        if ticker in self.pending_trades:
            for existing_trade in self.pending_trades[ticker]:
                existing_action = existing_trade.get('action', '')
                existing_side = 'YES' if existing_action == 'BUY_YES' else 'NO'
                if existing_side != current_side:
                    self.log(f"⏭️  SKIPPED: {ticker} - already have {existing_side} position, won't take {current_side}")
                    return None

        # Check spread
        spread = (yes_ask - yes_bid) / 100  # Convert to dollars
        if spread > self.max_spread:
            self.log(f"⏭️  SKIPPED: {ticker} - spread ${spread:.2f} > max ${self.max_spread:.2f} "
                    f"(YES: {yes_mid_cents:.0f}¢, T-{time_remaining:.1f}min, cell: {cell['side']})")
            return None

        # Determine action and price
        side = cell['side']
        if side == 'YES':
            action = 'BUY_YES'
            price = yes_ask / 100  # Convert to dollars
        else:
            action = 'BUY_NO'
            no_ask = market.get('no_ask', 0)
            if no_ask == 0:
                self.log(f"⏭️  SKIPPED: {ticker} - no NO liquidity "
                        f"(YES: {yes_mid_cents:.0f}¢, T-{time_remaining:.1f}min)")
                return None
            price = no_ask / 100  # Convert to dollars

        # Calculate bet size using Kelly on AVAILABLE balance (not committed capital)
        kelly = cell['kelly']
        available = self.get_available_balance()
        bet_size = available * kelly * self.fractional_kelly

        # Round to nearest dollar, minimum $1, but don't bet more than available
        bet_size = max(1, min(round(bet_size), available))

        if bet_size < 1:
            return None  # Not enough available capital

        return {
            'action': action,
            'price': price,
            'yes_mid': yes_mid_cents / 100,  # Store in dollars for consistency
            'spread': spread,
            'time_remaining': time_remaining,
            'time_fraction': time_fraction,
            'kelly': kelly,
            'bet_size': bet_size,
            'cell': cell
        }

    def place_paper_trade(self, market, opportunity):
        ticker = market['ticker']

        # Get bet size from opportunity (Kelly-based)
        bet_size = opportunity.get('bet_size', 100)
        cell = opportunity.get('cell', {})

        trade = {
            'ticker': ticker,
            'action': opportunity['action'],
            'price': opportunity['price'],
            'yes_mid': opportunity['yes_mid'],
            'spread': opportunity['spread'],
            'bet_size': bet_size,
            'time_remaining': opportunity['time_remaining'],
            'time_fraction': opportunity.get('time_fraction', 0),
            'kelly': opportunity.get('kelly', 0),
            'cell_ev': cell.get('ev', 0),
            'cell_p_win': cell.get('p_win', 0),
            'cell': cell,  # Store full cell for restoration
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'close_time': market['close_time'],
            'strike': market.get('floor_strike'),
            'status': 'pending'
        }

        # Store as list to allow multiple trades per market
        if ticker not in self.pending_trades:
            self.pending_trades[ticker] = []
        self.pending_trades[ticker].append(trade)

        # Track which cell we traded (so we don't re-enter same cell)
        cell_key = (tuple(cell.get('price_bucket', [0, 0])), tuple(cell.get('time_bucket', [0, 0])))
        if ticker not in self.seen_markets:
            self.seen_markets = getattr(self, 'seen_markets', {})
            if not isinstance(self.seen_markets, dict):
                self.seen_markets = {}
            self.seen_markets[ticker] = set()
        self.seen_markets[ticker].add(cell_key)

        kelly_pct = opportunity.get('kelly', 0) * 100
        ev_pct = cell.get('ev', 0)
        p_win = cell.get('p_win', 0) * 100

        self.log(f"📝 PAPER TRADE: {opportunity['action']} ${bet_size} @ ${opportunity['price']:.2f} "
                f"(YES: ${opportunity['yes_mid']:.2f}, T-{opportunity['time_remaining']:.1f}min, "
                f"Kelly: {kelly_pct:.0f}%, EV: {ev_pct:+.1f}%, P(win): {p_win:.0f}%) "
                f"on {ticker}")

        return trade

    def check_settled_trades(self):
        settled_tickers = []

        for ticker, trades in self.pending_trades.items():
            try:
                market = self.get_market_details(ticker)
                if market:
                    result = market.get('result')
                    # Only resolve if result is actually populated
                    if result in ['yes', 'no']:
                        # Resolve all trades for this market
                        for trade in trades:
                            self.resolve_trade(trade, result)
                        settled_tickers.append(ticker)
                    # else: keep waiting - market may be closed but not yet settled
            except Exception as e:
                self.log(f"Error checking {ticker}: {e}")

        for ticker in settled_tickers:
            del self.pending_trades[ticker]
            # Also clean up seen_markets for this ticker
            if ticker in self.seen_markets:
                del self.seen_markets[ticker]

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
        """Load previous trades and balance from paper_trades_v2.json"""
        try:
            with open(TRADES_FILE, "r") as f:
                data = json.load(f)

            self.trades = data.get('trades', [])
            self.balance = data.get('final_balance', self.starting_balance)

            # Restore pending trades (now a dict of lists)
            self.pending_trades = data.get('pending_trades', {})

            # Rebuild seen_markets from pending trades
            self.seen_markets = {}
            for ticker, trades in self.pending_trades.items():
                self.seen_markets[ticker] = set()
                for trade in trades:
                    # Reconstruct cell key from trade data
                    cell = trade.get('cell', {})
                    if cell:
                        cell_key = (tuple(cell.get('price_bucket', [0, 0])),
                                    tuple(cell.get('time_bucket', [0, 0])))
                        self.seen_markets[ticker].add(cell_key)

            # Count total pending trades
            total_pending = sum(len(trades) for trades in self.pending_trades.values())

            self.log(f"📂 Loaded {len(self.trades)} trades from {TRADES_FILE}")
            self.log(f"   Balance: ${self.balance:.2f}")
            self.log(f"   Markets with pending trades: {len(self.pending_trades)}")
            if total_pending:
                self.log(f"   ⏳ {total_pending} pending trades restored (will check for settlement)")

            return True
        except FileNotFoundError:
            self.log(f"⚠️  No {TRADES_FILE} found, starting fresh")
            return False
        except Exception as e:
            self.log(f"❌ Error loading {TRADES_FILE}: {e}")
            return False

    def backtest(self, data_file=BACKTEST_DATA_FILE):
        """Run backtest on historical data using decision table."""
        self.log(f"📊 Running table-driven backtest on {data_file}...")

        if not self.decision_table:
            self.log("❌ No decision table loaded. Cannot backtest.")
            return

        try:
            with open(data_file, "r") as f:
                markets = json.load(f)
        except FileNotFoundError:
            self.log(f"❌ No {data_file} found. Run with --download-data first.")
            return

        self.log(f"   Loaded {len(markets)} markets")
        self.log(f"   Decision table: {len(self.decision_table)} cells")
        self.log(f"   Fractional Kelly: {self.fractional_kelly}")
        self.log(f"   Max spread: ${self.max_spread:.2f}")
        self.log("")

        trades = []
        skipped_spread = 0
        skipped_no_liquidity = 0
        skipped_no_candles = 0
        skipped_no_match = 0

        for market in markets:
            ticker = market['ticker']
            result = market['result']
            candles = market.get('candlesticks', [])

            if not candles:
                skipped_no_candles += 1
                continue

            close_time = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))

            # Check each candle for a matching cell
            for candle in candles:
                end_ts = candle.get('end_period_ts') or candle.get('end_ts') or candle.get('timestamp')
                if isinstance(end_ts, int):
                    candle_time = datetime.fromtimestamp(end_ts, tz=timezone.utc)
                else:
                    candle_time = datetime.fromisoformat(str(end_ts).replace('Z', '+00:00'))

                time_remaining = (close_time - candle_time).total_seconds() / 60
                time_fraction = 1 - (time_remaining / self.window_minutes)
                time_fraction = max(0, min(1, time_fraction))

                # Extract prices from candle
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

                # Keep in cents for table lookup
                if yes_bid > 100:
                    yes_bid = yes_bid  # Already in cents
                if yes_ask > 100:
                    yes_ask = yes_ask

                if yes_bid <= 0 or yes_ask <= 0:
                    continue

                yes_mid_cents = (yes_bid + yes_ask) / 2

                # Find matching cell
                cell = self.find_matching_cell(yes_mid_cents, time_fraction)
                if not cell:
                    continue

                # Check spread (convert to dollars)
                spread = (yes_ask - yes_bid) / 100
                if spread > self.max_spread:
                    skipped_spread += 1
                    continue

                # We have a match - execute trade
                side = cell['side']
                kelly = cell['kelly']
                bet_size = self.balance * kelly * self.fractional_kelly
                bet_size = max(1, round(bet_size))

                if side == 'YES':
                    action = 'BUY_YES'
                    price = yes_ask / 100
                    won = (result == 'yes')
                else:
                    action = 'BUY_NO'
                    price = 1 - (yes_bid / 100)
                    won = (result == 'no')

                if won:
                    pnl = bet_size * (1 - price)
                else:
                    pnl = -bet_size * price

                fees = self.calculate_fees(bet_size, price)
                pnl -= fees

                self.balance += pnl

                trade = {
                    'ticker': ticker,
                    'action': action,
                    'price': price,
                    'yes_mid': yes_mid_cents / 100,
                    'spread': spread,
                    'time_remaining': time_remaining,
                    'time_fraction': time_fraction,
                    'bet_size': bet_size,
                    'kelly': kelly,
                    'cell_ev': cell.get('ev', 0),
                    'result': result,
                    'won': won,
                    'pnl': pnl,
                    'fees': fees
                }
                trades.append(trade)

                # Only one trade per market
                break
            else:
                skipped_no_match += 1

        # Calculate stats
        self.trades = trades
        wins = sum(1 for t in trades if t['won'])
        losses = len(trades) - wins
        total_pnl = sum(t['pnl'] for t in trades)

        # Separate stats by side
        no_trades = [t for t in trades if t['action'] == 'BUY_NO']
        yes_trades = [t for t in trades if t['action'] == 'BUY_YES']

        no_wins = sum(1 for t in no_trades if t['won'])
        yes_wins = sum(1 for t in yes_trades if t['won'])

        no_pnl = sum(t['pnl'] for t in no_trades)
        yes_pnl = sum(t['pnl'] for t in yes_trades)

        # Calculate Sharpe-like ratio
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
        self.log("BACKTEST RESULTS (Table-Driven)")
        self.log("=" * 60)
        self.log(f"Total markets analyzed: {len(markets)}")
        self.log(f"  Skipped (no cell match): {skipped_no_match}")
        self.log(f"  Skipped (spread > max): {skipped_spread}")
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
            self.log(f"BUY NO ({len(no_trades)} trades):")
            self.log(f"  {no_wins}W / {len(no_trades)-no_wins}L ({100*no_wins/len(no_trades):.1f}% win rate)")
            self.log(f"  PnL: ${no_pnl:+.2f}")
            self.log(f"  Avg PnL: ${no_pnl/len(no_trades):+.2f}")

        if yes_trades:
            self.log(f"BUY YES ({len(yes_trades)} trades):")
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

        # First, resolve any pending trades (now stored as lists)
        if pending_trades:
            total_pending = sum(len(t) if isinstance(t, list) else 1 for t in pending_trades.values())
            self.log(f"   Checking {total_pending} pending trades across {len(pending_trades)} markets...")
            for ticker, trade_list in pending_trades.items():
                # Handle both old format (single trade) and new format (list)
                if not isinstance(trade_list, list):
                    trade_list = [trade_list]

                market = self.get_market_details(ticker)
                if not market:
                    self.log(f"⚠️  Could not fetch {ticker}, keeping as pending")
                    self.pending_trades[ticker] = trade_list
                    continue

                result = market.get('result')
                if result in ['yes', 'no']:
                    # Resolve all trades for this market
                    for trade in trade_list:
                        action = trade['action']
                        price = trade['price']
                        bet_size = trade.get('bet_size', 100)

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
                    self.pending_trades[ticker] = trade_list

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
        self.log(f"🚀 Starting table-driven paper trading bot")
        self.log(f"   Balance: ${self.balance:.2f}")
        self.log(f"   Decision table: {len(self.decision_table)} cells")
        self.log(f"   Fractional Kelly: {self.fractional_kelly} ({self.fractional_kelly*100:.0f}%)")
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
        description='Table-driven Kalshi Paper Trading Bot for KXBTC15M markets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kalshi_paper_trader_v2.py --decision-table decision_table.json
  python kalshi_paper_trader_v2.py --decision-table decision_table.json --hours 4
  python kalshi_paper_trader_v2.py --decision-table decision_table.json --resume
  python kalshi_paper_trader_v2.py --recheck
  python kalshi_paper_trader_v2.py --download-data
  python kalshi_paper_trader_v2.py --backtest --decision-table decision_table.json

Generate decision table with:
  python price_to_probability.py --kalshi kalshi_backtest_data.json --export-table decision_table.json --min-composite 0.5
        """
    )

    # Decision table (required for trading)
    parser.add_argument('--decision-table', '-t', default=DECISION_TABLE_FILE,
                        help=f'Decision table JSON from price_to_probability.py (default: {DECISION_TABLE_FILE})')

    # Paper trading args
    parser.add_argument('--resume', action='store_true',
                        help='Resume from paper_trades_v2.json (load previous balance and seen markets)')
    parser.add_argument('--recheck', action='store_true',
                        help='Recheck all trades in paper_trades_v2.json and fix results from Kalshi API')
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

    # Create bot with decision table
    bot = PaperTradingBot(starting_balance=STARTING_BALANCE,
                          decision_table_path=args.decision_table)

    # Check if we have a decision table for trading modes
    if not bot.decision_table and not args.recheck:
        print(f"❌ No decision table loaded. Generate one with:")
        print(f"   python price_to_probability.py --kalshi kalshi_backtest_data.json --export-table {args.decision_table} --min-composite 0.5")
        exit(1)

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