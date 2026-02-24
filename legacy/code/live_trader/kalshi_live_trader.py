# kalshi_live_trader.py
# Table-driven LIVE trading bot using decision tables from price_to_probability.py
# WARNING: This bot places REAL trades with REAL money!

import os
import requests
import base64
import time
import json
import math
import argparse
import uuid
from datetime import datetime, timezone
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

# ============== CONFIG ==============
RUN_DURATION_HOURS = 8  # How long to run
TRADES_FILE = "live_trades.json"
DECISION_TABLE_FILE = "decision_table.json"
BACKTEST_DATA_FILE = "kalshi_backtest_data.json"

# Safety settings
DRY_RUN = True  # Set to False to place real trades - CHANGE THIS ONLY WHEN READY
MAX_POSITION_SIZE = 100  # Maximum contracts per trade (safety limit)
MIN_BALANCE_RESERVE = 5  # Keep at least this much uninvested
STARTING_BALANCE = 1000  # Fallback for dry run / paper trading

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


def api_post(path: str, data: dict = {}):
    headers = get_headers("POST", path)
    resp = requests.post(BASE_URL + path, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()


def api_delete(path: str):
    headers = get_headers("DELETE", path)
    resp = requests.delete(BASE_URL + path, headers=headers)
    resp.raise_for_status()
    return resp.json() if resp.text else {}


# ============== ACCOUNT ==============
def get_balance():
    """Get current account balance."""
    try:
        data = api_get("/trade-api/v2/portfolio/balance")
        return data.get("balance", 0) / 100  # Convert cents to dollars
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return None


def place_order(ticker: str, side: str, count: int, price_cents: int):
    """
    Place a limit order on Kalshi.

    Args:
        ticker: Market ticker (e.g., "KXBTC15M-26JAN261600-00")
        side: "yes" or "no"
        count: Number of contracts
        price_cents: Limit price in cents (1-99)

    Returns:
        Order response dict or None on failure

    Note: Caller (place_trade) handles dry_run logic - this function always attempts real orders.
    """
    # Safety checks
    if count > MAX_POSITION_SIZE:
        print(f"⚠️  Order size {count} exceeds max {MAX_POSITION_SIZE}, reducing")
        count = MAX_POSITION_SIZE

    if count < 1:
        print(f"⚠️  Order size must be at least 1")
        return None

    order_data = {
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "count": count,
        "type": "limit",
        "yes_price": price_cents if side == "yes" else None,
        "no_price": price_cents if side == "no" else None,
        "client_order_id": str(uuid.uuid4()),
    }

    # Remove None values
    order_data = {k: v for k, v in order_data.items() if v is not None}

    try:
        response = api_post("/trade-api/v2/portfolio/orders", order_data)
        return response.get("order", response)
    except requests.exceptions.HTTPError as e:
        print(f"❌ Order failed: {e}")
        print(f"   Response: {e.response.text if e.response else 'No response'}")
        return None
    except Exception as e:
        print(f"❌ Order error: {e}")
        return None


def get_positions():
    """Get current open positions."""
    try:
        data = api_get("/trade-api/v2/portfolio/positions")
        return data.get("market_positions", [])
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return []


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
class LiveTradingBot:
    def __init__(self, starting_balance=None, decision_table_path=None, dry_run=True):
        self.dry_run = dry_run
        self.trades = []
        self.pending_trades = {}  # ticker -> list of trades
        self.seen_markets = {}    # ticker -> set of (price_bucket, time_bucket) tuples traded

        # Get balance from API or use provided starting balance
        if dry_run:
            self.balance = starting_balance or STARTING_BALANCE
            self.starting_balance = self.balance
            self.log(f"🧪 DRY RUN MODE - Using simulated balance: ${self.balance:.2f}")
        else:
            # Fetch real balance from Kalshi API
            api_balance = get_balance()
            if api_balance is not None:
                self.balance = api_balance
                self.starting_balance = api_balance
                self.log(f"💰 LIVE MODE - Account balance: ${self.balance:.2f}")
            else:
                raise RuntimeError("❌ Could not fetch account balance from Kalshi API")

        # Track sessions separately from the full log file. The log file can span multiple
        # restarts; this helps keep balance snapshots interpretable over time.
        self.session_id = uuid.uuid4().hex
        self.session_started_at = datetime.now(timezone.utc).isoformat()
        self.session_start_balance = self.balance

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
        # In live mode, Kalshi already deducted cost from balance when order was placed
        # So API balance IS the available balance - no need to subtract pending exposure
        if not self.dry_run:
            return max(0, self.balance)

        # In dry run mode, we need to track pending exposure manually
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
        # (Don't hedge ourselves by accident)
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

    def place_trade(self, market, opportunity):
        """Place a trade - either simulated (dry run) or real (live)."""
        ticker = market['ticker']

        # Get bet size from opportunity (Kelly-based)
        bet_size = opportunity.get('bet_size', 100)
        cell = opportunity.get('cell', {})

        # Apply safety limits
        bet_size = min(bet_size, MAX_POSITION_SIZE)

        # Keep minimum reserve
        available_after = self.get_available_balance() - bet_size
        if available_after < MIN_BALANCE_RESERVE:
            self.log(f"⚠️  Would drop below reserve (${MIN_BALANCE_RESERVE}), reducing bet size")
            bet_size = max(1, int(self.get_available_balance() - MIN_BALANCE_RESERVE))
            if bet_size < 1:
                self.log(f"⚠️  Not enough available capital above reserve, skipping")
                return None

        # Determine side and price in cents for API
        action = opportunity['action']
        price_dollars = opportunity['price']
        price_cents = int(round(price_dollars * 100))

        if action == 'BUY_YES':
            side = 'yes'
        else:
            side = 'no'

        # Place the order (real or simulated)
        order_result = None
        if not self.dry_run:
            # LIVE ORDER
            self.log(f"🔴 PLACING LIVE ORDER: {side.upper()} {bet_size} @ {price_cents}¢ on {ticker}")
            order_result = place_order(ticker, side, bet_size, price_cents)
            if order_result is None:
                self.log(f"❌ Order failed, not recording trade")
                return None

            # Check if order filled - may need to wait if resting
            fill_count = order_result.get('fill_count', 0)
            order_status = order_result.get('status', '')
            order_id = order_result.get('order_id')

            if order_status == 'resting' and fill_count == 0:
                self.log(f"⏳ Order resting, waiting for fill...")
                # Wait up to 10 seconds for fill, checking every 2 seconds
                for _ in range(5):
                    time.sleep(2)
                    try:
                        order_check = api_get(f"/trade-api/v2/portfolio/orders/{order_id}")
                        order_data = order_check.get('order', order_check)
                        fill_count = order_data.get('fill_count', 0)
                        order_status = order_data.get('status', '')
                        if fill_count > 0 or order_status == 'executed':
                            self.log(f"✅ Order filled: {fill_count} contracts")
                            order_result = order_data
                            break
                    except Exception as e:
                        self.log(f"⚠️  Error checking order: {e}")

                # If still not filled after waiting, cancel
                if fill_count == 0:
                    self.log(f"⚠️  Order still not filled after 10s. Cancelling...")
                    try:
                        if order_id:
                            cancel_resp = api_delete(f"/trade-api/v2/portfolio/orders/{order_id}")
                            self.log(f"🚫 Order cancelled: {order_id}")
                            self.log(f"   Cancel response: {cancel_resp}")
                    except Exception as e:
                        self.log(f"❌ Could not cancel order: {e}")
                    return None

            # Update bet_size to actual fill count (in case of partial fill)
            if fill_count != bet_size:
                self.log(f"ℹ️  Partial fill: {fill_count}/{bet_size} contracts")
                bet_size = fill_count

            self.log(f"✅ Order executed: {order_status}, {fill_count} contracts filled")
        else:
            # DRY RUN - simulate the order
            order_result = {"dry_run": True, "ticker": ticker, "side": side, "count": bet_size, "price": price_cents}

        trade = {
            'ticker': ticker,
            'action': action,
            'price': price_dollars,
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
            'status': 'pending',
            'dry_run': self.dry_run,
            'order_result': order_result
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

        mode_emoji = "🧪" if self.dry_run else "💵"
        mode_label = "DRY RUN" if self.dry_run else "LIVE"
        self.log(f"{mode_emoji} {mode_label} TRADE: {action} ${bet_size} @ ${price_dollars:.2f} "
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

        # In live mode, sync balance from API instead of calculating internally
        # (Kalshi already deducted cost when order was placed, so our internal calc would double-count)
        if not self.dry_run:
            # Small delay to let Kalshi process the payout
            time.sleep(2)
            old_balance = self.balance
            api_balance = get_balance()
            if api_balance is not None:
                self.balance = api_balance
                actual_change = self.balance - old_balance
                self.log(f"💰 Balance after settlement: ${self.balance:.2f} (change: ${actual_change:+.2f})")
            else:
                # Fallback to internal calculation if API fails
                self.balance += pnl
        else:
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
        # Preserve log-level metadata across restarts. Otherwise, if the bot is restarted
        # with --resume, the new run's starting balance can overwrite the log's original
        # starting balance, making (starting_balance, final_balance, trades) inconsistent.
        existing = {}
        try:
            with open(TRADES_FILE, "r") as f:
                existing = json.load(f)
        except FileNotFoundError:
            existing = {}
        except Exception:
            # If the file is corrupted or partially written, overwrite with current state.
            existing = {}

        log_created_at = existing.get('log_created_at') or datetime.now(timezone.utc).isoformat()

        # Keep the first starting balance for the whole file once trades exist.
        existing_trades = existing.get('trades', [])
        starting_balance_for_file = existing.get('starting_balance', self.starting_balance) if existing_trades else self.starting_balance

        # Keep a lightweight balance timeline for later reconciliation/debugging.
        balance_history = existing.get('balance_history', [])
        balance_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'balance': self.balance,
            'source': 'api' if not self.dry_run else 'sim',
            'session_id': self.session_id,
        })

        # Update-or-append this session summary.
        sessions = existing.get('sessions', [])
        session_summary = {
            'session_id': self.session_id,
            'started_at': self.session_started_at,
            'ended_at': datetime.now(timezone.utc).isoformat(),
            'starting_balance': self.session_start_balance,
            'ending_balance': self.balance,
            'dry_run': self.dry_run,
            'n_trades_in_file': len(self.trades),
            'n_pending_markets': len(self.pending_trades),
        }
        replaced = False
        for i, s in enumerate(sessions):
            if isinstance(s, dict) and s.get('session_id') == self.session_id:
                sessions[i] = session_summary
                replaced = True
                break
        if not replaced:
            sessions.append(session_summary)

        with open(TRADES_FILE, "w") as f:
            json.dump({
                'log_created_at': log_created_at,
                'starting_balance': starting_balance_for_file,
                'final_balance': self.balance,
                'trades': self.trades,
                'pending_trades': self.pending_trades,  # Full trade objects, not just keys
                'balance_history': balance_history,
                'sessions': sessions,
            }, f, indent=2)
        self.log(f"💾 Saved results to {TRADES_FILE}")

    def load_from_file(self):
        """Load previous trades from live_trades.json. In live mode, balance comes from API."""
        try:
            with open(TRADES_FILE, "r") as f:
                data = json.load(f)

            self.trades = data.get('trades', [])
            # Preserve file-level starting balance so log stays interpretable across restarts.
            self.starting_balance = data.get('starting_balance', self.starting_balance)

            # In live mode, always use API balance (file balance may be stale/wrong)
            if not self.dry_run:
                api_balance = get_balance()
                if api_balance is not None:
                    self.balance = api_balance
                    self.log(f"💰 Balance from API: ${self.balance:.2f} (ignoring saved ${data.get('final_balance', 0):.2f})")
                else:
                    self.balance = data.get('final_balance', self.starting_balance)
                    self.log(f"⚠️  Could not fetch API balance, using saved: ${self.balance:.2f}")
            else:
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
            bet_size = trade.get('bet_size', 100)  # Default fallback

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

    def sync_balance_from_api(self):
        """Sync balance from Kalshi API (live mode only)."""
        if self.dry_run:
            return  # No sync needed in dry run

        api_balance = get_balance()
        if api_balance is not None:
            old_balance = self.balance
            self.balance = api_balance
            if abs(old_balance - api_balance) > 0.01:
                self.log(f"💰 Balance synced: ${old_balance:.2f} → ${api_balance:.2f}")

    def run(self, duration_minutes=60):
        mode = "DRY RUN 🧪" if self.dry_run else "LIVE 🔴"
        self.log(f"🚀 Starting table-driven trading bot ({mode})")
        self.log(f"   Balance: ${self.balance:.2f}")
        self.log(f"   Decision table: {len(self.decision_table)} cells")
        self.log(f"   Fractional Kelly: {self.fractional_kelly} ({self.fractional_kelly*100:.0f}%)")
        self.log(f"   Max spread: ${self.max_spread:.2f}")
        self.log(f"   Max position size: ${MAX_POSITION_SIZE}")
        self.log(f"   Min balance reserve: ${MIN_BALANCE_RESERVE}")
        self.log(f"   Running for {duration_minutes} minutes...\n")

        if not self.dry_run:
            self.log("⚠️  WARNING: LIVE TRADING ENABLED - REAL MONEY AT RISK ⚠️")

        start_time = time.time()
        check_interval = 0.25  # Poll every 250ms (4 calls/sec, leaves headroom for orders)
        balance_sync_interval = 300  # Sync balance every 5 minutes in live mode
        last_balance_sync = time.time()

        try:
            while (time.time() - start_time) < duration_minutes * 60:
                markets = self.get_active_markets()

                for market in markets:
                    opportunity = self.check_for_opportunity(market)
                    if opportunity:
                        self.place_trade(market, opportunity)

                self.check_settled_trades()

                # Periodically sync balance from API in live mode
                if not self.dry_run and (time.time() - last_balance_sync) > balance_sync_interval:
                    self.sync_balance_from_api()
                    last_balance_sync = time.time()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            self.log("\n⏹️ Stopped by user")

        self.print_stats()
        self.save_results()


# ============== MAIN ==============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Table-driven Kalshi Trading Bot for KXBTC15M markets (Live or Dry Run)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (simulated, no real trades)
  python kalshi_live_trader.py --dry-run
  python kalshi_live_trader.py --dry-run --hours 4
  python kalshi_live_trader.py --dry-run --resume

  # LIVE TRADING (real money!)
  python kalshi_live_trader.py --live
  python kalshi_live_trader.py --live --hours 4

  # Other commands
  python kalshi_live_trader.py --recheck
  python kalshi_live_trader.py --download-data
  python kalshi_live_trader.py --backtest

Generate decision table with:
  python price_to_probability.py --kalshi kalshi_backtest_data.json --export-table decision_table.json --min-composite 0.5
        """
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--dry-run', action='store_true',
                           help='Run in dry run mode (simulated trades, no real money) - DEFAULT')
    mode_group.add_argument('--live', action='store_true',
                           help='⚠️  LIVE TRADING - Places real orders with real money!')

    # Decision table (required for trading)
    parser.add_argument('--decision-table', '-t', default=DECISION_TABLE_FILE,
                        help=f'Decision table JSON from price_to_probability.py (default: {DECISION_TABLE_FILE})')

    # Trading args
    parser.add_argument('--resume', action='store_true',
                        help='Resume from live_trades.json (load previous balance and seen markets)')
    parser.add_argument('--recheck', action='store_true',
                        help='Recheck all trades in live_trades.json and fix results from Kalshi API')
    parser.add_argument('--hours', type=int, default=RUN_DURATION_HOURS,
                        help=f'Hours to run trading (default: {RUN_DURATION_HOURS})')

    # Backtest args
    parser.add_argument('--download-data', action='store_true',
                        help='Download ALL historical market data for backtesting (slow, ~10-15 min)')
    parser.add_argument('--update-data', action='store_true',
                        help='Download only NEW markets since last download (fast)')
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest on historical data in kalshi_backtest_data.json')

    args = parser.parse_args()

    # Determine if we're in live mode (default to dry run if neither specified)
    is_live = args.live

    # Download data mode (full)
    if args.download_data:
        download_backtest_data(update_only=False)
        exit(0)

    # Update data mode (diff only)
    if args.update_data:
        download_backtest_data(update_only=True)
        exit(0)

    # Create bot with decision table
    bot = LiveTradingBot(starting_balance=STARTING_BALANCE,
                         decision_table_path=args.decision_table,
                         dry_run=not is_live)

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

    # Safety confirmation for live trading
    if is_live:
        print("\n" + "="*60)
        print("⚠️  WARNING: LIVE TRADING MODE ⚠️")
        print("="*60)
        print(f"Account balance: ${bot.balance:.2f}")
        print(f"This will place REAL orders with REAL money!")
        print("")
        confirm = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirm != "CONFIRM":
            print("Aborted. Use --dry-run for simulated trading.")
            exit(1)
        print("="*60 + "\n")

    # Run in 1-hour chunks, save after each
    hours = args.hours
    for hour in range(hours):
        mode_str = "LIVE 🔴" if is_live else "DRY RUN 🧪"
        print(f"\n{'='*50}")
        print(f"HOUR {hour+1}/{hours} ({mode_str})")
        print(f"{'='*50}")
        bot.run(duration_minutes=60)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    bot.print_stats()
