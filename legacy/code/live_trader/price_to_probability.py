#!/usr/bin/env python3
"""
Price-to-Probability Translator

Converts any price series into probability space by:
1. Slicing data into time windows (synthetic prediction markets)
2. Each window "resolves YES" if close > open, "NO" if close < open
3. Building a surface: (distance_from_open, time_elapsed) → P(resolves YES)

This lets you translate any price movement into a continuation probability.
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import argparse

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plotting disabled")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, using basic CSV parsing")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class PriceToProbability:
    """Convert price data to probability space."""

    def __init__(self, window_minutes=15, normalize_by_volatility=False,
                 confidence_k=30, trade_exponent=0.5, recency_half_life=50,
                 mode='real', decay_half_life_days=7.0):
        """
        Args:
            window_minutes: Size of each synthetic market window
            normalize_by_volatility: If True, normalize distance by rolling volatility
            confidence_k: Samples needed for 50% confidence weight in composite score
            trade_exponent: Exponent for trade frequency in composite (0.5=sqrt, 1.0=linear)
            recency_half_life: Windows until weight decays to 50% (higher = slower decay)
            mode: 'real' for real market trading (returns relative to entry),
                  'prediction' for prediction markets (binary payoff)
            decay_half_life_days: Half-life in days for time-weighted p(win) calculation
                                  (e.g., 7.0 means data from 7 days ago has 50% weight)
        """
        self.window_minutes = window_minutes
        self.normalize_by_volatility = normalize_by_volatility
        self.confidence_k = confidence_k
        self.trade_exponent = trade_exponent
        self.recency_half_life = recency_half_life
        self.mode = mode
        self.decay_half_life_days = decay_half_life_days

        # Reference time for time-weighted calculations (set when loading data)
        self.reference_time = None

        # Store markets list for time-weighted analysis
        self.markets_data = []

        # The surface data
        # Key: (price_bucket, time_bucket) in prediction mode
        # Key: (distance_bucket, time_bucket) in real mode
        # Value: {"yes": count, "no": count, ...}
        self.surface_data = defaultdict(lambda: {"yes": 0, "no": 0})

        # Raw windows for analysis
        self.windows = []

        # Configuration for bucketing (will be auto-calibrated or manually set)
        self.distance_buckets = None  # Set during build_surface or manually
        self.time_buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Fraction of window elapsed

        # For prediction market mode: YES price buckets (in cents, 0-100)
        self.price_buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # For manual zoom
        self.plot_distance_range = None  # (min, max) in decimal
        self.plot_time_range = None  # (min, max) as fraction

    def load_csv(self, filepath, timestamp_col='timestamp', price_col='close',
                 date_format=None):
        """Load price data from CSV."""
        if HAS_PANDAS:
            df = pd.read_csv(filepath)

            # Parse timestamps
            if date_format:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=date_format)
            else:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])

            # Sort by time
            df = df.sort_values(timestamp_col)

            # Extract as list of (timestamp, price)
            data = list(zip(df[timestamp_col].tolist(), df[price_col].tolist()))
            return data
        else:
            # Basic CSV parsing
            data = []
            with open(filepath, 'r') as f:
                header = f.readline().strip().split(',')
                ts_idx = header.index(timestamp_col)
                price_idx = header.index(price_col)

                for line in f:
                    parts = line.strip().split(',')
                    ts = datetime.fromisoformat(parts[ts_idx].replace('Z', '+00:00'))
                    price = float(parts[price_idx])
                    data.append((ts, price))

            data.sort(key=lambda x: x[0])
            return data

    def load_json(self, filepath, timestamp_key='timestamp', price_key='price'):
        """Load price data from JSON."""
        with open(filepath, 'r') as f:
            raw = json.load(f)

        data = []
        for item in raw:
            if isinstance(item.get(timestamp_key), str):
                ts = datetime.fromisoformat(item[timestamp_key].replace('Z', '+00:00'))
            else:
                ts = datetime.fromtimestamp(item[timestamp_key])
            price = float(item[price_key])
            data.append((ts, price))

        data.sort(key=lambda x: x[0])
        return data

    def slice_into_windows(self, data):
        """
        Slice price data into non-overlapping windows.
        Each window is a synthetic prediction market.

        Args:
            data: List of (timestamp, price) tuples, sorted by time

        Returns:
            List of window dicts with open, close, samples, outcome
        """
        if not data:
            return []

        windows = []
        window_delta = timedelta(minutes=self.window_minutes)

        # Find window boundaries
        start_time = data[0][0]
        # Align to clean boundaries (e.g., 15-min marks)
        start_time = start_time.replace(second=0, microsecond=0)
        minute = start_time.minute
        aligned_minute = (minute // self.window_minutes) * self.window_minutes
        start_time = start_time.replace(minute=aligned_minute)

        current_window_start = start_time
        current_window = {
            'start': current_window_start,
            'end': current_window_start + window_delta,
            'samples': [],
            'open_price': None,
            'close_price': None
        }

        for ts, price in data:
            # Check if we've moved past current window
            while ts >= current_window['end']:
                # Finalize current window if it has data
                if current_window['samples']:
                    current_window['open_price'] = current_window['samples'][0][1]
                    current_window['close_price'] = current_window['samples'][-1][1]
                    current_window['outcome'] = 'YES' if current_window['close_price'] > current_window['open_price'] else 'NO'
                    windows.append(current_window)

                # Start new window
                current_window_start = current_window['end']
                current_window = {
                    'start': current_window_start,
                    'end': current_window_start + window_delta,
                    'samples': [],
                    'open_price': None,
                    'close_price': None
                }

            # Add sample to current window
            current_window['samples'].append((ts, price))

        # Don't forget last window
        if current_window['samples']:
            current_window['open_price'] = current_window['samples'][0][1]
            current_window['close_price'] = current_window['samples'][-1][1]
            current_window['outcome'] = 'YES' if current_window['close_price'] > current_window['open_price'] else 'NO'
            windows.append(current_window)

        self.windows = windows
        return windows

    def calibrate_distance_buckets(self, windows, n_buckets=10):
        """
        Auto-calibrate distance buckets based on actual move sizes in the data.
        Uses percentiles to create buckets that evenly distribute the data.
        """
        # Collect all distances from all samples
        all_distances = []
        for window in windows:
            if not window['samples'] or not window['open_price']:
                continue
            open_price = window['open_price']
            for ts, price in window['samples']:
                dist = abs((price - open_price) / open_price)
                all_distances.append(dist)

        if not all_distances:
            # Fallback to default buckets
            self.distance_buckets = [0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, float('inf')]
            return

        # Use percentiles to create buckets
        percentiles = np.linspace(0, 99, n_buckets + 1)
        bucket_edges = [0] + [np.percentile(all_distances, p) for p in percentiles[1:]] + [float('inf')]

        # Remove duplicates and ensure monotonic
        unique_edges = [bucket_edges[0]]
        for edge in bucket_edges[1:]:
            if edge > unique_edges[-1]:
                unique_edges.append(edge)

        self.distance_buckets = unique_edges

        print(f"Auto-calibrated {len(self.distance_buckets)-1} distance buckets based on data")
        print(f"  Range: {self.distance_buckets[1]*100:.3f}% to {self.distance_buckets[-2]*100:.2f}%")

    def set_distance_buckets(self, bucket_edges_pct):
        """
        Manually set distance bucket edges (in percentage points).
        Example: [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5] for fine-grained low-vol analysis
        """
        self.distance_buckets = [b / 100 for b in bucket_edges_pct] + [float('inf')]

    def get_distance_bucket(self, pct_distance):
        """Get the bucket index for a given percentage distance from open."""
        if self.distance_buckets is None:
            # Default fallback
            self.distance_buckets = [0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, float('inf')]

        abs_dist = abs(pct_distance)
        for i, threshold in enumerate(self.distance_buckets[1:], 1):
            if abs_dist < threshold:
                return i - 1
        return len(self.distance_buckets) - 2

    def get_time_bucket(self, time_fraction):
        """Get the bucket index for time elapsed (0 to 1)."""
        for i, threshold in enumerate(self.time_buckets[1:], 1):
            if time_fraction < threshold:
                return i - 1
        return len(self.time_buckets) - 2

    def get_price_bucket(self, yes_price_cents):
        """Get the bucket index for YES price in cents (0-100) for prediction market mode."""
        for i, threshold in enumerate(self.price_buckets[1:], 1):
            if yes_price_cents < threshold:
                return i - 1
        return len(self.price_buckets) - 2

    def load_kalshi_json(self, filepath):
        """
        Load Kalshi prediction market data and build the surface.

        Expected format: List of markets, each with:
        - ticker: market identifier
        - strike: strike price
        - result: "yes" or "no"
        - close_time: when market resolved
        - candlesticks: list of 1-minute candles with yes_bid, yes_ask, price
        """
        print(f"Loading Kalshi data from {filepath}...")

        with open(filepath, 'r') as f:
            markets = json.load(f)

        print(f"Loaded {len(markets)} markets")

        # Initialize surface data for prediction market mode
        self.surface_data = defaultdict(lambda: {
            "yes": 0,
            "no": 0,
            "win_magnitudes": [],  # profit when YES wins (buying YES)
            "loss_magnitudes": [],  # loss when NO wins (buying YES)
            "windows_yes": set(),
            "windows_no": set(),
            "yes_timestamps": [],  # timestamps for each YES observation (for time decay)
            "no_timestamps": [],   # timestamps for each NO observation (for time decay)
        })

        # Store markets for time-weighted analysis
        self.markets_data = markets

        # Process each market
        valid_markets = 0
        total_samples = 0

        # Track the most recent close_time for reference
        latest_close_time = None

        for market_idx, market in enumerate(markets):
            result = market.get('result', '').lower()
            if result not in ['yes', 'no']:
                continue

            candlesticks = market.get('candlesticks', [])
            if not candlesticks:
                continue

            # Determine window duration from candlesticks
            if len(candlesticks) < 2:
                continue

            # Parse close_time for time-weighted calculations
            close_time_str = market.get('close_time', '')
            if close_time_str:
                try:
                    close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                    if latest_close_time is None or close_time > latest_close_time:
                        latest_close_time = close_time
                except:
                    close_time = None
            else:
                close_time = None

            # Each candlestick is 1 minute, total window is number of candlesticks
            n_candles = len(candlesticks)

            valid_markets += 1

            # Process each candlestick
            for candle_idx, candle in enumerate(candlesticks):
                # Get YES price (use mid of bid/ask, or last trade price)
                yes_bid = candle.get('yes_bid', {}).get('close', 0)
                yes_ask = candle.get('yes_ask', {}).get('close', 100)

                # Use midpoint if both exist, otherwise use last trade price
                if yes_bid > 0 and yes_ask < 100:
                    yes_price = (yes_bid + yes_ask) / 2
                else:
                    # Fall back to last trade price
                    yes_price = candle.get('price', {}).get('close', 50)

                if yes_price is None or yes_price <= 0 or yes_price >= 100:
                    continue

                # Calculate time fraction (how far through the window)
                time_fraction = (candle_idx + 1) / n_candles

                # Get buckets
                price_bucket = self.get_price_bucket(yes_price)
                time_bucket = self.get_time_bucket(time_fraction)

                key = (price_bucket, time_bucket)

                # Record outcome and magnitudes for EV calculation
                # If buying YES at yes_price cents:
                #   - YES wins: profit = (100 - yes_price) / 100
                #   - NO wins: loss = yes_price / 100
                if result == 'yes':
                    self.surface_data[key]["yes"] += 1
                    self.surface_data[key]["windows_yes"].add(market_idx)
                    profit = (100 - yes_price) / 100  # As decimal
                    self.surface_data[key]["win_magnitudes"].append(profit)
                    if close_time:
                        self.surface_data[key]["yes_timestamps"].append(close_time)
                else:
                    self.surface_data[key]["no"] += 1
                    self.surface_data[key]["windows_no"].add(market_idx)
                    loss = yes_price / 100  # As decimal
                    self.surface_data[key]["loss_magnitudes"].append(loss)
                    if close_time:
                        self.surface_data[key]["no_timestamps"].append(close_time)

                total_samples += 1

        # Set reference time to the latest close_time
        self.reference_time = latest_close_time

        # Store windows for recency calculation
        self.windows = [{'idx': i} for i in range(valid_markets)]

        print(f"Processed {valid_markets} valid markets, {total_samples} samples")

        # Print bucket distribution
        print(f"\nPrice buckets: {self.price_buckets}")
        print(f"Time buckets: {self.time_buckets}")

        return self.surface_data

    def build_surface(self, windows=None, auto_calibrate=True):
        """
        Build the probability surface from windows.

        For each sample point in each window:
        - Calculate distance from open (as %)
        - Calculate time elapsed (as fraction of window)
        - Record whether the window resolved YES or NO
        - Track the magnitude of gains/losses for EV calculation

        Args:
            windows: List of window dicts (uses self.windows if None)
            auto_calibrate: If True, auto-calibrate distance buckets to fit the data
        """
        if windows is None:
            windows = self.windows

        # Auto-calibrate distance buckets if needed
        if auto_calibrate and self.distance_buckets is None:
            self.calibrate_distance_buckets(windows)

        # Extended surface data with magnitude tracking
        self.surface_data = defaultdict(lambda: {
            "yes": 0,
            "no": 0,
            "win_magnitudes": [],  # % gain when leader wins
            "loss_magnitudes": [],  # % loss when leader loses
            "windows_yes": set(),  # unique windows that resolved YES
            "windows_no": set()    # unique windows that resolved NO
        })

        for window_idx, window in enumerate(windows):
            if not window['samples'] or len(window['samples']) < 2:
                continue

            open_price = window['open_price']
            close_price = window['close_price']
            outcome = window['outcome']
            window_duration = (window['end'] - window['start']).total_seconds()

            # Calculate the final move from open to close (as %)
            final_move = (close_price - open_price) / open_price

            # Sample points throughout the window
            for ts, price in window['samples']:
                # Calculate distance from open (as percentage)
                pct_distance = (price - open_price) / open_price

                # Calculate time elapsed (as fraction)
                elapsed = (ts - window['start']).total_seconds()
                time_fraction = elapsed / window_duration

                # Get buckets
                dist_bucket = self.get_distance_bucket(pct_distance)
                time_bucket = self.get_time_bucket(time_fraction)

                # Determine if we're above or below open
                side = "above" if pct_distance >= 0 else "below"

                # Record: did the window resolve in the direction of current position?
                # If price is above open and window resolved YES, the "leader" won
                # If price is below open and window resolved NO, the "leader" won
                if side == "above":
                    leader_won = (outcome == "YES")
                    # If we're long (above open), gain/loss is final_move - current_distance
                    # Actually simpler: if leader won, gain = |final_move| - |current_distance|
                    # if leader lost, loss = |current_distance| + |final_move| (price went other way)
                    remaining_gain = abs(final_move) - abs(pct_distance) if leader_won else None
                    remaining_loss = abs(pct_distance) + abs(final_move) if not leader_won else None
                else:
                    leader_won = (outcome == "NO")
                    remaining_gain = abs(final_move) - abs(pct_distance) if leader_won else None
                    remaining_loss = abs(pct_distance) + abs(final_move) if not leader_won else None

                key = (dist_bucket, time_bucket)
                if leader_won:
                    self.surface_data[key]["yes"] += 1
                    self.surface_data[key]["windows_yes"].add(window_idx)
                    if remaining_gain is not None:
                        self.surface_data[key]["win_magnitudes"].append(remaining_gain)
                else:
                    self.surface_data[key]["no"] += 1
                    self.surface_data[key]["windows_no"].add(window_idx)
                    if remaining_loss is not None:
                        self.surface_data[key]["loss_magnitudes"].append(remaining_loss)

        return self.surface_data

    def get_probability(self, pct_distance, time_fraction):
        """
        Query the surface: given distance from open and time elapsed,
        what's the probability the leader wins (price stays on current side)?

        Args:
            pct_distance: Current price distance from open as decimal (0.01 = 1%)
            time_fraction: Time elapsed as fraction of window (0.5 = halfway)

        Returns:
            Probability (0-1) that price closes on current side of open
        """
        dist_bucket = self.get_distance_bucket(pct_distance)
        time_bucket = self.get_time_bucket(time_fraction)

        key = (dist_bucket, time_bucket)
        data = self.surface_data[key]

        total = data["yes"] + data["no"]
        if total == 0:
            return 0.5  # No data, assume 50/50

        return data["yes"] / total

    def get_recency_score(self, window_indices, total_windows):
        """
        Calculate recency score for a set of windows using exponential decay.

        Returns a value between 0 and 1, where:
        - 1.0 = all trades happened in the most recent window
        - 0.5 = average trade age equals half_life
        - Lower = trades are older/more decayed
        """
        if not window_indices or total_windows == 0:
            return 0.0

        total_weight = 0.0
        for window_idx in window_indices:
            # Age = how many windows ago (0 = most recent)
            age = total_windows - 1 - window_idx
            # Exponential decay with half-life
            weight = 0.5 ** (age / self.recency_half_life)
            total_weight += weight

        # Normalize by number of windows to get average recency weight
        avg_weight = total_weight / len(window_indices)
        return avg_weight

    def get_time_weighted_p_win(self, key):
        """
        Calculate time-weighted P(win) for a cell using exponential decay.

        Recent observations are weighted more heavily than older ones.
        Uses decay_half_life_days parameter.

        Args:
            key: (x_bucket, time_bucket) tuple

        Returns:
            dict with:
                - weighted_p_win: time-weighted probability
                - weighted_yes: sum of weights for YES observations
                - weighted_no: sum of weights for NO observations
                - unweighted_p_win: simple count-based probability for comparison
        """
        import math

        data = self.surface_data.get(key, {})
        yes_timestamps = data.get("yes_timestamps", [])
        no_timestamps = data.get("no_timestamps", [])
        yes_count = data.get("yes", 0)
        no_count = data.get("no", 0)

        # Calculate unweighted p_win
        total_count = yes_count + no_count
        if total_count == 0:
            return {
                "weighted_p_win": 0.5,
                "weighted_yes": 0,
                "weighted_no": 0,
                "unweighted_p_win": 0.5,
                "total_weight": 0
            }

        unweighted_p_win = yes_count / total_count

        # If no reference time or no timestamps, fall back to unweighted
        if self.reference_time is None or (not yes_timestamps and not no_timestamps):
            return {
                "weighted_p_win": unweighted_p_win,
                "weighted_yes": yes_count,
                "weighted_no": no_count,
                "unweighted_p_win": unweighted_p_win,
                "total_weight": total_count
            }

        # Calculate decay constant: ln(2) / half_life_days
        decay_constant = math.log(2) / self.decay_half_life_days

        def calculate_weight(timestamp):
            """Calculate exponential decay weight based on age."""
            age_seconds = (self.reference_time - timestamp).total_seconds()
            age_days = age_seconds / (24 * 3600)
            if age_days < 0:
                age_days = 0  # Future timestamps get full weight
            return math.exp(-decay_constant * age_days)

        # Calculate weighted sums
        weighted_yes = sum(calculate_weight(ts) for ts in yes_timestamps)
        weighted_no = sum(calculate_weight(ts) for ts in no_timestamps)

        # If we have timestamps for only some observations, scale up proportionally
        # This handles cases where some observations don't have timestamps
        if len(yes_timestamps) < yes_count and len(yes_timestamps) > 0:
            weighted_yes *= yes_count / len(yes_timestamps)
        elif len(yes_timestamps) == 0 and yes_count > 0:
            # No timestamps for YES, use unweighted count with average decay
            avg_decay = 0.5  # Assume half decayed on average
            weighted_yes = yes_count * avg_decay

        if len(no_timestamps) < no_count and len(no_timestamps) > 0:
            weighted_no *= no_count / len(no_timestamps)
        elif len(no_timestamps) == 0 and no_count > 0:
            avg_decay = 0.5
            weighted_no = no_count * avg_decay

        total_weight = weighted_yes + weighted_no
        if total_weight == 0:
            return {
                "weighted_p_win": unweighted_p_win,
                "weighted_yes": 0,
                "weighted_no": 0,
                "unweighted_p_win": unweighted_p_win,
                "total_weight": 0
            }

        weighted_p_win = weighted_yes / total_weight

        return {
            "weighted_p_win": weighted_p_win,
            "weighted_yes": weighted_yes,
            "weighted_no": weighted_no,
            "unweighted_p_win": unweighted_p_win,
            "total_weight": total_weight
        }

    def get_ev(self, pct_distance, time_fraction):
        """
        Calculate expected value for entering at this point and holding to window close.

        Returns:
            dict with: p_win, avg_win, avg_loss, ev, kelly_fraction, n_samples, trades
        """
        dist_bucket = self.get_distance_bucket(pct_distance)
        time_bucket = self.get_time_bucket(time_fraction)

        key = (dist_bucket, time_bucket)
        data = self.surface_data[key]

        total = data["yes"] + data["no"]
        if total == 0:
            return {"p_win": 0.5, "avg_win": 0, "avg_loss": 0, "ev": 0, "kelly": 0, "n": 0, "trades": 0}

        p_win = data["yes"] / total
        p_loss = 1 - p_win

        # Average magnitudes
        avg_win = np.mean(data["win_magnitudes"]) if data["win_magnitudes"] else 0
        avg_loss = np.mean(data["loss_magnitudes"]) if data["loss_magnitudes"] else 0

        # Expected value (as % of position)
        ev = p_win * avg_win - p_loss * avg_loss

        # Kelly criterion: f* = (p * b - q) / b where b = win/loss ratio
        # Or simplified: f* = p - q/b = p - (1-p) * (avg_loss / avg_win)
        if avg_win > 0 and avg_loss > 0:
            b = avg_win / avg_loss  # odds ratio
            kelly = (p_win * b - p_loss) / b
            kelly = max(0, kelly)  # Don't go negative
        elif p_win == 1.0 and ev > 0:
            kelly = float('inf')  # 100% win rate with positive EV = bet max
        else:
            kelly = 0

        # Unique trade opportunities
        all_windows = data["windows_yes"] | data["windows_no"]
        trades = len(all_windows)

        # Recency score
        total_windows = len(self.windows) if self.windows else 1
        recency = self.get_recency_score(all_windows, total_windows)

        return {
            "p_win": p_win,
            "avg_win": avg_win * 100,  # Convert to %
            "avg_loss": avg_loss * 100,
            "ev": ev * 100,
            "kelly": kelly,
            "n": total,
            "trades": trades,
            "recency": recency
        }

    def get_ev_matrix(self):
        """
        Calculate EV for all buckets.

        Returns:
            (ev_matrix, p_win_matrix, kelly_matrix, dist_labels, time_labels, counts, trade_freq, recency_matrix)
        """
        # Determine which buckets to use based on mode
        if self.mode == 'prediction':
            x_buckets = self.price_buckets
        else:
            if self.distance_buckets is None:
                return None, None, None, [], [], None, None, None
            x_buckets = self.distance_buckets

        n_x = len(x_buckets) - 1
        n_time = len(self.time_buckets) - 1

        ev_matrix = np.zeros((n_x, n_time))
        p_win_matrix = np.zeros((n_x, n_time))
        kelly_matrix = np.zeros((n_x, n_time))
        counts = np.zeros((n_x, n_time))
        trade_freq = np.zeros((n_x, n_time))
        recency_matrix = np.zeros((n_x, n_time))

        total_windows = len(self.windows) if self.windows else 1

        for (x_bucket, time_bucket), data in self.surface_data.items():
            total = data["yes"] + data["no"]
            if total > 0:
                # Use time-weighted p(win) instead of simple counts
                key = (x_bucket, time_bucket)
                tw_result = self.get_time_weighted_p_win(key)
                p_win = tw_result["weighted_p_win"]
                p_loss = 1 - p_win

                # Calculate EV using bucket midpoint for consistency
                if self.mode == 'prediction':
                    # Use midpoint of price bucket (in decimal, 0-1)
                    bucket_low = x_buckets[x_bucket]
                    bucket_high = x_buckets[x_bucket + 1] if x_bucket + 1 < len(x_buckets) else 100
                    midpoint = (bucket_low + bucket_high) / 2 / 100  # Convert cents to decimal

                    # EV for buying YES at midpoint price
                    # Win: profit = 1 - midpoint, Loss: lose midpoint
                    avg_win = 1 - midpoint
                    avg_loss = midpoint
                    ev = p_win * avg_win - p_loss * avg_loss
                else:
                    # Real market mode: use observed magnitudes
                    avg_win = np.mean(data["win_magnitudes"]) if data["win_magnitudes"] else 0
                    avg_loss = np.mean(data["loss_magnitudes"]) if data["loss_magnitudes"] else 0
                    ev = p_win * avg_win - p_loss * avg_loss

                # Kelly criterion
                if avg_win > 0 and avg_loss > 0:
                    b = avg_win / avg_loss
                    kelly = max(0, (p_win * b - p_loss) / b)
                elif p_win == 1.0 and ev > 0:
                    kelly = float('inf')  # 100% win rate with positive EV
                else:
                    kelly = 0

                all_windows = data["windows_yes"] | data["windows_no"]

                ev_matrix[x_bucket, time_bucket] = ev * 100  # As percentage
                p_win_matrix[x_bucket, time_bucket] = p_win
                kelly_matrix[x_bucket, time_bucket] = kelly
                counts[x_bucket, time_bucket] = total
                trade_freq[x_bucket, time_bucket] = len(all_windows)
                recency_matrix[x_bucket, time_bucket] = self.get_recency_score(all_windows, total_windows)

        # Labels
        x_labels = []
        if self.mode == 'prediction':
            # Price buckets in cents
            for i in range(len(x_buckets) - 1):
                low = x_buckets[i]
                high = x_buckets[i + 1]
                x_labels.append(f"{low}-{high}¢")
        else:
            # Distance buckets in percent
            for i in range(len(x_buckets) - 1):
                low = x_buckets[i] * 100
                high = x_buckets[i + 1] * 100
                if high == float('inf'):
                    x_labels.append(f">{low:.2f}%")
                else:
                    x_labels.append(f"{low:.2f}-{high:.2f}%")

        time_labels = []
        for i in range(len(self.time_buckets) - 1):
            low_pct = self.time_buckets[i]
            high_pct = self.time_buckets[i + 1]
            time_labels.append(f"{low_pct*100:.0f}-{high_pct*100:.0f}%")

        return ev_matrix, p_win_matrix, kelly_matrix, x_labels, time_labels, counts, trade_freq, recency_matrix

    def get_surface_matrix(self, for_plot=False):
        """
        Convert surface data to a 2D matrix for visualization.

        Args:
            for_plot: If True, use multiline labels suitable for plots

        Returns:
            (matrix, distance_labels, time_labels, counts, trade_freq)
            - counts: total samples (for statistical reliability)
            - trade_freq: unique windows (for trade opportunity frequency)
        """
        # Determine which buckets to use based on mode
        if self.mode == 'prediction':
            x_buckets = self.price_buckets
        else:
            x_buckets = self.distance_buckets

        n_x = len(x_buckets) - 1
        n_time = len(self.time_buckets) - 1

        matrix = np.zeros((n_x, n_time))
        counts = np.zeros((n_x, n_time))
        trade_freq = np.zeros((n_x, n_time))

        for (x_bucket, time_bucket), data in self.surface_data.items():
            total = data["yes"] + data["no"]
            if total > 0:
                # Use time-weighted P(win) for consistency with EV calculations
                key = (x_bucket, time_bucket)
                tw_result = self.get_time_weighted_p_win(key)
                matrix[x_bucket, time_bucket] = tw_result["weighted_p_win"]
                counts[x_bucket, time_bucket] = total
                # Unique windows = trade opportunities
                unique_windows = len(data["windows_yes"] | data["windows_no"])
                trade_freq[x_bucket, time_bucket] = unique_windows

        # X-axis labels
        x_labels = []
        if self.mode == 'prediction':
            # Price buckets in cents
            for i in range(len(x_buckets) - 1):
                low = x_buckets[i]
                high = x_buckets[i + 1]
                x_labels.append(f"{low}-{high}¢")
        else:
            # Distance buckets in percent
            for i in range(len(x_buckets) - 1):
                low = x_buckets[i] * 100
                high = x_buckets[i + 1] * 100
                if high == float('inf'):
                    x_labels.append(f">{low:.2f}%")
                else:
                    x_labels.append(f"{low:.2f}-{high:.2f}%")

        # Time labels
        time_labels = []
        for i in range(len(self.time_buckets) - 1):
            low_pct = self.time_buckets[i]
            high_pct = self.time_buckets[i + 1]
            mid_pct = (low_pct + high_pct) / 2

            # Human-friendly elapsed time at midpoint
            elapsed_str = format_time_elapsed(self.window_minutes, mid_pct)

            if for_plot:
                # Multiline for plot readability
                time_labels.append(f"{low_pct*100:.0f}-{high_pct*100:.0f}%\n({elapsed_str})")
            else:
                # Single line for table
                time_labels.append(f"{low_pct*100:.0f}-{high_pct*100:.0f}%")

        return matrix, x_labels, time_labels, counts, trade_freq

    def print_ev_surface(self):
        """Print the EV surface as a table."""
        ev_matrix, p_win_matrix, kelly_matrix, x_labels, time_labels, counts, trade_freq, recency_matrix = self.get_ev_matrix()

        if ev_matrix is None:
            print("No surface data available")
            return

        window_str = format_window_size(self.window_minutes)

        print("\n" + "="*80)
        if self.mode == 'prediction':
            print(f"EXPECTED VALUE SURFACE (% gain per contract) - Prediction Markets")
            print("="*80)
            print("EV = P(YES) × (1 - price) - P(NO) × price")
            print("Positive EV = buy YES, Negative = buy NO (or avoid)")
            x_header = "YES Price"
        else:
            print(f"EXPECTED VALUE SURFACE (% gain per trade) - {window_str} windows")
            print("="*80)
            print("EV = P(win) × avg_win - P(loss) × avg_loss")
            print("Positive EV = edge exists, Negative = avoid")
            x_header = "Distance"
        print()

        # Header
        print(f"{x_header:<15}", end="")
        for tl in time_labels:
            print(f"{tl:>10}", end="")
        print()
        print("-" * (15 + 10 * len(time_labels)))

        # Find max positive EV for highlighting
        max_ev = np.max(ev_matrix)

        # Rows
        for i, xl in enumerate(x_labels):
            print(f"{xl:<15}", end="")
            for j in range(len(time_labels)):
                ev = ev_matrix[i, j]
                count = counts[i, j]
                if count > 5:  # Need minimum samples
                    if ev > 0:
                        print(f"{ev:>+9.3f}%", end="")
                    else:
                        print(f"{ev:>9.3f}%", end="")
                else:
                    print(f"{'--':>10}", end="")
            print()

        # Summary of best opportunities
        print("\n" + "="*60)
        print("TOP EV OPPORTUNITIES (min 10 samples)")
        print("="*60)

        # Calculate composite scores
        # Composite = EV * trade_freq^exp
        composite = ev_matrix * np.power(trade_freq, self.trade_exponent)

        opportunities = []
        for i in range(len(x_labels)):
            for j in range(len(time_labels)):
                if counts[i, j] >= 10:
                    # Get avg win/loss from surface data for Kelly NO calculation
                    key = (i, j)
                    data = self.surface_data.get(key, {})
                    avg_win = np.mean(data.get("win_magnitudes", [0])) * 100 if data.get("win_magnitudes") else 0
                    avg_loss = np.mean(data.get("loss_magnitudes", [0])) * 100 if data.get("loss_magnitudes") else 0

                    opportunities.append({
                        "x_label": x_labels[i],
                        "time": time_labels[j],
                        "ev": ev_matrix[i, j],
                        "p_win": p_win_matrix[i, j],
                        "kelly": kelly_matrix[i, j],
                        "n": int(counts[i, j]),
                        "trades": int(trade_freq[i, j]),
                        "recency": recency_matrix[i, j],
                        "composite": composite[i, j],
                        "avg_win": avg_win,
                        "avg_loss": avg_loss
                    })

        # Sort by composite score descending
        opportunities.sort(key=lambda x: x["composite"], reverse=True)

        print(f"{x_header:<18} {'Time':<12} {'EV':>10} {'P(win)':>10} {'Kelly':>10} {'Samples':>8} {'Trades':>8} {'Score':>10}")
        print("-" * 104)

        for opp in opportunities[:10]:
            if opp["composite"] > 0:
                kelly_str = "MAX" if opp['kelly'] == float('inf') else f"{opp['kelly']*100:.1f}%"
                print(f"{opp['x_label']:<18} {opp['time']:<12} {opp['ev']:>+9.3f}% {opp['p_win']*100:>9.1f}% {kelly_str:>10} {opp['n']:>8} {opp['trades']:>8} {opp['composite']:>+9.2f}")

        # NO opportunities (negative YES EV = positive NO EV)
        print("\nTOP NO OPPORTUNITIES (buy NO / sell YES)")
        print("-" * 104)
        if self.mode == 'prediction':
            print(f"{'YES @':<10} {'NO @':<10} {'Time':<12} {'EV(NO)':>10} {'P(NO)':>10} {'Kelly':>10} {'Samples':>8} {'Trades':>8} {'Score':>10}")
        else:
            print(f"{x_header:<18} {'Time':<12} {'EV(short)':>10} {'P(lose)':>10} {'Kelly':>10} {'Samples':>8} {'Trades':>8} {'Score':>10}")
        print("-" * 104)

        # For NO opportunities, flip the metrics
        no_opportunities = []
        for opp in opportunities:
            ev_no = -opp['ev']  # Flip EV
            p_no = 1 - opp['p_win']  # P(NO) = 1 - P(YES)

            # Calculate Kelly for NO side
            if opp['ev'] < 0 and p_no > 0:  # Negative YES EV means positive NO EV
                # Kelly for NO: use inverted odds
                avg_win_no = opp.get('avg_loss', 0) / 100 if opp.get('avg_loss') else 0
                avg_loss_no = opp.get('avg_win', 0) / 100 if opp.get('avg_win') else 0

                if avg_win_no > 0 and avg_loss_no > 0:
                    b = avg_win_no / avg_loss_no
                    kelly_no = max(0, (p_no * b - opp['p_win']) / b)
                elif p_no == 1.0 and ev_no > 0:
                    kelly_no = float('inf')
                else:
                    kelly_no = 0

                # Composite for NO (use absolute EV, flipped)
                composite_no = ev_no * (opp['trades'] ** self.trade_exponent)

                # Calculate NO price label (complement of YES price)
                if self.mode == 'prediction' and '¢' in opp['x_label']:
                    parts = opp['x_label'].replace('¢', '').split('-')
                    yes_low, yes_high = int(parts[0]), int(parts[1])
                    no_low, no_high = 100 - yes_high, 100 - yes_low
                    no_label = f"{no_low}-{no_high}¢"
                else:
                    no_label = opp['x_label']

                if composite_no > 0:
                    no_opportunities.append({
                        "yes_label": opp['x_label'],
                        "no_label": no_label,
                        "time": opp['time'],
                        "ev_no": ev_no,
                        "p_no": p_no,
                        "kelly_no": kelly_no,
                        "n": opp['n'],
                        "trades": opp['trades'],
                        "recency": opp['recency'],
                        "composite_no": composite_no
                    })

        no_opportunities.sort(key=lambda x: x["composite_no"], reverse=True)

        for opp in no_opportunities[:10]:
            kelly_str = "MAX" if opp['kelly_no'] == float('inf') else f"{opp['kelly_no']*100:.1f}%"
            if self.mode == 'prediction':
                print(f"{opp['yes_label']:<10} {opp['no_label']:<10} {opp['time']:<12} {opp['ev_no']:>+9.3f}% {opp['p_no']*100:>9.1f}% {kelly_str:>10} {opp['n']:>8} {opp['trades']:>8} {opp['composite_no']:>+9.2f}")
            else:
                print(f"{opp['yes_label']:<18} {opp['time']:<12} {opp['ev_no']:>+9.3f}% {opp['p_no']*100:>9.1f}% {kelly_str:>10} {opp['n']:>8} {opp['trades']:>8} {opp['composite_no']:>+9.2f}")

    def export_decision_table(self, output_path, min_composite=0.1, fractional_kelly=0.25):
        """
        Export a decision table for bot consumption.

        Only includes cells where |composite| > min_composite.

        Args:
            output_path: Path to write JSON file
            min_composite: Minimum absolute composite score to include
            fractional_kelly: Kelly fraction multiplier (e.g., 0.25 for quarter Kelly)
        """
        ev_matrix, p_win_matrix, kelly_matrix, x_labels, time_labels, counts, trade_freq, recency_matrix = self.get_ev_matrix()

        if ev_matrix is None:
            print("No surface data available for export")
            return

        # Determine which buckets to use based on mode
        if self.mode == 'prediction':
            x_buckets = self.price_buckets
        else:
            x_buckets = self.distance_buckets

        # Calculate composite scores (same logic as print_ev_surface)
        # Composite = EV * trade_freq^exp
        composite_yes = ev_matrix * np.power(trade_freq, self.trade_exponent)

        cells = []

        for i in range(len(x_labels)):
            for j in range(len(time_labels)):
                if counts[i, j] < 10:  # Skip low-sample cells
                    continue

                # Get raw data for this cell
                key = (i, j)
                data = self.surface_data.get(key, {})
                avg_win_yes = np.mean(data.get("win_magnitudes", [0])) if data.get("win_magnitudes") else 0
                avg_loss_yes = np.mean(data.get("loss_magnitudes", [0])) if data.get("loss_magnitudes") else 0

                ev = ev_matrix[i, j]
                p_win = p_win_matrix[i, j]
                kelly = kelly_matrix[i, j]
                composite = composite_yes[i, j]

                # Check YES side (positive composite)
                if composite >= min_composite:
                    # Determine price/time bucket boundaries
                    x_low = x_buckets[i]
                    x_high = x_buckets[i + 1] if i + 1 < len(x_buckets) else float('inf')
                    t_low = self.time_buckets[j]
                    t_high = self.time_buckets[j + 1] if j + 1 < len(self.time_buckets) else 1.0

                    # Cap kelly at 1.0 for practical use
                    kelly_capped = 1.0 if kelly == float('inf') else min(1.0, kelly)

                    cells.append({
                        "price_bucket": [x_low, x_high],
                        "time_bucket": [t_low, t_high],
                        "side": "YES",
                        "kelly": round(kelly_capped, 4),
                        "composite": round(composite, 4),
                        "ev": round(ev, 4),
                        "p_win": round(p_win, 4),
                        "samples": int(counts[i, j]),
                        "trades": int(trade_freq[i, j])
                    })

                # Check NO side (negative EV for YES = positive EV for NO)
                if ev < 0:
                    p_no = 1 - p_win
                    ev_no = -ev

                    # Calculate Kelly for NO side
                    avg_win_no = avg_loss_yes  # YES loss = NO win
                    avg_loss_no = avg_win_yes  # YES win = NO loss

                    if avg_win_no > 0 and avg_loss_no > 0:
                        b = avg_win_no / avg_loss_no
                        kelly_no = max(0, (p_no * b - p_win) / b)
                    elif p_no == 1.0 and ev_no > 0:
                        kelly_no = float('inf')
                    else:
                        kelly_no = 0

                    composite_no = ev_no * (trade_freq[i, j] ** self.trade_exponent)

                    if composite_no >= min_composite:
                        x_low = x_buckets[i]
                        x_high = x_buckets[i + 1] if i + 1 < len(x_buckets) else float('inf')
                        t_low = self.time_buckets[j]
                        t_high = self.time_buckets[j + 1] if j + 1 < len(self.time_buckets) else 1.0

                        kelly_no_capped = 1.0 if kelly_no == float('inf') else min(1.0, kelly_no)

                        cells.append({
                            "price_bucket": [x_low, x_high],
                            "time_bucket": [t_low, t_high],
                            "side": "NO",
                            "kelly": round(kelly_no_capped, 4),
                            "composite": round(composite_no, 4),
                            "ev": round(ev_no, 4),
                            "p_win": round(p_no, 4),
                            "samples": int(counts[i, j]),
                            "trades": int(trade_freq[i, j])
                        })

        # Sort by composite score descending
        cells.sort(key=lambda x: x["composite"], reverse=True)

        output = {
            "generated": datetime.now().isoformat(),
            "mode": self.mode,
            "window_minutes": self.window_minutes,
            "min_composite_threshold": min_composite,
            "fractional_kelly": fractional_kelly,
            "decay_half_life_days": self.decay_half_life_days,
            "total_cells": len(cells),
            "cells": cells
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nExported {len(cells)} cells to {output_path}")
        print(f"  Min composite threshold: {min_composite}")
        print(f"  Fractional Kelly: {fractional_kelly}")

        # Summary by side
        yes_cells = [c for c in cells if c["side"] == "YES"]
        no_cells = [c for c in cells if c["side"] == "NO"]
        print(f"  YES cells: {len(yes_cells)}")
        print(f"  NO cells: {len(no_cells)}")

    def print_surface(self):
        """Print the probability surface as a table."""
        matrix, x_labels, time_labels, counts, trade_freq = self.get_surface_matrix()

        print("\n" + "="*80)
        if self.mode == 'prediction':
            print("PROBABILITY SURFACE: P(YES wins) given YES price and time elapsed")
            x_header = "YES Price"
        else:
            print("PROBABILITY SURFACE: P(leader wins) given distance and time elapsed")
            x_header = "Distance"
        print("="*80)
        print(f"Window size: {self.window_minutes} minutes")
        print(f"Total windows analyzed: {len(self.windows)}")
        print()

        # Header
        print(f"{x_header:<15}", end="")
        for tl in time_labels:
            print(f"{tl:>10}", end="")
        print()
        print("-" * (15 + 10 * len(time_labels)))

        # Rows
        for i, xl in enumerate(x_labels):
            print(f"{xl:<15}", end="")
            for j in range(len(time_labels)):
                prob = matrix[i, j]
                count = counts[i, j]
                if count > 0:
                    print(f"{prob*100:>9.1f}%", end="")
                else:
                    print(f"{'--':>10}", end="")
            print()

        print()
        if self.mode == 'prediction':
            print("Interpretation:")
            print("  - P(YES) > YES_price = positive EV buying YES")
            print("  - P(YES) < YES_price = positive EV buying NO")
            print("  - P(YES) ≈ YES_price = fairly priced, no edge")
        else:
            print("Interpretation:")
            print("  - High % at small distance = mean reversion (moves don't hold)")
            print("  - Low % at small distance = momentum (small moves continue)")
            print("  - High % at large distance = trends persist")
            print("  - Values near 50% = random / no edge")

    def plot_surface(self, output_path=None):
        """Plot the probability surface as a heatmap."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, cannot plot")
            return

        matrix, x_labels, time_labels, counts, trade_freq = self.get_surface_matrix(for_plot=True)

        # Apply zoom if set
        x_start, x_end = 0, len(x_labels)
        time_start, time_end = 0, len(time_labels)

        if self.plot_distance_range and self.mode != 'prediction':
            min_dist, max_dist = self.plot_distance_range
            for i, edge in enumerate(self.distance_buckets[:-1]):
                if edge >= min_dist and x_start == 0:
                    x_start = i
                if edge <= max_dist:
                    x_end = i + 1

        if self.plot_time_range:
            min_time, max_time = self.plot_time_range
            time_start = int(min_time * 10)
            time_end = int(max_time * 10)

        # Slice the data
        matrix = matrix[x_start:x_end, time_start:time_end]
        counts = counts[x_start:x_end, time_start:time_end]
        trade_freq = trade_freq[x_start:x_end, time_start:time_end]
        x_labels = x_labels[x_start:x_end]
        time_labels = time_labels[time_start:time_end]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

        # Format window size for title
        window_str = format_window_size(self.window_minutes)

        # Set labels based on mode
        if self.mode == 'prediction':
            y_label = 'YES Price (cents)'
            prob_title = 'P(YES Wins) - Prediction Market'
        else:
            y_label = 'Distance from Open'
            prob_title = f'P(Leader Wins) - {window_str} windows'

        # Heatmap of probabilities
        im1 = ax1.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax1.set_xticks(range(len(time_labels)))
        ax1.set_xticklabels(time_labels, rotation=45, ha='right')
        ax1.set_yticks(range(len(x_labels)))
        ax1.set_yticklabels(x_labels)
        ax1.set_xlabel('Time Elapsed (% of window)')
        ax1.set_ylabel(y_label)
        ax1.set_title(prob_title)
        plt.colorbar(im1, ax=ax1, label='Probability')

        # Add text annotations
        for i in range(len(x_labels)):
            for j in range(len(time_labels)):
                if counts[i, j] > 0:
                    text = f'{matrix[i,j]*100:.0f}%'
                    color = 'white' if matrix[i,j] < 0.3 or matrix[i,j] > 0.7 else 'black'
                    ax1.text(j, i, text, ha='center', va='center', color=color, fontsize=8)

        # Heatmap of sample counts (statistical reliability)
        im2 = ax2.imshow(counts, aspect='auto', cmap='Blues')
        ax2.set_xticks(range(len(time_labels)))
        ax2.set_xticklabels(time_labels, rotation=45, ha='right')
        ax2.set_yticks(range(len(x_labels)))
        ax2.set_yticklabels(x_labels)
        ax2.set_xlabel('Time Elapsed (% of window)')
        ax2.set_ylabel(y_label)
        ax2.set_title('Sample Count (statistical reliability)')
        plt.colorbar(im2, ax=ax2, label='Samples')

        # Add text annotations for sample counts
        max_count = np.max(counts)
        for i in range(len(x_labels)):
            for j in range(len(time_labels)):
                if counts[i, j] > 0:
                    text = f'{int(counts[i, j])}'
                    color = 'white' if counts[i, j] > max_count * 0.5 else 'black'
                    ax2.text(j, i, text, ha='center', va='center', color=color, fontsize=7)

        # Heatmap of trade frequency (unique windows)
        im3 = ax3.imshow(trade_freq, aspect='auto', cmap='Oranges')
        ax3.set_xticks(range(len(time_labels)))
        ax3.set_xticklabels(time_labels, rotation=45, ha='right')
        ax3.set_yticks(range(len(x_labels)))
        ax3.set_yticklabels(x_labels)
        ax3.set_xlabel('Time Elapsed (% of window)')
        ax3.set_ylabel(y_label)
        ax3.set_title(f'Trade Opportunities ({len(self.windows)} total windows)')
        plt.colorbar(im3, ax=ax3, label='Unique Windows')

        # Add text annotations for trade frequency
        max_trades = np.max(trade_freq)
        for i in range(len(x_labels)):
            for j in range(len(time_labels)):
                if trade_freq[i, j] > 0:
                    text = f'{int(trade_freq[i, j])}'
                    color = 'white' if trade_freq[i, j] > max_trades * 0.5 else 'black'
                    ax3.text(j, i, text, ha='center', va='center', color=color, fontsize=7)

        plt.tight_layout()

        if output_path:
            # Auto-add .png extension if not present
            if not output_path.endswith('.png'):
                output_path = output_path + '.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_ev_surface(self, output_path=None):
        """Plot the EV surface as a heatmap."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, cannot plot")
            return

        ev_matrix, p_win_matrix, kelly_matrix, x_labels, time_labels, counts, trade_freq, recency_matrix = self.get_ev_matrix()

        if ev_matrix is None:
            print("No surface data available")
            return

        # Apply zoom if set
        x_start, x_end = 0, len(x_labels)
        time_start, time_end = 0, len(time_labels)

        if self.plot_distance_range and self.mode != 'prediction':
            min_dist, max_dist = self.plot_distance_range
            for i, edge in enumerate(self.distance_buckets[:-1]):
                if edge >= min_dist and x_start == 0:
                    x_start = i
                if edge <= max_dist:
                    x_end = i + 1

        if self.plot_time_range:
            min_time, max_time = self.plot_time_range
            time_start = int(min_time * 10)
            time_end = int(max_time * 10)

        # Slice
        ev_matrix = ev_matrix[x_start:x_end, time_start:time_end]
        kelly_matrix = kelly_matrix[x_start:x_end, time_start:time_end]
        counts = counts[x_start:x_end, time_start:time_end]
        trade_freq = trade_freq[x_start:x_end, time_start:time_end]
        recency_matrix = recency_matrix[x_start:x_end, time_start:time_end]
        x_labels = x_labels[x_start:x_end]
        time_labels = time_labels[time_start:time_end]

        # Add human-friendly time to labels
        plot_time_labels = []
        for i, tl in enumerate(time_labels):
            idx = time_start + i
            if idx < len(self.time_buckets) - 1:
                mid_pct = (self.time_buckets[idx] + self.time_buckets[idx + 1]) / 2
                elapsed_str = format_time_elapsed(self.window_minutes, mid_pct)
                plot_time_labels.append(f"{tl}\n({elapsed_str})")
            else:
                plot_time_labels.append(tl)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        window_str = format_window_size(self.window_minutes)

        # Set labels based on mode
        if self.mode == 'prediction':
            y_label = 'YES Price (cents)'
            ev_title = 'Expected Value (%) - Prediction Market\nGreen = buy YES, Red = buy NO'
        else:
            y_label = 'Distance from Open'
            ev_title = f'Expected Value (%) - {window_str} windows\nGreen = positive EV, Red = negative'

        # Mask cells with insufficient data
        ev_masked = np.ma.masked_where(counts < 10, ev_matrix)

        # EV heatmap - use diverging colormap centered at 0
        max_abs_ev = max(abs(np.nanmin(ev_matrix)), abs(np.nanmax(ev_matrix)), 0.01)
        im1 = ax1.imshow(ev_masked, aspect='auto', cmap='RdYlGn', vmin=-max_abs_ev, vmax=max_abs_ev)
        ax1.set_xticks(range(len(plot_time_labels)))
        ax1.set_xticklabels(plot_time_labels, rotation=45, ha='right', fontsize=8)
        ax1.set_yticks(range(len(x_labels)))
        ax1.set_yticklabels(x_labels)
        ax1.set_xlabel('Time Elapsed')
        ax1.set_ylabel(y_label)
        ax1.set_title(ev_title)
        plt.colorbar(im1, ax=ax1, label='EV (%)')

        # Add text annotations
        for i in range(len(x_labels)):
            for j in range(len(time_labels)):
                if counts[i, j] >= 10:
                    ev = ev_matrix[i, j]
                    text = f'{ev:+.2f}%' if abs(ev) >= 0.005 else '0'
                    color = 'white' if abs(ev) > max_abs_ev * 0.5 else 'black'
                    ax1.text(j, i, text, ha='center', va='center', color=color, fontsize=7)

        # Kelly fraction heatmap - show both YES (positive) and NO (negative) sides
        # Calculate Kelly for both sides: positive for YES, negative for NO
        kelly_both = np.zeros_like(kelly_matrix)
        for i in range(kelly_matrix.shape[0]):
            for j in range(kelly_matrix.shape[1]):
                if counts[i, j] < 10 or abs(ev_matrix[i, j]) < 0.005:
                    continue

                ev = ev_matrix[i, j]
                if ev >= 0:
                    # YES side - use existing Kelly
                    k = kelly_matrix[i, j]
                    kelly_both[i, j] = 1.0 if np.isinf(k) else k
                else:
                    # NO side - calculate Kelly for NO
                    key = (i, j)
                    data = self.surface_data.get(key, {})
                    p_no = 1 - p_win_matrix[i, j]

                    avg_win_no = np.mean(data.get("loss_magnitudes", [0])) if data.get("loss_magnitudes") else 0
                    avg_loss_no = np.mean(data.get("win_magnitudes", [0])) if data.get("win_magnitudes") else 0

                    if avg_win_no > 0 and avg_loss_no > 0:
                        b = avg_win_no / avg_loss_no
                        kelly_no = max(0, (p_no * b - (1 - p_no)) / b)
                    elif p_no == 1.0:
                        kelly_no = 1.0
                    else:
                        kelly_no = 0

                    # Store as negative to indicate NO side
                    kelly_both[i, j] = -min(1.0, kelly_no)

        kelly_masked = np.ma.masked_where((counts < 10) | (np.abs(ev_matrix) < 0.005) | (np.abs(kelly_both) < 0.01), kelly_both)
        max_kelly = max(abs(np.nanmin(kelly_both)), abs(np.nanmax(kelly_both)), 0.5)
        im2 = ax2.imshow(kelly_masked, aspect='auto', cmap='RdYlGn', vmin=-max_kelly, vmax=max_kelly)
        ax2.set_xticks(range(len(plot_time_labels)))
        ax2.set_xticklabels(plot_time_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_yticks(range(len(x_labels)))
        ax2.set_yticklabels(x_labels)
        ax2.set_xlabel('Time Elapsed')
        ax2.set_ylabel(y_label)
        if self.mode == 'prediction':
            ax2.set_title(f'Kelly Fraction (Green=YES, Red=NO)\n|value| = bet size, sign = direction')
        else:
            ax2.set_title(f'Kelly Fraction (optimal bet size)\n0 = no bet, 0.5 = bet 50% of bankroll')
        plt.colorbar(im2, ax=ax2, label='Kelly (+YES / -NO)')

        # Add text annotations for Kelly
        for i in range(len(x_labels)):
            for j in range(len(time_labels)):
                if counts[i, j] >= 10 and abs(ev_matrix[i, j]) >= 0.005:
                    k = kelly_both[i, j]
                    if abs(k) >= 0.99:
                        text = 'MAX' if k > 0 else '-MAX'
                        color = 'white'
                    elif abs(k) > 0.01:
                        text = f'{k*100:+.0f}%'
                        color = 'white' if abs(k) > 0.25 else 'black'
                    else:
                        continue
                    ax2.text(j, i, text, ha='center', va='center', color=color, fontsize=7)

        # Composite score: EV * trade_freq^exp
        composite = np.zeros_like(ev_matrix)

        for i in range(ev_matrix.shape[0]):
            for j in range(ev_matrix.shape[1]):
                if counts[i, j] < 10:
                    continue

                ev = ev_matrix[i, j]
                tf = trade_freq[i, j]

                if ev >= 0:
                    # YES side
                    composite[i, j] = ev * (tf ** self.trade_exponent)
                else:
                    # NO side - store as negative to indicate NO side
                    ev_no = -ev  # Flip to positive
                    composite[i, j] = -ev_no * (tf ** self.trade_exponent)

        # Mask cells with insufficient data or zero composite
        composite_masked = np.ma.masked_where((counts < 10) | (np.abs(composite) < 0.001), composite)

        # Composite heatmap
        max_abs_composite = max(abs(np.nanmin(composite)), abs(np.nanmax(composite)), 0.01)
        im3 = ax3.imshow(composite_masked, aspect='auto', cmap='RdYlGn', vmin=-max_abs_composite, vmax=max_abs_composite)
        ax3.set_xticks(range(len(plot_time_labels)))
        ax3.set_xticklabels(plot_time_labels, rotation=45, ha='right', fontsize=8)
        ax3.set_yticks(range(len(x_labels)))
        ax3.set_yticklabels(x_labels)
        ax3.set_xlabel('Time Elapsed')
        ax3.set_ylabel(y_label)
        exp_str = "√" if self.trade_exponent == 0.5 else f"^{self.trade_exponent}"
        if self.mode == 'prediction':
            ax3.set_title(f'Composite Score (Green=YES, Red=NO)\nEV × trades{exp_str}')
        else:
            ax3.set_title(f'Composite Score\nEV × trades{exp_str}')
        plt.colorbar(im3, ax=ax3, label='Score (+YES / -NO)')

        # Add text annotations for composite
        for i in range(len(x_labels)):
            for j in range(len(time_labels)):
                if counts[i, j] >= 10:
                    score = composite[i, j]
                    if abs(score) >= 0.001:
                        text = f'{score:+.2f}'
                        color = 'white' if abs(score) > max_abs_composite * 0.5 else 'black'
                        ax3.text(j, i, text, ha='center', va='center', color=color, fontsize=7)

        plt.tight_layout()

        if output_path:
            # Save with _ev suffix
            if output_path.endswith('.png'):
                ev_path = output_path.replace('.png', '_ev.png')
            else:
                ev_path = output_path + '_ev.png'
            plt.savefig(ev_path, dpi=150, bbox_inches='tight')
            print(f"Saved EV plot to {ev_path}")
        else:
            plt.show()

        plt.close()

    def find_breakout_threshold(self, time_fraction=0.8, target_prob=0.7):
        """
        Find the distance threshold where continuation probability exceeds target.

        Args:
            time_fraction: How far into the window (0.8 = 80% elapsed)
            target_prob: Target probability for "breakout" classification

        Returns:
            Minimum distance (as %) where P(continuation) >= target_prob
        """
        time_bucket = self.get_time_bucket(time_fraction)

        for i, threshold in enumerate(self.distance_buckets[1:]):
            if threshold == float('inf'):
                continue

            key = (i, time_bucket)
            data = self.surface_data[key]
            total = data["yes"] + data["no"]

            if total > 10:  # Need enough samples
                prob = data["yes"] / total
                if prob >= target_prob:
                    return threshold * 100  # Return as percentage

        return None

    def analyze_for_trading(self):
        """Generate trading insights from the surface."""
        print("\n" + "="*60)
        print("TRADING ANALYSIS")
        print("="*60)

        if self.mode == 'prediction':
            # Prediction market analysis
            print("\n1. MISPRICING ZONES (where market price differs from actual probability):")

            for t_frac in [0.3, 0.5, 0.7, 0.9]:
                t_bucket = self.get_time_bucket(t_frac)
                mispricings = []

                for p_bucket in range(len(self.price_buckets) - 1):
                    key = (p_bucket, t_bucket)
                    data = self.surface_data[key]
                    total = data["yes"] + data["no"]

                    if total > 20:
                        actual_prob = data["yes"] / total
                        # Market price is midpoint of bucket
                        market_price = (self.price_buckets[p_bucket] + self.price_buckets[p_bucket + 1]) / 2 / 100
                        edge = actual_prob - market_price

                        if abs(edge) > 0.05:  # 5% edge
                            direction = "BUY YES" if edge > 0 else "BUY NO"
                            mispricings.append(f"{self.price_buckets[p_bucket]}-{self.price_buckets[p_bucket+1]}¢ ({direction}, {abs(edge)*100:.0f}% edge)")

                if mispricings:
                    print(f"   At T={t_frac*100:.0f}%: {', '.join(mispricings)}")

            print(f"\n2. OVERALL STATISTICS:")
            print(f"   Total markets: {len(self.windows)}")

        else:
            # Real market analysis
            print("\n1. MEAN REVERSION ZONES (where small moves tend to fail):")
            for t_frac in [0.3, 0.5, 0.7, 0.9]:
                t_bucket = self.get_time_bucket(t_frac)

                reversion_zones = []
                for d_bucket in range(3):  # Small distances
                    key = (d_bucket, t_bucket)
                    data = self.surface_data[key]
                    total = data["yes"] + data["no"]

                    if total > 20:
                        prob = data["yes"] / total
                        if prob < 0.55:  # Leader loses often
                            low = self.distance_buckets[d_bucket] * 100
                            high = self.distance_buckets[d_bucket + 1] * 100
                            reversion_zones.append(f"{low:.2f}-{high:.2f}% (P={prob*100:.0f}%)")

                if reversion_zones:
                    print(f"   At T={t_frac*100:.0f}%: {', '.join(reversion_zones)}")

            # Find breakout thresholds
            print("\n2. BREAKOUT THRESHOLDS (where moves become likely to hold):")
            for t_frac in [0.5, 0.7, 0.9]:
                threshold = self.find_breakout_threshold(t_frac, 0.65)
                if threshold:
                    print(f"   At T={t_frac*100:.0f}%: move >{threshold:.2f}% likely to hold")

            # Overall statistics
            print(f"\n3. OVERALL STATISTICS:")
            print(f"   Total windows: {len(self.windows)}")
            if self.windows:
                yes_count = sum(1 for w in self.windows if w.get('outcome') == 'YES')
                print(f"   YES outcomes: {yes_count} ({100*yes_count/len(self.windows):.1f}%)")
                print(f"   NO outcomes: {len(self.windows) - yes_count} ({100*(len(self.windows)-yes_count)/len(self.windows):.1f}%)")

            # Calculate average move size
            if self.windows and self.windows[0].get('open_price'):
                moves = [abs(w['close_price'] - w['open_price']) / w['open_price'] * 100
                         for w in self.windows if w.get('open_price')]
                if moves:
                    print(f"   Average move size: {np.mean(moves):.3f}%")
                    print(f"   Median move size: {np.median(moves):.3f}%")
                    print(f"   Max move size: {np.max(moves):.3f}%")

    def save_surface(self, filepath):
        """Save the surface data to JSON."""
        # Auto-add .json extension if not present
        if not filepath.endswith('.json'):
            filepath = filepath + '.json'

        # Convert defaultdict to regular dict with string keys
        # Convert sets to lists for JSON serialization
        surface_serializable = {}
        for k, v in self.surface_data.items():
            surface_serializable[f"{k[0]},{k[1]}"] = {
                "yes": v["yes"],
                "no": v["no"],
                "win_magnitudes": v["win_magnitudes"],
                "loss_magnitudes": v["loss_magnitudes"],
                "windows_yes": list(v["windows_yes"]),
                "windows_no": list(v["windows_no"])
            }

        data = {
            "window_minutes": self.window_minutes,
            "distance_buckets": self.distance_buckets,
            "time_buckets": self.time_buckets,
            "confidence_k": self.confidence_k,
            "trade_exponent": self.trade_exponent,
            "recency_half_life": self.recency_half_life,
            "surface": surface_serializable,
            "total_windows": len(self.windows),
            "created": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved surface to {filepath}")

    def export_llm_analysis(self, filepath):
        """
        Export a JSON file designed for LLM parsing with all computed metrics.
        Contains human-readable analysis of each bucket.
        """
        # Auto-add .json extension if not present
        if not filepath.endswith('.json'):
            filepath = filepath + '.json'

        window_str = format_window_size(self.window_minutes)

        # Build the analysis for each bucket
        buckets = []
        for (dist_bucket, time_bucket), data in self.surface_data.items():
            total_samples = data["yes"] + data["no"]
            if total_samples == 0:
                continue

            # Get bucket ranges
            dist_low = self.distance_buckets[dist_bucket] * 100
            dist_high = self.distance_buckets[dist_bucket + 1] * 100 if dist_bucket + 1 < len(self.distance_buckets) else float('inf')

            time_low = self.time_buckets[time_bucket]
            time_high = self.time_buckets[time_bucket + 1] if time_bucket + 1 < len(self.time_buckets) else 1.0

            # Calculate metrics
            p_win = data["yes"] / total_samples
            p_loss = 1 - p_win

            avg_win = np.mean(data["win_magnitudes"]) * 100 if data["win_magnitudes"] else 0
            avg_loss = np.mean(data["loss_magnitudes"]) * 100 if data["loss_magnitudes"] else 0

            ev = p_win * (avg_win / 100) - p_loss * (avg_loss / 100)
            ev_pct = ev * 100

            # Kelly
            if avg_win > 0 and avg_loss > 0:
                b = avg_win / avg_loss
                kelly = max(0, (p_win * b - p_loss) / b)
            elif p_win == 1.0 and ev > 0:
                kelly = "MAX"
            else:
                kelly = 0

            # Trade frequency and recency
            all_windows = data["windows_yes"] | data["windows_no"]
            trade_freq = len(all_windows)
            total_windows = len(self.windows) if self.windows else 1
            recency = self.get_recency_score(all_windows, total_windows)

            # Composite = EV * trade_freq^exp
            composite = ev_pct * (trade_freq ** self.trade_exponent)

            # Time in human-readable format
            time_mid = (time_low + time_high) / 2
            elapsed_min = self.window_minutes * time_mid

            bucket_data = {
                "distance_range": f"{dist_low:.2f}% - {dist_high:.2f}%" if dist_high != float('inf') else f">{dist_low:.2f}%",
                "distance_bucket": dist_bucket,
                "time_range": f"{time_low*100:.0f}% - {time_high*100:.0f}%",
                "time_elapsed_minutes": f"{self.window_minutes * time_low:.1f}m - {self.window_minutes * time_high:.1f}m",
                "time_bucket": time_bucket,
                "p_win": round(p_win * 100, 2),
                "p_loss": round(p_loss * 100, 2),
                "avg_win_pct": round(avg_win, 4),
                "avg_loss_pct": round(avg_loss, 4),
                "ev_pct": round(ev_pct, 4),
                "kelly": kelly if kelly == "MAX" else round(kelly * 100, 2),
                "kelly_unit": "MAX" if kelly == "MAX" else "percent",
                "samples": total_samples,
                "trade_opportunities": trade_freq,
                "recency": round(recency * 100, 2),
                "composite_score": round(composite, 4),
                "confidence_weight": round(confidence * 100, 2),
                "wins": data["yes"],
                "losses": data["no"]
            }
            buckets.append(bucket_data)

        # Sort by composite score descending
        buckets.sort(key=lambda x: x["composite_score"], reverse=True)

        # Summary statistics
        total_windows = len(self.windows)
        yes_outcomes = sum(1 for w in self.windows if w.get('outcome') == 'YES') if self.windows else 0

        # Top opportunities (positive composite)
        top_opportunities = [b for b in buckets if b["composite_score"] > 0][:10]

        # Worst setups (negative composite)
        worst_setups = [b for b in buckets if b["composite_score"] < 0]
        worst_setups.sort(key=lambda x: x["composite_score"])
        worst_setups = worst_setups[:5]

        output = {
            "metadata": {
                "description": "Price-to-probability analysis for LLM parsing",
                "window_size_minutes": self.window_minutes,
                "window_size_human": window_str,
                "total_windows_analyzed": total_windows,
                "yes_outcomes": yes_outcomes,
                "no_outcomes": total_windows - yes_outcomes,
                "composite_formula": f"EV × trades^{self.trade_exponent}",
                "decay_half_life_days": self.decay_half_life_days,
                "trade_exponent": self.trade_exponent,
                "created": datetime.now().isoformat()
            },
            "interpretation_guide": {
                "p_win": "Probability that price closes on the same side of open as current position (leader wins)",
                "ev_pct": "Expected value per trade as percentage of position size",
                "kelly": "Optimal bet size as percentage of bankroll (MAX = 100% win rate)",
                "samples": "Number of data points observed (higher = more statistically reliable)",
                "trade_opportunities": "Unique windows that hit this bucket (actual tradeable setups)",
                "composite_score": "Combined score ranking opportunities by EV scaled by trade frequency"
            },
            "top_opportunities": top_opportunities,
            "worst_setups": worst_setups,
            "all_buckets": buckets
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Exported LLM analysis to {filepath}")
        return filepath

    def load_surface(self, filepath):
        """Load a previously saved surface."""
        # Auto-add .json extension if not present
        if not filepath.endswith('.json'):
            filepath = filepath + '.json'

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.window_minutes = data["window_minutes"]
        self.distance_buckets = data["distance_buckets"]
        self.time_buckets = data["time_buckets"]
        self.confidence_k = data.get("confidence_k", 30)  # Default for old files
        self.trade_exponent = data.get("trade_exponent", 0.5)  # Default for old files
        self.recency_half_life = data.get("recency_half_life", 50)  # Default for old files

        self.surface_data = defaultdict(lambda: {
            "yes": 0, "no": 0,
            "win_magnitudes": [], "loss_magnitudes": [],
            "windows_yes": set(), "windows_no": set()
        })
        for key_str, value in data["surface"].items():
            parts = key_str.split(",")
            key = (int(parts[0]), int(parts[1]))
            # Handle both old format (just yes/no) and new format (with windows)
            self.surface_data[key] = {
                "yes": value.get("yes", 0),
                "no": value.get("no", 0),
                "win_magnitudes": value.get("win_magnitudes", []),
                "loss_magnitudes": value.get("loss_magnitudes", []),
                "windows_yes": set(value.get("windows_yes", [])),
                "windows_no": set(value.get("windows_no", []))
            }

        print(f"Loaded surface from {filepath}")
        print(f"  Window size: {self.window_minutes} min")
        print(f"  Total windows: {data['total_windows']}")


def format_time_elapsed(window_minutes, time_fraction):
    """Format time elapsed in human-readable units."""
    elapsed_minutes = window_minutes * time_fraction

    if window_minutes < 60:
        return f"{elapsed_minutes:.1f}m"
    elif window_minutes < 1440:  # Less than a day
        hours = elapsed_minutes / 60
        return f"{hours:.1f}h"
    else:
        days = elapsed_minutes / 1440
        return f"{days:.2f}d"


def format_window_size(window_minutes):
    """Format window size in human-readable units."""
    if window_minutes < 60:
        return f"{window_minutes}m"
    elif window_minutes < 1440:
        hours = window_minutes / 60
        if hours == int(hours):
            return f"{int(hours)}h"
        return f"{hours:.1f}h"
    else:
        days = window_minutes / 1440
        if days == int(days):
            return f"{int(days)}d"
        return f"{days:.1f}d"


def parse_time_input(time_str, window_minutes):
    """
    Parse human-friendly time input and convert to fraction of window.

    Accepts:
        - Percentage: "80", "80%", "50%"
        - Minutes: "12m", "12min", "12 min"
        - Hours: "2h", "2hr", "2 hours"
        - Days: "0.5d", "1day", "2 days"

    Returns:
        time_fraction (0-1)
    """
    time_str = time_str.strip().lower()

    # Check for percentage (default if just a number or ends with %)
    if time_str.endswith('%'):
        return float(time_str[:-1]) / 100

    # Check for minutes
    if 'm' in time_str and 'mo' not in time_str:  # 'm', 'min', 'mins', 'minute', 'minutes'
        val = float(''.join(c for c in time_str if c.isdigit() or c == '.'))
        return val / window_minutes

    # Check for hours
    if 'h' in time_str:  # 'h', 'hr', 'hrs', 'hour', 'hours'
        val = float(''.join(c for c in time_str if c.isdigit() or c == '.'))
        return (val * 60) / window_minutes

    # Check for days
    if 'd' in time_str:  # 'd', 'day', 'days'
        val = float(''.join(c for c in time_str if c.isdigit() or c == '.'))
        return (val * 1440) / window_minutes

    # Default: assume percentage
    try:
        return float(time_str) / 100
    except ValueError:
        raise ValueError(f"Could not parse time: {time_str}")


def interactive_query(p2p):
    """Interactive mode to query probabilities."""
    window_str = format_window_size(p2p.window_minutes)

    print("\n" + "="*60)
    print("INTERACTIVE QUERY MODE")
    print(f"Window size: {window_str} ({p2p.window_minutes} minutes)")
    print("="*60)
    print("Enter: distance% time")
    print("  Distance: percentage from open (e.g., 0.5, 1.2)")
    print("  Time: percentage OR human format")
    print(f"    - '80' or '80%' = 80% of {window_str}")
    if p2p.window_minutes <= 60:
        print(f"    - '12m' or '12min' = 12 minutes into window")
    if p2p.window_minutes >= 60:
        print(f"    - '2h' or '2hr' = 2 hours into window")
    if p2p.window_minutes >= 1440:
        print(f"    - '0.5d' or '1day' = days into window")
    print("Type 'q' to quit")
    print("="*60 + "\n")

    while True:
        try:
            inp = input("Distance% Time: ").strip()
            if inp.lower() == 'q':
                break

            parts = inp.split(None, 1)  # Split into max 2 parts
            if len(parts) != 2:
                print("Enter two values: distance% and time")
                continue

            distance_pct = float(parts[0].replace('%', ''))
            time_frac = parse_time_input(parts[1], p2p.window_minutes)

            # Clamp time_frac to valid range
            if time_frac < 0 or time_frac > 1:
                print(f"  Warning: time {parts[1]} = {time_frac*100:.1f}% is outside 0-100% range")
                time_frac = max(0, min(1, time_frac))

            # Convert distance to decimal
            distance = distance_pct / 100

            # Get probability and EV
            prob = p2p.get_probability(distance, time_frac)
            ev_data = p2p.get_ev(distance, time_frac)

            # Show time in actual units
            time_str = format_time_elapsed(p2p.window_minutes, time_frac)
            remaining_str = format_time_elapsed(p2p.window_minutes, 1 - time_frac)
            time_pct = time_frac * 100

            kelly_str = "MAX" if ev_data['kelly'] == float('inf') else f"{ev_data['kelly']*100:.1f}%"

            # Calculate composite score = EV * trade_freq^exp * confidence
            # Kelly removed (use for sizing only), recency removed (baked into time-weighted P(win))
            # Composite = EV * trade_freq^exp
            composite = ev_data['ev'] * (ev_data['trades'] ** p2p.trade_exponent)

            print(f"  Window: {window_str} | Elapsed: {time_str} ({time_pct:.0f}%) | Remaining: {remaining_str}")
            print(f"  P(continuation) = {prob*100:.1f}%")
            print(f"  Avg win: +{ev_data['avg_win']:.3f}% | Avg loss: -{ev_data['avg_loss']:.3f}%")
            print(f"  EV: {ev_data['ev']:+.4f}% | Kelly: {kelly_str} | Samples: {ev_data['n']} | Trades: {ev_data['trades']}")
            print(f"  Composite Score: {composite:+.3f}")

            if ev_data['ev'] > 0.01:
                print(f"  → POSITIVE EV - tradeable edge")
            elif ev_data['ev'] < -0.01:
                print(f"  → NEGATIVE EV - avoid this setup")
            else:
                print(f"  → NEUTRAL - no clear edge")

            print()

        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            break


def fetch_crypto_data(symbol='BTC', days=30, interval='1m'):
    """
    Fetch crypto data from a free API.
    Uses CoinGecko or falls back to generating sample data.
    """
    import urllib.request
    import ssl

    print(f"Fetching {symbol} data ({days} days, {interval} interval)...")

    # For 1-minute data, we'd need a different source
    # CoinGecko free tier only has hourly for recent data
    # Let's try Binance public API for 1m candles

    try:
        # Binance public API - no auth needed
        # Max 1000 candles per request
        base_url = "https://api.binance.com/api/v3/klines"

        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)

        # Fetch in chunks (1000 candles = ~16.6 hours of 1m data)
        candles_needed = days * 24 * 60  # 1 candle per minute
        chunks = (candles_needed // 1000) + 1

        for i in range(min(chunks, 30)):  # Cap at 30 chunks (~20 days of 1m data)
            params = f"symbol={symbol}USDT&interval={interval}&limit=1000"
            if all_data:
                # Get data before the earliest we have
                earliest = all_data[0][0]
                params += f"&endTime={earliest - 1}"

            url = f"{base_url}?{params}"

            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                raw = json.loads(response.read().decode())

            if not raw:
                break

            all_data = raw + all_data
            print(f"  Fetched chunk {i+1}/{chunks} ({len(all_data)} candles total)")

            if len(raw) < 1000:
                break  # No more data

        # Convert to our format
        # Binance format: [open_time, open, high, low, close, volume, close_time, ...]
        data = []
        for candle in all_data:
            ts = datetime.fromtimestamp(candle[0] / 1000)
            close_price = float(candle[4])
            data.append((ts, close_price))

        print(f"Fetched {len(data)} data points")
        return data

    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Generating synthetic data instead...")
        return generate_synthetic_data(days * 24 * 60)


def fetch_yfinance_data(symbol, period='7d', interval='1m'):
    """
    Fetch data from Yahoo Finance using yfinance.

    Args:
        symbol: Ticker symbol (e.g., 'SPY', 'AAPL', 'BTC-USD')
        period: How far back to fetch ('1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max')
        interval: Candle interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

    Note: 1m data only available for last 7 days, 2m-90m for last 60 days
    """
    if not HAS_YFINANCE:
        print("yfinance not installed. Run: pip install yfinance")
        return None

    # Auto-adjust incompatible period/interval combinations
    original_period = period
    original_interval = interval

    # 1m data: max 7 days
    if interval == '1m' and period not in ['1d', '5d', '7d']:
        print(f"Warning: 1m interval only supports up to 7 days. Adjusting period from {period} to 7d")
        period = '7d'

    # 2m-30m data: max 60 days
    if interval in ['2m', '5m', '15m', '30m'] and period in ['3mo', '6mo', '1y', '2y', '5y', '10y', 'max']:
        print(f"Warning: {interval} interval only supports up to 60 days. Adjusting period to 1mo")
        period = '1mo'

    print(f"Fetching {symbol} from Yahoo Finance (period={period}, interval={interval})...")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            print(f"No data returned for {symbol}")
            return None

        # Convert to our format
        data = []
        for idx, row in df.iterrows():
            ts = idx.to_pydatetime()
            if ts.tzinfo:
                ts = ts.replace(tzinfo=None)  # Remove timezone for simplicity
            price = row['Close']
            data.append((ts, price))

        print(f"Fetched {len(data)} data points from {data[0][0]} to {data[-1][0]}")
        return data

    except Exception as e:
        print(f"Error fetching from yfinance: {e}")
        return None


def generate_synthetic_data(n_points=10000, volatility=0.001):
    """Generate synthetic price data for testing."""
    print(f"Generating {n_points} synthetic data points...")

    data = []
    price = 100.0
    ts = datetime.now() - timedelta(minutes=n_points)

    for i in range(n_points):
        # Random walk with slight mean reversion
        change = np.random.normal(0, volatility)
        # Add occasional larger moves
        if np.random.random() < 0.01:
            change *= 5

        price *= (1 + change)
        data.append((ts, price))
        ts += timedelta(minutes=1)

    return data


def main():
    parser = argparse.ArgumentParser(description='Convert price data to probability space')
    parser.add_argument('--input', '-i', help='Input CSV file with price data')
    parser.add_argument('--yfinance', '-y', help='Fetch from Yahoo Finance (e.g., SPY, AAPL, BTC-USD, ^SPX)')
    parser.add_argument('--period', default='7d', help='yfinance period: 1d, 5d, 7d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max (default: 7d)')
    parser.add_argument('--interval', default='1m', help='yfinance interval: 1m, 5m, 15m, 30m, 1h, 1d (default: 1m). Note: 1m only last 7 days, 5m-30m last 60 days')
    parser.add_argument('--fetch', '-f', help='Fetch crypto from Binance (e.g., BTC, ETH) - may be geo-blocked')
    parser.add_argument('--days', '-d', type=int, default=7, help='Days of data for Binance/synthetic (default: 7)')
    parser.add_argument('--synthetic', '-s', action='store_true', help='Use synthetic data for testing')
    parser.add_argument('--window', '-w', type=int, default=15, help='Window size in MINUTES (default: 15). Each window becomes a synthetic prediction market.')
    parser.add_argument('--timestamp-col', default='timestamp', help='Timestamp column name')
    parser.add_argument('--price-col', default='close', help='Price column name')
    parser.add_argument('--output', '-o', help='Output path for plot')
    parser.add_argument('--save-surface', help='Save surface data to JSON')
    parser.add_argument('--load-surface', help='Load existing surface from JSON')
    parser.add_argument('--interactive', '-I', action='store_true', help='Interactive query mode')
    parser.add_argument('--compare', '-c', nargs='+', help='Compare multiple window sizes')
    parser.add_argument('--buckets', '-b', help='Custom distance buckets in %%, comma-separated (e.g., "0,0.05,0.1,0.15,0.2,0.3,0.5")')
    parser.add_argument('--zoom-dist', help='Zoom plot to distance range in %% (e.g., "0,0.3")')
    parser.add_argument('--zoom-time', help='Zoom plot to time range in %% (e.g., "50,100")')
    parser.add_argument('--confidence-k', type=float, default=30, help='Samples needed for 50%% confidence weight in composite score (default: 30)')
    parser.add_argument('--trade-exp', type=float, default=0.5, help='Exponent for trade frequency in composite score (0.5=sqrt, 1.0=linear, default: 0.5)')
    parser.add_argument('--recency-hl', type=float, default=50, help='Half-life for recency decay in windows (default: 50). Higher = slower decay.')
    parser.add_argument('--export-llm', help='Export LLM-readable analysis JSON with all computed metrics')
    parser.add_argument('--mode', choices=['real', 'prediction'], default='real',
                        help='Mode: "real" for real market trading (returns relative to entry), "prediction" for prediction markets (binary payoff). Default: real')
    parser.add_argument('--kalshi', '-k', help='Load Kalshi prediction market data from JSON file')
    parser.add_argument('--price-buckets', help='Custom YES price buckets in cents for prediction mode, comma-separated (e.g., "0,5,10,20,30,40,50,60,70,80,90,95,100")')
    parser.add_argument('--export-table', help='Export decision table JSON for bot consumption')
    parser.add_argument('--min-composite', type=float, default=0.1, help='Minimum |composite| score for export-table (default: 0.1)')
    parser.add_argument('--fractional-kelly', type=float, default=0.25, help='Fractional Kelly multiplier for export-table (default: 0.25 = quarter Kelly)')
    parser.add_argument('--decay-half-life', type=float, default=7.0,
                        help='Half-life in days for time-weighted P(win) calculation (default: 7.0). '
                             'Data from N days ago has 0.5^(N/half_life) weight. '
                             'Lower = more weight on recent data. Set to 0 to disable time weighting.')

    args = parser.parse_args()

    # Auto-set prediction mode if using Kalshi data
    mode = args.mode
    if args.kalshi and mode == 'real':
        mode = 'prediction'
        print("Auto-switching to prediction mode for Kalshi data")

    # Handle decay_half_life = 0 meaning disabled (use very large value)
    decay_hl = args.decay_half_life if args.decay_half_life > 0 else 99999.0

    p2p = PriceToProbability(window_minutes=args.window,
                              confidence_k=args.confidence_k,
                              trade_exponent=args.trade_exp,
                              recency_half_life=args.recency_hl,
                              mode=mode,
                              decay_half_life_days=decay_hl)

    if args.decay_half_life > 0:
        print(f"Time-weighted P(win) enabled with half-life of {args.decay_half_life} days")

    # Handle custom buckets
    if args.buckets:
        bucket_edges = [float(x) for x in args.buckets.split(',')]
        p2p.set_distance_buckets(bucket_edges)
        print(f"Using custom distance buckets: {bucket_edges}%")

    # Handle custom price buckets for prediction mode
    if args.price_buckets:
        price_edges = [float(x) for x in args.price_buckets.split(',')]
        p2p.price_buckets = price_edges
        print(f"Using custom price buckets: {price_edges}¢")

    # Handle zoom settings
    if args.zoom_dist:
        parts = [float(x)/100 for x in args.zoom_dist.split(',')]
        p2p.plot_distance_range = (parts[0], parts[1])
        print(f"Zoom distance: {args.zoom_dist}%")

    if args.zoom_time:
        parts = [float(x)/100 for x in args.zoom_time.split(',')]
        p2p.plot_time_range = (parts[0], parts[1])
        print(f"Zoom time: {args.zoom_time}%")

    data = None
    kalshi_loaded = False

    if args.kalshi:
        # Load Kalshi prediction market data
        p2p.load_kalshi_json(args.kalshi)
        kalshi_loaded = True
    elif args.load_surface:
        p2p.load_surface(args.load_surface)
    elif args.yfinance:
        data = fetch_yfinance_data(args.yfinance, args.period, args.interval)
        if not data:
            return
    elif args.fetch:
        data = fetch_crypto_data(args.fetch, args.days)
    elif args.synthetic:
        data = generate_synthetic_data(args.days * 24 * 60)
    elif args.input:
        print(f"Loading data from {args.input}...")
        data = p2p.load_csv(args.input, args.timestamp_col, args.price_col)
        print(f"Loaded {len(data)} data points")
    else:
        print("Price-to-Probability Translator")
        print("="*40)
        print("\nUsage examples:")
        print("  # Fetch SPY from Yahoo Finance (1-min data, last 7 days, 15-min windows)")
        print("  python price_to_probability.py -y SPY --period 7d --interval 1m --window 15 -I")
        print("")
        print("  # Fetch BTC-USD with daily candles, 1-week windows")
        print("  python price_to_probability.py -y BTC-USD --period 2y --interval 1d --window 10080 -I")
        print("  # (10080 min = 1 week)")
        print("")
        print("  # Use synthetic data for testing")
        print("  python price_to_probability.py --synthetic --window 15 -I")
        print("")
        print("  # Load from CSV")
        print("  python price_to_probability.py -i data.csv -w 15 --interactive")
        print("")
        print("  # Compare multiple timeframes")
        print("  python price_to_probability.py -y SPY --period 7d --compare 5 15 60")
        print("")
        print("  # Save/load surfaces")
        print("  python price_to_probability.py -y SPY --window 15 --save-surface spy_15min.json")
        print("  python price_to_probability.py --load-surface spy_15min.json -I")
        print("")
        print("Note: --window is in MINUTES. For longer windows:")
        print("  60 = 1 hour, 1440 = 1 day, 10080 = 1 week")
        print("")
        print("  # Prediction market mode (Kalshi)")
        print("  python price_to_probability.py --kalshi kalshi_backtest_data.json -I")
        print("  python price_to_probability.py --kalshi data.json --price-buckets '0,5,10,20,30,40,50,60,70,80,90,95,100'")
        return

    # Build surface if we have data (skip for Kalshi - already built)
    if data and not kalshi_loaded:
        # Handle comparison mode
        if args.compare:
            print("\n" + "="*60)
            print("MULTI-TIMEFRAME COMPARISON")
            print("="*60)

            for window_size in [int(w) for w in args.compare]:
                print(f"\n--- {window_size} MINUTE WINDOWS ---")
                p2p_tmp = PriceToProbability(window_minutes=window_size,
                                             confidence_k=args.confidence_k,
                                             trade_exponent=args.trade_exp,
                                             recency_half_life=args.recency_hl)
                windows = p2p_tmp.slice_into_windows(data)
                p2p_tmp.build_surface()

                # Quick summary
                print(f"Windows: {len(windows)}")

                # Find mean reversion at small moves, late in window
                t_bucket = p2p_tmp.get_time_bucket(0.8)
                d_bucket = p2p_tmp.get_distance_bucket(0.003)  # 0.3%
                key = (d_bucket, t_bucket)
                sd = p2p_tmp.surface_data[key]
                total = sd["yes"] + sd["no"]
                if total > 10:
                    prob = sd["yes"] / total
                    print(f"P(leader wins) at 0.3% move, 80% elapsed: {prob*100:.1f}%")

                # Save each surface
                if args.save_surface:
                    base = args.save_surface.replace('.json', '')
                    p2p_tmp.save_surface(f"{base}_{window_size}min.json")

            return

        # Normal single-window mode
        print(f"Slicing into {args.window}-minute windows...")
        windows = p2p.slice_into_windows(data)
        print(f"Created {len(windows)} windows")

        print("Building probability surface...")
        p2p.build_surface()

    # Check if we have any data to work with
    if not data and not kalshi_loaded and not args.load_surface:
        return

    # Print surfaces
    p2p.print_surface()
    p2p.print_ev_surface()

    # Trading analysis
    p2p.analyze_for_trading()

    # Plot if matplotlib available
    if HAS_MATPLOTLIB:
        output = args.output or f"probability_surface_{args.window}min.png"
        p2p.plot_surface(output)
        p2p.plot_ev_surface(output)

    # Save surface if requested
    if args.save_surface:
        p2p.save_surface(args.save_surface)

    # Export decision table if requested
    if args.export_table:
        p2p.export_decision_table(args.export_table, args.min_composite, args.fractional_kelly)

    # Export LLM analysis if requested
    if args.export_llm:
        p2p.export_llm_analysis(args.export_llm)

    # Interactive mode
    if args.interactive:
        interactive_query(p2p)


if __name__ == "__main__":
    main()
