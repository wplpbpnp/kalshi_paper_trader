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


class PriceToProbability:
    """Convert price data to probability space."""

    def __init__(self, window_minutes=15,
                 trade_exponent=0.5,
                 mode='real', decay_half_life_days=7.0):
        """
        Args:
            window_minutes: Size of each synthetic market window
            trade_exponent: Exponent for trade frequency in composite (0.5=sqrt, 1.0=linear)
            mode: 'real' for real market trading (returns relative to entry),
                  'prediction' for prediction markets (binary payoff)
            decay_half_life_days: Half-life in days for time-weighted p(win) calculation
                                  (e.g., 7.0 means data from 7 days ago has 50% weight)
        """
        self.window_minutes = window_minutes
        self.trade_exponent = trade_exponent
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
        self.time_buckets = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]  # Fraction of window elapsed

        # For prediction market mode: YES price buckets (in cents, 0-100)
        self.price_buckets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

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
            candlesticks = market.get('candlesticks', [])

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
                
                yes_bid = candle.get('yes_bid', {}).get('close', 0)
                yes_ask = candle.get('yes_ask', {}).get('close', 100)

                yes_price = yes_ask
                if not yes_price:
                    # Fall back to last trade price
                    yes_price = candle.get('price', {}).get('close', 50)

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

        def hedge_p_win(p_win, n):
            """
            Adjust p_win based on the confidence given by the number of samples.
            Use the most conservative bound
            """
            print("hedge_p_win\n")
            print(p_win)
            if p_win == 0:
                return p_win

            Z = 1.96 # 95% confidence bound
            p_hat = abs(p_win)

            f1 = 1 / (1 + ((Z**2)/n))
            f2 = (Z**2)/(2*n)
            p = (p_hat + f2) * f1
            print(p)

            bound = f1 * Z * (((p_hat*(1-p_hat)/n) + ((Z**2)/(4*n**2))) ** 0.5)
            print(bound)
            if p > 0.5:
                p -= bound
            else:
                p += bound
            print(p)
            return np.clip(p, 0, 1)

        weighted_p_win = hedge_p_win(weighted_p_win, yes_count)

        return {
            "weighted_p_win": weighted_p_win,
            "weighted_yes": weighted_yes,
            "weighted_no": weighted_no,
            "unweighted_p_win": unweighted_p_win,
            "total_weight": total_weight
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

        return ev_matrix, p_win_matrix, kelly_matrix, x_labels, time_labels, counts, trade_freq

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
        ev_matrix, p_win_matrix, kelly_matrix, x_labels, time_labels, counts, trade_freq = self.get_ev_matrix()

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
        ev_matrix, p_win_matrix, kelly_matrix, x_labels, time_labels, counts, trade_freq = self.get_ev_matrix()

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

        # Slice the data
        matrix = matrix[x_start:x_end, time_start:time_end]
        counts = counts[x_start:x_end, time_start:time_end]
        trade_freq = trade_freq[x_start:x_end, time_start:time_end]
        x_labels = x_labels[x_start:x_end]
        time_labels = time_labels[time_start:time_end]

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(40, 14))

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

        ev_matrix, p_win_matrix, kelly_matrix, x_labels, time_labels, counts, trade_freq = self.get_ev_matrix()

        if ev_matrix is None:
            print("No surface data available")
            return

        # Apply zoom if set
        x_start, x_end = 0, len(x_labels)
        time_start, time_end = 0, len(time_labels)

        # Slice
        ev_matrix = ev_matrix[x_start:x_end, time_start:time_end]
        kelly_matrix = kelly_matrix[x_start:x_end, time_start:time_end]
        counts = counts[x_start:x_end, time_start:time_end]
        trade_freq = trade_freq[x_start:x_end, time_start:time_end]
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

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(40, 14))

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

def main():
    parser = argparse.ArgumentParser(description='Convert price data to probability space')
    parser.add_argument('--window', '-w', type=int, default=15, help='Window size in MINUTES (default: 15). Each window becomes a synthetic prediction market.')
    parser.add_argument('--output', '-o', help='Output path for plot')
    parser.add_argument('--mode', choices=['real', 'prediction'], default='real',
                        help='Mode: "real" for real market trading (returns relative to entry), "prediction" for prediction markets (binary payoff). Default: real')
    parser.add_argument('--kalshi', '-k', help='Load Kalshi prediction market data from JSON file')
    parser.add_argument('--export-table', help='Export decision table JSON for bot consumption')
    parser.add_argument('--min-composite', type=float, default=0.1, help='Minimum |composite| score for export-table (default: 0.1)')
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
                              mode=mode,
                              decay_half_life_days=decay_hl)

    if args.decay_half_life > 0:
        print(f"Time-weighted P(win) enabled with half-life of {args.decay_half_life} days")

    data = None
    kalshi_loaded = False

    if args.kalshi:
        # Load Kalshi prediction market data
        p2p.load_kalshi_json(args.kalshi)
        kalshi_loaded = True

    # Check if we have any data to work with
    if not data and not kalshi_loaded and not args.load_surface:
        return

    # Print surfaces
    p2p.print_surface()
    p2p.print_ev_surface()

    # Plot if matplotlib available
    if HAS_MATPLOTLIB:
        output = args.output or f"probability_surface_{args.window}min.png"
        p2p.plot_surface(output)
        p2p.plot_ev_surface(output)

    # Export decision table if requested
    if args.export_table:
        p2p.export_decision_table(args.export_table, args.min_composite)

if __name__ == "__main__":
    main()
