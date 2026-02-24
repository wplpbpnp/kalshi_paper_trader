#!/usr/bin/env python3
"""
Neural Network Price-to-Probability Model

Simple NN that learns P(YES wins) from (price, time_remaining, spread).
No arbitrary buckets - continuous inputs, continuous output.
"""

import json
import numpy as np
from datetime import datetime
import argparse

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Run: pip install torch")


class ProbabilityNet(nn.Module):
    """Feedforward network for P(YES wins)."""

    def __init__(self, input_size=6, hidden_size=16, dropout=0.3, num_layers=2):
        super().__init__()

        layers = []
        # First hidden layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMProbabilityNet(nn.Module):
    """LSTM network that predicts P(YES wins) at every timestep."""

    def __init__(self, input_size=3, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,      # features per timestep: [price, spread, time_remaining]
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True            # input shape: (batch, seq_len, features)
        )

        # Output layer: hidden state → P(YES)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        """
        x: (batch, max_seq_len, input_size) - padded sequences
        lengths: (batch,) - actual lengths of each sequence

        Returns: (batch, max_seq_len, 1) - P(YES) at each timestep
        """
        # Pack padded sequence for efficient LSTM processing
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(x)

        # Predict at every timestep
        predictions = self.output(lstm_out)
        return predictions


def load_kalshi_data(filepath, decay_half_life_days=7.0):
    """
    Load Kalshi data and extract features.

    Returns:
        features: np.array of shape (n_samples, 6) - [yes_price, time_remaining, spread, price_change, start_price, max_price]
        labels: np.array of shape (n_samples,) - 1 if YES won, 0 if NO won
        weights: np.array of shape (n_samples,) - time decay weights
    """
    print(f"Loading data from {filepath}...")

    with open(filepath, 'r') as f:
        markets = json.load(f)

    features = []
    labels = []
    timestamps = []

    for market in markets:
        result = market.get('result', '').lower()
        if result not in ['yes', 'no']:
            continue

        candlesticks = market.get('candlesticks', [])
        if len(candlesticks) < 2:
            continue

        label = 1.0 if result == 'yes' else 0.0
        n_candles = len(candlesticks)

        # Parse close time for weighting
        close_time_str = market.get('close_time', '')
        try:
            close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
        except:
            close_time = None

        # Get starting price for momentum features
        first_candle = candlesticks[0]
        first_bid = first_candle.get('yes_bid', {}).get('close', 0) or 0
        first_ask = first_candle.get('yes_ask', {}).get('close', 100) or 100
        if first_bid > 0 and first_ask < 100:
            start_price = (first_bid + first_ask) / 2
        else:
            start_price = first_candle.get('price', {}).get('close', 50) or 50

        if start_price is None or start_price <= 0 or start_price >= 100:
            start_price = 50  # Default fallback

        max_price_seen = start_price
        min_price_seen = start_price

        for candle_idx, candle in enumerate(candlesticks):
            # Extract features
            yes_bid = candle.get('yes_bid', {}).get('close', 0) or 0
            yes_ask = candle.get('yes_ask', {}).get('close', 100) or 100

            # Yes price (midpoint or last trade)
            if yes_bid > 0 and yes_ask < 100:
                yes_price = (yes_bid + yes_ask) / 2
                spread = yes_ask - yes_bid
            else:
                yes_price = candle.get('price', {}).get('close', 50) or 50
                spread = 10  # Default spread if not available

            if yes_price is None or yes_price <= 0 or yes_price >= 100:
                continue

            # Update max/min tracking
            max_price_seen = max(max_price_seen, yes_price)
            min_price_seen = min(min_price_seen, yes_price)

            # Time remaining (1.0 = start of window, 0.0 = end)
            time_remaining = 1.0 - (candle_idx + 1) / n_candles

            # Momentum features
            price_change = yes_price - start_price  # How much price moved from start

            # Normalize features to roughly -1 to 1 or 0 to 1 range
            features.append([
                yes_price / 100,           # 0-1: current price
                time_remaining,             # 0-1: time remaining
                min(spread, 20) / 20,       # 0-1: spread (capped)
                price_change / 50,          # ~-1 to 1: change from start (normalized by 50 cents)
                start_price / 100,          # 0-1: where market started
                max_price_seen / 100,       # 0-1: highest price seen so far
            ])
            labels.append(label)
            timestamps.append(close_time)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    print(f"Loaded {len(features)} samples from {len(markets)} markets")
    print(f"YES outcomes: {labels.sum():.0f} ({labels.mean()*100:.1f}%)")

    # Calculate time weights
    if decay_half_life_days > 0 and timestamps[0] is not None:
        reference_time = max(t for t in timestamps if t is not None)
        decay_constant = np.log(2) / decay_half_life_days

        weights = []
        for ts in timestamps:
            if ts is not None:
                age_days = (reference_time - ts).total_seconds() / (24 * 3600)
                weight = np.exp(-decay_constant * max(0, age_days))
            else:
                weight = 0.5  # Default for missing timestamps
            weights.append(weight)
        weights = np.array(weights, dtype=np.float32)
    else:
        weights = np.ones(len(labels), dtype=np.float32)

    return features, labels, weights


def load_kalshi_sequences(filepath, decay_half_life_days=7.0, max_seq_len=15):
    """
    Load Kalshi data as sequences for LSTM.

    Returns:
        sequences: list of np.arrays, each (seq_len, 3) - [price, spread, time_remaining]
        labels: np.array (n_markets,) - 1 if YES won, 0 if NO won
        weights: np.array (n_markets,) - time decay weights
    """
    print(f"Loading sequences from {filepath}...")

    with open(filepath, 'r') as f:
        markets = json.load(f)

    sequences = []
    labels = []
    timestamps = []

    for market in markets:
        result = market.get('result', '').lower()
        if result not in ['yes', 'no']:
            continue

        candlesticks = market.get('candlesticks', [])
        if len(candlesticks) < 2:
            continue

        label = 1.0 if result == 'yes' else 0.0
        n_candles = len(candlesticks)

        # Parse close time for weighting
        close_time_str = market.get('close_time', '')
        try:
            close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
        except:
            close_time = None

        # Build sequence for this market
        seq = []
        for candle_idx, candle in enumerate(candlesticks):
            yes_bid = candle.get('yes_bid', {}).get('close', 0) or 0
            yes_ask = candle.get('yes_ask', {}).get('close', 100) or 100

            if yes_bid > 0 and yes_ask < 100:
                yes_price = (yes_bid + yes_ask) / 2
                spread = yes_ask - yes_bid
            else:
                yes_price = candle.get('price', {}).get('close', 50) or 50
                spread = 10

            if yes_price is None or yes_price <= 0 or yes_price >= 100:
                continue

            time_remaining = 1.0 - (candle_idx + 1) / n_candles

            seq.append([
                yes_price / 100,           # 0-1
                min(spread, 20) / 20,       # 0-1
                time_remaining              # 0-1
            ])

        if len(seq) >= 2:
            sequences.append(np.array(seq, dtype=np.float32))
            labels.append(label)
            timestamps.append(close_time)

    labels = np.array(labels, dtype=np.float32)

    # Calculate time weights per market
    if decay_half_life_days > 0 and timestamps[0] is not None:
        reference_time = max(t for t in timestamps if t is not None)
        decay_constant = np.log(2) / decay_half_life_days

        weights = []
        for ts in timestamps:
            if ts is not None:
                age_days = (reference_time - ts).total_seconds() / (24 * 3600)
                weight = np.exp(-decay_constant * max(0, age_days))
            else:
                weight = 0.5
            weights.append(weight)
        weights = np.array(weights, dtype=np.float32)
    else:
        weights = np.ones(len(labels), dtype=np.float32)

    print(f"Loaded {len(sequences)} market sequences")
    print(f"Sequence lengths: min={min(len(s) for s in sequences)}, max={max(len(s) for s in sequences)}, avg={np.mean([len(s) for s in sequences]):.1f}")
    print(f"YES outcomes: {labels.sum():.0f} ({labels.mean()*100:.1f}%)")

    return sequences, labels, weights


def train_lstm_model(sequences, labels, weights, epochs=100, lr=0.001, val_split=0.2,
                     hidden_size=32, dropout=0.2, weight_decay=0.01, patience=20, num_layers=2):
    """Train the LSTM network on sequences."""

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Split into train/validation
    n = len(sequences)
    indices = np.random.permutation(n)
    val_size = int(n * val_split)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_seqs = [sequences[i] for i in train_idx]
    train_labels = labels[train_idx]
    train_weights = weights[train_idx]

    val_seqs = [sequences[i] for i in val_idx]
    val_labels = labels[val_idx]
    val_weights = weights[val_idx]

    def pad_sequences(seqs):
        """Pad sequences to same length, return padded tensor and lengths."""
        lengths = torch.tensor([len(s) for s in seqs])
        max_len = max(lengths)
        padded = torch.zeros(len(seqs), max_len, seqs[0].shape[1])
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = torch.tensor(seq)
        return padded, lengths

    X_train, train_lengths = pad_sequences(train_seqs)
    y_train = torch.tensor(train_labels)
    w_train = torch.tensor(train_weights)

    X_val, val_lengths = pad_sequences(val_seqs)
    y_val = torch.tensor(val_labels)
    w_val = torch.tensor(val_weights)

    # Create model
    input_size = sequences[0].shape[1]  # features per timestep
    model = LSTMProbabilityNet(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def sequence_loss(predictions, labels, lengths, market_weights):
        """
        Loss: average BCE across all timesteps of all sequences.
        Each prediction at each timestep is compared to the market outcome.
        Weighted by market recency.
        """
        total_loss = 0
        total_weight = 0

        for i in range(len(predictions)):
            seq_len = lengths[i].item()
            # All predictions for this sequence compared to same label
            seq_preds = predictions[i, :seq_len, 0]  # (seq_len,)
            seq_labels = labels[i].expand(seq_len)    # same label repeated
            bce = nn.functional.binary_cross_entropy(seq_preds, seq_labels, reduction='sum')
            total_loss += bce * market_weights[i]
            total_weight += seq_len * market_weights[i]

        return total_loss / total_weight

    def sequence_accuracy(predictions, labels, lengths, market_weights):
        """Weighted accuracy using final prediction of each sequence."""
        correct_weighted = 0
        total_weight = 0

        for i in range(len(predictions)):
            seq_len = lengths[i].item()
            final_pred = predictions[i, seq_len - 1, 0]
            correct = ((final_pred > 0.5) == (labels[i] > 0.5)).float()
            correct_weighted += correct * market_weights[i]
            total_weight += market_weights[i]

        return correct_weighted / total_weight

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train, train_lengths)
        loss = sequence_loss(predictions, y_train, train_lengths, w_train)

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val, val_lengths)
            val_loss = sequence_loss(val_preds, y_val, val_lengths, w_val)
            val_acc = sequence_accuracy(val_preds, y_val, val_lengths, w_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.3f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_model_state)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"LSTM: {num_layers} layers, {hidden_size} hidden units, {dropout:.0%} dropout")

    return model


def train_model(features, labels, weights, epochs=100, lr=0.001, val_split=0.2,
                hidden_size=16, dropout=0.3, weight_decay=0.01, patience=20, num_layers=2):
    """Train the neural network with regularization and early stopping."""

    # Split into train/validation
    n = len(features)
    indices = np.random.permutation(n)
    val_size = int(n * val_split)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train = torch.tensor(features[train_idx])
    y_train = torch.tensor(labels[train_idx]).unsqueeze(1)
    w_train = torch.tensor(weights[train_idx]).unsqueeze(1)

    X_val = torch.tensor(features[val_idx])
    y_val = torch.tensor(labels[val_idx]).unsqueeze(1)
    w_val = torch.tensor(weights[val_idx]).unsqueeze(1)

    # Create model
    model = ProbabilityNet(input_size=features.shape[1], hidden_size=hidden_size,
                           dropout=dropout, num_layers=num_layers)

    # L2 regularization via weight_decay parameter
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Weighted BCE loss
    def weighted_bce(pred, target, weight):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        return (bce * weight).mean()

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(X_train)
        loss = weighted_bce(pred, y_train, w_train)

        loss.backward()
        optimizer.step()

        # Validation (weighted same as training)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = weighted_bce(val_pred, y_val, w_val)

            # Weighted accuracy
            correct = ((val_pred > 0.5) == y_val).float()
            val_acc = (correct * w_val).sum() / w_val.sum()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.3f}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model: {num_layers} layers, {hidden_size} hidden units, {dropout:.0%} dropout, {weight_decay} weight decay")

    return model


def print_probability_surface(model, spread=0.05, input_size=6):
    """Print probability surface similar to the bucketed version."""

    print("\n" + "="*80)
    print("NEURAL NETWORK PROBABILITY SURFACE: P(YES wins)")
    print("="*80)
    print(f"Spread fixed at {spread*20:.0f}¢, showing RISING prices (start=30¢)\n")

    # Price points (bucket midpoints for comparison)
    prices = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    # Time points (fraction remaining)
    times = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    time_labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]

    # Header
    print(f"{'YES Price':<12}", end="")
    for tl in time_labels:
        print(f"{tl:>9}", end="")
    print()
    print("-" * (12 + 9 * len(time_labels)))

    model.eval()
    with torch.no_grad():
        for price in prices:
            print(f"{price-5}-{price+5}¢".ljust(12), end="")
            for time_remaining in times:
                # Simulate RISING price scenario: started at 30, now at current price
                start_price = 30
                price_change = price - start_price
                max_price = max(price, start_price)

                if input_size == 6:
                    x = torch.tensor([[price/100, time_remaining, spread,
                                      price_change/50, start_price/100, max_price/100]])
                else:
                    x = torch.tensor([[price/100, time_remaining, spread]])
                p = model(x).item()
                print(f"{p*100:>8.1f}%", end="")
            print()


def print_falling_probability_surface(model, spread=0.05, input_size=6):
    """Print probability surface for FALLING prices."""

    print("\n" + "="*80)
    print("NEURAL NETWORK PROBABILITY SURFACE: P(YES wins)")
    print("="*80)
    print(f"Spread fixed at {spread*20:.0f}¢, showing FALLING prices (start=70¢)\n")

    prices = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    times = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    time_labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]

    print(f"{'YES Price':<12}", end="")
    for tl in time_labels:
        print(f"{tl:>9}", end="")
    print()
    print("-" * (12 + 9 * len(time_labels)))

    model.eval()
    with torch.no_grad():
        for price in prices:
            print(f"{price-5}-{price+5}¢".ljust(12), end="")
            for time_remaining in times:
                # Simulate FALLING price scenario: started at 70, now at current price
                start_price = 70
                price_change = price - start_price
                max_price = max(price, start_price)

                if input_size == 6:
                    x = torch.tensor([[price/100, time_remaining, spread,
                                      price_change/50, start_price/100, max_price/100]])
                else:
                    x = torch.tensor([[price/100, time_remaining, spread]])
                p = model(x).item()
                print(f"{p*100:>8.1f}%", end="")
            print()


def analyze_lstm_by_timestep(model, sequences, labels, weights):
    """Analyze LSTM predictions at each timestep."""

    model.eval()

    # Group predictions by timestep (relative to end)
    # timestep 0 = final candle, timestep -1 = second to last, etc.
    timestep_preds = {}  # {timestep: [(pred, label, weight), ...]}

    def pad_sequences(seqs):
        lengths = torch.tensor([len(s) for s in seqs])
        max_len = max(lengths)
        padded = torch.zeros(len(seqs), max_len, seqs[0].shape[1])
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = torch.tensor(seq)
        return padded, lengths

    X, lengths = pad_sequences(sequences)

    with torch.no_grad():
        preds = model(X, lengths)

    for i in range(len(sequences)):
        seq_len = lengths[i].item()
        label = labels[i]
        weight = weights[i]

        for j in range(seq_len):
            # timestep relative to end (0 = final, -1 = second to last, etc.)
            timestep = j - seq_len + 1
            pred = preds[i, j, 0].item()

            if timestep not in timestep_preds:
                timestep_preds[timestep] = []
            timestep_preds[timestep].append((pred, label, weight))

    print("\n" + "="*80)
    print("LSTM ANALYSIS BY TIMESTEP (relative to settlement)")
    print("="*80)
    print(f"{'Timestep':<10} {'Samples':<10} {'BCE Loss':<12} {'Pred|YES':<12} {'Pred|NO':<12} {'Spread':<10}")
    print("-" * 76)

    for ts in sorted(timestep_preds.keys()):
        data = timestep_preds[ts]
        preds_arr = np.array([d[0] for d in data])
        labels_arr = np.array([d[1] for d in data])
        weights_arr = np.array([d[2] for d in data])

        # Split by outcome
        yes_mask = labels_arr == 1
        no_mask = labels_arr == 0

        if yes_mask.sum() > 0:
            pred_when_yes = np.average(preds_arr[yes_mask], weights=weights_arr[yes_mask])
        else:
            pred_when_yes = 0.5

        if no_mask.sum() > 0:
            pred_when_no = np.average(preds_arr[no_mask], weights=weights_arr[no_mask])
        else:
            pred_when_no = 0.5

        # BCE loss
        eps = 1e-7
        bce = -np.average(
            labels_arr * np.log(preds_arr + eps) + (1 - labels_arr) * np.log(1 - preds_arr + eps),
            weights=weights_arr
        )

        # Spread between predictions (higher = more discriminating)
        spread = pred_when_yes - pred_when_no

        ts_label = f"t{ts}" if ts <= 0 else f"t+{ts}"
        print(f"{ts_label:<10} {len(data):<10} {bce:>10.4f} {pred_when_yes*100:>10.1f}% {pred_when_no*100:>10.1f}% {spread*100:>+8.1f}%")


def analyze_lstm_calibration(model, sequences, labels, weights):
    """Analyze calibration: when we predict X%, does YES win X% of the time?"""

    model.eval()

    def pad_sequences(seqs):
        lengths = torch.tensor([len(s) for s in seqs])
        max_len = max(lengths)
        padded = torch.zeros(len(seqs), max_len, seqs[0].shape[1])
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = torch.tensor(seq)
        return padded, lengths

    X, lengths = pad_sequences(sequences)

    with torch.no_grad():
        preds = model(X, lengths)

    # Collect all predictions with their outcomes, binned by prediction value
    # Also track by timestep
    pred_bins = {}  # {(pred_bucket, timestep_bucket): [(pred, label, weight), ...]}

    pred_buckets = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                    (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    time_buckets = [(-14, -10), (-10, -6), (-6, -3), (-3, 0)]  # relative to end

    for i in range(len(sequences)):
        seq_len = lengths[i].item()
        label = labels[i]
        weight = weights[i]

        for j in range(seq_len):
            timestep = j - seq_len + 1  # relative to end
            pred = preds[i, j, 0].item() * 100  # convert to percentage

            # Find prediction bucket
            pred_bucket = None
            for low, high in pred_buckets:
                if low <= pred < high:
                    pred_bucket = (low, high)
                    break
            if pred_bucket is None:
                pred_bucket = (90, 100)  # catch 100%

            # Find time bucket
            time_bucket = None
            for low, high in time_buckets:
                if low <= timestep < high:
                    time_bucket = (low, high)
                    break
            if time_bucket is None:
                time_bucket = (-3, 0)

            key = (pred_bucket, time_bucket)
            if key not in pred_bins:
                pred_bins[key] = []
            pred_bins[key].append((pred / 100, label, weight))

    print("\n" + "="*80)
    print("LSTM CALIBRATION: Predicted P(YES) vs Actual Win Rate")
    print("="*80)

    for time_bucket in time_buckets:
        t_label = f"t{time_bucket[0]} to t{time_bucket[1]}"
        print(f"\n{t_label}:")
        print(f"{'Predicted':<12} {'Samples':<10} {'Avg Pred':<12} {'Actual Win%':<14} {'Error':<10}")
        print("-" * 58)

        for pred_bucket in pred_buckets:
            key = (pred_bucket, time_bucket)
            if key not in pred_bins or len(pred_bins[key]) < 10:
                continue

            data = pred_bins[key]
            preds_arr = np.array([d[0] for d in data])
            labels_arr = np.array([d[1] for d in data])
            weights_arr = np.array([d[2] for d in data])

            avg_pred = np.average(preds_arr, weights=weights_arr)
            actual_win = np.average(labels_arr, weights=weights_arr)
            error = avg_pred - actual_win

            bucket_label = f"{pred_bucket[0]}-{pred_bucket[1]}%"
            print(f"{bucket_label:<12} {len(data):<10} {avg_pred*100:>10.1f}% {actual_win*100:>12.1f}% {error*100:>+8.1f}%")


def compute_lstm_ev_table(model, sequences, labels, weights, conservative_shrink=0.1, trade_exponent=0.5):
    """
    Compute EV and composite scores by price bucket and time bucket.
    Uses conservative probability (shrunk 10% toward 50%).
    """
    model.eval()

    def pad_sequences(seqs):
        lengths = torch.tensor([len(s) for s in seqs])
        max_len = max(lengths)
        padded = torch.zeros(len(seqs), max_len, seqs[0].shape[1])
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = torch.tensor(seq)
        return padded, lengths

    X, lengths = pad_sequences(sequences)

    with torch.no_grad():
        preds = model(X, lengths)

    # Buckets
    price_buckets = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                     (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    time_buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    trajectory_types = ["rising", "falling", "stable"]

    # Collect data by bucket
    bucket_data = {}  # {(price_bucket, time_bucket, trajectory): [(pred, label, weight, price), ...]}

    for i in range(len(sequences)):
        seq_len = lengths[i].item()
        label = labels[i]
        weight = weights[i]
        seq = sequences[i]

        # Get starting price for trajectory classification
        start_price = seq[0][0] * 100

        for j in range(seq_len):
            pred = preds[i, j, 0].item()
            price = seq[j][0] * 100  # convert back to cents
            time_remaining = seq[j][2]  # already 0-1

            # Classify trajectory based on price change from start
            price_change = price - start_price
            if price_change > 5:
                trajectory = "rising"
            elif price_change < -5:
                trajectory = "falling"
            else:
                trajectory = "stable"

            # Find price bucket
            price_bucket = None
            for low, high in price_buckets:
                if low <= price < high:
                    price_bucket = (low, high)
                    break
            if price_bucket is None:
                price_bucket = (90, 100)

            # Find time bucket
            time_bucket = None
            for low, high in time_buckets:
                if low <= time_remaining < high:
                    time_bucket = (low, high)
                    break
            if time_bucket is None:
                time_bucket = (0.8, 1.0)

            key = (price_bucket, time_bucket, trajectory)
            if key not in bucket_data:
                bucket_data[key] = []
            bucket_data[key].append((pred, label, weight, price))

    # Debug: count trajectories
    traj_counts = {'stable': 0, 'rising': 0, 'falling': 0}
    for key in bucket_data:
        traj_counts[key[2]] += len(bucket_data[key])
    print(f"\nTrajectory counts in bucket_data: {traj_counts}")

    print("\n" + "="*115)
    print("LSTM EV TABLE (Conservative: predictions shrunk 10% toward 50%)")
    print("="*115)
    print(f"{'Price':<12} {'Time':<12} {'Traj':<10} {'Samples':<10} {'P(YES)':<10} {'Cons P':<10} {'Side':<6} {'EV':<10} {'Trades':<10} {'Composite':<10}")
    print("-" * 115)

    results = []

    for price_bucket in price_buckets:
        for time_bucket in time_buckets:
            for trajectory in trajectory_types:
                key = (price_bucket, time_bucket, trajectory)
                if key not in bucket_data or len(bucket_data[key]) < 10:
                    continue

                data = bucket_data[key]
                preds_arr = np.array([d[0] for d in data])
                labels_arr = np.array([d[1] for d in data])
                weights_arr = np.array([d[2] for d in data])
                prices_arr = np.array([d[3] for d in data])

                # Weighted average prediction
                avg_pred = np.average(preds_arr, weights=weights_arr)

                # Conservative prediction (shrink toward 50%)
                cons_pred = avg_pred + (0.5 - avg_pred) * conservative_shrink

                # Midpoint price for EV calculation
                midpoint = (price_bucket[0] + price_bucket[1]) / 2 / 100

                # Determine best side and calculate EV
                # For YES: EV = P(YES) * (1 - price) - P(NO) * price
                # For NO: EV = P(NO) * (1 - (1-price)) - P(YES) * (1-price) = P(NO) * price - P(YES) * (1-price)
                ev_yes = cons_pred * (1 - midpoint) - (1 - cons_pred) * midpoint
                ev_no = (1 - cons_pred) * midpoint - cons_pred * (1 - midpoint)

                if ev_yes > ev_no and ev_yes > 0:
                    side = "YES"
                    ev = ev_yes
                elif ev_no > ev_yes and ev_no > 0:
                    side = "NO"
                    ev = ev_no
                else:
                    side = "-"
                    ev = max(ev_yes, ev_no)

                # Trade count (weighted)
                trades = weights_arr.sum()

                # Composite score
                composite = ev * (trades ** trade_exponent) if ev > 0 else 0

                price_label = f"{price_bucket[0]}-{price_bucket[1]}¢"
                time_label = f"{time_bucket[0]:.1f}-{time_bucket[1]:.1f}"

                print(f"{price_label:<12} {time_label:<12} {trajectory:<10} {len(data):<10} {avg_pred*100:>8.1f}% {cons_pred*100:>8.1f}% {side:<6} {ev*100:>+8.2f}% {trades:>8.1f} {composite:>10.4f}")

                if ev > 0:
                    results.append({
                        'price_bucket': price_bucket,
                        'time_bucket': time_bucket,
                        'trajectory': trajectory,
                        'side': side,
                        'p_win': cons_pred if side == "YES" else 1 - cons_pred,
                        'ev': ev * 100,
                        'trades': trades,
                        'composite': composite,
                        'samples': len(data)
                    })

    # Sort by composite and show top opportunities
    results.sort(key=lambda x: x['composite'], reverse=True)

    print("\n" + "="*90)
    print("TOP TRADING OPPORTUNITIES (sorted by composite score)")
    print("="*90)
    print(f"{'Rank':<6} {'Price':<12} {'Time':<12} {'Traj':<10} {'Side':<6} {'EV':<10} {'P(win)':<10} {'Composite':<10}")
    print("-" * 90)

    for i, r in enumerate(results[:20]):
        price_label = f"{r['price_bucket'][0]}-{r['price_bucket'][1]}¢"
        time_label = f"{r['time_bucket'][0]:.1f}-{r['time_bucket'][1]:.1f}"
        print(f"{i+1:<6} {price_label:<12} {time_label:<12} {r['trajectory']:<10} {r['side']:<6} {r['ev']:>+8.2f}% {r['p_win']*100:>8.1f}% {r['composite']:>10.4f}")

    return results


def print_lstm_probability_surface(model, spread=0.05):
    """Print probability surfaces for LSTM model with different price trajectories."""

    model.eval()

    print("\n" + "="*80)
    print("LSTM PROBABILITY SURFACE: P(YES wins)")
    print("="*80)

    # Test different scenarios
    scenarios = [
        ("STABLE at 50¢", lambda t: 50),
        ("RISING 30→70¢", lambda t: 30 + 40 * t),
        ("FALLING 70→30¢", lambda t: 70 - 40 * t),
        ("SPIKE then FALL (30→70→50)", lambda t: 30 + 80*t if t < 0.5 else 110 - 80*t),
        ("DIP then RISE (70→30→50)", lambda t: 70 - 80*t if t < 0.5 else -10 + 80*t),
    ]

    for name, price_fn in scenarios:
        print(f"\n{name}:")
        print("-" * 70)

        # Build a 15-step sequence
        seq = []
        for i in range(15):
            t = i / 14  # 0 to 1
            price = price_fn(t)
            time_remaining = 1.0 - (i + 1) / 15
            seq.append([price / 100, spread, time_remaining])

        seq_tensor = torch.tensor([seq], dtype=torch.float32)
        lengths = torch.tensor([15])

        with torch.no_grad():
            preds = model(seq_tensor, lengths)

        # Print predictions at key points
        print(f"{'Time':<12}", end="")
        for i in [0, 2, 4, 6, 8, 10, 12, 14]:
            t_label = f"t-{14-i}"
            print(f"{t_label:>8}", end="")
        print()

        print(f"{'Price':<12}", end="")
        for i in [0, 2, 4, 6, 8, 10, 12, 14]:
            print(f"{seq[i][0]*100:>7.0f}¢", end="")
        print()

        print(f"{'P(YES)':<12}", end="")
        for i in [0, 2, 4, 6, 8, 10, 12, 14]:
            p = preds[0, i, 0].item()
            print(f"{p*100:>7.1f}%", end="")
        print()


def print_ev_surface(model, spread=0.05, input_size=6):
    """Print EV surface for both rising and falling prices."""

    prices = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    times = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    time_labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]

    for scenario, start_price in [("RISING (started at 30¢)", 30), ("FALLING (started at 70¢)", 70)]:
        print("\n" + "="*80)
        print(f"EV SURFACE - {scenario}: P(YES wins)")
        print("="*80)
        print("EV = P(YES) × (1 - price) - P(NO) × price\n")

        print(f"{'YES Price':<12}", end="")
        for tl in time_labels:
            print(f"{tl:>9}", end="")
        print()
        print("-" * (12 + 9 * len(time_labels)))

        model.eval()
        with torch.no_grad():
            for price in prices:
                print(f"{price-5}-{price+5}¢".ljust(12), end="")
                for time_remaining in times:
                    price_change = price - start_price
                    max_price = max(price, start_price)

                    if input_size == 6:
                        x = torch.tensor([[price/100, time_remaining, spread,
                                          price_change/50, start_price/100, max_price/100]])
                    else:
                        x = torch.tensor([[price/100, time_remaining, spread]])
                    p_win = model(x).item()

                    # EV using bucket midpoint
                    midpoint = price / 100
                    ev = p_win * (1 - midpoint) - (1 - p_win) * midpoint

                    print(f"{ev*100:>+8.2f}%", end="")
                print()


def export_model(model, filepath, input_size=6, hidden_size=8, dropout=0.5):
    """Export model for use by trading bot."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'dropout': dropout
    }, filepath)
    print(f"\nModel saved to {filepath}")


def main():
    if not HAS_TORCH:
        print("PyTorch required. Install with: pip install torch")
        return

    parser = argparse.ArgumentParser(description='Neural Network Probability Model')
    parser.add_argument('--kalshi', '-k', required=True, help='Kalshi JSON data file')
    parser.add_argument('--decay-half-life', type=float, default=7.0, help='Half-life for time weighting (days)')
    parser.add_argument('--epochs', type=int, default=500, help='Max training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=16, help='Hidden layer size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='L2 regularization')
    parser.add_argument('--lstm', action='store_true', help='Use LSTM instead of feedforward')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--export', '-o', help='Export model to file')

    args = parser.parse_args()

    if args.lstm:
        # LSTM mode - use sequences
        sequences, labels, weights = load_kalshi_sequences(args.kalshi, args.decay_half_life)

        print(f"\nTraining LSTM with max {args.epochs} epochs, decay half-life = {args.decay_half_life} days...")
        model = train_lstm_model(
            sequences, labels, weights,
            epochs=args.epochs,
            lr=args.lr,
            hidden_size=args.hidden,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_layers=args.layers
        )

        # Analyze and print LSTM results
        analyze_lstm_by_timestep(model, sequences, labels, weights)
        analyze_lstm_calibration(model, sequences, labels, weights)
        ev_results = compute_lstm_ev_table(model, sequences, labels, weights)
        print_lstm_probability_surface(model)

    else:
        # Feedforward mode
        features, labels, weights = load_kalshi_data(args.kalshi, args.decay_half_life)

        print(f"\nTraining with max {args.epochs} epochs, decay half-life = {args.decay_half_life} days...")
        model = train_model(
            features, labels, weights,
            epochs=args.epochs,
            lr=args.lr,
            hidden_size=args.hidden,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_layers=args.layers
        )

        # Print surfaces
        input_size = features.shape[1]
        print_probability_surface(model, input_size=input_size)
        print_falling_probability_surface(model, input_size=input_size)
        print_ev_surface(model, input_size=input_size)

        # Export if requested
        if args.export:
            export_model(model, args.export, input_size=input_size,
                         hidden_size=args.hidden, dropout=args.dropout)


if __name__ == "__main__":
    main()
