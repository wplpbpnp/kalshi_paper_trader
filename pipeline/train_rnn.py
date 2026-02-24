from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import build_fixed_sequence, iter_snapshot_files, load_label_map, read_snapshot_file
from .model import RnnBinaryClassifier, RnnConfig


@dataclass(frozen=True)
class TrainArgs:
    snapshot_glob: str
    labels_json: str
    seq_len: int = 240
    min_records: int = 30
    val_frac: float = 0.2
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    seed: int = 7
    device: str = "cpu"
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainResult:
    model: RnnBinaryClassifier
    train_count: int
    val_count: int
    val_loss: float
    val_accuracy: float
    tickers_used: List[str]


def _build_dataset(args: TrainArgs) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = load_label_map(args.labels_json)
    files = list(iter_snapshot_files(args.snapshot_glob))
    xs: List[List[List[float]]] = []
    ys: List[int] = []
    tickers: List[str] = []
    snapshot_tickers: set[str] = set()
    files_with_records = 0
    files_meeting_min_records = 0
    unlabeled_examples: List[str] = []

    for path in files:
        records = read_snapshot_file(path)
        if records:
            files_with_records += 1
            snapshot_tickers.add(records[-1].market_ticker)
        if len(records) < args.min_records:
            continue
        files_meeting_min_records += 1
        ticker = records[-1].market_ticker
        if ticker not in labels:
            if ticker not in unlabeled_examples and len(unlabeled_examples) < 5:
                unlabeled_examples.append(ticker)
            continue
        seq = build_fixed_sequence(records, args.seq_len)
        xs.append(seq)
        ys.append(labels[ticker])
        tickers.append(ticker)

    if not xs:
        overlap = len(snapshot_tickers.intersection(set(labels.keys())))
        msg = (
            "No labeled training samples found. "
            f"snapshot_files={len(files)}, "
            f"files_with_records={files_with_records}, "
            f"files_meeting_min_records={files_meeting_min_records}, "
            f"snapshot_tickers={len(snapshot_tickers)}, "
            f"labels={len(labels)}, "
            f"ticker_overlap={overlap}. "
            "This usually means your snapshot markets are newer than your settlement labels."
        )
        if unlabeled_examples:
            msg += f" Unlabeled ticker examples: {', '.join(unlabeled_examples)}."
        raise RuntimeError(msg)

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return x, y, tickers


def _split(
    x: np.ndarray, y: np.ndarray, tickers: List[str], val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_frac)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return (
        x[train_idx],
        y[train_idx],
        x[val_idx],
        y[val_idx],
        [tickers[int(i)] for i in train_idx],
        [tickers[int(i)] for i in val_idx],
    )


def train(args: TrainArgs) -> TrainResult:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y, tickers = _build_dataset(args)
    x_tr, y_tr, x_va, y_va, tr_tickers, va_tickers = _split(x, y, tickers, args.val_frac, args.seed)

    tr_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    va_ds = TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    cfg = RnnConfig(
        input_dim=x.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = RnnBinaryClassifier(cfg).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(max(1, args.epochs)):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    val_loss = 0.0
    val_n = 0
    val_correct = 0
    with torch.no_grad():
        for xb, yb in va_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).float()
            val_correct += int((pred == yb).sum().item())
            val_n += int(yb.numel())
            val_loss += float(loss.item()) * int(yb.numel())

    mean_val_loss = val_loss / max(1, val_n)
    val_acc = float(val_correct / max(1, val_n))

    return TrainResult(
        model=model,
        train_count=len(tr_tickers),
        val_count=len(va_tickers),
        val_loss=mean_val_loss,
        val_accuracy=val_acc,
        tickers_used=tickers,
    )
