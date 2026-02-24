from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RnnConfig:
    input_dim: int = 6
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1


class RnnBinaryClassifier(nn.Module):
    """
    LSTM classifier over fixed-length market microstructure sequences.
    """

    def __init__(self, cfg: RnnConfig):
        super().__init__()
        self.cfg = cfg
        self.rnn = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.rnn(x)
        # Use the final timestep representation.
        last = out[:, -1, :]
        logits = self.head(last).squeeze(-1)
        return logits


def save_checkpoint(path: str, model: RnnBinaryClassifier) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": model.cfg.input_dim,
            "hidden_size": model.cfg.hidden_size,
            "num_layers": model.cfg.num_layers,
            "dropout": model.cfg.dropout,
        },
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[RnnBinaryClassifier, Dict]:
    payload = torch.load(path, map_location=device)
    cfg_raw = payload.get("config") or {}
    cfg = RnnConfig(
        input_dim=int(cfg_raw.get("input_dim", 6)),
        hidden_size=int(cfg_raw.get("hidden_size", 64)),
        num_layers=int(cfg_raw.get("num_layers", 1)),
        dropout=float(cfg_raw.get("dropout", 0.1)),
    )
    model = RnnBinaryClassifier(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload

