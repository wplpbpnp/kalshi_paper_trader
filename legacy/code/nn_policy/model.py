from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PolicyConfig:
    obs_dim: int
    hidden: Sequence[int] = (128, 128)
    n_actions: int = 4
    size_buckets: Sequence[int] = (0, 1, 2, 3, 5)


class PolicyNet(nn.Module):
    """
    Black-box policy:
      - categorical over actions
      - categorical over size buckets (used only on buy/close)
      - value head for advantage estimation (A2C-style)
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        in_dim = cfg.obs_dim
        for h in cfg.hidden:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        self.pi_action = nn.Linear(in_dim, cfg.n_actions)
        self.pi_size = nn.Linear(in_dim, len(cfg.size_buckets))
        self.v = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (logits_action, logits_size, value).
        """
        x = self.backbone(obs)
        return self.pi_action(x), self.pi_size(x), self.v(x).squeeze(-1)

