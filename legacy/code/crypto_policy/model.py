from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PolicyConfig:
    obs_dim: int
    hidden: Sequence[int] = (64,)
    n_actions: int = 4


class PolicyNet(nn.Module):
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

        self.pi = nn.Linear(in_dim, cfg.n_actions)
        self.v = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        return self.pi(x), self.v(x).squeeze(-1)

