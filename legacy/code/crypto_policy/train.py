from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .env import SeriesEnv
from .model import PolicyNet


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-4
    gamma: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    device: str = "cpu"


def _sample_batch(logits: torch.Tensor, deterministic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = torch.distributions.Categorical(logits=logits)
    if deterministic:
        a = torch.argmax(logits, dim=-1)
    else:
        a = dist.sample()
    return a, dist.log_prob(a), dist.entropy()


def train_epoch_batch(envs: list[SeriesEnv], model: PolicyNet, opt: optim.Optimizer, cfg: TrainConfig) -> dict[str, float]:
    if not envs:
        return {"loss": 0.0, "mean_pnl_bp": 0.0}

    model.train()
    obs = np.stack([e.reset() for e in envs], axis=0)
    B = obs.shape[0]
    done = np.zeros((B,), dtype=bool)

    logp_steps: list[torch.Tensor] = []
    v_steps: list[torch.Tensor] = []
    ent_steps: list[torch.Tensor] = []
    rew_steps: list[torch.Tensor] = []
    alive_steps: list[torch.Tensor] = []

    while True:
        if done.all():
            break

        alive = ~done
        alive_t = torch.from_numpy(alive.astype(np.float32)).to(cfg.device)

        obs_t = torch.from_numpy(obs).to(cfg.device)
        logits, v = model(obs_t)

        mask_np = np.stack([e.valid_action_mask() for e in envs], axis=0)
        mask = torch.from_numpy(mask_np).to(cfg.device)
        logits = logits.masked_fill(~mask, -1e9)

        a, logp, ent = _sample_batch(logits, deterministic=False)

        a_np = a.detach().cpu().numpy().astype(int)
        next_obs = obs.copy()
        r_vec = np.zeros((B,), dtype=np.float32)
        for i, e in enumerate(envs):
            if done[i]:
                continue
            o2, r, d, _info = e.step(int(a_np[i]))
            next_obs[i] = o2
            r_vec[i] = float(r)
            done[i] = bool(d)
        obs = next_obs

        logp_steps.append(logp * alive_t)
        v_steps.append(v * alive_t)
        ent_steps.append(ent * alive_t)
        rew_steps.append(torch.from_numpy(r_vec).to(cfg.device) * alive_t)
        alive_steps.append(alive_t)

    logp_t = torch.stack(logp_steps, dim=0)
    v_t = torch.stack(v_steps, dim=0)
    ent_t = torch.stack(ent_steps, dim=0)
    rew_t = torch.stack(rew_steps, dim=0)
    alive_t = torch.stack(alive_steps, dim=0)

    if cfg.gamma == 1.0:
        returns = torch.flip(torch.cumsum(torch.flip(rew_t, dims=[0]), dim=0), dims=[0])
    else:
        T = rew_t.shape[0]
        returns = torch.zeros_like(rew_t)
        g = torch.zeros((B,), dtype=torch.float32, device=cfg.device)
        for t in reversed(range(T)):
            g = rew_t[t] + cfg.gamma * g
            returns[t] = g

    adv = returns - v_t.detach()
    denom = alive_t.sum().clamp_min(1.0)

    policy_loss = -((logp_t * adv) * alive_t).sum() / denom
    value_loss = (((returns - v_t) ** 2) * alive_t).sum() / denom
    entropy = (ent_t * alive_t).sum() / denom
    loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * (-entropy)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()

    pnls = [float(e.equity_bp) for e in envs]
    return {"loss": float(loss.item()), "policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item()), "entropy": float(entropy.item()), "mean_pnl_bp": float(mean(pnls))}


@torch.no_grad()
def eval_policy(env: SeriesEnv, model: PolicyNet, device: str) -> dict[str, float]:
    model.eval()
    obs = env.reset()
    total = 0.0
    while True:
        o = torch.from_numpy(obs).to(device)
        logits, _v = model(o.unsqueeze(0))
        logits = logits.squeeze(0)
        mask = torch.from_numpy(env.valid_action_mask()).to(device)
        logits = logits.masked_fill(~mask, -1e9)
        a = int(torch.argmax(logits).item())
        obs, r, done, info = env.step(a)
        total += float(r)
        if done:
            log = env.action_log()
            steps = len(log)
            turns = sum(1 for s in log if s.executed)
            in_mkt = sum(1 for s in log if s.pos != 0)
            long_ct = sum(1 for s in log if s.pos == 1)
            short_ct = sum(1 for s in log if s.pos == -1)
            pos_after = int(log[-1].pos) if log else int(env.pos)
            # Reconstruct turnover (sum of abs(delta_pos)) from the action log.
            turnover = 0.0
            cost_log_bp = 0.0
            prev = 0
            for s in log:
                d = abs(int(s.pos) - int(prev))
                turnover += d
                if d:
                    # Match env cost model: linear bp -> log-wealth bp via log(1 - cost_frac).
                    cost_bp = float(env.fee_bps + env.slippage_bps) * float(d)
                    cost_frac = min(0.999999, max(0.0, cost_bp / 1e4))
                    cost_log_bp += float(np.log1p(-cost_frac) * 1e4)
                prev = int(s.pos)
            return {
                "episode_pnl_bp": float(info.get("episode_pnl_bp", total)),
                "steps": float(steps),
                "turns": float(turns),
                "turn_rate": float(turns / steps) if steps else 0.0,
                "time_in_mkt": float(in_mkt / steps) if steps else 0.0,
                "time_long": float(long_ct / steps) if steps else 0.0,
                "time_short": float(short_ct / steps) if steps else 0.0,
                "pos_after": float(pos_after),
                "turnover": float(turnover),
                "cost_log_bp": float(cost_log_bp),
            }
