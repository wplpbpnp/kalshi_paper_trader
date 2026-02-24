from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .env import EpisodeEnv
from .model import PolicyNet, PolicyConfig


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-4
    gamma: float = 1.0  # episodic, short horizon; keep undiscounted by default
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    device: str = "cpu"


@dataclass
class EpisodeRollout:
    obs: list[torch.Tensor]
    act: list[int]
    size_idx: list[int]
    logp: list[torch.Tensor]
    val: list[torch.Tensor]
    ent: list[torch.Tensor]
    rew: list[float]
    done: bool
    pnl_cents: int


def _discounted_returns(rews: list[float], gamma: float) -> torch.Tensor:
    out = []
    g = 0.0
    for r in reversed(rews):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


def _sample_categorical(logits: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return int(a.item()), dist.log_prob(a), dist.entropy()


def _sample_categorical_batch(logits: torch.Tensor, deterministic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    logits: [B, K]
    Returns (a_idx [B], logp [B], entropy [B])
    """
    dist = torch.distributions.Categorical(logits=logits)
    if deterministic:
        a = torch.argmax(logits, dim=-1)
    else:
        a = dist.sample()
    logp = dist.log_prob(a)
    ent = dist.entropy()
    return a, logp, ent


def rollout_batch(
    envs: list[EpisodeEnv],
    model: PolicyNet,
    *,
    device: str,
    deterministic: bool,
) -> tuple[list[int], list[list[float]]]:
    """
    Roll out a batch of episodes in parallel (one forward pass per timestep).

    Returns:
      - pnls_cents per episode
      - rewards per episode (list of per-step rewards) for training
    """
    if not envs:
        return [], []

    # Reset and collect initial obs.
    obs = np.stack([e.reset() for e in envs], axis=0)  # [B, obs_dim]
    B = obs.shape[0]

    # Episodes can have variable candle lengths; run until all are done.
    done = np.zeros((B,), dtype=bool)
    rewards: list[list[float]] = [[] for _ in range(B)]

    while True:
        if done.all():
            break

        obs_t = torch.from_numpy(obs).to(device)
        logits_a, logits_s, v = model(obs_t)

        # Action mask
        mask_np = np.stack([e.valid_action_mask() for e in envs], axis=0)  # [B, 4]
        mask = torch.from_numpy(mask_np).to(device)
        logits_a = logits_a.masked_fill(~mask, -1e9)

        a, logp_a, ent_a = _sample_categorical_batch(logits_a, deterministic=deterministic)

        # Size bucket: only for buy/close; otherwise use 0.
        sidx = torch.zeros((B,), dtype=torch.long, device=device)
        logp_s = torch.zeros((B,), dtype=torch.float32, device=device)
        ent_s = torch.zeros((B,), dtype=torch.float32, device=device)

        needs_size = (a == 1) | (a == 2) | (a == 3)
        if needs_size.any():
            logits_s2 = logits_s.clone()
            logits_s2[:, 0] = -1e9  # mask out size=0
            s_all, lp_all, ent_all = _sample_categorical_batch(logits_s2, deterministic=deterministic)
            sidx = torch.where(needs_size, s_all, sidx)
            logp_s = torch.where(needs_size, lp_all, logp_s)
            ent_s = torch.where(needs_size, ent_all, ent_s)

        # Step envs sequentially (env logic is python), but inference is batched.
        a_np = a.detach().cpu().numpy().astype(int)
        sidx_np = sidx.detach().cpu().numpy().astype(int)

        next_obs = obs.copy()
        for i, e in enumerate(envs):
            if done[i]:
                # Already done; keep dummy obs/reward.
                rewards[i].append(0.0)
                continue
            size = int(model.cfg.size_buckets[int(sidx_np[i])])
            o2, r, d, info = e.step(int(a_np[i]), size)
            rewards[i].append(float(r))
            next_obs[i] = o2
            done[i] = bool(d)

        obs = next_obs

    pnls = [int(e.cash_cents) for e in envs]
    return pnls, rewards


def rollout_episode(
    env: EpisodeEnv,
    model: PolicyNet,
    *,
    train: bool,
    device: str,
    deterministic: bool = False,
) -> EpisodeRollout:
    obs = env.reset()

    obs_t: list[torch.Tensor] = []
    acts: list[int] = []
    size_idxs: list[int] = []
    logps: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    ents: list[torch.Tensor] = []
    rews: list[float] = []

    while True:
        o = torch.from_numpy(obs).to(device)
        logits_a, logits_s, v = model(o.unsqueeze(0))
        logits_a = logits_a.squeeze(0)
        logits_s = logits_s.squeeze(0)

        # Mask invalid actions to avoid learning pathologies around "invalid noop" moves.
        mask = torch.from_numpy(env.valid_action_mask()).to(device)
        logits_a = logits_a.masked_fill(~mask, -1e9)

        if deterministic:
            dist_a = torch.distributions.Categorical(logits=logits_a)
            a = int(torch.argmax(logits_a).item())
            logp_a = dist_a.log_prob(torch.tensor(a, device=device))
            ent_a = dist_a.entropy()
        else:
            a, logp_a, ent_a = _sample_categorical(logits_a)

        # Size bucket: only meaningful for buy/close, but we still sample a bucket
        # (masked so "0 contracts" isn't chosen for buy/close).
        if a in (1, 2, 3):
            # Mask out size=0 by setting logit to -inf.
            logits_s2 = logits_s.clone()
            logits_s2[0] = -1e9
            if deterministic:
                dist_s = torch.distributions.Categorical(logits=logits_s2)
                sidx = int(torch.argmax(logits_s2).item())
                logp_s = dist_s.log_prob(torch.tensor(sidx, device=device))
                ent_s = dist_s.entropy()
            else:
                sidx, logp_s, ent_s = _sample_categorical(logits_s2)
        else:
            sidx = 0
            # Treat size decision as deterministic "0" for hold.
            logp_s = torch.tensor(0.0, device=device)
            ent_s = torch.tensor(0.0, device=device)

        size = model.cfg.size_buckets[sidx]

        next_obs, r, done, info = env.step(a, int(size))

        obs_t.append(o)
        acts.append(a)
        size_idxs.append(sidx)
        logps.append(logp_a + logp_s)
        vals.append(v.squeeze(0))
        ents.append(ent_a + ent_s)
        rews.append(float(r))

        obs = next_obs
        if done:
            pnl = int(info.get("episode_pnl_cents", 0))
            break

    return EpisodeRollout(
        obs=obs_t,
        act=acts,
        size_idx=size_idxs,
        logp=logps,
        val=vals,
        ent=ents,
        rew=rews,
        done=True,
        pnl_cents=pnl,
    )


@torch.no_grad()
def rollout_batch_eval(
    envs: list[EpisodeEnv],
    model: PolicyNet,
    *,
    device: str,
) -> tuple[list[int], int, int]:
    """
    Deterministic batched rollout for evaluation.
    Returns (pnls_cents, total_buys, total_closes).
    """
    if not envs:
        return [], 0, 0

    obs = np.stack([e.reset() for e in envs], axis=0)
    B = obs.shape[0]
    done = np.zeros((B,), dtype=bool)

    while True:
        if done.all():
            break

        obs_t = torch.from_numpy(obs).to(device)
        logits_a, logits_s, _v = model(obs_t)

        mask_np = np.stack([e.valid_action_mask() for e in envs], axis=0)
        mask = torch.from_numpy(mask_np).to(device)
        logits_a = logits_a.masked_fill(~mask, -1e9)

        a, _logp_a, _ent_a = _sample_categorical_batch(logits_a, deterministic=True)

        # Size for buy/close actions; otherwise 0.
        sidx = torch.zeros((B,), dtype=torch.long, device=device)
        needs_size = (a == 1) | (a == 2) | (a == 3)
        if needs_size.any():
            logits_s2 = logits_s.clone()
            logits_s2[:, 0] = -1e9
            s_all, _lp_all, _ent_all = _sample_categorical_batch(logits_s2, deterministic=True)
            sidx = torch.where(needs_size, s_all, sidx)

        a_np = a.detach().cpu().numpy().astype(int)
        sidx_np = sidx.detach().cpu().numpy().astype(int)

        next_obs = obs.copy()
        for i, e in enumerate(envs):
            if done[i]:
                continue
            size = int(model.cfg.size_buckets[int(sidx_np[i])])
            o2, _r, d, _info = e.step(int(a_np[i]), size)
            next_obs[i] = o2
            done[i] = bool(d)
        obs = next_obs

    pnls = [int(e.cash_cents) for e in envs]
    total_buys = sum(int(e.buy_count) for e in envs)
    total_closes = sum(int(e.close_count) for e in envs)
    return pnls, total_buys, total_closes


def train_epoch_batch(
    envs: list[EpisodeEnv],
    model: PolicyNet,
    opt: optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    """
    Batched rollout + A2C-style update.

    This keeps the environment step in Python (Kalshi-specific), but batches model
    inference and loss computation to reduce overhead and improve MPS utilization.
    """
    if not envs:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "mean_pnl_cents": 0.0,
            "std_pnl_cents": 0.0,
        }

    model.train()
    obs = np.stack([e.reset() for e in envs], axis=0)  # [B, obs_dim]
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
        alive_t = torch.from_numpy(alive.astype(np.float32)).to(cfg.device)  # [B]

        obs_t = torch.from_numpy(obs).to(cfg.device)
        logits_a, logits_s, v = model(obs_t)  # [B,4], [B,S], [B]

        mask_np = np.stack([e.valid_action_mask() for e in envs], axis=0)
        mask = torch.from_numpy(mask_np).to(cfg.device)
        logits_a = logits_a.masked_fill(~mask, -1e9)

        a, logp_a, ent_a = _sample_categorical_batch(logits_a, deterministic=False)

        # Size bucket decision
        sidx = torch.zeros((B,), dtype=torch.long, device=cfg.device)
        logp_s = torch.zeros((B,), dtype=torch.float32, device=cfg.device)
        ent_s = torch.zeros((B,), dtype=torch.float32, device=cfg.device)
        needs_size = (a == 1) | (a == 2) | (a == 3)
        if needs_size.any():
            logits_s2 = logits_s.clone()
            logits_s2[:, 0] = -1e9
            s_all, lp_all, ent_all = _sample_categorical_batch(logits_s2, deterministic=False)
            sidx = torch.where(needs_size, s_all, sidx)
            logp_s = torch.where(needs_size, lp_all, logp_s)
            ent_s = torch.where(needs_size, ent_all, ent_s)

        # Step envs.
        a_np = a.detach().cpu().numpy().astype(int)
        sidx_np = sidx.detach().cpu().numpy().astype(int)
        next_obs = obs.copy()
        r_vec = np.zeros((B,), dtype=np.float32)
        for i, e in enumerate(envs):
            if done[i]:
                continue
            size = int(model.cfg.size_buckets[int(sidx_np[i])])
            o2, r, d, _info = e.step(int(a_np[i]), size)
            next_obs[i] = o2
            r_vec[i] = float(r)
            done[i] = bool(d)
        obs = next_obs

        # Store tensors for loss. Mask out steps after done.
        logp_steps.append((logp_a + logp_s) * alive_t)
        v_steps.append(v * alive_t)
        ent_steps.append((ent_a + ent_s) * alive_t)
        rew_steps.append(torch.from_numpy(r_vec).to(cfg.device) * alive_t)
        alive_steps.append(alive_t)

    # Stack: [T, B]
    logp_t = torch.stack(logp_steps, dim=0)
    v_t = torch.stack(v_steps, dim=0)
    ent_t = torch.stack(ent_steps, dim=0)
    rew_t = torch.stack(rew_steps, dim=0)
    alive_t = torch.stack(alive_steps, dim=0)

    # Returns with gamma (reverse discounted cumulative sum)
    if cfg.gamma == 1.0:
        returns = torch.flip(torch.cumsum(torch.flip(rew_t, dims=[0]), dim=0), dims=[0])
    else:
        # Manual discounted return.
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
    entropy_loss = -entropy
    loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()

    pnls = np.asarray([e.cash_cents for e in envs], dtype=np.float32)
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "mean_pnl_cents": float(pnls.mean()) if pnls.size else 0.0,
        "std_pnl_cents": float(pnls.std()) if pnls.size else 0.0,
    }


def train_epoch(
    envs: Iterable[EpisodeEnv],
    model: PolicyNet,
    opt: optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    # Keep train_epoch as the "simple" baseline; walkforward uses it.
    # Batched rollouts are implemented separately to keep changes localized.
    model.train()
    rollouts: list[EpisodeRollout] = []
    for env in envs:
        rollouts.append(rollout_episode(env, model, train=True, device=cfg.device))

    logp = torch.cat([torch.stack(r.logp) for r in rollouts]).to(cfg.device)
    v = torch.cat([torch.stack(r.val) for r in rollouts]).to(cfg.device)
    ent = torch.cat([torch.stack(r.ent) for r in rollouts]).to(cfg.device)

    returns = torch.cat([_discounted_returns(r.rew, cfg.gamma) for r in rollouts]).to(cfg.device)
    adv = returns - v.detach()

    policy_loss = -(logp * adv).mean()
    value_loss = ((returns - v) ** 2).mean()
    entropy_loss = -ent.mean()
    loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()

    pnls = [r.pnl_cents for r in rollouts]
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(ent.mean().item()),
        "mean_pnl_cents": float(np.mean(pnls)) if pnls else 0.0,
        "std_pnl_cents": float(np.std(pnls)) if pnls else 0.0,
    }


@torch.no_grad()
def eval_policy(envs: Iterable[EpisodeEnv], model: PolicyNet, device: str) -> dict[str, float]:
    model.eval()
    pnls = []
    for env in envs:
        r = rollout_episode(env, model, train=False, device=device, deterministic=True)
        pnls.append(r.pnl_cents)
    pnls_np = np.asarray(pnls, dtype=np.float32)
    return {
        "episodes": float(len(pnls)),
        "mean_pnl_cents": float(pnls_np.mean()) if pnls else 0.0,
        "std_pnl_cents": float(pnls_np.std()) if pnls else 0.0,
    }
