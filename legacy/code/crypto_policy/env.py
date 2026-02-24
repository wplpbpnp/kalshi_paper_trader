from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Optional

import numpy as np

from .data import Bar


@dataclass
class StepInfo:
    t: int
    action: int
    executed: bool
    pos: int
    equity: float


@dataclass(frozen=True)
class SeriesCache:
    """
    Precomputed arrays derived from the full bar series, reused across many episodes.
    """

    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    quote_vol: np.ndarray
    taker_q: np.ndarray
    log_ret: np.ndarray
    range_: np.ndarray
    taker_ratio: np.ndarray
    tod_sin: np.ndarray
    tod_cos: np.ndarray


def _infer_bars_per_day(bars: list[Bar], *, fallback: int) -> int:
    """
    Infer bars/day from the dataset (median daily bar count).

    This matters for non-24/7 venues (equities/ETFs) where bar_minutes->24h is wrong.
    """
    by_day: dict[str, int] = {}
    for b in bars:
        by_day[b.date_utc] = by_day.get(b.date_utc, 0) + 1
    counts = [c for c in by_day.values() if c > 0]
    if not counts:
        return int(max(1, fallback))
    try:
        m = int(median(counts))
    except Exception:
        m = int(max(1, fallback))
    # Clamp to a sensible range.
    return int(max(1, min(m, int(max(1, fallback)))))


def build_series_cache(bars: list[Bar], *, bar_minutes: int, bars_per_day: int | None = None) -> SeriesCache:
    # Build arrays for the full series for fast slicing.
    n = len(bars)
    close = np.asarray([b.close for b in bars], dtype=np.float32)
    high = np.asarray([b.high for b in bars], dtype=np.float32)
    low = np.asarray([b.low for b in bars], dtype=np.float32)
    quote_vol = np.asarray([b.quote_volume for b in bars], dtype=np.float32)
    taker_q = np.asarray([b.taker_buy_quote_vol for b in bars], dtype=np.float32)

    # Close-to-close log returns (additive under compounding):
    # log_ret[t] corresponds to log(close[t] / close[t-1]) for t>=1, else 0.
    log_ret = np.zeros((n,), dtype=np.float32)
    log_ret[1:] = np.log(close[1:] / np.maximum(1e-12, close[:-1]))

    # Range proxy
    range_ = (high - low) / np.maximum(1e-12, close)

    # Order flow proxy: taker buy quote / total quote volume in [0,1] if volumes exist
    denom = np.maximum(1e-12, quote_vol)
    taker_ratio = np.clip(taker_q / denom, 0.0, 1.0)

    # Precompute per-index time-of-day sin/cos for the given bar size.
    fallback = int(round((24 * 60) / max(1, int(bar_minutes))))
    bpd = int(bars_per_day) if bars_per_day is not None and int(bars_per_day) > 0 else _infer_bars_per_day(bars, fallback=fallback)
    idx = np.arange(n, dtype=np.int32)
    tod = (idx % bpd) / max(1, bpd)
    tod_sin = np.sin(2 * np.pi * tod).astype(np.float32)
    tod_cos = np.cos(2 * np.pi * tod).astype(np.float32)

    return SeriesCache(
        close=close,
        high=high,
        low=low,
        quote_vol=quote_vol,
        taker_q=taker_q,
        log_ret=log_ret,
        range_=range_,
        taker_ratio=taker_ratio,
        tod_sin=tod_sin,
        tod_cos=tod_cos,
    )


class SeriesEnv:
    """
    Episodic environment over a contiguous slice of OHLCV bars.

    Position is in {-1, 0, +1} representing short/flat/long at 1x notional.

    Reward is in *basis points* (bp) of log-wealth change, using close-to-close log returns:

      r_t = pos_after_action * log(close[t+1]/close[t]) * 1e4 + log(1 - costs_frac) * 1e4

    Costs are linear in turnover:
      costs_bp = (fee_bps + slippage_bps) * abs(delta_pos)

    Action space (discrete):
      0 = hold (keep current pos)
      1 = go long  (+1)
      2 = go short (-1)
      3 = go flat  (0)

    Observation = engineered features from last `lookback` bars + time features + current pos.
    Feature modes:
      - basic: [log_ret, range, log1p(volume), taker_ratio-0.5] per bar
      - prices: [log_ret, range, log1p(volume)] per bar (for venues without order-flow fields)
    """

    def __init__(
        self,
        bars: list[Bar],
        *,
        start: int,
        length: int,
        bar_minutes: int,
        lookback: int = 16,
        fee_bps: float = 1.0,
        slippage_bps: float = 1.0,
        features: str = "basic",
        one_decision_per_episode: bool = False,
        allow_one_exit: bool = False,
        force_flat_at_end: bool = False,
        require_position: bool = False,
        long_only: bool = False,
        cache: Optional[SeriesCache] = None,
    ):
        self.bars = bars
        self.start = int(start)
        self.length = int(length)
        self.bar_minutes = int(bar_minutes)
        self.lookback = int(lookback)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.features = str(features)
        self.one_decision_per_episode = bool(one_decision_per_episode)
        self.allow_one_exit = bool(allow_one_exit)
        self.force_flat_at_end = bool(force_flat_at_end)
        self.require_position = bool(require_position)
        self.long_only = bool(long_only)

        self.t = 0
        self.pos = 0
        self.equity_bp = 0.0
        self._log: list[StepInfo] = []
        self._decision_made = False
        self._exit_used = False

        self._cache = cache if cache is not None else build_series_cache(bars, bar_minutes=self.bar_minutes)

    def reset(self) -> np.ndarray:
        self.t = 0
        self.pos = 0
        self.equity_bp = 0.0
        self._log = []
        self._decision_made = False
        self._exit_used = False
        return self._obs()

    def done(self) -> bool:
        # Need t+1 for return calculation; stop at last usable bar in episode.
        return self.t >= (self.length - 1)

    def valid_action_mask(self) -> np.ndarray:
        if self.done():
            return np.zeros((4,), dtype=bool)

        if not self.one_decision_per_episode:
            # All actions always valid; costs apply on turnover.
            mask = np.ones((4,), dtype=bool)
            if self.long_only:
                mask[2] = False  # disallow go short
            return mask

        # One-decision mode:
        # - At the first step, choose target pos once (flat/long/short).
        # - Thereafter, force hold; optionally allow a single exit to flat.
        mask = np.zeros((4,), dtype=bool)
        if not self._decision_made:
            # First decision: choose pos (including staying flat).
            mask[:] = True
            # If we're in one-decision mode and require a position, disallow the
            # "stay flat" outcomes at the first decision. From a flat start,
            # that means disallow hold (0) and go-flat (3).
            if self.require_position:
                mask[0] = False
                mask[3] = False
            if self.long_only:
                mask[2] = False  # disallow go short
            return mask

        # After initial decision:
        # Always allow hold.
        mask[0] = True

        # Optional one-time exit to flat if currently in a position.
        if self.allow_one_exit and (self.pos != 0) and (not self._exit_used):
            mask[3] = True
        return mask

    def _obs(self) -> np.ndarray:
        # Observation for the current step uses info up to current bar close (index i).
        i = self.start + self.t
        n = self._cache.close.shape[0]
        if i < 0 or i >= n:
            return np.zeros((1,), dtype=np.float32)

        feats: list[float] = []

        # Time features.
        feats.append(float(self.t) / max(1.0, float(self.length - 1)))
        feats.append(float(self._cache.tod_sin[i]))
        feats.append(float(self._cache.tod_cos[i]))

        mode = str(self.features).lower().strip()
        use_taker = mode not in ("prices", "price", "prices-only", "price-only")

        # Lookback features
        for k in range(self.lookback):
            j = i - (self.lookback - 1 - k)
            if j <= 0 or j >= n:
                feats.extend([0.0, 0.0, 0.0] + ([0.0] if use_taker else []))
                continue
            feats.append(float(self._cache.log_ret[j]))  # already small
            feats.append(float(self._cache.range_[j]))
            feats.append(float(np.log1p(self._cache.quote_vol[j])))
            if use_taker:
                feats.append(float(self._cache.taker_ratio[j] - 0.5))  # centered

        feats.append(float(self.pos))
        return np.asarray(feats, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done():
            return self._obs(), 0.0, True, {"episode_pnl_bp": float(self.equity_bp)}

        action = int(action)
        # Enforce action constraints early (treat invalid actions as hold).
        if self.one_decision_per_episode:
            mask = self.valid_action_mask()
            if action < 0 or action >= 4 or (not bool(mask[action])):
                action = 0

        tgt = self.pos
        if action == 1:
            tgt = 1
        elif action == 2:
            tgt = -1
        elif action == 3:
            tgt = 0
        if self.long_only:
            tgt = 0 if tgt == -1 else tgt

        delta = abs(tgt - self.pos)
        cost_bp = (self.fee_bps + self.slippage_bps) * float(delta)
        # Convert linear bp cost to log-wealth change in bp.
        cost_frac = min(0.999999, max(0.0, cost_bp / 1e4))
        cost_log_bp = math.log1p(-cost_frac) * 1e4  # negative or 0

        # Fast path: in one-decision mode without an exit option, the entire episode
        # reward is determined by the initial target position and future returns.
        # This collapses the episode to a single step (contextual bandit).
        if self.one_decision_per_episode and (not self._decision_made) and (not self.allow_one_exit):
            i = self.start + self.t
            end = min(self._cache.close.shape[0], self.start + self.length)
            # Sum close-to-close log returns for bars (i+1 .. end-1), because log_ret[j] is log ret from j-1->j.
            if i + 1 < end:
                pnl = float(tgt) * float(self._cache.log_ret[i + 1 : end].sum()) * 1e4
            else:
                pnl = 0.0
            reward_bp = pnl + cost_log_bp

            # Apply the action and log it.
            self.pos = tgt
            self.equity_bp += reward_bp
            self._log.append(StepInfo(t=self.t, action=action, executed=(delta != 0), pos=self.pos, equity=float(self.equity_bp)))

            self._decision_made = True

            # Optional liquidation at end.
            if self.force_flat_at_end and (self.pos != 0):
                liq_delta = abs(self.pos)
                liq_cost_bp = (self.fee_bps + self.slippage_bps) * float(liq_delta)
                liq_frac = min(0.999999, max(0.0, liq_cost_bp / 1e4))
                liq_cost_log_bp = math.log1p(-liq_frac) * 1e4
                self.equity_bp += liq_cost_log_bp
                reward_bp += liq_cost_log_bp
                self.pos = 0
                self._log.append(StepInfo(t=self.length - 1, action=3, executed=True, pos=self.pos, equity=float(self.equity_bp)))

            # Jump to end.
            self.t = self.length - 1
            return self._obs(), reward_bp, True, {"episode_pnl_bp": float(self.equity_bp)}

        # Apply action at current close; earn next bar's close-to-close return.
        i = self.start + self.t
        r_next = float(self._cache.log_ret[i + 1])
        reward_bp = float(tgt) * r_next * 1e4 + cost_log_bp

        self.pos = tgt
        self.equity_bp += reward_bp

        self._log.append(StepInfo(t=self.t, action=action, executed=(delta != 0), pos=self.pos, equity=float(self.equity_bp)))

        if self.one_decision_per_episode:
            if not self._decision_made:
                self._decision_made = True
            elif (action == 3) and (delta != 0):
                # Exited to flat after having made a decision.
                self._exit_used = True

        self.t += 1
        done = self.done()
        if done and self.force_flat_at_end and (self.pos != 0):
            # Liquidate at the episode end so successive episodes/windows are comparable.
            liq_delta = abs(0 - self.pos)
            liq_cost_bp = (self.fee_bps + self.slippage_bps) * float(liq_delta)
            liq_frac = min(0.999999, max(0.0, liq_cost_bp / 1e4))
            liq_cost_log_bp = math.log1p(-liq_frac) * 1e4
            self.equity_bp += liq_cost_log_bp
            reward_bp += liq_cost_log_bp
            self.pos = 0
            # Log liquidation as a final synthetic step so turnover/cost accounting includes it.
            self._log.append(
                StepInfo(t=self.t, action=3, executed=True, pos=self.pos, equity=float(self.equity_bp))
            )

        info = {"episode_pnl_bp": float(self.equity_bp)} if done else {}
        return self._obs(), reward_bp, done, info

    def action_log(self) -> list[StepInfo]:
        return list(self._log)
