from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .imports import add_maker_strat_to_path

add_maker_strat_to_path()

from data import Market15m  # type: ignore  # noqa: E402
from sim import _taker_fee_cents  # type: ignore  # noqa: E402


def _clip_price(x: int) -> int:
    return max(1, min(99, int(x)))


@dataclass
class StepInfo:
    ticker: str
    close_date: str
    minute: int
    action: int
    size: int
    executed: bool
    pos_side: int
    cash_cents: int


class EpisodeEnv:
    """
    One-episode environment for a single 15-minute Kalshi market.

    Observations are derived solely from candle top-of-book:
      - [minute_idx, time_to_end]
      - for last `lookback` minutes: yes_bid_open, yes_ask_open, spread
      - current position side (YES=+1, NO=-1, flat=0)
      - entry price (0 if flat)
    All prices are normalized to [0,1] by dividing by 100.

    Actions (discrete):
      0 = hold/no-op
      1 = buy YES (taker at ask.open) if flat
      2 = buy NO  (taker at ask.open) if flat
      3 = close (taker at bid.open) if in position

    Size is an integer number of contracts to trade for buy/close actions.
    """

    def __init__(
        self,
        market: Market15m,
        *,
        lookback: int = 5,
        max_contracts: int = 5,
        allow_close: bool = True,
        features: str = "raw",
    ):
        self.m = market
        self.lookback = int(lookback)
        self.max_contracts = int(max_contracts)
        self.allow_close = bool(allow_close)
        self.features = str(features)

        self.t = 0
        self.pos_side = 0  # -1 NO, 0 flat, +1 YES
        self.pos_qty = 0
        self.entry_px = 0
        self.cash_cents = 0

        self.buy_count = 0
        self.close_count = 0
        self._entered_once = False

        # Track for logging.
        self._actions_taken: list[StepInfo] = []

        # Cache per-minute market features so training isn't dominated by Python loops.
        self._build_cache()

    def _build_cache(self) -> None:
        n = len(self.m.candles)
        # Precompute YES/NO bid/ask/spread at candle open; -1 indicates invalid/missing.
        self._yes_bid = np.full((n,), -1, dtype=np.int16)
        self._yes_ask = np.full((n,), -1, dtype=np.int16)
        self._yes_spread = np.full((n,), -1, dtype=np.int16)
        self._no_bid = np.full((n,), -1, dtype=np.int16)
        self._no_ask = np.full((n,), -1, dtype=np.int16)
        self._no_spread = np.full((n,), -1, dtype=np.int16)
        self._has_yes = np.zeros((n,), dtype=bool)
        self._has_no = np.zeros((n,), dtype=bool)
        # Derived YES-only signals (cents); kept as floats for convenience.
        self._mid = np.zeros((n,), dtype=np.float32)
        self._spr = np.zeros((n,), dtype=np.float32)

        for t in range(n):
            y = self._yes_book_open(t)
            if y is not None:
                bid, ask, spr = y
                self._yes_bid[t] = bid
                self._yes_ask[t] = ask
                self._yes_spread[t] = spr
                self._has_yes[t] = True
                self._mid[t] = (float(bid) + float(ask)) / 2.0
                self._spr[t] = float(spr)

            no = self._no_book_open(t)
            if no is not None:
                bid, ask, spr = no
                self._no_bid[t] = bid
                self._no_ask[t] = ask
                self._no_spread[t] = spr
                self._has_no[t] = True

        # Precompute base observation features that don't depend on position state.
        # Shape: [n, base_dim]
        base_dim = 2 + 3 * self.lookback
        mode = self.features.lower()
        if mode in ("deltas", "raw+deltas"):
            base_dim += 5  # mid, mom1, mom3, mom5, spread
        elif mode in ("deltas+vol", "raw+deltas+vol"):
            base_dim += 9  # deltas block + vol/spread-change block

        self._base_obs = np.zeros((n, base_dim), dtype=np.float32)
        for t in range(n):
            minute_idx = 0.0 if n <= 1 else (t / (n - 1))
            time_to_end = 0.0 if n <= 1 else ((n - 1 - t) / (n - 1))
            j = 0
            self._base_obs[t, j] = minute_idx
            self._base_obs[t, j + 1] = time_to_end
            j += 2
            for k in range(self.lookback):
                minute = t - (self.lookback - 1 - k)
                if 0 <= minute < n and self._has_yes[minute]:
                    bid = float(self._yes_bid[minute]) / 100.0
                    ask = float(self._yes_ask[minute]) / 100.0
                    spr = float(self._yes_spread[minute]) / 100.0
                    self._base_obs[t, j : j + 3] = (bid, ask, spr)
                else:
                    self._base_obs[t, j : j + 3] = (0.0, 0.0, 0.0)
                j += 3

            if mode in ("deltas", "raw+deltas", "deltas+vol", "raw+deltas+vol"):
                # Helper features derived from the same candle stream (no new information).
                mid = float(self._mid[t]) if self._has_yes[t] else 0.0
                spr_c = float(self._spr[t]) if self._has_yes[t] else 0.0

                def mid_at(dt: int) -> float:
                    i = t - dt
                    if 0 <= i < n and self._has_yes[i]:
                        return float(self._mid[i])
                    return 0.0

                def spr_at(dt: int) -> float:
                    i = t - dt
                    if 0 <= i < n and self._has_yes[i]:
                        return float(self._spr[i])
                    return 0.0

                mom1 = mid - mid_at(1)
                mom3 = mid - mid_at(3)
                mom5 = mid - mid_at(5)

                self._base_obs[t, j : j + 5] = (mid / 100.0, mom1 / 100.0, mom3 / 100.0, mom5 / 100.0, spr_c / 100.0)
                j += 5

                if mode in ("deltas+vol", "raw+deltas+vol"):
                    # Rolling std of 1-step mid changes (over last K changes).
                    def vol_k(k: int) -> float:
                        diffs = []
                        for d in range(1, k + 1):
                            a = mid_at(d - 1)
                            b = mid_at(d)
                            if a != 0.0 and b != 0.0:
                                diffs.append(a - b)
                        if len(diffs) < 2:
                            return 0.0
                        x = np.asarray(diffs, dtype=np.float32)
                        return float(x.std())

                    vol3 = vol_k(3)
                    vol5 = vol_k(5)
                    spr1 = spr_c - spr_at(1)
                    spr3 = spr_c - spr_at(3)
                    self._base_obs[t, j : j + 4] = (vol3 / 100.0, vol5 / 100.0, spr1 / 100.0, spr3 / 100.0)
                    j += 4

    def reset(self) -> np.ndarray:
        self.t = 0
        self.pos_side = 0
        self.pos_qty = 0
        self.entry_px = 0
        self.cash_cents = 0
        self.buy_count = 0
        self.close_count = 0
        self._entered_once = False
        self._actions_taken = []
        return self._obs()

    def done(self) -> bool:
        return self.t >= len(self.m.candles)

    def _yes_book_open(self, minute: int) -> Optional[tuple[int, int, int]]:
        if minute < 0 or minute >= len(self.m.candles):
            return None
        c = self.m.candles[minute]
        yb = c.yes_bid.open
        ya = c.yes_ask.open
        if yb is None or ya is None or ya <= yb:
            return None
        bid = _clip_price(yb)
        ask = _clip_price(ya)
        spread = max(1, ask - bid)
        return bid, ask, spread

    def _no_book_open(self, minute: int) -> Optional[tuple[int, int, int]]:
        # Parity mapping from YES book.
        y = self._yes_book_open(minute)
        if y is None:
            return None
        yb, ya, _ = y
        bid = _clip_price(100 - ya)
        ask = _clip_price(100 - yb)
        spread = max(1, ask - bid)
        if ask <= bid:
            return None
        return bid, ask, spread

    def _obs(self) -> np.ndarray:
        n = len(self.m.candles)
        t = self.t
        if n == 0 or t < 0 or t >= n:
            base_dim = 2 + 3 * self.lookback
            mode = self.features.lower()
            if mode in ("deltas", "raw+deltas"):
                base_dim += 5
            elif mode in ("deltas+vol", "raw+deltas+vol"):
                base_dim += 9
            return np.zeros((base_dim + 3,), dtype=np.float32)
        base = self._base_obs[t]
        pos = np.asarray(
            [
                float(self.pos_side),
                (self.entry_px / 100.0) if self.pos_side != 0 else 0.0,
                (self.pos_qty / max(1, self.max_contracts)) if self.pos_side != 0 else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, pos], axis=0)

    def valid_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of shape (4,) for actions [hold, buy_yes, buy_no, close]
        that are valid at the current timestep.
        """
        if self.done():
            return np.zeros((4,), dtype=bool)
        has_yes = bool(self._has_yes[self.t])
        has_no = bool(self._has_no[self.t])
        if self.pos_side == 0:
            # Limit to 1 entry per episode: once we have entered and later flattened,
            # disallow additional entries for the remainder of the episode.
            if self._entered_once:
                return np.asarray([True, False, False, False], dtype=bool)
            return np.asarray([True, has_yes, has_no, False], dtype=bool)
        # In a position: allow close only if the corresponding book exists.
        if not self.allow_close:
            return np.asarray([True, False, False, False], dtype=bool)
        if self.pos_side == +1:
            return np.asarray([True, False, False, has_yes], dtype=bool)
        return np.asarray([True, False, False, has_no], dtype=bool)

    def step(self, action: int, size: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Returns (obs, reward, done, info).
        Reward is delta cash for this step, in cents.
        """
        if self.done():
            return self._obs(), 0.0, True, {}

        action = int(action)
        size = int(size)
        size = max(0, min(self.max_contracts, size))

        reward_cents = 0
        info: dict = {}
        executed = False

        # Current books (from cache).
        yes = None
        no = None
        if self._has_yes[self.t]:
            yes = (int(self._yes_bid[self.t]), int(self._yes_ask[self.t]), int(self._yes_spread[self.t]))
        if self._has_no[self.t]:
            no = (int(self._no_bid[self.t]), int(self._no_ask[self.t]), int(self._no_spread[self.t]))

        if action == 1 and self.pos_side == 0 and size > 0 and yes is not None:
            # Buy YES at ask (taker)
            _, ask, _ = yes
            px = _clip_price(ask)
            fee = _taker_fee_cents(px) * size
            self.cash_cents -= px * size
            self.cash_cents -= fee
            reward_cents -= (px * size + fee)
            self.pos_side = +1
            self.pos_qty = size
            self.entry_px = px
            executed = True
            self.buy_count += 1
            self._entered_once = True
        elif action == 2 and self.pos_side == 0 and size > 0 and no is not None:
            # Buy NO at ask (taker)
            _, ask, _ = no
            px = _clip_price(ask)
            fee = _taker_fee_cents(px) * size
            self.cash_cents -= px * size
            self.cash_cents -= fee
            reward_cents -= (px * size + fee)
            self.pos_side = -1
            self.pos_qty = size
            self.entry_px = px
            executed = True
            self.buy_count += 1
            self._entered_once = True
        elif action == 3 and self.allow_close and self.pos_side != 0 and size > 0:
            # Close position (taker) at bid of the held side.
            qty = min(self.pos_qty, size)
            book = yes if self.pos_side == +1 else no
            if book is not None:
                bid, _, _ = book
                px = _clip_price(bid)
                fee = _taker_fee_cents(px) * qty
                self.cash_cents += px * qty
                self.cash_cents -= fee
                reward_cents += (px * qty - fee)
                self.pos_qty -= qty
                if self.pos_qty == 0:
                    self.pos_side = 0
                    self.entry_px = 0
                executed = True
                self.close_count += 1

        self._actions_taken.append(
            StepInfo(
                ticker=self.m.ticker,
                close_date=self.m.close_date,
                minute=self.t,
                action=action,
                size=size,
                executed=executed,
                pos_side=self.pos_side,
                cash_cents=self.cash_cents,
            )
        )

        self.t += 1

        done = self.done()
        if done:
            # Settle any remaining open position.
            if self.pos_side != 0 and self.pos_qty > 0:
                settles_yes = self.m.settles_yes()
                win = settles_yes if self.pos_side == +1 else (not settles_yes)
                payoff = 100 if win else 0
                self.cash_cents += payoff * self.pos_qty
                reward_cents += payoff * self.pos_qty
                self.pos_side = 0
                self.pos_qty = 0
                self.entry_px = 0

            info["episode_pnl_cents"] = self.cash_cents

        return self._obs(), float(reward_cents), done, info

    def action_log(self) -> list[StepInfo]:
        return list(self._actions_taken)
