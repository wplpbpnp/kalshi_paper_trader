from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from data import Market15m, OHLC


Side = Literal["YES", "NO"]
Fallback = Literal["settle", "taker_exit"]


@dataclass(frozen=True)
class Strategy:
    side: Side
    entry_minute: int
    entry_improve_cents: int  # 0 = join best bid; >0 = improve within spread
    ttl_minutes: int
    tp_cents: int  # 0 means "no take-profit attempt"
    stop_minute: int  # absolute minute index where we begin "flatten" attempts
    stop_exit_ttl_minutes: int  # how long we attempt maker-flatten after stop_minute
    stop_exit_improve_cents: int  # how aggressively to price the stop-limit sell (reduce ask by this many cents)
    max_entry_spread_cents: int = 100  # skip entry if spread is wider than this (cents)
    max_stop_spread_cents: int = 100  # skip stop-maker attempts if spread is wider than this (cents)
    fallback: Fallback = "settle"
    require_resting_entry: bool = True
    require_resting_exit: bool = True
    allow_same_candle_exit: bool = False


@dataclass
class Trade:
    ticker: str
    side: Side
    entry_minute: int
    entry_px: int
    fill_minute: int
    exit_px: Optional[int]
    exit_minute: Optional[int]
    pnl_cents: int
    filled: bool
    exited: bool
    settled: bool


def _quotes_for_side(m: Market15m, minute: int, side: Side) -> tuple[OHLC, OHLC]:
    c = m.candles[minute]
    if side == "YES":
        return c.yes_bid, c.yes_ask
    return c.no_bid, c.no_ask


def _is_win(m: Market15m, side: Side) -> bool:
    # Long YES wins if market settles YES; long NO wins if market settles NO.
    y = m.settles_yes()
    return y if side == "YES" else (not y)

def _taker_fee_cents(price_cents: int) -> int:
    """
    Approx Kalshi fee per contract for a taker fill:
      fee_dollars = round_up(0.07 * P * (1-P)) where P in dollars.
    Convert to cents and round up to nearest cent.
    """
    p = max(0.01, min(0.99, price_cents / 100.0))
    fee_cents = 7.0 * p * (1.0 - p)  # 0.07 dollars -> 7 cents
    return int(fee_cents) if abs(fee_cents - int(fee_cents)) < 1e-12 else int(fee_cents) + 1


def simulate_market(m: Market15m, s: Strategy) -> Optional[Trade]:
    n = len(m.candles)
    if n == 0:
        return None
    if s.entry_minute < 0 or s.entry_minute >= n:
        return None
    if s.ttl_minutes <= 0:
        return None
    if s.tp_cents < 0:
        return None
    if s.stop_minute < 0:
        return None
    if s.stop_exit_ttl_minutes <= 0:
        return None
    if s.stop_exit_improve_cents < 0:
        return None

    bid0, ask0 = _quotes_for_side(m, s.entry_minute, s.side)
    if bid0.open is None or ask0.open is None:
        return None
    if ask0.open <= bid0.open:
        # Crossed/invalid book snapshot; skip.
        return None
    entry_spread = ask0.open - bid0.open
    if entry_spread > s.max_entry_spread_cents:
        return None

    # Entry: join/improve the best bid but do not cross the ask.
    entry_px = bid0.open + s.entry_improve_cents
    entry_px = min(entry_px, ask0.open - 1)
    entry_px = max(1, min(99, entry_px))

    if s.require_resting_entry and entry_px >= ask0.open:
        return None

    # Find first fill: when best ask trades down to <= our bid within TTL.
    end_fill = min(n - 1, s.entry_minute + s.ttl_minutes - 1)
    fill_minute: Optional[int] = None
    for t in range(s.entry_minute, end_fill + 1):
        _, ask = _quotes_for_side(m, t, s.side)
        if ask.low is None:
            continue
        if ask.low <= entry_px:
            fill_minute = t
            break

    if fill_minute is None:
        return Trade(
            ticker=m.ticker,
            side=s.side,
            entry_minute=s.entry_minute,
            entry_px=entry_px,
            fill_minute=-1,
            exit_px=None,
            exit_minute=None,
            pnl_cents=0,
            filled=False,
            exited=False,
            settled=False,
        )

    # Attempt TP exits (if configured) until stop_minute.
    last_minute = n - 1
    exit_start = fill_minute if s.allow_same_candle_exit else min(last_minute, fill_minute + 1)
    tp_end = min(last_minute, max(exit_start, s.stop_minute - 1))

    if s.tp_cents > 0 and exit_start <= tp_end:
        tp_px = max(1, min(99, entry_px + s.tp_cents))
        bid_at_tp_start, ask_at_tp_start = _quotes_for_side(m, exit_start, s.side)
        if bid_at_tp_start.open is None or ask_at_tp_start.open is None:
            return None
        # If we require resting exits, ensure TP isn't instantly marketable (would be taker).
        if not (s.require_resting_exit and tp_px <= bid_at_tp_start.open):
            tp_filled: Optional[int] = None
            for t in range(exit_start, tp_end + 1):
                bid, _ = _quotes_for_side(m, t, s.side)
                if bid.high is None:
                    continue
                if bid.high >= tp_px:
                    tp_filled = t
                    break
            if tp_filled is not None:
                pnl = tp_px - entry_px
                return Trade(
                    ticker=m.ticker,
                    side=s.side,
                    entry_minute=s.entry_minute,
                    entry_px=entry_px,
                    fill_minute=fill_minute,
                    exit_px=tp_px,
                    exit_minute=tp_filled,
                    pnl_cents=pnl,
                    filled=True,
                    exited=True,
                    settled=False,
                )

    # Time-stop: attempt maker-flatten starting at stop_minute (or after fill).
    stop_start = max(exit_start, min(last_minute, s.stop_minute))
    stop_bid0, stop_ask0 = _quotes_for_side(m, stop_start, s.side)
    if stop_bid0.open is None or stop_ask0.open is None:
        return None
    if stop_ask0.open <= stop_bid0.open:
        return None
    stop_spread = stop_ask0.open - stop_bid0.open
    if stop_spread <= s.max_stop_spread_cents:
        # Price an exit ask: join current ask, optionally improve it, but never cross the bid.
        stop_exit_px = stop_ask0.open - s.stop_exit_improve_cents
        stop_exit_px = max(stop_bid0.open + 1, stop_exit_px)
        stop_exit_px = max(1, min(99, stop_exit_px))

        # Only count as maker if it doesn't instantly cross.
        if not (s.require_resting_exit and stop_exit_px <= stop_bid0.open):
            stop_end = min(last_minute, stop_start + s.stop_exit_ttl_minutes - 1)
            stop_filled: Optional[int] = None
            for t in range(stop_start, stop_end + 1):
                bid, _ = _quotes_for_side(m, t, s.side)
                if bid.high is None:
                    continue
                if bid.high >= stop_exit_px:
                    stop_filled = t
                    break
            if stop_filled is not None:
                pnl = stop_exit_px - entry_px
                return Trade(
                    ticker=m.ticker,
                    side=s.side,
                    entry_minute=s.entry_minute,
                    entry_px=entry_px,
                    fill_minute=fill_minute,
                    exit_px=stop_exit_px,
                    exit_minute=stop_filled,
                    pnl_cents=pnl,
                    filled=True,
                    exited=True,
                    settled=False,
                )

        # Maker stop didn't fill within TTL; optionally force taker flatten at stop_end.
        if s.fallback == "taker_exit":
            stop_end = min(last_minute, stop_start + s.stop_exit_ttl_minutes - 1)
            bid_end, _ = _quotes_for_side(m, stop_end, s.side)
            px = bid_end.close if bid_end.close is not None else bid_end.open
            if px is None:
                return None
            taker_exit_px = max(0, min(100, int(px)))
            fee = _taker_fee_cents(taker_exit_px)
            pnl = taker_exit_px - entry_px - fee
            return Trade(
                ticker=m.ticker,
                side=s.side,
                entry_minute=s.entry_minute,
                entry_px=entry_px,
                fill_minute=fill_minute,
                exit_px=taker_exit_px,
                exit_minute=stop_end,
                pnl_cents=pnl,
                filled=True,
                exited=True,
                settled=False,
            )

    # settle
    payoff = 100 if _is_win(m, s.side) else 0
    pnl = payoff - entry_px
    return Trade(
        ticker=m.ticker,
        side=s.side,
        entry_minute=s.entry_minute,
        entry_px=entry_px,
        fill_minute=fill_minute,
        exit_px=None,
        exit_minute=None,
        pnl_cents=pnl,
        filled=True,
        exited=False,
        settled=True,
    )
