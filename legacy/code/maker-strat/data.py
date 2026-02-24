from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Optional


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


@dataclass(frozen=True)
class OHLC:
    open: Optional[int]
    high: Optional[int]
    low: Optional[int]
    close: Optional[int]

    def ok(self) -> bool:
        return self.open is not None and self.high is not None and self.low is not None and self.close is not None


def _parse_ohlc(d: dict[str, Any]) -> OHLC:
    return OHLC(
        open=_to_int(d.get("open")),
        high=_to_int(d.get("high")),
        low=_to_int(d.get("low")),
        close=_to_int(d.get("close")),
    )


def _comp(x: Optional[int]) -> Optional[int]:
    return None if x is None else 100 - x


def complement_ohlc(ohlc: OHLC) -> OHLC:
    """
    Convert YES -> NO using parity mapping:
      NO_bid = 100 - YES_ask
      NO_ask = 100 - YES_bid

    For OHLC, complement flips high/low.
    """
    return OHLC(
        open=_comp(ohlc.open),
        high=_comp(ohlc.low),
        low=_comp(ohlc.high),
        close=_comp(ohlc.close),
    )


@dataclass(frozen=True)
class Candle:
    end_period_ts: int
    yes_bid: OHLC
    yes_ask: OHLC

    @property
    def no_bid(self) -> OHLC:
        return complement_ohlc(self.yes_ask)

    @property
    def no_ask(self) -> OHLC:
        return complement_ohlc(self.yes_bid)


@dataclass(frozen=True)
class Market15m:
    ticker: str
    close_time: str  # ISO string
    strike: float
    result: str  # "yes" or "no"
    candles: tuple[Candle, ...]

    @property
    def close_date(self) -> str:
        # Used for block bootstrap grouping (dependence within day).
        dt = datetime.strptime(self.close_time, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%Y-%m-%d")

    def settles_yes(self) -> bool:
        return self.result.lower() == "yes"


def load_markets(path: str | Path) -> list[Market15m]:
    p = Path(path)
    raw = json.loads(p.read_text())
    out: list[Market15m] = []

    for m in raw:
        candles_raw = m.get("candlesticks") or []
        candles: list[Candle] = []
        for c in candles_raw:
            yb = c.get("yes_bid") or {}
            ya = c.get("yes_ask") or {}
            candles.append(
                Candle(
                    end_period_ts=int(c.get("end_period_ts")),
                    yes_bid=_parse_ohlc(yb),
                    yes_ask=_parse_ohlc(ya),
                )
            )

        # Some markets can have missing/noisy candles; keep them and let the simulator skip invalid points.
        out.append(
            Market15m(
                ticker=m["ticker"],
                close_time=m["close_time"],
                strike=float(m.get("strike") or 0.0),
                result=str(m.get("result") or "").lower(),
                candles=tuple(candles),
            )
        )

    return out


def iter_minutes(market: Market15m) -> Iterable[int]:
    return range(len(market.candles))

