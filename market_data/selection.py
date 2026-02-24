from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso_z(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


@dataclass
class MarketSelection:
    ticker: str
    close_time: datetime
    market: Dict[str, Any]


def pick_active_open_market(markets: list[Dict[str, Any]]) -> Optional[MarketSelection]:
    now = utcnow()
    best_active: Optional[MarketSelection] = None
    best_future: Optional[MarketSelection] = None

    for m in markets:
        ticker = m.get("ticker")
        close_time_s = m.get("close_time")
        if not ticker or not close_time_s:
            continue

        try:
            close_time = parse_iso_z(close_time_s)
        except Exception:
            continue

        open_time = None
        open_time_s = m.get("open_time")
        if open_time_s:
            try:
                open_time = parse_iso_z(open_time_s)
            except Exception:
                open_time = None

        if open_time is not None and open_time <= now < close_time:
            if best_active is None or close_time < best_active.close_time:
                best_active = MarketSelection(ticker=ticker, close_time=close_time, market=m)
            continue

        if close_time > now:
            if best_future is None or close_time < best_future.close_time:
                best_future = MarketSelection(ticker=ticker, close_time=close_time, market=m)

    return best_active or best_future

