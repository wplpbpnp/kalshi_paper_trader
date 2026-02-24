from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Bar:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trade_count: int
    taker_buy_base_vol: float
    taker_buy_quote_vol: float

    @property
    def date_utc(self) -> str:
        dt = datetime.fromtimestamp(self.open_time_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")


def load_bars_csv(path: str | Path) -> list[Bar]:
    p = Path(path)
    out: list[Bar] = []
    with p.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # This loader is intentionally permissive so disparate sources can be merged into
            # a single "universal" OHLCV schema. Missing microstructure fields are treated
            # as 0, and quote_volume defaults to volume when absent.
            vol = float(row.get("volume") or 0.0)
            out.append(
                Bar(
                    open_time_ms=int(row["open_time_ms"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=vol,
                    quote_volume=float(row.get("quote_volume") or vol),
                    trade_count=int(float(row.get("trade_count") or 0)),
                    taker_buy_base_vol=float(row.get("taker_buy_base_vol") or 0.0),
                    taker_buy_quote_vol=float(row.get("taker_buy_quote_vol") or 0.0),
                )
            )
    out.sort(key=lambda b: b.open_time_ms)
    return out


def iter_days(bars: Iterable[Bar]) -> list[str]:
    return sorted({b.date_utc for b in bars})
