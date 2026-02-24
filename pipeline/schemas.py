from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SnapshotRecord:
    ts_ms: int
    market_ticker: str
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    yes_mid: Optional[float]
    spread: Optional[int]


@dataclass(frozen=True)
class StrategySpec:
    """
    Universal strategy payload.

    `strategy_type` can be a built-in runtime type (example: `rnn_edge_v1`) or
    `plugin`, in which case `plugin` must include `module` and `class`.
    """

    schema_version: str
    strategy_id: str
    created_at_utc: str
    strategy_type: str
    markets: Dict[str, Any]
    model: Dict[str, Any]
    signals: Dict[str, Any]
    execution: Dict[str, Any]
    risk: Dict[str, Any]
    plugin: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "created_at_utc": self.created_at_utc,
            "strategy_type": self.strategy_type,
            "markets": self.markets,
            "model": self.model,
            "signals": self.signals,
            "execution": self.execution,
            "risk": self.risk,
        }
        if self.plugin is not None:
            out["plugin"] = self.plugin
        return out

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "StrategySpec":
        return StrategySpec(
            schema_version=str(raw.get("schema_version", "strategy/v1")),
            strategy_id=str(raw.get("strategy_id", "unnamed-strategy")),
            created_at_utc=str(raw.get("created_at_utc", utc_now_iso())),
            strategy_type=str(raw.get("strategy_type", "rnn_edge_v1")),
            markets=dict(raw.get("markets") or {}),
            model=dict(raw.get("model") or {}),
            signals=dict(raw.get("signals") or {}),
            execution=dict(raw.get("execution") or {}),
            risk=dict(raw.get("risk") or {}),
            plugin=dict(raw.get("plugin") or {}) if raw.get("plugin") is not None else None,
        )

