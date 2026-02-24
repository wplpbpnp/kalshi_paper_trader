from __future__ import annotations

import importlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import torch

from .data import feature_vector
from .model import load_checkpoint
from .schemas import SnapshotRecord, StrategySpec


@dataclass(frozen=True)
class OrderIntent:
    side: str  # "yes" or "no"
    count: int
    limit_price_cents: int
    reason: str


@dataclass
class RuntimeState:
    market_ticker: str
    position_side: str  # "flat"|"yes"|"no"
    seconds_to_close: Optional[float]
    now_ts: float


class StrategyEngine(Protocol):
    def on_new_market(self, ticker: str) -> None: ...

    def on_snapshot(self, snapshot: SnapshotRecord, state: RuntimeState) -> List[OrderIntent]: ...


class RnnEdgeStrategy:
    """
    Built-in strategy type: `rnn_edge_v1`.

    Emits one entry order when the model probability crosses thresholds.
    """

    def __init__(self, spec: StrategySpec, strategy_path: str):
        model_rel = str(spec.model.get("model_path", "")).strip()
        if not model_rel:
            raise ValueError("Strategy model.model_path is required")
        base = Path(strategy_path).resolve().parent
        model_path = Path(model_rel)
        if not model_path.is_absolute():
            model_path = (base / model_path).resolve()
        self.model, _ = load_checkpoint(str(model_path), device=str(spec.model.get("device", "cpu")))
        self.model_device = str(spec.model.get("device", "cpu"))
        self.seq_len = int(spec.model.get("seq_len", 240))
        self.long_threshold = float(spec.signals.get("long_threshold", 0.58))
        self.short_threshold = float(spec.signals.get("short_threshold", 0.42))
        self.allow_no = bool(spec.execution.get("allow_no", True))
        self.count = int(spec.execution.get("contracts", 1))
        self.price_offset_cents = int(spec.execution.get("price_offset_cents", 0))
        self.min_seconds_between_trades = float(spec.risk.get("min_seconds_between_trades", 30))
        self.max_entries_per_market = int(spec.risk.get("max_entries_per_market", 1))
        self._buf: deque[list[float]] = deque(maxlen=self.seq_len)
        self._last_trade_ts: float = 0.0
        self._entries_this_market: int = 0

    def on_new_market(self, ticker: str) -> None:
        self._buf.clear()
        self._entries_this_market = 0

    def _predict_prob_yes(self) -> float:
        seq = list(self._buf)
        if len(seq) < self.seq_len:
            pad = [[0.0] * len(seq[0] if seq else [0.0] * 6) for _ in range(self.seq_len - len(seq))]
            seq = pad + seq
        x = torch.tensor([seq], dtype=torch.float32, device=self.model_device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).item()
        return float(prob)

    def on_snapshot(self, snapshot: SnapshotRecord, state: RuntimeState) -> List[OrderIntent]:
        self._buf.append(feature_vector(snapshot))
        if len(self._buf) < 16:
            return []
        if self._entries_this_market >= self.max_entries_per_market:
            return []
        if state.position_side != "flat":
            return []
        if (state.now_ts - self._last_trade_ts) < self.min_seconds_between_trades:
            return []

        p_yes = self._predict_prob_yes()
        intents: List[OrderIntent] = []
        if p_yes >= self.long_threshold:
            ask = snapshot.yes_ask if snapshot.yes_ask is not None else 99
            px = max(1, min(99, int(ask) + self.price_offset_cents))
            intents.append(
                OrderIntent(
                    side="yes",
                    count=self.count,
                    limit_price_cents=px,
                    reason=f"rnn_p_yes={p_yes:.4f}>=long_threshold={self.long_threshold:.4f}",
                )
            )
        elif p_yes <= self.short_threshold and self.allow_no:
            ask_no = snapshot.no_ask
            if ask_no is None and snapshot.yes_bid is not None:
                ask_no = 100 - int(snapshot.yes_bid)
            ask_no = 99 if ask_no is None else int(ask_no)
            px = max(1, min(99, ask_no + self.price_offset_cents))
            intents.append(
                OrderIntent(
                    side="no",
                    count=self.count,
                    limit_price_cents=px,
                    reason=f"rnn_p_yes={p_yes:.4f}<=short_threshold={self.short_threshold:.4f}",
                )
            )

        if intents:
            self._entries_this_market += 1
            self._last_trade_ts = float(state.now_ts)
        return intents


def load_strategy_spec(path: str) -> StrategySpec:
    raw = json.loads(Path(path).read_text())
    return StrategySpec.from_dict(raw)


def _load_plugin(module_name: str, class_name: str):
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls


def build_engine(path: str) -> StrategyEngine:
    spec = load_strategy_spec(path)
    stype = spec.strategy_type

    if stype == "rnn_edge_v1":
        return RnnEdgeStrategy(spec, strategy_path=path)

    if stype == "plugin":
        if not spec.plugin:
            raise ValueError("strategy_type=plugin requires plugin.module and plugin.class")
        module_name = str(spec.plugin.get("module", "")).strip()
        class_name = str(spec.plugin.get("class", "")).strip()
        if not module_name or not class_name:
            raise ValueError("plugin.module and plugin.class are required")
        cls = _load_plugin(module_name, class_name)
        return cls(spec=spec, strategy_path=path)

    raise ValueError(f"Unknown strategy_type: {stype}")
