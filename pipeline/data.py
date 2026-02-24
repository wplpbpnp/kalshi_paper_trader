from __future__ import annotations

import glob
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .schemas import SnapshotRecord


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _open_jsonl(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("r")


def iter_snapshot_files(path_or_glob: str) -> Iterable[Path]:
    p = Path(path_or_glob)
    if p.exists() and p.is_file():
        return [p]
    if p.exists() and p.is_dir():
        files = sorted(p.glob("*.snap_*ms.jsonl*"))
        return files
    files = sorted(Path(x) for x in glob.glob(path_or_glob))
    return files


def read_snapshot_file(path: Path) -> List[SnapshotRecord]:
    out: List[SnapshotRecord] = []
    try:
        with _open_jsonl(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except Exception:
                    continue
                ts_ms = _to_int(raw.get("ts_ms"))
                ticker = raw.get("market_ticker")
                if ts_ms is None or not isinstance(ticker, str) or not ticker:
                    continue
                yes_bid = _to_int(raw.get("yes_bid"))
                yes_ask = _to_int(raw.get("yes_ask"))
                no_bid = _to_int(raw.get("no_bid"))
                no_ask = _to_int(raw.get("no_ask"))
                yes_mid = _to_float(raw.get("yes_mid"))
                spread = _to_int(raw.get("spread"))
                if spread is None and yes_bid is not None and yes_ask is not None:
                    spread = yes_ask - yes_bid
                out.append(
                    SnapshotRecord(
                        ts_ms=ts_ms,
                        market_ticker=ticker,
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        no_bid=no_bid,
                        no_ask=no_ask,
                        yes_mid=yes_mid,
                        spread=spread,
                    )
                )
    except (EOFError, OSError):
        # Corrupted/truncated snapshot files can happen during interrupted captures.
        # Keep any successfully parsed rows and let callers decide whether to skip.
        pass
    out.sort(key=lambda r: r.ts_ms)
    return out


def load_label_map(path: str) -> Dict[str, int]:
    """
    Supported label inputs:
    - {"TICKER":"yes"/"no"/1/0, ...}
    - [{"ticker":"...", "result":"yes"|"no"}, ...]
    """
    raw = json.loads(Path(path).read_text())
    out: Dict[str, int] = {}
    if isinstance(raw, dict):
        items = raw.items()
        for ticker, val in items:
            out[str(ticker)] = _parse_label(val)
        return {k: v for k, v in out.items() if v in (0, 1)}

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker", ""))
            if not ticker:
                continue
            result = item.get("result")
            out[ticker] = _parse_label(result)
        return {k: v for k, v in out.items() if v in (0, 1)}

    raise ValueError(f"Unsupported labels JSON shape in {path}")


def _parse_label(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 1 if float(v) > 0 else 0
    s = str(v).strip().lower()
    if s in ("yes", "y", "1", "true", "win"):
        return 1
    if s in ("no", "n", "0", "false", "lose", "loss"):
        return 0
    return -1


def feature_vector(rec: SnapshotRecord) -> List[float]:
    """
    Canonical feature vector used by the RNN edge model.
    """
    yes_bid = 0.0 if rec.yes_bid is None else rec.yes_bid / 100.0
    yes_ask = 0.0 if rec.yes_ask is None else rec.yes_ask / 100.0
    no_bid = 0.0 if rec.no_bid is None else rec.no_bid / 100.0
    no_ask = 0.0 if rec.no_ask is None else rec.no_ask / 100.0
    yes_mid = rec.yes_mid
    if yes_mid is None and rec.yes_bid is not None and rec.yes_ask is not None:
        yes_mid = 0.5 * (rec.yes_bid + rec.yes_ask)
    mid = 0.0 if yes_mid is None else float(yes_mid) / 100.0
    spread = 0.0 if rec.spread is None else float(rec.spread) / 100.0
    return [yes_bid, yes_ask, no_bid, no_ask, mid, spread]


def build_fixed_sequence(records: List[SnapshotRecord], seq_len: int) -> List[List[float]]:
    """
    Left-pad with zeros and keep the latest `seq_len` samples.
    """
    feats = [feature_vector(r) for r in records[-seq_len:]]
    if not feats:
        feats = []
    feat_dim = 6
    if len(feats) < seq_len:
        pad = [[0.0] * feat_dim for _ in range(seq_len - len(feats))]
        feats = pad + feats
    return feats


def sample_last_record(records: List[SnapshotRecord]) -> Optional[SnapshotRecord]:
    if not records:
        return None
    return records[-1]
