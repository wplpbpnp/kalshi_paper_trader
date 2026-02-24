from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_runtime_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def cfg_get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    return cfg.get(key, default)


def cfg_get_path(cfg: Dict[str, Any], key: str, root: Path, default: Optional[str] = None) -> Optional[str]:
    raw = cfg.get(key, default)
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((root / p).resolve())

