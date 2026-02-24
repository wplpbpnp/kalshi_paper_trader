from __future__ import annotations

"""
This repo already has a workable Kalshi candle schema loader and a fee model in
`kalshi_paper_trader/maker-strat/`. Those scripts are not a Python package, so
we add their directory to sys.path for reuse.
"""

from pathlib import Path
import sys


def add_maker_strat_to_path() -> None:
    root = Path(__file__).resolve().parents[2]
    p = root / "kalshi_paper_trader" / "maker-strat"
    sys.path.insert(0, str(p))

