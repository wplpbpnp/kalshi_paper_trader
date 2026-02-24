import argparse
import gzip
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _iter_raw(path: str) -> Iterable[Dict[str, Any]]:
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _parse_ts_ms(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _event_ts_ms(rec: Dict[str, Any]) -> int:
    recv_ts = rec.get("recv_ts_ms")
    msg = rec.get("msg") or {}
    body = msg.get("msg") if isinstance(msg, dict) else {}
    ts = None
    if isinstance(msg, dict):
        ts = msg.get("ts")
    if ts is None and isinstance(body, dict):
        ts = body.get("ts")
    parsed = _parse_ts_ms(ts)
    if parsed is not None:
        return parsed
    if recv_ts is None:
        return 0
    return int(recv_ts)


def _top_sum(levels: Dict[int, int], depth: int) -> Optional[int]:
    if depth <= 0 or not levels:
        return None
    items = sorted(levels.items(), key=lambda kv: kv[0], reverse=True)[:depth]
    return int(sum(sz for _, sz in items))


def _best_price(levels: Dict[int, int]) -> Optional[int]:
    return max(levels) if levels else None


@dataclass
class BarState:
    flow_yes: int = 0
    flow_no: int = 0
    trade_count: int = 0
    trade_events: int = 0
    last_trade_yes_price: Optional[int] = None
    last_trade_no_price: Optional[int] = None
    last_trade_ts_ms: Optional[int] = None

    def reset(self) -> None:
        self.flow_yes = 0
        self.flow_no = 0
        self.trade_count = 0
        self.trade_events = 0
        self.last_trade_yes_price = None
        self.last_trade_no_price = None
        self.last_trade_ts_ms = None


def _finalize_bar(
    *,
    bar_start_ms: int,
    market_ticker: str,
    book_yes: Dict[int, int],
    book_no: Dict[int, int],
    depth: int,
    bar: BarState,
) -> Dict[str, Any]:
    yes_bid = _best_price(book_yes)
    no_bid = _best_price(book_no)

    yes_ask = (100 - no_bid) if no_bid is not None else None
    no_ask = (100 - yes_bid) if yes_bid is not None else None

    yes_mid = None
    if yes_bid is not None and yes_ask is not None:
        yes_mid = 0.5 * (yes_bid + yes_ask)

    spread = None
    if yes_bid is not None and yes_ask is not None:
        spread = yes_ask - yes_bid

    sum_yes = _top_sum(book_yes, depth)
    sum_no = _top_sum(book_no, depth)

    imbalance = None
    if sum_yes is not None and sum_no is not None:
        denom = sum_yes + sum_no
        if denom > 0:
            imbalance = (sum_yes - sum_no) / float(denom)

    flow_imbalance = None
    flow_total = bar.flow_yes + bar.flow_no
    if flow_total > 0:
        flow_imbalance = (bar.flow_yes - bar.flow_no) / float(flow_total)

    return {
        "bar_start_ms": bar_start_ms,
        "market_ticker": market_ticker,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "yes_mid": yes_mid,
        "spread": spread,
        "depth_sum_yes": sum_yes,
        "depth_sum_no": sum_no,
        "book_imbalance": imbalance,
        "flow_yes": bar.flow_yes,
        "flow_no": bar.flow_no,
        "flow_imbalance": flow_imbalance,
        "trade_count": bar.trade_count,
        "trade_events": bar.trade_events,
        "last_trade_yes_price": bar.last_trade_yes_price,
        "last_trade_no_price": bar.last_trade_no_price,
        "last_trade_ts_ms": bar.last_trade_ts_ms,
    }


def _analyze_file(path: str, bar_ms: int, depth: int) -> List[Dict[str, Any]]:
    book_yes: Dict[int, int] = {}
    book_no: Dict[int, int] = {}
    bar = BarState()
    bars: List[Dict[str, Any]] = []

    cur_bar_start: Optional[int] = None
    market_ticker = os.path.basename(path).split(".")[0]

    def advance_to(ts_ms: int):
        nonlocal cur_bar_start
        if cur_bar_start is None:
            cur_bar_start = (ts_ms // bar_ms) * bar_ms
            return
        while ts_ms >= cur_bar_start + bar_ms:
            bars.append(
                _finalize_bar(
                    bar_start_ms=cur_bar_start,
                    market_ticker=market_ticker,
                    book_yes=book_yes,
                    book_no=book_no,
                    depth=depth,
                    bar=bar,
                )
            )
            cur_bar_start += bar_ms
            bar.reset()

    for rec in _iter_raw(path):
        msg = rec.get("msg") or {}
        if not isinstance(msg, dict):
            continue
        mtype = msg.get("type")
        body = msg.get("msg") if isinstance(msg.get("msg"), dict) else {}
        ts_ms = _event_ts_ms(rec)
        if ts_ms <= 0:
            continue
        advance_to(ts_ms)

        if mtype == "orderbook_snapshot":
            yes = body.get("yes") or []
            no = body.get("no") or []
            book_yes.clear()
            book_no.clear()
            for p, s in yes:
                pi = _safe_int(p)
                si = _safe_int(s)
                if pi is None or si is None:
                    continue
                if si > 0:
                    book_yes[pi] = si
            for p, s in no:
                pi = _safe_int(p)
                si = _safe_int(s)
                if pi is None or si is None:
                    continue
                if si > 0:
                    book_no[pi] = si

        elif mtype == "orderbook_delta":
            price = _safe_int(body.get("price"))
            delta = _safe_int(body.get("delta"))
            side = body.get("side")
            if price is None or delta is None or side not in ("yes", "no"):
                continue
            levels = book_yes if side == "yes" else book_no
            new_sz = levels.get(price, 0) + delta
            if new_sz <= 0:
                levels.pop(price, None)
            else:
                levels[price] = new_sz

        elif mtype == "trade":
            taker = body.get("taker_side")
            cnt = _safe_int(body.get("count")) or 0
            if taker == "yes":
                bar.flow_yes += cnt
            elif taker == "no":
                bar.flow_no += cnt
            if cnt > 0:
                bar.trade_count += cnt
                bar.trade_events += 1
            bar.last_trade_yes_price = _safe_int(body.get("yes_price"))
            bar.last_trade_no_price = _safe_int(body.get("no_price"))
            bar.last_trade_ts_ms = ts_ms

    if cur_bar_start is not None:
        bars.append(
            _finalize_bar(
                bar_start_ms=cur_bar_start,
                market_ticker=market_ticker,
                book_yes=book_yes,
                book_no=book_no,
                depth=depth,
                bar=bar,
            )
        )

    return bars


def _corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _sign_acc(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    n = int(mask.sum())
    if n < 5:
        return None
    same = np.sign(x[mask]) == np.sign(y[mask])
    return float(np.mean(same))


def _summarize(df: pd.DataFrame, horizons: List[int], label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"bars={len(df)} trades={int(df['trade_events'].sum())} contracts={int(df['trade_count'].sum())}")
    print(f"mid_valid={int(df['yes_mid'].notna().sum())} spread_med={df['spread'].median()}")
    for h in horizons:
        col = f"mid_fwd_{h}"
        corr_imb = _corr(df["book_imbalance"].to_numpy(), df[col].to_numpy())
        corr_flow = _corr(df["flow_imbalance"].to_numpy(), df[col].to_numpy())
        acc_imb = _sign_acc(df["book_imbalance"].to_numpy(), df[col].to_numpy())
        acc_flow = _sign_acc(df["flow_imbalance"].to_numpy(), df[col].to_numpy())
        print(
            f"h={h} bars: corr(imb,Δmid)={corr_imb} corr(flow,Δmid)={corr_flow} "
            f"signacc(imb)={acc_imb} signacc(flow)={acc_flow}"
        )

    for thresh in (0.1, 0.2, 0.3):
        mask_pos = df["book_imbalance"] > thresh
        mask_neg = df["book_imbalance"] < -thresh
        if mask_pos.any() or mask_neg.any():
            mean_pos = df.loc[mask_pos, "mid_fwd_1"].mean()
            mean_neg = df.loc[mask_neg, "mid_fwd_1"].mean()
            print(f"imb>{thresh}: mean Δmid@1={mean_pos} | imb<-{thresh}: mean Δmid@1={mean_neg}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Basic orderflow/book analysis for Kalshi WS raw logs.")
    ap.add_argument(
        "--data-dir",
        default=os.path.join("kalshi_paper_trader", "market_data", "data"),
        help="Directory with *.raw.jsonl(.gz) files",
    )
    ap.add_argument(
        "--raw-glob",
        default="*.raw.jsonl.gz",
        help="Glob for raw files inside data-dir",
    )
    ap.add_argument("--bar-ms", type=int, default=1000, help="Aggregation bar size in ms")
    ap.add_argument("--depth", type=int, default=5, help="Depth levels for book imbalance (top N bids)")
    ap.add_argument(
        "--horizons",
        default="1,5,10",
        help="Comma-separated forward horizons in bars (e.g., 1,5,10)",
    )
    ap.add_argument("--out-csv", default="", help="Optional path to write per-bar features")
    ap.add_argument("--per-market", action="store_true", help="Print per-market summaries")
    args = ap.parse_args()

    if args.bar_ms <= 0:
        raise SystemExit("bar-ms must be > 0")

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if not horizons:
        raise SystemExit("horizons must be non-empty")

    data_dir = args.data_dir
    raw_paths = sorted(
        [os.path.join(data_dir, p) for p in os.listdir(data_dir) if p.endswith(args.raw_glob.split("*")[-1])]
    )
    if args.raw_glob != "*.raw.jsonl.gz":
        import glob

        raw_paths = sorted(glob.glob(os.path.join(data_dir, args.raw_glob)))

    if not raw_paths:
        print("No raw files found.")
        return 1

    all_bars: List[Dict[str, Any]] = []
    for path in raw_paths:
        bars = _analyze_file(path, bar_ms=args.bar_ms, depth=args.depth)
        all_bars.extend(bars)

    df = pd.DataFrame(all_bars)
    if df.empty:
        print("No bars produced.")
        return 1

    df = df.sort_values(["market_ticker", "bar_start_ms"]).reset_index(drop=True)
    df["yes_mid"] = df["yes_mid"].astype(float)

    for h in horizons:
        df[f"mid_fwd_{h}"] = df.groupby("market_ticker")["yes_mid"].shift(-h) - df["yes_mid"]

    if args.per_market:
        for ticker, g in df.groupby("market_ticker"):
            _summarize(g, horizons, label=ticker)

    _summarize(df, horizons, label="ALL")

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
