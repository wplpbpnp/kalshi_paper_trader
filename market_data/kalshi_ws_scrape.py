"""
WebSocket-based real-time Kalshi data logger (supports 100ms snapshots).

Logs:
- Raw WS messages as JSONL (optional)
- Fixed-interval snapshots (default 100ms) derived from order book + last trade

For KXBTC15M, the order book is expressed as bid stacks for YES and NO. Asks
are derived via complement:
  yes_ask = 100 - no_bid
  no_ask  = 100 - yes_bid
"""

import argparse
import asyncio
from collections import deque
import gzip
import json
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, TextIO, Tuple

from .kalshi_http import KalshiHttpClient
from .selection import parse_iso_z, pick_active_open_market
from .kalshi_ws import KalshiWsClient, KalshiWsConfig


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _open_jsonl(path: str, gzip_level: Optional[int]) -> TextIO:
    if gzip_level is None:
        return open(path, "a", buffering=1)
    return gzip.open(path, "at", compresslevel=gzip_level)


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


@dataclass
class OrderBookState:
    yes_bids: Dict[int, int]
    no_bids: Dict[int, int]
    sid: Optional[int] = None
    last_seq: Optional[int] = None
    last_ts: Optional[str] = None

    def reset(self) -> None:
        self.yes_bids.clear()
        self.no_bids.clear()
        self.sid = None
        self.last_seq = None
        self.last_ts = None

    def best_yes_bid(self) -> Optional[int]:
        return max(self.yes_bids) if self.yes_bids else None

    def best_no_bid(self) -> Optional[int]:
        return max(self.no_bids) if self.no_bids else None

    def best_yes_ask(self) -> Optional[int]:
        no_bid = self.best_no_bid()
        return (100 - no_bid) if no_bid is not None else None

    def best_no_ask(self) -> Optional[int]:
        yes_bid = self.best_yes_bid()
        return (100 - yes_bid) if yes_bid is not None else None


@dataclass
class LastTradeState:
    ts: Optional[str] = None
    taker_side: Optional[str] = None
    count: Optional[int] = None
    yes_price: Optional[int] = None
    no_price: Optional[int] = None


@dataclass
class BestQuoteState:
    ts: Optional[str] = None
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    last_price: Optional[int] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None


def _fmt_price(p: Optional[int]) -> str:
    return "—" if p is None else f"{p:02d}"


def _fmt_ms(v: Optional[int]) -> str:
    return "—" if v is None else f"{v}"


def _book_top(levels: Dict[int, int], depth: int) -> list[tuple[int, int]]:
    if not levels or depth <= 0:
        return []
    return sorted(levels.items(), key=lambda kv: kv[0], reverse=True)[:depth]


def _payload(msg: Dict[str, Any]) -> Dict[str, Any]:
    payload = msg.get("msg")
    return payload if isinstance(payload, dict) else {}


async def _snapshot_writer(
    *,
    stop_flag: Dict[str, bool],
    out_fh: TextIO,
    snapshot_ms: int,
    market_ticker: str,
    book: OrderBookState,
    quote: BestQuoteState,
    last_trade: LastTradeState,
    last_msg_recv_ts_ms: Dict[str, Optional[int]],
) -> None:
    interval_s = max(1, snapshot_ms) / 1000.0
    next_t = time.time()

    while not stop_flag["flag"]:
        now = time.time()
        if now < next_t:
            try:
                await asyncio.sleep(next_t - now)
            except asyncio.CancelledError:
                return
            continue
        next_t += interval_s

        yes_bid = book.best_yes_bid()
        yes_ask = book.best_yes_ask()
        no_bid = book.best_no_bid()
        no_ask = book.best_no_ask()

        # Fallback to ticker quotes if we don't have a populated book (or if one side is missing).
        if yes_bid is None:
            yes_bid = quote.yes_bid
        if yes_ask is None:
            yes_ask = quote.yes_ask
        if no_bid is None and yes_ask is not None:
            no_bid = 100 - yes_ask
        if no_ask is None and yes_bid is not None:
            no_ask = 100 - yes_bid

        yes_mid = None
        if yes_bid is not None and yes_ask is not None:
            yes_mid = 0.5 * (yes_bid + yes_ask)

        no_mid = None
        if no_bid is not None and no_ask is not None:
            no_mid = 0.5 * (no_bid + no_ask)

        rec = {
            "ts_ms": int(now * 1000),
            "market_ticker": market_ticker,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "yes_mid": yes_mid,
            "no_mid": no_mid,
            "book_sid": book.sid,
            "book_seq": book.last_seq,
            "book_ts": book.last_ts,
            "ticker_ts": quote.ts,
            "ticker_last_price": quote.last_price,
            "ticker_open_interest": quote.open_interest,
            "ticker_volume": quote.volume,
            "last_trade_ts": last_trade.ts,
            "last_trade_taker_side": last_trade.taker_side,
            "last_trade_count": last_trade.count,
            "last_trade_yes_price": last_trade.yes_price,
            "last_trade_no_price": last_trade.no_price,
            "last_ws_msg_recv_ts_ms": last_msg_recv_ts_ms.get("value"),
            "ws_msg_age_ms": (
                (int(now * 1000) - last_msg_recv_ts_ms["value"])
                if last_msg_recv_ts_ms.get("value") is not None
                else None
            ),
        }
        out_fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        out_fh.flush()


async def _heartbeat_printer(
    *,
    stop_flag: Dict[str, bool],
    heartbeat_ms: int,
    market_ticker: str,
    close_dt: Optional[datetime],
    floor_strike: Optional[float],
    cap_strike: Optional[float],
    book: OrderBookState,
    quote: BestQuoteState,
    last_trade: LastTradeState,
    last_msg_recv_ts_ms: Dict[str, Optional[int]],
    trades_buf: deque,
    book_depth: int,
    trade_lines: int,
) -> None:
    interval_s = max(100, heartbeat_ms) / 1000.0
    while not stop_flag["flag"]:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return

        now = _utcnow()
        now_ms = int(time.time() * 1000)
        last_recv = last_msg_recv_ts_ms.get("value")
        age_ms = (now_ms - last_recv) if last_recv is not None else None

        yes_bid = book.best_yes_bid() or quote.yes_bid
        yes_ask = book.best_yes_ask() or quote.yes_ask
        no_bid = book.best_no_bid()
        if no_bid is None and yes_ask is not None:
            no_bid = 100 - yes_ask
        no_ask = book.best_no_ask()
        if no_ask is None and yes_bid is not None:
            no_ask = 100 - yes_bid

        spread = None
        if yes_bid is not None and yes_ask is not None:
            spread = yes_ask - yes_bid

        ttc = None
        if close_dt is not None:
            ttc = max(0.0, (close_dt - now).total_seconds())

        strike_s = "—"
        if floor_strike is not None:
            try:
                strike_s = f"{float(floor_strike):.2f}"
            except Exception:
                strike_s = str(floor_strike)

        header = (
            f"[{now.isoformat()}] {market_ticker} "
            f"ttc_s={('—' if ttc is None else f'{ttc:0.1f}')}"
            f" ws_age_ms={_fmt_ms(age_ms)}"
            f" strike={strike_s}"
            f" YES {_fmt_price(yes_bid)}/{_fmt_price(yes_ask)}"
            f" NO {_fmt_price(no_bid)}/{_fmt_price(no_ask)}"
            f" spr={('—' if spread is None else spread)}"
            f" vol={quote.volume if quote.volume is not None else '—'}"
            f" oi={quote.open_interest if quote.open_interest is not None else '—'}"
        )
        print(header, flush=True)

        if book_depth > 0:
            ytop = _book_top(book.yes_bids, book_depth)
            ntop = _book_top(book.no_bids, book_depth)
            if ytop:
                print(
                    "  YES bids: "
                    + " ".join([f"{p}:{s}" for p, s in ytop]),
                    flush=True,
                )
            if ntop:
                print(
                    "  NO  bids: "
                    + " ".join([f"{p}:{s}" for p, s in ntop]),
                    flush=True,
                )

        if trade_lines > 0:
            recent = list(trades_buf)[-trade_lines:]
            if recent:
                print("  Trades:", flush=True)
                for t in recent:
                    print(
                        f"    ts={t.get('ts','—')} taker={t.get('taker_side','—')} "
                        f"cnt={t.get('count','—')} yes={t.get('yes_price','—')} no={t.get('no_price','—')}",
                        flush=True,
                    )


async def _run_for_market(
    *,
    http: KalshiHttpClient,
    ws_cfg: KalshiWsConfig,
    market_ticker: str,
    close_time_iso: Optional[str],
    floor_strike: Optional[float],
    cap_strike: Optional[float],
    out_dir: str,
    gzip_level: Optional[int],
    snapshot_ms: int,
    include_raw: bool,
    auth: bool,
    stop_flag: Dict[str, bool],
    heartbeat_ms: int,
    book_depth: int,
    trade_lines: int,
) -> None:
    _ensure_dir(out_dir)

    suffix = "jsonl.gz" if gzip_level is not None else "jsonl"
    raw_path = os.path.join(out_dir, f"{market_ticker}.raw.{suffix}")
    snap_path = os.path.join(out_dir, f"{market_ticker}.snap_{snapshot_ms}ms.{suffix}")
    meta_path = os.path.join(out_dir, f"{market_ticker}.meta.json")

    with open(meta_path, "w") as f:
        json.dump(
            {
                "market_ticker": market_ticker,
                "close_time": close_time_iso,
                "floor_strike": floor_strike,
                "cap_strike": cap_strike,
                "selected_at": _utcnow().isoformat(),
                "ws_url": ws_cfg.ws_url,
                "snapshot_ms": snapshot_ms,
                "raw_enabled": include_raw,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    raw_fh: Optional[TextIO] = _open_jsonl(raw_path, gzip_level) if include_raw else None
    snap_fh: TextIO = _open_jsonl(snap_path, gzip_level)

    book = OrderBookState(yes_bids={}, no_bids={})
    quote = BestQuoteState()
    last_trade = LastTradeState()
    last_msg_recv_ts_ms: Dict[str, Optional[int]] = {"value": None}
    trades_buf: deque = deque(maxlen=500)

    ws_client = KalshiWsClient(http_client=http, cfg=ws_cfg, auth=auth)

    market_stop = {"flag": False}

    close_dt: Optional[datetime] = None
    if close_time_iso:
        try:
            close_dt = parse_iso_z(close_time_iso)
        except Exception:
            close_dt = None

    snapshot_task = asyncio.create_task(
        _snapshot_writer(
            stop_flag=market_stop,
            out_fh=snap_fh,
            snapshot_ms=snapshot_ms,
            market_ticker=market_ticker,
            book=book,
            quote=quote,
            last_trade=last_trade,
            last_msg_recv_ts_ms=last_msg_recv_ts_ms,
        )
    )

    heartbeat_task: Optional[asyncio.Task] = None
    if heartbeat_ms and heartbeat_ms > 0:
        heartbeat_task = asyncio.create_task(
            _heartbeat_printer(
                stop_flag=market_stop,
                heartbeat_ms=heartbeat_ms,
                market_ticker=market_ticker,
                close_dt=close_dt,
                floor_strike=floor_strike,
                cap_strike=cap_strike,
                book=book,
                quote=quote,
                last_trade=last_trade,
                last_msg_recv_ts_ms=last_msg_recv_ts_ms,
                trades_buf=trades_buf,
                book_depth=book_depth,
                trade_lines=trade_lines,
            )
        )

    try:
        backoff_s = 0.25
        while not stop_flag["flag"] and not market_stop["flag"]:
            if close_dt is not None and _utcnow() >= (close_dt + timedelta(seconds=5)):
                market_stop["flag"] = True
                break

            try:
                async with ws_client.connect() as ws:
                    backoff_s = 0.25

                    # Subscribe to book + trades for this market.
                    # orderbook_delta emits an initial orderbook_snapshot and then deltas.
                    channels = ["ticker", "trade"]
                    if auth:
                        channels.insert(1, "orderbook_delta")

                    await ws_client.send_json(
                        ws,
                        {
                            "id": 1,
                            "cmd": "subscribe",
                            "params": {"channels": channels, "market_tickers": [market_ticker]},
                        },
                    )

                    while not stop_flag["flag"] and not market_stop["flag"]:
                        if close_dt is not None and _utcnow() >= (close_dt + timedelta(seconds=5)):
                            market_stop["flag"] = True
                            break

                        try:
                            msg = await ws_client.recv_json(ws, timeout_s=5.0)
                        except asyncio.TimeoutError:
                            # Allow time-based shutdown / rollover even during quiet periods.
                            continue

                        recv_ms = int(time.time() * 1000)
                        last_msg_recv_ts_ms["value"] = recv_ms

                        if raw_fh is not None:
                            raw_fh.write(
                                json.dumps(
                                    {"recv_ts_ms": recv_ms, "market_ticker": market_ticker, "msg": msg},
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            raw_fh.flush()

                        mtype = msg.get("type")
                        body = _payload(msg)

                        if mtype in ("ticker", "ticker_v2"):
                            quote.ts = msg.get("ts") or body.get("ts")
                            quote.yes_bid = _safe_int(body.get("yes_bid"))
                            quote.yes_ask = _safe_int(body.get("yes_ask"))
                            quote.last_price = _safe_int(body.get("last_price"))
                            quote.open_interest = _safe_int(body.get("open_interest"))
                            quote.volume = _safe_int(body.get("volume"))

                        elif mtype == "orderbook_snapshot":
                            yes = body.get("yes") or []
                            no = body.get("no") or []
                            book.reset()
                            book.sid = _safe_int(body.get("sid"))
                            book.last_seq = _safe_int(body.get("seq"))
                            book.last_ts = msg.get("ts") or body.get("ts")
                            for p, s in yes:
                                pi = _safe_int(p)
                                si = _safe_int(s)
                                if pi is None or si is None:
                                    continue
                                if si > 0:
                                    book.yes_bids[pi] = si
                            for p, s in no:
                                pi = _safe_int(p)
                                si = _safe_int(s)
                                if pi is None or si is None:
                                    continue
                                if si > 0:
                                    book.no_bids[pi] = si

                        elif mtype == "orderbook_delta":
                            if not auth:
                                continue

                            sid = _safe_int(body.get("sid"))
                            seq = _safe_int(body.get("seq"))
                            if book.sid is None:
                                book.sid = sid
                            if sid is not None and book.sid is not None and sid != book.sid:
                                # Different subscription; ignore.
                                continue

                            if book.last_seq is not None and seq is not None and seq != book.last_seq + 1:
                                # Sequence gap: force resubscribe by breaking the inner loop.
                                book.reset()
                                break

                            book.last_seq = seq
                            book.last_ts = msg.get("ts") or body.get("ts")

                            price = _safe_int(body.get("price"))
                            delta = _safe_int(body.get("delta"))
                            side = body.get("side")
                            if price is None or delta is None or side not in ("yes", "no"):
                                continue
                            levels = book.yes_bids if side == "yes" else book.no_bids
                            new_sz = levels.get(price, 0) + delta
                            if new_sz <= 0:
                                levels.pop(price, None)
                            else:
                                levels[price] = new_sz

                        elif mtype == "trade":
                            last_trade.ts = msg.get("ts") or body.get("ts")
                            last_trade.taker_side = body.get("taker_side")
                            last_trade.count = _safe_int(body.get("count"))
                            last_trade.yes_price = _safe_int(body.get("yes_price"))
                            last_trade.no_price = _safe_int(body.get("no_price"))
                            if (
                                last_trade.ts is not None
                                or last_trade.count is not None
                                or last_trade.yes_price is not None
                                or last_trade.no_price is not None
                            ):
                                trades_buf.append(
                                    {
                                        "ts": last_trade.ts,
                                        "taker_side": last_trade.taker_side,
                                        "count": last_trade.count,
                                        "yes_price": last_trade.yes_price,
                                        "no_price": last_trade.no_price,
                                    }
                                )

                        # Market ended? Stop condition handled by outer loop via close_time.

            except Exception:
                await asyncio.sleep(min(5.0, backoff_s))
                backoff_s = min(5.0, backoff_s * 2.0)

    finally:
        market_stop["flag"] = True
        snapshot_task.cancel()
        if heartbeat_task is not None:
            heartbeat_task.cancel()
        try:
            await snapshot_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        if heartbeat_task is not None:
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if raw_fh is not None:
            raw_fh.close()
        snap_fh.close()


def _select_market(
    http: KalshiHttpClient, series: str
) -> Tuple[str, Optional[str], Optional[float], Optional[float]]:
    resp = http.get(
        "/trade-api/v2/markets",
        params={"series_ticker": series, "status": "open", "limit": 200},
    )
    markets = resp.get("markets", [])
    sel = pick_active_open_market(markets)
    if sel is None:
        raise RuntimeError(f"No open markets found for series {series}")
    close_time_iso = sel.market.get("close_time")
    floor_strike = sel.market.get("floor_strike")
    cap_strike = sel.market.get("cap_strike")
    return sel.ticker, close_time_iso, floor_strike, cap_strike


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-time Kalshi WS scraper (raw + fixed-interval snapshots).")
    ap.add_argument("--series", default="KXBTC15M", help="Series ticker (default: KXBTC15M)")
    ap.add_argument("--market-ticker", default="", help="Override market ticker (skip auto-selection)")
    ap.add_argument(
        "--base-url",
        default="https://api.elections.kalshi.com",
        help="Kalshi REST base URL (for market selection)",
    )
    ap.add_argument(
        "--ws-url",
        default="wss://api.elections.kalshi.com/trade-api/ws/v2",
        help="Kalshi WS URL",
    )
    ap.add_argument(
        "--env-file",
        default=os.path.join("kalshi_paper_trader", ".env"),
        help="Path to .env containing KALSHI_API_KEY=... (optional if env var set)",
    )
    ap.add_argument(
        "--pem-file",
        default=os.path.join("kalshi_paper_trader", "kalshi_secret.pem"),
        help="Path to kalshi_secret.pem",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join("kalshi_paper_trader", "market_data", "data"),
        help="Output directory for logs",
    )
    ap.add_argument("--snapshot-ms", type=int, default=100, help="Snapshot interval in milliseconds")
    ap.add_argument(
        "--heartbeat-ms",
        type=int,
        default=1000,
        help="Print a live heartbeat every N ms (0 disables). Default 1000.",
    )
    ap.add_argument(
        "--book-depth",
        type=int,
        default=3,
        help="Heartbeat: print top N bid levels for YES/NO books (default 3).",
    )
    ap.add_argument(
        "--trade-lines",
        type=int,
        default=5,
        help="Heartbeat: print last N trades (default 5).",
    )
    ap.add_argument("--no-raw", action="store_true", help="Disable raw WS message logging")
    ap.add_argument("--no-auth", action="store_true", help="Disable WS auth (orderbook_delta will fail)")
    ap.add_argument(
        "--gzip",
        type=int,
        default=6,
        help="Gzip level (1-9). Set 0 to disable gzip.",
    )
    ap.add_argument(
        "--duration-min",
        type=float,
        default=0.0,
        help="Stop after N minutes (0 = run until Ctrl-C).",
    )
    args = ap.parse_args()

    gzip_level: Optional[int] = None if args.gzip == 0 else max(1, min(9, args.gzip))

    http = KalshiHttpClient.from_files(
        base_url=args.base_url,
        env_file=args.env_file if args.env_file and os.path.exists(args.env_file) else None,
        pem_file=args.pem_file,
        timeout_s=10.0,
    )

    ws_cfg = KalshiWsConfig(ws_url=args.ws_url)

    stop = {"flag": False}

    def _on_signal(_sig, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    start_time = time.time()
    duration_s = max(0.0, float(args.duration_min)) * 60.0

    async def _runner():
        no_market_backoff_s = 0.5
        while not stop["flag"]:
            if duration_s and (time.time() - start_time) >= duration_s:
                break

            if args.market_ticker:
                ticker = args.market_ticker
                close_time_iso = None
                floor_strike = None
                cap_strike = None
            else:
                try:
                    ticker, close_time_iso, floor_strike, cap_strike = _select_market(http, args.series)
                    no_market_backoff_s = 0.5
                except RuntimeError as e:
                    # Transiently, the series may have no open market right after a close.
                    # Retry instead of crashing.
                    if "No open markets found for series" in str(e):
                        await asyncio.sleep(min(5.0, no_market_backoff_s))
                        no_market_backoff_s = min(5.0, no_market_backoff_s * 2.0)
                        continue
                    raise

            await _run_for_market(
                http=http,
                ws_cfg=ws_cfg,
                market_ticker=ticker,
                close_time_iso=close_time_iso,
                floor_strike=floor_strike,
                cap_strike=cap_strike,
                out_dir=args.out_dir,
                gzip_level=gzip_level,
                snapshot_ms=max(1, args.snapshot_ms),
                include_raw=not args.no_raw,
                auth=not args.no_auth,
                stop_flag=stop,
                heartbeat_ms=max(0, int(args.heartbeat_ms)),
                book_depth=max(0, int(args.book_depth)),
                trade_lines=max(0, int(args.trade_lines)),
            )

            # If user pinned a market ticker, don't loop/roll.
            if args.market_ticker:
                break

            # If not pinned, sleep briefly then select the next active market.
            await asyncio.sleep(0.5)

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        return 130
    except asyncio.CancelledError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
