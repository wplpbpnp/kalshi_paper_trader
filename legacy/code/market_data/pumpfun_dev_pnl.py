import argparse
import base64
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests


PUMPFUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
CREATE_DISCRIMINATORS = {
    bytes.fromhex("181ec828051c0777"),  # older create
    bytes.fromhex("e76f2bb75b8568b1"),  # CreateV2
    bytes.fromhex("eb17665cbbe362dc"),  # CreateV2 variant
}


def _utcnow_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _cache_path(cache_dir: str, prefix: str, key: str) -> str:
    safe = key.replace("/", "_").replace(":", "_")
    return os.path.join(cache_dir, f"{prefix}_{safe}.json")


def _load_cache(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


class Budget:
    def __init__(self, total: int, cost_sig: int, cost_tx: int):
        self.total = total
        self.cost_sig = cost_sig
        self.cost_tx = cost_tx
        self.spent = 0

    def can_spend(self, cost: int) -> bool:
        return (self.spent + cost) <= self.total

    def spend(self, cost: int) -> bool:
        if not self.can_spend(cost):
            return False
        self.spent += cost
        return True

    def remaining(self) -> int:
        return max(0, self.total - self.spent)


def _rpc(
    url: str,
    method: str,
    params: List[Any],
    *,
    budget: Budget,
    cost: int,
    cache_path: Optional[str] = None,
    timeout_s: float = 30.0,
) -> Any:
    if cache_path:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    if not budget.spend(cost):
        raise RuntimeError("Credit budget exceeded.")

    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    backoff = 0.5
    for _ in range(6):
        resp = requests.post(url, json=payload, timeout=timeout_s)
        if resp.status_code in (429, 500, 502, 503):
            time.sleep(backoff)
            backoff = min(5.0, backoff * 2.0)
            continue
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        if cache_path:
            _save_cache(cache_path, data["result"])
        return data["result"]
    raise RuntimeError("RPC failed after retries.")


def _get_signatures_for_address(
    url: str,
    address: str,
    *,
    before: Optional[str],
    limit: int,
    budget: Budget,
    cache_dir: str,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": limit}
    if before:
        params["before"] = before
    cache_key = f"{address}_{before or 'none'}_{limit}"
    cache_path = _cache_path(cache_dir, "sigs", cache_key)
    return _rpc(
        url,
        "getSignaturesForAddress",
        [address, params],
        budget=budget,
        cost=budget.cost_sig,
        cache_path=cache_path,
    )


def _get_transaction(
    url: str,
    signature: str,
    *,
    budget: Budget,
    cache_dir: str,
) -> Optional[Dict[str, Any]]:
    cache_path = _cache_path(cache_dir, "tx", signature)
    return _rpc(
        url,
        "getTransaction",
        [
            signature,
            {
                "encoding": "json",
                "maxSupportedTransactionVersion": 0,
            },
        ],
        budget=budget,
        cost=budget.cost_tx,
        cache_path=cache_path,
    )


def _account_keys(tx: Dict[str, Any]) -> List[str]:
    msg = tx.get("transaction", {}).get("message", {})
    keys = msg.get("accountKeys", []) or []
    if keys and isinstance(keys[0], dict):
        keys = [k.get("pubkey") for k in keys]
    meta = tx.get("meta", {}) or {}
    loaded = meta.get("loadedAddresses") or {}
    if loaded:
        keys = keys + loaded.get("writable", []) + loaded.get("readonly", [])
    return keys


def _resolve_account(accounts: List[Any], idx: int, keys: List[str]) -> Optional[str]:
    if idx >= len(accounts):
        return None
    acc = accounts[idx]
    if isinstance(acc, int):
        if 0 <= acc < len(keys):
            return keys[acc]
        return None
    if isinstance(acc, str):
        return acc
    return None


def _instruction_program_id(instr: Dict[str, Any], keys: List[str]) -> Optional[str]:
    if "programId" in instr:
        return instr.get("programId")
    if "programIdIndex" in instr:
        idx = instr.get("programIdIndex")
        if isinstance(idx, int) and 0 <= idx < len(keys):
            return keys[idx]
    return None


def _instruction_data(instr: Dict[str, Any]) -> Optional[bytes]:
    data = instr.get("data")
    if not data:
        return None
    try:
        return base64.b64decode(data)
    except Exception:
        return None


def _primary_signer(tx: Dict[str, Any]) -> Optional[str]:
    msg = tx.get("transaction", {}).get("message", {})
    header = msg.get("header") or {}
    num_signers = int(header.get("numRequiredSignatures") or 0)
    keys = msg.get("accountKeys", []) or []
    if keys and isinstance(keys[0], dict):
        keys = [k.get("pubkey") for k in keys]
    if not keys or num_signers <= 0:
        return None
    return keys[0]


def _extract_create(tx: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    keys = _account_keys(tx)
    msg = tx.get("transaction", {}).get("message", {})
    instrs = msg.get("instructions", []) or []
    header = msg.get("header") or {}
    num_signers = int(header.get("numRequiredSignatures") or 0)
    signers = keys[:num_signers] if num_signers and keys else []
    logs = tx.get("meta", {}).get("logMessages", []) or []
    has_create_log = any("Instruction: Create" in line for line in logs)
    for instr in instrs:
        program_id = _instruction_program_id(instr, keys)
        if program_id != PUMPFUN_PROGRAM_ID:
            continue
        raw = _instruction_data(instr)
        is_create = False
        if raw and len(raw) >= 8 and raw[:8] in CREATE_DISCRIMINATORS:
            is_create = True
        elif has_create_log:
            is_create = True
        if not is_create:
            continue
        accounts = instr.get("accounts") or []
        creator = signers[0] if signers else None
        mint = _resolve_account(accounts, 0, keys)
        if mint is None and len(signers) >= 2:
            mint = signers[1]
        if creator and mint:
            return creator, mint
    return None


def _tx_has_pumpfun(tx: Dict[str, Any]) -> bool:
    keys = _account_keys(tx)
    msg = tx.get("transaction", {}).get("message", {})
    instrs = msg.get("instructions", []) or []
    for instr in instrs:
        program_id = _instruction_program_id(instr, keys)
        if program_id == PUMPFUN_PROGRAM_ID:
            return True
    return False


def _tx_mints(tx: Dict[str, Any]) -> List[str]:
    meta = tx.get("meta", {}) or {}
    mints = set()
    for key in ("preTokenBalances", "postTokenBalances"):
        for bal in meta.get(key, []) or []:
            mint = bal.get("mint")
            if mint:
                mints.add(mint)
    return list(mints)


def _token_deltas_for_owner(tx: Dict[str, Any], owner: str) -> Dict[str, float]:
    meta = tx.get("meta", {}) or {}
    pre = meta.get("preTokenBalances", []) or []
    post = meta.get("postTokenBalances", []) or []

    def to_map(bals: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
        out: Dict[str, Tuple[int, int]] = {}
        for bal in bals:
            if bal.get("owner") != owner:
                continue
            mint = bal.get("mint")
            ui = bal.get("uiTokenAmount") or {}
            amt_str = ui.get("amount")
            dec = ui.get("decimals", 0)
            if not mint or amt_str is None:
                continue
            try:
                amt = int(amt_str)
            except Exception:
                continue
            out[mint] = (amt, int(dec))
        return out

    pre_map = to_map(pre)
    post_map = to_map(post)
    mints = set(pre_map.keys()) | set(post_map.keys())
    deltas: Dict[str, float] = {}
    for m in mints:
        pre_amt, pre_dec = pre_map.get(m, (0, None))
        post_amt, post_dec = post_map.get(m, (0, None))
        dec = post_dec if post_dec is not None else (pre_dec if pre_dec is not None else 0)
        if dec is None:
            dec = 0
        delta = (post_amt - pre_amt) / (10 ** int(dec))
        if delta != 0:
            deltas[m] = float(delta)
    return deltas


def _lamports_delta(tx: Dict[str, Any], pubkey: str) -> Optional[int]:
    keys = _account_keys(tx)
    if pubkey not in keys:
        return None
    idx = keys.index(pubkey)
    meta = tx.get("meta", {}) or {}
    pre = meta.get("preBalances", [])
    post = meta.get("postBalances", [])
    if idx >= len(pre) or idx >= len(post):
        return None
    return int(post[idx]) - int(pre[idx])


def _collect_create_sigs(
    url: str,
    cutoff_ts: int,
    *,
    max_sigs: int,
    budget: Budget,
    cache_dir: str,
) -> List[str]:
    sigs: List[str] = []
    before = None
    while len(sigs) < max_sigs and budget.remaining() >= budget.cost_sig:
        batch = _get_signatures_for_address(
            url, PUMPFUN_PROGRAM_ID, before=before, limit=1000, budget=budget, cache_dir=cache_dir
        )
        if not batch:
            break
        stop = False
        for rec in batch:
            bt = rec.get("blockTime")
            sig = rec.get("signature")
            if not sig or bt is None:
                continue
            if int(bt) < cutoff_ts:
                stop = True
                break
            sigs.append(sig)
            if len(sigs) >= max_sigs:
                break
        if stop or len(sigs) >= max_sigs:
            break
        before = batch[-1].get("signature")
        if not before:
            break
    return sigs


def _summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {}
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick-and-dirty Pump.fun dev PnL sampler (24h window).")
    ap.add_argument("--api-key", default=os.environ.get("HELIUS_API_KEY", ""), help="HELIUS_API_KEY")
    ap.add_argument("--rpc-url", default="", help="Override RPC URL")
    ap.add_argument("--window-hours", type=float, default=24.0)
    ap.add_argument("--target-creates", type=int, default=300)
    ap.add_argument("--max-sigs", type=int, default=20000)
    ap.add_argument("--max-create-txs", type=int, default=12000)
    ap.add_argument("--max-devs", type=int, default=400)
    ap.add_argument("--max-dev-sigs", type=int, default=80)
    ap.add_argument("--credit-budget", type=int, default=200_000)
    ap.add_argument("--cost-sig", type=int, default=1)
    ap.add_argument("--cost-tx", type=int, default=10)
    ap.add_argument("--cache-dir", default=os.path.join("kalshi_paper_trader", "market_data", "pumpfun_cache"))
    ap.add_argument("--out-csv", default=os.path.join("kalshi_paper_trader", "market_data", "pumpfun_dev_pnl.csv"))
    ap.add_argument(
        "--mint-csv",
        default=os.path.join("kalshi_paper_trader", "market_data", "pumpfun_mint_pnl.csv"),
        help="Optional path to write per-mint PnL",
    )
    ap.add_argument("--calc-outside-buys", action="store_true", help="Estimate outside buy volume per mint")
    ap.add_argument("--max-mints-outside", type=int, default=100)
    ap.add_argument("--max-mint-sigs", type=int, default=80)
    ap.add_argument("--max-mint-txs", type=int, default=4000)
    ap.add_argument("--sell-buckets", default="5,15,30,60,120")
    ap.add_argument("--sol-usd", type=float, default=0.0, help="Optional SOL/USD for $10k threshold")
    args = ap.parse_args()

    if not args.api_key and not args.rpc_url:
        raise SystemExit("Provide --api-key or --rpc-url.")

    rpc_url = args.rpc_url or f"https://mainnet.helius-rpc.com/?api-key={args.api_key}"
    os.makedirs(args.cache_dir, exist_ok=True)

    cutoff_ts = _utcnow_ts() - int(args.window_hours * 3600)
    budget = Budget(args.credit_budget, args.cost_sig, args.cost_tx)

    print(f"Cutoff: {datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat()}")
    print(f"Credit budget: {budget.total} (sig={budget.cost_sig}, tx={budget.cost_tx})")

    dev_mints: Dict[str, List[str]] = {}
    mint_create_ts: Dict[str, int] = {}
    scanned_sigs = 0
    fetched_txs = 0
    creates_found = 0
    before = None
    while (
        scanned_sigs < args.max_sigs
        and fetched_txs < args.max_create_txs
        and budget.remaining() >= budget.cost_sig
    ):
        batch = _get_signatures_for_address(
            rpc_url, PUMPFUN_PROGRAM_ID, before=before, limit=1000, budget=budget, cache_dir=args.cache_dir
        )
        if not batch:
            break
        stop = False
        for rec in batch:
            bt = rec.get("blockTime")
            sig = rec.get("signature")
            if not sig or bt is None:
                continue
            if int(bt) < cutoff_ts:
                stop = True
                break
            scanned_sigs += 1
            if fetched_txs >= args.max_create_txs or budget.remaining() < budget.cost_tx:
                stop = True
                break
            tx = _get_transaction(rpc_url, sig, budget=budget, cache_dir=args.cache_dir)
            fetched_txs += 1
            if not tx:
                continue
            res = _extract_create(tx)
            if not res:
                continue
            creator, mint = res
            creates_found += 1
            dev_mints.setdefault(creator, []).append(mint)
            bt = tx.get("blockTime")
            if bt is not None and mint not in mint_create_ts:
                mint_create_ts[mint] = int(bt)
            if len(dev_mints) >= args.max_devs or creates_found >= args.target_creates:
                stop = True
                break
            if creates_found % 25 == 0:
                print(
                    f"Creates={creates_found} scanned_sigs={scanned_sigs} fetched_txs={fetched_txs} "
                    f"devs={len(dev_mints)} budget_left={budget.remaining()}"
                )
        if stop:
            break
        before = batch[-1].get("signature")
        if not before:
            break

    print(f"Creates found (<=24h): {creates_found} scanned_sigs={scanned_sigs} fetched_txs={fetched_txs}")

    devs = list(dev_mints.keys())
    print(f"Creators sampled: {len(devs)}")

    rows: List[Dict[str, Any]] = []
    mint_pnl: Dict[str, float] = {}
    mint_token_pos: Dict[str, float] = {}
    mint_last_price: Dict[str, float] = {}
    mint_has_sell: Dict[str, bool] = {}
    mint_first_sell_ts: Dict[str, int] = {}
    mint_creator: Dict[str, str] = {}
    for creator, mints in dev_mints.items():
        for m in mints:
            mint_creator[m] = creator
    for idx, dev in enumerate(devs, start=1):
        if budget.remaining() < budget.cost_sig:
            break
        before = None
        dev_sigs: List[Dict[str, Any]] = []
        while len(dev_sigs) < args.max_dev_sigs and budget.remaining() >= budget.cost_sig:
            batch = _get_signatures_for_address(
                rpc_url, dev, before=before, limit=1000, budget=budget, cache_dir=args.cache_dir
            )
            if not batch:
                break
            stop = False
            for rec in batch:
                bt = rec.get("blockTime")
                if bt is None:
                    continue
                if int(bt) < cutoff_ts:
                    stop = True
                    break
                dev_sigs.append(rec)
                if len(dev_sigs) >= args.max_dev_sigs:
                    break
            if stop or len(dev_sigs) >= args.max_dev_sigs:
                break
            before = batch[-1].get("signature")
            if not before:
                break

        pnl_lamports = 0
        pump_txs = 0
        total_txs = 0
        own_pnl_lamports = 0
        own_pump_txs = 0
        mint_set = set(dev_mints.get(dev, []))
        for rec in dev_sigs:
            sig = rec.get("signature")
            if not sig:
                continue
            if budget.remaining() < budget.cost_tx:
                break
            tx = _get_transaction(rpc_url, sig, budget=budget, cache_dir=args.cache_dir)
            if not tx or tx.get("meta", {}).get("err") is not None:
                continue
            total_txs += 1
            if not _tx_has_pumpfun(tx):
                continue
            pump_txs += 1
            delta = _lamports_delta(tx, dev)
            if delta is not None:
                pnl_lamports += delta

            if mint_set:
                token_deltas = _token_deltas_for_owner(tx, dev)
                tx_mints = set(token_deltas.keys())
                own_mints = mint_set.intersection(tx_mints)
                if own_mints:
                    bt = tx.get("blockTime")
                    own_ok = True
                    if bt is not None:
                        own_ok = False
                        for m in own_mints:
                            ts = mint_create_ts.get(m)
                            if ts is None or int(bt) >= int(ts):
                                own_ok = True
                                break
                    if own_ok:
                        own_pump_txs += 1
                        if delta is not None:
                            own_pnl_lamports += delta
                            abs_weights = {m: abs(token_deltas.get(m, 0.0)) for m in own_mints}
                            total_w = sum(abs_weights.values())
                            if total_w > 0:
                                for m, w in abs_weights.items():
                                    mint_pnl[m] = mint_pnl.get(m, 0.0) + (delta / 1e9) * (w / total_w)
                            else:
                                for m in own_mints:
                                    mint_pnl[m] = mint_pnl.get(m, 0.0) + (delta / 1e9) / max(1, len(own_mints))

                        # Track inventory + last price per mint
                        nonzero_mints = [m for m in own_mints if token_deltas.get(m)]
                        for m in nonzero_mints:
                            d = token_deltas.get(m, 0.0)
                            mint_token_pos[m] = mint_token_pos.get(m, 0.0) + d
                            if d < 0:
                                mint_has_sell[m] = True
                                bt = tx.get("blockTime")
                                if bt is not None:
                                    prev = mint_first_sell_ts.get(m)
                                    if prev is None or int(bt) < int(prev):
                                        mint_first_sell_ts[m] = int(bt)
                        if delta is not None and len(nonzero_mints) == 1:
                            m = nonzero_mints[0]
                            d = token_deltas.get(m, 0.0)
                            if d != 0:
                                price = abs((delta / 1e9) / d)
                                if price > 0:
                                    mint_last_price[m] = price

        rows.append(
            {
                "creator": dev,
                "mints_created": len(dev_mints.get(dev, [])),
                "pumpfun_txs": pump_txs,
                "sampled_txs": total_txs,
                "pnl_sol": pnl_lamports / 1e9,
                "own_pumpfun_txs": own_pump_txs,
                "own_pnl_sol": own_pnl_lamports / 1e9,
            }
        )

        if idx % 50 == 0:
            print(f"Dev {idx}/{len(devs)} budget_left={budget.remaining()}")

    if not rows:
        print("No dev rows produced.")
        return 1

    pnl_values = [r["pnl_sol"] for r in rows]
    own_pnl_values = [r["own_pnl_sol"] for r in rows]
    stats = _summarize(pnl_values)
    own_stats = _summarize(own_pnl_values)

    pos_rate = float(np.mean(np.array(pnl_values) > 0.0))
    own_pos_rate = float(np.mean(np.array(own_pnl_values) > 0.0))
    print("\n=== DEV PNL SUMMARY (ALL PUMPFUN TXS, REALIZED SOL) ===")
    print(f"devs_analyzed={len(rows)} pos_rate={pos_rate}")
    for k in ("mean", "median", "p90", "p95", "p99", "min", "max"):
        if k in stats:
            print(f"{k}={stats[k]}")

    print("\n=== DEV PNL SUMMARY (OWN MINTS ONLY, REALIZED SOL) ===")
    print(f"devs_analyzed={len(rows)} pos_rate={own_pos_rate}")
    for k in ("mean", "median", "p90", "p95", "p99", "min", "max"):
        if k in own_stats:
            print(f"{k}={own_stats[k]}")

    outside_vol: Dict[str, float] = {}
    outside_count: Dict[str, int] = {}

    if mint_pnl:
        mint_values = list(mint_pnl.values())
        mint_stats = _summarize(mint_values)
        mint_pos_rate = float(np.mean(np.array(mint_values) > 0.0))
        print("\n=== MINT PNL SUMMARY (OWN MINTS ONLY, REALIZED SOL) ===")
        print(f"mints_analyzed={len(mint_values)} pos_rate={mint_pos_rate}")
        for k in ("mean", "median", "p90", "p95", "p99", "min", "max"):
            if k in mint_stats:
                print(f"{k}={mint_stats[k]}")

        mint_total = {}
        for m, pnl in mint_pnl.items():
            inv = mint_token_pos.get(m, 0.0)
            mark = mint_last_price.get(m, 0.0)
            mint_total[m] = pnl + inv * mark
        total_values = list(mint_total.values())
        total_stats = _summarize(total_values)
        total_pos_rate = float(np.mean(np.array(total_values) > 0.0))
        print("\n=== MINT PNL SUMMARY (REALIZED + MARKED INVENTORY) ===")
        print(f"mints_analyzed={len(total_values)} pos_rate={total_pos_rate}")
        for k in ("mean", "median", "p90", "p95", "p99", "min", "max"):
            if k in total_stats:
                print(f"{k}={total_stats[k]}")

        sell_mints = [m for m in mint_pnl.keys() if mint_has_sell.get(m)]
        hold_mints = [m for m in mint_pnl.keys() if not mint_has_sell.get(m)]
        if sell_mints:
            vals = [mint_total[m] for m in sell_mints]
            stats = _summarize(vals)
            pos = float(np.mean(np.array(vals) > 0.0))
            print("\n=== MINTS WITH DEV SELL (REALIZED + MARKED) ===")
            print(f"mints_analyzed={len(vals)} pos_rate={pos}")
            for k in ("mean", "median", "p90", "p95", "p99", "min", "max"):
                if k in stats:
                    print(f"{k}={stats[k]}")
        if hold_mints:
            vals = [mint_total[m] for m in hold_mints]
            stats = _summarize(vals)
            pos = float(np.mean(np.array(vals) > 0.0))
            print("\n=== MINTS WITH NO DEV SELL (REALIZED + MARKED) ===")
            print(f"mints_analyzed={len(vals)} pos_rate={pos}")
            for k in ("mean", "median", "p90", "p95", "p99", "min", "max"):
                if k in stats:
                    print(f"{k}={stats[k]}")

        buckets = [int(x) for x in args.sell_buckets.split(",") if x.strip()]
        buckets = sorted({b for b in buckets if b > 0})
        if buckets:
            print("\n=== MINT PNL BY TIME-TO-DEV-SELL (REALIZED + MARKED) ===")
            for b in buckets:
                vals = []
                for m, total in mint_total.items():
                    sell_ts = mint_first_sell_ts.get(m)
                    create_ts = mint_create_ts.get(m)
                    if sell_ts is None or create_ts is None:
                        continue
                    dt_min = (sell_ts - create_ts) / 60.0
                    if dt_min <= b:
                        vals.append(total)
                if vals:
                    stats = _summarize(vals)
                    pos = float(np.mean(np.array(vals) > 0.0))
                    print(f"<= {b} min: n={len(vals)} pos_rate={pos} mean={stats.get('mean')} median={stats.get('median')}")

            no_sell_vals = [mint_total[m] for m in mint_total.keys() if not mint_has_sell.get(m)]
            if no_sell_vals:
                stats = _summarize(no_sell_vals)
                pos = float(np.mean(np.array(no_sell_vals) > 0.0))
                print(f"no sell: n={len(no_sell_vals)} pos_rate={pos} mean={stats.get('mean')} median={stats.get('median')}")

        if args.calc_outside_buys:
            window_end = _utcnow_ts()
            mints = list(mint_pnl.keys())[: args.max_mints_outside]
            scanned = 0
            fetched = 0
            for m in mints:
                if budget.remaining() < budget.cost_sig:
                    break
                creator = mint_creator.get(m)
                if not creator:
                    continue
                create_ts = mint_create_ts.get(m)
                if create_ts is None:
                    continue
                end_ts = mint_first_sell_ts.get(m, window_end)
                before = None
                sigs_scanned = 0
                while sigs_scanned < args.max_mint_sigs and budget.remaining() >= budget.cost_sig:
                    batch = _get_signatures_for_address(
                        rpc_url, m, before=before, limit=1000, budget=budget, cache_dir=args.cache_dir
                    )
                    if not batch:
                        break
                    stop = False
                    for rec in batch:
                        bt = rec.get("blockTime")
                        sig = rec.get("signature")
                        if not sig or bt is None:
                            continue
                        if int(bt) < int(create_ts):
                            stop = True
                            break
                        if int(bt) > int(end_ts):
                            continue
                        sigs_scanned += 1
                        scanned += 1
                        if fetched >= args.max_mint_txs or budget.remaining() < budget.cost_tx:
                            stop = True
                            break
                        tx = _get_transaction(rpc_url, sig, budget=budget, cache_dir=args.cache_dir)
                        fetched += 1
                        if not tx or tx.get("meta", {}).get("err") is not None:
                            continue
                        if not _tx_has_pumpfun(tx):
                            continue
                        signer = _primary_signer(tx)
                        if not signer or signer == creator:
                            continue
                        deltas = _token_deltas_for_owner(tx, signer)
                        d = deltas.get(m)
                        if d and d > 0:
                            outside_vol[m] = outside_vol.get(m, 0.0) + d
                            outside_count[m] = outside_count.get(m, 0) + 1
                    if stop:
                        break
                    before = batch[-1].get("signature")
                    if not before:
                        break

            if outside_vol:
                vols = list(outside_vol.values())
                stats = _summarize(vols)
                print("\n=== OUTSIDE BUY VOLUME BEFORE DEV SELL (TOKENS) ===")
                print(f"mints_analyzed={len(vols)} scanned_sigs={scanned} fetched_txs={fetched}")
                for k in ("mean", "median", "p90", "p95", "p99", "min", "max"):
                    if k in stats:
                        print(f"{k}={stats[k]}")

    if args.sol_usd > 0:
        usd = np.array(pnl_values) * args.sol_usd
        pct_10k = float(np.mean(usd >= 10_000))
        print(f"pct_over_$10k={pct_10k}")

    with open(args.out_csv, "w") as f:
        f.write("creator,mints_created,own_pumpfun_txs,own_pnl_sol\n")
        for r in rows:
            f.write(
                f"{r['creator']},{r['mints_created']},{r['own_pumpfun_txs']},{r['own_pnl_sol']}\n"
            )

    if args.mint_csv:
        with open(args.mint_csv, "w") as f:
            f.write(
                "mint,creator,create_ts,own_pnl_sol,inventory,mark_price,own_pnl_marked,has_sell,first_sell_ts,"
                "outside_buy_volume,outside_buy_count\n"
            )
            for m, pnl in mint_pnl.items():
                inv = mint_token_pos.get(m, 0.0)
                mark = mint_last_price.get(m, 0.0)
                total = pnl + inv * mark
                f.write(
                    f"{m},{mint_creator.get(m,'')},{mint_create_ts.get(m,'')},{pnl},"
                    f"{inv},{mark},{total},{int(bool(mint_has_sell.get(m)))},{mint_first_sell_ts.get(m,'')},"
                    f"{outside_vol.get(m,0.0)},{outside_count.get(m,0)}\n"
                )
        print(f"Wrote {args.mint_csv}")
    print(f"Wrote {args.out_csv}")
    print(f"Credits spent (estimate): {budget.spent} / {budget.total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
