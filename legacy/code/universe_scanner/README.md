# Universe Scanner (WIP)

Goal: run the same battery of checks/backtests across a small, diverse universe of assets and converge on candidates that remain good under realistic costs and stability checks.

This repo already has a working RL-style walk-forward harness in:
`kalshi_paper_trader/crypto_policy/`

The plan is to feed *any* asset into that harness by normalizing disparate data sources (Binance, broker APIs, etc.) into a single universal OHLCV CSV schema.

## Universal OHLCV CSV schema

Required columns:
- `open_time_ms` (int): bar open timestamp in milliseconds since epoch (UTC)
- `open` (float)
- `high` (float)
- `low` (float)
- `close` (float)
- `volume` (float): base volume if available, else any consistent proxy

Optional columns (safe to omit; default to 0):
- `quote_volume` (float): if missing, loader defaults it to `volume`
- `trade_count` (int)
- `taker_buy_base_vol` (float)
- `taker_buy_quote_vol` (float)

Notes:
- The RL env uses log returns on `close` and treats `volume`/`quote_volume` only as features.
- For venues without trade/flow fields, run with `--features prices` to avoid injecting meaningless constants.

## Starter universe (10 diverse assets)

You can keep this list as a reference when building your own data adapters:
- Equity index (ETF): `SPY` (US equities)
- Bonds (ETF): `TLT` (long Treasuries)
- Gold (ETF): `GLD`
- FX pair: `EURUSD` (or broker symbol equivalent)
- Crypto majors: `BTCUSDT`, `ETHUSDT`
- Crypto microcap: `PENGUUSDT`
- Commodity futures: `CL` (crude oil), `GC` (gold), `ES` (S&P 500)

The important part is not the exact tickers, it's to cover different regimes (risk-on, rates, FX, commodities, high-vol crypto).

