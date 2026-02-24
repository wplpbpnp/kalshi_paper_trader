# crypto_policy

Utilities for experimenting with longer-horizon trading using exchange OHLCV (e.g., 30m bars),
as a contrast to short-horizon Kalshi candle-only models where microstructure dominates.

## Download data (Binance.US default)

BTCUSDT 30m bars (example: last 60 days):

```sh
./kalshi_paper_trader/venv/bin/python kalshi_paper_trader/crypto_policy/download_binance_klines.py \
  --exchange binanceus \
  --symbol BTCUSDT \
  --interval 30m \
  --start 2025-12-01T00:00:00Z \
  --out kalshi_paper_trader/crypto_policy/data/BTCUSDT_30m.csv
```

PENGUUSDT 30m bars:

```sh
./kalshi_paper_trader/venv/bin/python kalshi_paper_trader/crypto_policy/download_binance_klines.py \
  --exchange binanceus \
  --symbol PENGUUSDT \
  --interval 30m \
  --start 2025-12-01T00:00:00Z \
  --out kalshi_paper_trader/crypto_policy/data/PENGUUSDT_30m.csv
```

Notes:
- Output directory also gets an `assumptions.json` describing the source and parameters.
- If Binance.US doesn't have the symbol, try `--exchange binance` or a different venue.

## Walk-forward backtest (toy RL policy)

BTCUSDT 30m, walk-forward (example):

```sh
./kalshi_paper_trader/venv/bin/python -m kalshi_paper_trader.crypto_policy.walkforward \
  --data kalshi_paper_trader/crypto_policy/data/BTCUSDT_30m.csv \
  --outdir /tmp/crypto_bt_btc \
  --bar-minutes 30 \
  --train-days 21 --test-days 7 --step-days 7 \
  --episode-days 7 --lookback 16 \
  --fee-bps 1 --slippage-bps 1 \
  --epochs 8 --batch-episodes 128 \
  --lr 3e-4 --gamma 1.0 \
  --entropy-coef 0.01 --value-coef 0.5 --grad-clip 1.0 \
  --hidden 64 --hidden 64 \
  --device cpu
```

Outputs:
- `summary.json`: aggregate test PnL stats across windows (bp).
- `windows.csv`: per-window PnL plus flat/long/short baselines.
- `assumptions.json`: run parameters + explicit modeling assumptions.

## Seed sweep

Run multiple seeds and collect alpha vs baselines:

```sh
./kalshi_paper_trader/venv/bin/python -m kalshi_paper_trader.crypto_policy.seed_sweep \
  --data kalshi_paper_trader/crypto_policy/data/BTCUSDT_30m.csv \
  --outdir /tmp/crypto_seed_sweep \
  --seeds 1-20 \
  --bar-minutes 30 \
  --train-days 21 --test-days 7 --step-days 7 \
  --episode-days 7 --lookback 16 \
  --fee-bps 0.5 --slippage-bps 0.5 \
  --one-decision-per-episode --require-position --force-flat-at-end \
  --epochs 30 --batch-episodes 128 \
  --lr 3e-4 --gamma 1.0 --entropy-coef 0.01 \
  --hidden 64 --device cpu
```

Outputs:
- `results.csv`: per-seed totals and alpha vs always-long/always-short baselines.
- `seed_XXXX/`: full walk-forward output for each seed.

`results.csv` also includes risk metrics:
- `*_sharpe`: annualized sharpe, treating each test window as one period (periods/year = 365 / test_days).
- `*_max_drawdown_bp`: max drawdown of cumulative PnL across windows (bp).

Long-only mode:
- Add `--long-only` to disallow shorts (pos in {0,+1}). This is useful when testing "risk reduction" policies against long-only benchmarks.
