# Kalshi Minimal Pipeline

This repo now has a single primary workflow with three scripts:

1. `scripts/download_highres.py`
2. `scripts/train_edge_rnn.py`
3. `scripts/run_live.py`

The goal is to keep one canonical market-data format and one canonical strategy format.

Helper script:

- `scripts/build_labels.py` (build/update ticker settlement labels from archived backtest JSON)

## Repository Layout (Phase 4)

Active paths:

- `scripts/`
- `pipeline/`
- `market_data/` (shared Kalshi HTTP/WS clients + low-level collectors)
- `config/` (centralized runtime defaults)
- `secrets/` (local secret files, not committed)
- `pipeline_data/` (runtime data; high-res snapshots, labels, generated strategies)

Legacy paths:

- `legacy/code/`
- `legacy/artifacts/`

Legacy content was moved out of the root so only the minimal pipeline is prominent.

## Canonical Data Format

The downloader writes snapshot JSONL records (optionally gzip) with fields like:

- `ts_ms`
- `market_ticker`
- `yes_bid`, `yes_ask`
- `no_bid`, `no_ask`
- `yes_mid`
- `spread`

Use the downloader output as the single training/live input format.

## Canonical Strategy Format

Strategies are JSON files (`strategy/v1`) with:

- `strategy_type` (built-in or plugin)
- `markets` (for example `series_ticker`)
- `model` (paths and model settings)
- `signals` (thresholds)
- `execution` (order behavior)
- `risk` (trade throttles/safety)

Built-in strategy type in this refactor:

- `rnn_edge_v1` (LSTM classifier that emits entry intents)

You can also load arbitrary strategy code using `strategy_type=plugin` with:

- `plugin.module`
- `plugin.class`

## End-to-End Usage

### 0) Configure runtime defaults

Edit `config/runtime.json` once:

- `env_file`
- `pem_file`
- `series`
- paths for data/labels/strategy outputs

Default secret locations are:

- `secrets/.env`
- `secrets/kalshi_secret.pem`

### Quick Commands

```bash
make help
make download
make labels
make sync-labels
make train
make smoke-train
make evaluate
make live-dry
```

You can override defaults:

```bash
make train CONFIG=config/runtime.json PYTHON=./venv/bin/python
make live-dry STRATEGY=pipeline_data/strategies/rnn_edge_latest/strategy.json SERIES=KXBTC15M
make evaluate STRATEGY=pipeline_data/strategies/rnn_edge_latest/strategy.json EVAL_ARGS="--mode walkforward --train-markets 200 --test-markets 50 --step-markets 50 --epochs 3"
make sync-labels SYNC_ARGS="--output pipeline_data/labels/aligned.json --unresolved-out pipeline_data/labels/unresolved.json"
```

`make train` now automatically runs a post-train evaluation pass (fixed replay mode by default).

```bash
# Disable auto-eval if needed
make train EVAL_AFTER_TRAIN=0

# Customize post-train evaluation settings
make train POST_TRAIN_EVAL_ARGS="--mode fixed --test-frac 0.2 --out pipeline_data/strategies/rnn_edge_latest/eval_after_train.json"
```

### 1) Download high-resolution data

```bash
python scripts/download_highres.py \
  --config config/runtime.json \
  --series KXBTC15M \
  --snapshot-ms 100 \
  --out-dir pipeline_data/highres
```

### 2) Train RNN edge model and emit strategy file

If you need to refresh labels from archived historical data:

```bash
python scripts/build_labels.py --config config/runtime.json
```

To align labels with your current snapshots (and pull settled outcomes for missing tickers from Kalshi API):

```bash
python scripts/sync_labels_from_api.py \
  --config config/runtime.json \
  --output pipeline_data/labels/aligned.json \
  --unresolved-out pipeline_data/labels/unresolved.json
```

Then train against aligned labels:

```bash
python scripts/train_edge_rnn.py \
  --config config/runtime.json \
  --labels pipeline_data/labels/aligned.json
```

```bash
python scripts/train_edge_rnn.py \
  --config config/runtime.json \
  --snapshots "pipeline_data/highres/*.snap_*ms.jsonl.gz" \
  --labels pipeline_data/labels/kalshi_settlements.json \
  --outdir pipeline_data/strategies/rnn_edge_latest \
  --device cpu
```

Outputs:

- `pipeline_data/strategies/rnn_edge_latest/model.pt`
- `pipeline_data/strategies/rnn_edge_latest/strategy.json`
- `pipeline_data/strategies/rnn_edge_latest/metrics.json`

### 2b) Evaluate strategy edge (out-of-sample)

Walk-forward (default):

```bash
python scripts/evaluate_strategy.py \
  --config config/runtime.json \
  --strategy pipeline_data/strategies/rnn_edge_latest/strategy.json \
  --mode walkforward \
  --train-markets 200 \
  --test-markets 50 \
  --step-markets 50 \
  --epochs 3 \
  --out pipeline_data/strategies/rnn_edge_latest/eval_walkforward.json
```

Fixed strategy replay on latest holdout:

```bash
python scripts/evaluate_strategy.py \
  --config config/runtime.json \
  --strategy pipeline_data/strategies/rnn_edge_latest/strategy.json \
  --mode fixed \
  --test-frac 0.2 \
  --out pipeline_data/strategies/rnn_edge_latest/eval_fixed.json
```

### 3) Run the strategy executor

Dry-run (default):

```bash
python scripts/run_live.py \
  --config config/runtime.json \
  --strategy pipeline_data/strategies/rnn_edge_latest/strategy.json \
  --series KXBTC15M
```

Live:

```bash
python scripts/run_live.py \
  --config config/runtime.json \
  --strategy pipeline_data/strategies/rnn_edge_latest/strategy.json \
  --series KXBTC15M \
  --live
```

## New Core Modules

- `pipeline/schemas.py`: unified data/strategy schema objects
- `pipeline/data.py`: snapshot IO + feature extraction
- `pipeline/model.py`: RNN model + checkpoint IO
- `pipeline/train_rnn.py`: train/eval pipeline
- `pipeline/strategy_runtime.py`: strategy loader and execution runtime

## Legacy Code

Older scripts and experiments are archived under `legacy/` for reference. The intended path is the three scripts above.
