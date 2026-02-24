# Legacy Archive

This folder contains the pre-refactor code and outputs.

- `legacy/code/`: old strategy research stacks, backtesters, and trading entrypoints
- `legacy/artifacts/`: generated outputs and historical experiment files
- `legacy/code/market_data/`: archived market-data helper scripts no longer used by the active pipeline
- `legacy/artifacts/market_data/`: archived pumpfun caches and legacy market-data artifacts
- `legacy/artifacts/kalshi_backtest_data.json`: archived historical backtest dataset (superseded by `pipeline_data/labels/kalshi_settlements.json` for training labels)

The active implementation lives at repo root in:

- `scripts/`
- `pipeline/`
- `market_data/`
