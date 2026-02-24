# Runtime Config

`config/runtime.json` is the single default config for active scripts:

- `scripts/download_highres.py`
- `scripts/train_edge_rnn.py`
- `scripts/run_live.py`

You can override with `--config <path>` and/or individual CLI flags.

Sensitive files are referenced by path here but should live under `secrets/` and remain untracked.

