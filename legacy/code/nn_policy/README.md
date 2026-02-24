# nn_policy

"Vibes-based" policy learning for Kalshi 15-minute BTC markets using only candle data.

This is intentionally black-box: the model outputs actions (buy YES / buy NO / close / hold) and position size.

Important limitations:
- Uses 1-minute candle top-of-book (bid/ask OHLC), not full order books; maker/queue effects are not modeled.
- This is offline training on historical episodes; it is very easy to overfit and produce fake edge.

## Quickstart

Train on one contiguous time slice and evaluate on a later slice:

```sh
./kalshi_paper_trader/venv/bin/python -m kalshi_paper_trader.nn_policy.walkforward \
  --data kalshi_paper_trader/edge_finder/data/KXBTC15M_candles.json \
  --outdir /tmp/nn_policy_bt \
  --train-days 14 --test-days 3 --step-days 3 \
  --epochs 8 --batch-episodes 128 \
  --lr 3e-4 --hidden 128 --hidden 128 \
  --lookback 5 \
  --max-contracts 5
```

Outputs:
- `windows.csv` per walk-forward window (train/test dates + test PnL)
- `trades.csv` per episode (final realized PnL, actions taken)
- `summary.json` overall
