# maker-strat

Goal: scan + backtest **maker-first** strategies on Kalshi BTC 15m markets using the
historical 1-minute candlesticks you already downloaded (order book best bid/ask OHLC).

This is intentionally "from scratch" and self-contained so you can iterate without
touching the older scanner code.

Key idea (why this exists):
- "Prices diverge toward 0/100 near expiry" is just uncertainty collapsing. It does **not**
  imply you can exit profitably unless you have edge *conditional on getting filled*.
- Maker fills change the economics (often $0 fees), but introduce **selection**:
  you get filled when someone chooses to trade with your resting order.

Data source used (local):
- `kalshi_paper_trader/edge_finder/data/KXBTC15M_candles.json`

How we simulate a "resting buy" fill (approximation):
- If you place a bid at price `p`, we treat you as filled when the market's best ask
  trades down to `<= p` within your TTL window (using candle ask LOW).
- Exit similarly: your resting sell at `p_take` fills when best bid reaches `>= p_take`
  (using candle bid HIGH).

Run a basic parameter scan:
```bash
./kalshi_paper_trader/venv/bin/python kalshi_paper_trader/maker-strat/scan.py \
  --data kalshi_paper_trader/edge_finder/data/KXBTC15M_candles.json \
  --side NO \
  --entry-minutes 0-10 \
  --entry-improve 0,1,2 \
  --ttl-minutes 1,2,3 \
  --tp-cents 1,2,3,4,5 \
  --stop-minute 12 \
  --stop-exit-ttl-minutes 1,2 \
  --stop-exit-improve 0,1,2 \
  --max-entry-spread-cents 30 \
  --max-stop-spread-cents 30 \
  --fallback taker_exit \
  --bootstrap 200
```

Outputs:
- `kalshi_paper_trader/maker-strat/results.csv`
