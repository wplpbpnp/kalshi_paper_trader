# market_data (minimal)

This folder now contains only the modules used by the active 3-script pipeline:

- `kalshi_http.py`: signed REST client with timeout-enabled `get/post/delete`
- `kalshi_ws.py`: websocket wrapper client
- `selection.py`: active-market selection helpers
- `kalshi_ws_scrape.py`: high-resolution websocket scraper

Archived tools were moved to:

- `legacy/code/market_data/`
- `legacy/artifacts/market_data/`

Primary entrypoint for collection is:

```bash
python scripts/download_highres.py --config config/runtime.json --series KXBTC15M --snapshot-ms 100
```
