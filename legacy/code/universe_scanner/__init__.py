"""
Universe scanner wrapper around `kalshi_paper_trader.crypto_policy`.

This layer is intentionally thin: the user normalizes disparate data sources into
the universal OHLCV CSV schema; the scanner runs the same evaluation battery
across a list of assets and writes consolidated results.
"""

