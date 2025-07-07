# Data Pipeline

## Data Sources

- **Market Data:** Polygon.io, Yahoo Finance, yfinance, etc.
- **Options Chains:** Polygon.io, via REST API.
- **Economic Indicators:** (Planned) FRED, CPI, Unemployment, etc.
- **News/NLP Signals:** (Planned) Scrape SEC, Reddit, X/Twitter, Fed transcripts.
- **Weather/Geo Data:** (Planned) NOAA, BOM, satellite feeds.

## Data Flow

1. Fetch historical EOD/Intraday data for symbols.
2. Download options chains for each backtest date.
3. Preprocess and clean raw data.
4. Feed data into strategy models for P/L calculation.

## Configuration

- Data provider APIs set in `bot/env`.
- Custom pipeline logic in `data_pipeline.py`.

See [Backtesting](backtesting.md) for data usage details.

