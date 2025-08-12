# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **quantitative options trading bot** built in Python that generates trading signals by aggregating sentiment analysis, volatility forecasting, and event detection. The system supports both backtesting historical strategies and live trading execution.

**Core Strategy**: Delta-neutral options strategies (iron condors, butterflies) and volatility-based trades (straddles, strangles) using AI-driven sentiment analysis and macro event detection.

## Architecture

The codebase follows a modular pipeline architecture:

```
Config → DataPipeline → SignalGenerator → StrategyEngine → ExecutionEngine
    ↓         ↓             ↓              ↓             ↓
  env vars  market data   ML models    option strats   broker APIs
```

### Key Components

- **`config.py`**: Central configuration hub with API keys loaded from environment variables
- **`data_pipeline.py`**: Market data ingestion from Polygon.io, yfinance, FRED
- **`signal_generator.py`**: Aggregates sentiment (FinBERT/OpenAI), volatility (GARCH/EWMA), and event plugins
- **`bot/`**: Core trading logic including backtesting, execution engine, and options strategies
- **`plugins/`**: Event detection modules (unusual flow, cyber breaches, macro releases)

### Data Sources Integration

The system integrates multiple data feeds:
- **Market Data**: Polygon.io (primary), yfinance (fallback)
- **Sentiment**: Reddit (PRAW), NewsAPI, GNews, Stocktwits, Google Trends
- **Events**: Custom plugins for options flow, security breaches, economic releases
- **ML Models**: FinBERT for sentiment, GARCH/EWMA for volatility forecasting

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your API keys
```

### Required Environment Variables
```bash
# Core market data (required)
export POLYGON_KEY=your_polygon_key

# Optional sentiment sources
export OPENAI_API_KEY=your_openai_key
export REDDIT_CLIENT_ID=your_reddit_id
export REDDIT_CLIENT_SECRET=your_reddit_secret
export NEWSAPI_KEY=your_newsapi_key
export GNEWS_API_KEY=your_gnews_key

# Model configuration
export NLP_MODEL_NAME=ProsusAI/finbert
export VOL_MODEL=garch  # or "ewma"
export EDGE_THRESHOLD=0.8
```

### Running the System

```bash
# Run backtest on historical data
python main.py --mode backtest

# Run live trading loop (single iteration)
python main.py --mode live --once

# Test signal generation for specific symbols
python batch_signal_test.py

# Run individual test scripts
python signal_test.py
python fullsignaltest.py
```

### Code Quality

**Linting**: Use `ruff` for linting (included in requirements.txt)
```bash
ruff check .
```

**Formatting**: Use `black` for code formatting (included in requirements.txt)
```bash
black .
```

**Testing**: Run individual test files manually (no unified test framework configured yet)
```bash
python batch_signal_test.py
python signal_test.py
```

## Key File Structure

```
├── main.py                 # CLI entry point
├── config.py              # Configuration and API keys
├── data_pipeline.py       # Market data fetching
├── signal_generator.py    # ML models and signal aggregation
├── bot/
│   ├── backtest.py        # Options backtesting engine
│   ├── engine.py          # Strategy, execution, and risk management
│   ├── polygon_client.py  # Polygon.io API client
│   └── strategy/
│       └── iron_condor.py # Iron condor options strategy
├── plugins/
│   └── events.py          # Custom event detection plugins
├── cache/                 # Cached market data (auto-generated)
├── docs/                  # Project documentation
└── requirements.txt       # Python dependencies
```

## Development Notes

### Adding New Data Sources
1. Add API key to `config.py` and `env.example`
2. Implement fetcher method in `data_pipeline.py`
3. Integrate in `signal_generator.py`
4. Add optional import with graceful fallback

### Adding New Trading Strategies
1. Create new strategy class in `bot/strategy/`
2. Implement required methods: position sizing, P&L calculation
3. Register in `bot/engine.py` StrategyEngine
4. Add backtest support in `bot/backtest.py`

### Caching
- Market data is automatically cached in `cache/` directory using pickle files
- Hash-based cache keys ensure consistency across API calls
- Cache significantly speeds up backtesting and development

### Security
- All API keys should be stored in environment variables, never committed to git
- Use `.env` file for local development (already in .gitignore)
- The `env.example` file shows required variables without exposing secrets

## Current Limitations

- No unified test framework (tests are individual scripts)
- Execution engine is placeholder (prints trades instead of placing them)
- No CI/CD pipeline configured
- Docker deployment not yet implemented

Refer to `readme.md` for detailed setup instructions and `project_plan.txt` for the strategic overview.