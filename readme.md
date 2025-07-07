# Quant Options Trading Bot

A modular, AI-driven options trading bot for automated strategies on equities and ETFs.  
Supports iron condors, volatility plays, directional signals, and more.  
Focus: S&P 500, ASX 200, high-volume tech stocks, sector ETFs.

## Features

- End-to-end data pipeline: real-time & historical options/stock data, macro indicators
- AI modules for NLP news analysis & time-series forecasting
- Robust backtesting with slippage, margin, and volatility modelling
- Modular quant strategies: iron condor, straddle, directional, etc.
- Interactive Brokers API integration (planned)
- Risk management: drawdown limits, Kelly sizing, VaR, correlation checks

## Installation

1. Clone the repo:
    ```sh
    git clone git@github.com:YOURUSERNAME/trading-bot.git
    cd trading-bot
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Add your API keys:
    - Copy `.env.example` to `.env` and fill out values.

## Quick Start

```sh
python main.py

Project Structure

.
├── bot/              # Core logic, strategies, and data pipeline
├── main.py           # Entrypoint
├── requirements.txt  # Python dependencies
├── config.py         # Bot configuration
├── plugins/          # Optional extra modules
├── tests/            # Unit tests
└── readme.md         # Project documentation

Development

    Code style: Black

    Lint: flake8

    Testing: pytest

TODO

Expand AI NLP pipeline

Broker execution API

Live trading support
