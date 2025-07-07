# Quant Options Bot

Modular Python framework for generating, back‑testing and executing option strategies driven by sentiment, volatility and event triggers.

---

## Features

| Layer | Tech | Notes |
| ----- | ---- | ----- |
|       |      |       |

| **Data**        | *yfinance*, FRED, Polygon            | Live OHLCV, macro time‑series, option snapshots                |
| --------------- | ------------------------------------ | -------------------------------------------------------------- |
| **Signals**     | FinBERT, GARCH/EWMA, plugin registry | Sentiment, vol forecast, unusual‑flow, breach and macro events |
| **Back‑tester** | Polygon open‑close API               | 7‑day iron‐condor P/L with chain‑based strikes                 |
| **Execution**   | stub                                 | Ready for IB‑insync / Tradier integration                      |

---

## Quick start

```bash
# clone and enter repo
git clone https://github.com/icysummer42/quant-options-bot.git
cd quant-options-bot

# create env and install deps
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# export keys (free tiers work)
export POLYGON_KEY=YOUR_KEY
export TE_API_KEY=YOUR_KEY   # optional

# run back‑test
python main.py --mode backtest

# one live loop (prints debug only)
python main.py --mode live --once
```

---

## Configuration

All tunables live in ``

```text
symbols_equity        # tickers to scan
unusual_options_min_premium  # sweep threshold
edge_threshold        # live trade trigger
```

API keys are read from environment variables so you can keep credentials out of git.

---

## File map

```
config.py            # parameters & keys
data_pipeline.py     # market + macro fetchers
plugins.py           # event‑trigger plugins
signal_generator.py  # sentiment, vol, score
data_pipeline.py
main.py              # back‑tester + live loop
requirements.txt     # pinned deps
```

---

## Roadmap

- IBKR execution wiring
- Option P/L with greeks, slippage, commissions
- Streamlit risk dashboard
- CI workflow (pytest + Ruff)
- Docker image for scheduled deployment

