# Quickstart

## Prerequisites
- Python 3.11+
- Git
- (Optional) virtualenv

## Setup

```bash
git clone <your repo url>
cd trading-bot
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp bot/env.example bot/env  # Edit bot/env with your API keys

# Run a backtest

python3 main.py --backtest

---

