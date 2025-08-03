# Backtesting Engine

## Features

- **Simulates multi-year trading strategies**
- **Adjusts for:**
  - Slippage and commissions
  - Margin usage and capital requirements
  - Volatility skew and IV crush
  - Liquidity/option availability

## How It Works

- Iterates over date ranges and symbols
- For each, opens/closes virtual trades and records P/L
- Uses real historical market and option prices

## Running Backtests

```bash
python3 main.py

## Output

    Console P/L report for each symbol

    Debug logs for troubleshooting

Future Plans

    Parallelized multi-core backtesting

    Store full trade logs in CSV/Parquet

    Visualization dashboards

See Troubleshooting for known issues.


---

### **`docs/troubleshooting.md`**

```markdown
# Troubleshooting

## Common Issues

### No Spot Price / No Data

- Ensure backtest dates are **weekdays/market days** (not weekends/holidays).
- Double-check your API key(s) are valid and not rate-limited.
- Make sure data provider limits are not exceeded.

### Package/Install Errors

- Run `pip install --upgrade pip`
- Check for missing dependencies in `requirements.txt`.

### Git/GitHub Issues

- Ensure `.gitignore` excludes `__pycache__`, `env`, `cache/`, and other non-code files.
- See [GitHub documentation](https://docs.github.com/) for workflow errors.

## Getting Help

- Open an issue on the GitHub repo with error logs and details.

