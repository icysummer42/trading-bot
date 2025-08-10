"""Market & macro data ingestion layer."""
from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
import os
import requests

# Optional libs
try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore

class DataPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.polygon_key = getattr(cfg, "polygon_key", None) or os.getenv("POLYGON_API_KEY")

    def get_close_series(self, symbol: str, start=None, end=None):
        """Load historical close prices for a symbol as a pd.Series."""
        print(f"[DEBUG] In pipeline: polygon_key={self.polygon_key!r}")
        assert self.polygon_key, "Polygon key missing inside DataPipeline!"

        # Default: last 1 year
        if not start:
            start = (dt.date.today() - dt.timedelta(days=365)).isoformat()
        if not end:
            end = dt.date.today().isoformat()

        # --- Polygon API ---
        if self.polygon_key:
            print(f"[INFO] Fetching close prices from Polygon for {symbol} ({start} to {end}) ...")
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start}/{end}"
                f"?adjusted=true&sort=asc&apiKey={self.polygon_key}"
            )
            r = requests.get(url, timeout=10)
            print(f"[DEBUG] Polygon status: {r.status_code}")
            if r.status_code == 200:
                data = r.json().get("results", [])
                print(f"[DEBUG] Polygon results length: {len(data)}")
                if data:
                    closes = pd.Series(
                        [row["c"] for row in data],
                        index=pd.to_datetime([dt.datetime.fromtimestamp(row["t"]/1000).date() for row in data])
                    )
                    closes.name = "close"
                    print(f"[DEBUG] Polygon close prices:\n{closes.head()}")
                    return closes
                else:
                    print(f"[WARN] Polygon returned empty data for {symbol}")
            else:
                print(f"[WARN] Polygon API failed: {r.status_code} - {r.text[:200]}")

        # --- yfinance fallback ---
        if yf is not None:
            print(f"[INFO] Fetching close prices from yfinance for {symbol} ({start} to {end}) ...")
            df = yf.download(symbol, start=start, end=end)
            if not df.empty and "Close" in df:
                closes = df["Close"]
                closes.name = "close"
                print(f"[DEBUG] yfinance close prices:\n{closes.head()}")
                return closes
            else:
                print(f"[WARN] yfinance returned no data or empty series for {symbol}.")
        else:
            print("[ERROR] yfinance not installed.")

        print("[ERROR] Could not fetch close price series for", symbol, "from any source. Returning empty series.")
        return pd.Series(dtype=float)
