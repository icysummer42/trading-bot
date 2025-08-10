"""Market & macro data ingestion layer."""
from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from typing import Dict
from config import Config
import requests, warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

# Optional libs
try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore

try:
    import pandas_datareader.data as web
except ImportError:
    web = None  # type: ignore

class DataPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.polygon_key = getattr(cfg, "polygon_key", None) or os.getenv("POLYGON_API_KEY")
        self.finnhub_key = getattr(cfg, "finnhub_api_key", None) or os.getenv("FINNHUB_API_KEY")
    
    def get_close_series(self, symbol: str, start=None, end=None):
        """Load historical close prices for a symbol as a pd.Series."""
        # Default: last 1 year
        if not start:
            start = (dt.date.today() - dt.timedelta(days=365)).isoformat()
        if not end:
            end = dt.date.today().isoformat()

        # --- Polygon API ---
        if self.polygon_key:
            print("[INFO] Fetching close prices from Polygon...")
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start}/{end}"
                f"?adjusted=true&sort=asc&apiKey={self.polygon_key}"
            )
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json().get("results", [])
                if data:
                    closes = pd.Series([row["c"] for row in data], 
                        index=pd.to_datetime([dt.datetime.fromtimestamp(row["t"]/1000).date() for row in data]))
                    closes.name = "close"
                    print(f"[DEBUG] Polygon close prices: {closes.head()}")
                    return closes
            print("[WARN] Polygon API failed or returned no data.")

        # --- Finnhub API ---
        if self.finnhub_key:
            print("[INFO] Fetching close prices from Finnhub...")
            url = (
                f"https://finnhub.io/api/v1/stock/candle?symbol={symbol.upper()}&resolution=D"
                f"&from={(pd.to_datetime(start) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')}"
                f"&to={(pd.to_datetime(end) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')}"
                f"&token={self.finnhub_key}"
            )
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                d = r.json()
                if d.get("c"):
                    closes = pd.Series(d["c"], 
                        index=pd.to_datetime(d["t"], unit="s"))
                    closes.name = "close"
                    print(f"[DEBUG] Finnhub close prices: {closes.head()}")
                    return closes
            print("[WARN] Finnhub API failed or returned no data.")

        # --- Yahoo Finance fallback ---
        if yf is not None:
            print("[INFO] Fetching close prices from yfinance...")
            df = yf.download(symbol, start=start, end=end)
            if "Close" in df and not df["Close"].empty:
                closes = df["Close"]
                closes.name = "close"
                print(f"[DEBUG] yfinance close prices: {closes.head()}")
                return closes
            else:
                print("[WARN] yfinance returned no data or empty series.")
        else:
            print("[ERROR] yfinance not installed.")

        # Last defense: empty series
        print(f"[ERROR] Could not fetch close price series for {symbol} from any source. Returning empty series.")
        return pd.Series([], name="close")
