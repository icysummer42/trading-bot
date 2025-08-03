"""Market & macro data ingestion layer."""
from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from typing import Dict
from config import Config
import requests, warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Optional libs
try: import yfinance as yf
except ImportError: yf = None  # type: ignore

try: import pandas_datareader.data as web
except ImportError: web = None  # type: ignore

class DataPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── Equities ──────────────────────────────────────────────────────────
    def fetch_equity_prices(self, sym: str) -> pd.DataFrame:
        if yf is None:
            return self._mock(sym)
        try:
            df = yf.Ticker(sym).history(start=self.cfg.data_start,
                                        end=self.cfg.data_end,
                                        auto_adjust=False)
            df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                               "Close": "close", "Volume": "volume"}, inplace=True)
            df["symbol"] = sym
            return df
        except Exception:
            return self._mock(sym)

    def _mock(self, sym: str) -> pd.DataFrame:
        idx = pd.date_range(self.cfg.data_start, dt.date.today(), freq="B")
        df = pd.DataFrame({
            "open":   np.random.rand(len(idx))*100,
            "high":   np.random.rand(len(idx))*101,
            "low":    np.random.rand(len(idx))*99,
            "close":  np.random.rand(len(idx))*100,
            "volume": np.random.randint(1e6, 5e6, len(idx)),
        }, index=idx)
        df["symbol"] = sym
        return df

    # ── Options chain (nearest expiry, coarse) ────────────────────────────
    def fetch_options_chain(self, sym: str) -> pd.DataFrame:
        if yf is None:
            return pd.DataFrame()
        try:
            t = yf.Ticker(sym)
            exp = t.options[:1]
            chains = [pd.concat([t.option_chain(e).calls.assign(type="call", expiry=e),
                                  t.option_chain(e).puts.assign(type="put", expiry=e)])
                      for e in exp]
            return pd.concat(chains, ignore_index=True)
        except Exception:
            return pd.DataFrame()

    # ── Macro (FRED) ──────────────────────────────────────────────────────
    def fetch_macro(self) -> pd.DataFrame:
        if web is None or not self.cfg.fred_key:
            return pd.DataFrame()
        try:
            ids: Dict[str, str] = {"CPI": "CPIAUCSL", "FED": "FEDFUNDS"}
            return pd.DataFrame({k: web.DataReader(v, "fred", self.cfg.data_start,
                                                   self.cfg.data_end,
                                                   api_key=self.cfg.fred_key)[v]
                                 for k, v in ids.items()})
        except Exception:
            return pd.DataFrame()
