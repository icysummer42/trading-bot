"""Configuration and constants for the Quant Options Bot.

This dataclass stores all core symbols, model parameters, API keys, and feature switches.
API keys and settings are loaded from environment variables (preferred for secrets).
"""

from __future__ import annotations
import os
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class Config:
    """
    Configuration hub: symbols, API keys, model params, event thresholds.
    Read secrets from environment variables for security.
    """

    # Universe: equities and ETFs
    symbols_equity: List[str] = field(default_factory=lambda: ["AAPL", "TSLA", "NVDA"])
    symbols_etf: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "XLK", "XLE"])

    # Date range for historical fetches
    data_start: str = "2015-01-01"
    data_end: str = dt.date.today().isoformat()

    # === API Keys and Integrations ===

    # Polygon.io
    polygon_key: str = os.getenv("POLYGON_KEY", "")
    # Finnhub
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    # Reddit (PRAW)
    reddit: Dict[str, str] = field(default_factory=lambda: {
        "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
        "user_agent": os.getenv("REDDIT_USER_AGENT", "sentiment-bot")
    })
    # Stocktwits
    stocktwits_token: str = os.getenv("STOCKTWITS_TOKEN", "")
    # NewsAPI.org
    newsapi_key: str = os.getenv("NEWSAPI_KEY", "")
    # GNews (Google News API)
    gnews_key: str = os.getenv("GNEWS_API_KEY", "")
    # Google Trends (no key required for pytrends)
    # QuiverQuant
    quiver_api_key: str = os.getenv("QUIVER_API_KEY", "")
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Other optional APIs
    te_api_key: str = os.getenv("TE_API_KEY", "")
    fred_key: str = os.getenv("FRED_API_KEY", "")
    hibp_key: str = os.getenv("HIBP_API_KEY", "")

    # === Modelling ===
    nlp_model_name: str = os.getenv("NLP_MODEL_NAME", "ProsusAI/finbert")
    vol_model: str = os.getenv("VOL_MODEL", "garch")  # "ewma" or "garch"
    ewma_lambda: float = float(os.getenv("EWMA_LAMBDA", 0.94))

    # === Event thresholds ===
    unusual_options_min_premium: float = float(os.getenv("UNUSUAL_OPTIONS_MIN_PREMIUM", 1_000_000.0))  # USD
    cyber_severity_threshold: int = int(os.getenv("CYBER_SEVERITY_THRESHOLD", 7))
    macro_surprise_sigma: float = float(os.getenv("MACRO_SURPRISE_SIGMA", 1.0))

    # === Trading/Scoring ===
    edge_threshold: float = float(os.getenv("EDGE_THRESHOLD", 0.8))
    # Scoring weights can be overridden at runtime (signal_weights as dict)
    signal_weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """
        Allow override of signal_weights from environment (as comma-separated).
        """
        # Parse weights if available as env var (e.g. "sentiment=0.5,volatility=0.3,events=0.2")
        wstr = os.getenv("SIGNAL_WEIGHTS", "")
        if wstr:
            try:
                parts = [kv.strip().split("=") for kv in wstr.split(",") if "=" in kv]
                self.signal_weights = {k: float(v) for k, v in parts}
            except Exception as e:
                print(f"[WARN] Failed to parse SIGNAL_WEIGHTS: {e}")
        # Backward compatibility for legacy vars
        if self.signal_weights is None:
            s = os.getenv("SIGNAL_SENTIMENT_WEIGHT")
            v = os.getenv("SIGNAL_VOLATILITY_WEIGHT")
            e = os.getenv("SIGNAL_EVENTS_WEIGHT")
            if any([s, v, e]):
                self.signal_weights = {
                    "sentiment": float(s or 0.5),
                    "volatility": float(v or 0.3),
                    "events": float(e or 0.2),
                }
