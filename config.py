"""Configuration and constants for the Quant Options Bot."""
from __future__ import annotations
import datetime as dt, os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    """Central place for symbols, API keys, thresholds and model params."""

    # Universe
    symbols_equity: List[str] = field(default_factory=lambda: ["AAPL", "TSLA", "NVDA"])
    symbols_etf: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "XLK", "XLE"])

    # Date range for historical fetches
    data_start: str = "2015-01-01"
    data_end: str = dt.date.today().isoformat()

    # Third‑party API keys (read from environment for security)
    polygon_key: str = os.getenv("POLYGON_KEY", "")
    te_api_key: str = os.getenv("TE_API_KEY", "")
    fred_key: str = os.getenv("FRED_API_KEY", "")
    hibp_key: str = os.getenv("HIBP_API_KEY", "")

    # Modelling
    nlp_model_name: str = "ProsusAI/finbert"
    vol_model: str = "garch"  # or "ewma"
    ewma_lambda: float = 0.94

    # Event thresholds
    unusual_options_min_premium: float = 1_000_000.0  # USD
    cyber_severity_threshold: int = 7
    macro_surprise_sigma: float = 1.0

    # Trading
    edge_threshold: float = 0.8
