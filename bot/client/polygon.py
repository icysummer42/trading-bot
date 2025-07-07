"""Very thin Polygon API wrapper used by the back-tester."""

from __future__ import annotations

import datetime as dt
import requests
from typing import Optional


class PolygonClient:
    """Wrap the handful of Polygon endpoints we need."""

    BASE = "https://api.polygon.io"

    def __init__(self, cfg) -> None:
        self.key: str = cfg.polygon_api_key
        self.s: requests.Session = requests.Session()

    # --------------------------------------------------------------------- #
    # NEW helper – used by IronCondorStrategy
    # --------------------------------------------------------------------- #
    def spot(self, symbol: str, trade_date: dt.date) -> Optional[float]:
        """
        Return the *adjusted* close price for `symbol` on `trade_date`.

        Falls back to None if Polygon has no data (e.g., weekend/holiday)
        or a network timeout occurs.
        """
        url = (
            f"{self.BASE}/v1/open-close/{symbol.upper()}/{trade_date}"
            f"?adjusted=true&apiKey={self.key}"
        )
        try:
            r = self.s.get(url, timeout=4)
            if not r.ok or not r.headers.get("content-type", "").startswith("application/json"):
                return None
            data = r.json()
            # Polygon returns: { 'status':'OK', 'close':123.45, ... }
            return float(data.get("close")) if "close" in data else None
        except (requests.exceptions.RequestException, ValueError):
            return None

    # --------------------------------------------------------------------- #
    # (Optional) cacheable helper for option contract close prices
    # --------------------------------------------------------------------- #
    def option_close(self, occ_ticker: str, trade_date: dt.date) -> Optional[float]:
        url = (
            f"{self.BASE}/v2/aggs/ticker/{occ_ticker}/range/1/day/"
            f"{trade_date}/{trade_date}?apiKey={self.key}"
        )
        try:
            r = self.s.get(url, timeout=4)
            if not r.ok or not r.headers.get("content-type", "").startswith("application/json"):
                return None
            results = r.json().get("results")
            return float(results[0]["c"]) if results else None
        except (requests.exceptions.RequestException, ValueError, KeyError):
            return None
