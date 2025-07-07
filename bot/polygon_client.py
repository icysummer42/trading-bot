import datetime as dt
import time
from typing import List

import pandas as pd
import requests

class PolygonClient:
    """Thin wrapper around Polygon.io endpoints with basic retry."""

    def __init__(self, api_key: str | None = None):
        self.s = requests.Session()
        if api_key:
            self.s.params = {"apiKey": api_key}

    def _get(self, url: str, timeout: int = 6, retries: int = 6):
        for attempt in range(retries):
            r = self.s.get(url, timeout=timeout)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            return r
        return r

    def chain_on(self, sym: str, date: dt.date, expiry: dt.date) -> pd.DataFrame:
        url = (
            "https://api.polygon.io/v3/reference/options/contracts?"
            f"underlying_ticker={sym.upper()}&expiration_date={expiry}&as_of={date}&limit=1000"
        )
        r = self._get(url)
        if r.status_code != 200:
            return pd.DataFrame()
        return pd.json_normalize(r.json().get("results", []))

    def snapshot_chain(self, sym: str, date: dt.date) -> pd.DataFrame:
        url = (
            "https://api.polygon.io/v3/reference/options/contracts?"
            f"underlying_ticker={sym.upper()}&as_of={date}&limit=1000"
        )
        r = self._get(url)
        if r.status_code != 200:
            return pd.DataFrame()
        return pd.json_normalize(r.json().get("results", []))

    def agg_close(self, ticker: str, date: dt.date) -> float | None:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}"
        r = self._get(url, timeout=4)
        if r.status_code != 200:
            return None
        res = r.json().get("results", [])
        return float(res[0]["c"]) if res else None

    def open_close(self, ticker: str, date: dt.date) -> float | None:
        url = f"https://api.polygon.io/v1/open-close/{ticker}/{date}"
        try:
            r = self._get(url, timeout=4)
        except Exception:
            return None
        if r.status_code != 200:
            return None
        return r.json().get("close")

    def expiries_on(self, sym: str, as_of: dt.date) -> List[dt.date]:
        url = (
            "https://api.polygon.io/v3/reference/options/contracts?"
            f"underlying_ticker={sym.upper()}&as_of={as_of}&limit=1000"
        )
        r = self._get(url)
        if r.status_code != 200:
            return []
        rows = r.json().get("results", [])
        return sorted(
            {dt.datetime.strptime(row["expiration_date"], "%Y-%m-%d").date() for row in rows}
        )

    def spot(self, sym: str, date: dt.date) -> float | None:
        """Return the closing price (spot) for the symbol on a given date."""
        return self.agg_close(sym, date)
