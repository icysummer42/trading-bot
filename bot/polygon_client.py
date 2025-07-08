import datetime as dt
from typing import List

import pandas as pd
import requests
import os, pickle, hashlib, functools

import threading
import time

def disk_cache(cache_dir="cache", version="v1"):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            os.makedirs(cache_dir, exist_ok=True)
            # Version key avoids stale cache
            key_str = f"{version}:{fn.__name__}:{str(args)}:{str(kwargs)}"
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"{key_hash}.pkl")

            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    result = pickle.load(f)
                print(f"[CACHE HIT] {fn.__name__} {args} {kwargs}")
                return result

            result = fn(*args, **kwargs)

            # Only cache valid results: not None, not empty DataFrame
            should_cache = (
                result is not None and
                (not hasattr(result, "empty") or not result.empty)
            )
            if should_cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
                print(f"[CACHE MISS - SAVED] {fn.__name__} {args} {kwargs}")
            else:
                print(f"[CACHE MISS - NOT CACHED] {fn.__name__} {args} {kwargs}")
            return result
        return wrapper
    return decorator

class AdaptiveRateLimiter:
    """
    Adaptive rate limiter for API calls.
    - Backs off aggressively on 429, recovers slowly after success.
    """
    def __init__(self, min_interval=0.01, max_interval=2.0, backoff_factor=2.0, recovery_factor=1.05):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self._interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0

    def wait(self):
        with self._lock:
            now = time.time()
            wait_time = self._interval - (now - self._last_call)
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_call = time.time()

    def on_success(self):
        with self._lock:
            self._interval = max(self.min_interval, self._interval / self.recovery_factor)

    def on_rate_limit(self):
        with self._lock:
            self._interval = min(self.max_interval, self._interval * self.backoff_factor)

class PolygonClient:
    """Thin wrapper around Polygon.io endpoints with basic retry."""

    def __init__(self, api_key: str | None = None):
        self.s = requests.Session()
        if api_key:
            self.s.params = {"apiKey": api_key}
        self.rate_limiter = AdaptiveRateLimiter(min_interval=1.0, max_interval=10.0)

    def _get(self, url: str, timeout: int = 6, retries: int = 6):
        last_exc = None
        for attempt in range(retries):
            self.rate_limiter.wait()
            now = time.time()
            now_hr = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"[REQ] {now_hr} ({now:.3f}) {url} [interval={self.rate_limiter._interval:.3f}s]")
            try:
                r = self.s.get(url, timeout=timeout)
                if r.status_code == 429:
                    print(f"[WARN] 429 Rate limit hit. Backing off (attempt {attempt+1}/{retries})")
                    self.rate_limiter.on_rate_limit()
                    continue
                self.rate_limiter.on_success()
                return r
            except requests.exceptions.Timeout as e:
                print(f"[ERROR] Timeout for {url} (attempt {attempt+1}/{retries})")
                last_exc = e
                self.rate_limiter.on_rate_limit()
                continue
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Network error for {url} (attempt {attempt+1}/{retries}): {e}")
                last_exc = e
                self.rate_limiter.on_rate_limit()
                continue
        print(f"[ERROR] All retries failed for {url}")
        if last_exc:
            print(f"Last exception: {last_exc}")
        return None

    @disk_cache(cache_dir='cache/chain_on', version='v1')
    def chain_on(self, sym: str, date: dt.date, expiry: dt.date) -> pd.DataFrame:
        url = ("https://api.polygon.io/v3/reference/options/contracts?"
        f"underlying_ticker={sym.upper()}&expiration_date={expiry}&as_of={date}&limit=1000")
        r = self._get(url)
        if r is None or r.status_code != 200:
            print(f"[ERROR] chain_on: No response or bad status for {sym} {date} {expiry}")
            return pd.DataFrame()
        try:
            data = r.json().get("results", [])
        except Exception as e:
            print(f"[ERROR] chain_on: Failed to parse JSON for {sym} {date} {expiry}: {e}")
            return pd.DataFrame()
        if not data:
            print(f"[WARN] chain_on: No contracts for {sym} {date} {expiry}")
            return pd.DataFrame()
        return pd.json_normalize(data)

    @disk_cache(cache_dir='cache/snapshot_chain', version='v1')
    def snapshot_chain(self, sym: str, date: dt.date) -> pd.DataFrame:
        url = (
            "https://api.polygon.io/v3/reference/options/contracts?"
            f"underlying_ticker={sym.upper()}&as_of={date}&limit=1000"
        )
        r = self._get(url)
        if r is None or r.status_code != 200:
            print(f"[ERROR] snapshot_chain: No response or bad status for {sym} {date}")
            return pd.DataFrame()
        try:
            data = r.json().get("results", [])
        except Exception as e:
            print(f"[ERROR] snapshot_chain: Failed to parse JSON for {sym} {date}: {e}")
            return pd.DataFrame()
        if not data:
            print(f"[WARN] snapshot_chain: No contracts for {sym} {date}")
            return pd.DataFrame()
        return pd.json_normalize(data)

    @disk_cache(cache_dir='cache/agg_close', version='v1')
    def agg_close(self, ticker: str, date: dt.date) -> float | None:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}"
        r = self._get(url, timeout=4)
        if r is None or r.status_code != 200:
            print(f"[ERROR] agg_close: No response or error status for {ticker} on {date}")
            return None
        res = r.json().get("results", [])
        if not res:
            print(f"[WARN] agg_close: No price results for {ticker} on {date}")
            return None
        return float(res[0]["c"])

    @disk_cache(cache_dir='cache/open_close', version='v1')
    def open_close(self, ticker: str, date: dt.date) -> float | None:
        url = f"https://api.polygon.io/v1/open-close/{ticker}/{date}"
        try:
            r = self._get(url, timeout=4)
        except Exception as e:
            print(f"[ERROR] open_close: Exception for {ticker} {date}: {e}")
            return None
        if r is None or r.status_code != 200:
            print(f"[ERROR] open_close: No response or bad status for {ticker} {date}")
            return None
        try:
            close = r.json().get("close")
        except Exception as e:
            print(f"[ERROR] open_close: Failed to parse JSON for {ticker} {date}: {e}")
            return None
        if close is None:
            print(f"[WARN] open_close: No close price for {ticker} {date}")
        return close

    @disk_cache(cache_dir='cache/expires_on', version='v1')
    def expiries_on(self, sym: str, as_of: dt.date) -> List[dt.date]:
        url = (
            "https://api.polygon.io/v3/reference/options/contracts?"
            f"underlying_ticker={sym.upper()}&as_of={as_of}&limit=1000"
        )
        r = self._get(url)
        if r is None or r.status_code != 200:
            print(f"[ERROR] expiries_on: No response or bad status for {sym} {as_of}")
            return []
        try:
            rows = r.json().get("results", [])
        except Exception as e:
            print(f"[ERROR] expiries_on: Failed to parse JSON for {sym} {as_of}: {e}")
            return []
        if not rows:
            print(f"[WARN] expiries_on: No expiries for {sym} {as_of}")
            return []
        try:
            return sorted({
                dt.datetime.strptime(row["expiration_date"], "%Y-%m-%d").date()
                for row in rows if "expiration_date" in row
            })
        except Exception as e:
            print(f"[ERROR] expiries_on: Date parse error for {sym} {as_of}: {e}")
            return []

    @disk_cache(cache_dir='cache/spot', version='v1')
    def spot(self, sym: str, date: dt.date) -> float | None:
        """Return the closing price (spot) for the symbol on a given date."""
        return self.agg_close(sym, date)
