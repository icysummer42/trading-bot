"""Event‑trigger plugins: unusual flow, breaches, macro releases."""
from __future__ import annotations
import datetime as dt, requests
from typing import Any, Dict, List
from config import Config

class EventPlugin:
    """Interface for plugin objects."""
    def check(self) -> Dict[str, Any]:
        raise NotImplementedError

# ── Polygon unusual options flow ──────────────────────────────────────────
class UnusualOptionsFlowPlugin(EventPlugin):
    def __init__(self, cfg: Config):
        self.key = cfg.polygon_key
        self.min_notional = cfg.unusual_options_min_premium
        self.symbols: List[str] = cfg.symbols_equity + cfg.symbols_etf
        self.s = requests.Session()
        if self.key:
            self.s.params = {"apiKey": self.key}

    def _snapshot(self, underlying: str):
        try:
            url = f"https://api.polygon.io/v3/snapshot/options/{underlying}"
            return self.s.get(url, timeout=4).json().get("results", [])
        except Exception:
            return []

    def check(self):
        if not self.key:
            return {}
        for sym in self.symbols:
            for opt in self._snapshot(sym):
                q = opt.get("last_quote", {})
                notional = q.get("p", 0)*q.get("s", 0)*100
                if notional >= self.min_notional:
                    ctype = opt.get("details", {}).get("contract_type", "call").lower()
                    return {"unusual_flow": {"dir": "bull" if ctype=="call" else "bear",
                                              "mag": notional/self.min_notional}}
        return {}

# ── CISA breach feed ──────────────────────────────────────────────────────
class CyberSecurityBreachPlugin(EventPlugin):
    def __init__(self, cfg: Config):
        self.threshold = cfg.cyber_severity_threshold
    def check(self):
        try:
            vul = requests.get("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
                               timeout=3).json().get("vulnerabilities", [])
            if vul and vul[0].get("cvssScore", 0) >= self.threshold:
                return {"cyber_breach": {"ticker": None, "sev": vul[0]["cvssScore"]}}
        except Exception:
            pass
        return {}

# ── Trading Economics macro calendar ──────────────────────────────────────
class TopTierReleasePlugin(EventPlugin):
    def __init__(self, cfg: Config):
        self.key = cfg.te_api_key
        self.sigma = cfg.macro_surprise_sigma
    def check(self):
        if not self.key:
            return {}
        today = dt.date.today().isoformat()
        try:
            evs = [e for e in requests.get(
                f"https://api.tradingeconomics.com/calendar/events?c={self.key}&d1={today}&d2={today}",
                timeout=4).json() if e.get("Importance") == 3]
            if not evs:
                return {}
            ev = evs[0]
            try:
                surprise = (float(ev["Actual"])-float(ev["Consensus"]))/self.sigma
            except Exception:
                surprise = 0.0
            return {"macro_release": {"name": ev.get("Event", "macro"), "surprise": surprise}}
        except Exception:
            return {}
