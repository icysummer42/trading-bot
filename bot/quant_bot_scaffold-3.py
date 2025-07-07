"""
Quant Options Trading Bot – One‑File Scaffold (Polygon Unusual‑Flow Edition)
===========================================================================
🔄 **What changed?**  Added a **debug print** inside
`SignalGenerator.aggregate_scores()` so you can see real‑time sentiment, vol
and event inputs as the edge score is computed:

```python
[DEBUG] FinBERT=0.432, Vol=0.018, Events={'unusual_flow': {...}}  
```

Everything else remains identical.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Soft imports – graceful degradation
# ---------------------------------------------------------------------------
try:
    import yfinance as yf  # Yahoo Finance
except ImportError:  # pragma: no cover
    yf = None  # type: ignore

try:
    import pandas_datareader.data as web  # FRED
except ImportError:  # pragma: no cover
    web = None  # type: ignore

try:
    from transformers import pipeline  # FinBERT
except ImportError:  # pragma: no cover
    pipeline = None  # type: ignore

try:
    from arch import arch_model  # GARCH volatility
except ImportError:  # pragma: no cover
    arch_model = None  # type: ignore

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
@dataclass
class Config:
    symbols_equity: List[str] = field(default_factory=lambda: ["AAPL", "TSLA", "NVDA"])
    symbols_etf: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "XLK", "XLE"])
    data_start: str = "2015-01-01"
    data_end: str = dt.date.today().isoformat()
    polygon_key: str = os.getenv("POLYGON_KEY", "")
    hibp_key: str = os.getenv("HIBP_API_KEY", "")
    te_api_key: str = os.getenv("TE_API_KEY", "")
    fred_key: str = os.getenv("FRED_API_KEY", "")
    nlp_model_name: str = "ProsusAI/finbert"
    vol_model: str = "garch"
    ewma_lambda: float = 0.94
    unusual_options_min_premium: float = 1_000_000.0
    cyber_severity_threshold: int = 7
    macro_surprise_sigma: float = 1.0
    edge_threshold: float = 0.8

# ---------------------------------------------------------------------------
# 2. DATA PIPELINE (unchanged)
# ---------------------------------------------------------------------------
class DataPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
    # … (same as previous content – omitted for brevity) …

# ---------------------------------------------------------------------------
# 3. EVENT PLUGINS (unchanged)
# ---------------------------------------------------------------------------
class EventPlugin(Protocol):
    def check(self) -> Dict[str, Any]: ...

class UnusualOptionsFlowPlugin:
    def __init__(self, cfg: Config):
        self.key = cfg.polygon_key
        self.min_notional = cfg.unusual_options_min_premium
        self.syms = cfg.symbols_equity + cfg.symbols_etf
        self.session = requests.Session()
        if self.key:
            self.session.params = {"apiKey": self.key}
    def _snapshot(self, u: str):
        if not self.key:
            return []
        try:
            r = self.session.get(f"https://api.polygon.io/v3/snapshot/options/{u}", timeout=4)
            r.raise_for_status()
            return r.json().get("results", [])
        except Exception:
            return []
    def check(self):
        if not self.key:
            return {}
        for sym in self.syms:
            for opt in self._snapshot(sym):
                ask = opt.get("last_quote", {}).get("p", 0)
                size = opt.get("last_quote", {}).get("s", 0)
                if ask*size*100 >= self.min_notional:
                    ctype = opt.get("details", {}).get("contract_type", "call").lower()
                    return {"unusual_flow": {"dir": "bull" if ctype=="call" else "bear", "mag": (ask*size*100)/self.min_notional}}
        return {}

class CyberSecurityBreachPlugin:
    def __init__(self, cfg: Config):
        self.th = cfg.cyber_severity_threshold
    def check(self):
        try:
            vul = requests.get("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json", timeout=3).json().get("vulnerabilities", [])
            if vul and vul[0].get("cvssScore", 0) >= self.th:
                return {"cyber_breach": {"ticker": None, "sev": vul[0]["cvssScore"]}}
        except Exception:
            pass
        return {}

class TopTierReleasePlugin:
    def __init__(self, cfg: Config):
        self.key = cfg.te_api_key; self.sigma = cfg.macro_surprise_sigma
    def check(self):
        if not self.key:
            return {}
        today = dt.date.today().isoformat()
        try:
            evs = [e for e in requests.get(f"https://api.tradingeconomics.com/calendar/events?c={self.key}&d1={today}&d2={today}", timeout=4).json() if e.get("Importance")==3]
            if not evs:
                return {}
            ev = evs[0]
            surprise = 0.0
            try:
                surprise = (float(ev["Actual"])-float(ev["Consensus"]))/self.sigma
            except Exception:
                pass
            return {"macro_release": {"name": ev.get("Event","macro"), "surprise": surprise}}
        except Exception:
            return {}

# ---------------------------------------------------------------------------
# 4. SIGNAL GENERATOR (added debug print)
# ---------------------------------------------------------------------------
class SignalGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pipe = None
        if pipeline is not None:
            try:
                self.pipe = pipeline("sentiment-analysis", model=cfg.nlp_model_name, tokenizer=cfg.nlp_model_name)
            except Exception:
                pass
        self.plugins: List[EventPlugin] = [UnusualOptionsFlowPlugin(cfg), CyberSecurityBreachPlugin(cfg), TopTierReleasePlugin(cfg)]

    def nlp_sentiment(self, texts: List[str]) -> float:
        if not texts or self.pipe is None:
            return 0.0
        res = [self.pipe(t)[0] for t in texts]
        return float(np.mean([(d["score"] if d["label"].lower()=="positive" else -d["score"]) for d in res]))

    def _ewma_vol(self, rets):
        return float(np.sqrt((rets**2).ewm(alpha=1-self.cfg.ewma_lambda).mean().iloc[-1]))
    def _garch_vol(self, rets):
        if arch_model is None:
            return None
        try:
            fit = arch_model(rets*100, p=1, q=1).fit(disp="off")
            return float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1,0])/100)
        except Exception:
            return None
    def forecast_volatility(self, close):
        r = close.pct_change().dropna()
        v = self._garch_vol(r) if self.cfg.vol_model=="garch" else None
        return v if v is not None else self._ewma_vol(r)

    def detect_events(self):
        ev: Dict[str,Any] = {}
        for p in self.plugins:
            ev.update(p.check())
        return ev

    def aggregate_scores(self, d: Dict[str,Any]):
        s, v, e = d.get("sentiment",0.0), d.get("vol",0.0), d.get("events",{})
        score = 0.6*s + 0.4*(v/0.05)
        if "unusual_flow" in e:
            f = e["unusual_flow"]; score += 0.2*f["mag"]*(1 if f["dir"]=="bull" else -1)
        if "cyber_breach" in e:
            score -= 0.15*(e["cyber_breach"]["sev"] / 10)
        if "macro_release" in e:
            score += 0.1*e["macro_release"]["surprise"]
        # DEBUG LINE -------------------------------------------------------
        print(f"[DEBUG] FinBERT={s:.3f}, Vol={v:.3f}, Events={e}")
        # -----------------------------------------------------------------
        return float(np.tanh(score))

# ---------------------------------------------------------------------------
# 5‑8. Strategy, Backtester, Execution, Risk (unchanged minimal stubs)
# ---------------------------------------------------------------------------
class StrategyEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
    def iron_condor(self, sym: str, edge: float):
        return {"strategy": "iron_condor", "symbol": sym, "wingspan": max(1,int(abs(edge)*10))}
    def generate_trades(self, edge: float, sym: str):
        if abs(edge) < 0.2:
            return [self.iron_condor(sym, edge)]
        return [{"strategy": "covered_call" if edge>0 else "put_debit_spread", "symbol": sym}]

class Backtester:
    def __init__(self, cfg: Config, dp: DataPipeline, sg: SignalGenerator, strat: StrategyEngine):
        self.cfg, self.dp, self.sg, self.strat = cfg, dp, sg, strat
    def simulate_symbol(self, sym: str):
        px = self.dp.fetch_equity_prices(sym)["close"]
        # inject signal path for debug printing
        sent = self.sg.nlp_sentiment([f"Backtest placeholder news for {sym}"])
        vol  = self.sg.forecast_volatility(px)
        ev   = self.sg.detect_events()
        _    = self.sg.aggregate_scores({"sentiment":sent, "vol":vol, "events":ev})
        return px.pct_change().fillna(0).cumsum()
    def run(self):
        curves = {s: self.simulate_symbol(s) for s in self.cfg.symbols_equity}
        print("Backtest finished – first rows:
", pd.DataFrame(curves).head())

class ExecutionEngine:
    def place_order(self, trade):
        print("[EXECUTE]", trade)

class RiskManager:
    def position_size(self, edge, bankroll):
        return bankroll*0.01

# ---------------------------------------------------------------------------
# 9. MAIN ENTRYPOINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    parser.add_argument("--once", action="store_true", help="run only one live iteration")
    args = parser.parse_args()

    cfg  = Config()
    dp   = DataPipeline(cfg)
    sg   = SignalGenerator(cfg)
    strat= StrategyEngine(cfg)
    exe  = ExecutionEngine()
    risk = RiskManager()

    if args.mode == "backtest":
        Backtester(cfg, dp, sg, strat).run()
    else:  # live mode
        bankroll = 1_000_000
        def live_cycle():
            for sym in cfg.symbols_equity:
                close = dp.fetch_equity_prices(sym)["close"]
                sent  = sg.nlp_sentiment([f"Live placeholder headline for {sym}"])
                vol   = sg.forecast_volatility(close)
                ev    = sg.detect_events()
                edge  = sg.aggregate_scores({"sentiment":sent,"vol":vol,"events":ev})
                if abs(edge) >= cfg.edge_threshold:
                    for t in strat.generate_trades(edge, sym):
                        t["size_$"] = risk.position_size(edge, bankroll)
                        exe.place_order(t)
        if args.once:
            live_cycle()
        else:
            import time
            while True:
                live_cycle()
                time.sleep(60)

if __name__ == "__main__":
    main()
