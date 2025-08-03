"""Sentiment, volatility forecasting, and edge‑score aggregation."""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List
from config import Config
from data_pipeline import DataPipeline
from plugins import UnusualOptionsFlowPlugin, CyberSecurityBreachPlugin, TopTierReleasePlugin

try:
    from transformers import pipeline as hf_pipeline  # FinBERT sentiment
except ImportError:
    hf_pipeline = None  # type: ignore

try:
    from arch import arch_model
except ImportError:
    arch_model = None  # type: ignore

class SignalGenerator:
    def __init__(self, cfg: Config, dp: DataPipeline):
        self.cfg = cfg; self.dp = dp
        self.pipe = None
        if hf_pipeline is not None:
            try:
                self.pipe = hf_pipeline("sentiment-analysis",
                                       model=cfg.nlp_model_name,
                                       tokenizer=cfg.nlp_model_name)
            except Exception:
                pass
        # Event plugins
        self.plugins = [UnusualOptionsFlowPlugin(cfg),
                        CyberSecurityBreachPlugin(cfg),
                        TopTierReleasePlugin(cfg)]

    # ── Sentiment ─────────────────────────────────────────────────────────
    def nlp_sentiment(self, texts: List[str]) -> float:
        if not texts or self.pipe is None:
            return 0.0
        res = [self.pipe(t)[0] for t in texts]
        return float(np.mean([(d["score"] if d["label"].lower()=="positive" else -d["score"]) for d in res]))

    # ── Volatility forecast ───────────────────────────────────────────────
    def _ewma_vol(self, rets):
        lam = self.cfg.ewma_lambda
        return float(np.sqrt((rets**2).ewm(alpha=1-lam).mean().iloc[-1]))
    def _garch_vol(self, rets):
        if arch_model is None:
            return None
        try:
            fit = arch_model(rets*100, p=1, q=1).fit(disp="off")
            return float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1,0])/100)
        except Exception:
            return None
    def forecast_volatility(self, close):
        rets = close.pct_change().dropna()
        vol  = self._garch_vol(rets) if self.cfg.vol_model == "garch" else None
        return vol if vol is not None else self._ewma_vol(rets)

    # ── Event aggregation ────────────────────────────────────────────────
    def detect_events(self):
        ev: Dict[str, Any] = {}
        for p in self.plugins:
            ev.update(p.check())
        return ev

    # ── Final edge score ─────────────────────────────────────────────────
    def aggregate_scores(self, sentiment: float, vol: float, events: Dict[str,Any]):
        score = 0.6*sentiment + 0.4*(vol/0.05)
        if "unusual_flow" in events:
            f = events["unusual_flow"]; score += 0.2*f["mag"]*(1 if f["dir"]=="bull" else -1)
        if "cyber_breach" in events:
            score -= 0.15*(events["cyber_breach"]["sev"]/10)
        if "macro_release" in events:
            score += 0.1*events["macro_release"]["surprise"]
        # DEBUG
        print(f"[DEBUG] FinBERT={sentiment:.3f}, Vol={vol:.3f}, Events={events}")
        return float(np.tanh(score))
