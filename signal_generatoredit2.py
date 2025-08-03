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
        # Configurable weights
        self.weights = getattr(cfg, "signal_weights", {
            "sentiment": 0.5,
            "volatility": 0.3,
            "events": 0.2
        })

    # ── Sentiment ─────────────────────────────────────────────
    def fetch_headlines(self, symbol: str) -> List[str]:
        """Fetch or stub recent news headlines for the symbol."""
        # TODO: Integrate with real news API, eg Finnhub, NewsAPI, Yahoo
        return [
            f"{symbol} launches new product, stock rallies",
            f"Analysts remain bullish on {symbol} amid market volatility"
        ]
    
    def nlp_sentiment(self, texts: List[str]) -> float:
        if not texts or self.pipe is None:
            return 0.0
        res = [self.pipe(t)[0] for t in texts]
        scores = [(d["score"] if d["label"].lower() == "positive" else -d["score"]) for d in res]
        sent_mean = float(np.mean(scores))
        print(f"[DEBUG] Sentiment inputs: {texts} -> Scores: {scores} -> Mean: {sent_mean:.3f}")
        return sent_mean

    def symbol_sentiment(self, symbol: str) -> float:
        headlines = self.fetch_headlines(symbol)
        return self.nlp_sentiment(headlines)

    # ── Volatility forecast ─────────────────────────────
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
        if vol is not None:
            print(f"[DEBUG] GARCH Vol: {vol:.4f}")
            return vol
        ewma = self._ewma_vol(rets)
        print(f"[DEBUG] EWMA Vol: {ewma:.4f}")
        return ewma

    # ── Event aggregation ─────────────────────────────
    def detect_events(self, symbol: str) -> Dict[str, Any]:
        ev: Dict[str, Any] = {}
        for p in self.plugins:
            out = p.check(symbol=symbol) if "symbol" in p.check.__code__.co_varnames else p.check()
            ev.update(out)
        print(f"[DEBUG] Events: {ev}")
        return ev

    # ── Final edge score ─────────────────────────────
    def aggregate_scores(self, sentiment: float, vol: float, events: Dict[str,Any]) -> float:
        w = self.weights
        score = w["sentiment"] * sentiment + w["volatility"] * (vol/0.05)
        # Simple linear event adjustments (or aggregate to a float score)
        if "unusual_flow" in events:
            f = events["unusual_flow"]; score += 0.2*f["mag"]*(1 if f["dir"]=="bull" else -1)
        if "cyber_breach" in events:
            score -= 0.15*(events["cyber_breach"]["sev"]/10)
        if "macro_release" in events:
            score += 0.1*events["macro_release"]["surprise"]
        print(f"[DEBUG] Weights: {w} | Sentiment: {sentiment:.3f} | Vol: {vol:.3f} | Events: {events} | Raw score: {score:.3f}")
        return float(np.tanh(score))

    # ── Convenience: Single call for a symbol ────────
    def get_signal_score(self, symbol: str, close_series) -> float:
        sentiment = self.symbol_sentiment(symbol)
        vol = self.forecast_volatility(close_series)
        events = self.detect_events(symbol)
        return self.aggregate_scores(sentiment, vol, events)
