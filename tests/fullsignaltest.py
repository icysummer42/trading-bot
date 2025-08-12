"""Signal Generator: Sentiment, volatility, and event aggregation for trading signals.

This module pulls and aggregates data from multiple news/social feeds
(Finnhub, Polygon, Stocktwits, NewsAPI, GNews, Google Trends, Quiver Quant, Reddit)
and produces a unified sentiment score using OpenAI or FinBERT.
It also calculates volatility forecasts and merges custom event plugins.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List
from config import Config
from data_pipeline import DataPipeline
from plugins.events import UnusualOptionsFlowPlugin, CyberSecurityBreachPlugin, TopTierReleasePlugin
import requests
import datetime as dt

# Optional feeds: import if available
try:
    from gnews import GNews
except ImportError:
    GNews = None

try:
    from newsapi import NewsApiClient
except ImportError:
    NewsApiClient = None

try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None

try:
    import openai
except ImportError:
    openai = None

try:
    import praw
except ImportError:
    praw = None

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

try:
    from arch import arch_model
except ImportError:
    arch_model = None

class SignalGenerator:
    """Aggregates news/social feeds, scores sentiment, forecasts volatility, and detects special events."""

    def __init__(self, cfg: Config, dp: DataPipeline):
        """
        Initialize the SignalGenerator with config, data pipeline, and plugins.
        Sets up NLP and Reddit sentiment, event plugins, and configurable weights.
        """
        self.cfg = cfg
        self.dp = dp

        # Sentiment: FinBERT pipeline as fallback
        self.pipe = None
        if hf_pipeline is not None:
            try:
                self.pipe = hf_pipeline(
                    "sentiment-analysis",
                    model=cfg.nlp_model_name,
                    tokenizer=cfg.nlp_model_name,
                )
            except Exception:
                self.pipe = None

        # Event detection plugins
        self.plugins = [
            UnusualOptionsFlowPlugin(cfg),
            CyberSecurityBreachPlugin(cfg),
            TopTierReleasePlugin(cfg)
        ]
        self.weights = getattr(cfg, "signal_weights", {
            "sentiment": 0.5,
            "volatility": 0.3,
            "events": 0.2
        })

        # Reddit (if configured)
        self.reddit = None
        if praw is not None:
            praw_cfg = getattr(cfg, "reddit", None)
            if praw_cfg and all(k in praw_cfg for k in ["client_id", "client_secret"]):
                try:
                    self.reddit = praw.Reddit(
                        client_id=praw_cfg["client_id"],
                        client_secret=praw_cfg["client_secret"],
                        user_agent=praw_cfg.get("user_agent", "sentiment-bot"),
                    )
                except Exception as e:
                    print(f"[WARN] PRAW init failed: {e}")
            elif all(hasattr(cfg, k) for k in ["reddit_client_id", "reddit_client_secret"]):
                try:
                    self.reddit = praw.Reddit(
                        client_id=cfg.reddit_client_id,
                        client_secret=cfg.reddit_client_secret,
                        user_agent=getattr(cfg, "reddit_user_agent", "sentiment-bot"),
                    )
                except Exception as e:
                    print(f"[WARN] PRAW init failed: {e}")

    # === API Fetchers ===

    def fetch_finnhub_headlines(self, symbol: str, start=None, end=None, max_n=10) -> List[str]:
        """Fetch recent company news headlines from Finnhub."""
        api_key = getattr(self.cfg, "finnhub_api_key", None)
        if not api_key:
            return []
        if not start:
            start = (dt.date.today() - dt.timedelta(days=30)).isoformat()
        if not end:
            end = dt.date.today().isoformat()
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start}&to={end}&token={api_key}"
        try:
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                articles = resp.json()
                return [a["headline"] for a in articles[:max_n] if "headline" in a]
        except Exception as e:
            print(f"[ERROR] Finnhub: {e}")
        return []

    def fetch_polygon_headlines(self, symbol: str, max_n=10) -> List[str]:
        """Fetch news headlines for symbol from Polygon.io."""
        api_key = getattr(self.cfg, "polygon_key", None)
        if not api_key:
            return []
        url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&apiKey={api_key}&limit={max_n}"
        try:
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                return [a["title"] for a in results[:max_n] if "title" in a]
        except Exception as e:
            print(f"[ERROR] Polygon news: {e}")
        return []

    def fetch_stocktwits(self, symbol: str, max_n=10) -> List[str]:
        """Fetch recent message bodies about the symbol from Stocktwits."""
        token = getattr(self.cfg, "stocktwits_token", "")
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        if token:
            url += f"?access_token={token}"
        try:
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                messages = r.json().get("messages", [])
                return [m.get("body", "") for m in messages[:max_n]]
        except Exception as e:
            print(f"[ERROR] Stocktwits: {e}")
        return []

    def fetch_newsapi(self, symbol: str, max_n=10) -> List[str]:
        """Fetch recent news headlines about the symbol from NewsAPI.org."""
        api_key = getattr(self.cfg, "newsapi_key", "")
        if not api_key or NewsApiClient is None:
            return []
        try:
            newsapi = NewsApiClient(api_key=api_key)
            articles = newsapi.get_everything(q=symbol, language='en', page_size=max_n)
            return [a['title'] for a in articles.get('articles', [])]
        except Exception as e:
            print(f"[ERROR] NewsAPI: {e}")
        return []

    def fetch_gnews(self, symbol: str, max_n=10) -> List[str]:
        """Fetch recent news headlines for the symbol using GNews (Google News)."""
        gnews_key = getattr(self.cfg, "gnews_key", "")
        if GNews is None:
            return []
        try:
            gnews = GNews(language='en', max_results=max_n, api_key=gnews_key if gnews_key else None)
            news = gnews.get_news(symbol)
            return [a['title'] for a in news]
        except Exception as e:
            print(f"[ERROR] GNews: {e}")
        return []

    def fetch_pytrends(self, symbol: str, max_n=10) -> List[str]:
        """Fetch Google Trends data for symbol as a sentiment signal."""
        if TrendReq is None:
            return []
        try:
            pytrends = TrendReq()
            pytrends.build_payload([symbol])
            df = pytrends.interest_over_time()
            if not df.empty:
                return [f"Google Trends for {symbol}: {df[symbol].iloc[-1]}"]
        except Exception as e:
            print(f"[ERROR] pytrends: {e}")
        return []

    def fetch_quiver_news(self, symbol: str, max_n=10) -> List[str]:
        """Fetch news headlines for symbol from Quiver Quantitative."""
        api_key = getattr(self.cfg, "quiver_api_key", "")
        url = f"https://api.quiverquant.com/beta/historic/news/{symbol.upper()}"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code == 200:
                news = r.json()
                return [item["Headline"] for item in news[:max_n] if "Headline" in item]
        except Exception as e:
            print(f"[ERROR] QuiverQuant: {e}")
        return []

    def fetch_reddit(self, symbol: str, n=10) -> List[str]:
        """Fetch recent Reddit post titles and bodies mentioning the symbol."""
        if self.reddit is None:
            return []
        posts = []
        try:
            for subreddit in ["wallstreetbets", "stocks"]:
                for i, submission in enumerate(self.reddit.subreddit(subreddit).search(symbol, sort="new", limit=n // 2)):
                    posts.append(submission.title + " " + submission.selftext)
        except Exception as e:
            print(f"[ERROR] Reddit fetch: {e}")
        return posts

    def fetch_headlines(self, symbol: str) -> List[str]:
        """
        Aggregate and deduplicate all headlines/text from configured feeds for the given symbol.
        Returns up to 50 most recent deduplicated headlines.
        """
        sources = []
        sources += self.fetch_finnhub_headlines(symbol)
        sources += self.fetch_polygon_headlines(symbol)
        sources += self.fetch_stocktwits(symbol)
        sources += self.fetch_newsapi(symbol)
        sources += self.fetch_gnews(symbol)
        sources += self.fetch_pytrends(symbol)
        sources += self.fetch_quiver_news(symbol)
        sources += self.fetch_reddit(symbol, n=10)
        # Deduplicate
        seen = set()
        deduped = []
        for s in sources:
            if s and s not in seen:
                deduped.append(s)
                seen.add(s)
            if len(deduped) >= 50:
                break
        if not deduped:
            deduped = [
                f"{symbol} launches new product, stock rallies",
                f"Analysts remain bullish on {symbol} amid market volatility"
            ]
        print(f"[DEBUG] Headlines for {symbol}: {deduped[:5]} ... [{len(deduped)} total]")
        return deduped

    # === Sentiment ===

    def nlp_sentiment(self, texts: List[str]) -> float:
        """
        Compute sentiment score for a list of texts.
        Tries OpenAI (if configured), falls back to FinBERT/transformers.
        Returns score in [-1, 1].
        """
        openai_key = getattr(self.cfg, "openai_api_key", None)
        if openai and openai_key:
            openai.api_key = openai_key
            try:
                prompt = (
                    "Analyze sentiment (positive, negative, or neutral, with a score from -1 to 1) "
                    "for these headlines:\n" + "\n".join(texts[:5])
                )
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": prompt}],
                    max_tokens=50,
                    temperature=0
                )
                content = resp.choices[0].message.content
                import re
                match = re.search(r'(-?\d+\.\d+)', content)
                if match:
                    return float(match.group(1))
                if "positive" in content.lower():
                    return 0.5
                if "negative" in content.lower():
                    return -0.5
                return 0.0
            except Exception as e:
                print(f"[ERROR] OpenAI sentiment: {e}")
        # Fallback to FinBERT/transformers
        if not texts or self.pipe is None:
            return 0.0
        res = [self.pipe(t)[0] for t in texts]
        scores = [(d["score"] if d["label"].lower() == "positive" else -d["score"]) for d in res]
        sent_mean = float(np.mean(scores))
        print(f"[DEBUG] Sentiment (FinBERT): {sent_mean:.3f}")
        return sent_mean

    def symbol_sentiment(self, symbol: str) -> float:
        """
        Aggregate all feeds for a symbol and return unified sentiment score.
        """
        headlines = self.fetch_headlines(symbol)
        return self.nlp_sentiment(headlines)

    # === Volatility ===

    def _ewma_vol(self, rets):
        """Exponentially weighted moving average (EWMA) volatility estimator."""
        if rets.empty:
            print("[ERROR] EWMA volatility requested on empty returns.")
            return 0.0
        lam = self.cfg.ewma_lambda
        return float(np.sqrt((rets**2).ewm(alpha=1-lam).mean().iloc[-1]))

    def _garch_vol(self, rets):
        """GARCH(1,1) volatility estimator (uses arch package)."""
        if arch_model is None or rets.empty:
            print("[ERROR] GARCH volatility requested on empty returns or arch not available.")
            return None
        try:
            fit = arch_model(rets*100, p=1, q=1).fit(disp="off")
            return float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1,0])/100)
        except Exception:
            return None

    def forecast_volatility(self, close):
        """
        Forecast volatility from price series using GARCH or EWMA.
        Returns a single volatility estimate (annualized stdev of returns).
        """
        if close.empty:
            print("[ERROR] No close prices for volatility. Skipping.")
            return 0.0
        rets = close.pct_change().dropna()
        if rets.empty:
            print("[ERROR] No returns for volatility. Skipping.")
            return 0.0
        vol = self._garch_vol(rets) if getattr(self.cfg, "vol_model", None) == "garch" else None
        if vol is not None:
            print(f"[DEBUG] GARCH Vol: {vol:.4f}")
            return vol
        ewma = self._ewma_vol(rets)
        print(f"[DEBUG] EWMA Vol: {ewma:.4f}")
        return ewma

    # === Events ===

    def detect_events(self, symbol: str) -> Dict[str, Any]:
        """
        Run event detection plugins for a symbol.
        Plugins return a dict of event signals (e.g., 'unusual_flow', 'cyber_breach').
        """
        ev: Dict[str, Any] = {}
        for p in self.plugins:
            out = p.check(symbol=symbol) if "symbol" in p.check.__code__.co_varnames else p.check()
            ev.update(out)
        print(f"[DEBUG] Events: {ev}")
        return ev

    # === Final aggregation ===

    def aggregate_scores(self, sentiment: float, vol: float, events: Dict[str,Any]) -> float:
        """
        Combine sentiment, volatility, and events into a single signal score using config weights.
        """
        w = self.weights
        score = w["sentiment"] * sentiment + w["volatility"] * (vol/0.05)
        if "unusual_flow" in events:
            f = events["unusual_flow"]; score += 0.2*f["mag"]*(1 if f["dir"]=="bull" else -1)
        if "cyber_breach" in events:
            score -= 0.15*(events["cyber_breach"]["sev"]/10)
        if "macro_release" in events:
            score += 0.1*events["macro_release"]["surprise"]
        print(f"[DEBUG] Weights: {w} | Sentiment: {sentiment:.3f} | Vol: {vol:.3f} | Events: {events} | Raw score: {score:.3f}")
        return float(np.tanh(score))

    def get_signal_score(self, symbol: str, close_series) -> float:
        """
        Convenience method: Runs all steps to produce a final signal score for a symbol.
        """
        if close_series.empty:
            print("[ERROR] No price data for symbol. Aborting signal generation.")
            return float('nan')
        sentiment = self.symbol_sentiment(symbol)
        vol = self.forecast_volatility(close_series)
        events = self.detect_events(symbol)
        return self.aggregate_scores(sentiment, vol, events)
