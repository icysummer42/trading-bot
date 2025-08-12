from dotenv import load_dotenv
import certifi
import os

# --- Load .env and set SSL certificate bundle early ---
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

# --- Debug: Print all API keys (truncated for security if desired) ---
def _trunc(key):
    return (key[:6] + "..." if key else "")

print("[DEBUG] POLYGON_KEY:", _trunc(os.getenv("POLYGON_KEY")))
print("[DEBUG] FINNHUB_API_KEY:", _trunc(os.getenv("FINNHUB_API_KEY")))
print("[DEBUG] REDDIT_CLIENT_ID:", _trunc(os.getenv("REDDIT_CLIENT_ID")))
print("[DEBUG] STOCKTWITS_TOKEN:", _trunc(os.getenv("STOCKTWITS_TOKEN")))
print("[DEBUG] NEWSAPI_KEY:", _trunc(os.getenv("NEWSAPI_KEY")))
print("[DEBUG] GNEWS_KEY:", _trunc(os.getenv("GNEWS_KEY")))
print("[DEBUG] QUIVER_API_KEY:", _trunc(os.getenv("QUIVER_API_KEY")))
print("[DEBUG] OPENAI_API_KEY:", _trunc(os.getenv("OPENAI_API_KEY")))

import pandas as pd
import datetime as dt

today = dt.date.today()
yesterday = today - dt.timedelta(days=1)

from signal_generator import SignalGenerator
from config import Config
from data_pipeline import DataPipeline

# --- Setup config with all necessary API keys and params ---
cfg = Config()
cfg.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
cfg.polygon_key = os.getenv("POLYGON_KEY")
cfg.newsapi_key = os.getenv("NEWSAPI_KEY")
cfg.gnews_key = os.getenv("GNEWS_KEY")
cfg.stocktwits_token = os.getenv("STOCKTWITS_TOKEN")
cfg.quiver_api_key = os.getenv("QUIVER_API_KEY")
cfg.openai_api_key = os.getenv("OPENAI_API_KEY")
cfg.nlp_model_name = os.getenv("NLP_MODEL_NAME", "ProsusAI/finbert")
cfg.ewma_lambda = float(os.getenv("EWMA_LAMBDA", 0.94))
cfg.vol_model = os.getenv("VOL_MODEL", "ewma")  # or "garch"
cfg.signal_weights = {
    "sentiment": float(os.getenv("SIGNAL_SENTIMENT_WEIGHT", 0.5)),
    "volatility": float(os.getenv("SIGNAL_VOLATILITY_WEIGHT", 0.3)),
    "events": float(os.getenv("SIGNAL_EVENTS_WEIGHT", 0.2)),
}
cfg.reddit = {
    "client_id": os.getenv("REDDIT_CLIENT_ID"),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    "user_agent": os.getenv("REDDIT_USER_AGENT", "sentiment-bot"),
}

# --- Initialize pipeline and generator ---
dp = DataPipeline(cfg)
symbol = "msft"  # or any symbol you want to test

# --- Get historical close prices for symbol ---
try:
    close_series = dp.get_close_series(symbol, start="2024-06-01", end=str(yesterday))
    print(f"Close prices head:\n{close_series.head()}")
except Exception as e:
    print("[ERROR] Failed to fetch close prices:", e)
    close_series = pd.Series([150, 152, 151, 153, 152])  # fallback

if close_series is None or getattr(close_series, "empty", False):
    print("[ERROR] No price data for symbol after fetching. Aborting.")
    exit(1)

# --- Run the signal generator ---
sg = SignalGenerator(cfg, dp)
score = sg.get_signal_score(symbol, close_series)
print(f"\n=== Final Signal Score for {symbol}: {score:.3f} ===")
print("Close prices head:\n", close_series.head())
print("Close prices len:", len(close_series))
