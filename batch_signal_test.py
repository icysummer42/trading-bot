from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import datetime as dt
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from signal_generator import SignalGenerator
from config import Config
from data_pipeline import DataPipeline
from logger import get_logger

logger = get_logger("batchrun")

SYMBOLS = ["AAPL", "TSLA", "NVDA", "SPY", "QQQ", "XLK", "XLE"]  # Your watchlist

def score_one(symbol, cfg, dp):
    row = {
        "symbol": symbol,
        "score": "",
        "status": "OK",
        "sentiment_source": "",
        "error_message": "",
        "n_headlines": 0,
        "date": str(dt.date.today())
    }
    try:
        close_series = dp.get_close_series(symbol, start="2024-06-01", end=str(dt.date.today()))
        if close_series is None or close_series.empty:
            row["status"] = "FAIL"
            row["error_message"] = "No price data"
            logger.error(f"No price data for {symbol}, skipping.")
            return row
        sg = SignalGenerator(cfg, dp)
        # Make sure symbol_sentiment returns (score, source)
        out = sg.symbol_sentiment(symbol)
        if isinstance(out, tuple):
            sent_score, sent_source = out
        else:
            sent_score, sent_source = out, ""
        vol = sg.forecast_volatility(close_series)
        events = sg.detect_events(symbol)
        score = sg.aggregate_scores(sent_score, vol, events)
        headlines = sg.fetch_headlines(symbol)
        row.update({
            "score": round(score, 3),
            "sentiment_source": sent_source,
            "n_headlines": len(headlines)
        })
        logger.info(f"[DONE] {symbol}: score={score:.3f} (sentiment: {sent_source})")
    except Exception as e:
        row["status"] = "FAIL"
        row["error_message"] = str(e)
        logger.exception(f"Exception scoring {symbol}: {e}")
    return row

def main():
    # --- Load config ---
    cfg = Config()
    cfg.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
    cfg.polygon_key = os.getenv("POLYGON_KEY")
    cfg.openai_api_key = os.getenv("OPENAI_API_KEY")
    cfg.newsapi_key = os.getenv("NEWSAPI_KEY")
    cfg.gnews_key = os.getenv("GNEWS_KEY")
    cfg.quiver_api_key = os.getenv("QUIVER_API_KEY")
    cfg.stocktwits_token = os.getenv("STOCKTWITS_TOKEN")
    cfg.nlp_model_name = os.getenv("NLP_MODEL_NAME", "ProsusAI/finbert")
    cfg.ewma_lambda = float(os.getenv("EWMA_LAMBDA", 0.94))
    cfg.vol_model = os.getenv("VOL_MODEL", "ewma")
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
    logger.debug(f"POLYGON_KEY: {cfg.polygon_key!r}")

    dp = DataPipeline(cfg)
    logger.info(f"Batch scoring {len(SYMBOLS)} symbols...")

    results = []
    with ThreadPoolExecutor(max_workers=min(len(SYMBOLS), 4)) as executor:
        futures = {executor.submit(score_one, symbol, cfg, dp): symbol for symbol in SYMBOLS}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)  # <--- THIS WAS MISSING

    # Write CSV output
    fieldnames = ["symbol", "score", "status", "sentiment_source", "n_headlines", "error_message", "date"]
    with open("batch_signal_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logger.info(f"Batch scoring complete. Results saved to batch_signal_scores.csv")

if __name__ == "__main__":
    main()
