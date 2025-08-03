import os
import pandas as pd
from signal_generator import SignalGenerator
from config import Config
from data_pipeline import DataPipeline  # <--- use your real pipeline

# Setup config with env variables
cfg = Config()
cfg.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
cfg.polygon_key = os.getenv("POLYGON_API_KEY")
cfg.nlp_model_name = "ProsusAI/finbert"  # or your preferred model

# Add any other config needed by your plugins here

# Initialize your real data pipeline
dp = DataPipeline(cfg)
symbol = "AAPL"

# Always try to load close price series from your pipeline
try:
    close_series = dp.get_close_series(symbol)  # <<--- (fix variable name)
except Exception:
    # fallback to CSV if not implemented yet
    #df = pd.read_csv("AAPL_close.csv")
    #close_series = df["close"]
    close_series = pd.Series([150, 152, 151, 153, 152])  # Fallback dummy series

sg = SignalGenerator(cfg, dp)
print("Signal score:", sg.get_signal_score(symbol, close_series))
print("Close prices head:", close_series.head())
