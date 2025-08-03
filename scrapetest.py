import certifi, os
os.environ["SSL_CERT_FILE"] = certifi.where()

try:
    import snscrape.modules.twitter as sntwitter
except ImportError:
    print("[FAIL] snscrape not installed")
    exit()

query = "TSLA"
try:
    tweets = [tweet.content for tweet in sntwitter.TwitterSearchScraper(query).get_items()]
    print(f"[OK] Fetched {len(tweets)} tweets. First tweet: {tweets[0] if tweets else 'NONE'}")
except Exception as e:
    print("[ERROR] snscrape failed:", e)
