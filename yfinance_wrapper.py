
def get_fixed_yfinance_data(symbol, period="1mo", session=None):
    """
    Fixed yfinance data fetcher with error handling and retries.
    """
    import yfinance as yf
    import requests
    import time
    from datetime import datetime, timedelta
    
    if session is None:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
    
    # Try multiple approaches
    approaches = [
        lambda: yf.Ticker(symbol, session=session).history(period=period, auto_adjust=True),
        lambda: yf.download(symbol, period=period, progress=False, session=session),
        lambda: yf.Ticker(symbol).history(period=period, interval="1d"),
    ]
    
    for i, approach in enumerate(approaches):
        try:
            data = approach()
            if not data.empty:
                print(f"yfinance approach {i+1} succeeded for {symbol}")
                return data
        except Exception as e:
            print(f"yfinance approach {i+1} failed: {e}")
            time.sleep(1)  # Brief delay between attempts
    
    print(f"All yfinance approaches failed for {symbol}")
    return None
