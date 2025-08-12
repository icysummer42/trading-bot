#!/usr/bin/env python3
"""
yfinance API Diagnostic Tool

Diagnoses and fixes yfinance API connection issues to replace mock data
with real market data in the rate-limited pipeline.
"""

import yfinance as yf
import requests
import json
import time
import sys
from datetime import datetime, timedelta

def test_basic_connection():
    """Test basic internet connectivity and yfinance endpoints."""
    print("🔍 Testing Basic Connectivity")
    print("-" * 40)
    
    # Test basic internet
    try:
        response = requests.get("https://httpbin.org/get", timeout=10)
        print(f"✅ Internet connection: {response.status_code}")
    except Exception as e:
        print(f"❌ Internet connection failed: {e}")
        return False
    
    # Test Yahoo Finance directly
    try:
        response = requests.get("https://finance.yahoo.com", timeout=10)
        print(f"✅ Yahoo Finance website: {response.status_code}")
    except Exception as e:
        print(f"❌ Yahoo Finance website failed: {e}")
        return False
    
    return True

def test_yfinance_methods():
    """Test different yfinance methods and configurations."""
    print("\n📊 Testing yfinance Methods")
    print("-" * 40)
    
    symbol = "AAPL"
    
    # Method 1: Basic ticker history
    print("1. Basic ticker.history():")
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="5d")
        if not data.empty:
            print(f"   ✅ Got {len(data)} days of data")
            print(f"   Latest close: ${data['Close'].iloc[-1]:.2f}")
        else:
            print("   ⚠️  Empty data returned")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 2: With different parameters
    print("2. With session and different params:")
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        ticker = yf.Ticker(symbol, session=session)
        data = ticker.history(period="1mo", interval="1d", auto_adjust=True, prepost=True)
        if not data.empty:
            print(f"   ✅ Got {len(data)} days of data")
            print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        else:
            print("   ⚠️  Empty data returned")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 3: Using download function
    print("3. Using yf.download():")
    try:
        data = yf.download(symbol, period="5d", progress=False)
        if not data.empty:
            print(f"   ✅ Got {len(data)} days of data")
        else:
            print("   ⚠️  Empty data returned")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 4: Multiple symbols
    print("4. Multiple symbols:")
    try:
        symbols = ["AAPL", "MSFT", "GOOGL"]
        data = yf.download(symbols, period="5d", progress=False)
        if not data.empty:
            print(f"   ✅ Got data for {len(symbols)} symbols")
            print(f"   Shape: {data.shape}")
        else:
            print("   ⚠️  Empty data returned")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_proxy_and_headers():
    """Test different proxy settings and headers."""
    print("\n🌐 Testing Proxy and Headers")
    print("-" * 40)
    
    symbol = "AAPL"
    
    # Different user agents
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]
    
    for i, ua in enumerate(user_agents):
        print(f"{i+1}. User Agent: {ua[:50]}...")
        try:
            session = requests.Session()
            session.headers.update({'User-Agent': ua})
            
            ticker = yf.Ticker(symbol, session=session)
            data = ticker.history(period="5d")
            
            if not data.empty:
                print(f"   ✅ Success with {len(data)} days")
                return session  # Return working session
            else:
                print("   ⚠️  Empty data")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return None

def test_rate_limiting():
    """Test if rate limiting is the issue."""
    print("\n⏱️  Testing Rate Limiting")
    print("-" * 40)
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    for symbol in symbols:
        print(f"Fetching {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d")
            if not data.empty:
                print(f"   ✅ {symbol}: {len(data)} days")
            else:
                print(f"   ⚠️  {symbol}: Empty data")
            
            # Small delay between requests
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")

def test_alternative_endpoints():
    """Test alternative data sources if yfinance fails."""
    print("\n🔄 Testing Alternative Sources")
    print("-" * 40)
    
    symbol = "AAPL"
    
    # Test Alpha Vantage (if API key available)
    print("1. Alpha Vantage:")
    try:
        # This would require API key from configuration
        print("   ⚠️  Requires API key from configuration")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test Polygon (if API key available)
    print("2. Polygon.io:")
    try:
        # This would require API key from configuration
        print("   ⚠️  Requires API key from configuration")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def fix_yfinance_config():
    """Apply fixes for common yfinance issues."""
    print("\n🔧 Applying yfinance Fixes")
    print("-" * 40)
    
    fixes_applied = []
    
    # Fix 1: Set proper session with headers
    try:
        import yfinance as yf
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Test with session
        ticker = yf.Ticker("AAPL", session=session)
        data = ticker.history(period="5d")
        
        if not data.empty:
            fixes_applied.append("Custom session with headers")
            print("   ✅ Custom session fix successful")
            return session
        else:
            print("   ⚠️  Custom session didn't help")
            
    except Exception as e:
        print(f"   ❌ Session fix error: {e}")
    
    # Fix 2: Try different time periods
    for period in ["5d", "1mo", "3mo"]:
        try:
            ticker = yf.Ticker("AAPL")
            data = ticker.history(period=period)
            if not data.empty:
                fixes_applied.append(f"Period {period} works")
                print(f"   ✅ Period {period} successful")
                break
        except Exception as e:
            print(f"   ⚠️  Period {period} failed: {e}")
    
    return None

def create_fixed_yfinance_wrapper():
    """Create a wrapper function with fixes applied."""
    print("\n🛠️  Creating Fixed yfinance Wrapper")
    print("-" * 40)
    
    wrapper_code = '''
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
'''
    
    # Save the wrapper to a file
    with open('/home/quantbot/project/yfinance_wrapper.py', 'w') as f:
        f.write(wrapper_code)
    
    print("✅ Created yfinance_wrapper.py with fixed implementation")
    
    return wrapper_code

def run_full_diagnostic():
    """Run complete diagnostic suite."""
    print("🚀 yfinance API Diagnostic Suite")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Test basic connectivity
    if not test_basic_connection():
        print("\n❌ Basic connectivity failed - check internet connection")
        return False
    
    # Test yfinance methods
    test_yfinance_methods()
    
    # Test headers and user agents
    working_session = test_proxy_and_headers()
    
    # Test rate limiting
    test_rate_limiting()
    
    # Test alternatives
    test_alternative_endpoints()
    
    # Apply fixes
    fixed_session = fix_yfinance_config()
    
    # Create wrapper
    create_fixed_yfinance_wrapper()
    
    # Final test with best approach
    print("\n🎯 Final Test with Best Approach")
    print("-" * 40)
    
    session = working_session or fixed_session
    if session:
        try:
            ticker = yf.Ticker("AAPL", session=session)
            data = ticker.history(period="1mo")
            if not data.empty:
                print(f"✅ Final test successful: {len(data)} days of AAPL data")
                print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
                print(f"   Latest close: ${data['Close'].iloc[-1]:.2f}")
                return True
            else:
                print("⚠️  Final test returned empty data")
        except Exception as e:
            print(f"❌ Final test error: {e}")
    
    print("\n📋 Diagnostic Summary")
    print("-" * 40)
    print("• Check yfinance_wrapper.py for improved implementation")
    print("• Consider using alternative data sources")
    print("• May need to implement retry logic with delays")
    
    return False

if __name__ == "__main__":
    success = run_full_diagnostic()
    sys.exit(0 if success else 1)