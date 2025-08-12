#!/usr/bin/env python3
"""
Simple test for the unified data pipeline with proper environment loading
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value
                print(f"Loaded {key}")

def test_pipeline():
    """Test the unified data pipeline"""
    load_env()
    
    # Now import after environment is loaded
    from data_pipeline_unified import UnifiedDataPipeline
    from config import Config
    
    print(f"\n🔧 Environment Check:")
    print(f"POLYGON_KEY: {'✅' if os.getenv('POLYGON_KEY') else '❌'}")
    print(f"FINNHUB_API_KEY: {'✅' if os.getenv('FINNHUB_API_KEY') else '❌'}")
    
    # Initialize pipeline
    config = Config()
    pipeline = UnifiedDataPipeline(config)
    
    print(f"\n🚀 Testing Unified Data Pipeline...")
    
    # Test 1: Health Check
    print(f"\n📊 Health Check:")
    health = pipeline.health_check()
    print(f"Overall Status: {health['overall_status']}")
    for source, status in health['sources'].items():
        print(f"  {source}: {status['status']}")
    
    # Test 2: Close Series
    print(f"\n📈 Testing Close Series (AAPL):")
    try:
        close_data = pipeline.get_close_series("AAPL", start="2024-08-01", end="2024-08-10")
        if not close_data.empty:
            print(f"✅ Success: {len(close_data)} data points")
            print(f"   Latest price: ${close_data.iloc[-1]:.2f}")
            print(f"   Date range: {close_data.index[0]} to {close_data.index[-1]}")
        else:
            print(f"❌ No data returned")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Equity Prices (OHLCV)
    print(f"\n📊 Testing Equity Prices (AAPL):")
    try:
        equity_data = pipeline.fetch_equity_prices("AAPL")
        if not equity_data.empty:
            print(f"✅ Success: {len(equity_data)} days of data")
            print(f"   Columns: {list(equity_data.columns)}")
            print(f"   Latest close: ${equity_data['close'].iloc[-1]:.2f}")
        else:
            print(f"❌ No data returned")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Options Chain
    print(f"\n🎯 Testing Options Chain (AAPL):")
    try:
        options_data = pipeline.fetch_options_chain("AAPL")
        if not options_data.empty:
            calls = len(options_data[options_data['type'] == 'call'])
            puts = len(options_data[options_data['type'] == 'put'])
            print(f"✅ Success: {calls} calls, {puts} puts")
            print(f"   Strike range: ${options_data['strike'].min():.2f} - ${options_data['strike'].max():.2f}")
        else:
            print(f"❌ No data returned")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Cache Performance
    print(f"\n⚡ Testing Cache Performance:")
    import time
    
    # First fetch
    start = time.time()
    data1 = pipeline.get_close_series("MSFT", start="2024-08-01", end="2024-08-10")
    time1 = time.time() - start
    
    # Second fetch (cached)
    start = time.time()
    data2 = pipeline.get_close_series("MSFT", start="2024-08-01", end="2024-08-10")
    time2 = time.time() - start
    
    print(f"   First fetch: {time1:.3f}s")
    print(f"   Second fetch: {time2:.3f}s")
    if time2 < time1 * 0.5:
        print(f"✅ Cache working: {time1/time2:.1f}x speedup")
    else:
        print(f"⚠️  Cache may not be optimal")
    
    print(f"\n🎉 Pipeline test completed!")

if __name__ == "__main__":
    test_pipeline()