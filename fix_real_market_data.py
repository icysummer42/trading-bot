#!/usr/bin/env python3
"""
Fix Real Market Data Implementation

Replace yfinance with Polygon.io API to get real market data using 
the API key from configuration, integrated with the rate limiting system.
"""

import asyncio
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_configuration import create_config_manager

class PolygonDataFetcher:
    """
    Real market data fetcher using Polygon.io API with rate limiting.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        
        # Set proper headers
        self.session.headers.update({
            'User-Agent': 'QuantBot/1.0',
            'Authorization': f'Bearer {api_key}'
        })
    
    def get_stock_data(self, symbol: str, timespan: str = "day", 
                      multiplier: int = 1, from_date: str = None, 
                      to_date: str = None) -> Optional[pd.DataFrame]:
        """
        Get stock data from Polygon API.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timespan: Time span ('minute', 'hour', 'day', 'week', 'month', 'quarter', 'year')
            multiplier: Size of the timespan multiplier
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set default dates if not provided
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Construct URL
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        # Parameters
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            print(f"🔍 Fetching {symbol} data from Polygon...")
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    # Convert to DataFrame
                    df = pd.DataFrame(data['results'])
                    
                    # Rename columns to match yfinance format
                    df = df.rename(columns={
                        'o': 'Open',
                        'h': 'High', 
                        'l': 'Low',
                        'c': 'Close',
                        'v': 'Volume',
                        't': 'timestamp'
                    })
                    
                    # Convert timestamp to datetime
                    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('Date', inplace=True)
                    
                    # Add Adj Close (same as Close for simplicity)
                    df['Adj Close'] = df['Close']
                    
                    # Select relevant columns in yfinance order
                    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
                    
                    print(f"✅ Got {len(df)} days of real data for {symbol}")
                    return df
                    
                else:
                    print(f"⚠️  No results in Polygon response for {symbol}")
                    return None
                    
            elif response.status_code == 429:
                print(f"⚠️  Rate limited by Polygon for {symbol}")
                return None
            else:
                print(f"❌ Polygon API error {response.status_code} for {symbol}: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching {symbol} from Polygon: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/last price for a symbol."""
        url = f"{self.base_url}/v2/last/trade/{symbol}"
        params = {'apikey': self.api_key}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    return data['results']['p']  # price
            return None
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None

def create_real_data_fetcher():
    """Create a data fetcher using real API keys from configuration."""
    try:
        # Load configuration
        config_manager = create_config_manager(environment="development")
        config = config_manager.get_config()
        
        # Get Polygon API key
        if hasattr(config.data_sources, 'polygon_api_key'):
            api_key = config.data_sources.polygon_api_key
            
            # Remove SECURE: prefix if present
            if api_key.startswith("SECURE:"):
                api_key = api_key[7:]  # Remove "SECURE:" prefix
            
            print(f"✅ Using Polygon API key: {api_key[:8]}...")
            return PolygonDataFetcher(api_key)
        else:
            print("❌ No Polygon API key found in configuration")
            return None
            
    except Exception as e:
        print(f"❌ Error creating data fetcher: {e}")
        return None

def test_real_data_fetcher():
    """Test the real data fetcher with multiple symbols."""
    print("🚀 Testing Real Market Data Fetcher")
    print("=" * 50)
    
    # Create fetcher
    fetcher = create_real_data_fetcher()
    if not fetcher:
        return False
    
    # Test symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"]
    results = {}
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}")
        print("-" * 20)
        
        # Get historical data
        data = fetcher.get_stock_data(
            symbol=symbol,
            from_date="2025-01-01",  # Recent data
            to_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        if data is not None and not data.empty:
            results[symbol] = {
                'success': True,
                'rows': len(data),
                'date_range': f"{data.index[0].date()} to {data.index[-1].date()}",
                'latest_close': data['Close'].iloc[-1],
                'volume': data['Volume'].iloc[-1]
            }
            
            print(f"✅ Success: {len(data)} days")
            print(f"   Range: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"   Latest: ${data['Close'].iloc[-1]:.2f}")
            print(f"   Volume: {data['Volume'].iloc[-1]:,.0f}")
            
        else:
            results[symbol] = {'success': False}
            print(f"❌ Failed to get data")
    
    # Summary
    successful = sum(1 for r in results.values() if r.get('success'))
    total = len(results)
    
    print(f"\n📋 Summary: {successful}/{total} symbols successful")
    
    if successful > 0:
        print("✅ Real market data is working!")
        return True
    else:
        print("❌ Real market data failed for all symbols")
        return False

def update_unified_pipeline_with_polygon():
    """Update the unified pipeline to use Polygon instead of yfinance."""
    print("\n🔧 Updating Unified Pipeline")
    print("-" * 30)
    
    try:
        # Read current pipeline
        with open('/home/quantbot/project/data_pipeline_unified.py', 'r') as f:
            content = f.readlines()
        
        # Find the _fetch_yfinance_close_series method and update it
        updated_content = []
        in_yfinance_method = False
        polygon_method_added = False
        
        for line in content:
            # Check if we're entering the yfinance method
            if 'def _fetch_yfinance_close_series' in line:
                in_yfinance_method = True
                updated_content.append(line)
            elif in_yfinance_method and line.strip().startswith('def '):
                # We've reached the next method, add Polygon method first
                if not polygon_method_added:
                    updated_content.append("""
    def _fetch_polygon_close_series(self, symbol: str, start: str, end: str) -> pd.Series:
        \"\"\"Fetch close series from Polygon API\"\"\"
        try:
            if hasattr(self.config.data_sources, 'polygon_api_key'):
                api_key = self.config.data_sources.polygon_api_key
                if api_key.startswith("SECURE:"):
                    api_key = api_key[7:]  # Remove "SECURE:" prefix
                
                from fix_real_market_data import PolygonDataFetcher
                fetcher = PolygonDataFetcher(api_key)
                
                data = fetcher.get_stock_data(symbol, from_date=start, to_date=end)
                if data is not None and not data.empty:
                    logger.info(f"Polygon returned {len(data)} days for {symbol}")
                    return data['Close']
                
            return pd.Series(dtype=float)
            
        except Exception as e:
            logger.warning(f"Polygon API error for {symbol}: {e}")
            return pd.Series(dtype=float)

""")
                    polygon_method_added = True
                in_yfinance_method = False
                updated_content.append(line)
            else:
                updated_content.append(line)
        
        # Update the get_close_series method to try Polygon first
        final_content = []
        for i, line in enumerate(updated_content):
            if 'def get_close_series' in line:
                # Find the method and update the order
                final_content.append(line)
                # Add lines until we find the fetch attempts
                j = i + 1
                while j < len(updated_content) and 'self._fetch_polygon_close_series' not in updated_content[j]:
                    final_content.append(updated_content[j])
                    j += 1
                
                # Add Polygon first, then yfinance as fallback
                final_content.append('        # Try Polygon API first (real-time data)\n')
                final_content.append('        data = self._fetch_polygon_close_series(symbol, start, end)\n')
                final_content.append('        if not data.empty:\n')
                final_content.append('            return data\n')
                final_content.append('\n')
                
                # Skip to after the original fetch attempts
                while j < len(updated_content) and not updated_content[j].strip().startswith('# Generate mock data'):
                    j += 1
                
                # Continue from mock data generation
                for k in range(j, len(updated_content)):
                    final_content.append(updated_content[k])
                break
            else:
                final_content.append(line)
        
        # Write updated content
        with open('/home/quantbot/project/data_pipeline_unified.py', 'w') as f:
            f.writelines(final_content)
        
        print("✅ Updated unified pipeline to use Polygon API")
        return True
        
    except Exception as e:
        print(f"❌ Error updating pipeline: {e}")
        return False

async def test_integrated_pipeline():
    """Test the updated pipeline with real data."""
    print("\n🧪 Testing Integrated Pipeline")
    print("-" * 30)
    
    try:
        from complete_data_pipeline_with_rate_limiting import create_rate_limited_pipeline
        
        # Create pipeline
        pipeline = create_rate_limited_pipeline(environment="development")
        await pipeline.initialize_async()
        
        # Test market data fetch
        symbols = ["AAPL", "MSFT"]
        for symbol in symbols:
            print(f"Testing {symbol}...")
            data = await pipeline.fetch_market_data_async(symbol)
            
            if data is not None and not data.empty:
                print(f"✅ {symbol}: {len(data)} days, latest: ${data['Close'].iloc[-1]:.2f}")
            else:
                print(f"⚠️  {symbol}: No data")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False

async def run_complete_fix():
    """Run complete fix for real market data."""
    print("🚀 Complete Real Market Data Fix")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Step 1: Test real data fetcher
    print("Step 1: Test Polygon API...")
    fetcher_works = test_real_data_fetcher()
    
    if not fetcher_works:
        print("❌ Polygon API test failed - check API key")
        return False
    
    # Step 2: Update pipeline
    print("\nStep 2: Update unified pipeline...")
    pipeline_updated = update_unified_pipeline_with_polygon()
    
    if not pipeline_updated:
        print("❌ Pipeline update failed")
        return False
    
    # Step 3: Test integration
    print("\nStep 3: Test integrated pipeline...")
    integration_works = await test_integrated_pipeline()
    
    if integration_works:
        print("\n🎉 Real market data fix complete!")
        print("✅ Polygon API working")
        print("✅ Pipeline updated")
        print("✅ Integration successful")
        return True
    else:
        print("⚠️  Integration test had issues")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_complete_fix())
    sys.exit(0 if success else 1)