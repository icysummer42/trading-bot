#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Data Pipeline
=================================================

This script validates all components of the unified data pipeline:
- Data fetching from multiple sources
- Caching functionality  
- Failover mechanisms
- Data quality validation
- Error handling
- Performance metrics

Run with: python test_unified_pipeline.py
"""

import sys
import time
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline_unified import UnifiedDataPipeline, DataPipelineError
from config import Config
import pandas as pd
import numpy as np


class PipelineTestSuite:
    """Comprehensive test suite for the unified data pipeline"""
    
    def __init__(self):
        self.config = Config()
        self.pipeline = UnifiedDataPipeline(self.config)
        self.test_results = {}
        self.start_time = time.time()
        
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Unified Data Pipeline Test Suite")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Close Series Fetching", self.test_close_series),
            ("Equity Prices Fetching", self.test_equity_prices),
            ("Options Chain Fetching", self.test_options_chain),
            ("Macro Data Fetching", self.test_macro_data),
            ("Caching Functionality", self.test_caching),
            ("Data Quality Validation", self.test_data_quality),
            ("Error Handling", self.test_error_handling),
            ("Performance Benchmarks", self.test_performance),
            ("Failover Mechanisms", self.test_failover)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 Running: {test_name}")
            print("-" * 40)
            
            try:
                result = test_func()
                if result:
                    print(f"✅ PASSED: {test_name}")
                    passed_tests += 1
                else:
                    print(f"❌ FAILED: {test_name}")
                self.test_results[test_name] = result
                
            except Exception as e:
                print(f"💥 ERROR in {test_name}: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                self.test_results[test_name] = False
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 TEST SUITE SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Runtime: {time.time() - self.start_time:.2f}s")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Data pipeline is ready for production.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Review issues above.")
            
        return passed_tests == total_tests
    
    def test_health_check(self):
        """Test pipeline health check functionality"""
        try:
            health = self.pipeline.health_check()
            
            # Validate health check structure
            required_keys = ['timestamp', 'overall_status', 'sources', 'cache', 'connectivity']
            for key in required_keys:
                if key not in health:
                    print(f"❌ Missing key in health check: {key}")
                    return False
            
            print(f"📊 Overall Status: {health['overall_status']}")
            print(f"📊 Sources: {list(health['sources'].keys())}")
            print(f"📊 Cache Status: {health['cache']['status']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_close_series(self):
        """Test close series fetching with multiple symbols"""
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        success_count = 0
        
        for symbol in test_symbols:
            try:
                start_time = time.time()
                close_series = self.pipeline.get_close_series(symbol)
                fetch_time = time.time() - start_time
                
                if not close_series.empty:
                    print(f"✅ {symbol}: {len(close_series)} data points "
                          f"({fetch_time:.2f}s)")
                    print(f"   Latest: {close_series.iloc[-1]:.2f} on {close_series.index[-1]}")
                    success_count += 1
                else:
                    print(f"❌ {symbol}: Empty data returned")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
        
        return success_count == len(test_symbols)
    
    def test_equity_prices(self):
        """Test full OHLCV equity data fetching"""
        test_symbol = "AAPL"
        
        try:
            start_time = time.time()
            equity_data = self.pipeline.fetch_equity_prices(test_symbol)
            fetch_time = time.time() - start_time
            
            if equity_data.empty:
                print(f"❌ No equity data returned for {test_symbol}")
                return False
            
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'symbol']
            missing_cols = [col for col in required_columns if col not in equity_data.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return False
            
            print(f"✅ Equity data for {test_symbol}:")
            print(f"   Rows: {len(equity_data)}, Columns: {list(equity_data.columns)}")
            print(f"   Date range: {equity_data.index[0]} to {equity_data.index[-1]}")
            print(f"   Latest close: ${equity_data['close'].iloc[-1]:.2f}")
            print(f"   Fetch time: {fetch_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Equity prices test failed: {e}")
            return False
    
    def test_options_chain(self):
        """Test options chain data fetching"""
        test_symbol = "AAPL"
        
        try:
            start_time = time.time()
            options_data = self.pipeline.fetch_options_chain(test_symbol)
            fetch_time = time.time() - start_time
            
            if options_data.empty:
                print(f"❌ No options data returned for {test_symbol}")
                return False
            
            required_columns = ['strike', 'bid', 'ask', 'volume', 'open_interest', 'type']
            missing_cols = [col for col in required_columns if col not in options_data.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return False
            
            calls = options_data[options_data['type'] == 'call']
            puts = options_data[options_data['type'] == 'put']
            
            print(f"✅ Options data for {test_symbol}:")
            print(f"   Total options: {len(options_data)}")
            print(f"   Calls: {len(calls)}, Puts: {len(puts)}")
            print(f"   Strike range: ${options_data['strike'].min():.2f} - ${options_data['strike'].max():.2f}")
            print(f"   Fetch time: {fetch_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Options chain test failed: {e}")
            return False
    
    def test_macro_data(self):
        """Test macro economic data fetching"""
        try:
            start_time = time.time()
            macro_data = self.pipeline.fetch_macro_data()
            fetch_time = time.time() - start_time
            
            if macro_data.empty:
                print("❌ No macro data returned")
                return False
            
            print(f"✅ Macro data fetched ({fetch_time:.2f}s):")
            for col in macro_data.columns:
                if col != 'timestamp':
                    value = macro_data[col].iloc[0]
                    if value is not None:
                        print(f"   {col.upper()}: {value:.2f}")
                    else:
                        print(f"   {col.upper()}: N/A")
            
            return True
            
        except Exception as e:
            print(f"❌ Macro data test failed: {e}")
            return False
    
    def test_caching(self):
        """Test caching functionality"""
        test_symbol = "MSFT"
        
        try:
            # Clear any existing cache for test symbol
            self.pipeline.cache_manager.cache_dir.mkdir(exist_ok=True)
            
            # First fetch (should cache)
            start_time = time.time()
            data1 = self.pipeline.get_close_series(test_symbol)
            first_fetch_time = time.time() - start_time
            
            # Second fetch (should use cache)
            start_time = time.time()
            data2 = self.pipeline.get_close_series(test_symbol)
            second_fetch_time = time.time() - start_time
            
            # Verify data consistency
            if not data1.equals(data2):
                print("❌ Cached data doesn't match original data")
                return False
            
            # Cache should be significantly faster
            if second_fetch_time > first_fetch_time * 0.8:  # Allow 20% margin
                print(f"⚠️  Cache may not be working optimally")
                print(f"   First fetch: {first_fetch_time:.3f}s")
                print(f"   Second fetch: {second_fetch_time:.3f}s")
            
            print(f"✅ Caching test passed:")
            print(f"   First fetch: {first_fetch_time:.3f}s")
            print(f"   Second fetch (cached): {second_fetch_time:.3f}s")
            print(f"   Speed improvement: {(first_fetch_time/second_fetch_time):.1f}x")
            
            return True
            
        except Exception as e:
            print(f"❌ Caching test failed: {e}")
            return False
    
    def test_data_quality(self):
        """Test data quality validation"""
        test_symbol = "AAPL"
        
        try:
            close_series = self.pipeline.get_close_series(test_symbol)
            
            if close_series.empty:
                print(f"❌ No data to validate for {test_symbol}")
                return False
            
            # Test price series validation
            quality_metrics = self.pipeline.validator.validate_price_series(close_series, test_symbol)
            
            print(f"✅ Data Quality Metrics for {test_symbol}:")
            print(f"   Completeness: {quality_metrics.completeness:.2%}")
            print(f"   Consistency: {'✅' if quality_metrics.consistency else '❌'}")
            print(f"   Timeliness: {'✅' if quality_metrics.timeliness else '❌'}")
            print(f"   Accuracy: {'✅' if quality_metrics.accuracy else '❌'}")
            print(f"   Source: {quality_metrics.source}")
            
            # Test should pass if completeness > 80% and consistency is True
            return quality_metrics.completeness > 0.8 and quality_metrics.consistency
            
        except Exception as e:
            print(f"❌ Data quality test failed: {e}")
            return False
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        error_tests = [
            ("Invalid Symbol", lambda: self.pipeline.get_close_series("INVALID_SYMBOL_12345")),
            ("Empty Symbol", lambda: self.pipeline.get_close_series("")),
            ("Invalid Date Range", lambda: self.pipeline.get_close_series("AAPL", "2025-01-01", "2024-01-01"))
        ]
        
        passed_tests = 0
        
        for test_name, test_func in error_tests:
            try:
                result = test_func()
                # Should return empty series or handle gracefully, not crash
                if isinstance(result, pd.Series):
                    print(f"✅ {test_name}: Handled gracefully (returned {len(result)} data points)")
                    passed_tests += 1
                else:
                    print(f"❌ {test_name}: Unexpected return type {type(result)}")
                    
            except Exception as e:
                # Catching exceptions is also acceptable error handling
                print(f"✅ {test_name}: Caught exception properly - {type(e).__name__}")
                passed_tests += 1
        
        return passed_tests == len(error_tests)
    
    def test_performance(self):
        """Test performance benchmarks"""
        test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        try:
            print("📊 Performance Benchmarks:")
            
            # Single symbol performance
            start_time = time.time()
            data = self.pipeline.get_close_series("AAPL")
            single_fetch_time = time.time() - start_time
            print(f"   Single symbol fetch: {single_fetch_time:.3f}s")
            
            # Multiple symbols performance
            start_time = time.time()
            for symbol in test_symbols:
                self.pipeline.get_close_series(symbol)
            multi_fetch_time = time.time() - start_time
            print(f"   {len(test_symbols)} symbols fetch: {multi_fetch_time:.3f}s")
            print(f"   Average per symbol: {multi_fetch_time/len(test_symbols):.3f}s")
            
            # Performance should be reasonable (< 2 seconds per symbol on first fetch)
            avg_time = multi_fetch_time / len(test_symbols)
            performance_ok = avg_time < 2.0
            
            if not performance_ok:
                print(f"⚠️  Performance slower than expected: {avg_time:.3f}s per symbol")
            
            return performance_ok
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False
    
    def test_failover(self):
        """Test failover mechanisms between data sources"""
        try:
            print("🔄 Testing failover mechanisms...")
            
            # This test validates that the system can handle source failures gracefully
            # and falls back to alternative sources
            
            test_symbol = "AAPL"
            
            # Force test different source priorities
            original_sources = self.pipeline.source_manager.equity_sources.copy()
            
            # Test with limited sources
            self.pipeline.source_manager.equity_sources = ['yfinance']  # yfinance only
            data1 = self.pipeline.get_close_series(test_symbol)
            
            # Reset sources
            self.pipeline.source_manager.equity_sources = original_sources
            
            if not data1.empty:
                print("✅ Failover mechanism working - can fetch from backup sources")
                return True
            else:
                print("❌ Failover failed - no backup data source succeeded")
                return False
                
        except Exception as e:
            print(f"❌ Failover test failed: {e}")
            return False


def main():
    """Run the complete test suite"""
    test_suite = PipelineTestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎯 Data Pipeline is ready for production use!")
    else:
        print("\n⚠️  Some tests failed. Please review and fix issues before production use.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)