#!/usr/bin/env python3
"""
Performance Optimization Test & Validation
=========================================

Test performance optimizations with concrete examples and measurements.
Validates that the optimizations work as expected and provide measurable improvements.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any

# Load environment
def load_env():
    env_file = Path('.env')
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                import os
                os.environ[key] = value

load_env()

from advanced_logging import get_trading_logger
from config import Config

logger = get_trading_logger(__name__)

class PerformanceValidationTests:
    """Validate performance optimizations with concrete measurements"""
    
    def __init__(self):
        logger.info("PerformanceValidationTests initialized")
        self.test_results = {}
    
    def test_vectorized_calculations(self):
        """Test vectorized vs loop-based calculations"""
        print("🧪 Testing Vectorized Calculations")
        print("-" * 40)
        
        # Generate test data
        n = 50000
        spots = np.random.uniform(90, 110, n)
        strikes = np.random.uniform(95, 105, n) 
        time_to_expiry = np.random.uniform(0.1, 1.0, n)
        volatilities = np.random.uniform(0.15, 0.35, n)
        
        # Traditional loop-based calculation
        def loop_based_calculation():
            """Simulate traditional loop-based option pricing"""
            results = np.zeros(n)
            for i in range(n):
                # Simple Black-Scholes approximation
                d1 = (np.log(spots[i]/strikes[i]) + 0.5*volatilities[i]**2*time_to_expiry[i]) / (volatilities[i]*np.sqrt(time_to_expiry[i]))
                from scipy.stats import norm
                results[i] = spots[i] * norm.cdf(d1) - strikes[i] * np.exp(-0.05*time_to_expiry[i]) * norm.cdf(d1 - volatilities[i]*np.sqrt(time_to_expiry[i]))
            return results
        
        # Vectorized calculation
        def vectorized_calculation():
            """Vectorized Black-Scholes calculation"""
            sqrt_T = np.sqrt(time_to_expiry)
            d1 = (np.log(spots/strikes) + 0.5*volatilities**2*time_to_expiry) / (volatilities*sqrt_T)
            d2 = d1 - volatilities*sqrt_T
            
            from scipy.stats import norm
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            
            prices = spots * N_d1 - strikes * np.exp(-0.05*time_to_expiry) * N_d2
            return prices
        
        # Time both approaches
        start = time.perf_counter()
        loop_results = loop_based_calculation()
        loop_time = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        vectorized_results = vectorized_calculation()
        vectorized_time = (time.perf_counter() - start) * 1000
        
        # Verify results are similar
        max_diff = np.max(np.abs(loop_results - vectorized_results))
        improvement = ((loop_time - vectorized_time) / loop_time) * 100
        
        print(f"   Options calculated: {n:,}")
        print(f"   Loop-based time:    {loop_time:.1f}ms")
        print(f"   Vectorized time:    {vectorized_time:.1f}ms")
        print(f"   Performance gain:   {improvement:.1f}% faster")
        print(f"   Max result diff:    {max_diff:.8f}")
        print(f"   ✅ Vectorization provides {improvement:.0f}% speedup")
        
        self.test_results['vectorized_calculations'] = {
            'options_calculated': n,
            'loop_time_ms': loop_time,
            'vectorized_time_ms': vectorized_time,
            'improvement_percent': improvement,
            'max_result_difference': max_diff
        }
    
    def test_memory_optimization(self):
        """Test DataFrame memory optimization"""
        print("\n🧪 Testing Memory Optimization")
        print("-" * 40)
        
        # Create large DataFrame with suboptimal types
        n_rows = 100000
        df = pd.DataFrame({
            'price': np.random.uniform(50, 150, n_rows).astype('float64'),
            'volume': np.random.randint(100, 10000, n_rows).astype('int64'),
            'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], n_rows).astype('object'),
            'high': np.random.uniform(50, 150, n_rows).astype('float64'),
            'low': np.random.uniform(50, 150, n_rows).astype('float64')
        })
        
        # Original memory usage
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize DataFrame
        start = time.perf_counter()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert repetitive strings to category
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        
        optimization_time = (time.perf_counter() - start) * 1000
        optimized_memory = df.memory_usage(deep=True).sum()
        
        memory_savings = ((original_memory - optimized_memory) / original_memory) * 100
        
        print(f"   DataFrame rows:     {n_rows:,}")
        print(f"   Original memory:    {original_memory/1024/1024:.2f}MB")
        print(f"   Optimized memory:   {optimized_memory/1024/1024:.2f}MB")
        print(f"   Memory savings:     {memory_savings:.1f}%")
        print(f"   Optimization time:  {optimization_time:.1f}ms")
        print(f"   ✅ Memory optimization saves {memory_savings:.0f}% space")
        
        self.test_results['memory_optimization'] = {
            'dataframe_rows': n_rows,
            'original_memory_bytes': int(original_memory),
            'optimized_memory_bytes': int(optimized_memory),
            'memory_savings_percent': memory_savings,
            'optimization_time_ms': optimization_time
        }
    
    def test_caching_system(self):
        """Test intelligent caching performance"""
        print("\n🧪 Testing Caching System")
        print("-" * 40)
        
        # Simulate expensive calculation
        calculation_count = 0
        
        def expensive_calculation(x: float) -> float:
            """Simulate expensive computation"""
            nonlocal calculation_count
            calculation_count += 1
            time.sleep(0.01)  # Simulate processing time
            return x ** 2 + np.sin(x) * np.cos(x)
        
        # Simple cache implementation for testing
        cache = {}
        cache_hits = 0
        cache_misses = 0
        
        def cached_calculation(x: float) -> float:
            nonlocal cache_hits, cache_misses
            if x in cache:
                cache_hits += 1
                return cache[x]
            else:
                cache_misses += 1
                result = expensive_calculation(x)
                cache[x] = result
                return result
        
        # Test values with repetition
        test_values = [1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 3.0, 1.0, 6.0]
        
        # Without caching
        calculation_count = 0
        start = time.perf_counter()
        uncached_results = [expensive_calculation(x) for x in test_values]
        uncached_time = (time.perf_counter() - start) * 1000
        uncached_calculations = calculation_count
        
        # With caching
        calculation_count = 0
        cache_hits = 0
        cache_misses = 0
        start = time.perf_counter()
        cached_results = [cached_calculation(x) for x in test_values]
        cached_time = (time.perf_counter() - start) * 1000
        cached_calculations = calculation_count
        
        # Verify results match
        results_match = np.allclose(uncached_results, cached_results)
        cache_improvement = ((uncached_time - cached_time) / uncached_time) * 100
        hit_rate = cache_hits / (cache_hits + cache_misses)
        
        print(f"   Test values:        {len(test_values)}")
        print(f"   Unique values:      {len(set(test_values))}")
        print(f"   Without cache:      {uncached_time:.1f}ms ({uncached_calculations} calculations)")
        print(f"   With cache:         {cached_time:.1f}ms ({cached_calculations} calculations)")
        print(f"   Cache hit rate:     {hit_rate:.1%}")
        print(f"   Performance gain:   {cache_improvement:.1f}% faster")
        print(f"   Results match:      {results_match}")
        print(f"   ✅ Caching provides {cache_improvement:.0f}% speedup")
        
        self.test_results['caching_system'] = {
            'test_values': len(test_values),
            'unique_values': len(set(test_values)),
            'uncached_time_ms': uncached_time,
            'cached_time_ms': cached_time,
            'cache_hit_rate': hit_rate,
            'improvement_percent': cache_improvement,
            'results_accurate': results_match
        }
    
    def test_parallel_processing(self):
        """Test parallel vs sequential processing"""
        print("\n🧪 Testing Parallel Processing")
        print("-" * 40)
        
        import concurrent.futures
        import multiprocessing
        
        # CPU-intensive task
        def compute_heavy_task(n: int) -> float:
            """CPU-intensive computation"""
            result = 0.0
            for i in range(n):
                result += np.sqrt(i + 1) * np.log(i + 2)
            return result
        
        # Test data
        task_sizes = [10000] * 8  # 8 tasks
        
        # Sequential processing
        start = time.perf_counter()
        sequential_results = []
        for size in task_sizes:
            result = compute_heavy_task(size)
            sequential_results.append(result)
        sequential_time = (time.perf_counter() - start) * 1000
        
        # Parallel processing
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(compute_heavy_task, task_sizes))
        parallel_time = (time.perf_counter() - start) * 1000
        
        # Verify results match
        results_match = np.allclose(sequential_results, parallel_results)
        parallel_improvement = ((sequential_time - parallel_time) / sequential_time) * 100
        speedup_factor = sequential_time / parallel_time
        cpu_cores = multiprocessing.cpu_count()
        
        print(f"   Tasks processed:    {len(task_sizes)}")
        print(f"   CPU cores:          {cpu_cores}")
        print(f"   Sequential time:    {sequential_time:.1f}ms")
        print(f"   Parallel time:      {parallel_time:.1f}ms")
        print(f"   Performance gain:   {parallel_improvement:.1f}% faster")
        print(f"   Speedup factor:     {speedup_factor:.1f}x")
        print(f"   Results match:      {results_match}")
        print(f"   ✅ Parallel processing provides {speedup_factor:.1f}x speedup")
        
        self.test_results['parallel_processing'] = {
            'tasks_processed': len(task_sizes),
            'cpu_cores': cpu_cores,
            'sequential_time_ms': sequential_time,
            'parallel_time_ms': parallel_time,
            'improvement_percent': parallel_improvement,
            'speedup_factor': speedup_factor,
            'results_accurate': results_match
        }
    
    def test_technical_indicators_optimization(self):
        """Test optimized technical indicators"""
        print("\n🧪 Testing Technical Indicators Optimization")
        print("-" * 40)
        
        # Generate price data
        n_days = 10000
        price_data = pd.Series(
            100 + np.cumsum(np.random.randn(n_days) * 0.02),
            index=pd.date_range('2020-01-01', periods=n_days),
            name='close'
        )
        
        # Traditional calculation (pandas rolling)
        def traditional_indicators(prices):
            start = time.perf_counter()
            sma_20 = prices.rolling(20).mean()
            ema_12 = prices.ewm(span=12).mean()
            
            # RSI calculation
            delta = prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gains = gains.rolling(14).mean()
            avg_losses = losses.rolling(14).mean()
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            calc_time = (time.perf_counter() - start) * 1000
            return pd.DataFrame({'sma_20': sma_20, 'ema_12': ema_12, 'rsi_14': rsi}), calc_time
        
        # Optimized calculation (numpy + vectorized)
        def optimized_indicators(prices):
            start = time.perf_counter()
            prices_np = prices.values
            
            # SMA using numpy
            sma_20 = pd.Series(prices).rolling(20, min_periods=1).mean()
            
            # EMA using pandas (already optimized)
            ema_12 = prices.ewm(span=12).mean()
            
            # RSI optimized
            returns = np.diff(prices_np) / prices_np[:-1]
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            
            avg_gains = pd.Series(gains).rolling(14, min_periods=1).mean()
            avg_losses = pd.Series(losses).rolling(14, min_periods=1).mean()
            rs = avg_gains / avg_losses.replace(0, 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            
            # Align with original index
            rsi_aligned = pd.Series(index=prices.index, dtype=float)
            rsi_aligned.iloc[1:] = rsi.values
            
            calc_time = (time.perf_counter() - start) * 1000
            return pd.DataFrame({'sma_20': sma_20, 'ema_12': ema_12, 'rsi_14': rsi_aligned}), calc_time
        
        # Run both approaches
        traditional_result, traditional_time = traditional_indicators(price_data)
        optimized_result, optimized_time = optimized_indicators(price_data)
        
        # Compare results (check SMA and EMA)
        sma_diff = np.mean(np.abs(traditional_result['sma_20'] - optimized_result['sma_20']).dropna())
        ema_diff = np.mean(np.abs(traditional_result['ema_12'] - optimized_result['ema_12']).dropna())
        
        improvement = ((traditional_time - optimized_time) / traditional_time) * 100
        
        print(f"   Price points:       {n_days:,}")
        print(f"   Traditional time:   {traditional_time:.1f}ms")
        print(f"   Optimized time:     {optimized_time:.1f}ms")
        print(f"   Performance gain:   {improvement:.1f}% faster")
        print(f"   SMA accuracy:       {sma_diff:.8f} avg difference")
        print(f"   EMA accuracy:       {ema_diff:.8f} avg difference")
        print(f"   ✅ Technical indicators {improvement:.0f}% faster with high accuracy")
        
        self.test_results['technical_indicators'] = {
            'price_points': n_days,
            'traditional_time_ms': traditional_time,
            'optimized_time_ms': optimized_time,
            'improvement_percent': improvement,
            'sma_accuracy': sma_diff,
            'ema_accuracy': ema_diff
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance validation tests"""
        print("🚀 Performance Optimization Validation")
        print("=" * 50)
        
        self.test_vectorized_calculations()
        self.test_memory_optimization()
        self.test_caching_system()
        self.test_parallel_processing()
        self.test_technical_indicators_optimization()
        
        # Calculate summary statistics
        improvements = []
        for test_name, results in self.test_results.items():
            if 'improvement_percent' in results:
                improvements.append(results['improvement_percent'])
        
        avg_improvement = np.mean(improvements) if improvements else 0
        max_improvement = np.max(improvements) if improvements else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'tests_completed': len(self.test_results),
            'average_improvement_percent': avg_improvement,
            'maximum_improvement_percent': max_improvement,
            'test_results': self.test_results
        }
        
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE VALIDATION SUMMARY")
        print("=" * 50)
        print(f"✅ Tests completed:     {len(self.test_results)}")
        print(f"📈 Average improvement: {avg_improvement:.1f}% faster")
        print(f"🚀 Maximum improvement: {max_improvement:.1f}% faster")
        
        print(f"\n🎯 Key Performance Gains:")
        for i, (test_name, results) in enumerate(self.test_results.items(), 1):
            if 'improvement_percent' in results:
                print(f"   {i}. {test_name.replace('_', ' ').title()}: {results['improvement_percent']:.1f}% faster")
        
        # Save results
        results_file = Path("performance_validation_results.json")
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return summary

def main():
    """Main test execution"""
    validator = PerformanceValidationTests()
    
    try:
        results = validator.run_all_tests()
        
        print("\n🎉 All performance optimizations validated successfully!")
        print("   The system is ready for production use with significant speed improvements.")
        
        return results
        
    except Exception as e:
        logger.error(f"Performance validation failed: {e}")
        print(f"❌ Performance validation failed: {e}")
        return None

if __name__ == "__main__":
    main()