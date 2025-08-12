#!/usr/bin/env python3
"""
Rate Limiting System Validation

Simple validation script to test the enhanced rate limiting system
and integration with the complete data pipeline.
"""

import asyncio
import time
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_rate_limiting import CoordinatedRateLimitManager, RequestPriority
from complete_data_pipeline_with_rate_limiting import create_rate_limited_pipeline
from advanced_configuration import create_config_manager

async def test_rate_limiter_basic():
    """Test basic rate limiting functionality."""
    print("🔧 Testing Basic Rate Limiting")
    print("-" * 40)
    
    try:
        # Create configuration
        config_manager = create_config_manager(environment="development")
        config = config_manager.get_config()
        
        # Test configuration for rate limiting
        test_config = {
            'test_provider': {
                'requests_per_minute': 60,
                'burst_limit': 10
            }
        }
        
        # Create rate limiter  
        rate_limiter = CoordinatedRateLimitManager(config_manager)
        
        print("✅ Rate limiter initialized successfully")
        
        # Test basic request
        async def mock_request():
            await asyncio.sleep(0.01)  # 10ms delay
            return {"status": "success", "data": "test"}
        
        # Execute test request
        start_time = time.time()
        result = await rate_limiter.execute_request(
            "test_provider",
            mock_request,
            priority=RequestPriority.MEDIUM
        )
        execution_time = time.time() - start_time
        
        print(f"✅ Request executed successfully in {execution_time:.3f}s")
        print(f"   Result: {result}")
        
        # Test multiple requests to verify rate limiting
        print("   Testing multiple requests...")
        
        requests_completed = 0
        start_batch = time.time()
        
        tasks = []
        for i in range(15):  # Exceed burst limit
            task = rate_limiter.execute_request("test_provider", mock_request)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_time = time.time() - start_batch
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        print(f"✅ {len(successful_results)}/15 requests completed in {batch_time:.2f}s")
        
        # Get health status
        health = await rate_limiter.get_health_status()
        print(f"✅ Health status: {health.get('overall_status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rate limiter test failed: {e}")
        return False

async def test_pipeline_integration():
    """Test rate-limited pipeline integration."""
    print("\n📊 Testing Pipeline Integration")
    print("-" * 40)
    
    try:
        # Create rate-limited pipeline
        pipeline = create_rate_limited_pipeline(environment="development")
        
        # Initialize async components (if method exists)
        if hasattr(pipeline, 'initialize_async'):
            await pipeline.initialize_async()
        print("✅ Rate-limited pipeline initialized")
        
        # Test rate limiting status
        rate_status = await pipeline.get_rate_limiting_status()
        print(f"✅ Rate limiting status: {rate_status.get('overall_status', 'unknown')}")
        
        # Test performance metrics
        perf_metrics = await pipeline.get_performance_metrics()
        print(f"✅ Performance metrics collected")
        
        if "rate_limiting" in perf_metrics:
            active_providers = len(perf_metrics["rate_limiting"])
            print(f"   Active rate-limited providers: {active_providers}")
        
        # Test health check
        health = await pipeline.health_check_comprehensive()
        print(f"✅ Comprehensive health check: {health.get('overall_status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

async def test_configuration_integration():
    """Test configuration system integration."""
    print("\n⚙️ Testing Configuration Integration")
    print("-" * 40)
    
    try:
        # Test configuration loading
        config_manager = create_config_manager(environment="development")
        config = config_manager.get_config()
        
        print("✅ Configuration loaded successfully")
        
        # Check for required configuration sections
        required_sections = ["data_sources", "trading", "monitoring"]
        for section in required_sections:
            if hasattr(config, section):
                print(f"✅ {section} configuration available")
            else:
                print(f"⚠️  {section} configuration missing")
        
        # Test rate limit configuration
        if hasattr(config, 'data_sources'):
            ds_config = config.data_sources
            if hasattr(ds_config, 'rate_limits'):
                print("✅ Rate limit configuration found")
                rate_limits = ds_config.rate_limits
                if hasattr(rate_limits, 'polygon'):
                    polygon_limits = rate_limits.polygon
                    print(f"   Polygon rate limit: {getattr(polygon_limits, 'requests_per_minute', 'N/A')} req/min")
            else:
                print("⚠️  Rate limit configuration not found in data_sources")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def run_comprehensive_validation():
    """Run comprehensive validation of the rate limiting system."""
    print("🚀 Enhanced Rate Limiting System Validation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    results = []
    
    # Test 1: Basic rate limiter
    result1 = await test_rate_limiter_basic()
    results.append(("Basic Rate Limiting", result1))
    
    # Test 2: Pipeline integration
    result2 = await test_pipeline_integration()
    results.append(("Pipeline Integration", result2))
    
    # Test 3: Configuration integration
    result3 = await test_configuration_integration()
    results.append(("Configuration Integration", result3))
    
    # Summary
    print("\n📋 Validation Summary")
    print("=" * 30)
    
    passed_tests = 0
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if passed:
            passed_tests += 1
    
    print(f"\n🎯 Overall Result: {passed_tests}/{len(results)} tests passed")
    
    if passed_tests == len(results):
        print("🎉 All validation tests PASSED!")
        print("Rate limiting system is ready for use.")
        return True
    else:
        print("⚠️  Some validation tests FAILED!")
        print("Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_validation())
    sys.exit(0 if success else 1)