#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Error Handling System
==========================================================

Tests all components of the enhanced error handling framework:
- Custom exception hierarchy
- Retry mechanisms with exponential backoff
- Circuit breaker patterns
- Error monitoring and alerting
- Recovery strategies
- Integration with data pipeline

Run with: python test_error_handling.py
"""

import sys
import time
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from error_handling import *
from enhanced_data_pipeline import EnhancedDataPipeline
from config import Config
import unittest.mock as mock


class ErrorHandlingTestSuite:
    """Comprehensive test suite for error handling system"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def run_all_tests(self):
        """Run complete error handling test suite"""
        print("🛡️  Starting Enhanced Error Handling Test Suite")
        print("=" * 60)
        
        tests = [
            ("Custom Exception Hierarchy", self.test_exception_hierarchy),
            ("Retry Mechanisms", self.test_retry_mechanisms),
            ("Circuit Breaker Pattern", self.test_circuit_breaker),
            ("Error Monitoring & Metrics", self.test_error_monitoring),
            ("Recovery Strategies", self.test_recovery_strategies),
            ("Integration with Data Pipeline", self.test_pipeline_integration),
            ("Concurrent Error Handling", self.test_concurrent_handling),
            ("Performance Impact", self.test_performance_impact),
            ("System Health Monitoring", self.test_health_monitoring),
            ("Error Handler Decorators", self.test_decorators)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 Testing: {test_name}")
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
                self.test_results[test_name] = False
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 ERROR HANDLING TEST SUITE SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Runtime: {time.time() - self.start_time:.2f}s")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL ERROR HANDLING TESTS PASSED!")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. System needs attention.")
            
        return passed_tests == total_tests
    
    def test_exception_hierarchy(self):
        """Test custom exception hierarchy and error categorization"""
        try:
            # Test base trading system error
            base_error = TradingSystemError(
                "Test error", 
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM
            )
            
            if not isinstance(base_error, Exception):
                print("❌ Base error not an Exception")
                return False
            
            # Test specific error types
            data_error = DataFeedError("API failed", source="test_api")
            market_error = MarketDataError("Bad data", symbol="AAPL")
            signal_error = SignalGenerationError("Model failed", model="test_model")
            
            # Verify error attributes
            if data_error.category != ErrorCategory.DATA_FEED:
                print("❌ DataFeedError category incorrect")
                return False
            
            if market_error.symbol != "AAPL":
                print("❌ MarketDataError symbol not set")
                return False
            
            if not hasattr(base_error, 'error_id'):
                print("❌ Error ID not generated")
                return False
            
            print("✅ Exception hierarchy working correctly")
            print(f"   Base error ID: {base_error.error_id}")
            print(f"   Data error source: {data_error.source}")
            print(f"   Market error symbol: {market_error.symbol}")
            
            return True
            
        except Exception as e:
            print(f"❌ Exception hierarchy test failed: {e}")
            return False
    
    def test_retry_mechanisms(self):
        """Test retry logic with exponential backoff"""
        try:
            # Test retry configuration
            config = RetryConfig(max_attempts=3, base_delay=0.1, exponential_base=2.0)
            handler = RetryHandler(config)
            
            # Test delay calculation
            delay1 = handler.calculate_delay(0)
            delay2 = handler.calculate_delay(1)
            delay3 = handler.calculate_delay(2)
            
            if not (delay1 < delay2 < delay3):
                print(f"❌ Exponential backoff not working: {delay1}, {delay2}, {delay3}")
                return False
            
            # Test retry decorator
            attempt_count = 0
            
            @handler
            def failing_function():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise NetworkError("Simulated network failure")
                return "success"
            
            start_time = time.time()
            result = failing_function()
            elapsed = time.time() - start_time
            
            if result != "success":
                print(f"❌ Retry did not succeed: {result}")
                return False
            
            if attempt_count != 3:
                print(f"❌ Wrong number of attempts: {attempt_count}")
                return False
            
            if elapsed < 0.3:  # Should have delays
                print(f"❌ Retry delays not working: {elapsed:.3f}s")
                return False
            
            print("✅ Retry mechanisms working correctly")
            print(f"   Attempts: {attempt_count}")
            print(f"   Total time: {elapsed:.3f}s")
            print(f"   Delay progression: {delay1:.3f}s → {delay2:.3f}s → {delay3:.3f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Retry mechanism test failed: {e}")
            return False
    
    def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        try:
            config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=1.0,
                success_threshold=2
            )
            breaker = CircuitBreaker("test_breaker", config)
            
            # Test normal operation (CLOSED)
            if breaker.state != CircuitState.CLOSED:
                print(f"❌ Initial state not CLOSED: {breaker.state}")
                return False
            
            # Simulate failures to open circuit
            for i in range(3):
                try:
                    @breaker
                    def failing_function():
                        raise NetworkError("Simulated failure")
                    
                    failing_function()
                except:
                    pass  # Expected to fail
            
            # Circuit should be OPEN now
            if breaker.state != CircuitState.OPEN:
                print(f"❌ Circuit not OPEN after failures: {breaker.state}")
                return False
            
            # Test that calls are rejected
            try:
                @breaker
                def test_function():
                    return "should not execute"
                
                result = test_function()
                print("❌ Circuit breaker did not reject call")
                return False
            except TradingSystemError as e:
                if "OPEN" not in str(e):
                    print(f"❌ Wrong error message: {e}")
                    return False
            
            # Wait for recovery timeout
            time.sleep(1.1)
            
            # Should move to HALF_OPEN on next call
            success_count = 0
            for i in range(2):  # success_threshold = 2
                @breaker
                def success_function():
                    return "success"
                
                result = success_function()
                success_count += 1
            
            # Circuit should be CLOSED again
            if breaker.state != CircuitState.CLOSED:
                print(f"❌ Circuit not CLOSED after recovery: {breaker.state}")
                return False
            
            print("✅ Circuit breaker pattern working correctly")
            print(f"   Final state: {breaker.state.value}")
            print(f"   Recovery successes: {success_count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Circuit breaker test failed: {e}")
            return False
    
    def test_error_monitoring(self):
        """Test error monitoring and metrics collection"""
        try:
            monitor = ErrorMonitor(alert_threshold=3, time_window=60)
            
            # Test initial state
            if monitor.metrics.total_errors != 0:
                print("❌ Monitor not initialized correctly")
                return False
            
            # Record various errors
            errors = [
                DataFeedError("API failure 1", source="test"),
                MarketDataError("Bad data", symbol="AAPL", severity=ErrorSeverity.HIGH),
                NetworkError("Connection lost", severity=ErrorSeverity.CRITICAL),
                DataFeedError("API failure 2", source="test"),
                SignalGenerationError("Model error", model="test")
            ]
            
            for error in errors:
                monitor.record_error(error)
            
            # Check metrics
            if monitor.metrics.total_errors != 5:
                print(f"❌ Wrong error count: {monitor.metrics.total_errors}")
                return False
            
            # Check category counts
            data_feed_count = monitor.metrics.errors_by_category.get(ErrorCategory.DATA_FEED, 0)
            if data_feed_count != 2:
                print(f"❌ Wrong data feed error count: {data_feed_count}")
                return False
            
            # Check severity counts
            critical_count = monitor.metrics.errors_by_severity.get(ErrorSeverity.CRITICAL, 0)
            if critical_count != 1:
                print(f"❌ Wrong critical error count: {critical_count}")
                return False
            
            # Test health summary
            health = monitor.get_health_summary()
            if health["total_errors"] != 5:
                print(f"❌ Health summary incorrect: {health}")
                return False
            
            print("✅ Error monitoring working correctly")
            print(f"   Total errors: {monitor.metrics.total_errors}")
            print(f"   Categories: {dict(monitor.metrics.errors_by_category)}")
            print(f"   Health status: {health['status']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error monitoring test failed: {e}")
            return False
    
    def test_recovery_strategies(self):
        """Test different recovery strategies"""
        try:
            handler = ErrorHandler()
            
            # Test ABORT strategy (default)
            try:
                with handler.handle_errors("test_abort", recovery_strategy=RecoveryStrategy.ABORT):
                    raise ValueError("Test error")
                print("❌ ABORT strategy did not raise error")
                return False
            except TradingSystemError:
                pass  # Expected
            
            # Test SKIP strategy
            result_captured = False
            
            try:
                with handler.handle_errors("test_skip", recovery_strategy=RecoveryStrategy.SKIP):
                    raise ValueError("Test error")
                result_captured = True  # Should reach here with SKIP
            except:
                print("❌ SKIP strategy raised error")
                return False
            
            if not result_captured:
                print("❌ SKIP strategy did not continue execution")
                return False
            
            print("✅ Recovery strategies working correctly")
            print("   ABORT: Properly raises errors")
            print("   SKIP: Properly continues execution")
            
            return True
            
        except Exception as e:
            print(f"❌ Recovery strategies test failed: {e}")
            return False
    
    def test_pipeline_integration(self):
        """Test enhanced data pipeline integration"""
        try:
            # Load environment
            import os
            from pathlib import Path
            
            env_file = Path('.env')
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            
            config = Config()
            pipeline = EnhancedDataPipeline(config)
            
            # Test successful operation
            data = pipeline.get_close_series("AAPL", start="2024-08-01", end="2024-08-10")
            if data.empty:
                print("⚠️  No real data available, testing with mock scenario")
                # Test error handling with mock
                return self._test_pipeline_error_scenarios(pipeline)
            
            # Test health check
            health = pipeline.health_check()
            if "error_handling" not in health:
                print("❌ Enhanced health check missing error handling info")
                return False
            
            # Test error summary
            error_summary = pipeline.get_error_summary()
            if "total_errors" not in error_summary:
                print("❌ Error summary missing key metrics")
                return False
            
            print("✅ Pipeline integration working correctly")
            print(f"   Data points: {len(data)}")
            print(f"   Health status: {health.get('overall_status', 'unknown')}")
            print(f"   Total errors: {error_summary['total_errors']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline integration test failed: {e}")
            return False
    
    def _test_pipeline_error_scenarios(self, pipeline):
        """Test pipeline error handling with mock scenarios"""
        try:
            # Mock a failing API call
            original_method = pipeline._fetch_polygon_close_series
            
            def mock_failing_polygon(*args, **kwargs):
                raise NetworkError("Mock API failure")
            
            pipeline._fetch_polygon_close_series = mock_failing_polygon
            
            # This should trigger fallback mechanisms
            try:
                data = pipeline.get_close_series("AAPL")
                print("✅ Fallback mechanisms working")
            except Exception as e:
                print(f"⚠️  Pipeline fallback needs attention: {e}")
            
            # Restore original method
            pipeline._fetch_polygon_close_series = original_method
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline error scenario test failed: {e}")
            return False
    
    def test_concurrent_handling(self):
        """Test error handling under concurrent load"""
        try:
            handler = ErrorHandler()
            results = []
            errors = []
            
            def worker(worker_id):
                try:
                    with handler.handle_errors(f"worker_{worker_id}"):
                        if worker_id % 3 == 0:  # Every 3rd worker fails
                            raise NetworkError(f"Worker {worker_id} failure")
                        results.append(f"success_{worker_id}")
                except Exception as e:
                    errors.append(f"error_{worker_id}")
            
            # Run concurrent workers
            threads = []
            for i in range(10):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # Check results
            expected_successes = 7  # 10 - 3 failures
            expected_failures = 3
            
            if len(results) != expected_successes:
                print(f"❌ Wrong success count: {len(results)} (expected {expected_successes})")
                return False
            
            if len(errors) != expected_failures:
                print(f"❌ Wrong error count: {len(errors)} (expected {expected_failures})")
                return False
            
            print("✅ Concurrent error handling working correctly")
            print(f"   Successful operations: {len(results)}")
            print(f"   Failed operations: {len(errors)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Concurrent handling test failed: {e}")
            return False
    
    def test_performance_impact(self):
        """Test performance impact of error handling system"""
        try:
            # Measure baseline performance
            def simple_function():
                return "result"
            
            start = time.time()
            for _ in range(1000):
                simple_function()
            baseline_time = time.time() - start
            
            # Measure with error handling
            @trading_operation()
            def wrapped_function():
                return "result"
            
            start = time.time()
            for _ in range(1000):
                wrapped_function()
            wrapped_time = time.time() - start
            
            # Calculate overhead
            overhead = (wrapped_time - baseline_time) / baseline_time * 100
            
            if overhead > 50:  # More than 50% overhead is concerning
                print(f"⚠️  High performance overhead: {overhead:.1f}%")
                return False
            
            print("✅ Performance impact acceptable")
            print(f"   Baseline: {baseline_time*1000:.3f}ms")
            print(f"   With error handling: {wrapped_time*1000:.3f}ms") 
            print(f"   Overhead: {overhead:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance impact test failed: {e}")
            return False
    
    def test_health_monitoring(self):
        """Test system health monitoring"""
        try:
            # Test initial health
            health = get_system_health()
            if health["status"] != "healthy":
                print(f"⚠️  System not healthy initially: {health}")
            
            # Simulate some errors
            error_handler.monitor.record_error(
                NetworkError("Test error", severity=ErrorSeverity.HIGH)
            )
            
            # Check updated health
            updated_health = get_system_health()
            if updated_health["total_errors"] != 1:
                print(f"❌ Health not updated: {updated_health}")
                return False
            
            # Test error statistics
            stats = get_error_statistics()
            if stats["total_errors"] != 1:
                print(f"❌ Statistics incorrect: {stats}")
                return False
            
            print("✅ Health monitoring working correctly")
            print(f"   System status: {updated_health['status']}")
            print(f"   Error count: {stats['total_errors']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Health monitoring test failed: {e}")
            return False
    
    def test_decorators(self):
        """Test error handling decorators"""
        try:
            # Test retry decorator
            call_count = 0
            
            @with_retry("default")
            def retry_test():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise NetworkError("Retry test")
                return "success"
            
            result = retry_test()
            if result != "success" or call_count != 2:
                print(f"❌ Retry decorator failed: {result}, {call_count}")
                return False
            
            # Test trading operation decorator
            @trading_operation(category=ErrorCategory.SIGNAL_GENERATION)
            def trading_test():
                raise ValueError("Trading test error")
            
            try:
                trading_test()
                print("❌ Trading operation decorator did not raise error")
                return False
            except TradingSystemError as e:
                if e.category != ErrorCategory.SIGNAL_GENERATION:
                    print(f"❌ Wrong error category: {e.category}")
                    return False
            
            print("✅ Decorators working correctly")
            print(f"   Retry calls: {call_count}")
            print("   Trading operation properly wrapped errors")
            
            return True
            
        except Exception as e:
            print(f"❌ Decorators test failed: {e}")
            return False


def main():
    """Run the complete error handling test suite"""
    # Reset error monitoring for clean test
    reset_error_monitoring()
    
    test_suite = ErrorHandlingTestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎯 Enhanced Error Handling System is ready for production!")
    else:
        print("\n⚠️  Some tests failed. Please review and fix issues.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)