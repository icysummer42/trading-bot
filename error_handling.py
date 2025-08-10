"""
Enhanced Error Handling & Resilience System
==========================================

Professional error handling framework for the quantitative options trading bot.
Provides comprehensive error classification, recovery strategies, circuit breakers,
and resilience patterns for production trading operations.

Features:
- Hierarchical exception taxonomy
- Automatic retry mechanisms with exponential backoff
- Circuit breaker patterns to prevent cascade failures
- Error recovery strategies specific to trading operations
- Comprehensive error logging and alerting
- Performance impact monitoring
- Graceful degradation strategies
"""

from __future__ import annotations
import time
import traceback
import functools
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from contextlib import contextmanager
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Classification of error severity levels"""
    LOW = "low"           # Minor issues, system continues normally
    MEDIUM = "medium"     # Noticeable impact, but system stable
    HIGH = "high"         # Significant impact, requires attention
    CRITICAL = "critical" # System stability at risk, immediate action needed


class ErrorCategory(Enum):
    """Categories of errors in the trading system"""
    DATA_FEED = "data_feed"         # Data source/API issues
    MARKET_DATA = "market_data"     # Data quality/validation issues
    SIGNAL_GENERATION = "signal"    # ML/signal processing issues
    STRATEGY = "strategy"           # Strategy execution issues
    RISK_MANAGEMENT = "risk"        # Risk system issues
    EXECUTION = "execution"         # Trade execution issues
    NETWORK = "network"            # Network connectivity issues
    EXTERNAL_API = "external_api"  # Third-party API issues
    CONFIGURATION = "config"       # Configuration/setup issues
    SYSTEM = "system"              # General system issues


class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    RETRY = "retry"                 # Retry with backoff
    FALLBACK = "fallback"          # Use alternative method/source
    SKIP = "skip"                  # Skip and continue with next
    DEGRADE = "degrade"            # Reduce functionality gracefully
    CIRCUIT_BREAK = "circuit_break" # Stop operations temporarily
    ABORT = "abort"                # Stop current operation completely


# ═══════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTION HIERARCHY
# ═══════════════════════════════════════════════════════════════════

class TradingSystemError(Exception):
    """Base exception for all trading system errors"""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now()
        self.error_id = f"{category.value}_{int(time.time()*1000)}"


class DataFeedError(TradingSystemError):
    """Data feed/source related errors"""
    def __init__(self, message: str, source: str = "unknown", **kwargs):
        super().__init__(message, ErrorCategory.DATA_FEED, **kwargs)
        self.source = source


class MarketDataError(TradingSystemError):
    """Market data quality/validation errors"""
    def __init__(self, message: str, symbol: str = "unknown", **kwargs):
        super().__init__(message, ErrorCategory.MARKET_DATA, **kwargs)
        self.symbol = symbol


class SignalGenerationError(TradingSystemError):
    """Signal processing/ML errors"""
    def __init__(self, message: str, model: str = "unknown", **kwargs):
        super().__init__(message, ErrorCategory.SIGNAL_GENERATION, **kwargs)
        self.model = model


class StrategyExecutionError(TradingSystemError):
    """Strategy execution errors"""
    def __init__(self, message: str, strategy: str = "unknown", **kwargs):
        super().__init__(message, ErrorCategory.STRATEGY, **kwargs)
        self.strategy = strategy


class RiskManagementError(TradingSystemError):
    """Risk management system errors"""
    def __init__(self, message: str, risk_type: str = "unknown", **kwargs):
        super().__init__(message, ErrorCategory.RISK_MANAGEMENT, **kwargs)
        self.risk_type = risk_type


class ExecutionError(TradingSystemError):
    """Trade execution errors"""
    def __init__(self, message: str, order_type: str = "unknown", **kwargs):
        super().__init__(message, ErrorCategory.EXECUTION, **kwargs)
        self.order_type = order_type


class NetworkError(TradingSystemError):
    """Network connectivity errors"""
    def __init__(self, message: str, endpoint: str = "unknown", **kwargs):
        super().__init__(message, ErrorCategory.NETWORK, **kwargs)
        self.endpoint = endpoint


class ExternalAPIError(TradingSystemError):
    """External API errors"""
    def __init__(self, message: str, api_provider: str = "unknown", **kwargs):
        super().__init__(message, ErrorCategory.EXTERNAL_API, **kwargs)
        self.api_provider = api_provider


class ConfigurationError(TradingSystemError):
    """Configuration/setup errors"""
    def __init__(self, message: str, config_key: str = "unknown", **kwargs):
        super().__init__(message, ErrorCategory.CONFIGURATION, **kwargs)
        self.config_key = config_key


# ═══════════════════════════════════════════════════════════════════
# ERROR RECOVERY & RETRY MECHANISMS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (NetworkError, ExternalAPIError, DataFeedError)


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff"""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            # Add jitter to prevent thundering herd
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if exception should be retried"""
        if attempt >= self.config.max_attempts:
            return False
            
        return isinstance(exception, self.config.retryable_exceptions)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to functions"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not self.should_retry(e, attempt):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise
                    
                    if attempt < self.config.max_attempts - 1:
                        delay = self.calculate_delay(attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                     f"Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    
            # All retries exhausted
            logger.error(f"All {self.config.max_attempts} attempts failed for {func.__name__}")
            raise last_exception
            
        return wrapper


# ═══════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER PATTERN
# ═══════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject calls
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying recovery
    success_threshold: int = 3      # Successes to close circuit
    monitored_exceptions: tuple = (NetworkError, ExternalAPIError)


class CircuitBreaker:
    """Circuit breaker implementation for external dependencies"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
        
    def _should_allow_call(self) -> bool:
        """Check if call should be allowed based on circuit state"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if (time.time() - self.last_failure_time) >= self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        return False
    
    def _record_success(self):
        """Record successful operation"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} CLOSED after recovery")
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self, exception: Exception):
        """Record failed operation"""
        if not isinstance(exception, self.config.monitored_exceptions):
            return
            
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} OPENED due to failures")
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} re-OPENED during recovery test")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add circuit breaker protection"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._should_allow_call():
                raise TradingSystemError(
                    f"Circuit breaker {self.name} is OPEN",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.HIGH,
                    context={"circuit_state": self.state.value}
                )
            
            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure(e)
                raise
                
        return wrapper


# ═══════════════════════════════════════════════════════════════════
# ERROR MONITORING & ALERTING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ErrorMetrics:
    """Error monitoring metrics"""
    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=dict)
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=dict)
    recent_errors: List[TradingSystemError] = field(default_factory=list)
    first_error_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None


class ErrorMonitor:
    """Monitors and tracks error patterns across the system"""
    
    def __init__(self, alert_threshold: int = 10, time_window: int = 300):
        self.metrics = ErrorMetrics()
        self.alert_threshold = alert_threshold  # Errors per time window
        self.time_window = time_window  # Seconds
        self.alerts_sent = set()
        
    def record_error(self, error: TradingSystemError):
        """Record an error occurrence"""
        self.metrics.total_errors += 1
        
        # Update category counts
        if error.category not in self.metrics.errors_by_category:
            self.metrics.errors_by_category[error.category] = 0
        self.metrics.errors_by_category[error.category] += 1
        
        # Update severity counts
        if error.severity not in self.metrics.errors_by_severity:
            self.metrics.errors_by_severity[error.severity] = 0
        self.metrics.errors_by_severity[error.severity] += 1
        
        # Update recent errors (keep last 100)
        self.metrics.recent_errors.append(error)
        if len(self.metrics.recent_errors) > 100:
            self.metrics.recent_errors.pop(0)
        
        # Update timestamps
        if self.metrics.first_error_time is None:
            self.metrics.first_error_time = error.timestamp
        self.metrics.last_error_time = error.timestamp
        
        # Check for alert conditions
        self._check_alert_conditions(error)
    
    def _check_alert_conditions(self, error: TradingSystemError):
        """Check if error conditions warrant alerts"""
        # Immediate alert for critical errors
        if error.severity == ErrorSeverity.CRITICAL:
            self._send_alert(f"CRITICAL ERROR: {error.message}", error)
            return
        
        # Alert for high frequency of errors
        recent_time = datetime.now() - timedelta(seconds=self.time_window)
        recent_errors = [e for e in self.metrics.recent_errors if e.timestamp >= recent_time]
        
        if len(recent_errors) >= self.alert_threshold:
            alert_key = f"frequency_{error.category.value}_{int(time.time() / self.time_window)}"
            if alert_key not in self.alerts_sent:
                self._send_alert(f"High error frequency: {len(recent_errors)} {error.category.value} errors in {self.time_window}s", error)
                self.alerts_sent.add(alert_key)
    
    def _send_alert(self, message: str, error: TradingSystemError):
        """Send alert (can be extended with email, Slack, etc.)"""
        logger.error(f"ALERT: {message}")
        logger.error(f"Error details: {error.error_id} - {error.message}")
        
        # TODO: Implement actual alerting mechanisms
        # - Email notifications
        # - Slack/Discord webhooks
        # - SMS alerts for critical issues
        # - Dashboard notifications
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get system health summary based on error patterns"""
        if self.metrics.total_errors == 0:
            return {"status": "healthy", "total_errors": 0}
        
        recent_time = datetime.now() - timedelta(seconds=self.time_window)
        recent_errors = [e for e in self.metrics.recent_errors if e.timestamp >= recent_time]
        
        critical_errors = sum(1 for e in recent_errors if e.severity == ErrorSeverity.CRITICAL)
        high_errors = sum(1 for e in recent_errors if e.severity == ErrorSeverity.HIGH)
        
        if critical_errors > 0:
            status = "critical"
        elif high_errors > 3 or len(recent_errors) > self.alert_threshold:
            status = "degraded"
        elif len(recent_errors) > self.alert_threshold // 2:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "total_errors": self.metrics.total_errors,
            "recent_errors": len(recent_errors),
            "critical_errors": critical_errors,
            "high_errors": high_errors,
            "categories": dict(self.metrics.errors_by_category),
            "last_error": self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None
        }


# ═══════════════════════════════════════════════════════════════════
# ERROR HANDLING CONTEXT MANAGERS
# ═══════════════════════════════════════════════════════════════════

class ErrorHandler:
    """Central error handling system"""
    
    def __init__(self):
        self.monitor = ErrorMonitor()
        self.retry_configs = {}
        self.circuit_breakers = {}
        
    def register_retry_config(self, name: str, config: RetryConfig):
        """Register a retry configuration"""
        self.retry_configs[name] = config
        
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig):
        """Register a circuit breaker"""
        self.circuit_breakers[name] = CircuitBreaker(name, config)
    
    def get_retry_handler(self, name: str = "default") -> RetryHandler:
        """Get retry handler by name"""
        config = self.retry_configs.get(name, RetryConfig())
        return RetryHandler(config)
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get circuit breaker by name"""
        if name not in self.circuit_breakers:
            # Create default circuit breaker
            self.register_circuit_breaker(name, CircuitBreakerConfig())
        return self.circuit_breakers[name]
    
    @contextmanager
    def handle_errors(
        self, 
        operation: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.ABORT,
        context: Optional[Dict[str, Any]] = None
    ):
        """Context manager for comprehensive error handling"""
        try:
            yield
        except TradingSystemError as e:
            # Already a trading system error, just record and re-raise
            self.monitor.record_error(e)
            logger.error(f"Trading system error in {operation}: {e.message}")
            raise
        except Exception as e:
            # Convert to trading system error
            trading_error = TradingSystemError(
                message=f"Error in {operation}: {str(e)}",
                category=category,
                severity=severity,
                context=context or {},
                original_exception=e
            )
            
            self.monitor.record_error(trading_error)
            logger.error(f"Unhandled error in {operation}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Apply recovery strategy
            if recovery_strategy == RecoveryStrategy.ABORT:
                raise trading_error
            elif recovery_strategy == RecoveryStrategy.SKIP:
                logger.warning(f"Skipping {operation} due to error: {e}")
                return
            else:
                # Other recovery strategies can be implemented here
                raise trading_error


# ═══════════════════════════════════════════════════════════════════
# GLOBAL ERROR HANDLER INSTANCE
# ═══════════════════════════════════════════════════════════════════

# Global instance for system-wide error handling
error_handler = ErrorHandler()

# Register default configurations
error_handler.register_retry_config("data_feed", RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    retryable_exceptions=(DataFeedError, NetworkError, ExternalAPIError)
))

error_handler.register_retry_config("api_calls", RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=30.0,
    retryable_exceptions=(NetworkError, ExternalAPIError)
))

error_handler.register_circuit_breaker("polygon_api", CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    monitored_exceptions=(NetworkError, ExternalAPIError)
))

error_handler.register_circuit_breaker("yfinance", CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0
))


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE DECORATORS
# ═══════════════════════════════════════════════════════════════════

def with_retry(config_name: str = "default"):
    """Decorator for adding retry logic"""
    def decorator(func):
        return error_handler.get_retry_handler(config_name)(func)
    return decorator


def with_circuit_breaker(breaker_name: str):
    """Decorator for adding circuit breaker protection"""
    def decorator(func):
        return error_handler.get_circuit_breaker(breaker_name)(func)
    return decorator


def trading_operation(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    recovery: RecoveryStrategy = RecoveryStrategy.ABORT
):
    """Decorator for wrapping trading operations with error handling"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with error_handler.handle_errors(
                operation=func.__name__,
                category=category,
                severity=severity,
                recovery_strategy=recovery,
                context={"function": func.__name__, "args_count": len(args)}
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_system_health() -> Dict[str, Any]:
    """Get overall system health status"""
    return error_handler.monitor.get_health_summary()


def get_error_statistics() -> Dict[str, Any]:
    """Get detailed error statistics"""
    metrics = error_handler.monitor.metrics
    return {
        "total_errors": metrics.total_errors,
        "errors_by_category": dict(metrics.errors_by_category),
        "errors_by_severity": dict(metrics.errors_by_severity),
        "first_error": metrics.first_error_time.isoformat() if metrics.first_error_time else None,
        "last_error": metrics.last_error_time.isoformat() if metrics.last_error_time else None,
        "recent_errors_count": len(metrics.recent_errors)
    }


def reset_error_monitoring():
    """Reset error monitoring (useful for testing)"""
    global error_handler
    error_handler = ErrorHandler()