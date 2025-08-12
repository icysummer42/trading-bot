#!/usr/bin/env python3
"""
Enhanced API Rate Limiting System
================================

Enterprise-grade rate limiting system for all data sources in the quantitative trading bot.
Coordinates rate limits across all APIs with intelligent throttling, quota management, and failover.

Key Features:
- Centralized rate limiting across all 9+ data sources
- Dynamic throttling based on API quotas and response times
- Intelligent backoff strategies (exponential, linear, custom)
- Rate limit sharing and coordination between components
- Quota tracking and budget management
- Circuit breaker integration for failed APIs
- Real-time rate limit monitoring and alerting
- Priority-based request queuing
- Distributed rate limiting support

Integrates with:
- complete_data_pipeline.py (all external data sources)
- advanced_configuration.py (rate limit configurations)
- advanced_logging.py (monitoring and alerts)
- error_handling.py (circuit breakers and failover)
"""

import asyncio
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum
import heapq
import json
import logging
from abc import ABC, abstractmethod
import statistics
import random

# Enhanced logging and monitoring
try:
    from advanced_logging import get_trading_logger, metrics_collector
    logger = get_trading_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    metrics_collector = None

# Configuration management
try:
    from advanced_configuration import AdvancedConfigurationManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logger.warning("Configuration management not available")

# Error handling integration
try:
    from error_handling import CircuitBreakerManager, TradingSystemError
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False
    logger.warning("Error handling integration not available")


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"


class BackoffStrategy(Enum):
    """Backoff strategies for rate limit violations"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    JITTER = "jitter"
    ADAPTIVE = "adaptive"


class RequestPriority(Enum):
    """Request priority levels"""
    CRITICAL = 1    # Trading decisions, real-time data
    HIGH = 2        # Market data, options chains
    MEDIUM = 3      # News, sentiment analysis
    LOW = 4         # Historical data, batch processing


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a data source"""
    provider: str
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    requests_per_day: int = 100000
    burst_limit: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_factor: float = 2.0
    max_backoff_seconds: float = 300.0
    timeout_seconds: float = 30.0
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


@dataclass
class APIQuota:
    """API quota tracking"""
    provider: str
    daily_limit: int
    daily_used: int = 0
    hourly_limit: int = 3600
    hourly_used: int = 0
    minute_limit: int = 60
    minute_used: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    quota_warnings_sent: List[str] = field(default_factory=list)


@dataclass
class RequestMetrics:
    """Request performance metrics"""
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    avg_response_time_ms: float = 0.0
    last_request_time: datetime = field(default_factory=datetime.now)
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class QueuedRequest:
    """Queued API request"""
    provider: str
    request_func: Callable
    args: tuple
    kwargs: dict
    priority: RequestPriority
    created_at: datetime
    timeout: float
    future: asyncio.Future = None
    
    def __lt__(self, other):
        # For priority queue ordering
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


class RateLimitViolationError(Exception):
    """Rate limit violation error"""
    def __init__(self, provider: str, retry_after: float = None):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for {provider}" + 
                        (f", retry after {retry_after}s" if retry_after else ""))


class TokenBucket:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens based on elapsed time
            new_tokens = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens to be available"""
        with self._lock:
            if self.tokens >= tokens:
                return 0.0
            
            needed_tokens = tokens - self.tokens
            return needed_tokens / self.refill_rate


class SlidingWindowCounter:
    """Sliding window rate limiter implementation"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # seconds
        self.max_requests = max_requests
        self.requests = deque()
        self._lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self._lock:
            now = time.time()
            
            # Remove old requests outside window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def get_wait_time(self) -> float:
        """Get time to wait for next allowed request"""
        with self._lock:
            if len(self.requests) < self.max_requests:
                return 0.0
            
            oldest_request = self.requests[0]
            return oldest_request + self.window_size - time.time()


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on API responses"""
    
    def __init__(self, initial_rate: float, min_rate: float = 0.1, max_rate: float = 10.0):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.success_count = 0
        self.failure_count = 0
        self.last_adjustment = time.time()
        self._lock = threading.Lock()
    
    def record_success(self, response_time_ms: float):
        """Record successful request"""
        with self._lock:
            self.success_count += 1
            
            # Increase rate if we're getting fast responses
            if response_time_ms < 100 and self.success_count > 10:
                self._adjust_rate(1.1)  # 10% increase
    
    def record_failure(self, is_rate_limited: bool = False):
        """Record failed request"""
        with self._lock:
            self.failure_count += 1
            
            if is_rate_limited:
                self._adjust_rate(0.5)  # 50% decrease for rate limits
            else:
                self._adjust_rate(0.9)  # 10% decrease for other failures
    
    def _adjust_rate(self, factor: float):
        """Adjust rate with bounds checking"""
        new_rate = self.current_rate * factor
        self.current_rate = max(self.min_rate, min(self.max_rate, new_rate))
        self.last_adjustment = time.time()
        
        logger.debug(f"Adjusted rate to {self.current_rate:.2f} req/s (factor: {factor})")
    
    def get_current_rate(self) -> float:
        """Get current rate limit"""
        return self.current_rate


class EnhancedRateLimiter:
    """Enhanced rate limiter for a single data source"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.provider = config.provider
        
        # Initialize rate limiting components based on strategy
        self._setup_rate_limiter()
        
        # Metrics and monitoring
        self.metrics = RequestMetrics(provider=config.provider)
        self.quota = APIQuota(
            provider=config.provider,
            daily_limit=config.requests_per_day,
            hourly_limit=config.requests_per_hour,
            minute_limit=config.requests_per_minute
        )
        
        # Circuit breaker integration
        self.circuit_breaker = None
        if ERROR_HANDLING_AVAILABLE:
            self.circuit_breaker = CircuitBreakerManager().get_breaker(
                f"rate_limiter_{config.provider}",
                failure_threshold=config.circuit_breaker_threshold,
                timeout=config.circuit_breaker_timeout
            )
        
        # Backoff state
        self.current_backoff = 0.0
        self.consecutive_failures = 0
        self.last_failure_time = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Enhanced rate limiter initialized for {self.provider}")
    
    def _setup_rate_limiter(self):
        """Setup rate limiter based on strategy"""
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            self.limiter = TokenBucket(
                capacity=self.config.burst_limit,
                refill_rate=self.config.requests_per_second
            )
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            self.limiter = SlidingWindowCounter(
                window_size=60,  # 1 minute window
                max_requests=self.config.requests_per_minute
            )
        elif self.config.strategy == RateLimitStrategy.ADAPTIVE:
            self.limiter = AdaptiveRateLimiter(
                initial_rate=self.config.requests_per_second
            )
        else:
            # Default to sliding window
            self.limiter = SlidingWindowCounter(
                window_size=60,
                max_requests=self.config.requests_per_minute
            )
    
    async def acquire(self, priority: RequestPriority = RequestPriority.MEDIUM) -> bool:
        """Acquire permission to make a request"""
        with self._lock:
            # Check circuit breaker
            if self.circuit_breaker and not self.circuit_breaker.is_closed():
                logger.warning(f"Circuit breaker open for {self.provider}")
                return False
            
            # Check if we're in backoff period
            if self._is_in_backoff():
                wait_time = self._get_backoff_wait_time()
                logger.debug(f"In backoff period for {self.provider}, wait {wait_time:.2f}s")
                return False
            
            # Check quota limits
            if not self._check_quota():
                logger.warning(f"Quota exceeded for {self.provider}")
                return False
            
            # Check rate limit
            if isinstance(self.limiter, TokenBucket):
                if not self.limiter.consume():
                    return False
            elif isinstance(self.limiter, SlidingWindowCounter):
                if not self.limiter.is_allowed():
                    return False
            elif isinstance(self.limiter, AdaptiveRateLimiter):
                # Simple time-based check for adaptive limiter
                if hasattr(self, '_last_request_time'):
                    min_interval = 1.0 / self.limiter.get_current_rate()
                    elapsed = time.time() - self._last_request_time
                    if elapsed < min_interval:
                        return False
                
                self._last_request_time = time.time()
            
            return True
    
    def record_request_success(self, response_time_ms: float):
        """Record successful request"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.response_times.append(response_time_ms)
            
            # Update average response time
            if self.metrics.response_times:
                self.metrics.avg_response_time_ms = statistics.mean(self.metrics.response_times)
            
            # Update quota usage
            self._update_quota_usage()
            
            # Reset backoff on success
            if self.consecutive_failures > 0:
                self.consecutive_failures = 0
                self.current_backoff = 0.0
                logger.debug(f"Reset backoff for {self.provider} after success")
            
            # Record success for adaptive limiter
            if isinstance(self.limiter, AdaptiveRateLimiter):
                self.limiter.record_success(response_time_ms)
            
            # Update metrics collector (if available and has update_metrics method)
            if metrics_collector and hasattr(metrics_collector, 'update_metrics'):
                try:
                    metrics_collector.update_metrics({
                        f"api_requests_total_{self.provider}": self.metrics.total_requests,
                        f"api_success_rate_{self.provider}": self.get_success_rate(),
                        f"api_avg_response_time_{self.provider}": response_time_ms
                    })
                except Exception as e:
                    logger.debug(f"Metrics collector update failed: {e}")
    
    def record_request_failure(self, is_rate_limited: bool = False, error: Exception = None):
        """Record failed request"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            
            if is_rate_limited:
                self.metrics.rate_limited_requests += 1
            
            # Update quota usage (even for failures, some APIs count them)
            self._update_quota_usage()
            
            # Handle backoff
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            self._calculate_backoff()
            
            # Record failure for adaptive limiter
            if isinstance(self.limiter, AdaptiveRateLimiter):
                self.limiter.record_failure(is_rate_limited)
            
            # Circuit breaker handling
            if self.circuit_breaker:
                if is_rate_limited:
                    self.circuit_breaker.record_failure()
                else:
                    self.circuit_breaker.record_failure()
            
            logger.warning(f"Request failure for {self.provider}: rate_limited={is_rate_limited}, "
                          f"consecutive_failures={self.consecutive_failures}")
    
    def _check_quota(self) -> bool:
        """Check if quota limits are exceeded"""
        now = datetime.now()
        
        # Reset counters if needed
        if (now - self.quota.last_reset).total_seconds() >= 86400:  # Daily reset
            self.quota.daily_used = 0
            self.quota.hourly_used = 0
            self.quota.minute_used = 0
            self.quota.last_reset = now
        elif (now - self.quota.last_reset).total_seconds() >= 3600:  # Hourly reset
            self.quota.hourly_used = 0
            self.quota.minute_used = 0
        elif (now - self.quota.last_reset).total_seconds() >= 60:  # Minute reset
            self.quota.minute_used = 0
        
        # Check limits
        if self.quota.daily_used >= self.quota.daily_limit:
            self._send_quota_warning("daily")
            return False
        
        if self.quota.hourly_used >= self.quota.hourly_limit:
            self._send_quota_warning("hourly")
            return False
        
        if self.quota.minute_used >= self.quota.minute_limit:
            return False
        
        return True
    
    def _update_quota_usage(self):
        """Update quota usage counters"""
        self.quota.daily_used += 1
        self.quota.hourly_used += 1
        self.quota.minute_used += 1
    
    def _send_quota_warning(self, period: str):
        """Send quota warning (avoid spam)"""
        warning_key = f"{period}_quota_{datetime.now().strftime('%Y%m%d%H')}"
        
        if warning_key not in self.quota.quota_warnings_sent:
            logger.error(f"Quota limit reached for {self.provider} ({period})")
            self.quota.quota_warnings_sent.append(warning_key)
            
            # Keep only recent warnings
            if len(self.quota.quota_warnings_sent) > 10:
                self.quota.quota_warnings_sent = self.quota.quota_warnings_sent[-5:]
    
    def _is_in_backoff(self) -> bool:
        """Check if we're currently in backoff period"""
        if self.consecutive_failures == 0 or not self.last_failure_time:
            return False
        
        elapsed = time.time() - self.last_failure_time
        return elapsed < self.current_backoff
    
    def _get_backoff_wait_time(self) -> float:
        """Get remaining backoff wait time"""
        if not self._is_in_backoff():
            return 0.0
        
        elapsed = time.time() - self.last_failure_time
        return max(0.0, self.current_backoff - elapsed)
    
    def _calculate_backoff(self):
        """Calculate backoff time based on strategy"""
        if self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            self.current_backoff = min(
                self.config.backoff_factor ** (self.consecutive_failures - 1),
                self.config.max_backoff_seconds
            )
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            self.current_backoff = min(
                self.consecutive_failures * self.config.backoff_factor,
                self.config.max_backoff_seconds
            )
        elif self.config.backoff_strategy == BackoffStrategy.FIBONACCI:
            fib_value = self._fibonacci(self.consecutive_failures)
            self.current_backoff = min(
                fib_value * self.config.backoff_factor,
                self.config.max_backoff_seconds
            )
        elif self.config.backoff_strategy == BackoffStrategy.JITTER:
            base_backoff = self.config.backoff_factor ** (self.consecutive_failures - 1)
            jitter = base_backoff * 0.1 * random.random()  # 10% jitter
            self.current_backoff = min(
                base_backoff + jitter,
                self.config.max_backoff_seconds
            )
        
        logger.debug(f"Calculated backoff for {self.provider}: {self.current_backoff:.2f}s")
    
    def _fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number"""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)
    
    def get_success_rate(self) -> float:
        """Get success rate percentage"""
        if self.metrics.total_requests == 0:
            return 0.0
        return (self.metrics.successful_requests / self.metrics.total_requests) * 100
    
    def get_status(self) -> Dict[str, Any]:
        """Get current limiter status"""
        return {
            "provider": self.provider,
            "strategy": self.config.strategy.value,
            "total_requests": self.metrics.total_requests,
            "success_rate": self.get_success_rate(),
            "avg_response_time_ms": self.metrics.avg_response_time_ms,
            "consecutive_failures": self.consecutive_failures,
            "current_backoff": self.current_backoff,
            "in_backoff": self._is_in_backoff(),
            "quota_usage": {
                "daily": f"{self.quota.daily_used}/{self.quota.daily_limit}",
                "hourly": f"{self.quota.hourly_used}/{self.quota.hourly_limit}",
                "minute": f"{self.quota.minute_used}/{self.quota.minute_limit}"
            },
            "circuit_breaker_open": self.circuit_breaker and not self.circuit_breaker.is_closed() if self.circuit_breaker else False
        }


class CoordinatedRateLimitManager:
    """Centralized rate limit manager for all data sources"""
    
    def __init__(self, config_manager: 'AdvancedConfigurationManager' = None):
        self.config_manager = config_manager
        self.rate_limiters: Dict[str, EnhancedRateLimiter] = {}
        self.request_queue = asyncio.PriorityQueue()
        self.active_requests: Dict[str, int] = defaultdict(int)
        self.global_metrics = defaultdict(int)
        
        # Request processing
        self.queue_processor_running = False
        self.max_concurrent_requests = 50
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize rate limiters from configuration
        self._initialize_rate_limiters()
        
        logger.info("CoordinatedRateLimitManager initialized")
    
    def _initialize_rate_limiters(self):
        """Initialize rate limiters from configuration"""
        # Default configurations for all known data sources
        default_configs = {
            "polygon": RateLimitConfig(
                provider="polygon",
                requests_per_second=5.0,
                requests_per_minute=300,
                requests_per_hour=18000,
                requests_per_day=100000,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                backoff_factor=1.5
            ),
            "finnhub": RateLimitConfig(
                provider="finnhub",
                requests_per_second=1.0,
                requests_per_minute=60,
                requests_per_hour=3600,
                requests_per_day=50000,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                backoff_factor=2.0
            ),
            "alpha_vantage": RateLimitConfig(
                provider="alpha_vantage",
                requests_per_second=0.1,
                requests_per_minute=5,
                requests_per_hour=300,
                requests_per_day=500,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                backoff_strategy=BackoffStrategy.LINEAR,
                backoff_factor=10.0
            ),
            "newsapi": RateLimitConfig(
                provider="newsapi",
                requests_per_second=2.0,
                requests_per_minute=100,
                requests_per_hour=1000,
                requests_per_day=1000,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                backoff_factor=1.5
            ),
            "openweather": RateLimitConfig(
                provider="openweather",
                requests_per_second=1.0,
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=1000,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                backoff_factor=2.0
            ),
            "fred": RateLimitConfig(
                provider="fred",
                requests_per_second=2.0,
                requests_per_minute=120,
                requests_per_hour=1000,
                requests_per_day=100000,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                backoff_factor=1.2
            ),
            "stocktwits": RateLimitConfig(
                provider="stocktwits",
                requests_per_second=3.0,
                requests_per_minute=200,
                requests_per_hour=400,
                requests_per_day=400,
                strategy=RateLimitStrategy.ADAPTIVE,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                backoff_factor=2.0
            ),
            "gnews": RateLimitConfig(
                provider="gnews",
                requests_per_second=0.2,
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=100,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                backoff_factor=3.0
            ),
            "reddit": RateLimitConfig(
                provider="reddit",
                requests_per_second=1.0,
                requests_per_minute=60,
                requests_per_hour=600,
                requests_per_day=1000,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                backoff_factor=2.0
            )
        }
        
        # Override with configuration manager settings if available
        if self.config_manager and CONFIG_AVAILABLE:
            try:
                config = self.config_manager.get_config()
                rate_limits = config.data_sources.rate_limits
                
                for provider, default_config in default_configs.items():
                    if provider in rate_limits:
                        provider_limits = rate_limits[provider]
                        
                        # Update configuration from config manager
                        if hasattr(provider_limits, 'requests_per_minute'):
                            default_config.requests_per_minute = provider_limits.requests_per_minute
                        if hasattr(provider_limits, 'requests_per_hour'):
                            default_config.requests_per_hour = provider_limits.requests_per_hour
                        if hasattr(provider_limits, 'requests_per_day'):
                            default_config.requests_per_day = provider_limits.requests_per_day
                        if hasattr(provider_limits, 'backoff_factor'):
                            default_config.backoff_factor = provider_limits.backoff_factor
                        if hasattr(provider_limits, 'max_retries'):
                            default_config.max_retries = provider_limits.max_retries
                        if hasattr(provider_limits, 'timeout_seconds'):
                            default_config.timeout_seconds = provider_limits.timeout_seconds
                
            except Exception as e:
                logger.warning(f"Failed to load rate limits from configuration: {e}")
        
        # Create rate limiters
        for provider, config in default_configs.items():
            self.rate_limiters[provider] = EnhancedRateLimiter(config)
            logger.debug(f"Rate limiter created for {provider}")
    
    async def execute_request(self, 
                            provider: str,
                            request_func: Callable,
                            priority: RequestPriority = RequestPriority.MEDIUM,
                            timeout: float = None,
                            *args, **kwargs) -> Any:
        """Execute API request with rate limiting"""
        
        if provider not in self.rate_limiters:
            raise ValueError(f"Unknown provider: {provider}")
        
        rate_limiter = self.rate_limiters[provider]
        timeout = timeout or rate_limiter.config.timeout_seconds
        
        # Try to acquire rate limit permission
        if not await rate_limiter.acquire(priority):
            # Queue the request if rate limited
            return await self._queue_request(provider, request_func, priority, timeout, *args, **kwargs)
        
        # Execute request immediately
        return await self._execute_with_monitoring(rate_limiter, request_func, *args, **kwargs)
    
    async def _queue_request(self, 
                           provider: str,
                           request_func: Callable,
                           priority: RequestPriority,
                           timeout: float,
                           *args, **kwargs) -> Any:
        """Queue request for later execution"""
        
        future = asyncio.Future()
        queued_request = QueuedRequest(
            provider=provider,
            request_func=request_func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            created_at=datetime.now(),
            timeout=timeout,
            future=future
        )
        
        await self.request_queue.put((priority.value, time.time(), queued_request))
        
        # Start queue processor if not running
        if not self.queue_processor_running:
            asyncio.create_task(self._process_queue())
        
        logger.debug(f"Queued {provider} request with priority {priority.value}")
        return await future
    
    async def _process_queue(self):
        """Process queued requests"""
        self.queue_processor_running = True
        
        try:
            while not self.request_queue.empty():
                try:
                    # Get highest priority request
                    _, _, queued_request = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=1.0
                    )
                    
                    # Check if request hasn't timed out
                    if (datetime.now() - queued_request.created_at).total_seconds() > queued_request.timeout:
                        queued_request.future.set_exception(asyncio.TimeoutError("Request timed out in queue"))
                        continue
                    
                    # Try to acquire rate limit permission
                    rate_limiter = self.rate_limiters[queued_request.provider]
                    if await rate_limiter.acquire(queued_request.priority):
                        # Execute request
                        try:
                            result = await self._execute_with_monitoring(
                                rate_limiter,
                                queued_request.request_func,
                                *queued_request.args,
                                **queued_request.kwargs
                            )
                            queued_request.future.set_result(result)
                        except Exception as e:
                            queued_request.future.set_exception(e)
                    else:
                        # Put back in queue if still rate limited
                        await self.request_queue.put((
                            queued_request.priority.value,
                            time.time(),
                            queued_request
                        ))
                        
                        # Wait a bit before trying next request
                        await asyncio.sleep(0.1)
                
                except asyncio.TimeoutError:
                    # No more requests in queue
                    break
                except Exception as e:
                    logger.error(f"Error processing request queue: {e}")
                    
        finally:
            self.queue_processor_running = False
    
    async def _execute_with_monitoring(self, 
                                     rate_limiter: EnhancedRateLimiter,
                                     request_func: Callable,
                                     *args, **kwargs) -> Any:
        """Execute request with performance monitoring"""
        
        start_time = time.time()
        provider = rate_limiter.provider
        
        with self._lock:
            self.active_requests[provider] += 1
            self.global_metrics["total_requests"] += 1
        
        try:
            # Execute the request
            if asyncio.iscoroutinefunction(request_func):
                result = await request_func(*args, **kwargs)
            else:
                result = request_func(*args, **kwargs)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Record success
            rate_limiter.record_request_success(response_time_ms)
            
            with self._lock:
                self.global_metrics["successful_requests"] += 1
            
            logger.debug(f"Successful request to {provider}: {response_time_ms:.1f}ms")
            return result
            
        except Exception as e:
            # Determine if it's a rate limit error
            is_rate_limited = self._is_rate_limit_error(e)
            
            # Record failure
            rate_limiter.record_request_failure(is_rate_limited, e)
            
            with self._lock:
                self.global_metrics["failed_requests"] += 1
                if is_rate_limited:
                    self.global_metrics["rate_limited_requests"] += 1
            
            logger.warning(f"Request failed for {provider}: {e}")
            raise
            
        finally:
            with self._lock:
                self.active_requests[provider] -= 1
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Determine if error is due to rate limiting"""
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit", "too many requests", "429", "quota exceeded",
            "throttled", "rate exceeded", "api limit"
        ]
        
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def get_provider_status(self, provider: str) -> Dict[str, Any]:
        """Get status for specific provider"""
        if provider not in self.rate_limiters:
            return {"error": f"Provider {provider} not found"}
        
        return self.rate_limiters[provider].get_status()
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global rate limiting status"""
        provider_stats = {}
        for provider, limiter in self.rate_limiters.items():
            provider_stats[provider] = limiter.get_status()
        
        queue_size = self.request_queue.qsize()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "global_metrics": dict(self.global_metrics),
            "active_requests": dict(self.active_requests),
            "queue_size": queue_size,
            "providers": provider_stats,
            "total_providers": len(self.rate_limiters)
        }
    
    def update_rate_limits(self, provider: str, new_limits: Dict[str, Any]):
        """Update rate limits for a provider"""
        if provider not in self.rate_limiters:
            logger.warning(f"Cannot update limits for unknown provider: {provider}")
            return
        
        rate_limiter = self.rate_limiters[provider]
        config = rate_limiter.config
        
        # Update configuration
        for key, value in new_limits.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Updated {key} for {provider}: {value}")
        
        # Reinitialize limiter with new config
        self.rate_limiters[provider] = EnhancedRateLimiter(config)
        logger.info(f"Rate limiter reinitialized for {provider}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all rate limiters"""
        health_results = {}
        
        for provider, limiter in self.rate_limiters.items():
            status = limiter.get_status()
            
            # Determine health based on success rate and backoff status
            is_healthy = (
                status["success_rate"] > 80 and
                not status["in_backoff"] and
                not status["circuit_breaker_open"]
            )
            
            health_results[provider] = {
                "healthy": is_healthy,
                "success_rate": status["success_rate"],
                "in_backoff": status["in_backoff"],
                "circuit_breaker_open": status["circuit_breaker_open"],
                "total_requests": status["total_requests"]
            }
        
        overall_health = all(result["healthy"] for result in health_results.values())
        
        return {
            "overall_healthy": overall_health,
            "providers": health_results,
            "timestamp": datetime.now().isoformat()
        }


# Factory function
def create_rate_limit_manager(config_manager: 'AdvancedConfigurationManager' = None) -> CoordinatedRateLimitManager:
    """Create rate limit manager with configuration"""
    return CoordinatedRateLimitManager(config_manager)


# Example usage and testing
if __name__ == "__main__":
    async def demo_rate_limiting():
        """Demonstrate enhanced rate limiting system"""
        print("🚦 Enhanced API Rate Limiting System Demo")
        print("=" * 50)
        
        # Create rate limit manager
        rate_manager = create_rate_limit_manager()
        
        # Mock API functions
        async def mock_polygon_request(symbol: str):
            await asyncio.sleep(0.1)  # Simulate API latency
            return f"Polygon data for {symbol}"
        
        def mock_finnhub_request(symbol: str):
            time.sleep(0.05)  # Simulate API latency
            return f"Finnhub data for {symbol}"
        
        # Test requests with different priorities
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        print("\n🔄 Testing coordinated rate limiting...")
        
        # Submit multiple requests
        tasks = []
        for i, symbol in enumerate(symbols):
            # Alternate between providers and priorities
            if i % 2 == 0:
                priority = RequestPriority.HIGH if i < 2 else RequestPriority.MEDIUM
                task = rate_manager.execute_request(
                    "polygon", mock_polygon_request, priority, None, symbol
                )
            else:
                priority = RequestPriority.MEDIUM
                task = rate_manager.execute_request(
                    "finnhub", mock_finnhub_request, priority, None, symbol
                )
            
            tasks.append(task)
        
        # Execute requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"✅ Completed {len([r for r in results if not isinstance(r, Exception)])} requests")
        print(f"❌ Failed {len([r for r in results if isinstance(r, Exception)])} requests")
        
        # Show status
        print(f"\n📊 Global Rate Limiting Status:")
        global_status = rate_manager.get_global_status()
        
        print(f"   Total requests: {global_status['global_metrics']['total_requests']}")
        print(f"   Successful: {global_status['global_metrics']['successful_requests']}")
        print(f"   Queue size: {global_status['queue_size']}")
        
        print(f"\n📈 Provider Status:")
        for provider, status in global_status['providers'].items():
            if status['total_requests'] > 0:
                print(f"   {provider}: {status['success_rate']:.1f}% success rate, "
                      f"{status['total_requests']} requests")
        
        # Health check
        print(f"\n💚 Health Check:")
        health = await rate_manager.health_check()
        print(f"   Overall healthy: {health['overall_healthy']}")
        
        for provider, health_info in health['providers'].items():
            if health_info['total_requests'] > 0:
                status_emoji = "✅" if health_info['healthy'] else "❌"
                print(f"   {status_emoji} {provider}: {health_info['success_rate']:.1f}% success")
        
        print(f"\n🎉 Enhanced Rate Limiting Demo Complete!")
        
    # Run demo
    asyncio.run(demo_rate_limiting())