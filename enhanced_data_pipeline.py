"""
Enhanced Data Pipeline with Professional Error Handling
======================================================

This module wraps the existing unified data pipeline with the enhanced error handling system,
providing production-grade resilience for live trading operations.

Features:
- Automatic retry for transient failures
- Circuit breakers for external API protection  
- Graceful degradation when data sources fail
- Comprehensive error monitoring and alerting
- Recovery strategies specific to trading operations
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime, timedelta

from data_pipeline import UnifiedDataPipeline, DataPipelineError
from error_handling import (
    error_handler, 
    with_retry, 
    with_circuit_breaker, 
    trading_operation,
    DataFeedError,
    MarketDataError,
    NetworkError,
    ExternalAPIError,
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy
)
import logging

logger = logging.getLogger(__name__)


class EnhancedDataPipeline(UnifiedDataPipeline):
    """Enhanced data pipeline with professional error handling and resilience"""
    
    def __init__(self, config):
        super().__init__(config)
        self.error_handler = error_handler
        logger.info("EnhancedDataPipeline initialized with error handling")
    
    @with_circuit_breaker("polygon_api")
    @with_retry("data_feed")
    @trading_operation(
        category=ErrorCategory.DATA_FEED,
        severity=ErrorSeverity.HIGH,
        recovery=RecoveryStrategy.FALLBACK
    )
    def get_close_series(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.Series:
        """
        Get historical close prices with enhanced error handling
        
        Includes:
        - Circuit breaker protection for API stability
        - Automatic retry for transient failures
        - Graceful fallback to alternative sources
        - Data quality validation with error alerts
        """
        try:
            # Call parent method with comprehensive error context
            with self.error_handler.handle_errors(
                operation=f"get_close_series_{symbol}",
                category=ErrorCategory.MARKET_DATA,
                severity=ErrorSeverity.MEDIUM,
                context={"symbol": symbol, "start": start, "end": end}
            ):
                result = super().get_close_series(symbol, start, end)
                
                # Enhanced validation
                if result.empty:
                    raise MarketDataError(
                        f"No price data available for {symbol}",
                        symbol=symbol,
                        severity=ErrorSeverity.HIGH,
                        context={"date_range": f"{start} to {end}"}
                    )
                
                # Data quality checks
                self._validate_price_series(result, symbol)
                
                return result
                
        except DataPipelineError as e:
            # Convert to enhanced error with more context
            raise DataFeedError(
                f"Data pipeline error for {symbol}: {str(e)}",
                source="unified_pipeline",
                severity=ErrorSeverity.HIGH,
                context={"symbol": symbol, "original_error": str(e)}
            )
    
    def _validate_price_series(self, data: pd.Series, symbol: str):
        """Enhanced price data validation with specific error handling"""
        if data.empty:
            return  # Already handled above
        
        # Check for reasonable price ranges
        if (data <= 0).any():
            raise MarketDataError(
                f"Invalid price data for {symbol}: non-positive values detected",
                symbol=symbol,
                severity=ErrorSeverity.HIGH
            )
        
        # Check for extreme price movements
        price_changes = data.pct_change().abs()
        max_change = price_changes.max()
        
        if max_change > 0.5:  # 50% single-day change
            logger.warning(f"Extreme price movement detected for {symbol}: {max_change:.1%}")
            # Don't raise error, but log for monitoring
        
        # Check data recency
        latest_date = data.index[-1]
        if hasattr(latest_date, 'date'):
            days_old = (datetime.now().date() - latest_date.date()).days
        else:
            days_old = (datetime.now().date() - latest_date).days
            
        if days_old > 7:
            logger.warning(f"Stale data for {symbol}: {days_old} days old")
    
    @with_retry("data_feed")
    @trading_operation(
        category=ErrorCategory.DATA_FEED,
        severity=ErrorSeverity.MEDIUM,
        recovery=RecoveryStrategy.FALLBACK
    )
    def fetch_equity_prices(self, symbol: str) -> pd.DataFrame:
        """Enhanced equity price fetching with error handling"""
        try:
            with self.error_handler.handle_errors(
                operation=f"fetch_equity_prices_{symbol}",
                category=ErrorCategory.MARKET_DATA,
                context={"symbol": symbol}
            ):
                result = super().fetch_equity_prices(symbol)
                
                if result.empty:
                    raise MarketDataError(
                        f"No equity price data available for {symbol}",
                        symbol=symbol,
                        severity=ErrorSeverity.MEDIUM
                    )
                
                # Validate OHLCV data
                self._validate_equity_data(result, symbol)
                
                return result
                
        except Exception as e:
            if isinstance(e, (MarketDataError, DataFeedError)):
                raise
            
            # Convert unexpected errors
            raise DataFeedError(
                f"Unexpected error fetching equity data for {symbol}: {str(e)}",
                source="equity_fetcher",
                severity=ErrorSeverity.HIGH,
                original_exception=e
            )
    
    def _validate_equity_data(self, data: pd.DataFrame, symbol: str):
        """Validate OHLCV equity data"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise MarketDataError(
                f"Missing columns in equity data for {symbol}: {missing_columns}",
                symbol=symbol,
                severity=ErrorSeverity.HIGH,
                context={"missing_columns": missing_columns}
            )
        
        # Validate OHLC relationships
        invalid_ohlc = ((data['high'] < data['low']) | 
                       (data['high'] < data['open']) | 
                       (data['high'] < data['close']) |
                       (data['low'] > data['open']) |
                       (data['low'] > data['close'])).any()
        
        if invalid_ohlc:
            raise MarketDataError(
                f"Invalid OHLC relationships in data for {symbol}",
                symbol=symbol,
                severity=ErrorSeverity.HIGH
            )
    
    @with_retry("data_feed")
    @trading_operation(
        category=ErrorCategory.DATA_FEED,
        severity=ErrorSeverity.MEDIUM,
        recovery=RecoveryStrategy.FALLBACK
    )
    def fetch_options_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Enhanced options chain fetching with error handling"""
        try:
            with self.error_handler.handle_errors(
                operation=f"fetch_options_chain_{symbol}",
                category=ErrorCategory.MARKET_DATA,
                context={"symbol": symbol, "expiry": expiry}
            ):
                result = super().fetch_options_chain(symbol, expiry)
                
                if result.empty:
                    logger.warning(f"No options data available for {symbol}, using mock data")
                    # Don't raise error - options data is often unavailable
                else:
                    self._validate_options_data(result, symbol)
                
                return result
                
        except Exception as e:
            if isinstance(e, MarketDataError):
                raise
                
            # Log but don't fail for options data - it's often unavailable
            logger.warning(f"Options data fetch failed for {symbol}: {str(e)}")
            return super()._generate_mock_options_data(symbol)
    
    def _validate_options_data(self, data: pd.DataFrame, symbol: str):
        """Validate options chain data"""
        required_columns = ['strike', 'bid', 'ask', 'type']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing optional columns in options data for {symbol}: {missing_columns}")
            # Don't raise error for options - data formats vary
        
        # Validate bid-ask spreads
        if 'bid' in data.columns and 'ask' in data.columns:
            invalid_spreads = (data['ask'] < data['bid']).any()
            if invalid_spreads:
                logger.warning(f"Invalid bid-ask spreads detected in options data for {symbol}")
    
    @with_retry("api_calls")
    @trading_operation(
        category=ErrorCategory.DATA_FEED,
        severity=ErrorSeverity.LOW,
        recovery=RecoveryStrategy.SKIP
    )
    def fetch_macro_data(self) -> pd.DataFrame:
        """Enhanced macro data fetching with error handling"""
        try:
            with self.error_handler.handle_errors(
                operation="fetch_macro_data",
                category=ErrorCategory.MARKET_DATA,
                severity=ErrorSeverity.LOW,  # Macro data is nice-to-have
                recovery_strategy=RecoveryStrategy.SKIP
            ):
                result = super().fetch_macro_data()
                
                # Log macro data status
                non_null_count = sum(1 for col in result.columns 
                                   if col != 'timestamp' and result[col].iloc[0] is not None)
                total_indicators = len(result.columns) - 1  # Exclude timestamp
                
                logger.info(f"Macro data fetched: {non_null_count}/{total_indicators} indicators available")
                
                return result
                
        except Exception as e:
            # Macro data is optional - don't fail the system
            logger.warning(f"Macro data fetch failed: {str(e)}")
            # Return empty DataFrame with timestamp
            return pd.DataFrame([{"timestamp": datetime.now()}])
    
    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with error handling metrics"""
        # Get base health check
        base_health = super().health_check()
        
        # Add error handling metrics
        error_health = self.error_handler.monitor.get_health_summary()
        
        # Combine results
        enhanced_health = {
            **base_health,
            "error_handling": error_health,
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "last_failure": cb.last_failure_time
                }
                for name, cb in self.error_handler.circuit_breakers.items()
            }
        }
        
        # Determine overall status
        error_status = error_health["status"]
        base_status = base_health["overall_status"]
        
        if error_status == "critical" or base_status == "critical":
            enhanced_health["overall_status"] = "critical"
        elif error_status == "degraded" or base_status == "degraded":
            enhanced_health["overall_status"] = "degraded"
        elif error_status == "warning":
            enhanced_health["overall_status"] = "warning"
        else:
            enhanced_health["overall_status"] = "healthy"
        
        return enhanced_health
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary for monitoring"""
        from error_handling import get_error_statistics
        return get_error_statistics()
    
    def reset_circuit_breakers(self):
        """Reset all circuit breakers (useful for recovery operations)"""
        for cb in self.error_handler.circuit_breakers.values():
            cb.state = cb.CircuitState.CLOSED
            cb.failure_count = 0
            cb.success_count = 0
        logger.info("All circuit breakers reset to CLOSED state")


# Convenience function to create enhanced pipeline
def create_enhanced_pipeline(config) -> EnhancedDataPipeline:
    """Factory function to create enhanced data pipeline"""
    return EnhancedDataPipeline(config)