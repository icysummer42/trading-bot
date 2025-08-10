"""
Advanced Logging & Monitoring System
===================================

Professional logging and monitoring system for quantitative options trading.
Designed for 24/7 production operations with real-time metrics, structured logging,
and comprehensive alerting capabilities.

Features:
- Structured JSON logging with trading-specific fields
- Performance metrics collection and export
- Real-time system health monitoring
- Integration with Grafana/Prometheus
- Alert management system
- Trading-specific KPIs and dashboards
- Compliance-ready audit logging
"""

from __future__ import annotations
import os
import sys
import json
import time
import threading
import logging
import logging.handlers
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
from pathlib import Path
import psutil
import socket

# Metrics collection
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠️  Prometheus client not available. Install with: pip install prometheus-client")


class LogLevel(Enum):
    """Enhanced log levels for trading systems"""
    TRACE = "TRACE"      # Detailed execution flow
    DEBUG = "DEBUG"      # Development debugging
    INFO = "INFO"        # General information
    BUSINESS = "BUSINESS" # Business logic events (trades, signals)
    WARNING = "WARNING"   # Warning conditions
    ERROR = "ERROR"      # Error conditions
    CRITICAL = "CRITICAL" # Critical system failures
    AUDIT = "AUDIT"      # Compliance/audit events


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"       # Monotonically increasing (errors, trades)
    GAUGE = "gauge"          # Point-in-time value (prices, positions)
    HISTOGRAM = "histogram"   # Distribution (latency, execution time)
    TIMER = "timer"          # Timing measurements


@dataclass
class TradingLogRecord:
    """Structured log record for trading operations"""
    timestamp: str
    level: str
    logger: str
    message: str
    module: str
    function: str
    line_number: int
    
    # Trading-specific fields
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    position_id: Optional[str] = None
    order_id: Optional[str] = None
    trade_value: Optional[float] = None
    pnl: Optional[float] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    
    # System context
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Performance metrics  
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Error context
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class TradingFormatter(logging.Formatter):
    """Custom formatter for trading system logs"""
    
    def __init__(self):
        super().__init__()
        
    def format(self, record):
        """Format log record as structured JSON"""
        # Get basic record info
        log_record = TradingLogRecord(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            module=record.module if hasattr(record, 'module') else 'unknown',
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process
        )
        
        # Add trading-specific context if available
        for attr in ['symbol', 'strategy', 'position_id', 'order_id', 'trade_value', 
                    'pnl', 'execution_time_ms', 'error_type', 'correlation_id']:
            if hasattr(record, attr):
                setattr(log_record, attr, getattr(record, attr))
        
        # Add exception info if present
        if record.exc_info:
            import traceback
            log_record.stack_trace = traceback.format_exception(*record.exc_info)
        
        return json.dumps(log_record.to_dict(), default=str)


class MetricsCollector:
    """Collects and exports system and trading metrics"""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics = {}
        self.custom_registry = CollectorRegistry() if self.enable_prometheus else None
        
        if self.enable_prometheus:
            self._setup_prometheus_metrics()
        
        # System metrics collection
        self._start_system_metrics_collection()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        if not self.enable_prometheus:
            return
            
        # Trading metrics
        self.metrics.update({
            'trades_total': Counter('trading_trades_total', 'Total number of trades', 
                                  ['symbol', 'strategy', 'side'], registry=self.custom_registry),
            'trade_pnl': Histogram('trading_trade_pnl', 'Trade P&L distribution',
                                 ['symbol', 'strategy'], registry=self.custom_registry),
            'positions_open': Gauge('trading_positions_open', 'Number of open positions',
                                  ['symbol', 'strategy'], registry=self.custom_registry),
            'portfolio_value': Gauge('trading_portfolio_value', 'Total portfolio value',
                                   registry=self.custom_registry),
            
            # System metrics
            'api_requests_total': Counter('api_requests_total', 'API requests',
                                        ['endpoint', 'status'], registry=self.custom_registry),
            'api_request_duration': Histogram('api_request_duration_seconds', 'API request duration',
                                            ['endpoint'], registry=self.custom_registry),
            'error_count': Counter('system_errors_total', 'System errors',
                                 ['component', 'error_type'], registry=self.custom_registry),
            'system_health': Gauge('system_health_score', 'System health (0-100)',
                                 registry=self.custom_registry),
            
            # Performance metrics
            'cpu_usage': Gauge('system_cpu_usage_percent', 'CPU usage percentage',
                             registry=self.custom_registry),
            'memory_usage': Gauge('system_memory_usage_mb', 'Memory usage in MB',
                                registry=self.custom_registry),
            'disk_usage': Gauge('system_disk_usage_percent', 'Disk usage percentage',
                              registry=self.custom_registry)
        })
    
    def _start_system_metrics_collection(self):
        """Start background thread for system metrics collection"""
        def collect_system_metrics():
            while True:
                try:
                    if self.enable_prometheus:
                        # CPU usage
                        cpu_percent = psutil.cpu_percent(interval=1)
                        self.metrics['cpu_usage'].set(cpu_percent)
                        
                        # Memory usage
                        memory = psutil.virtual_memory()
                        self.metrics['memory_usage'].set(memory.used / 1024 / 1024)  # MB
                        
                        # Disk usage
                        disk = psutil.disk_usage('/')
                        disk_percent = (disk.used / disk.total) * 100
                        self.metrics['disk_usage'].set(disk_percent)
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    logging.getLogger(__name__).error(f"Error collecting system metrics: {e}")
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def record_trade(self, symbol: str, strategy: str, side: str, pnl: float):
        """Record a trade execution"""
        if self.enable_prometheus:
            self.metrics['trades_total'].labels(symbol=symbol, strategy=strategy, side=side).inc()
            self.metrics['trade_pnl'].labels(symbol=symbol, strategy=strategy).observe(pnl)
    
    def update_positions(self, positions_by_symbol: Dict[str, int]):
        """Update open positions count"""
        if self.enable_prometheus:
            for symbol, count in positions_by_symbol.items():
                # Extract strategy from first position (simplified)
                strategy = "multi"  # Could be enhanced to track per strategy
                self.metrics['positions_open'].labels(symbol=symbol, strategy=strategy).set(count)
    
    def update_portfolio_value(self, value: float):
        """Update total portfolio value"""
        if self.enable_prometheus:
            self.metrics['portfolio_value'].set(value)
    
    def record_api_request(self, endpoint: str, status: str, duration: float):
        """Record API request metrics"""
        if self.enable_prometheus:
            self.metrics['api_requests_total'].labels(endpoint=endpoint, status=status).inc()
            self.metrics['api_request_duration'].labels(endpoint=endpoint).observe(duration)
    
    def record_error(self, component: str, error_type: str):
        """Record system error"""
        if self.enable_prometheus:
            self.metrics['error_count'].labels(component=component, error_type=error_type).inc()
    
    def update_health_score(self, score: float):
        """Update system health score (0-100)"""
        if self.enable_prometheus:
            self.metrics['system_health'].set(score)
    
    def export_to_prometheus_gateway(self, gateway_url: str, job_name: str = 'trading_bot'):
        """Export metrics to Prometheus Push Gateway"""
        if self.enable_prometheus:
            try:
                push_to_gateway(gateway_url, job=job_name, registry=self.custom_registry)
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to push metrics to gateway: {e}")
    
    def start_prometheus_server(self, port: int = 8000):
        """Start Prometheus metrics server"""
        if self.enable_prometheus:
            try:
                start_http_server(port, registry=self.custom_registry)
                logging.getLogger(__name__).info(f"Prometheus metrics server started on port {port}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to start Prometheus server: {e}")


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # Python expression
    severity: str   # info, warning, critical
    message: str
    cooldown_seconds: int = 300  # 5 minutes
    enabled: bool = True
    last_triggered: Optional[datetime] = None


class AlertManager:
    """Manages alerts and notifications for trading system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.rules: Dict[str, AlertRule] = {}
        self.alert_channels = []
        self.metrics_cache = {}
        
        if config_path and Path(config_path).exists():
            self._load_alert_rules(config_path)
        else:
            self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules for trading system"""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                condition="error_rate_per_minute > 10",
                severity="warning",
                message="High error rate detected: {error_rate_per_minute} errors/minute"
            ),
            AlertRule(
                name="system_unhealthy",
                condition="system_health_score < 50",
                severity="critical",
                message="System health critical: {system_health_score}% health score"
            ),
            AlertRule(
                name="high_cpu_usage",
                condition="cpu_usage > 80",
                severity="warning",
                message="High CPU usage: {cpu_usage}%"
            ),
            AlertRule(
                name="high_memory_usage", 
                condition="memory_usage_percent > 85",
                severity="warning",
                message="High memory usage: {memory_usage_percent}%"
            ),
            AlertRule(
                name="api_down",
                condition="api_success_rate < 0.5",
                severity="critical",
                message="API success rate critical: {api_success_rate:.1%}"
            ),
            AlertRule(
                name="large_portfolio_drawdown",
                condition="portfolio_drawdown_percent > 10",
                severity="critical",
                message="Large portfolio drawdown: {portfolio_drawdown_percent:.1f}%"
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.name] = rule
    
    def _load_alert_rules(self, config_path: str):
        """Load alert rules from configuration file"""
        try:
            with open(config_path, 'r') as f:
                rules_data = json.load(f)
                
            for rule_data in rules_data.get('alert_rules', []):
                rule = AlertRule(**rule_data)
                self.rules[rule.name] = rule
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load alert rules: {e}")
            self._setup_default_rules()
    
    def add_alert_channel(self, channel):
        """Add alert delivery channel (email, slack, etc.)"""
        self.alert_channels.append(channel)
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics cache for alert evaluation"""
        self.metrics_cache.update(metrics)
        self._evaluate_alerts()
    
    def _evaluate_alerts(self):
        """Evaluate all alert rules against current metrics"""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
                
            # Check cooldown
            if (rule.last_triggered and 
                (datetime.now() - rule.last_triggered).total_seconds() < rule.cooldown_seconds):
                continue
            
            try:
                # Evaluate condition
                if eval(rule.condition, {"__builtins__": {}}, self.metrics_cache):
                    self._trigger_alert(rule)
                    
            except Exception as e:
                logging.getLogger(__name__).error(f"Error evaluating alert rule {rule_name}: {e}")
    
    def _trigger_alert(self, rule: AlertRule):
        """Trigger alert and send notifications"""
        rule.last_triggered = datetime.now()
        
        # Format message with current metrics
        try:
            formatted_message = rule.message.format(**self.metrics_cache)
        except:
            formatted_message = rule.message
        
        alert_data = {
            'rule_name': rule.name,
            'severity': rule.severity,
            'message': formatted_message,
            'timestamp': rule.last_triggered.isoformat(),
            'metrics': self.metrics_cache.copy()
        }
        
        # Log alert
        logger = logging.getLogger(__name__)
        if rule.severity == 'critical':
            logger.critical(f"ALERT: {formatted_message}")
        elif rule.severity == 'warning':
            logger.warning(f"ALERT: {formatted_message}")
        else:
            logger.info(f"ALERT: {formatted_message}")
        
        # Send to alert channels
        for channel in self.alert_channels:
            try:
                channel.send_alert(alert_data)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.__class__.__name__}: {e}")


class TradingLogger:
    """Enhanced logger for trading systems with structured logging"""
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Add custom log level for business events
        logging.addLevelName(25, "BUSINESS")
        logging.addLevelName(35, "AUDIT")
        
        self._setup_handlers()
        
        # Context tracking
        self.context = {}
        self._local = threading.local()
    
    def _setup_handlers(self):
        """Setup log handlers"""
        if self.logger.handlers:
            return  # Already setup
        
        formatter = TradingFormatter()
        
        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for production
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "trading_system.jsonl",
            maxBytes=100*1024*1024,  # 100MB
            backupCount=10
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Separate file for business events
        business_handler = logging.handlers.RotatingFileHandler(
            log_dir / "business_events.jsonl",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=20
        )
        business_handler.setFormatter(formatter)
        business_handler.setLevel(25)  # BUSINESS level
        self.logger.addHandler(business_handler)
    
    def set_context(self, **kwargs):
        """Set logging context for current thread"""
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        self._local.context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        if hasattr(self._local, 'context'):
            self._local.context = {}
    
    def _log_with_context(self, level, msg, **kwargs):
        """Log message with context"""
        extra = kwargs.copy()
        
        # Add thread-local context
        if hasattr(self._local, 'context'):
            extra.update(self._local.context)
        
        # Add global context
        extra.update(self.context)
        
        self.logger.log(level, msg, extra=extra)
    
    def trace(self, msg, **kwargs):
        """Trace level logging"""
        self._log_with_context(5, msg, **kwargs)
    
    def debug(self, msg, **kwargs):
        """Debug level logging"""
        self._log_with_context(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg, **kwargs):
        """Info level logging"""
        self._log_with_context(logging.INFO, msg, **kwargs)
    
    def business(self, msg, **kwargs):
        """Business event logging"""
        self._log_with_context(25, msg, **kwargs)
    
    def warning(self, msg, **kwargs):
        """Warning level logging"""
        self._log_with_context(logging.WARNING, msg, **kwargs)
    
    def error(self, msg, **kwargs):
        """Error level logging"""
        self._log_with_context(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg, **kwargs):
        """Critical level logging"""
        self._log_with_context(logging.CRITICAL, msg, **kwargs)
    
    def audit(self, msg, **kwargs):
        """Audit level logging"""
        self._log_with_context(35, msg, **kwargs)
    
    @contextmanager
    def operation_context(self, operation: str, **context):
        """Context manager for operation logging"""
        start_time = time.time()
        operation_id = f"{operation}_{int(time.time()*1000)}"
        
        # Set context
        old_context = getattr(self._local, 'context', {}).copy()
        self.set_context(operation=operation, operation_id=operation_id, **context)
        
        try:
            self.info(f"Started {operation}", operation_start=True)
            yield operation_id
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.error(f"Failed {operation}: {e}", 
                      operation_failed=True,
                      execution_time_ms=execution_time,
                      error_type=type(e).__name__)
            raise
            
        finally:
            execution_time = (time.time() - start_time) * 1000
            self.info(f"Completed {operation}",
                     operation_completed=True,
                     execution_time_ms=execution_time)
            
            # Restore old context
            if hasattr(self._local, 'context'):
                self._local.context = old_context


# Global instances
metrics_collector = MetricsCollector()
alert_manager = AlertManager()

# Factory function for getting logger
def get_trading_logger(name: str) -> TradingLogger:
    """Get trading logger instance"""
    return TradingLogger(name)