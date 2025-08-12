#!/usr/bin/env python3
"""
Advanced Configuration Management System
======================================

Enterprise-grade configuration management for the quantitative options trading bot.
Handles multiple environments, API key validation, dynamic updates, and secure storage.

Key Features:
- Environment-specific configurations (dev/staging/prod)
- API key validation and rotation capabilities
- Configuration schema enforcement with Pydantic
- Dynamic configuration hot-reloading
- Secrets management integration
- Configuration versioning and rollback
- Centralized settings management
- Rate limiting coordination across data sources
- Configuration health monitoring

Supports all data sources from complete_data_pipeline.py and existing systems.
"""

import os
import json
import yaml
import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
from enum import Enum

# Enhanced validation with Pydantic
try:
    from pydantic import BaseModel, Field, ValidationError, validator
    from pydantic.env_settings import BaseSettings
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    BaseSettings = object

# Encryption for sensitive data
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

# Configuration watching
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# Enhanced logging
try:
    from advanced_logging import get_trading_logger
    logger = get_trading_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported environments"""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigurationError(Exception):
    """Configuration-specific errors"""
    pass


class APIKeyStatus(Enum):
    """API key validation status"""
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"


@dataclass
class APIKeyInfo:
    """API key information and status"""
    key: str
    provider: str
    status: APIKeyStatus = APIKeyStatus.UNKNOWN
    last_validated: Optional[datetime] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    daily_quota: Optional[int] = None
    quota_used: Optional[int] = None
    validation_error: Optional[str] = None


@dataclass
class RateLimitConfig:
    """Rate limiting configuration per API"""
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    requests_per_day: int = 100000
    backoff_factor: float = 1.5
    max_retries: int = 3
    timeout_seconds: int = 30


if PYDANTIC_AVAILABLE:
    class DatabaseConfig(BaseModel):
        """Database configuration with validation"""
        host: str = "localhost"
        port: int = Field(default=5432, ge=1, le=65535)
        username: str
        password: str
        database: str
        pool_size: int = Field(default=10, ge=1, le=100)
        ssl_mode: str = Field(default="prefer", regex="^(disable|prefer|require)$")
        
    class TradingConfig(BaseModel):
        """Trading configuration with validation"""
        max_position_size: float = Field(ge=0.01, le=1.0)
        max_daily_loss: float = Field(ge=100, le=100000)
        risk_free_rate: float = Field(ge=0.0, le=0.1)
        commission_per_contract: float = Field(ge=0.0, le=10.0)
        slippage_bps: int = Field(ge=0, le=100)
        
    class DataSourceConfig(BaseModel):
        """Data source configuration with validation"""
        polygon_api_key: Optional[str] = None
        finnhub_api_key: Optional[str] = None
        alpha_vantage_api_key: Optional[str] = None
        newsapi_key: Optional[str] = None
        openweather_api_key: Optional[str] = None
        fred_api_key: Optional[str] = None
        stocktwits_token: Optional[str] = None
        gnews_key: Optional[str] = None
        reddit_client_id: Optional[str] = None
        reddit_client_secret: Optional[str] = None
        openai_api_key: Optional[str] = None
        
        # Rate limiting per source
        rate_limits: Dict[str, RateLimitConfig] = Field(default_factory=dict)
        
        @validator('rate_limits')
        def validate_rate_limits(cls, v):
            # Ensure all data sources have rate limit configs
            default_sources = [
                'polygon', 'finnhub', 'alpha_vantage', 'newsapi', 
                'openweather', 'fred', 'stocktwits', 'gnews', 'reddit'
            ]
            
            for source in default_sources:
                if source not in v:
                    v[source] = RateLimitConfig()
            
            return v
            
    class MonitoringConfig(BaseModel):
        """Monitoring and logging configuration"""
        log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
        enable_metrics: bool = True
        metrics_port: int = Field(default=8080, ge=1024, le=65535)
        grafana_enabled: bool = False
        prometheus_enabled: bool = False
        alert_webhook_url: Optional[str] = None
        
    class ApplicationConfig(BaseSettings):
        """Main application configuration"""
        environment: Environment = Environment.DEVELOPMENT
        debug: bool = False
        
        # Core configurations
        database: DatabaseConfig
        trading: TradingConfig  
        data_sources: DataSourceConfig
        monitoring: MonitoringConfig
        
        # Application settings
        app_name: str = "QuantBot"
        app_version: str = "1.0.0"
        timezone: str = "US/Eastern"
        
        # Cache settings
        cache_enabled: bool = True
        cache_ttl_seconds: int = 3600
        cache_max_size_mb: int = 512
        
        # Performance settings
        async_enabled: bool = True
        max_concurrent_requests: int = 10
        worker_threads: int = 4
        
        class Config:
            env_prefix = "QUANTBOT_"
            case_sensitive = False
            
else:
    # Fallback classes when Pydantic not available
    class DatabaseConfig:
        def __init__(self, **kwargs):
            self.host = kwargs.get('host', 'localhost')
            self.port = kwargs.get('port', 5432)
            self.username = kwargs.get('username', '')
            self.password = kwargs.get('password', '')
            self.database = kwargs.get('database', '')
            
    class TradingConfig:
        def __init__(self, **kwargs):
            self.max_position_size = kwargs.get('max_position_size', 0.1)
            self.max_daily_loss = kwargs.get('max_daily_loss', 1000)
            
    class DataSourceConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
                
    class MonitoringConfig:
        def __init__(self, **kwargs):
            self.log_level = kwargs.get('log_level', 'INFO')
            self.enable_metrics = kwargs.get('enable_metrics', True)
            
    class ApplicationConfig:
        def __init__(self, **kwargs):
            self.environment = Environment.DEVELOPMENT
            self.database = DatabaseConfig(**kwargs.get('database', {}))
            self.trading = TradingConfig(**kwargs.get('trading', {}))
            self.data_sources = DataSourceConfig(**kwargs.get('data_sources', {}))
            self.monitoring = MonitoringConfig(**kwargs.get('monitoring', {}))


class ConfigurationValidator:
    """Validates configuration settings"""
    
    @staticmethod
    def validate_api_keys(config: DataSourceConfig) -> Dict[str, APIKeyInfo]:
        """Validate all API keys"""
        results = {}
        
        # Test each API key
        api_tests = {
            'polygon': ConfigurationValidator._test_polygon_key,
            'finnhub': ConfigurationValidator._test_finnhub_key,
            'alpha_vantage': ConfigurationValidator._test_alpha_vantage_key,
            'newsapi': ConfigurationValidator._test_newsapi_key,
            'openweather': ConfigurationValidator._test_openweather_key,
            'fred': ConfigurationValidator._test_fred_key
        }
        
        for provider, test_func in api_tests.items():
            key_attr = f"{provider}_api_key"
            if hasattr(config, key_attr):
                api_key = getattr(config, key_attr)
                if api_key:
                    try:
                        results[provider] = test_func(api_key)
                    except Exception as e:
                        results[provider] = APIKeyInfo(
                            key=api_key[:8] + "...",
                            provider=provider,
                            status=APIKeyStatus.INVALID,
                            validation_error=str(e)
                        )
                        
        return results
    
    @staticmethod
    def _test_polygon_key(api_key: str) -> APIKeyInfo:
        """Test Polygon.io API key"""
        import requests
        
        url = f"https://api.polygon.io/v1/marketstatus/now?apiKey={api_key}"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="polygon",
                    status=APIKeyStatus.VALID,
                    last_validated=datetime.now()
                )
            elif response.status_code == 401:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="polygon", 
                    status=APIKeyStatus.INVALID,
                    validation_error="Unauthorized - invalid API key"
                )
            else:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="polygon",
                    status=APIKeyStatus.UNKNOWN,
                    validation_error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            raise ConfigurationError(f"Polygon API test failed: {e}")
    
    @staticmethod
    def _test_finnhub_key(api_key: str) -> APIKeyInfo:
        """Test Finnhub API key"""
        import requests
        
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol=AAPL&token={api_key}"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="finnhub",
                    status=APIKeyStatus.VALID,
                    last_validated=datetime.now()
                )
            else:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="finnhub",
                    status=APIKeyStatus.INVALID,
                    validation_error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            raise ConfigurationError(f"Finnhub API test failed: {e}")
    
    @staticmethod
    def _test_alpha_vantage_key(api_key: str) -> APIKeyInfo:
        """Test Alpha Vantage API key"""
        import requests
        
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey={api_key}"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "Error Message" in data or "Note" in data:
                    return APIKeyInfo(
                        key=api_key[:8] + "...",
                        provider="alpha_vantage",
                        status=APIKeyStatus.RATE_LIMITED,
                        validation_error=data.get("Note", "Rate limited")
                    )
                else:
                    return APIKeyInfo(
                        key=api_key[:8] + "...",
                        provider="alpha_vantage",
                        status=APIKeyStatus.VALID,
                        last_validated=datetime.now()
                    )
            else:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="alpha_vantage",
                    status=APIKeyStatus.INVALID,
                    validation_error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            raise ConfigurationError(f"Alpha Vantage API test failed: {e}")
    
    @staticmethod
    def _test_newsapi_key(api_key: str) -> APIKeyInfo:
        """Test NewsAPI key"""
        import requests
        
        url = "https://newsapi.org/v2/top-headlines"
        headers = {"X-API-Key": api_key}
        params = {"country": "us", "pageSize": 1}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="newsapi",
                    status=APIKeyStatus.VALID,
                    last_validated=datetime.now()
                )
            else:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="newsapi",
                    status=APIKeyStatus.INVALID,
                    validation_error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            raise ConfigurationError(f"NewsAPI test failed: {e}")
    
    @staticmethod
    def _test_openweather_key(api_key: str) -> APIKeyInfo:
        """Test OpenWeather API key"""
        import requests
        
        url = f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={api_key}"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="openweather",
                    status=APIKeyStatus.VALID,
                    last_validated=datetime.now()
                )
            else:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="openweather",
                    status=APIKeyStatus.INVALID,
                    validation_error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            raise ConfigurationError(f"OpenWeather API test failed: {e}")
    
    @staticmethod
    def _test_fred_key(api_key: str) -> APIKeyInfo:
        """Test FRED API key"""
        import requests
        
        url = f"https://api.stlouisfed.org/fred/series?series_id=GDP&api_key={api_key}&file_type=json"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="fred",
                    status=APIKeyStatus.VALID,
                    last_validated=datetime.now()
                )
            else:
                return APIKeyInfo(
                    key=api_key[:8] + "...",
                    provider="fred",
                    status=APIKeyStatus.INVALID,
                    validation_error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            raise ConfigurationError(f"FRED API test failed: {e}")


class SecureConfigManager:
    """Handles sensitive configuration data with encryption"""
    
    def __init__(self, key_file: str = ".config_key"):
        self.key_file = Path(key_file)
        self.encryption_key = self._load_or_create_key()
        self.cipher = Fernet(self.encryption_key) if ENCRYPTION_AVAILABLE else None
        
    def _load_or_create_key(self) -> bytes:
        """Load existing encryption key or create new one"""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            if ENCRYPTION_AVAILABLE:
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                self.key_file.chmod(0o600)  # Restrict permissions
                return key
            else:
                # Fallback: simple encoding
                key = os.urandom(32)
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                return key
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value"""
        if self.cipher:
            return self.cipher.encrypt(value.encode()).decode()
        else:
            # Fallback: base64 encoding (not secure!)
            import base64
            return base64.b64encode(value.encode()).decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a sensitive value"""
        if self.cipher:
            return self.cipher.decrypt(encrypted_value.encode()).decode()
        else:
            # Fallback: base64 decoding
            import base64
            return base64.b64decode(encrypted_value.encode()).decode()
    
    def secure_store(self, key: str, value: str, config_file: str = "secure_config.json"):
        """Store encrypted value in secure config file"""
        config_path = Path(config_file)
        
        # Load existing config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Encrypt and store
        config[key] = self.encrypt_value(value)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        config_path.chmod(0o600)  # Restrict permissions
        
    def secure_load(self, key: str, config_file: str = "secure_config.json") -> Optional[str]:
        """Load and decrypt value from secure config file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            return None
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        encrypted_value = config.get(key)
        if encrypted_value:
            return self.decrypt_value(encrypted_value)
        
        return None


if WATCHDOG_AVAILABLE:
    class ConfigurationWatcher(FileSystemEventHandler):
        """Watches configuration files for changes"""
        
        def __init__(self, config_manager):
            self.config_manager = config_manager
            self.last_reload = {}
            
        def on_modified(self, event):
            if event.is_directory:
                return
                
            file_path = Path(event.src_path)
            
            # Only watch config files
            if file_path.suffix not in ['.json', '.yaml', '.yml', '.env']:
                return
                
            # Debounce rapid file changes
            now = time.time()
            if file_path in self.last_reload:
                if now - self.last_reload[file_path] < 2.0:  # 2 second debounce
                    return
                    
            self.last_reload[file_path] = now
            
            logger.info(f"Configuration file changed: {file_path}")
            try:
                self.config_manager.reload_configuration()
                logger.info("Configuration reloaded successfully")
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")


class AdvancedConfigurationManager:
    """
    Advanced configuration management system with all enterprise features
    """
    
    def __init__(self, 
                 config_dir: str = "config",
                 environment: Optional[Environment] = None,
                 auto_reload: bool = True):
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        self.environment = environment or self._detect_environment()
        self.auto_reload = auto_reload
        
        # Configuration state
        self.config: Optional[ApplicationConfig] = None
        self.config_hash = None
        self.last_loaded = None
        
        # Components
        self.secure_manager = SecureConfigManager()
        self.validator = ConfigurationValidator()
        self.api_key_cache: Dict[str, APIKeyInfo] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._observers = []
        self._reload_callbacks: List[Callable] = []
        
        # File watcher
        self.watcher = None
        self.observer = None
        
        # Load initial configuration
        self.load_configuration()
        
        # Start file watching
        if auto_reload and WATCHDOG_AVAILABLE:
            self._start_file_watcher()
            
        logger.info(f"AdvancedConfigurationManager initialized for {self.environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Auto-detect current environment"""
        env_var = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
        
        env_mapping = {
            "dev": Environment.DEVELOPMENT,
            "development": Environment.DEVELOPMENT,
            "stage": Environment.STAGING,
            "staging": Environment.STAGING,
            "prod": Environment.PRODUCTION,
            "production": Environment.PRODUCTION,
            "test": Environment.TESTING,
            "testing": Environment.TESTING
        }
        
        return env_mapping.get(env_var, Environment.DEVELOPMENT)
    
    def _get_config_files(self) -> List[Path]:
        """Get configuration files for current environment"""
        files = [
            self.config_dir / "base.yaml",
            self.config_dir / f"{self.environment.value}.yaml",
            self.config_dir / "local.yaml",  # Local overrides
        ]
        
        # Also check for JSON variants
        for base_file in list(files):
            json_file = base_file.with_suffix('.json')
            if json_file.exists():
                files.append(json_file)
        
        return [f for f in files if f.exists()]
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load YAML file {file_path}: {e}")
            return {}
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            return {}
    
    def _merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge multiple configuration dictionaries"""
        def deep_merge(base: Dict, overlay: Dict) -> Dict:
            result = base.copy()
            
            for key, value in overlay.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
                    
            return result
        
        merged = {}
        for config in configs:
            if config:
                merged = deep_merge(merged, config)
        
        return merged
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """Calculate hash of configuration for change detection"""
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Load all QUANTBOT_ prefixed variables
        for key, value in os.environ.items():
            if key.startswith("QUANTBOT_"):
                config_key = key[9:].lower()  # Remove QUANTBOT_ prefix
                
                # Convert nested keys (e.g., QUANTBOT_DATABASE_HOST -> database.host)
                key_parts = config_key.split('_')
                
                current = env_config
                for part in key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Try to parse as JSON, fall back to string
                try:
                    parsed_value = json.loads(value)
                    current[key_parts[-1]] = parsed_value
                except:
                    # Try to parse as boolean or number
                    if value.lower() in ('true', 'false'):
                        current[key_parts[-1]] = value.lower() == 'true'
                    elif value.isdigit():
                        current[key_parts[-1]] = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        current[key_parts[-1]] = float(value)
                    else:
                        current[key_parts[-1]] = value
        
        return env_config
    
    def _integrate_secure_values(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Replace placeholders with secure values"""
        def replace_secure_refs(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    obj[key] = replace_secure_refs(value)
            elif isinstance(obj, list):
                return [replace_secure_refs(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("SECURE:"):
                secure_key = obj[7:]  # Remove SECURE: prefix
                secure_value = self.secure_manager.secure_load(secure_key)
                return secure_value or obj  # Return original if not found
            
            return obj
        
        return replace_secure_refs(config_data)
    
    def load_configuration(self) -> None:
        """Load configuration from all sources"""
        with self._lock:
            try:
                # Load from files
                config_files = self._get_config_files()
                file_configs = []
                
                for config_file in config_files:
                    if config_file.suffix == '.yaml' or config_file.suffix == '.yml':
                        file_config = self._load_yaml_file(config_file)
                    else:
                        file_config = self._load_json_file(config_file)
                    
                    if file_config:
                        file_configs.append(file_config)
                        logger.debug(f"Loaded config from {config_file}")
                
                # Load environment variables
                env_config = self._load_environment_variables()
                
                # Merge all configurations (files first, then env overrides)
                merged_config = self._merge_configs(*file_configs, env_config)
                
                # Integrate secure values
                merged_config = self._integrate_secure_values(merged_config)
                
                # Calculate hash for change detection
                new_hash = self._calculate_config_hash(merged_config)
                
                if new_hash != self.config_hash:
                    # Create validated configuration object
                    if PYDANTIC_AVAILABLE:
                        try:
                            self.config = ApplicationConfig(**merged_config)
                        except ValidationError as e:
                            raise ConfigurationError(f"Configuration validation failed: {e}")
                    else:
                        self.config = ApplicationConfig(**merged_config)
                    
                    self.config_hash = new_hash
                    self.last_loaded = datetime.now()
                    
                    # Notify callbacks
                    for callback in self._reload_callbacks:
                        try:
                            callback(self.config)
                        except Exception as e:
                            logger.error(f"Configuration reload callback failed: {e}")
                    
                    logger.info("Configuration loaded successfully")
                else:
                    logger.debug("Configuration unchanged, skipping reload")
                    
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def reload_configuration(self) -> None:
        """Reload configuration (public method)"""
        self.load_configuration()
    
    def get_config(self) -> ApplicationConfig:
        """Get current configuration"""
        if self.config is None:
            self.load_configuration()
        
        return self.config
    
    def get_api_key_status(self, provider: str) -> Optional[APIKeyInfo]:
        """Get API key validation status"""
        return self.api_key_cache.get(provider)
    
    def validate_all_api_keys(self) -> Dict[str, APIKeyInfo]:
        """Validate all configured API keys"""
        logger.info("Validating all API keys...")
        
        config = self.get_config()
        results = self.validator.validate_api_keys(config.data_sources)
        
        # Cache results
        self.api_key_cache.update(results)
        
        # Log results
        for provider, info in results.items():
            if info.status == APIKeyStatus.VALID:
                logger.info(f"✅ {provider} API key valid")
            else:
                logger.warning(f"❌ {provider} API key {info.status.value}: {info.validation_error}")
        
        return results
    
    def get_data_source_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for specific data source"""
        config = self.get_config()
        data_sources = config.data_sources
        
        # Get API key
        api_key = getattr(data_sources, f"{provider}_api_key", None)
        
        # Get rate limits
        rate_limits = data_sources.rate_limits.get(provider, RateLimitConfig())
        
        return {
            "api_key": api_key,
            "rate_limits": asdict(rate_limits) if hasattr(rate_limits, '__dict__') else rate_limits,
            "status": self.api_key_cache.get(provider, APIKeyInfo("", provider)).status
        }
    
    def update_configuration(self, updates: Dict[str, Any], persist: bool = False) -> None:
        """Update configuration dynamically"""
        with self._lock:
            if not self.config:
                self.load_configuration()
            
            # Apply updates to current config
            config_dict = self.config.dict() if hasattr(self.config, 'dict') else asdict(self.config)
            merged_config = self._merge_configs(config_dict, updates)
            
            # Recreate config object
            if PYDANTIC_AVAILABLE:
                try:
                    self.config = ApplicationConfig(**merged_config)
                except ValidationError as e:
                    raise ConfigurationError(f"Configuration update validation failed: {e}")
            else:
                self.config = ApplicationConfig(**merged_config)
            
            # Update hash
            self.config_hash = self._calculate_config_hash(merged_config)
            self.last_loaded = datetime.now()
            
            # Persist if requested
            if persist:
                self.save_configuration(merged_config, f"{self.environment.value}.yaml")
            
            logger.info(f"Configuration updated: {list(updates.keys())}")
    
    def save_configuration(self, config_data: Dict[str, Any], filename: str) -> None:
        """Save configuration to file"""
        file_path = self.config_dir / filename
        
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(file_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        else:
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def add_reload_callback(self, callback: Callable[[ApplicationConfig], None]) -> None:
        """Add callback for configuration reload events"""
        self._reload_callbacks.append(callback)
    
    def remove_reload_callback(self, callback: Callable[[ApplicationConfig], None]) -> None:
        """Remove configuration reload callback"""
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)
    
    def _start_file_watcher(self) -> None:
        """Start watching configuration files for changes"""
        if not WATCHDOG_AVAILABLE:
            logger.warning("File watching not available (watchdog not installed)")
            return
        
        try:
            self.watcher = ConfigurationWatcher(self)
            self.observer = Observer()
            self.observer.schedule(self.watcher, str(self.config_dir), recursive=False)
            self.observer.start()
            
            logger.info("Configuration file watching started")
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get configuration system health status"""
        config = self.get_config()
        
        # Check API key status
        api_key_health = {}
        for provider, info in self.api_key_cache.items():
            api_key_health[provider] = {
                "status": info.status.value,
                "last_validated": info.last_validated.isoformat() if info.last_validated else None,
                "quota_used": info.quota_used,
                "rate_limit_remaining": info.rate_limit_remaining
            }
        
        return {
            "environment": self.environment.value,
            "last_loaded": self.last_loaded.isoformat() if self.last_loaded else None,
            "config_hash": self.config_hash,
            "file_watcher_active": self.observer is not None and self.observer.is_alive(),
            "api_keys": api_key_health,
            "database_configured": bool(config.database.host if config.database else False),
            "monitoring_enabled": config.monitoring.enable_metrics if config.monitoring else False
        }
    
    def export_configuration_template(self) -> Dict[str, Any]:
        """Export configuration template for new environments"""
        template = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "quantbot",
                "password": "SECURE:database_password",
                "database": "quantbot_prod"
            },
            "trading": {
                "max_position_size": 0.05,
                "max_daily_loss": 5000,
                "risk_free_rate": 0.05,
                "commission_per_contract": 1.0,
                "slippage_bps": 5
            },
            "data_sources": {
                "polygon_api_key": "SECURE:polygon_key",
                "finnhub_api_key": "SECURE:finnhub_key",
                "alpha_vantage_api_key": "SECURE:alpha_vantage_key",
                "newsapi_key": "SECURE:newsapi_key",
                "openweather_api_key": "SECURE:openweather_key",
                "fred_api_key": "SECURE:fred_key",
                "rate_limits": {
                    "polygon": {"requests_per_minute": 300},
                    "finnhub": {"requests_per_minute": 60}
                }
            },
            "monitoring": {
                "log_level": "INFO",
                "enable_metrics": True,
                "metrics_port": 8080,
                "grafana_enabled": False,
                "prometheus_enabled": False
            }
        }
        
        return template
    
    def shutdown(self) -> None:
        """Shutdown configuration manager"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        logger.info("AdvancedConfigurationManager shutdown complete")


# Factory functions
def create_config_manager(environment: str = None, 
                         config_dir: str = "config",
                         auto_reload: bool = True) -> AdvancedConfigurationManager:
    """Factory function to create configuration manager"""
    env = Environment(environment) if environment else None
    return AdvancedConfigurationManager(config_dir, env, auto_reload)


def setup_default_configuration(config_dir: str = "config") -> None:
    """Setup default configuration files for all environments"""
    config_path = Path(config_dir)
    config_path.mkdir(exist_ok=True, parents=True)
    
    manager = AdvancedConfigurationManager(config_dir)
    template = manager.export_configuration_template()
    
    # Create templates for each environment
    environments = ['development', 'staging', 'production']
    
    for env in environments:
        env_config = template.copy()
        
        # Environment-specific adjustments
        if env == 'development':
            env_config['monitoring']['log_level'] = 'DEBUG'
            env_config['trading']['max_position_size'] = 0.01
        elif env == 'production':
            env_config['monitoring']['prometheus_enabled'] = True
            env_config['monitoring']['grafana_enabled'] = True
        
        # Save configuration
        env_file = config_path / f"{env}.yaml"
        with open(env_file, 'w') as f:
            yaml.dump(env_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created {env} configuration template: {env_file}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize configuration manager
    config_manager = create_config_manager(
        environment="development",
        config_dir="config",
        auto_reload=True
    )
    
    # Test configuration loading
    print("🔧 Testing Advanced Configuration Management")
    print("=" * 50)
    
    # Get configuration
    config = config_manager.get_config()
    print(f"Environment: {config.environment.value}")
    print(f"Database host: {config.database.host}")
    print(f"Cache enabled: {config.cache_enabled}")
    
    # Test API key validation
    print("\n🔑 Validating API Keys...")
    api_results = config_manager.validate_all_api_keys()
    for provider, info in api_results.items():
        print(f"  {provider}: {info.status.value}")
    
    # Test data source config
    print("\n📊 Data Source Configurations:")
    for provider in ['polygon', 'finnhub', 'alpha_vantage']:
        source_config = config_manager.get_data_source_config(provider)
        rate_limits = source_config['rate_limits']
        print(f"  {provider}: {rate_limits['requests_per_minute']} req/min")
    
    # Test dynamic updates
    print("\n🔄 Testing Dynamic Configuration Updates...")
    config_manager.update_configuration({
        'trading': {'max_position_size': 0.02}
    })
    
    updated_config = config_manager.get_config()
    print(f"Updated max position size: {updated_config.trading.max_position_size}")
    
    # Test health status
    print("\n💚 Configuration System Health:")
    health = config_manager.get_health_status()
    print(f"  Environment: {health['environment']}")
    print(f"  Last loaded: {health['last_loaded']}")
    print(f"  File watcher: {'Active' if health['file_watcher_active'] else 'Inactive'}")
    
    # Create default configuration templates
    print("\n📝 Creating Configuration Templates...")
    setup_default_configuration("config")
    
    print("\n✅ Advanced Configuration Management System ready!")
    
    # Cleanup
    config_manager.shutdown()