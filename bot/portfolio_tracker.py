"""
Real-time Portfolio Position Tracker

Core portfolio tracking system that integrates with the trading engine
to provide real-time position monitoring, P&L calculation, and risk analytics.
"""

from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import pickle
import threading
import time
from pathlib import Path

from bot.enhanced_engine import ExecutionEngine, Position
from bot.risk_manager import AdvancedRiskManager
from bot.greeks import GreeksCalculator
from bot.polygon_client import PolygonClient
from logger import get_logger

logger = get_logger("portfolio_tracker")

@dataclass
class PositionUpdate:
    """Real-time position update."""
    timestamp: dt.datetime
    symbol: str
    strategy: str
    current_price: float
    unrealized_pnl: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    days_to_expiry: int
    position_value: float

@dataclass
class PortfolioSnapshot:
    """Complete portfolio snapshot at a point in time."""
    timestamp: dt.datetime
    total_value: float
    unrealized_pnl: float
    positions: List[PositionUpdate]
    portfolio_greeks: Dict[str, float]
    risk_metrics: Dict[str, float]
    alerts: List[Dict[str, Any]]

@dataclass
class RiskAlert:
    """Risk management alert."""
    timestamp: dt.datetime
    level: str  # 'info', 'warning', 'error'
    category: str  # 'position', 'portfolio', 'greek', 'limit'
    message: str
    symbol: Optional[str] = None
    current_value: Optional[float] = None
    limit_value: Optional[float] = None

class PortfolioTracker:
    """
    Real-time portfolio tracking system.
    
    Provides:
    - Real-time position monitoring
    - P&L calculation and tracking
    - Greeks exposure analysis
    - Risk limit monitoring
    - Alert generation
    - Historical performance tracking
    """
    
    def __init__(self, 
                 execution_engine: ExecutionEngine,
                 risk_manager: AdvancedRiskManager,
                 polygon_client: PolygonClient = None,
                 data_dir: str = "portfolio_data"):
        """
        Initialize portfolio tracker.
        
        Args:
            execution_engine: Trading execution engine
            risk_manager: Risk management system
            polygon_client: Market data client
            data_dir: Directory for storing portfolio data
        """
        self.execution_engine = execution_engine
        self.risk_manager = risk_manager
        self.polygon_client = polygon_client
        self.greeks_calc = GreeksCalculator()
        
        # Data storage
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Current state
        self.current_positions: Dict[str, Position] = {}
        self.position_updates: List[PositionUpdate] = []
        self.portfolio_snapshots: List[PortfolioSnapshot] = []
        self.active_alerts: List[RiskAlert] = []
        
        # Risk limits (configurable)
        self.risk_limits = {
            'max_position_var': 0.05,      # 5% VaR per position
            'max_portfolio_var': 0.03,     # 3% portfolio VaR
            'max_delta_exposure': 0.50,    # 50% delta exposure
            'max_gamma_exposure': 1.00,    # 100% gamma exposure
            'max_vega_exposure': 1000,     # $1000 vega exposure
            'max_theta_decay': -500,       # -$500 daily theta
            'max_position_size': 0.10,     # 10% of portfolio per position
            'max_concentration': 0.40,     # 40% max concentration
            'min_liquidity_days': 5,       # Min 5 days to expiry
        }
        
        # Performance tracking
        self.daily_pnl: List[Tuple[dt.date, float]] = []
        self.portfolio_value_history: List[Tuple[dt.datetime, float]] = []
        
        # Threading for real-time updates
        self.update_interval = 30  # seconds
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Load existing data
        self._load_historical_data()
    
    def start_monitoring(self):
        """Start real-time portfolio monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started real-time portfolio monitoring")
    
    def stop_monitoring(self):
        """Stop real-time portfolio monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped portfolio monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self.update_positions()
                self._check_risk_limits()
                self._generate_portfolio_snapshot()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def update_positions(self):
        """Update all position valuations and Greeks."""
        current_time = dt.datetime.now()
        
        # Get current positions from execution engine
        portfolio_summary = self.execution_engine.get_portfolio_summary()
        positions = portfolio_summary.get('positions', {})
        
        updated_positions = []
        
        for symbol, position_data in positions.items():
            try:
                # Get current market price
                current_price = self._get_current_price(symbol)
                if current_price is None:
                    continue
                
                # Calculate position update
                position_update = self._calculate_position_update(
                    symbol, position_data, current_price, current_time
                )
                
                if position_update:
                    updated_positions.append(position_update)
                    
            except Exception as e:
                logger.warning(f"Error updating position {symbol}: {e}")
        
        # Store updates
        self.position_updates.extend(updated_positions)
        
        # Keep only recent updates (last 1000)
        if len(self.position_updates) > 1000:
            self.position_updates = self.position_updates[-1000:]
        
        logger.debug(f"Updated {len(updated_positions)} positions")
    
    def _calculate_position_update(self, 
                                  symbol: str, 
                                  position_data: Dict, 
                                  current_price: float,
                                  timestamp: dt.datetime) -> Optional[PositionUpdate]:
        """Calculate position update with current Greeks and P&L."""
        try:
            strategy = position_data.get('strategy', 'unknown')
            size = position_data.get('size', 0)
            entry_price = position_data.get('entry_price', current_price)
            
            # Calculate unrealized P&L (simplified)
            unrealized_pnl = (current_price - entry_price) * size / entry_price * 0.01  # Rough estimate
            
            # Calculate Greeks (mock calculation - would use real options pricing)
            days_to_expiry = self._estimate_days_to_expiry(symbol)
            greeks = self._calculate_position_greeks(symbol, current_price, days_to_expiry)
            
            position_value = size * (1 + (current_price - entry_price) / entry_price)
            
            return PositionUpdate(
                timestamp=timestamp,
                symbol=symbol,
                strategy=strategy,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                delta=greeks['delta'],
                gamma=greeks['gamma'], 
                theta=greeks['theta'],
                vega=greeks['vega'],
                rho=greeks['rho'],
                days_to_expiry=days_to_expiry,
                position_value=position_value
            )
            
        except Exception as e:
            logger.warning(f"Error calculating position update for {symbol}: {e}")
            return None
    
    def _calculate_position_greeks(self, symbol: str, spot: float, days_to_expiry: int) -> Dict[str, float]:
        """Calculate position Greeks (simplified)."""
        # Mock Greeks calculation - in production would use actual option pricing
        tau = days_to_expiry / 365.0
        vol = 0.25  # Default volatility
        
        # Simplified Greeks (would be calculated properly with actual strikes/option types)
        return {
            'delta': np.random.uniform(-0.5, 0.5),
            'gamma': np.random.uniform(-0.1, 0.1),
            'theta': np.random.uniform(-20, 20),
            'vega': np.random.uniform(-50, 50),
            'rho': np.random.uniform(-10, 10)
        }
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        if self.polygon_client:
            try:
                return self.polygon_client.spot(symbol, dt.date.today())
            except:
                pass
        
        # Mock prices for demonstration
        mock_prices = {
            'AAPL': 185.50 + np.random.normal(0, 2),
            'TSLA': 245.25 + np.random.normal(0, 5),
            'NVDA': 875.30 + np.random.normal(0, 10),
            'SPY': 485.75 + np.random.normal(0, 2),
            'QQQ': 378.90 + np.random.normal(0, 3)
        }
        
        return mock_prices.get(symbol, 100.0 + np.random.normal(0, 1))
    
    def _estimate_days_to_expiry(self, symbol: str) -> int:
        """Estimate days to expiry for position."""
        # Mock estimation - in production would track actual expiry dates
        return np.random.randint(10, 45)
    
    def _check_risk_limits(self):
        """Check all risk limits and generate alerts."""
        current_time = dt.datetime.now()
        new_alerts = []
        
        if not self.position_updates:
            return
        
        # Get latest position updates
        latest_updates = self._get_latest_position_updates()
        
        # Portfolio-level checks
        portfolio_greeks = self._calculate_portfolio_greeks(latest_updates)
        portfolio_value = sum(update.position_value for update in latest_updates)
        
        # Check portfolio delta exposure
        portfolio_delta = abs(portfolio_greeks.get('delta', 0))
        if portfolio_delta > self.risk_limits['max_delta_exposure']:
            new_alerts.append(RiskAlert(
                timestamp=current_time,
                level='warning',
                category='greek',
                message=f"Portfolio delta exposure {portfolio_delta:.3f} exceeds limit {self.risk_limits['max_delta_exposure']:.3f}",
                current_value=portfolio_delta,
                limit_value=self.risk_limits['max_delta_exposure']
            ))
        
        # Check portfolio vega exposure
        portfolio_vega = abs(portfolio_greeks.get('vega', 0))
        if portfolio_vega > self.risk_limits['max_vega_exposure']:
            new_alerts.append(RiskAlert(
                timestamp=current_time,
                level='warning',
                category='greek',
                message=f"Portfolio vega exposure ${portfolio_vega:.0f} exceeds limit ${self.risk_limits['max_vega_exposure']:.0f}",
                current_value=portfolio_vega,
                limit_value=self.risk_limits['max_vega_exposure']
            ))
        
        # Check daily theta decay
        portfolio_theta = portfolio_greeks.get('theta', 0)
        if portfolio_theta < self.risk_limits['max_theta_decay']:
            new_alerts.append(RiskAlert(
                timestamp=current_time,
                level='info',
                category='greek',
                message=f"Daily theta decay ${portfolio_theta:.0f} exceeds ${self.risk_limits['max_theta_decay']:.0f}",
                current_value=portfolio_theta,
                limit_value=self.risk_limits['max_theta_decay']
            ))
        
        # Position-level checks
        for update in latest_updates:
            # Check position size
            position_pct = update.position_value / portfolio_value if portfolio_value > 0 else 0
            if position_pct > self.risk_limits['max_position_size']:
                new_alerts.append(RiskAlert(
                    timestamp=current_time,
                    level='warning',
                    category='position',
                    message=f"{update.symbol} position size {position_pct:.1%} exceeds limit {self.risk_limits['max_position_size']:.1%}",
                    symbol=update.symbol,
                    current_value=position_pct,
                    limit_value=self.risk_limits['max_position_size']
                ))
            
            # Check days to expiry
            if update.days_to_expiry < self.risk_limits['min_liquidity_days']:
                new_alerts.append(RiskAlert(
                    timestamp=current_time,
                    level='warning',
                    category='position',
                    message=f"{update.symbol} expires in {update.days_to_expiry} days (below {self.risk_limits['min_liquidity_days']} day limit)",
                    symbol=update.symbol,
                    current_value=update.days_to_expiry,
                    limit_value=self.risk_limits['min_liquidity_days']
                ))
        
        # Add new alerts
        self.active_alerts.extend(new_alerts)
        
        # Clean up old alerts (keep last 100)
        if len(self.active_alerts) > 100:
            self.active_alerts = self.active_alerts[-100:]
        
        if new_alerts:
            logger.info(f"Generated {len(new_alerts)} new risk alerts")
    
    def _get_latest_position_updates(self) -> List[PositionUpdate]:
        """Get latest update for each position."""
        if not self.position_updates:
            return []
        
        # Group by symbol and get latest
        latest_updates = {}
        for update in self.position_updates:
            if update.symbol not in latest_updates or update.timestamp > latest_updates[update.symbol].timestamp:
                latest_updates[update.symbol] = update
        
        return list(latest_updates.values())
    
    def _calculate_portfolio_greeks(self, position_updates: List[PositionUpdate]) -> Dict[str, float]:
        """Calculate aggregate portfolio Greeks."""
        portfolio_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        for update in position_updates:
            portfolio_greeks['delta'] += update.delta
            portfolio_greeks['gamma'] += update.gamma
            portfolio_greeks['theta'] += update.theta
            portfolio_greeks['vega'] += update.vega
            portfolio_greeks['rho'] += update.rho
        
        return portfolio_greeks
    
    def _generate_portfolio_snapshot(self):
        """Generate complete portfolio snapshot."""
        current_time = dt.datetime.now()
        latest_updates = self._get_latest_position_updates()
        
        if not latest_updates:
            return
        
        # Calculate portfolio metrics
        total_value = sum(update.position_value for update in latest_updates)
        unrealized_pnl = sum(update.unrealized_pnl for update in latest_updates)
        portfolio_greeks = self._calculate_portfolio_greeks(latest_updates)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(latest_updates, total_value)
        
        # Get recent alerts (last 10)
        recent_alerts = [
            {
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level,
                'category': alert.category,
                'message': alert.message,
                'symbol': alert.symbol
            }
            for alert in self.active_alerts[-10:]
        ]
        
        snapshot = PortfolioSnapshot(
            timestamp=current_time,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            positions=latest_updates,
            portfolio_greeks=portfolio_greeks,
            risk_metrics=risk_metrics,
            alerts=recent_alerts
        )
        
        self.portfolio_snapshots.append(snapshot)
        
        # Keep only recent snapshots (last 1000)
        if len(self.portfolio_snapshots) > 1000:
            self.portfolio_snapshots = self.portfolio_snapshots[-1000:]
        
        # Save to disk periodically
        if len(self.portfolio_snapshots) % 10 == 0:
            self._save_snapshot_data()
    
    def _calculate_risk_metrics(self, position_updates: List[PositionUpdate], total_value: float) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        if not position_updates:
            return {}
        
        # Calculate concentration (largest position %)
        position_values = [update.position_value for update in position_updates]
        max_position = max(position_values) if position_values else 0
        concentration = (max_position / total_value) if total_value > 0 else 0
        
        # Calculate portfolio volatility (simplified)
        pnls = [update.unrealized_pnl for update in position_updates]
        portfolio_vol = np.std(pnls) if len(pnls) > 1 else 0
        
        # Estimate VaR (simplified - 5% percentile)
        portfolio_var = np.percentile(pnls, 5) if len(pnls) > 1 else 0
        var_pct = abs(portfolio_var / total_value) if total_value > 0 else 0
        
        return {
            'concentration': concentration,
            'portfolio_volatility': portfolio_vol,
            'var_95_dollar': portfolio_var,
            'var_95_percent': var_pct,
            'total_positions': len(position_updates),
            'avg_dte': np.mean([update.days_to_expiry for update in position_updates])
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        if not self.portfolio_snapshots:
            return {'error': 'No portfolio data available'}
        
        latest_snapshot = self.portfolio_snapshots[-1]
        
        return {
            'timestamp': latest_snapshot.timestamp.isoformat(),
            'total_value': latest_snapshot.total_value,
            'unrealized_pnl': latest_snapshot.unrealized_pnl,
            'position_count': len(latest_snapshot.positions),
            'portfolio_greeks': latest_snapshot.portfolio_greeks,
            'risk_metrics': latest_snapshot.risk_metrics,
            'recent_alerts': latest_snapshot.alerts,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'strategy': pos.strategy,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'position_value': pos.position_value,
                    'days_to_expiry': pos.days_to_expiry,
                    'delta': pos.delta,
                    'theta': pos.theta,
                    'vega': pos.vega
                }
                for pos in latest_snapshot.positions
            ]
        }
    
    def get_historical_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get historical performance data."""
        cutoff_date = dt.datetime.now() - dt.timedelta(days=days)
        
        # Filter snapshots by date
        recent_snapshots = [
            snap for snap in self.portfolio_snapshots 
            if snap.timestamp >= cutoff_date
        ]
        
        if not recent_snapshots:
            return {'error': 'No historical data available'}
        
        # Extract time series data
        timestamps = [snap.timestamp for snap in recent_snapshots]
        portfolio_values = [snap.total_value for snap in recent_snapshots]
        pnls = [snap.unrealized_pnl for snap in recent_snapshots]
        
        # Calculate performance metrics
        if len(portfolio_values) > 1:
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            max_value = max(portfolio_values)
            min_value = min(portfolio_values)
            max_drawdown = (max_value - min_value) / max_value if max_value > 0 else 0
            volatility = np.std(np.diff(portfolio_values) / portfolio_values[:-1])
        else:
            total_return = 0
            max_drawdown = 0
            volatility = 0
        
        return {
            'period_days': days,
            'data_points': len(recent_snapshots),
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'current_pnl': pnls[-1] if pnls else 0,
            'max_pnl': max(pnls) if pnls else 0,
            'min_pnl': min(pnls) if pnls else 0,
            'time_series': {
                'timestamps': [ts.isoformat() for ts in timestamps],
                'portfolio_values': portfolio_values,
                'pnls': pnls
            }
        }
    
    def _save_snapshot_data(self):
        """Save portfolio snapshots to disk."""
        try:
            snapshot_file = self.data_dir / "portfolio_snapshots.pkl"
            with open(snapshot_file, 'wb') as f:
                pickle.dump(self.portfolio_snapshots[-100:], f)  # Save last 100 snapshots
            
            alerts_file = self.data_dir / "active_alerts.pkl"
            with open(alerts_file, 'wb') as f:
                pickle.dump(self.active_alerts[-50:], f)  # Save last 50 alerts
                
        except Exception as e:
            logger.warning(f"Error saving portfolio data: {e}")
    
    def _load_historical_data(self):
        """Load historical portfolio data from disk."""
        try:
            snapshot_file = self.data_dir / "portfolio_snapshots.pkl"
            if snapshot_file.exists():
                with open(snapshot_file, 'rb') as f:
                    self.portfolio_snapshots = pickle.load(f)
                logger.info(f"Loaded {len(self.portfolio_snapshots)} portfolio snapshots")
            
            alerts_file = self.data_dir / "active_alerts.pkl"
            if alerts_file.exists():
                with open(alerts_file, 'rb') as f:
                    self.active_alerts = pickle.load(f)
                logger.info(f"Loaded {len(self.active_alerts)} historical alerts")
                
        except Exception as e:
            logger.warning(f"Error loading historical portfolio data: {e}")
    
    def export_portfolio_data(self, filename: Optional[str] = None) -> str:
        """Export portfolio data to JSON file."""
        if filename is None:
            filename = f"portfolio_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_timestamp': dt.datetime.now().isoformat(),
            'portfolio_summary': self.get_portfolio_summary(),
            'historical_performance': self.get_historical_performance(30),
            'risk_limits': self.risk_limits,
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'level': alert.level,
                    'category': alert.category,
                    'message': alert.message,
                    'symbol': alert.symbol
                }
                for alert in self.active_alerts[-20:]
            ]
        }
        
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported portfolio data to {filepath}")
        return str(filepath)