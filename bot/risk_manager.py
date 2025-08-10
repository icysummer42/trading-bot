"""
Advanced Risk Management System for Options Trading Bot

Implements VaR, Kelly Criterion, drawdown controls, and portfolio correlation analysis.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from scipy.stats import norm
from logger import get_logger

logger = get_logger("risk_manager")

@dataclass
class Position:
    """Individual position representation."""
    symbol: str
    strategy: str
    size: float
    entry_price: float
    current_value: float
    entry_date: datetime
    unrealized_pnl: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

@dataclass
class Portfolio:
    """Portfolio state with positions and metrics."""
    positions: Dict[str, Position] = field(default_factory=dict)
    cash: float = 1_000_000.0
    total_value: float = 1_000_000.0
    daily_returns: List[float] = field(default_factory=list)
    high_water_mark: float = 1_000_000.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0

@dataclass
class RiskMetrics:
    """Risk metrics for a portfolio."""
    portfolio_var_95: float = 0.0
    portfolio_var_99: float = 0.0
    expected_shortfall: float = 0.0
    portfolio_beta: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    correlation_risk: float = 0.0
    concentration_risk: float = 0.0

class VarCalculator:
    """Value at Risk calculation using multiple methods."""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
    
    def historical_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Historical VaR from return series."""
        if len(returns) < 30:
            logger.warning("Insufficient data for reliable VaR calculation")
            return 0.0
        
        return -np.percentile(returns, (1 - confidence_level) * 100)
    
    def parametric_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Parametric VaR assuming normal distribution."""
        if len(returns) < 30:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = norm.ppf(1 - confidence_level)
        
        return -(mean + z_score * std)
    
    def monte_carlo_var(self, portfolio_value: float, expected_return: float, 
                       volatility: float, confidence_level: float = 0.95, 
                       num_simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        simulated_returns = np.random.normal(expected_return, volatility, num_simulations)
        simulated_values = portfolio_value * (1 + simulated_returns)
        losses = portfolio_value - simulated_values
        
        return np.percentile(losses, confidence_level * 100)
    
    def expected_shortfall(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var = self.historical_var(returns, confidence_level)
        tail_losses = returns[returns <= -var]
        
        if len(tail_losses) == 0:
            return var
        
        return -np.mean(tail_losses)

class KellyCriterion:
    """Kelly Criterion position sizing for optimal capital allocation."""
    
    def __init__(self, max_kelly_fraction: float = 0.25):
        self.max_kelly_fraction = max_kelly_fraction  # Cap Kelly at 25% to reduce risk
    
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, 
                               avg_loss: float) -> float:
        """
        Calculate Kelly fraction for position sizing.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount (positive)
            avg_loss: Average loss amount (positive)
        
        Returns:
            Kelly fraction (0-1)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly formula: f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Cap the Kelly fraction to reduce risk
        kelly_fraction = max(0, min(kelly_fraction, self.max_kelly_fraction))
        
        logger.debug(f"Kelly calculation: win_rate={win_rate:.3f}, b={b:.3f}, "
                    f"kelly_fraction={kelly_fraction:.3f}")
        
        return kelly_fraction
    
    def position_size(self, capital: float, kelly_fraction: float, 
                     risk_per_trade: float) -> float:
        """
        Calculate position size based on Kelly fraction.
        
        Args:
            capital: Available capital
            kelly_fraction: Kelly fraction (0-1)
            risk_per_trade: Maximum risk per trade as fraction of capital
        
        Returns:
            Position size in dollars
        """
        # Use the smaller of Kelly fraction or risk per trade limit
        sizing_fraction = min(kelly_fraction, risk_per_trade)
        return capital * sizing_fraction

class CorrelationAnalyzer:
    """Analyze correlations between portfolio positions."""
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
    
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for portfolio positions."""
        if returns_data.empty:
            return pd.DataFrame()
        
        return returns_data.corr()
    
    def portfolio_concentration_risk(self, positions: Dict[str, Position]) -> float:
        """Calculate concentration risk using Herfindahl-Hirschman Index."""
        if not positions:
            return 0.0
        
        total_value = sum(abs(pos.current_value) for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        # Calculate HHI
        hhi = sum((abs(pos.current_value) / total_value) ** 2 
                 for pos in positions.values())
        
        # Normalize to 0-1 scale (1 = maximum concentration)
        n = len(positions)
        normalized_hhi = (hhi - 1/n) / (1 - 1/n) if n > 1 else 1.0
        
        return max(0, min(1, normalized_hhi))
    
    def diversification_ratio(self, weights: np.ndarray, 
                            correlation_matrix: pd.DataFrame) -> float:
        """Calculate portfolio diversification ratio."""
        if correlation_matrix.empty or len(weights) == 0:
            return 1.0
        
        # Portfolio volatility
        portfolio_var = np.dot(weights, np.dot(correlation_matrix.values, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Weighted average of individual volatilities
        individual_vols = np.sqrt(np.diag(correlation_matrix.values))
        weighted_avg_vol = np.dot(weights, individual_vols)
        
        if portfolio_vol == 0:
            return 1.0
        
        return weighted_avg_vol / portfolio_vol

class AdvancedRiskManager:
    """
    Comprehensive risk management system with VaR, Kelly sizing, 
    drawdown controls, and correlation monitoring.
    """
    
    def __init__(self, config):
        self.config = config
        self.portfolio = Portfolio()
        self.var_calculator = VarCalculator()
        self.kelly_calculator = KellyCriterion(
            max_kelly_fraction=getattr(config, 'max_kelly_fraction', 0.25)
        )
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Risk limits from config
        self.max_portfolio_var = getattr(config, 'max_portfolio_var', 0.02)  # 2% daily VaR
        self.max_drawdown_limit = getattr(config, 'max_drawdown_limit', 0.15)  # 15% max drawdown
        self.max_position_size = getattr(config, 'max_position_size', 0.1)  # 10% per position
        self.max_correlation = getattr(config, 'max_correlation', 0.7)  # 70% max correlation
        
        self.risk_metrics = RiskMetrics()
        self.trade_history: List[Dict] = []
    
    def update_portfolio(self, positions: Dict[str, Position], 
                        market_data: Dict[str, float]) -> None:
        """Update portfolio state with current positions and market data."""
        self.portfolio.positions = positions
        
        # Calculate current portfolio value
        total_value = self.portfolio.cash + sum(pos.current_value for pos in positions.values())
        
        # Update daily return
        if self.portfolio.total_value > 0:
            daily_return = (total_value - self.portfolio.total_value) / self.portfolio.total_value
            self.portfolio.daily_returns.append(daily_return)
            
            # Keep only last 252 days (1 year)
            if len(self.portfolio.daily_returns) > 252:
                self.portfolio.daily_returns = self.portfolio.daily_returns[-252:]
        
        self.portfolio.total_value = total_value
        
        # Update high water mark and drawdown
        if total_value > self.portfolio.high_water_mark:
            self.portfolio.high_water_mark = total_value
            self.portfolio.current_drawdown = 0.0
        else:
            self.portfolio.current_drawdown = (self.portfolio.high_water_mark - total_value) / self.portfolio.high_water_mark
            self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, self.portfolio.current_drawdown)
        
        # Update risk metrics
        self._calculate_risk_metrics()
    
    def _calculate_risk_metrics(self) -> None:
        """Calculate all risk metrics for the current portfolio."""
        returns = np.array(self.portfolio.daily_returns)
        
        if len(returns) >= 30:
            # VaR calculations
            self.risk_metrics.portfolio_var_95 = self.var_calculator.historical_var(returns, 0.95)
            self.risk_metrics.portfolio_var_99 = self.var_calculator.historical_var(returns, 0.99)
            self.risk_metrics.expected_shortfall = self.var_calculator.expected_shortfall(returns, 0.95)
            
            # Sharpe ratio (assume risk-free rate = 2%)
            if len(returns) > 1:
                excess_returns = returns - 0.02/252  # Daily risk-free rate
                self.risk_metrics.sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        
        # Drawdown metrics
        self.risk_metrics.current_drawdown = self.portfolio.current_drawdown
        self.risk_metrics.max_drawdown = self.portfolio.max_drawdown
        
        # Concentration risk
        self.risk_metrics.concentration_risk = self.correlation_analyzer.portfolio_concentration_risk(
            self.portfolio.positions
        )
    
    def kelly_position_size(self, symbol: str, expected_return: float, 
                          volatility: float, confidence: float = 0.6) -> float:
        """
        Calculate position size using Kelly Criterion based on historical performance.
        
        Args:
            symbol: Trading symbol
            expected_return: Expected return for the trade
            volatility: Expected volatility
            confidence: Confidence in the expected return (0-1)
        
        Returns:
            Recommended position size in dollars
        """
        # Get historical performance for this strategy/symbol
        symbol_trades = [t for t in self.trade_history 
                        if t.get('symbol') == symbol]
        
        if len(symbol_trades) < 10:
            # Use conservative sizing for new strategies
            return self.portfolio.total_value * 0.02  # 2% of portfolio
        
        # Calculate win rate and average win/loss
        wins = [t['pnl'] for t in symbol_trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in symbol_trades if t['pnl'] < 0]
        
        if not wins or not losses:
            return self.portfolio.total_value * 0.02
        
        win_rate = len(wins) / len(symbol_trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        # Apply confidence adjustment
        adjusted_win_rate = win_rate * confidence + 0.5 * (1 - confidence)
        
        kelly_fraction = self.kelly_calculator.calculate_kelly_fraction(
            adjusted_win_rate, avg_win, avg_loss
        )
        
        max_risk_per_trade = self.max_position_size
        position_size = self.kelly_calculator.position_size(
            self.portfolio.total_value, kelly_fraction, max_risk_per_trade
        )
        
        logger.info(f"Kelly sizing for {symbol}: fraction={kelly_fraction:.3f}, "
                   f"size=${position_size:,.0f}")
        
        return position_size
    
    def check_risk_limits(self, proposed_trade: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if proposed trade violates risk limits.
        
        Args:
            proposed_trade: Dictionary with trade details
        
        Returns:
            (approved, reason) - True if trade is approved
        """
        # Check VaR limits
        if self.risk_metrics.portfolio_var_95 > self.max_portfolio_var:
            return False, f"Portfolio VaR ({self.risk_metrics.portfolio_var_95:.3f}) exceeds limit ({self.max_portfolio_var:.3f})"
        
        # Check drawdown limits
        if self.portfolio.current_drawdown > self.max_drawdown_limit:
            return False, f"Current drawdown ({self.portfolio.current_drawdown:.3f}) exceeds limit ({self.max_drawdown_limit:.3f})"
        
        # Check position size limits
        trade_size = proposed_trade.get('size', 0)
        if trade_size > self.portfolio.total_value * self.max_position_size:
            return False, f"Position size exceeds {self.max_position_size:.1%} of portfolio"
        
        # Check concentration limits
        if self.risk_metrics.concentration_risk > 0.8:
            return False, "Portfolio concentration risk too high"
        
        return True, "Trade approved"
    
    def calculate_position_greeks(self, position: Position, spot_price: float,
                                risk_free_rate: float = 0.02) -> Position:
        """Calculate Greeks for options positions (placeholder for now)."""
        # This would integrate with options pricing models
        # For now, return position as-is
        logger.debug(f"Greeks calculation for {position.symbol} - placeholder")
        return position
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        return {
            'portfolio_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'positions_count': len(self.portfolio.positions),
            'risk_metrics': {
                'var_95': self.risk_metrics.portfolio_var_95,
                'var_99': self.risk_metrics.portfolio_var_99,
                'expected_shortfall': self.risk_metrics.expected_shortfall,
                'sharpe_ratio': self.risk_metrics.sharpe_ratio,
                'max_drawdown': self.risk_metrics.max_drawdown,
                'current_drawdown': self.risk_metrics.current_drawdown,
                'concentration_risk': self.risk_metrics.concentration_risk,
            },
            'risk_limits': {
                'max_var_limit': self.max_portfolio_var,
                'max_drawdown_limit': self.max_drawdown_limit,
                'max_position_size': self.max_position_size,
            },
            'positions': {symbol: {
                'value': pos.current_value,
                'pnl': pos.unrealized_pnl,
                'strategy': pos.strategy
            } for symbol, pos in self.portfolio.positions.items()}
        }
    
    def record_trade(self, trade_result: Dict[str, Any]) -> None:
        """Record completed trade for performance tracking."""
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': trade_result.get('symbol'),
            'strategy': trade_result.get('strategy'),
            'pnl': trade_result.get('pnl', 0),
            'size': trade_result.get('size', 0),
            'duration_days': trade_result.get('duration_days', 0)
        }
        
        self.trade_history.append(trade_record)
        
        # Keep only last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        
        logger.info(f"Recorded trade: {trade_record}")