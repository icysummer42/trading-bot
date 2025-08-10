"""Enhanced execution layer with advanced risk management integration."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from bot.risk_manager import AdvancedRiskManager, Position
from bot.greeks import GreeksCalculator
from bot.strategy.strategy_factory import StrategyFactory
from logger import get_logger

logger = get_logger("engine")


class StrategyEngine:
    """Enhanced strategy engine with risk-aware trade generation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.risk_manager = AdvancedRiskManager(cfg)
        self.greeks_calculator = GreeksCalculator()
        
        # Initialize strategy factory if we have a polygon client
        self.strategy_factory = None
        if hasattr(cfg, 'polygon_key') and cfg.polygon_key:
            try:
                from bot.polygon_client import PolygonClient
                polygon_client = PolygonClient(cfg.polygon_key)
                self.strategy_factory = StrategyFactory(
                    client=polygon_client,
                    greeks_calc=self.greeks_calculator,
                    risk_free_rate=getattr(cfg, 'risk_free_rate', 0.02)
                )
            except Exception as e:
                logger.warning(f"Could not initialize strategy factory: {e}")
        
        # Enhanced strategy mapping with new strategies
        self.strategy_mappings = {
            'iron_condor': 'iron_condor',
            'long_straddle': 'long_straddle',
            'short_straddle': 'short_straddle', 
            'long_strangle': 'long_strangle',
            'short_strangle': 'short_strangle',
            'bull_call_spread': 'bull_call_spread',
            'bear_put_spread': 'bear_put_spread',
            'bear_call_spread': 'bear_call_spread',
            'bull_put_spread': 'bull_put_spread',
            'long_call_butterfly': 'long_call_butterfly',
            'covered_call': 'bull_call_spread',  # Simplified mapping
            'put_spread': 'bear_put_spread'      # Simplified mapping
        }

    def generate(self, edge: float, sym: str, market_data: Dict[str, Any] = None):
        """
        Generate trades based on edge score and current risk metrics.
        
        Args:
            edge: Signal strength (-1 to 1)
            sym: Symbol to trade
            market_data: Current market data for the symbol
        
        Returns:
            List of trade dictionaries
        """
        if market_data is None:
            market_data = {}
        
        # Get risk-adjusted position size
        spot_price = market_data.get('spot_price', 100.0)
        volatility = market_data.get('volatility', 0.2)
        
        # Calculate Kelly-based position size  
        position_size = self.risk_manager.kelly_position_size(
            symbol=sym,
            expected_return=edge * 0.05,  # Convert edge to expected return
            volatility=volatility
        )
        
        # Enhanced strategy selection using strategy factory
        trades = []
        
        # Determine market outlook from edge
        if edge > 0.3:
            market_outlook = 'bullish'
        elif edge < -0.3:
            market_outlook = 'bearish'
        else:
            market_outlook = 'neutral'
        
        # Determine volatility outlook
        if volatility > 0.35:
            vol_outlook = 'high'
        elif volatility < 0.20:
            vol_outlook = 'low'
        else:
            vol_outlook = 'medium'
        
        # Get strategy recommendation
        strategy_name = self._select_optimal_strategy(
            edge=edge, 
            volatility=volatility, 
            market_outlook=market_outlook,
            vol_outlook=vol_outlook
        )
        
        trade = {
            "strategy": strategy_name,
            "symbol": sym,
            "size": position_size,
            "edge": edge,
            "market_data": market_data,
            "market_outlook": market_outlook,
            "volatility_outlook": vol_outlook
        }
        
        # Check risk limits before adding trade
        approved, reason = self.risk_manager.check_risk_limits(trade)
        if approved:
            trades.append(trade)
            logger.info(f"Generated trade: {trade['strategy']} for {sym}, size=${position_size:,.0f}")
        else:
            logger.warning(f"Trade rejected for {sym}: {reason}")
        
        return trades
    
    def _select_optimal_strategy(self, edge: float, volatility: float,
                               market_outlook: str, vol_outlook: str) -> str:
        """
        Select optimal strategy based on market conditions.
        
        Args:
            edge: Signal strength (-1 to 1)
            volatility: Expected volatility
            market_outlook: 'bullish', 'bearish', 'neutral'
            vol_outlook: 'high', 'medium', 'low'
        
        Returns:
            Strategy name
        """
        # If strategy factory is available, use it for recommendations
        if self.strategy_factory:
            recommendations = self.strategy_factory.recommend_strategies(
                market_outlook=market_outlook,
                volatility_outlook=vol_outlook,
                risk_profile='moderate'
            )
            
            if recommendations:
                # Select based on edge strength and volatility
                if abs(edge) > 0.6:
                    # High conviction - prefer directional strategies
                    directional_strategies = ['bull_call_spread', 'bear_put_spread', 
                                           'bull_put_spread', 'bear_call_spread']
                    for strategy in directional_strategies:
                        if strategy in recommendations:
                            return strategy
                
                elif vol_outlook == 'high':
                    # High vol - prefer volatility strategies
                    vol_strategies = ['long_straddle', 'long_strangle']
                    for strategy in vol_strategies:
                        if strategy in recommendations:
                            return strategy
                
                elif vol_outlook == 'low':
                    # Low vol - prefer income strategies
                    income_strategies = ['short_strangle', 'iron_condor', 'long_call_butterfly']
                    for strategy in income_strategies:
                        if strategy in recommendations:
                            return strategy
                
                # Default to first recommendation
                return recommendations[0]
        
        # Fallback to simple logic if no strategy factory
        if abs(edge) < 0.2:
            return "iron_condor"
        elif edge > 0.5:
            return "bull_call_spread"
        elif edge < -0.5:
            return "bear_put_spread"
        elif volatility > 0.3:
            return "short_strangle"
        else:
            return "long_straddle"
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Get information about a strategy."""
        if self.strategy_factory:
            return self.strategy_factory.get_strategy_info(strategy_name)
        return {}
    
    def get_available_strategies(self) -> List[str]:
        """Get list of all available strategies."""
        if self.strategy_factory:
            return self.strategy_factory.get_available_strategies()
        return list(self.strategy_mappings.keys())


class ExecutionEngine:
    """Enhanced execution engine with risk management and position tracking."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Dict] = []
        self.executed_trades: List[Dict] = []
        
    def place(self, trade: Dict[str, Any]) -> bool:
        """
        Place a trade order with risk checks.
        
        Args:
            trade: Trade dictionary with strategy, symbol, size, etc.
        
        Returns:
            True if trade was placed successfully
        """
        symbol = trade.get('symbol', 'UNKNOWN')
        strategy = trade.get('strategy', 'UNKNOWN')
        size = trade.get('size', 0)
        
        logger.info(f"[EXECUTE] {strategy} for {symbol}, size=${size:,.0f}")
        
        # In a real implementation, this would:
        # 1. Connect to broker API (Interactive Brokers, etc.)
        # 2. Calculate exact strikes and expirations
        # 3. Place the actual option orders
        # 4. Monitor fills and update positions
        
        # For now, simulate the execution
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'strategy': strategy,
            'size': size,
            'status': 'EXECUTED',
            'edge': trade.get('edge', 0),
            'entry_price': trade.get('market_data', {}).get('spot_price', 100.0)
        }
        
        self.executed_trades.append(trade_record)
        
        # Update positions (simplified)
        if symbol in self.positions:
            self.positions[symbol].size += size
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                strategy=strategy,
                size=size,
                entry_price=trade_record['entry_price'],
                current_value=size,  # Simplified
                entry_date=trade_record['timestamp']
            )
        
        return True
    
    def update_positions(self, market_data: Dict[str, float]):
        """Update position values with current market data."""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                # Simplified P&L calculation (real implementation would be more complex)
                position.unrealized_pnl = (current_price - position.entry_price) * position.size / position.entry_price
                position.current_value = position.size + position.unrealized_pnl
        
        logger.debug(f"Updated {len(self.positions)} positions")
    
    def close_position(self, symbol: str, percentage: float = 1.0) -> bool:
        """
        Close a position (partial or full).
        
        Args:
            symbol: Symbol to close
            percentage: Percentage to close (0.0 to 1.0)
        
        Returns:
            True if position was closed
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        close_size = position.size * percentage
        
        logger.info(f"Closing {percentage:.1%} of {symbol} position (${close_size:,.0f})")
        
        # Simulate closing the position
        if percentage >= 1.0:
            del self.positions[symbol]
        else:
            position.size *= (1 - percentage)
        
        return True
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of current portfolio."""
        total_value = sum(pos.current_value for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'positions_count': len(self.positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'positions': {sym: {
                'size': pos.size,
                'value': pos.current_value,
                'pnl': pos.unrealized_pnl,
                'strategy': pos.strategy
            } for sym, pos in self.positions.items()}
        }


class LegacyRiskManager:
    """Legacy risk manager wrapper for backward compatibility."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        if cfg:
            from bot.risk_manager import AdvancedRiskManager as ARM
            self.advanced_rm = ARM(cfg)
        else:
            self.advanced_rm = None

    def size(self, edge: float, capital: float):
        """Position sizing method with Kelly criterion."""
        if self.advanced_rm:
            # Use Kelly criterion if advanced risk manager is available
            return self.advanced_rm.kelly_position_size(
                symbol="DEFAULT",
                expected_return=edge * 0.05,
                volatility=0.2
            )
        else:
            # Fallback to simple 1% rule
            return capital * 0.01