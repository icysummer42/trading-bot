"""
Strategy Factory for creating and managing options trading strategies.

Provides centralized strategy creation, backtesting, and execution management.
"""

from __future__ import annotations
import datetime as dt
from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass
import importlib

from bot.polygon_client import PolygonClient
from bot.greeks import GreeksCalculator
from bot.strategy.base_strategy import BaseOptionsStrategy, StrategyPosition

# Import all strategy classes
from bot.strategy.straddle import LongStraddle, ShortStraddle
from bot.strategy.strangle import LongStrangle, ShortStrangle
from bot.strategy.spreads import BullCallSpread, BearPutSpread, BearCallSpread, BullPutSpread
from bot.strategy.butterfly import LongCallButterfly, LongPutButterfly, ShortCallButterfly

@dataclass 
class StrategyConfig:
    """Configuration for strategy creation."""
    strategy_name: str
    parameters: Dict[str, Any]
    risk_profile: str  # 'conservative', 'moderate', 'aggressive'
    market_outlook: str  # 'bullish', 'bearish', 'neutral'
    volatility_outlook: str  # 'low', 'medium', 'high'

class StrategyFactory:
    """
    Factory class for creating and managing options strategies.
    
    Provides:
    - Strategy instantiation
    - Parameter optimization
    - Strategy selection based on market conditions
    - Risk-adjusted strategy recommendations
    """
    
    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        self.client = client
        self.greeks_calc = greeks_calc or GreeksCalculator()
        self.risk_free_rate = risk_free_rate
        
        # Registry of available strategies
        self.strategy_registry: Dict[str, Type[BaseOptionsStrategy]] = {
            # Volatility Strategies
            'long_straddle': LongStraddle,
            'short_straddle': ShortStraddle,
            'long_strangle': LongStrangle, 
            'short_strangle': ShortStrangle,
            
            # Directional Strategies
            'bull_call_spread': BullCallSpread,
            'bear_put_spread': BearPutSpread,
            'bear_call_spread': BearCallSpread,
            'bull_put_spread': BullPutSpread,
            
            # Neutral/Income Strategies
            'long_call_butterfly': LongCallButterfly,
            'long_put_butterfly': LongPutButterfly,
            'short_call_butterfly': ShortCallButterfly,
            
            # Legacy iron condor (from existing codebase)
            'iron_condor': None  # Will be handled separately
        }
        
        # Strategy characteristics for selection
        self.strategy_profiles = {
            'long_straddle': {
                'market_outlook': 'neutral',
                'volatility_outlook': 'high',
                'risk_profile': 'moderate',
                'max_loss': 'limited',
                'max_profit': 'unlimited',
                'description': 'Profits from big moves in either direction'
            },
            'short_straddle': {
                'market_outlook': 'neutral',
                'volatility_outlook': 'low',
                'risk_profile': 'aggressive',
                'max_loss': 'unlimited',
                'max_profit': 'limited',
                'description': 'Profits from low volatility, range-bound movement'
            },
            'long_strangle': {
                'market_outlook': 'neutral',
                'volatility_outlook': 'high',
                'risk_profile': 'moderate',
                'max_loss': 'limited',
                'max_profit': 'unlimited',
                'description': 'Cheaper volatility play than straddle'
            },
            'short_strangle': {
                'market_outlook': 'neutral',
                'volatility_outlook': 'low', 
                'risk_profile': 'aggressive',
                'max_loss': 'unlimited',
                'max_profit': 'limited',
                'description': 'Income strategy with wider profit range'
            },
            'bull_call_spread': {
                'market_outlook': 'bullish',
                'volatility_outlook': 'low',
                'risk_profile': 'conservative',
                'max_loss': 'limited',
                'max_profit': 'limited',
                'description': 'Limited risk bullish strategy'
            },
            'bear_put_spread': {
                'market_outlook': 'bearish',
                'volatility_outlook': 'low',
                'risk_profile': 'conservative',
                'max_loss': 'limited',
                'max_profit': 'limited',
                'description': 'Limited risk bearish strategy'
            },
            'bear_call_spread': {
                'market_outlook': 'bearish',
                'volatility_outlook': 'low',
                'risk_profile': 'moderate',
                'max_loss': 'limited',
                'max_profit': 'limited',
                'description': 'Credit spread for bearish outlook'
            },
            'bull_put_spread': {
                'market_outlook': 'bullish',
                'volatility_outlook': 'low',
                'risk_profile': 'moderate',
                'max_loss': 'limited',
                'max_profit': 'limited',
                'description': 'Credit spread for bullish outlook'
            },
            'long_call_butterfly': {
                'market_outlook': 'neutral',
                'volatility_outlook': 'low',
                'risk_profile': 'conservative',
                'max_loss': 'limited',
                'max_profit': 'limited',
                'description': 'Profits from minimal price movement'
            },
            'iron_condor': {
                'market_outlook': 'neutral',
                'volatility_outlook': 'low',
                'risk_profile': 'moderate',
                'max_loss': 'limited',
                'max_profit': 'limited',
                'description': 'Range-bound income strategy'
            }
        }

    def get_available_strategies(self) -> List[str]:
        """Get list of all available strategy names."""
        return list(self.strategy_registry.keys())

    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Get information about a specific strategy."""
        return self.strategy_profiles.get(strategy_name, {})

    def create_strategy(self, strategy_name: str) -> Optional[BaseOptionsStrategy]:
        """
        Create a strategy instance.
        
        Args:
            strategy_name: Name of the strategy to create
        
        Returns:
            Strategy instance or None if not found
        """
        if strategy_name not in self.strategy_registry:
            return None
        
        strategy_class = self.strategy_registry[strategy_name]
        if strategy_class is None:
            return None  # Handle legacy strategies separately
        
        return strategy_class(
            client=self.client,
            greeks_calc=self.greeks_calc,
            risk_free_rate=self.risk_free_rate
        )

    def recommend_strategies(self, market_outlook: str, volatility_outlook: str, 
                           risk_profile: str = 'moderate') -> List[str]:
        """
        Recommend strategies based on market conditions and risk profile.
        
        Args:
            market_outlook: 'bullish', 'bearish', or 'neutral'
            volatility_outlook: 'low', 'medium', or 'high'
            risk_profile: 'conservative', 'moderate', or 'aggressive'
        
        Returns:
            List of recommended strategy names
        """
        recommendations = []
        
        for strategy_name, profile in self.strategy_profiles.items():
            # Match market outlook
            if profile.get('market_outlook') != market_outlook and market_outlook != 'any':
                continue
            
            # Match volatility outlook
            if volatility_outlook == 'high' and profile.get('volatility_outlook') == 'high':
                recommendations.append(strategy_name)
            elif volatility_outlook == 'low' and profile.get('volatility_outlook') == 'low':
                recommendations.append(strategy_name)
            elif volatility_outlook == 'medium':
                # Medium vol can work with both high and low vol strategies
                recommendations.append(strategy_name)
            
            # Filter by risk profile if specified
            if risk_profile != 'any' and profile.get('risk_profile') != risk_profile:
                if strategy_name in recommendations:
                    recommendations.remove(strategy_name)
        
        return recommendations

    def create_position(self, strategy_name: str, symbol: str, spot: float, 
                       trade_date: dt.date, expiry: dt.date, 
                       **kwargs) -> Optional[StrategyPosition]:
        """
        Create a strategy position.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Underlying symbol
            spot: Current spot price
            trade_date: Trade date
            expiry: Expiry date
            **kwargs: Strategy-specific parameters
        
        Returns:
            StrategyPosition or None if creation fails
        """
        strategy = self.create_strategy(strategy_name)
        if strategy is None:
            return None
        
        return strategy.create_position(
            symbol=symbol,
            spot=spot,
            trade_date=trade_date,
            expiry=expiry,
            **kwargs
        )

    def optimize_parameters(self, strategy_name: str, symbol: str, 
                          historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize strategy parameters based on historical data.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Underlying symbol
            historical_data: Historical market data
        
        Returns:
            Optimized parameters dictionary
        """
        # This is a simplified optimization - in production, you'd use more sophisticated methods
        base_params = {}
        
        if 'straddle' in strategy_name or 'strangle' in strategy_name:
            # For volatility strategies, optimize based on realized vs implied vol
            realized_vol = historical_data.get('realized_volatility', 0.25)
            implied_vol = historical_data.get('implied_volatility', 0.25)
            
            # Adjust target deltas based on vol differential
            if implied_vol > realized_vol * 1.2:
                # IV is high, consider selling strategies or tighter deltas
                base_params['delta_call'] = 0.20
                base_params['delta_put'] = -0.20
            else:
                # IV is reasonable, use standard deltas
                base_params['delta_call'] = 0.25
                base_params['delta_put'] = -0.25
        
        elif 'spread' in strategy_name:
            # For spreads, optimize wing width based on expected move
            expected_move = historical_data.get('expected_move', spot * 0.05)
            base_params['spread_width'] = max(5, expected_move * 0.5)
        
        elif 'butterfly' in strategy_name:
            # For butterflies, optimize wing width based on volatility
            volatility = historical_data.get('volatility', 0.25)
            base_params['wing_width'] = max(5, spot * volatility * 0.1)
        
        return base_params

    def backtest_strategy(self, strategy_name: str, symbol: str, 
                         start_date: dt.date, end_date: dt.date,
                         **kwargs) -> Dict[str, Any]:
        """
        Backtest a strategy over a date range.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Underlying symbol  
            start_date: Start date for backtest
            end_date: End date for backtest
            **kwargs: Strategy parameters
        
        Returns:
            Backtest results dictionary
        """
        strategy = self.create_strategy(strategy_name)
        if strategy is None:
            return {'error': f'Strategy {strategy_name} not found'}
        
        results = {
            'strategy_name': strategy_name,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'win_rate': 0.0,
            'avg_winner': 0.0,
            'avg_loser': 0.0,
            'trades': []
        }
        
        # Simplified backtesting logic
        current_date = start_date
        trade_count = 0
        
        while current_date <= end_date and trade_count < 50:  # Limit for demo
            # Monthly trades
            expiry = strategy._next_monthly_expiry(current_date)
            spot = self.client.spot(symbol, current_date)
            
            if spot is None:
                current_date += dt.timedelta(days=1)
                continue
            
            position = strategy.create_position(
                symbol=symbol,
                spot=spot,
                trade_date=current_date,
                expiry=expiry,
                **kwargs
            )
            
            if position:
                # Simulate holding to expiry
                final_spot = self.client.spot(symbol, expiry)
                if final_spot:
                    # Update position with final price
                    updated_position = strategy.update_position(position, final_spot, expiry)
                    pnl = updated_position.unrealized_pnl
                    
                    results['trades'].append({
                        'entry_date': current_date,
                        'expiry_date': expiry,
                        'entry_spot': spot,
                        'exit_spot': final_spot,
                        'pnl': pnl
                    })
                    
                    results['total_pnl'] += pnl
                    results['total_trades'] += 1
                    
                    if pnl > 0:
                        results['winning_trades'] += 1
                        results['max_profit'] = max(results['max_profit'], pnl)
                    else:
                        results['losing_trades'] += 1
                        results['max_loss'] = min(results['max_loss'], pnl)
                    
                    trade_count += 1
            
            # Move to next month
            current_date = current_date.replace(day=1) + dt.timedelta(days=32)
            current_date = current_date.replace(day=1)
        
        # Calculate summary statistics
        if results['total_trades'] > 0:
            results['win_rate'] = results['winning_trades'] / results['total_trades']
            
            winning_pnls = [t['pnl'] for t in results['trades'] if t['pnl'] > 0]
            losing_pnls = [t['pnl'] for t in results['trades'] if t['pnl'] < 0]
            
            if winning_pnls:
                results['avg_winner'] = sum(winning_pnls) / len(winning_pnls)
            if losing_pnls:
                results['avg_loser'] = sum(losing_pnls) / len(losing_pnls)
        
        return results

    def get_strategy_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all available strategies."""
        summary = {}
        for strategy_name in self.strategy_registry.keys():
            summary[strategy_name] = {
                'profile': self.strategy_profiles.get(strategy_name, {}),
                'available': strategy_name in self.strategy_registry,
                'class_name': self.strategy_registry[strategy_name].__name__ if self.strategy_registry[strategy_name] else 'Legacy'
            }
        return summary