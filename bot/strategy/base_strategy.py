"""
Base strategy class for options trading strategies.

Provides common functionality for P&L calculation, Greeks analysis,
and strategy execution across all options strategies.
"""

from __future__ import annotations
import math
import datetime as dt
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from bot.polygon_client import PolygonClient
from bot.pricing import bs_price
from bot.greeks import GreeksCalculator

@dataclass
class OptionLeg:
    """Represents a single option leg in a strategy."""
    ticker: str
    strike: float
    expiry: dt.date
    is_call: bool
    position: int  # +1 for long, -1 for short
    quantity: int = 1
    entry_price: float = 0.0
    current_price: float = 0.0
    implied_vol: float = 0.20

@dataclass
class StrategyPosition:
    """Complete options strategy position."""
    symbol: str
    strategy_name: str
    legs: List[OptionLeg]
    entry_date: dt.date
    expiry: dt.date
    spot_at_entry: float
    current_spot: float = 0.0
    unrealized_pnl: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None

class BaseOptionsStrategy(ABC):
    """
    Abstract base class for all options trading strategies.
    
    Provides common functionality for:
    - P&L calculation
    - Greeks analysis
    - Strike selection
    - Position management
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        self.client = client
        self.greeks_calc = greeks_calc or GreeksCalculator()
        self.r = risk_free_rate
        self.strategy_name = self.__class__.__name__

    @staticmethod
    def _occ_ticker(root: str, expiry: dt.date, strike: float, is_call: bool) -> Optional[str]:
        """Generate OCC option ticker format."""
        if strike is None or math.isnan(strike):
            return None
        side = "C" if is_call else "P"
        return f"O:{root.upper()}{expiry.strftime('%y%m%d')}{side}{int(strike * 1000):08d}"

    @staticmethod
    def _next_monthly_expiry(trade_date: dt.date, min_days: int = 21) -> dt.date:
        """Find next monthly Friday at least min_days away."""
        target_date = trade_date + dt.timedelta(days=min_days)
        # Find third Friday of the month
        first_day = target_date.replace(day=1)
        first_friday = first_day + dt.timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + dt.timedelta(days=14)
        return third_friday

    @staticmethod
    def _weekly_expiry(trade_date: dt.date, days_out: int = 7) -> dt.date:
        """Find next Friday that's days_out away."""
        target_date = trade_date + dt.timedelta(days=days_out)
        days_to_friday = (4 - target_date.weekday()) % 7
        return target_date + dt.timedelta(days=days_to_friday)

    def _select_strikes_by_delta(self, chain, spot: float, target_deltas: List[float],
                                is_call: bool) -> List[float]:
        """
        Select strikes based on target delta values.
        
        Args:
            chain: Options chain DataFrame
            spot: Current spot price
            target_deltas: List of target delta values
            is_call: True for calls, False for puts
        
        Returns:
            List of selected strike prices
        """
        selected_strikes = []
        
        # Filter chain for the right option type
        option_type = "call" if is_call else "put"
        filtered_chain = self._filter_chain_by_type(chain, option_type)
        
        if filtered_chain.empty:
            return []
        
        # Calculate deltas for all strikes
        strike_col = self._get_strike_column(filtered_chain)
        strikes = filtered_chain[strike_col].values
        
        for target_delta in target_deltas:
            best_strike = None
            min_delta_diff = float('inf')
            
            for strike in strikes:
                # Estimate time to expiry (simplified to 30 days)
                tau = 30 / 365
                iv = 0.25  # Default IV
                
                calculated_delta = self.greeks_calc.delta(
                    spot=spot, 
                    strike=strike, 
                    iv=iv, 
                    tau=tau, 
                    r=self.r, 
                    is_call=is_call
                )
                
                delta_diff = abs(calculated_delta - target_delta)
                if delta_diff < min_delta_diff:
                    min_delta_diff = delta_diff
                    best_strike = strike
            
            if best_strike is not None:
                selected_strikes.append(best_strike)
        
        return selected_strikes

    def _filter_chain_by_type(self, chain, option_type: str):
        """Filter options chain by type (call/put)."""
        if "contract_type" in chain.columns:
            return chain[chain["contract_type"] == option_type]
        elif "option_type" in chain.columns:
            return chain[chain["option_type"] == option_type]
        elif "type" in chain.columns:
            return chain[chain["type"] == option_type]
        return chain

    def _get_strike_column(self, chain):
        """Get the strike price column name from chain."""
        if "strike_price" in chain.columns:
            return "strike_price"
        elif "strike" in chain.columns:
            return "strike"
        else:
            raise ValueError(f"No strike column found in: {list(chain.columns)}")

    def _calculate_leg_pnl(self, leg: OptionLeg, current_spot: float, 
                          current_date: dt.date) -> float:
        """Calculate P&L for a single option leg."""
        # Time to expiry
        tau = max(0, (leg.expiry - current_date).days / 365.0)
        
        if tau <= 0:
            # At expiration, use intrinsic value
            if leg.is_call:
                intrinsic = max(0, current_spot - leg.strike)
            else:
                intrinsic = max(0, leg.strike - current_spot)
            current_value = intrinsic
        else:
            # Use Black-Scholes pricing
            current_value = bs_price(
                spot=current_spot,
                strike=leg.strike,
                iv=leg.implied_vol,
                tau=tau,
                r=self.r,
                call=leg.is_call
            )
        
        # P&L = (current_value - entry_price) * position * quantity
        leg_pnl = (current_value - leg.entry_price) * leg.position * leg.quantity
        return leg_pnl

    def _calculate_position_greeks(self, position: StrategyPosition, 
                                 current_date: dt.date) -> Dict[str, float]:
        """Calculate total Greeks for the entire position."""
        total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        for leg in position.legs:
            tau = max(0, (leg.expiry - current_date).days / 365.0)
            
            if tau > 0:
                leg_greeks = self.greeks_calc.calculate_all_greeks(
                    spot=position.current_spot,
                    strike=leg.strike,
                    iv=leg.implied_vol,
                    tau=tau,
                    r=self.r,
                    is_call=leg.is_call
                )
                
                # Scale by position and quantity
                multiplier = leg.position * leg.quantity
                for greek_name, greek_value in leg_greeks.items():
                    total_greeks[greek_name] += greek_value * multiplier
        
        return total_greeks

    @abstractmethod
    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """
        Create a new strategy position.
        
        Args:
            symbol: Underlying symbol
            spot: Current spot price
            trade_date: Trade entry date
            expiry: Options expiry date
            **kwargs: Strategy-specific parameters
        
        Returns:
            StrategyPosition if successful, None otherwise
        """
        pass

    @abstractmethod
    def calculate_max_profit_loss(self, position: StrategyPosition) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate theoretical max profit and max loss for the position.
        
        Returns:
            (max_profit, max_loss) tuple
        """
        pass

    def update_position(self, position: StrategyPosition, current_spot: float, 
                       current_date: dt.date) -> StrategyPosition:
        """Update position with current market data."""
        position.current_spot = current_spot
        
        # Calculate total P&L
        total_pnl = sum(
            self._calculate_leg_pnl(leg, current_spot, current_date) 
            for leg in position.legs
        )
        position.unrealized_pnl = total_pnl
        
        # Update Greeks
        greeks = self._calculate_position_greeks(position, current_date)
        position.delta = greeks['delta']
        position.gamma = greeks['gamma']
        position.theta = greeks['theta']
        position.vega = greeks['vega']
        
        return position

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven points for the strategy."""
        # This is a simplified calculation - subclasses can override
        # for more complex strategies
        strikes = [leg.strike for leg in position.legs]
        if strikes:
            return [min(strikes), max(strikes)]
        return []

    def risk_reward_analysis(self, position: StrategyPosition) -> Dict[str, Any]:
        """Generate risk/reward analysis for the position."""
        max_profit, max_loss = self.calculate_max_profit_loss(position)
        breakevens = self.get_breakeven_points(position)
        
        return {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'current_pnl': position.unrealized_pnl,
            'breakeven_points': breakevens,
            'delta': position.delta,
            'gamma': position.gamma,
            'theta': position.theta,
            'vega': position.vega,
            'profit_potential': max_profit / abs(max_loss) if max_loss else None,
            'days_to_expiry': (position.expiry - dt.date.today()).days
        }