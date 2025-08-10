"""
Long and Short Straddle strategies implementation.

Long Straddle: Buy ATM call + Buy ATM put (volatility play)
Short Straddle: Sell ATM call + Sell ATM put (income strategy)
"""

from __future__ import annotations
import datetime as dt
from typing import List, Tuple, Optional
import numpy as np

from bot.strategy.base_strategy import BaseOptionsStrategy, StrategyPosition, OptionLeg
from bot.polygon_client import PolygonClient
from bot.greeks import GreeksCalculator

class LongStraddle(BaseOptionsStrategy):
    """
    Long Straddle Strategy: Buy ATM Call + Buy ATM Put
    
    Characteristics:
    - Market Outlook: Neutral (expecting big move in either direction)
    - Max Profit: Unlimited
    - Max Loss: Premium paid
    - Breakeven: Strike ± Total Premium
    - Best Market: High volatility, big moves
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        super().__init__(client, greeks_calc, risk_free_rate)
        self.strategy_name = "Long Straddle"

    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """
        Create a long straddle position.
        
        Args:
            symbol: Underlying symbol
            spot: Current spot price
            trade_date: Entry date
            expiry: Options expiry date
            **kwargs: Additional parameters (quantity, etc.)
        
        Returns:
            StrategyPosition or None if creation fails
        """
        quantity = kwargs.get('quantity', 1)
        
        # Get options chain
        chain = self.client.snapshot_chain(symbol, trade_date)
        if chain.empty:
            return None
        
        # Find ATM strike (closest to spot)
        strike_col = self._get_strike_column(chain)
        strikes = chain[strike_col].values
        atm_strike = strikes[np.argmin(np.abs(strikes - spot))]
        
        # Create option legs
        call_ticker = self._occ_ticker(symbol, expiry, atm_strike, True)
        put_ticker = self._occ_ticker(symbol, expiry, atm_strike, False)
        
        if not call_ticker or not put_ticker:
            return None
        
        # Get entry prices (simplified - in real trading, use bid/ask)
        call_price = self.client.agg_close(call_ticker, trade_date)
        put_price = self.client.agg_close(put_ticker, trade_date)
        
        if call_price is None or put_price is None:
            # Fallback to theoretical pricing
            tau = (expiry - trade_date).days / 365.0
            iv = kwargs.get('implied_vol', 0.25)
            
            from bot.pricing import bs_price
            call_price = bs_price(spot, atm_strike, iv, tau, self.r, True)
            put_price = bs_price(spot, atm_strike, iv, tau, self.r, False)
        
        # Create legs
        legs = [
            OptionLeg(
                ticker=call_ticker,
                strike=atm_strike,
                expiry=expiry,
                is_call=True,
                position=1,  # Long
                quantity=quantity,
                entry_price=call_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            OptionLeg(
                ticker=put_ticker,
                strike=atm_strike,
                expiry=expiry,
                is_call=False,
                position=1,  # Long
                quantity=quantity,
                entry_price=put_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            )
        ]
        
        position = StrategyPosition(
            symbol=symbol,
            strategy_name=self.strategy_name,
            legs=legs,
            entry_date=trade_date,
            expiry=expiry,
            spot_at_entry=spot,
            current_spot=spot
        )
        
        # Calculate max profit/loss
        max_profit, max_loss = self.calculate_max_profit_loss(position)
        position.max_profit = max_profit
        position.max_loss = max_loss
        
        return position

    def calculate_max_profit_loss(self, position: StrategyPosition) -> Tuple[Optional[float], Optional[float]]:
        """Calculate max profit (unlimited) and max loss (premium paid)."""
        # Max loss = total premium paid
        max_loss = sum(leg.entry_price * leg.quantity for leg in position.legs if leg.position > 0)
        
        # Max profit = unlimited (theoretically)
        max_profit = None  # Unlimited
        
        return max_profit, -max_loss  # Loss is negative

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven points for long straddle."""
        if not position.legs:
            return []
        
        strike = position.legs[0].strike  # Both legs have same strike (ATM)
        total_premium = sum(leg.entry_price * leg.quantity for leg in position.legs)
        
        # Breakeven points: Strike ± Total Premium
        return [strike - total_premium, strike + total_premium]

class ShortStraddle(BaseOptionsStrategy):
    """
    Short Straddle Strategy: Sell ATM Call + Sell ATM Put
    
    Characteristics:
    - Market Outlook: Neutral (expecting small moves)
    - Max Profit: Premium received
    - Max Loss: Unlimited
    - Breakeven: Strike ± Total Premium
    - Best Market: Low volatility, range-bound
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        super().__init__(client, greeks_calc, risk_free_rate)
        self.strategy_name = "Short Straddle"

    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """Create a short straddle position."""
        quantity = kwargs.get('quantity', 1)
        
        # Get options chain
        chain = self.client.snapshot_chain(symbol, trade_date)
        if chain.empty:
            return None
        
        # Find ATM strike
        strike_col = self._get_strike_column(chain)
        strikes = chain[strike_col].values
        atm_strike = strikes[np.argmin(np.abs(strikes - spot))]
        
        # Create option legs
        call_ticker = self._occ_ticker(symbol, expiry, atm_strike, True)
        put_ticker = self._occ_ticker(symbol, expiry, atm_strike, False)
        
        if not call_ticker or not put_ticker:
            return None
        
        # Get entry prices
        call_price = self.client.agg_close(call_ticker, trade_date)
        put_price = self.client.agg_close(put_ticker, trade_date)
        
        if call_price is None or put_price is None:
            # Fallback to theoretical pricing
            tau = (expiry - trade_date).days / 365.0
            iv = kwargs.get('implied_vol', 0.25)
            
            from bot.pricing import bs_price
            call_price = bs_price(spot, atm_strike, iv, tau, self.r, True)
            put_price = bs_price(spot, atm_strike, iv, tau, self.r, False)
        
        # Create legs (short positions)
        legs = [
            OptionLeg(
                ticker=call_ticker,
                strike=atm_strike,
                expiry=expiry,
                is_call=True,
                position=-1,  # Short
                quantity=quantity,
                entry_price=call_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            OptionLeg(
                ticker=put_ticker,
                strike=atm_strike,
                expiry=expiry,
                is_call=False,
                position=-1,  # Short
                quantity=quantity,
                entry_price=put_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            )
        ]
        
        position = StrategyPosition(
            symbol=symbol,
            strategy_name=self.strategy_name,
            legs=legs,
            entry_date=trade_date,
            expiry=expiry,
            spot_at_entry=spot,
            current_spot=spot
        )
        
        # Calculate max profit/loss
        max_profit, max_loss = self.calculate_max_profit_loss(position)
        position.max_profit = max_profit
        position.max_loss = max_loss
        
        return position

    def calculate_max_profit_loss(self, position: StrategyPosition) -> Tuple[Optional[float], Optional[float]]:
        """Calculate max profit (premium received) and max loss (unlimited)."""
        # Max profit = total premium received (when stock stays at strike)
        max_profit = sum(leg.entry_price * leg.quantity for leg in position.legs if leg.position < 0)
        
        # Max loss = unlimited (theoretically)
        max_loss = None  # Unlimited risk
        
        return max_profit, max_loss

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven points for short straddle."""
        if not position.legs:
            return []
        
        strike = position.legs[0].strike  # Both legs have same strike (ATM)
        total_premium = sum(leg.entry_price * leg.quantity for leg in position.legs)
        
        # Breakeven points: Strike ± Total Premium
        return [strike - total_premium, strike + total_premium]