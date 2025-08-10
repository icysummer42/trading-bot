"""
Long and Short Strangle strategies implementation.

Long Strangle: Buy OTM call + Buy OTM put (cheaper volatility play)
Short Strangle: Sell OTM call + Sell OTM put (income strategy with wider range)
"""

from __future__ import annotations
import datetime as dt
from typing import List, Tuple, Optional
import numpy as np

from bot.strategy.base_strategy import BaseOptionsStrategy, StrategyPosition, OptionLeg
from bot.polygon_client import PolygonClient
from bot.greeks import GreeksCalculator

class LongStrangle(BaseOptionsStrategy):
    """
    Long Strangle Strategy: Buy OTM Call + Buy OTM Put
    
    Characteristics:
    - Market Outlook: Neutral (expecting big move, cheaper than straddle)
    - Max Profit: Unlimited
    - Max Loss: Premium paid
    - Breakeven: Call Strike + Total Premium, Put Strike - Total Premium
    - Best Market: High volatility, big moves (cheaper entry than straddle)
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        super().__init__(client, greeks_calc, risk_free_rate)
        self.strategy_name = "Long Strangle"

    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """
        Create a long strangle position.
        
        Args:
            symbol: Underlying symbol
            spot: Current spot price
            trade_date: Entry date
            expiry: Options expiry date
            **kwargs: delta_call, delta_put, quantity, implied_vol
        
        Returns:
            StrategyPosition or None if creation fails
        """
        quantity = kwargs.get('quantity', 1)
        delta_call = kwargs.get('delta_call', 0.25)  # OTM call target delta
        delta_put = kwargs.get('delta_put', -0.25)   # OTM put target delta
        
        # Get options chain
        chain = self.client.snapshot_chain(symbol, trade_date)
        if chain.empty:
            return None
        
        # Find OTM call and put strikes
        call_strikes = self._select_strikes_by_delta(chain, spot, [delta_call], True)
        put_strikes = self._select_strikes_by_delta(chain, spot, [abs(delta_put)], False)
        
        if not call_strikes or not put_strikes:
            # Fallback: Use strikes based on percentage OTM
            strike_col = self._get_strike_column(chain)
            strikes = sorted(chain[strike_col].values)
            
            # Find strikes approximately 5-10% OTM
            call_strike = next((s for s in strikes if s > spot * 1.05), strikes[-1])
            put_strike = next((s for s in reversed(strikes) if s < spot * 0.95), strikes[0])
        else:
            call_strike = call_strikes[0]
            put_strike = put_strikes[0]
        
        # Create option legs
        call_ticker = self._occ_ticker(symbol, expiry, call_strike, True)
        put_ticker = self._occ_ticker(symbol, expiry, put_strike, False)
        
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
            call_price = bs_price(spot, call_strike, iv, tau, self.r, True)
            put_price = bs_price(spot, put_strike, iv, tau, self.r, False)
        
        # Create legs
        legs = [
            OptionLeg(
                ticker=call_ticker,
                strike=call_strike,
                expiry=expiry,
                is_call=True,
                position=1,  # Long
                quantity=quantity,
                entry_price=call_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            OptionLeg(
                ticker=put_ticker,
                strike=put_strike,
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
        
        return max_profit, -max_loss

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven points for long strangle."""
        if len(position.legs) < 2:
            return []
        
        call_leg = next(leg for leg in position.legs if leg.is_call)
        put_leg = next(leg for leg in position.legs if not leg.is_call)
        
        total_premium = sum(leg.entry_price * leg.quantity for leg in position.legs)
        
        # Breakeven points: Call Strike + Total Premium, Put Strike - Total Premium
        return [
            put_leg.strike - total_premium,
            call_leg.strike + total_premium
        ]

class ShortStrangle(BaseOptionsStrategy):
    """
    Short Strangle Strategy: Sell OTM Call + Sell OTM Put
    
    Characteristics:
    - Market Outlook: Neutral (expecting small moves within range)
    - Max Profit: Premium received
    - Max Loss: Unlimited
    - Breakeven: Call Strike + Total Premium, Put Strike - Total Premium
    - Best Market: Low volatility, range-bound (wider profit range than short straddle)
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        super().__init__(client, greeks_calc, risk_free_rate)
        self.strategy_name = "Short Strangle"

    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """Create a short strangle position."""
        quantity = kwargs.get('quantity', 1)
        delta_call = kwargs.get('delta_call', 0.20)  # OTM call target delta
        delta_put = kwargs.get('delta_put', -0.20)   # OTM put target delta
        
        # Get options chain
        chain = self.client.snapshot_chain(symbol, trade_date)
        if chain.empty:
            return None
        
        # Find OTM call and put strikes
        call_strikes = self._select_strikes_by_delta(chain, spot, [delta_call], True)
        put_strikes = self._select_strikes_by_delta(chain, spot, [abs(delta_put)], False)
        
        if not call_strikes or not put_strikes:
            # Fallback: Use strikes based on percentage OTM
            strike_col = self._get_strike_column(chain)
            strikes = sorted(chain[strike_col].values)
            
            # Find strikes approximately 5-10% OTM
            call_strike = next((s for s in strikes if s > spot * 1.05), strikes[-1])
            put_strike = next((s for s in reversed(strikes) if s < spot * 0.95), strikes[0])
        else:
            call_strike = call_strikes[0]
            put_strike = put_strikes[0]
        
        # Create option legs
        call_ticker = self._occ_ticker(symbol, expiry, call_strike, True)
        put_ticker = self._occ_ticker(symbol, expiry, put_strike, False)
        
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
            call_price = bs_price(spot, call_strike, iv, tau, self.r, True)
            put_price = bs_price(spot, put_strike, iv, tau, self.r, False)
        
        # Create legs (short positions)
        legs = [
            OptionLeg(
                ticker=call_ticker,
                strike=call_strike,
                expiry=expiry,
                is_call=True,
                position=-1,  # Short
                quantity=quantity,
                entry_price=call_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            OptionLeg(
                ticker=put_ticker,
                strike=put_strike,
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
        # Max profit = total premium received
        max_profit = sum(leg.entry_price * leg.quantity for leg in position.legs if leg.position < 0)
        
        # Max loss = unlimited (theoretically)
        max_loss = None  # Unlimited risk
        
        return max_profit, max_loss

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven points for short strangle."""
        if len(position.legs) < 2:
            return []
        
        call_leg = next(leg for leg in position.legs if leg.is_call)
        put_leg = next(leg for leg in position.legs if not leg.is_call)
        
        total_premium = sum(leg.entry_price * leg.quantity for leg in position.legs)
        
        # Breakeven points: Call Strike + Total Premium, Put Strike - Total Premium
        return [
            put_leg.strike - total_premium,
            call_leg.strike + total_premium
        ]