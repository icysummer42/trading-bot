"""
Butterfly spread strategies implementation.

Long Call Butterfly: Buy ITM call + Sell 2 ATM calls + Buy OTM call
Long Put Butterfly: Buy ITM put + Sell 2 ATM puts + Buy OTM put
Short Call Butterfly: Sell ITM call + Buy 2 ATM calls + Sell OTM call
Short Put Butterfly: Sell ITM put + Buy 2 ATM puts + Sell OTM put
"""

from __future__ import annotations
import datetime as dt
from typing import List, Tuple, Optional
import numpy as np

from bot.strategy.base_strategy import BaseOptionsStrategy, StrategyPosition, OptionLeg
from bot.polygon_client import PolygonClient
from bot.greeks import GreeksCalculator

class LongCallButterfly(BaseOptionsStrategy):
    """
    Long Call Butterfly: Buy Lower Strike + Sell 2 Middle Strikes + Buy Higher Strike
    
    Characteristics:
    - Market Outlook: Neutral (expect minimal movement around middle strike)
    - Max Profit: Middle Strike - Lower Strike - Net Debit
    - Max Loss: Net Debit Paid
    - Breakeven: Lower Strike + Net Debit, Upper Strike - Net Debit
    - Best Market: Low volatility, range-bound around middle strike
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        super().__init__(client, greeks_calc, risk_free_rate)
        self.strategy_name = "Long Call Butterfly"

    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """
        Create a long call butterfly position.
        
        Args:
            symbol: Underlying symbol
            spot: Current spot price
            trade_date: Entry date
            expiry: Options expiry date
            **kwargs: wing_width, quantity, implied_vol
        
        Returns:
            StrategyPosition or None if creation fails
        """
        quantity = kwargs.get('quantity', 1)
        wing_width = kwargs.get('wing_width', None)  # Distance between strikes
        
        # Get options chain
        chain = self.client.snapshot_chain(symbol, trade_date)
        if chain.empty:
            return None
        
        # Filter for calls only
        call_chain = self._filter_chain_by_type(chain, "call")
        if call_chain.empty:
            return None
        
        strike_col = self._get_strike_column(call_chain)
        strikes = sorted(call_chain[strike_col].values)
        
        # Select strikes for butterfly
        # Middle strike should be closest to current spot (ATM)
        middle_strike = min(strikes, key=lambda x: abs(x - spot))
        
        if wing_width:
            # Fixed wing width
            lower_strike = middle_strike - wing_width
            upper_strike = middle_strike + wing_width
            
            # Find closest available strikes
            lower_strike = min(strikes, key=lambda x: abs(x - lower_strike))
            upper_strike = min(strikes, key=lambda x: abs(x - upper_strike))
        else:
            # Use strikes based on percentage of spot
            lower_strike = min(strikes, key=lambda x: abs(x - spot * 0.95))
            upper_strike = min(strikes, key=lambda x: abs(x - spot * 1.05))
        
        # Ensure proper ordering: lower < middle < upper
        if not (lower_strike < middle_strike < upper_strike):
            return None
        
        # Ensure symmetric wings (approximately)
        wing1 = middle_strike - lower_strike
        wing2 = upper_strike - middle_strike
        if abs(wing1 - wing2) > wing1 * 0.1:  # Allow 10% asymmetry
            # Adjust to make symmetric
            wing_avg = (wing1 + wing2) / 2
            lower_strike = middle_strike - wing_avg
            upper_strike = middle_strike + wing_avg
            
            # Find closest available strikes
            lower_strike = min(strikes, key=lambda x: abs(x - lower_strike))
            upper_strike = min(strikes, key=lambda x: abs(x - upper_strike))
        
        # Create option tickers
        lower_call_ticker = self._occ_ticker(symbol, expiry, lower_strike, True)
        middle_call_ticker = self._occ_ticker(symbol, expiry, middle_strike, True)
        upper_call_ticker = self._occ_ticker(symbol, expiry, upper_strike, True)
        
        if not all([lower_call_ticker, middle_call_ticker, upper_call_ticker]):
            return None
        
        # Get entry prices
        lower_call_price = self.client.agg_close(lower_call_ticker, trade_date)
        middle_call_price = self.client.agg_close(middle_call_ticker, trade_date)
        upper_call_price = self.client.agg_close(upper_call_ticker, trade_date)
        
        if any(price is None for price in [lower_call_price, middle_call_price, upper_call_price]):
            # Fallback to theoretical pricing
            tau = (expiry - trade_date).days / 365.0
            iv = kwargs.get('implied_vol', 0.25)
            
            from bot.pricing import bs_price
            lower_call_price = bs_price(spot, lower_strike, iv, tau, self.r, True)
            middle_call_price = bs_price(spot, middle_strike, iv, tau, self.r, True)
            upper_call_price = bs_price(spot, upper_strike, iv, tau, self.r, True)
        
        # Create legs
        legs = [
            # Buy lower strike call
            OptionLeg(
                ticker=lower_call_ticker,
                strike=lower_strike,
                expiry=expiry,
                is_call=True,
                position=1,  # Long
                quantity=quantity,
                entry_price=lower_call_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            # Sell 2 middle strike calls
            OptionLeg(
                ticker=middle_call_ticker,
                strike=middle_strike,
                expiry=expiry,
                is_call=True,
                position=-1,  # Short
                quantity=2 * quantity,
                entry_price=middle_call_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            # Buy upper strike call
            OptionLeg(
                ticker=upper_call_ticker,
                strike=upper_strike,
                expiry=expiry,
                is_call=True,
                position=1,  # Long
                quantity=quantity,
                entry_price=upper_call_price,
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
        """Calculate max profit and max loss for long call butterfly."""
        # Sort legs by strike
        legs_by_strike = sorted(position.legs, key=lambda x: x.strike)
        lower_leg = legs_by_strike[0]
        middle_leg = legs_by_strike[1]  # This has quantity = 2
        upper_leg = legs_by_strike[2]
        
        # Net debit = what we pay
        net_debit = (
            (lower_leg.entry_price * lower_leg.quantity) +
            (upper_leg.entry_price * upper_leg.quantity) -
            (middle_leg.entry_price * middle_leg.quantity)
        )
        
        # Strike difference (should be same for both wings in symmetric butterfly)
        strike_diff = middle_leg.strike - lower_leg.strike
        
        # Max profit = Strike difference - Net debit (when stock exactly at middle strike)
        max_profit = (strike_diff * lower_leg.quantity) - net_debit
        
        # Max loss = Net debit (when stock below lower or above upper strike)
        max_loss = -net_debit
        
        return max_profit, max_loss

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven points for long call butterfly."""
        legs_by_strike = sorted(position.legs, key=lambda x: x.strike)
        lower_leg = legs_by_strike[0]
        middle_leg = legs_by_strike[1]
        upper_leg = legs_by_strike[2]
        
        # Net debit
        net_debit = (
            (lower_leg.entry_price * lower_leg.quantity) +
            (upper_leg.entry_price * upper_leg.quantity) -
            (middle_leg.entry_price * middle_leg.quantity)
        )
        
        net_debit_per_unit = net_debit / lower_leg.quantity
        
        # Breakeven points
        return [
            lower_leg.strike + net_debit_per_unit,
            upper_leg.strike - net_debit_per_unit
        ]

class LongPutButterfly(BaseOptionsStrategy):
    """
    Long Put Butterfly: Buy Lower Strike + Sell 2 Middle Strikes + Buy Higher Strike
    
    Similar to call butterfly but uses puts. Profit profile is identical.
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        super().__init__(client, greeks_calc, risk_free_rate)
        self.strategy_name = "Long Put Butterfly"

    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """Create a long put butterfly position."""
        quantity = kwargs.get('quantity', 1)
        wing_width = kwargs.get('wing_width', None)
        
        # Get options chain
        chain = self.client.snapshot_chain(symbol, trade_date)
        if chain.empty:
            return None
        
        # Filter for puts only
        put_chain = self._filter_chain_by_type(chain, "put")
        if put_chain.empty:
            return None
        
        strike_col = self._get_strike_column(put_chain)
        strikes = sorted(put_chain[strike_col].values)
        
        # Select strikes (same logic as call butterfly)
        middle_strike = min(strikes, key=lambda x: abs(x - spot))
        
        if wing_width:
            lower_strike = middle_strike - wing_width
            upper_strike = middle_strike + wing_width
            
            lower_strike = min(strikes, key=lambda x: abs(x - lower_strike))
            upper_strike = min(strikes, key=lambda x: abs(x - upper_strike))
        else:
            lower_strike = min(strikes, key=lambda x: abs(x - spot * 0.95))
            upper_strike = min(strikes, key=lambda x: abs(x - spot * 1.05))
        
        if not (lower_strike < middle_strike < upper_strike):
            return None
        
        # Create option tickers
        lower_put_ticker = self._occ_ticker(symbol, expiry, lower_strike, False)
        middle_put_ticker = self._occ_ticker(symbol, expiry, middle_strike, False)
        upper_put_ticker = self._occ_ticker(symbol, expiry, upper_strike, False)
        
        if not all([lower_put_ticker, middle_put_ticker, upper_put_ticker]):
            return None
        
        # Get entry prices
        lower_put_price = self.client.agg_close(lower_put_ticker, trade_date)
        middle_put_price = self.client.agg_close(middle_put_ticker, trade_date)
        upper_put_price = self.client.agg_close(upper_put_ticker, trade_date)
        
        if any(price is None for price in [lower_put_price, middle_put_price, upper_put_price]):
            # Fallback to theoretical pricing
            tau = (expiry - trade_date).days / 365.0
            iv = kwargs.get('implied_vol', 0.25)
            
            from bot.pricing import bs_price
            lower_put_price = bs_price(spot, lower_strike, iv, tau, self.r, False)
            middle_put_price = bs_price(spot, middle_strike, iv, tau, self.r, False)
            upper_put_price = bs_price(spot, upper_strike, iv, tau, self.r, False)
        
        # Create legs
        legs = [
            # Buy lower strike put
            OptionLeg(
                ticker=lower_put_ticker,
                strike=lower_strike,
                expiry=expiry,
                is_call=False,
                position=1,  # Long
                quantity=quantity,
                entry_price=lower_put_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            # Sell 2 middle strike puts
            OptionLeg(
                ticker=middle_put_ticker,
                strike=middle_strike,
                expiry=expiry,
                is_call=False,
                position=-1,  # Short
                quantity=2 * quantity,
                entry_price=middle_put_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            # Buy upper strike put
            OptionLeg(
                ticker=upper_put_ticker,
                strike=upper_strike,
                expiry=expiry,
                is_call=False,
                position=1,  # Long
                quantity=quantity,
                entry_price=upper_put_price,
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
        """Calculate max profit and max loss for long put butterfly."""
        # Same calculation as call butterfly
        legs_by_strike = sorted(position.legs, key=lambda x: x.strike)
        lower_leg = legs_by_strike[0]
        middle_leg = legs_by_strike[1]
        upper_leg = legs_by_strike[2]
        
        # Net debit
        net_debit = (
            (lower_leg.entry_price * lower_leg.quantity) +
            (upper_leg.entry_price * upper_leg.quantity) -
            (middle_leg.entry_price * middle_leg.quantity)
        )
        
        # Strike difference
        strike_diff = middle_leg.strike - lower_leg.strike
        
        # Max profit = Strike difference - Net debit
        max_profit = (strike_diff * lower_leg.quantity) - net_debit
        
        # Max loss = Net debit
        max_loss = -net_debit
        
        return max_profit, max_loss

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven points for long put butterfly."""
        legs_by_strike = sorted(position.legs, key=lambda x: x.strike)
        lower_leg = legs_by_strike[0]
        middle_leg = legs_by_strike[1]
        upper_leg = legs_by_strike[2]
        
        # Net debit
        net_debit = (
            (lower_leg.entry_price * lower_leg.quantity) +
            (upper_leg.entry_price * upper_leg.quantity) -
            (middle_leg.entry_price * middle_leg.quantity)
        )
        
        net_debit_per_unit = net_debit / lower_leg.quantity
        
        # Breakeven points
        return [
            lower_leg.strike + net_debit_per_unit,
            upper_leg.strike - net_debit_per_unit
        ]

class ShortCallButterfly(BaseOptionsStrategy):
    """
    Short Call Butterfly: Sell Lower Strike + Buy 2 Middle Strikes + Sell Higher Strike
    
    Characteristics:
    - Market Outlook: Expecting volatility (opposite of long butterfly)
    - Max Profit: Net Credit Received
    - Max Loss: Strike Difference - Net Credit
    - Best Market: High volatility, breakout moves
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        super().__init__(client, greeks_calc, risk_free_rate)
        self.strategy_name = "Short Call Butterfly"

    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """Create a short call butterfly position."""
        quantity = kwargs.get('quantity', 1)
        wing_width = kwargs.get('wing_width', None)
        
        # Get options chain
        chain = self.client.snapshot_chain(symbol, trade_date)
        if chain.empty:
            return None
        
        # Filter for calls only
        call_chain = self._filter_chain_by_type(chain, "call")
        if call_chain.empty:
            return None
        
        strike_col = self._get_strike_column(call_chain)
        strikes = sorted(call_chain[strike_col].values)
        
        # Select strikes (same as long butterfly)
        middle_strike = min(strikes, key=lambda x: abs(x - spot))
        
        if wing_width:
            lower_strike = middle_strike - wing_width
            upper_strike = middle_strike + wing_width
            
            lower_strike = min(strikes, key=lambda x: abs(x - lower_strike))
            upper_strike = min(strikes, key=lambda x: abs(x - upper_strike))
        else:
            lower_strike = min(strikes, key=lambda x: abs(x - spot * 0.95))
            upper_strike = min(strikes, key=lambda x: abs(x - spot * 1.05))
        
        if not (lower_strike < middle_strike < upper_strike):
            return None
        
        # Create option tickers
        lower_call_ticker = self._occ_ticker(symbol, expiry, lower_strike, True)
        middle_call_ticker = self._occ_ticker(symbol, expiry, middle_strike, True)
        upper_call_ticker = self._occ_ticker(symbol, expiry, upper_strike, True)
        
        if not all([lower_call_ticker, middle_call_ticker, upper_call_ticker]):
            return None
        
        # Get entry prices
        lower_call_price = self.client.agg_close(lower_call_ticker, trade_date)
        middle_call_price = self.client.agg_close(middle_call_ticker, trade_date)
        upper_call_price = self.client.agg_close(upper_call_ticker, trade_date)
        
        if any(price is None for price in [lower_call_price, middle_call_price, upper_call_price]):
            # Fallback to theoretical pricing
            tau = (expiry - trade_date).days / 365.0
            iv = kwargs.get('implied_vol', 0.25)
            
            from bot.pricing import bs_price
            lower_call_price = bs_price(spot, lower_strike, iv, tau, self.r, True)
            middle_call_price = bs_price(spot, middle_strike, iv, tau, self.r, True)
            upper_call_price = bs_price(spot, upper_strike, iv, tau, self.r, True)
        
        # Create legs (opposite positions from long butterfly)
        legs = [
            # Sell lower strike call
            OptionLeg(
                ticker=lower_call_ticker,
                strike=lower_strike,
                expiry=expiry,
                is_call=True,
                position=-1,  # Short
                quantity=quantity,
                entry_price=lower_call_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            # Buy 2 middle strike calls
            OptionLeg(
                ticker=middle_call_ticker,
                strike=middle_strike,
                expiry=expiry,
                is_call=True,
                position=1,  # Long
                quantity=2 * quantity,
                entry_price=middle_call_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            # Sell upper strike call
            OptionLeg(
                ticker=upper_call_ticker,
                strike=upper_strike,
                expiry=expiry,
                is_call=True,
                position=-1,  # Short
                quantity=quantity,
                entry_price=upper_call_price,
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
        """Calculate max profit and max loss for short call butterfly."""
        legs_by_strike = sorted(position.legs, key=lambda x: x.strike)
        lower_leg = legs_by_strike[0]
        middle_leg = legs_by_strike[1]
        upper_leg = legs_by_strike[2]
        
        # Net credit = what we receive (opposite of long butterfly)
        net_credit = (
            (lower_leg.entry_price * abs(lower_leg.position) * lower_leg.quantity) +
            (upper_leg.entry_price * abs(upper_leg.position) * upper_leg.quantity) -
            (middle_leg.entry_price * middle_leg.position * middle_leg.quantity)
        )
        
        # Strike difference
        strike_diff = middle_leg.strike - lower_leg.strike
        
        # Max profit = Net credit (when stock below lower or above upper strike)
        max_profit = net_credit
        
        # Max loss = Strike difference - Net credit (when stock exactly at middle strike)
        max_loss = -((strike_diff * lower_leg.quantity) - net_credit)
        
        return max_profit, max_loss

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven points for short call butterfly."""
        legs_by_strike = sorted(position.legs, key=lambda x: x.strike)
        lower_leg = legs_by_strike[0]
        middle_leg = legs_by_strike[1]
        upper_leg = legs_by_strike[2]
        
        # Net credit
        net_credit = (
            (lower_leg.entry_price * abs(lower_leg.position) * lower_leg.quantity) +
            (upper_leg.entry_price * abs(upper_leg.position) * upper_leg.quantity) -
            (middle_leg.entry_price * middle_leg.position * middle_leg.quantity)
        )
        
        net_credit_per_unit = net_credit / lower_leg.quantity
        
        # Breakeven points
        return [
            lower_leg.strike + net_credit_per_unit,
            upper_leg.strike - net_credit_per_unit
        ]