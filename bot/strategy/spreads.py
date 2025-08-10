"""
Bull and Bear spread strategies implementation.

Bull Call Spread: Buy ITM call + Sell OTM call (bullish, limited risk/reward)
Bull Put Spread: Sell ITM put + Buy OTM put (bullish, credit spread)
Bear Call Spread: Sell ITM call + Buy OTM call (bearish, credit spread)
Bear Put Spread: Buy ITM put + Sell OTM put (bearish, limited risk/reward)
"""

from __future__ import annotations
import datetime as dt
from typing import List, Tuple, Optional
import numpy as np

from bot.strategy.base_strategy import BaseOptionsStrategy, StrategyPosition, OptionLeg
from bot.polygon_client import PolygonClient
from bot.greeks import GreeksCalculator

class BullCallSpread(BaseOptionsStrategy):
    """
    Bull Call Spread: Buy Lower Strike Call + Sell Higher Strike Call
    
    Characteristics:
    - Market Outlook: Moderately Bullish
    - Max Profit: Strike Difference - Net Debit
    - Max Loss: Net Debit Paid
    - Breakeven: Lower Strike + Net Debit
    - Best Market: Moderate upward movement expected
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        super().__init__(client, greeks_calc, risk_free_rate)
        self.strategy_name = "Bull Call Spread"

    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """
        Create a bull call spread position.
        
        Args:
            symbol: Underlying symbol
            spot: Current spot price
            trade_date: Entry date
            expiry: Options expiry date
            **kwargs: long_delta, short_delta, quantity, implied_vol, spread_width
        
        Returns:
            StrategyPosition or None if creation fails
        """
        quantity = kwargs.get('quantity', 1)
        long_delta = kwargs.get('long_delta', 0.60)   # ITM call
        short_delta = kwargs.get('short_delta', 0.30) # OTM call
        spread_width = kwargs.get('spread_width', None)  # Optional fixed spread width
        
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
        
        # Select strikes
        if spread_width:
            # Fixed spread width approach
            long_strike = min(strikes, key=lambda x: abs(x - spot * 0.98))  # Slightly ITM
            short_strike = long_strike + spread_width
            # Ensure short strike exists
            if short_strike not in strikes:
                short_strike = min(strikes, key=lambda x: abs(x - short_strike))
        else:
            # Delta-based approach
            long_strikes = self._select_strikes_by_delta(call_chain, spot, [long_delta], True)
            short_strikes = self._select_strikes_by_delta(call_chain, spot, [short_delta], True)
            
            if not long_strikes or not short_strikes:
                # Fallback: percentage-based selection
                long_strike = next((s for s in strikes if s >= spot * 0.98), strikes[0])
                short_strike = next((s for s in strikes if s >= spot * 1.05), strikes[-1])
            else:
                long_strike = long_strikes[0]
                short_strike = short_strikes[0]
        
        # Ensure long strike < short strike
        if long_strike >= short_strike:
            return None
        
        # Create option legs
        long_call_ticker = self._occ_ticker(symbol, expiry, long_strike, True)
        short_call_ticker = self._occ_ticker(symbol, expiry, short_strike, True)
        
        if not long_call_ticker or not short_call_ticker:
            return None
        
        # Get entry prices
        long_call_price = self.client.agg_close(long_call_ticker, trade_date)
        short_call_price = self.client.agg_close(short_call_ticker, trade_date)
        
        if long_call_price is None or short_call_price is None:
            # Fallback to theoretical pricing
            tau = (expiry - trade_date).days / 365.0
            iv = kwargs.get('implied_vol', 0.25)
            
            from bot.pricing import bs_price
            long_call_price = bs_price(spot, long_strike, iv, tau, self.r, True)
            short_call_price = bs_price(spot, short_strike, iv, tau, self.r, True)
        
        # Create legs
        legs = [
            OptionLeg(
                ticker=long_call_ticker,
                strike=long_strike,
                expiry=expiry,
                is_call=True,
                position=1,  # Long
                quantity=quantity,
                entry_price=long_call_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            OptionLeg(
                ticker=short_call_ticker,
                strike=short_strike,
                expiry=expiry,
                is_call=True,
                position=-1,  # Short
                quantity=quantity,
                entry_price=short_call_price,
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
        """Calculate max profit and max loss for bull call spread."""
        long_leg = next(leg for leg in position.legs if leg.position > 0)
        short_leg = next(leg for leg in position.legs if leg.position < 0)
        
        # Net debit = long price - short price (what we pay)
        net_debit = (long_leg.entry_price - short_leg.entry_price) * long_leg.quantity
        
        # Strike difference
        strike_diff = short_leg.strike - long_leg.strike
        
        # Max profit = Strike difference - Net debit (when stock above short strike)
        max_profit = (strike_diff * long_leg.quantity) - net_debit
        
        # Max loss = Net debit (when stock below long strike)
        max_loss = -net_debit
        
        return max_profit, max_loss

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven point for bull call spread."""
        long_leg = next(leg for leg in position.legs if leg.position > 0)
        short_leg = next(leg for leg in position.legs if leg.position < 0)
        
        net_debit = long_leg.entry_price - short_leg.entry_price
        
        # Breakeven = Lower strike + Net debit
        return [long_leg.strike + net_debit]

class BearPutSpread(BaseOptionsStrategy):
    """
    Bear Put Spread: Buy Higher Strike Put + Sell Lower Strike Put
    
    Characteristics:
    - Market Outlook: Moderately Bearish
    - Max Profit: Strike Difference - Net Debit
    - Max Loss: Net Debit Paid
    - Breakeven: Higher Strike - Net Debit
    - Best Market: Moderate downward movement expected
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        super().__init__(client, greeks_calc, risk_free_rate)
        self.strategy_name = "Bear Put Spread"

    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """Create a bear put spread position."""
        quantity = kwargs.get('quantity', 1)
        long_delta = kwargs.get('long_delta', -0.60)   # ITM put (higher strike)
        short_delta = kwargs.get('short_delta', -0.30) # OTM put (lower strike)
        spread_width = kwargs.get('spread_width', None)
        
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
        
        # Select strikes
        if spread_width:
            # Fixed spread width approach
            long_strike = min(strikes, key=lambda x: abs(x - spot * 1.02))  # Slightly ITM
            short_strike = long_strike - spread_width
            # Ensure short strike exists
            if short_strike not in strikes:
                short_strike = min(strikes, key=lambda x: abs(x - short_strike))
        else:
            # Delta-based approach
            long_strikes = self._select_strikes_by_delta(put_chain, spot, [abs(long_delta)], False)
            short_strikes = self._select_strikes_by_delta(put_chain, spot, [abs(short_delta)], False)
            
            if not long_strikes or not short_strikes:
                # Fallback: percentage-based selection
                long_strike = next((s for s in reversed(strikes) if s <= spot * 1.02), strikes[-1])
                short_strike = next((s for s in reversed(strikes) if s <= spot * 0.95), strikes[0])
            else:
                long_strike = long_strikes[0]
                short_strike = short_strikes[0]
        
        # Ensure long strike > short strike
        if long_strike <= short_strike:
            return None
        
        # Create option legs
        long_put_ticker = self._occ_ticker(symbol, expiry, long_strike, False)
        short_put_ticker = self._occ_ticker(symbol, expiry, short_strike, False)
        
        if not long_put_ticker or not short_put_ticker:
            return None
        
        # Get entry prices
        long_put_price = self.client.agg_close(long_put_ticker, trade_date)
        short_put_price = self.client.agg_close(short_put_ticker, trade_date)
        
        if long_put_price is None or short_put_price is None:
            # Fallback to theoretical pricing
            tau = (expiry - trade_date).days / 365.0
            iv = kwargs.get('implied_vol', 0.25)
            
            from bot.pricing import bs_price
            long_put_price = bs_price(spot, long_strike, iv, tau, self.r, False)
            short_put_price = bs_price(spot, short_strike, iv, tau, self.r, False)
        
        # Create legs
        legs = [
            OptionLeg(
                ticker=long_put_ticker,
                strike=long_strike,
                expiry=expiry,
                is_call=False,
                position=1,  # Long
                quantity=quantity,
                entry_price=long_put_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            OptionLeg(
                ticker=short_put_ticker,
                strike=short_strike,
                expiry=expiry,
                is_call=False,
                position=-1,  # Short
                quantity=quantity,
                entry_price=short_put_price,
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
        """Calculate max profit and max loss for bear put spread."""
        long_leg = next(leg for leg in position.legs if leg.position > 0)
        short_leg = next(leg for leg in position.legs if leg.position < 0)
        
        # Net debit = long price - short price
        net_debit = (long_leg.entry_price - short_leg.entry_price) * long_leg.quantity
        
        # Strike difference
        strike_diff = long_leg.strike - short_leg.strike
        
        # Max profit = Strike difference - Net debit (when stock below short strike)
        max_profit = (strike_diff * long_leg.quantity) - net_debit
        
        # Max loss = Net debit (when stock above long strike)
        max_loss = -net_debit
        
        return max_profit, max_loss

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven point for bear put spread."""
        long_leg = next(leg for leg in position.legs if leg.position > 0)
        short_leg = next(leg for leg in position.legs if leg.position < 0)
        
        net_debit = long_leg.entry_price - short_leg.entry_price
        
        # Breakeven = Higher strike - Net debit
        return [long_leg.strike - net_debit]

class BearCallSpread(BaseOptionsStrategy):
    """
    Bear Call Spread: Sell Lower Strike Call + Buy Higher Strike Call
    
    Characteristics:
    - Market Outlook: Moderately Bearish
    - Max Profit: Net Credit Received
    - Max Loss: Strike Difference - Net Credit
    - Breakeven: Lower Strike + Net Credit
    - Best Market: Range-bound or modest decline
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        super().__init__(client, greeks_calc, risk_free_rate)
        self.strategy_name = "Bear Call Spread"

    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """Create a bear call spread position (credit spread)."""
        quantity = kwargs.get('quantity', 1)
        short_delta = kwargs.get('short_delta', 0.30)  # OTM call (lower strike) - sell
        long_delta = kwargs.get('long_delta', 0.15)    # More OTM call (higher strike) - buy
        
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
        
        # Select strikes (short strike < long strike)
        short_strikes = self._select_strikes_by_delta(call_chain, spot, [short_delta], True)
        long_strikes = self._select_strikes_by_delta(call_chain, spot, [long_delta], True)
        
        if not short_strikes or not long_strikes:
            # Fallback: percentage-based selection
            short_strike = next((s for s in strikes if s >= spot * 1.03), strikes[-1])
            long_strike = next((s for s in strikes if s >= spot * 1.08), strikes[-1])
        else:
            short_strike = short_strikes[0]
            long_strike = long_strikes[0]
        
        # Ensure short strike < long strike
        if short_strike >= long_strike:
            return None
        
        # Create option legs
        short_call_ticker = self._occ_ticker(symbol, expiry, short_strike, True)
        long_call_ticker = self._occ_ticker(symbol, expiry, long_strike, True)
        
        if not short_call_ticker or not long_call_ticker:
            return None
        
        # Get entry prices
        short_call_price = self.client.agg_close(short_call_ticker, trade_date)
        long_call_price = self.client.agg_close(long_call_ticker, trade_date)
        
        if short_call_price is None or long_call_price is None:
            # Fallback to theoretical pricing
            tau = (expiry - trade_date).days / 365.0
            iv = kwargs.get('implied_vol', 0.25)
            
            from bot.pricing import bs_price
            short_call_price = bs_price(spot, short_strike, iv, tau, self.r, True)
            long_call_price = bs_price(spot, long_strike, iv, tau, self.r, True)
        
        # Create legs
        legs = [
            OptionLeg(
                ticker=short_call_ticker,
                strike=short_strike,
                expiry=expiry,
                is_call=True,
                position=-1,  # Short
                quantity=quantity,
                entry_price=short_call_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            OptionLeg(
                ticker=long_call_ticker,
                strike=long_strike,
                expiry=expiry,
                is_call=True,
                position=1,   # Long
                quantity=quantity,
                entry_price=long_call_price,
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
        """Calculate max profit and max loss for bear call spread."""
        short_leg = next(leg for leg in position.legs if leg.position < 0)
        long_leg = next(leg for leg in position.legs if leg.position > 0)
        
        # Net credit = short price - long price (what we receive)
        net_credit = (short_leg.entry_price - long_leg.entry_price) * short_leg.quantity
        
        # Strike difference
        strike_diff = long_leg.strike - short_leg.strike
        
        # Max profit = Net credit (when stock below short strike)
        max_profit = net_credit
        
        # Max loss = Strike difference - Net credit (when stock above long strike)
        max_loss = -(strike_diff * short_leg.quantity - net_credit)
        
        return max_profit, max_loss

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven point for bear call spread."""
        short_leg = next(leg for leg in position.legs if leg.position < 0)
        long_leg = next(leg for leg in position.legs if leg.position > 0)
        
        net_credit = short_leg.entry_price - long_leg.entry_price
        
        # Breakeven = Lower strike + Net credit
        return [short_leg.strike + net_credit]

class BullPutSpread(BaseOptionsStrategy):
    """
    Bull Put Spread: Sell Higher Strike Put + Buy Lower Strike Put
    
    Characteristics:
    - Market Outlook: Moderately Bullish
    - Max Profit: Net Credit Received
    - Max Loss: Strike Difference - Net Credit
    - Breakeven: Higher Strike - Net Credit
    - Best Market: Range-bound or modest rise
    """

    def __init__(self, client: PolygonClient, greeks_calc: GreeksCalculator = None, 
                 risk_free_rate: float = 0.02):
        super().__init__(client, greeks_calc, risk_free_rate)
        self.strategy_name = "Bull Put Spread"

    def create_position(self, symbol: str, spot: float, trade_date: dt.date,
                       expiry: dt.date, **kwargs) -> Optional[StrategyPosition]:
        """Create a bull put spread position (credit spread)."""
        quantity = kwargs.get('quantity', 1)
        short_delta = kwargs.get('short_delta', -0.30)  # OTM put (higher strike) - sell
        long_delta = kwargs.get('long_delta', -0.15)    # More OTM put (lower strike) - buy
        
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
        
        # Select strikes (short strike > long strike)
        short_strikes = self._select_strikes_by_delta(put_chain, spot, [abs(short_delta)], False)
        long_strikes = self._select_strikes_by_delta(put_chain, spot, [abs(long_delta)], False)
        
        if not short_strikes or not long_strikes:
            # Fallback: percentage-based selection
            short_strike = next((s for s in reversed(strikes) if s <= spot * 0.97), strikes[0])
            long_strike = next((s for s in reversed(strikes) if s <= spot * 0.92), strikes[0])
        else:
            short_strike = short_strikes[0]
            long_strike = long_strikes[0]
        
        # Ensure short strike > long strike
        if short_strike <= long_strike:
            return None
        
        # Create option legs
        short_put_ticker = self._occ_ticker(symbol, expiry, short_strike, False)
        long_put_ticker = self._occ_ticker(symbol, expiry, long_strike, False)
        
        if not short_put_ticker or not long_put_ticker:
            return None
        
        # Get entry prices
        short_put_price = self.client.agg_close(short_put_ticker, trade_date)
        long_put_price = self.client.agg_close(long_put_ticker, trade_date)
        
        if short_put_price is None or long_put_price is None:
            # Fallback to theoretical pricing
            tau = (expiry - trade_date).days / 365.0
            iv = kwargs.get('implied_vol', 0.25)
            
            from bot.pricing import bs_price
            short_put_price = bs_price(spot, short_strike, iv, tau, self.r, False)
            long_put_price = bs_price(spot, long_strike, iv, tau, self.r, False)
        
        # Create legs
        legs = [
            OptionLeg(
                ticker=short_put_ticker,
                strike=short_strike,
                expiry=expiry,
                is_call=False,
                position=-1,  # Short
                quantity=quantity,
                entry_price=short_put_price,
                implied_vol=kwargs.get('implied_vol', 0.25)
            ),
            OptionLeg(
                ticker=long_put_ticker,
                strike=long_strike,
                expiry=expiry,
                is_call=False,
                position=1,   # Long
                quantity=quantity,
                entry_price=long_put_price,
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
        """Calculate max profit and max loss for bull put spread."""
        short_leg = next(leg for leg in position.legs if leg.position < 0)
        long_leg = next(leg for leg in position.legs if leg.position > 0)
        
        # Net credit = short price - long price
        net_credit = (short_leg.entry_price - long_leg.entry_price) * short_leg.quantity
        
        # Strike difference
        strike_diff = short_leg.strike - long_leg.strike
        
        # Max profit = Net credit (when stock above short strike)
        max_profit = net_credit
        
        # Max loss = Strike difference - Net credit (when stock below long strike)
        max_loss = -(strike_diff * short_leg.quantity - net_credit)
        
        return max_profit, max_loss

    def get_breakeven_points(self, position: StrategyPosition) -> List[float]:
        """Calculate breakeven point for bull put spread."""
        short_leg = next(leg for leg in position.legs if leg.position < 0)
        long_leg = next(leg for leg in position.legs if leg.position > 0)
        
        net_credit = short_leg.entry_price - long_leg.entry_price
        
        # Breakeven = Higher strike - Net credit
        return [short_leg.strike - net_credit]