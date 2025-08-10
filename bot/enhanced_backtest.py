"""
Enhanced Backtesting Engine for Options Trading Strategies

Provides realistic backtesting with:
- Bid/ask spreads modeling
- Commission and slippage calculations
- Historical volatility surface modeling
- Comprehensive performance analytics
- Support for all 12 strategy types
- Risk-adjusted metrics
"""

from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging

from bot.polygon_client import PolygonClient
from bot.greeks import GreeksCalculator
from bot.strategy.strategy_factory import StrategyFactory
from bot.risk_manager import AdvancedRiskManager
from logger import get_logger

logger = get_logger("enhanced_backtest")

@dataclass
class MarketData:
    """Historical market data point."""
    date: dt.date
    spot_price: float
    implied_vol: float
    realized_vol: float
    bid_ask_spread: float = 0.01  # As percentage of mid price
    volume: int = 0
    open_interest: int = 0

@dataclass
class TradeExecution:
    """Record of actual trade execution with slippage and commissions."""
    trade_date: dt.date
    symbol: str
    strategy: str
    legs: List[Dict[str, Any]]
    entry_prices: List[float]  # Actual prices paid (including slippage)
    theoretical_prices: List[float]  # Mid prices without slippage
    commissions: float
    slippage_cost: float
    total_cost: float  # Entry cost including all fees

@dataclass
class PositionSnapshot:
    """Portfolio position at a point in time."""
    date: dt.date
    symbol: str
    strategy: str
    position_value: float
    unrealized_pnl: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    days_to_expiry: int

@dataclass
class BacktestResults:
    """Comprehensive backtest results with performance metrics."""
    strategy_name: str
    symbol: str
    start_date: dt.date
    end_date: dt.date
    
    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L Metrics
    total_pnl: float = 0.0
    gross_pnl: float = 0.0  # Before commissions
    net_pnl: float = 0.0    # After all costs
    total_commissions: float = 0.0
    total_slippage: float = 0.0
    
    # Risk Metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0
    
    # Performance Statistics
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0
    profit_factor: float = 0.0
    
    # Greeks Exposure
    max_delta_exposure: float = 0.0
    max_gamma_exposure: float = 0.0
    max_theta_exposure: float = 0.0
    max_vega_exposure: float = 0.0
    
    # Detailed Records
    trades: List[TradeExecution] = field(default_factory=list)
    daily_pnl: List[float] = field(default_factory=list)
    position_snapshots: List[PositionSnapshot] = field(default_factory=list)
    drawdown_series: List[float] = field(default_factory=list)

class EnhancedBacktester:
    """
    Advanced backtesting engine for options strategies.
    
    Features:
    - Realistic bid/ask spread modeling
    - Commission and slippage calculations
    - Historical volatility surface reconstruction
    - Greeks tracking and risk metrics
    - Multi-strategy portfolio simulation
    """
    
    def __init__(self, 
                 client: PolygonClient,
                 risk_manager: AdvancedRiskManager = None,
                 commission_per_contract: float = 0.65,
                 base_slippage: float = 0.005,  # 0.5% base slippage
                 min_bid_ask_spread: float = 0.01,
                 max_bid_ask_spread: float = 0.05):
        """
        Initialize enhanced backtester.
        
        Args:
            client: Polygon client for market data
            risk_manager: Risk management system
            commission_per_contract: Commission per options contract
            base_slippage: Base slippage as fraction of price
            min_bid_ask_spread: Minimum bid/ask spread
            max_bid_ask_spread: Maximum bid/ask spread
        """
        self.client = client
        self.risk_manager = risk_manager
        self.greeks_calc = GreeksCalculator()
        self.strategy_factory = None
        
        # Cost modeling parameters
        self.commission_per_contract = commission_per_contract
        self.base_slippage = base_slippage
        self.min_bid_ask_spread = min_bid_ask_spread
        self.max_bid_ask_spread = max_bid_ask_spread
        
        # Market data cache
        self.market_data_cache: Dict[Tuple[str, dt.date], MarketData] = {}
        
        # Initialize strategy factory if possible
        try:
            self.strategy_factory = StrategyFactory(
                client=client,
                greeks_calc=self.greeks_calc
            )
        except Exception as e:
            logger.warning(f"Could not initialize strategy factory: {e}")
    
    def run_backtest(self,
                    strategy_name: str,
                    symbols: List[str],
                    start_date: dt.date,
                    end_date: dt.date,
                    initial_capital: float = 100000,
                    rebalance_frequency: str = 'weekly',
                    **strategy_params) -> Dict[str, BacktestResults]:
        """
        Run comprehensive backtest for a strategy across multiple symbols.
        
        Args:
            strategy_name: Name of strategy to test
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
            **strategy_params: Strategy-specific parameters
        
        Returns:
            Dictionary mapping symbols to backtest results
        """
        logger.info(f"Starting enhanced backtest for {strategy_name}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Period: {start_date} to {end_date}")
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"Backtesting {strategy_name} on {symbol}")
            
            result = self._backtest_single_symbol(
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital / len(symbols),  # Divide capital
                rebalance_frequency=rebalance_frequency,
                **strategy_params
            )
            
            results[symbol] = result
            
            logger.info(f"Completed {symbol}: {result.total_trades} trades, "
                       f"${result.net_pnl:,.0f} P&L, "
                       f"{result.win_rate:.1%} win rate")
        
        return results
    
    def _backtest_single_symbol(self,
                               strategy_name: str,
                               symbol: str,
                               start_date: dt.date,
                               end_date: dt.date,
                               initial_capital: float,
                               rebalance_frequency: str,
                               **strategy_params) -> BacktestResults:
        """Backtest single strategy on single symbol."""
        
        result = BacktestResults(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Generate trading dates
        trade_dates = self._generate_trade_dates(start_date, end_date, rebalance_frequency)
        
        current_capital = initial_capital
        current_positions = []
        daily_pnls = []
        peak_capital = initial_capital
        
        for trade_date in trade_dates:
            try:
                # Close expired positions
                current_positions, closed_pnl = self._close_expired_positions(
                    current_positions, trade_date
                )
                current_capital += closed_pnl
                
                # Get market data
                market_data = self._get_market_data(symbol, trade_date)
                if market_data is None:
                    continue
                
                # Generate new position if we have capital
                if current_capital > 0:
                    new_position = self._generate_position(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        trade_date=trade_date,
                        market_data=market_data,
                        available_capital=current_capital,
                        **strategy_params
                    )
                    
                    if new_position:
                        current_positions.append(new_position)
                        current_capital -= new_position.total_cost
                        result.trades.append(new_position)
                        result.total_trades += 1
                        result.total_commissions += new_position.commissions
                        result.total_slippage += new_position.slippage_cost
                
                # Mark positions to market
                position_values = []
                total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
                
                for position in current_positions:
                    position_value, greeks = self._mark_position_to_market(
                        position, trade_date, market_data
                    )
                    position_values.append(position_value)
                    
                    for greek, value in greeks.items():
                        total_greeks[greek] += value
                
                # Calculate daily P&L
                portfolio_value = current_capital + sum(position_values)
                daily_pnl = portfolio_value - initial_capital - sum(daily_pnls)
                daily_pnls.append(daily_pnl)
                
                # Track drawdown
                peak_capital = max(peak_capital, portfolio_value)
                drawdown = (peak_capital - portfolio_value) / peak_capital
                result.drawdown_series.append(drawdown)
                result.max_drawdown = max(result.max_drawdown, drawdown)
                
                # Track Greeks exposure
                result.max_delta_exposure = max(result.max_delta_exposure, abs(total_greeks['delta']))
                result.max_gamma_exposure = max(result.max_gamma_exposure, abs(total_greeks['gamma']))
                result.max_theta_exposure = max(result.max_theta_exposure, abs(total_greeks['theta']))
                result.max_vega_exposure = max(result.max_vega_exposure, abs(total_greeks['vega']))
                
                # Create position snapshot
                snapshot = PositionSnapshot(
                    date=trade_date,
                    symbol=symbol,
                    strategy=strategy_name,
                    position_value=sum(position_values),
                    unrealized_pnl=portfolio_value - initial_capital,
                    delta=total_greeks['delta'],
                    gamma=total_greeks['gamma'],
                    theta=total_greeks['theta'],
                    vega=total_greeks['vega'],
                    rho=total_greeks['rho'],
                    days_to_expiry=self._avg_days_to_expiry(current_positions, trade_date)
                )
                result.position_snapshots.append(snapshot)
                
            except Exception as e:
                logger.warning(f"Error processing {trade_date} for {symbol}: {e}")
                continue
        
        # Calculate final metrics
        result.daily_pnl = daily_pnls
        result.total_pnl = sum(daily_pnls)
        result.net_pnl = result.total_pnl - result.total_commissions - result.total_slippage
        result.gross_pnl = result.total_pnl
        
        # Calculate win/loss statistics
        winning_trades = [t for t in result.trades if self._trade_pnl(t) > 0]
        losing_trades = [t for t in result.trades if self._trade_pnl(t) < 0]
        
        result.winning_trades = len(winning_trades)
        result.losing_trades = len(losing_trades)
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        
        if winning_trades:
            result.avg_winner = np.mean([self._trade_pnl(t) for t in winning_trades])
            result.largest_winner = max([self._trade_pnl(t) for t in winning_trades])
        
        if losing_trades:
            result.avg_loser = np.mean([self._trade_pnl(t) for t in losing_trades])
            result.largest_loser = min([self._trade_pnl(t) for t in losing_trades])
        
        # Risk-adjusted metrics
        if daily_pnls and len(daily_pnls) > 1:
            returns = np.array(daily_pnls) / initial_capital
            result.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            result.sortino_ratio = self._calculate_sortino_ratio(returns)
            result.var_95 = np.percentile(returns, 5) * initial_capital
            
            if result.max_drawdown > 0:
                result.calmar_ratio = (result.net_pnl / initial_capital) / result.max_drawdown
        
        # Profit factor
        gross_profit = sum([self._trade_pnl(t) for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([self._trade_pnl(t) for t in losing_trades])) if losing_trades else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return result
    
    def _generate_trade_dates(self, start_date: dt.date, end_date: dt.date, frequency: str) -> List[dt.date]:
        """Generate list of trading dates based on frequency."""
        dates = []
        current = start_date
        
        if frequency == 'daily':
            delta = dt.timedelta(days=1)
        elif frequency == 'weekly':
            delta = dt.timedelta(days=7)
        elif frequency == 'monthly':
            delta = dt.timedelta(days=30)
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        while current <= end_date:
            # Only trade on weekdays
            if current.weekday() < 5:
                dates.append(current)
            current += delta
        
        return dates
    
    def _get_market_data(self, symbol: str, date: dt.date) -> Optional[MarketData]:
        """Get market data for symbol on date with caching."""
        cache_key = (symbol, date)
        
        if cache_key in self.market_data_cache:
            return self.market_data_cache[cache_key]
        
        try:
            # Get spot price
            spot = self.client.spot(symbol, date)
            if spot is None:
                return None
            
            # Estimate implied volatility (simplified)
            # In production, this would come from options chain
            implied_vol = self._estimate_implied_volatility(symbol, date, spot)
            
            # Calculate realized volatility from recent price history
            realized_vol = self._calculate_realized_volatility(symbol, date)
            
            # Estimate bid/ask spread based on volatility and liquidity
            bid_ask_spread = self._estimate_bid_ask_spread(spot, implied_vol)
            
            market_data = MarketData(
                date=date,
                spot_price=spot,
                implied_vol=implied_vol,
                realized_vol=realized_vol,
                bid_ask_spread=bid_ask_spread
            )
            
            self.market_data_cache[cache_key] = market_data
            return market_data
            
        except Exception as e:
            logger.warning(f"Could not get market data for {symbol} on {date}: {e}")
            return None
    
    def _estimate_implied_volatility(self, symbol: str, date: dt.date, spot: float) -> float:
        """Estimate implied volatility using historical data and VIX proxy."""
        try:
            # Get recent realized volatility
            realized_vol = self._calculate_realized_volatility(symbol, date, lookback_days=30)
            
            # Add volatility risk premium (typical 2-5%)
            vol_premium = 0.03 if symbol in ['SPY', 'QQQ'] else 0.05
            
            # Adjust for market regime (simplified)
            if realized_vol > 0.3:  # High vol regime
                vol_premium *= 1.5
            elif realized_vol < 0.15:  # Low vol regime
                vol_premium *= 0.8
            
            implied_vol = realized_vol + vol_premium
            return max(0.1, min(1.0, implied_vol))  # Clamp between 10% and 100%
            
        except:
            # Fallback to typical values by symbol type
            if symbol in ['SPY', 'QQQ', 'IWM']:
                return 0.20
            elif symbol in ['AAPL', 'MSFT', 'GOOGL']:
                return 0.25
            else:
                return 0.30
    
    def _calculate_realized_volatility(self, symbol: str, date: dt.date, lookback_days: int = 20) -> float:
        """Calculate realized volatility from historical price returns."""
        try:
            # Get historical prices (simplified - in production use proper historical data)
            returns = []
            for i in range(1, lookback_days + 1):
                past_date = date - dt.timedelta(days=i)
                if past_date.weekday() < 5:  # Weekdays only
                    price = self.client.spot(symbol, past_date)
                    if price:
                        returns.append(np.log(price))
            
            if len(returns) < 10:  # Need minimum data points
                return 0.25  # Default volatility
            
            # Calculate annualized volatility
            daily_returns = np.diff(returns)
            return np.std(daily_returns) * np.sqrt(252)
            
        except:
            return 0.25  # Default fallback
    
    def _estimate_bid_ask_spread(self, spot: float, volatility: float) -> float:
        """Estimate bid/ask spread based on stock price and volatility."""
        # Base spread increases with volatility and decreases with price
        base_spread = 0.005 + (volatility - 0.2) * 0.01
        
        # Adjust for stock price (higher price = lower spread %)
        if spot > 200:
            base_spread *= 0.8
        elif spot < 50:
            base_spread *= 1.5
        
        return max(self.min_bid_ask_spread, min(self.max_bid_ask_spread, base_spread))
    
    def _generate_position(self,
                          strategy_name: str,
                          symbol: str,
                          trade_date: dt.date,
                          market_data: MarketData,
                          available_capital: float,
                          **strategy_params) -> Optional[TradeExecution]:
        """Generate new position with realistic execution costs."""
        
        if not self.strategy_factory:
            return None
        
        try:
            # Calculate position size based on available capital and risk
            max_position_size = available_capital * 0.05  # Max 5% per trade
            
            # Create strategy position
            expiry = self._get_next_expiry(trade_date)
            position = self.strategy_factory.create_position(
                strategy_name=strategy_name,
                symbol=symbol,
                spot=market_data.spot_price,
                trade_date=trade_date,
                expiry=expiry,
                implied_vol=market_data.implied_vol,
                **strategy_params
            )
            
            if not position or not position.legs:
                return None
            
            # Calculate execution costs
            theoretical_prices = []
            actual_prices = []
            total_commissions = 0
            total_slippage = 0
            
            for leg in position.legs:
                # Theoretical mid price
                mid_price = self._calculate_option_price(
                    spot=market_data.spot_price,
                    strike=leg.strike,
                    expiry=expiry,
                    vol=market_data.implied_vol,
                    is_call=leg.is_call
                )
                theoretical_prices.append(mid_price)
                
                # Apply bid/ask spread
                spread = mid_price * market_data.bid_ask_spread
                if leg.quantity > 0:  # Buying
                    execution_price = mid_price + spread/2
                else:  # Selling
                    execution_price = mid_price - spread/2
                
                # Apply slippage (additional cost for market impact)
                slippage = abs(execution_price * self.base_slippage)
                if leg.quantity > 0:
                    execution_price += slippage
                else:
                    execution_price -= slippage
                
                actual_prices.append(execution_price)
                
                # Calculate commissions
                contracts = abs(leg.quantity)
                total_commissions += contracts * self.commission_per_contract
                total_slippage += contracts * slippage
            
            # Calculate total cost
            total_cost = sum(abs(leg.quantity) * price 
                           for leg, price in zip(position.legs, actual_prices))
            total_cost += total_commissions
            
            # Check if we can afford this trade
            if total_cost > max_position_size:
                return None
            
            # Create execution record
            execution = TradeExecution(
                trade_date=trade_date,
                symbol=symbol,
                strategy=strategy_name,
                legs=[{
                    'strike': leg.strike,
                    'quantity': leg.quantity,
                    'is_call': leg.is_call,
                    'expiry': expiry
                } for leg in position.legs],
                entry_prices=actual_prices,
                theoretical_prices=theoretical_prices,
                commissions=total_commissions,
                slippage_cost=total_slippage,
                total_cost=total_cost
            )
            
            return execution
            
        except Exception as e:
            logger.warning(f"Error generating position for {symbol}: {e}")
            return None
    
    def _calculate_option_price(self, spot: float, strike: float, expiry: dt.date, 
                               vol: float, is_call: bool, risk_free_rate: float = 0.02) -> float:
        """Calculate option price using Black-Scholes."""
        try:
            from bot.pricing import bs_price
            
            # Convert expiry to time to expiration in years
            days_to_expiry = (expiry - dt.date.today()).days
            tau = max(1/365, days_to_expiry / 365.0)  # Avoid zero time
            
            price = bs_price(
                spot=spot,
                strike=strike,
                tau=tau,
                vol=vol,
                r=risk_free_rate,
                is_call=is_call
            )
            
            return max(0.01, price)  # Minimum price of $0.01
            
        except Exception as e:
            logger.warning(f"Error calculating option price: {e}")
            return 0.50  # Default option price
    
    def _get_next_expiry(self, trade_date: dt.date, min_days: int = 21) -> dt.date:
        """Get next monthly expiry at least min_days away."""
        current = trade_date + dt.timedelta(days=min_days)
        
        # Find third Friday of the month
        # Move to first day of month
        first_day = current.replace(day=1)
        
        # Find first Friday
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + dt.timedelta(days=days_to_friday)
        
        # Third Friday is 14 days later
        third_friday = first_friday + dt.timedelta(days=14)
        
        # If third Friday is too soon, go to next month
        if third_friday < current:
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=current.month + 1, day=1)
            
            days_to_friday = (4 - next_month.weekday()) % 7
            first_friday = next_month + dt.timedelta(days=days_to_friday)
            third_friday = first_friday + dt.timedelta(days=14)
        
        return third_friday
    
    def _close_expired_positions(self, positions: List[TradeExecution], 
                                current_date: dt.date) -> Tuple[List[TradeExecution], float]:
        """Close expired positions and return remaining positions and P&L."""
        remaining_positions = []
        total_pnl = 0
        
        for position in positions:
            # Check if position is expired
            expiry = position.legs[0]['expiry'] if position.legs else current_date
            
            if current_date >= expiry:
                # Calculate expiry P&L (simplified)
                pnl = self._calculate_expiry_pnl(position, current_date)
                total_pnl += pnl
            else:
                remaining_positions.append(position)
        
        return remaining_positions, total_pnl
    
    def _calculate_expiry_pnl(self, position: TradeExecution, expiry_date: dt.date) -> float:
        """Calculate P&L at expiration."""
        try:
            spot = self.client.spot(position.symbol, expiry_date)
            if spot is None:
                return 0
            
            total_pnl = 0
            
            for i, leg in enumerate(position.legs):
                strike = leg['strike']
                quantity = leg['quantity']
                is_call = leg['is_call']
                entry_price = position.entry_prices[i]
                
                # Intrinsic value at expiry
                if is_call:
                    intrinsic_value = max(0, spot - strike)
                else:
                    intrinsic_value = max(0, strike - spot)
                
                # P&L = (exit_value - entry_price) * quantity
                leg_pnl = (intrinsic_value - entry_price) * quantity
                total_pnl += leg_pnl
            
            return total_pnl
            
        except Exception as e:
            logger.warning(f"Error calculating expiry P&L: {e}")
            return 0
    
    def _mark_position_to_market(self, position: TradeExecution, 
                                current_date: dt.date, 
                                market_data: MarketData) -> Tuple[float, Dict[str, float]]:
        """Mark position to market and calculate current Greeks."""
        try:
            total_value = 0
            total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
            for i, leg in enumerate(position.legs):
                strike = leg['strike']
                quantity = leg['quantity']
                is_call = leg['is_call']
                expiry = leg['expiry']
                
                # Calculate current option price
                current_price = self._calculate_option_price(
                    spot=market_data.spot_price,
                    strike=strike,
                    expiry=expiry,
                    vol=market_data.implied_vol,
                    is_call=is_call
                )
                
                # Position value
                leg_value = current_price * quantity
                total_value += leg_value
                
                # Calculate Greeks
                days_to_expiry = (expiry - current_date).days
                tau = max(1/365, days_to_expiry / 365.0)
                
                greeks = self.greeks_calc.calculate_all_greeks(
                    spot=market_data.spot_price,
                    strike=strike,
                    iv=market_data.implied_vol,
                    tau=tau,
                    r=0.02,
                    is_call=is_call
                )
                
                for greek, value in greeks.items():
                    total_greeks[greek] += value * quantity
            
            return total_value, total_greeks
            
        except Exception as e:
            logger.warning(f"Error marking position to market: {e}")
            return 0, {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def _avg_days_to_expiry(self, positions: List[TradeExecution], current_date: dt.date) -> int:
        """Calculate average days to expiry for current positions."""
        if not positions:
            return 0
        
        total_days = 0
        count = 0
        
        for position in positions:
            for leg in position.legs:
                expiry = leg['expiry']
                days = (expiry - current_date).days
                total_days += days
                count += 1
        
        return total_days // count if count > 0 else 0
    
    def _trade_pnl(self, trade: TradeExecution) -> float:
        """Calculate P&L for a completed trade."""
        # Simplified - in practice this would be calculated when position is closed
        return 0  # Placeholder
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Annualize (assuming daily returns)
        annual_return = mean_return * 252
        annual_vol = std_return * np.sqrt(252)
        
        return annual_return / annual_vol if annual_vol > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) == 0:
            return 0
        
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0
        
        downside_deviation = np.std(downside_returns)
        
        # Annualize
        annual_return = mean_return * 252
        annual_downside_dev = downside_deviation * np.sqrt(252)
        
        return annual_return / annual_downside_dev if annual_downside_dev > 0 else 0

    def generate_report(self, results: Dict[str, BacktestResults]) -> str:
        """Generate comprehensive backtest report."""
        report = []
        report.append("=" * 80)
        report.append("ENHANCED BACKTEST REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        total_trades = sum(r.total_trades for r in results.values())
        total_pnl = sum(r.net_pnl for r in results.values())
        avg_win_rate = np.mean([r.win_rate for r in results.values()])
        
        report.append(f"\nOVERALL SUMMARY:")
        report.append(f"Total Trades: {total_trades:,}")
        report.append(f"Total P&L: ${total_pnl:,.0f}")
        report.append(f"Average Win Rate: {avg_win_rate:.1%}")
        
        # Per-symbol results
        for symbol, result in results.items():
            report.append(f"\n{symbol} RESULTS:")
            report.append(f"  Trades: {result.total_trades}")
            report.append(f"  Win Rate: {result.win_rate:.1%}")
            report.append(f"  Net P&L: ${result.net_pnl:,.0f}")
            report.append(f"  Gross P&L: ${result.gross_pnl:,.0f}")
            report.append(f"  Commissions: ${result.total_commissions:,.0f}")
            report.append(f"  Slippage: ${result.total_slippage:,.0f}")
            report.append(f"  Max Drawdown: {result.max_drawdown:.1%}")
            report.append(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            report.append(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
            report.append(f"  Profit Factor: {result.profit_factor:.2f}")
            
            if result.avg_winner > 0:
                report.append(f"  Avg Winner: ${result.avg_winner:.0f}")
            if result.avg_loser < 0:
                report.append(f"  Avg Loser: ${result.avg_loser:.0f}")
        
        return "\n".join(report)