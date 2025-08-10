#!/usr/bin/env python3
"""
Test script for the Advanced Risk Management System

Demonstrates VaR calculations, Kelly Criterion, drawdown controls,
and portfolio correlation analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from bot.risk_manager import AdvancedRiskManager, Position, Portfolio
from bot.greeks import GreeksCalculator
from bot.enhanced_engine import StrategyEngine, ExecutionEngine

def generate_sample_returns(n_days=252, mean=0.0005, std=0.02):
    """Generate sample daily returns for testing."""
    return np.random.normal(mean, std, n_days)

def test_var_calculations():
    """Test VaR calculation methods."""
    print("\n=== Testing VaR Calculations ===")
    
    cfg = Config()
    risk_manager = AdvancedRiskManager(cfg)
    
    # Generate sample return data
    returns = generate_sample_returns(500, mean=0.001, std=0.015)
    
    # Calculate VaR using different methods
    hist_var_95 = risk_manager.var_calculator.historical_var(returns, 0.95)
    hist_var_99 = risk_manager.var_calculator.historical_var(returns, 0.99)
    param_var_95 = risk_manager.var_calculator.parametric_var(returns, 0.95)
    mc_var_95 = risk_manager.var_calculator.monte_carlo_var(
        portfolio_value=1000000,
        expected_return=np.mean(returns),
        volatility=np.std(returns),
        confidence_level=0.95
    )
    expected_shortfall = risk_manager.var_calculator.expected_shortfall(returns, 0.95)
    
    print(f"Historical VaR (95%): {hist_var_95:.4f}")
    print(f"Historical VaR (99%): {hist_var_99:.4f}")
    print(f"Parametric VaR (95%): {param_var_95:.4f}")
    print(f"Monte Carlo VaR (95%): ${mc_var_95:,.0f}")
    print(f"Expected Shortfall (95%): {expected_shortfall:.4f}")

def test_kelly_criterion():
    """Test Kelly Criterion position sizing."""
    print("\n=== Testing Kelly Criterion ===")
    
    cfg = Config()
    risk_manager = AdvancedRiskManager(cfg)
    
    # Simulate historical trade results
    sample_trades = [
        {'symbol': 'AAPL', 'pnl': 1500, 'strategy': 'iron_condor'},
        {'symbol': 'AAPL', 'pnl': -800, 'strategy': 'iron_condor'},
        {'symbol': 'AAPL', 'pnl': 2200, 'strategy': 'iron_condor'},
        {'symbol': 'AAPL', 'pnl': -600, 'strategy': 'iron_condor'},
        {'symbol': 'AAPL', 'pnl': 1800, 'strategy': 'iron_condor'},
        {'symbol': 'AAPL', 'pnl': -1200, 'strategy': 'iron_condor'},
        {'symbol': 'AAPL', 'pnl': 900, 'strategy': 'iron_condor'},
        {'symbol': 'AAPL', 'pnl': 1100, 'strategy': 'iron_condor'},
        {'symbol': 'AAPL', 'pnl': -400, 'strategy': 'iron_condor'},
        {'symbol': 'AAPL', 'pnl': 1600, 'strategy': 'iron_condor'},
    ]
    
    # Add trades to history
    risk_manager.trade_history = sample_trades
    
    # Test Kelly position sizing
    position_size = risk_manager.kelly_position_size(
        symbol='AAPL',
        expected_return=0.08,  # 8% expected return
        volatility=0.15,       # 15% volatility
        confidence=0.7         # 70% confidence
    )
    
    print(f"Recommended position size for AAPL: ${position_size:,.0f}")
    print(f"As percentage of portfolio: {position_size/risk_manager.portfolio.total_value:.2%}")

def test_portfolio_risk_metrics():
    """Test portfolio risk metrics calculation."""
    print("\n=== Testing Portfolio Risk Metrics ===")
    
    cfg = Config()
    risk_manager = AdvancedRiskManager(cfg)
    
    # Create sample portfolio
    positions = {
        'AAPL': Position(
            symbol='AAPL',
            strategy='iron_condor',
            size=50000,
            entry_price=150.0,
            current_value=52000,
            entry_date=datetime.now() - timedelta(days=10),
            unrealized_pnl=2000
        ),
        'TSLA': Position(
            symbol='TSLA',
            strategy='straddle',
            size=30000,
            entry_price=200.0,
            current_value=28500,
            entry_date=datetime.now() - timedelta(days=5),
            unrealized_pnl=-1500
        ),
        'SPY': Position(
            symbol='SPY',
            strategy='iron_condor',
            size=40000,
            entry_price=400.0,
            current_value=41200,
            entry_date=datetime.now() - timedelta(days=3),
            unrealized_pnl=1200
        )
    }
    
    # Generate sample daily returns
    returns = generate_sample_returns(100, mean=0.0008, std=0.018)
    risk_manager.portfolio.daily_returns = returns.tolist()
    
    # Update portfolio
    market_data = {'AAPL': 152.0, 'TSLA': 195.0, 'SPY': 403.0}
    risk_manager.update_portfolio(positions, market_data)
    
    # Generate risk report
    risk_report = risk_manager.generate_risk_report()
    
    print(f"Portfolio Value: ${risk_report['portfolio_value']:,.0f}")
    print(f"Positions: {risk_report['positions_count']}")
    print(f"VaR (95%): {risk_report['risk_metrics']['var_95']:.4f}")
    print(f"VaR (99%): {risk_report['risk_metrics']['var_99']:.4f}")
    print(f"Sharpe Ratio: {risk_report['risk_metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {risk_report['risk_metrics']['max_drawdown']:.2%}")
    print(f"Current Drawdown: {risk_report['risk_metrics']['current_drawdown']:.2%}")
    print(f"Concentration Risk: {risk_report['risk_metrics']['concentration_risk']:.3f}")

def test_greeks_calculation():
    """Test options Greeks calculations."""
    print("\n=== Testing Greeks Calculations ===")
    
    greeks_calc = GreeksCalculator()
    
    # Sample option parameters
    spot = 100.0
    strike = 105.0
    iv = 0.25  # 25% volatility
    tau = 30/365  # 30 days to expiration
    r = 0.02  # 2% risk-free rate
    
    # Calculate all Greeks for a call option
    call_greeks = greeks_calc.calculate_all_greeks(spot, strike, iv, tau, r, True)
    put_greeks = greeks_calc.calculate_all_greeks(spot, strike, iv, tau, r, False)
    
    print("Call Option Greeks:")
    for greek, value in call_greeks.items():
        print(f"  {greek.capitalize()}: {value:.4f}")
    
    print("\nPut Option Greeks:")
    for greek, value in put_greeks.items():
        print(f"  {greek.capitalize()}: {value:.4f}")
    
    # Test implied volatility calculation
    from bot.pricing import bs_price
    market_price = bs_price(spot, strike, 0.22, tau, r, True)  # Use 22% vol
    implied_vol = greeks_calc.implied_volatility(market_price, spot, strike, tau, r, True)
    
    print(f"\nImplied Volatility Test:")
    print(f"  Market Price: ${market_price:.2f}")
    print(f"  Calculated IV: {implied_vol:.4f} ({implied_vol*100:.2f}%)")

def test_risk_limits():
    """Test risk limit checks."""
    print("\n=== Testing Risk Limit Checks ===")
    
    cfg = Config()
    cfg.max_portfolio_var = 0.015  # 1.5% daily VaR limit
    cfg.max_drawdown_limit = 0.10  # 10% max drawdown
    cfg.max_position_size = 0.08   # 8% max position size
    
    risk_manager = AdvancedRiskManager(cfg)
    
    # Set up a portfolio with high risk
    high_risk_returns = generate_sample_returns(50, mean=-0.005, std=0.035)
    risk_manager.portfolio.daily_returns = high_risk_returns.tolist()
    risk_manager.portfolio.current_drawdown = 0.12  # 12% drawdown
    risk_manager._calculate_risk_metrics()
    
    # Test various trade sizes
    test_trades = [
        {'symbol': 'AAPL', 'size': 50000, 'strategy': 'iron_condor'},
        {'symbol': 'TSLA', 'size': 100000, 'strategy': 'straddle'},  # Large position
        {'symbol': 'SPY', 'size': 30000, 'strategy': 'strangle'},
    ]
    
    for trade in test_trades:
        approved, reason = risk_manager.check_risk_limits(trade)
        status = "✅ APPROVED" if approved else "❌ REJECTED"
        print(f"{trade['symbol']} ${trade['size']:,} - {status}: {reason}")

def test_enhanced_strategy_engine():
    """Test the enhanced strategy engine."""
    print("\n=== Testing Enhanced Strategy Engine ===")
    
    cfg = Config()
    strategy_engine = StrategyEngine(cfg)
    
    # Test different edge scenarios
    test_scenarios = [
        {'edge': 0.1, 'symbol': 'AAPL', 'volatility': 0.20},   # Low edge, normal vol
        {'edge': 0.7, 'symbol': 'TSLA', 'volatility': 0.35},   # High bullish edge, high vol
        {'edge': -0.6, 'symbol': 'SPY', 'volatility': 0.15},   # High bearish edge, low vol
        {'edge': 0.0, 'symbol': 'QQQ', 'volatility': 0.40},    # Neutral edge, high vol
    ]
    
    for scenario in test_scenarios:
        market_data = {
            'spot_price': 100.0,
            'volatility': scenario['volatility']
        }
        
        trades = strategy_engine.generate(
            edge=scenario['edge'],
            sym=scenario['symbol'], 
            market_data=market_data
        )
        
        print(f"\n{scenario['symbol']} (edge={scenario['edge']:.1f}, vol={scenario['volatility']:.0%}):")
        for trade in trades:
            print(f"  Strategy: {trade['strategy']}")
            print(f"  Size: ${trade['size']:,.0f}")

def main():
    """Run all risk management tests."""
    print("🚀 Advanced Risk Management System Tests")
    print("=" * 50)
    
    try:
        test_var_calculations()
        test_kelly_criterion()
        test_portfolio_risk_metrics()
        test_greeks_calculation()
        test_risk_limits()
        test_enhanced_strategy_engine()
        
        print("\n✅ All tests completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("   • VaR calculations (Historical, Parametric, Monte Carlo)")
        print("   • Kelly Criterion position sizing")
        print("   • Portfolio risk metrics (Sharpe, Drawdown, Concentration)")
        print("   • Options Greeks calculations")
        print("   • Risk limit enforcement")
        print("   • Enhanced strategy selection")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()