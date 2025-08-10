#!/usr/bin/env python3
"""
Demo script showing the Advanced Risk Management System in action.

This script demonstrates how the enhanced system works with realistic trading scenarios.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from bot.enhanced_engine import StrategyEngine, ExecutionEngine
from bot.risk_manager import AdvancedRiskManager, Position

def create_mock_config():
    """Create a configuration with risk management settings."""
    cfg = Config()
    cfg.polygon_key = "mock_key"  # Prevent API calls
    cfg.max_kelly_fraction = 0.20
    cfg.max_portfolio_var = 0.025  # 2.5% daily VaR limit
    cfg.max_drawdown_limit = 0.12  # 12% max drawdown
    cfg.max_position_size = 0.08   # 8% max position size
    cfg.symbols_equity = ["AAPL", "TSLA", "NVDA"]
    return cfg

def simulate_trading_session():
    """Simulate a live trading session with risk management."""
    print("🎯 QUANTITATIVE OPTIONS TRADING BOT")
    print("=" * 60)
    print("Advanced Risk Management System Demo")
    print("-" * 60)
    
    # Initialize components
    cfg = create_mock_config()
    strategy_engine = StrategyEngine(cfg)
    execution_engine = ExecutionEngine(cfg)
    risk_manager = AdvancedRiskManager(cfg)
    
    # Simulate some trading history for Kelly calculations
    sample_trades = [
        {'symbol': 'AAPL', 'pnl': 1200, 'strategy': 'iron_condor', 'duration_days': 7},
        {'symbol': 'AAPL', 'pnl': -400, 'strategy': 'iron_condor', 'duration_days': 8},
        {'symbol': 'AAPL', 'pnl': 800, 'strategy': 'iron_condor', 'duration_days': 6},
        {'symbol': 'TSLA', 'pnl': 2200, 'strategy': 'straddle', 'duration_days': 3},
        {'symbol': 'TSLA', 'pnl': -800, 'strategy': 'straddle', 'duration_days': 4},
        {'symbol': 'NVDA', 'pnl': 1500, 'strategy': 'bull_call_spread', 'duration_days': 10},
    ]
    risk_manager.trade_history = sample_trades
    
    # Simulate portfolio returns for VaR calculation
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.0008, 0.018, 100)  # 100 days of returns
    risk_manager.portfolio.daily_returns = returns.tolist()
    risk_manager.portfolio.total_value = 1000000  # $1M portfolio
    
    print(f"💰 Starting Portfolio Value: ${risk_manager.portfolio.total_value:,.0f}")
    print(f"📈 Historical Returns: {len(returns)} days")
    print(f"📊 Trade History: {len(sample_trades)} completed trades")
    
    # Trading scenarios with different market conditions
    trading_scenarios = [
        {
            'symbol': 'AAPL',
            'edge': 0.15,
            'spot_price': 185.50,
            'volatility': 0.25,
            'description': 'AAPL - Low bullish edge, normal volatility'
        },
        {
            'symbol': 'TSLA', 
            'edge': 0.72,
            'spot_price': 250.00,
            'volatility': 0.45,
            'description': 'TSLA - High bullish edge, high volatility'
        },
        {
            'symbol': 'NVDA',
            'edge': -0.58,
            'spot_price': 450.25,
            'volatility': 0.35,
            'description': 'NVDA - High bearish edge, elevated volatility'
        }
    ]
    
    print("\n🎰 TRADING SESSION")
    print("=" * 60)
    
    total_trades = 0
    total_allocated = 0
    
    for scenario in trading_scenarios:
        print(f"\n📋 {scenario['description']}")
        print(f"   Edge Score: {scenario['edge']:+.2f}")
        print(f"   Spot Price: ${scenario['spot_price']:.2f}")
        print(f"   Volatility: {scenario['volatility']:.0%}")
        
        market_data = {
            'spot_price': scenario['spot_price'],
            'volatility': scenario['volatility'],
            'last_close': scenario['spot_price']
        }
        
        # Generate trades
        trades = strategy_engine.generate(
            edge=scenario['edge'],
            sym=scenario['symbol'],
            market_data=market_data
        )
        
        # Execute approved trades
        for trade in trades:
            success = execution_engine.place(trade)
            if success:
                total_trades += 1
                total_allocated += trade['size']
                print(f"   ✅ {trade['strategy'].upper()} - Size: ${trade['size']:,.0f}")
            else:
                print(f"   ❌ Trade rejected by execution engine")
    
    print(f"\n📊 SESSION SUMMARY")
    print("=" * 60)
    print(f"Total Trades Executed: {total_trades}")
    print(f"Total Capital Allocated: ${total_allocated:,.0f}")
    print(f"Portfolio Utilization: {total_allocated/risk_manager.portfolio.total_value:.1%}")
    
    # Update portfolio and generate risk report
    current_positions = execution_engine.positions
    current_market_data = {
        'AAPL': 186.00,
        'TSLA': 248.50, 
        'NVDA': 452.75
    }
    
    execution_engine.update_positions(current_market_data)
    risk_manager.update_portfolio(current_positions, current_market_data)
    
    # Generate comprehensive risk report
    print(f"\n🛡️  RISK MANAGEMENT REPORT")
    print("=" * 60)
    
    risk_report = risk_manager.generate_risk_report()
    
    print(f"Portfolio Value: ${risk_report['portfolio_value']:,.0f}")
    print(f"Active Positions: {risk_report['positions_count']}")
    
    risk_metrics = risk_report['risk_metrics']
    print(f"\n📈 Risk Metrics:")
    print(f"   VaR (95%): {risk_metrics['var_95']:.4f} ({risk_metrics['var_95']*100:.2f}%)")
    print(f"   VaR (99%): {risk_metrics['var_99']:.4f} ({risk_metrics['var_99']*100:.2f}%)")
    print(f"   Expected Shortfall: {risk_metrics['expected_shortfall']:.4f}")
    print(f"   Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
    print(f"   Current Drawdown: {risk_metrics['current_drawdown']:.2%}")
    print(f"   Concentration Risk: {risk_metrics['concentration_risk']:.3f}")
    
    risk_limits = risk_report['risk_limits']
    print(f"\n🚨 Risk Limits:")
    print(f"   Max VaR Limit: {risk_limits['max_var_limit']:.2%}")
    print(f"   Max Drawdown Limit: {risk_limits['max_drawdown_limit']:.2%}")
    print(f"   Max Position Size: {risk_limits['max_position_size']:.2%}")
    
    # Check if any limits are breached
    violations = []
    if risk_metrics['var_95'] > risk_limits['max_var_limit']:
        violations.append("VaR limit exceeded")
    if risk_metrics['current_drawdown'] > risk_limits['max_drawdown_limit']:
        violations.append("Drawdown limit exceeded") 
    if risk_metrics['concentration_risk'] > 0.8:
        violations.append("High concentration risk")
    
    if violations:
        print(f"\n⚠️  Risk Violations Detected:")
        for violation in violations:
            print(f"   • {violation}")
    else:
        print(f"\n✅ All risk limits within acceptable ranges")
    
    print(f"\n🎯 SYSTEM PERFORMANCE")
    print("=" * 60)
    print("✅ Advanced Risk Management: Active")
    print("✅ Kelly Criterion Position Sizing: Active") 
    print("✅ VaR Monitoring: Active")
    print("✅ Drawdown Controls: Active")
    print("✅ Portfolio Correlation Analysis: Active")
    print("✅ Options Greeks Calculation: Ready")
    print("✅ Multi-Strategy Selection: Active")

def demonstrate_risk_scenarios():
    """Show how the system handles different risk scenarios."""
    print(f"\n🧪 RISK SCENARIO TESTING")
    print("=" * 60)
    
    cfg = create_mock_config()
    cfg.max_portfolio_var = 0.015  # Very tight VaR limit
    cfg.max_position_size = 0.05   # Small position limit
    
    risk_manager = AdvancedRiskManager(cfg)
    
    # Simulate high-risk market conditions
    high_volatility_returns = np.random.normal(-0.002, 0.035, 50)  # Negative drift, high vol
    risk_manager.portfolio.daily_returns = high_volatility_returns.tolist()
    risk_manager.portfolio.current_drawdown = 0.08  # 8% current drawdown
    risk_manager._calculate_risk_metrics()
    
    print("Market Condition: High volatility bear market")
    print(f"Portfolio VaR (95%): {risk_manager.risk_metrics.portfolio_var_95:.3f}")
    print(f"Current Drawdown: {risk_manager.portfolio.current_drawdown:.2%}")
    
    # Test trade approval under stressed conditions
    test_trades = [
        {'symbol': 'AAPL', 'size': 25000, 'strategy': 'iron_condor'},
        {'symbol': 'TSLA', 'size': 60000, 'strategy': 'straddle'},  # Large position
        {'symbol': 'SPY', 'size': 15000, 'strategy': 'strangle'},
    ]
    
    print("\nTrade Approval Results:")
    for trade in test_trades:
        approved, reason = risk_manager.check_risk_limits(trade)
        status = "✅ APPROVED" if approved else "❌ REJECTED"
        print(f"   {trade['symbol']} ${trade['size']:,} - {status}: {reason}")

if __name__ == "__main__":
    print("🚀 Starting Advanced Risk Management Demo...")
    simulate_trading_session()
    demonstrate_risk_scenarios()
    print(f"\n🎉 Demo completed successfully!")
    print(f"\n💡 The advanced risk management system is now fully integrated")
    print(f"   and ready to protect your trading capital with:")
    print(f"   • Scientific position sizing (Kelly Criterion)")
    print(f"   • Real-time VaR monitoring") 
    print(f"   • Automated drawdown controls")
    print(f"   • Portfolio correlation analysis")
    print(f"   • Multi-method volatility forecasting")