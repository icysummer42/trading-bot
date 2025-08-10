#!/usr/bin/env python3
"""
Advanced Options Strategies Demo

Showcases all implemented options strategies with realistic market scenarios,
demonstrating strategy selection, risk management, and portfolio optimization.
"""

import numpy as np
import datetime as dt
from typing import Dict, List
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from bot.enhanced_engine import StrategyEngine, ExecutionEngine
from bot.risk_manager import AdvancedRiskManager
from bot.strategy.strategy_factory import StrategyFactory

def create_demo_config():
    """Create demo configuration with realistic settings."""
    cfg = Config()
    cfg.polygon_key = None  # Demo mode - no API calls
    cfg.max_kelly_fraction = 0.20
    cfg.max_portfolio_var = 0.025
    cfg.max_drawdown_limit = 0.12
    cfg.max_position_size = 0.08
    cfg.symbols_equity = ["AAPL", "TSLA", "NVDA", "SPY", "QQQ"]
    return cfg

def simulate_market_scenarios():
    """Create diverse market scenarios for testing."""
    return [
        {
            'symbol': 'AAPL',
            'spot_price': 185.50,
            'volatility': 0.22,
            'edge': 0.15,
            'description': 'AAPL - Low bullish conviction, normal volatility',
            'expected_strategy': 'neutral/income'
        },
        {
            'symbol': 'TSLA',
            'spot_price': 245.25,
            'volatility': 0.65,
            'edge': 0.08,
            'description': 'TSLA - Neutral outlook, extreme volatility',
            'expected_strategy': 'volatility play'
        },
        {
            'symbol': 'NVDA', 
            'spot_price': 875.30,
            'volatility': 0.45,
            'edge': -0.72,
            'description': 'NVDA - Strong bearish signal, high volatility',
            'expected_strategy': 'bearish directional'
        },
        {
            'symbol': 'SPY',
            'spot_price': 485.75,
            'volatility': 0.12,
            'edge': 0.85,
            'description': 'SPY - Very bullish signal, low volatility',
            'expected_strategy': 'bullish directional'
        },
        {
            'symbol': 'QQQ',
            'spot_price': 378.90,
            'volatility': 0.18,
            'edge': -0.05,
            'description': 'QQQ - Neutral signal, low volatility',
            'expected_strategy': 'range-bound income'
        }
    ]

def demonstrate_strategy_selection():
    """Show how different market conditions lead to different strategy choices."""
    print("🎯 INTELLIGENT STRATEGY SELECTION")
    print("=" * 70)
    
    cfg = create_demo_config()
    strategy_engine = StrategyEngine(cfg)
    scenarios = simulate_market_scenarios()
    
    print("Market conditions automatically determine optimal strategy:\n")
    
    for scenario in scenarios:
        print(f"📊 {scenario['description']}")
        print(f"   Signal Strength: {scenario['edge']:+.2f}")
        print(f"   Volatility: {scenario['volatility']:.0%}")
        print(f"   Expected: {scenario['expected_strategy']}")
        
        # Market outlook classification
        if scenario['edge'] > 0.3:
            market_outlook = 'bullish'
        elif scenario['edge'] < -0.3:
            market_outlook = 'bearish'  
        else:
            market_outlook = 'neutral'
        
        # Volatility classification
        if scenario['volatility'] > 0.35:
            vol_outlook = 'high'
        elif scenario['volatility'] < 0.20:
            vol_outlook = 'low'
        else:
            vol_outlook = 'medium'
        
        # Get strategy recommendation
        selected_strategy = strategy_engine._select_optimal_strategy(
            edge=scenario['edge'],
            volatility=scenario['volatility'],
            market_outlook=market_outlook,
            vol_outlook=vol_outlook
        )
        
        # Get strategy details
        strategy_info = strategy_engine.get_strategy_info(selected_strategy)
        
        print(f"   🎲 Selected: {selected_strategy.replace('_', ' ').title()}")
        if strategy_info:
            print(f"   📋 Profile: {strategy_info.get('description', 'N/A')}")
            print(f"   💰 Max Profit: {strategy_info.get('max_profit', 'N/A')}")
            print(f"   ⚠️  Max Loss: {strategy_info.get('max_loss', 'N/A')}")
        print()

def demonstrate_portfolio_construction():
    """Show complete portfolio construction with risk management."""
    print("\n💼 INTELLIGENT PORTFOLIO CONSTRUCTION")
    print("=" * 70)
    
    cfg = create_demo_config()
    strategy_engine = StrategyEngine(cfg)
    execution_engine = ExecutionEngine(cfg)
    risk_manager = AdvancedRiskManager(cfg)
    
    scenarios = simulate_market_scenarios()
    
    print("Building diversified options portfolio with risk controls:\n")
    
    portfolio_value = 1_000_000
    total_allocated = 0
    successful_trades = 0
    rejected_trades = 0
    
    for scenario in scenarios:
        symbol = scenario['symbol']
        print(f"🏢 Analyzing {symbol}")
        print(f"   Spot: ${scenario['spot_price']:.2f}")
        print(f"   Signal: {scenario['edge']:+.3f}")
        print(f"   Volatility: {scenario['volatility']:.1%}")
        
        market_data = {
            'spot_price': scenario['spot_price'],
            'volatility': scenario['volatility'],
            'last_close': scenario['spot_price']
        }
        
        # Generate trade recommendation
        trades = strategy_engine.generate(
            edge=scenario['edge'],
            sym=symbol,
            market_data=market_data
        )
        
        for trade in trades:
            # Check position size as percentage of portfolio
            position_pct = trade['size'] / portfolio_value
            
            print(f"   📈 Recommended: {trade['strategy'].replace('_', ' ').title()}")
            print(f"   💵 Position Size: ${trade['size']:,.0f} ({position_pct:.1%} of portfolio)")
            
            # Execute with risk management
            success = execution_engine.place(trade)
            if success:
                total_allocated += trade['size']
                successful_trades += 1
                print(f"   ✅ Trade Executed")
            else:
                rejected_trades += 1
                print(f"   ❌ Trade Rejected by Risk Management")
        
        print()
    
    # Portfolio summary
    portfolio_utilization = total_allocated / portfolio_value
    
    print("📋 PORTFOLIO SUMMARY")
    print("-" * 30)
    print(f"Starting Capital: ${portfolio_value:,.0f}")
    print(f"Total Allocated: ${total_allocated:,.0f}")
    print(f"Portfolio Utilization: {portfolio_utilization:.1%}")
    print(f"Successful Trades: {successful_trades}")
    print(f"Rejected Trades: {rejected_trades}")
    print(f"Cash Remaining: ${portfolio_value - total_allocated:,.0f}")

def demonstrate_strategy_characteristics():
    """Show detailed characteristics of each strategy type."""
    print("\n📚 STRATEGY CHARACTERISTICS GUIDE")
    print("=" * 70)
    
    strategy_guide = {
        'Volatility Strategies': {
            'long_straddle': {
                'setup': 'Buy ATM Call + Buy ATM Put',
                'outlook': 'Big move expected (any direction)',
                'max_profit': 'Unlimited',
                'max_loss': 'Premium paid',
                'best_when': 'Before earnings, events, breakouts'
            },
            'short_straddle': {
                'setup': 'Sell ATM Call + Sell ATM Put', 
                'outlook': 'Expect low volatility, range-bound',
                'max_profit': 'Premium collected',
                'max_loss': 'Unlimited',
                'best_when': 'After volatility crush, stable periods'
            },
            'long_strangle': {
                'setup': 'Buy OTM Call + Buy OTM Put',
                'outlook': 'Big move expected (cheaper than straddle)',
                'max_profit': 'Unlimited',
                'max_loss': 'Premium paid',
                'best_when': 'Expecting volatility, limited budget'
            },
            'short_strangle': {
                'setup': 'Sell OTM Call + Sell OTM Put',
                'outlook': 'Expect moderate range-bound movement',
                'max_profit': 'Premium collected',
                'max_loss': 'Unlimited',
                'best_when': 'Income generation, wider profit zone'
            }
        },
        'Directional Strategies': {
            'bull_call_spread': {
                'setup': 'Buy ITM Call + Sell OTM Call',
                'outlook': 'Moderately bullish',
                'max_profit': 'Strike difference - net debit',
                'max_loss': 'Net debit paid',
                'best_when': 'Bullish but want to limit cost'
            },
            'bear_put_spread': {
                'setup': 'Buy ITM Put + Sell OTM Put',
                'outlook': 'Moderately bearish',
                'max_profit': 'Strike difference - net debit',
                'max_loss': 'Net debit paid',
                'best_when': 'Bearish but want to limit cost'
            },
            'bear_call_spread': {
                'setup': 'Sell ITM Call + Buy OTM Call',
                'outlook': 'Moderately bearish (collect premium)',
                'max_profit': 'Net premium collected',
                'max_loss': 'Strike difference - net premium',
                'best_when': 'Bearish, want immediate credit'
            },
            'bull_put_spread': {
                'setup': 'Sell ITM Put + Buy OTM Put',
                'outlook': 'Moderately bullish (collect premium)',
                'max_profit': 'Net premium collected',
                'max_loss': 'Strike difference - net premium',
                'best_when': 'Bullish, want immediate credit'
            }
        },
        'Range-Bound Strategies': {
            'long_call_butterfly': {
                'setup': 'Buy ITM Call + Sell 2 ATM Calls + Buy OTM Call',
                'outlook': 'Expect minimal movement around middle strike',
                'max_profit': 'Strike difference - net debit',
                'max_loss': 'Net debit paid',
                'best_when': 'Low volatility, precise price target'
            },
            'iron_condor': {
                'setup': 'Sell OTM Put + Buy Further OTM Put + Sell OTM Call + Buy Further OTM Call',
                'outlook': 'Expect price to stay within range',
                'max_profit': 'Net premium collected',
                'max_loss': 'Wing width - net premium',
                'best_when': 'Range-bound market, income generation'
            }
        }
    }
    
    for category, strategies in strategy_guide.items():
        print(f"\n🎪 {category.upper()}")
        print("-" * 50)
        
        for strategy_name, details in strategies.items():
            print(f"\n📋 {strategy_name.replace('_', ' ').title()}")
            print(f"   Setup: {details['setup']}")
            print(f"   Market Outlook: {details['outlook']}")
            print(f"   Max Profit: {details['max_profit']}")
            print(f"   Max Loss: {details['max_loss']}")
            print(f"   Best When: {details['best_when']}")

def demonstrate_risk_scenarios():
    """Show how risk management protects in various scenarios."""
    print("\n🛡️ RISK MANAGEMENT IN ACTION")
    print("=" * 70)
    
    cfg = create_demo_config()
    
    # Aggressive risk settings for demonstration
    cfg.max_portfolio_var = 0.015  # 1.5% VaR limit
    cfg.max_position_size = 0.06   # 6% position limit
    cfg.max_drawdown_limit = 0.08  # 8% drawdown limit
    
    risk_manager = AdvancedRiskManager(cfg)
    
    print("Testing trades under different risk conditions:\n")
    
    # Simulate market stress
    stress_returns = np.random.normal(-0.003, 0.04, 60)  # Volatile, declining market
    risk_manager.portfolio.daily_returns = stress_returns.tolist()
    risk_manager.portfolio.current_drawdown = 0.06  # 6% current drawdown
    risk_manager._calculate_risk_metrics()
    
    print(f"📉 Market Stress Conditions:")
    print(f"   Portfolio VaR (95%): {risk_manager.risk_metrics.portfolio_var_95:.3f}")
    print(f"   Current Drawdown: {risk_manager.portfolio.current_drawdown:.2%}")
    print(f"   Max VaR Limit: {risk_manager.max_portfolio_var:.3f}")
    print(f"   Drawdown Limit: {risk_manager.max_drawdown_limit:.2%}")
    
    # Test various trade sizes
    test_trades = [
        {'symbol': 'AAPL', 'size': 30000, 'strategy': 'long_straddle'},
        {'symbol': 'TSLA', 'size': 80000, 'strategy': 'bull_call_spread'},  # Large position
        {'symbol': 'NVDA', 'size': 45000, 'strategy': 'short_strangle'},
        {'symbol': 'SPY', 'size': 25000, 'strategy': 'iron_condor'},
    ]
    
    print(f"\n🔍 Trade Approval Results:")
    approved_count = 0
    rejected_count = 0
    
    for trade in test_trades:
        approved, reason = risk_manager.check_risk_limits(trade)
        status = "✅ APPROVED" if approved else "❌ REJECTED"
        
        print(f"   {trade['symbol']} {trade['strategy']} ${trade['size']:,}")
        print(f"   {status}: {reason}")
        
        if approved:
            approved_count += 1
        else:
            rejected_count += 1
        print()
    
    print(f"📊 Risk Management Summary:")
    print(f"   Approved Trades: {approved_count}")
    print(f"   Rejected Trades: {rejected_count}")
    print(f"   Protection Rate: {rejected_count / len(test_trades):.0%}")

def main():
    """Run the comprehensive strategies demonstration."""
    print("🚀 ADVANCED OPTIONS STRATEGIES SYSTEM")
    print("🎯 Intelligent • Risk-Aware • Diversified")
    print("=" * 70)
    
    try:
        demonstrate_strategy_selection()
        demonstrate_portfolio_construction()
        demonstrate_strategy_characteristics()
        demonstrate_risk_scenarios()
        
        print(f"\n🎉 DEMONSTRATION COMPLETED!")
        
        print(f"\n💡 SYSTEM CAPABILITIES SHOWCASED:")
        capabilities = [
            "✨ Intelligent strategy selection based on market signals",
            "🎯 12 professional options strategies implemented",
            "🧠 Advanced risk management with VaR and drawdown controls", 
            "⚡ Real-time Greeks calculation for all positions",
            "🏗️ Modular architecture for easy strategy addition",
            "📊 Portfolio-level risk monitoring and correlation analysis",
            "🔄 Seamless integration with existing trading infrastructure",
            "📈 Kelly Criterion position sizing for optimal capital allocation"
        ]
        
        for capability in capabilities:
            print(f"   {capability}")
        
        print(f"\n🔮 READY FOR PRODUCTION:")
        print(f"   • Connect to Interactive Brokers for live trading")
        print(f"   • Add more exotic strategies (condors, calendars, etc.)")
        print(f"   • Implement machine learning for strategy optimization") 
        print(f"   • Build web dashboard for real-time monitoring")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()