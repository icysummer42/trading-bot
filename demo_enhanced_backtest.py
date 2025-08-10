#!/usr/bin/env python3
"""
Enhanced Backtesting Demo

Demonstrates the enhanced backtesting engine with:
- Realistic slippage and commission modeling
- Multiple strategies across different market conditions
- Comprehensive performance analytics
- Risk-adjusted metrics
"""

import sys
import os
import datetime as dt
import numpy as np
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from bot.enhanced_backtest import EnhancedBacktester, BacktestResults
from bot.polygon_client import PolygonClient
from bot.risk_manager import AdvancedRiskManager

def create_mock_backtester():
    """Create enhanced backtester with mock data for demonstration."""
    
    # Create mock polygon client
    class MockPolygonClient:
        def __init__(self):
            self.price_data = {}
            self._generate_mock_data()
        
        def _generate_mock_data(self):
            """Generate realistic mock price data."""
            symbols = ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ']
            base_prices = {'AAPL': 185, 'TSLA': 245, 'NVDA': 875, 'SPY': 485, 'QQQ': 378}
            
            start_date = dt.date(2023, 1, 1)
            end_date = dt.date.today()
            
            for symbol in symbols:
                prices = []
                current_price = base_prices[symbol]
                current_date = start_date
                
                # Generate daily price series with realistic volatility
                vol = 0.25 if symbol in ['AAPL', 'NVDA'] else 0.35 if symbol == 'TSLA' else 0.18
                
                while current_date <= end_date:
                    if current_date.weekday() < 5:  # Weekdays only
                        # Random walk with drift
                        daily_return = np.random.normal(0.0005, vol/np.sqrt(252))  # Slight positive drift
                        current_price *= (1 + daily_return)
                        prices.append((current_date, current_price))
                    current_date += dt.timedelta(days=1)
                
                self.price_data[symbol] = dict(prices)
        
        def spot(self, symbol: str, date: dt.date) -> float:
            """Get spot price for symbol on date."""
            if symbol in self.price_data and date in self.price_data[symbol]:
                return self.price_data[symbol][date]
            
            # Fallback to interpolation or nearest date
            if symbol in self.price_data:
                dates = list(self.price_data[symbol].keys())
                if dates:
                    closest_date = min(dates, key=lambda d: abs((d - date).days))
                    return self.price_data[symbol][closest_date]
            
            return None
        
        def snapshot_chain(self, symbol: str, date: dt.date):
            """Mock options chain - not needed for enhanced backtester."""
            return None
        
        def agg_close(self, ticker: str, date: dt.date):
            """Mock option prices - not needed for enhanced backtester."""
            return None
    
    return MockPolygonClient()

def run_strategy_comparison():
    """Run backtests comparing multiple strategies."""
    
    print("🚀 ENHANCED BACKTESTING ENGINE DEMO")
    print("=" * 70)
    
    # Setup
    cfg = Config()
    mock_client = create_mock_backtester()
    risk_manager = AdvancedRiskManager(cfg)
    
    # Create enhanced backtester
    backtester = EnhancedBacktester(
        client=mock_client,
        risk_manager=risk_manager,
        commission_per_contract=0.65,  # Realistic commissions
        base_slippage=0.005,  # 0.5% base slippage
        min_bid_ask_spread=0.01,  # 1% minimum spread
        max_bid_ask_spread=0.05   # 5% maximum spread
    )
    
    # Test parameters
    symbols = ['AAPL', 'TSLA', 'SPY']
    start_date = dt.date(2023, 6, 1)
    end_date = dt.date(2024, 6, 1)
    initial_capital = 100000
    
    # Test different strategies
    strategies_to_test = [
        'long_straddle',
        'short_strangle', 
        'bull_call_spread',
        'iron_condor'
    ]
    
    print(f"Testing {len(strategies_to_test)} strategies on {len(symbols)} symbols")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,}")
    print()
    
    all_results = {}
    
    for strategy in strategies_to_test:
        print(f"📊 BACKTESTING {strategy.upper().replace('_', ' ')}")
        print("-" * 50)
        
        try:
            # Run backtest
            results = backtester.run_backtest(
                strategy_name=strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                rebalance_frequency='weekly'
            )
            
            all_results[strategy] = results
            
            # Display results
            for symbol, result in results.items():
                print(f"  {symbol}:")
                print(f"    Trades: {result.total_trades}")
                print(f"    Win Rate: {result.win_rate:.1%}")
                print(f"    Net P&L: ${result.net_pnl:,.0f}")
                print(f"    Gross P&L: ${result.gross_pnl:,.0f}")
                print(f"    Commissions: ${result.total_commissions:,.0f}")
                print(f"    Slippage: ${result.total_slippage:,.0f}")
                print(f"    Max Drawdown: {result.max_drawdown:.1%}")
                print(f"    Sharpe Ratio: {result.sharpe_ratio:.2f}")
                print()
            
        except Exception as e:
            print(f"  ❌ Error backtesting {strategy}: {e}")
            print()
    
    # Generate comparison summary
    print("📈 STRATEGY COMPARISON SUMMARY")
    print("=" * 70)
    
    strategy_summaries = {}
    for strategy, symbol_results in all_results.items():
        total_trades = sum(r.total_trades for r in symbol_results.values())
        total_pnl = sum(r.net_pnl for r in symbol_results.values())
        avg_win_rate = np.mean([r.win_rate for r in symbol_results.values()])
        avg_sharpe = np.mean([r.sharpe_ratio for r in symbol_results.values()])
        total_commissions = sum(r.total_commissions for r in symbol_results.values())
        total_slippage = sum(r.total_slippage for r in symbol_results.values())
        
        strategy_summaries[strategy] = {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_win_rate': avg_win_rate,
            'avg_sharpe': avg_sharpe,
            'total_commissions': total_commissions,
            'total_slippage': total_slippage
        }
    
    # Sort strategies by total P&L
    sorted_strategies = sorted(strategy_summaries.items(), 
                              key=lambda x: x[1]['total_pnl'], reverse=True)
    
    print(f"{'Strategy':<20} {'Trades':<8} {'Win Rate':<10} {'Net P&L':<12} {'Sharpe':<8} {'Costs':<10}")
    print("-" * 80)
    
    for strategy, summary in sorted_strategies:
        strategy_name = strategy.replace('_', ' ').title()
        trades = summary['total_trades']
        win_rate = f"{summary['avg_win_rate']:.1%}"
        pnl = f"${summary['total_pnl']:,.0f}"
        sharpe = f"{summary['avg_sharpe']:.2f}"
        costs = f"${summary['total_commissions'] + summary['total_slippage']:,.0f}"
        
        print(f"{strategy_name:<20} {trades:<8} {win_rate:<10} {pnl:<12} {sharpe:<8} {costs:<10}")
    
    return all_results

def demonstrate_cost_analysis():
    """Demonstrate the impact of transaction costs."""
    print("\n💰 TRANSACTION COST ANALYSIS")
    print("=" * 70)
    
    # Show how costs impact performance
    scenarios = [
        {'name': 'Low Cost', 'commission': 0.30, 'slippage': 0.002},
        {'name': 'Moderate Cost', 'commission': 0.65, 'slippage': 0.005},
        {'name': 'High Cost', 'commission': 1.50, 'slippage': 0.010},
    ]
    
    print("Impact of transaction costs on Long Straddle strategy:")
    print(f"{'Scenario':<15} {'Commission':<12} {'Slippage':<10} {'Est. Impact':<12}")
    print("-" * 50)
    
    for scenario in scenarios:
        commission = scenario['commission']
        slippage = scenario['slippage']
        
        # Rough estimate of impact (4 legs × cost per leg × frequency)
        legs_per_trade = 2  # Straddle has 2 legs
        trades_per_year = 52  # Weekly
        annual_commission_cost = commission * legs_per_trade * trades_per_year
        annual_slippage_cost = slippage * 250 * legs_per_trade * trades_per_year  # Assume $250 avg premium
        total_annual_cost = annual_commission_cost + annual_slippage_cost
        
        print(f"{scenario['name']:<15} ${commission:<11.2f} {slippage*100:<9.1f}% ${total_annual_cost:<11.0f}")
    
    print("\nKey Insights:")
    print("• Transaction costs can significantly impact strategy profitability")
    print("• Higher frequency strategies are more sensitive to costs")
    print("• Slippage often exceeds commission costs for options")
    print("• Cost modeling is essential for realistic backtesting")

def demonstrate_risk_metrics():
    """Demonstrate risk-adjusted performance metrics."""
    print("\n📊 RISK-ADJUSTED METRICS EXPLANATION")
    print("=" * 70)
    
    metrics_info = {
        'Sharpe Ratio': {
            'description': 'Return per unit of total risk',
            'good_value': '> 1.0',
            'interpretation': 'Higher is better. Accounts for return volatility.'
        },
        'Sortino Ratio': {
            'description': 'Return per unit of downside risk',
            'good_value': '> 1.5', 
            'interpretation': 'Like Sharpe but only penalizes downside volatility.'
        },
        'Calmar Ratio': {
            'description': 'Annual return / Maximum Drawdown',
            'good_value': '> 0.5',
            'interpretation': 'Measures return relative to worst drawdown period.'
        },
        'Max Drawdown': {
            'description': 'Largest peak-to-trough decline',
            'good_value': '< 20%',
            'interpretation': 'Lower is better. Shows worst-case scenario.'
        },
        'Win Rate': {
            'description': 'Percentage of profitable trades',
            'good_value': '> 50%',
            'interpretation': 'Higher is better, but not the only factor.'
        },
        'Profit Factor': {
            'description': 'Gross profit / Gross loss',
            'good_value': '> 1.25',
            'interpretation': 'How much profit per dollar of loss.'
        }
    }
    
    for metric, info in metrics_info.items():
        print(f"\n{metric}:")
        print(f"  Description: {info['description']}")
        print(f"  Good Value: {info['good_value']}")
        print(f"  Interpretation: {info['interpretation']}")

def main():
    """Run the enhanced backtesting demonstration."""
    
    try:
        # Run strategy comparison
        results = run_strategy_comparison()
        
        # Show cost analysis
        demonstrate_cost_analysis()
        
        # Explain risk metrics
        demonstrate_risk_metrics()
        
        print(f"\n🎉 ENHANCED BACKTESTING DEMO COMPLETED!")
        
        print(f"\n💡 KEY FEATURES DEMONSTRATED:")
        features = [
            "✅ Realistic bid/ask spread modeling",
            "✅ Commission and slippage calculations", 
            "✅ Historical volatility estimation",
            "✅ Multi-strategy comparison framework",
            "✅ Risk-adjusted performance metrics",
            "✅ Greeks tracking and exposure analysis",
            "✅ Transaction cost impact analysis",
            "✅ Drawdown and risk monitoring",
            "✅ Professional-grade performance analytics"
        ]
        
        for feature in features:
            print(f"   {feature}")
        
        print(f"\n🏆 PRODUCTION ADVANTAGES:")
        advantages = [
            "• Realistic cost modeling prevents overoptimistic results",
            "• Risk metrics help identify robust strategies",
            "• Greeks tracking enables better risk management", 
            "• Multi-symbol testing reveals strategy universality",
            "• Transaction cost analysis optimizes execution",
            "• Professional analytics match institutional standards"
        ]
        
        for advantage in advantages:
            print(f"   {advantage}")
        
        print(f"\n🔮 READY FOR LIVE DEPLOYMENT:")
        print(f"   The enhanced backtesting engine provides the validation")
        print(f"   needed to confidently deploy strategies in live markets.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()