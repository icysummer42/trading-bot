#!/usr/bin/env python3
"""
Portfolio Dashboard Demo

Demonstrates the portfolio monitoring capabilities including:
- Real-time position tracking
- P&L monitoring with Greeks
- Risk alerts and limit monitoring
- Performance analytics
- Interactive visualizations
"""

import sys
import os
import datetime as dt
import time
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from bot.enhanced_engine import ExecutionEngine, StrategyEngine
from bot.risk_manager import AdvancedRiskManager
from bot.portfolio_tracker import PortfolioTracker, RiskAlert
from logger import get_logger

logger = get_logger("portfolio_demo")

def create_demo_portfolio():
    """Create a demo portfolio with mock positions."""
    print("🏗️ Setting up demo portfolio...")
    
    cfg = Config()
    cfg.polygon_key = None  # Demo mode
    
    # Initialize components
    execution_engine = ExecutionEngine(cfg)
    risk_manager = AdvancedRiskManager(cfg)
    strategy_engine = StrategyEngine(cfg)
    
    # Create mock positions in execution engine
    mock_trades = [
        {
            'symbol': 'AAPL',
            'strategy': 'long_straddle',
            'size': 25000,
            'edge': 0.15,
            'market_data': {'spot_price': 185.50, 'volatility': 0.22}
        },
        {
            'symbol': 'TSLA', 
            'strategy': 'short_strangle',
            'size': 30000,
            'edge': -0.08,
            'market_data': {'spot_price': 245.25, 'volatility': 0.65}
        },
        {
            'symbol': 'NVDA',
            'strategy': 'bull_call_spread',
            'size': 40000,
            'edge': 0.72,
            'market_data': {'spot_price': 875.30, 'volatility': 0.45}
        },
        {
            'symbol': 'SPY',
            'strategy': 'iron_condor',
            'size': 20000,
            'edge': -0.05,
            'market_data': {'spot_price': 485.75, 'volatility': 0.18}
        },
        {
            'symbol': 'QQQ',
            'strategy': 'long_call_butterfly',
            'size': 15000,
            'edge': 0.25,
            'market_data': {'spot_price': 378.90, 'volatility': 0.20}
        }
    ]
    
    # Execute mock trades
    for trade in mock_trades:
        success = execution_engine.place(trade)
        if success:
            print(f"  ✅ Placed {trade['strategy']} on {trade['symbol']}: ${trade['size']:,}")
        else:
            print(f"  ❌ Failed to place {trade['strategy']} on {trade['symbol']}")
    
    return execution_engine, risk_manager, strategy_engine

def demonstrate_portfolio_tracking():
    """Demonstrate real-time portfolio tracking."""
    print("\n📊 PORTFOLIO TRACKING DEMONSTRATION")
    print("=" * 70)
    
    # Setup
    execution_engine, risk_manager, strategy_engine = create_demo_portfolio()
    
    # Initialize portfolio tracker
    portfolio_tracker = PortfolioTracker(
        execution_engine=execution_engine,
        risk_manager=risk_manager,
        data_dir="demo_portfolio_data"
    )
    
    print(f"\n🎯 Starting real-time monitoring...")
    
    # Start monitoring
    portfolio_tracker.start_monitoring()
    
    try:
        # Simulate real-time monitoring for demo
        for i in range(5):
            print(f"\n⏰ Update #{i+1} - {dt.datetime.now().strftime('%H:%M:%S')}")
            
            # Force position update
            portfolio_tracker.update_positions()
            
            # Get current portfolio summary
            summary = portfolio_tracker.get_portfolio_summary()
            
            if 'error' not in summary:
                print(f"   💼 Portfolio Value: ${summary['total_value']:,.0f}")
                print(f"   📈 Unrealized P&L: ${summary['unrealized_pnl']:+,.0f}")
                print(f"   🏢 Positions: {summary['position_count']}")
                
                # Show portfolio Greeks
                greeks = summary.get('portfolio_greeks', {})
                print(f"   🔢 Delta: {greeks.get('delta', 0):+.3f}")
                print(f"   ⚡ Gamma: {greeks.get('gamma', 0):+.3f}")
                print(f"   📉 Theta: ${greeks.get('theta', 0):+.0f}/day")
                print(f"   📊 Vega: ${greeks.get('vega', 0):+.0f}")
                
                # Show risk metrics
                risk_metrics = summary.get('risk_metrics', {})
                if 'concentration' in risk_metrics:
                    print(f"   ⚠️ Concentration: {risk_metrics['concentration']:.1%}")
                    print(f"   📏 VaR (95%): {risk_metrics.get('var_95_percent', 0):.1%}")
                
                # Show recent alerts
                alerts = summary.get('recent_alerts', [])
                if alerts:
                    print(f"   🚨 Active Alerts: {len(alerts)}")
                    for alert in alerts[-2:]:  # Show last 2 alerts
                        level_emoji = "🚨" if alert['level'] == 'error' else "⚠️" if alert['level'] == 'warning' else "ℹ️"
                        print(f"     {level_emoji} {alert['message']}")
                else:
                    print(f"   ✅ No active alerts")
                
                # Show position details
                positions = summary.get('positions', [])
                print(f"   \n   📋 Position Details:")
                for pos in positions[:3]:  # Show first 3 positions
                    pnl_emoji = "🟢" if pos['unrealized_pnl'] > 0 else "🔴" if pos['unrealized_pnl'] < 0 else "🟡"
                    print(f"     {pnl_emoji} {pos['symbol']} {pos['strategy']}: "
                          f"${pos['unrealized_pnl']:+,.0f} "
                          f"(Δ={pos['delta']:+.2f}, θ=${pos['theta']:+.0f})")
            else:
                print(f"   ❌ Error: {summary['error']}")
            
            time.sleep(2)  # Wait before next update
    
    finally:
        # Stop monitoring
        portfolio_tracker.stop_monitoring()
        print(f"\n⏹️ Stopped monitoring")
    
    return portfolio_tracker

def demonstrate_historical_performance(portfolio_tracker):
    """Demonstrate historical performance analytics."""
    print(f"\n📈 HISTORICAL PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Get historical performance
    performance = portfolio_tracker.get_historical_performance(days=30)
    
    if 'error' not in performance:
        print(f"📊 Performance Summary (Last {performance['period_days']} days):")
        print(f"   📈 Total Return: {performance['total_return']:+.2%}")
        print(f"   📉 Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"   📊 Volatility: {performance['volatility']:.2%}")
        print(f"   💰 Current P&L: ${performance['current_pnl']:+,.0f}")
        print(f"   🎯 Best P&L: ${performance['max_pnl']:+,.0f}")
        print(f"   💸 Worst P&L: ${performance['min_pnl']:+,.0f}")
        print(f"   📅 Data Points: {performance['data_points']}")
        
        # Show time series sample
        time_series = performance.get('time_series', {})
        if time_series and time_series.get('timestamps'):
            print(f"\n   📉 Recent P&L Trend:")
            timestamps = time_series['timestamps'][-5:]  # Last 5 points
            pnls = time_series['pnls'][-5:]
            
            for i, (ts, pnl) in enumerate(zip(timestamps, pnls)):
                time_str = dt.datetime.fromisoformat(ts).strftime('%m/%d %H:%M')
                trend = "📈" if i > 0 and pnl > pnls[i-1] else "📉" if i > 0 and pnl < pnls[i-1] else "➡️"
                print(f"     {trend} {time_str}: ${pnl:+,.0f}")
    else:
        print(f"❌ Error getting performance data: {performance['error']}")

def demonstrate_risk_analytics(portfolio_tracker):
    """Demonstrate risk analytics and alerts."""
    print(f"\n⚠️ RISK ANALYTICS DEMONSTRATION")
    print("=" * 70)
    
    # Get current portfolio summary
    summary = portfolio_tracker.get_portfolio_summary()
    
    if 'error' not in summary:
        risk_metrics = summary.get('risk_metrics', {})
        
        print(f"🛡️ Risk Metrics Analysis:")
        print(f"   📊 Total Positions: {risk_metrics.get('total_positions', 0)}")
        print(f"   🎯 Concentration Risk: {risk_metrics.get('concentration', 0):.1%}")
        print(f"   📏 Portfolio VaR (95%): {risk_metrics.get('var_95_percent', 0):.1%}")
        print(f"   📈 Portfolio Volatility: ${risk_metrics.get('portfolio_volatility', 0):,.0f}")
        print(f"   ⏰ Average Days to Expiry: {risk_metrics.get('avg_dte', 0):.0f}")
        
        # Risk limit analysis
        print(f"\n🚦 Risk Limit Status:")
        limits = portfolio_tracker.risk_limits
        
        # Check concentration
        concentration = risk_metrics.get('concentration', 0)
        max_concentration = limits.get('max_concentration', 0.4)
        concentration_status = "✅ OK" if concentration <= max_concentration else "⚠️ HIGH" if concentration <= max_concentration * 1.2 else "🚨 BREACH"
        print(f"   📊 Concentration: {concentration:.1%} (limit: {max_concentration:.1%}) {concentration_status}")
        
        # Check VaR
        portfolio_var = risk_metrics.get('var_95_percent', 0)
        max_var = limits.get('max_portfolio_var', 0.03)
        var_status = "✅ OK" if portfolio_var <= max_var else "⚠️ HIGH" if portfolio_var <= max_var * 1.2 else "🚨 BREACH"
        print(f"   📏 VaR (95%): {portfolio_var:.1%} (limit: {max_var:.1%}) {var_status}")
        
        # Check Greeks limits
        greeks = summary.get('portfolio_greeks', {})
        delta_exposure = abs(greeks.get('delta', 0))
        max_delta = limits.get('max_delta_exposure', 0.5)
        delta_status = "✅ OK" if delta_exposure <= max_delta else "⚠️ HIGH" if delta_exposure <= max_delta * 1.2 else "🚨 BREACH"
        print(f"   🔢 Delta Exposure: {delta_exposure:.3f} (limit: {max_delta:.3f}) {delta_status}")
        
        vega_exposure = abs(greeks.get('vega', 0))
        max_vega = limits.get('max_vega_exposure', 1000)
        vega_status = "✅ OK" if vega_exposure <= max_vega else "⚠️ HIGH" if vega_exposure <= max_vega * 1.2 else "🚨 BREACH"
        print(f"   📊 Vega Exposure: ${vega_exposure:.0f} (limit: ${max_vega:.0f}) {vega_status}")
        
        # Show active alerts
        alerts = summary.get('recent_alerts', [])
        print(f"\n🚨 Active Alerts ({len(alerts)}):")
        if alerts:
            for alert in alerts[-5:]:  # Show last 5 alerts
                level_emoji = "🚨" if alert['level'] == 'error' else "⚠️" if alert['level'] == 'warning' else "ℹ️"
                time_str = dt.datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                symbol_str = f" [{alert['symbol']}]" if alert.get('symbol') else ""
                print(f"     {level_emoji} {time_str}{symbol_str}: {alert['message']}")
        else:
            print(f"     ✅ No active alerts - all risk metrics within limits")

def demonstrate_data_export(portfolio_tracker):
    """Demonstrate portfolio data export."""
    print(f"\n💾 DATA EXPORT DEMONSTRATION")
    print("=" * 70)
    
    try:
        # Export portfolio data
        export_file = portfolio_tracker.export_portfolio_data()
        print(f"✅ Portfolio data exported to: {export_file}")
        
        # Show file size
        file_size = os.path.getsize(export_file) / 1024  # KB
        print(f"📁 Export file size: {file_size:.1f} KB")
        
        # Show export contents summary
        import json
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        print(f"📄 Export contains:")
        print(f"   📊 Portfolio summary with {len(export_data.get('portfolio_summary', {}).get('positions', []))} positions")
        print(f"   📈 Historical performance data")
        print(f"   ⚠️ {len(export_data.get('recent_alerts', []))} recent alerts")
        print(f"   🛡️ Risk limits configuration")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

def main():
    """Run the portfolio dashboard demonstration."""
    
    print("🚀 PORTFOLIO MONITORING DASHBOARD DEMO")
    print("🎯 Real-time • Risk-aware • Performance Analytics")
    print("=" * 70)
    
    try:
        # Run portfolio tracking demo
        portfolio_tracker = demonstrate_portfolio_tracking()
        
        # Run analytics demos
        demonstrate_historical_performance(portfolio_tracker)
        demonstrate_risk_analytics(portfolio_tracker)
        demonstrate_data_export(portfolio_tracker)
        
        print(f"\n🎉 PORTFOLIO DASHBOARD DEMO COMPLETED!")
        
        print(f"\n💡 KEY FEATURES DEMONSTRATED:")
        features = [
            "✅ Real-time position tracking and P&L monitoring",
            "✅ Portfolio Greeks aggregation and exposure analysis", 
            "✅ Risk limit monitoring with automated alerts",
            "✅ Historical performance analytics and metrics",
            "✅ Multi-strategy portfolio management",
            "✅ Concentration and correlation risk analysis",
            "✅ Interactive risk dashboard capabilities",
            "✅ Data export and reporting functionality"
        ]
        
        for feature in features:
            print(f"   {feature}")
        
        print(f"\n🏆 PRODUCTION ADVANTAGES:")
        advantages = [
            "• Real-time portfolio visibility prevents blind spots",
            "• Risk alerts enable proactive position management",
            "• Greeks aggregation identifies portfolio-level exposures", 
            "• Performance analytics guide strategy optimization",
            "• Historical tracking enables pattern recognition",
            "• Automated monitoring reduces manual oversight burden"
        ]
        
        for advantage in advantages:
            print(f"   {advantage}")
        
        print(f"\n🔮 READY FOR LIVE DEPLOYMENT:")
        print(f"   The portfolio dashboard provides institutional-grade")
        print(f"   monitoring and risk management for live trading operations.")
        
        print(f"\n🌐 WEB DASHBOARD:")
        print(f"   Run 'streamlit run portfolio_dashboard.py' to launch")
        print(f"   the interactive web interface for real-time monitoring.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()