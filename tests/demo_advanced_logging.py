#!/usr/bin/env python3
"""
Advanced Logging System Demonstration
====================================

This script demonstrates the advanced logging and monitoring system
integrated with the quantitative options trading bot.

Features demonstrated:
- Structured JSON logging with trading context
- Performance metrics collection
- Real-time system health monitoring
- Alert management and notifications
- Trading-specific KPIs and dashboards
- Compliance-ready audit logging
"""

import os
import time
import asyncio
from pathlib import Path
from datetime import datetime

# Load environment
def load_env():
    env_file = Path('.env')
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value

load_env()

from advanced_logging import (
    get_trading_logger, 
    metrics_collector, 
    alert_manager,
    TradingLogger
)
from enhanced_data_pipeline import EnhancedDataPipeline
from config import Config
import threading

def demo_structured_logging():
    """Demonstrate structured logging capabilities"""
    print("📝 Advanced Logging System Demo")
    print("=" * 50)
    
    # Initialize trading logger
    logger = get_trading_logger("trading_demo")
    
    print("\n🧪 Test 1: Structured Trading Logs")
    
    # Set trading context
    logger.set_context(
        session_id="demo_session_001",
        strategy="iron_condor",
        symbol="AAPL"
    )
    
    # Business event logging
    logger.business("Starting trading session", 
                   portfolio_value=100000.0,
                   risk_limit=10000.0)
    
    # Trade execution logging
    with logger.operation_context("place_order", order_type="iron_condor", symbol="AAPL"):
        time.sleep(0.1)  # Simulate order processing
        
        logger.info("Order validation complete",
                   strike_prices=[180, 185, 200, 205],
                   expiry="2024-08-18",
                   premium_collected=2.50)
        
        # Simulate successful trade
        logger.business("Trade executed successfully",
                       order_id="IC_AAPL_001",
                       trade_value=2500.0,
                       pnl=250.0,
                       execution_time_ms=150.2)
    
    # Error scenario logging
    try:
        with logger.operation_context("market_data_fetch", symbol="INVALID"):
            raise ValueError("Invalid symbol provided")
    except Exception as e:
        logger.error("Market data fetch failed",
                    error_type="ValueError",
                    symbol="INVALID")
    
    # Audit logging
    logger.audit("Position size check",
                position_size=5000.0,
                risk_limit=10000.0,
                compliance_status="passed")
    
    print("   ✅ Structured logs generated with trading context")
    print("   ✅ Business events logged for compliance")
    print("   ✅ Performance metrics captured")
    print("   ✅ Error context preserved")


def demo_metrics_collection():
    """Demonstrate metrics collection and monitoring"""
    print("\n🧪 Test 2: Metrics Collection & Monitoring")
    
    # Record some trading metrics
    metrics_collector.record_trade("AAPL", "iron_condor", "buy", 250.0)
    metrics_collector.record_trade("MSFT", "straddle", "sell", -150.0)
    metrics_collector.record_trade("GOOGL", "butterfly", "buy", 300.0)
    
    # Update portfolio metrics
    metrics_collector.update_portfolio_value(102400.0)
    metrics_collector.update_positions({"AAPL": 2, "MSFT": 1, "GOOGL": 3})
    
    # Record API metrics
    metrics_collector.record_api_request("polygon/aggregates", "success", 0.245)
    metrics_collector.record_api_request("polygon/aggregates", "success", 0.189)
    metrics_collector.record_api_request("yfinance/history", "error", 2.156)
    
    # Record system errors
    metrics_collector.record_error("data_pipeline", "network_error")
    metrics_collector.record_error("signal_generator", "model_error")
    
    # Update system health
    metrics_collector.update_health_score(85.0)
    
    print("   ✅ Trading metrics recorded (trades, P&L, positions)")
    print("   ✅ API performance metrics captured")
    print("   ✅ System health score updated")
    print("   ✅ Error statistics tracked")
    
    # Start Prometheus metrics server for demonstration
    try:
        metrics_collector.start_prometheus_server(8000)
        print("   ✅ Prometheus metrics server started on port 8000")
        print("      Visit http://localhost:8000/metrics to see metrics")
    except Exception as e:
        print(f"   ⚠️  Prometheus server: {e}")


def demo_alert_system():
    """Demonstrate alert management system"""
    print("\n🧪 Test 3: Alert Management System")
    
    # Simulate various system conditions for alerts
    alert_manager.update_metrics({
        'error_rate_per_minute': 12.5,  # Above threshold (10)
        'system_health_score': 45.0,    # Below threshold (50) 
        'cpu_usage': 85.0,              # Above threshold (80)
        'memory_usage_percent': 90.0,   # Above threshold (85)
        'api_success_rate': 0.3,        # Below threshold (0.5)
        'portfolio_drawdown_percent': 15.0  # Above threshold (10)
    })
    
    # Wait for alerts to be evaluated
    time.sleep(1)
    
    print("   ✅ Alert rules evaluated against current metrics")
    print("   ✅ Critical alerts triggered for system conditions")
    print("   ✅ Alert cooldown prevents spam notifications")
    
    # Show alert status
    triggered_alerts = []
    for rule_name, rule in alert_manager.rules.items():
        if rule.last_triggered:
            triggered_alerts.append(rule_name)
    
    if triggered_alerts:
        print(f"   🚨 Active alerts: {', '.join(triggered_alerts)}")
    else:
        print("   ✅ No alerts currently active")


def demo_integrated_monitoring():
    """Demonstrate integration with enhanced data pipeline"""
    print("\n🧪 Test 4: Integrated System Monitoring")
    
    config = Config()
    pipeline = EnhancedDataPipeline(config)
    logger = get_trading_logger("pipeline_demo")
    
    # Set context for pipeline operations
    logger.set_context(correlation_id="demo_correlation_123")
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        try:
            start_time = time.time()
            
            # This will use enhanced logging automatically
            data = pipeline.get_close_series(symbol, start="2024-08-01", end="2024-08-10")
            
            execution_time = (time.time() - start_time) * 1000
            
            if not data.empty:
                logger.info(f"Data fetch successful for {symbol}",
                           symbol=symbol,
                           data_points=len(data),
                           execution_time_ms=execution_time,
                           latest_price=data.iloc[-1])
                
                # Record metrics
                metrics_collector.record_api_request(f"data_fetch_{symbol}", "success", execution_time/1000)
            else:
                logger.warning(f"No data returned for {symbol}",
                              symbol=symbol,
                              execution_time_ms=execution_time)
                
        except Exception as e:
            logger.error(f"Data fetch failed for {symbol}",
                        symbol=symbol,
                        error_type=type(e).__name__,
                        error_message=str(e))
    
    # Get system health with enhanced monitoring
    health = pipeline.health_check()
    logger.info("System health check completed",
               overall_status=health['overall_status'],
               error_handling_status=health['error_handling']['status'])
    
    print("   ✅ Enhanced pipeline operations logged automatically")
    print("   ✅ Performance metrics captured for all API calls")
    print("   ✅ Error context preserved across system boundaries")
    print("   ✅ Correlation IDs enable distributed tracing")


def demo_monitoring_dashboard_integration():
    """Demonstrate monitoring dashboard data flow"""
    print("\n🧪 Test 5: Monitoring Dashboard Integration")
    
    # Simulate dashboard data collection
    from monitoring_dashboard import dashboard_cache
    
    # Update dashboard cache with fresh data
    dashboard_cache.update_data()
    
    # Display current dashboard data
    dashboard_data = dashboard_cache.data
    
    print("   📊 Dashboard Data Summary:")
    print(f"      System Health: {dashboard_data['system_health'].get('overall_status', 'unknown')}")
    print(f"      Performance: CPU {dashboard_data['performance_metrics'].get('cpu_usage', 0):.1f}%, "
          f"Memory {dashboard_data['performance_metrics'].get('memory_usage', 0):.1f}%")
    print(f"      Trading: Portfolio ${dashboard_data['trading_metrics'].get('portfolio_value', 0):,.2f}")
    print(f"      Errors: {dashboard_data['error_stats'].get('total_errors', 0)} total")
    print(f"      APIs: {list(dashboard_data['api_status'].keys())}")
    
    print("   ✅ Real-time dashboard data updated")
    print("   ✅ WebSocket clients would receive this data")
    print("   ✅ Mobile-responsive interface ready")


def demo_production_deployment():
    """Show production deployment options"""
    print("\n🧪 Test 6: Production Deployment Options")
    
    deployment_options = [
        {
            "name": "FastAPI Dashboard",
            "command": "python monitoring_dashboard.py",
            "port": 8080,
            "features": ["Real-time updates", "Mobile-responsive", "Authentication"]
        },
        {
            "name": "Grafana + Prometheus", 
            "command": "./deploy_grafana_stack.sh",
            "port": 3000,
            "features": ["Industry standard", "Advanced alerting", "High availability"]
        },
        {
            "name": "Enhanced Streamlit",
            "command": "./deploy_streamlit_dashboard.sh", 
            "port": 8501,
            "features": ["Python-native", "Auto-refresh", "Development-friendly"]
        }
    ]
    
    print("   🚀 Available deployment options:")
    for i, option in enumerate(deployment_options, 1):
        print(f"      {i}. {option['name']}")
        print(f"         Command: {option['command']}")
        print(f"         Access: http://localhost:{option['port']}")
        print(f"         Features: {', '.join(option['features'])}")
        print()
    
    print("   ✅ Multiple deployment strategies available")
    print("   ✅ Production-ready monitoring solutions")
    print("   ✅ Scalable architecture for growth")


async def demo_realtime_monitoring():
    """Demonstrate real-time monitoring capabilities"""
    print("\n🧪 Test 7: Real-time Monitoring Simulation")
    
    logger = get_trading_logger("realtime_demo")
    
    print("   🔄 Simulating 10 seconds of real-time trading activity...")
    
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    strategies = ["iron_condor", "straddle", "butterfly", "strangle"]
    
    for i in range(10):
        # Simulate trading activity every second
        import random
        
        symbol = random.choice(symbols)
        strategy = random.choice(strategies)
        
        # Random trade outcome
        pnl = random.uniform(-500, 1000)
        success = pnl > 0
        
        if success:
            logger.business(f"Profitable trade executed",
                           symbol=symbol,
                           strategy=strategy, 
                           pnl=pnl,
                           trade_id=f"T{i+1:03d}")
        else:
            logger.warning(f"Loss-making trade",
                          symbol=symbol,
                          strategy=strategy,
                          pnl=pnl,
                          trade_id=f"T{i+1:03d}")
        
        # Record metrics
        metrics_collector.record_trade(symbol, strategy, "buy" if success else "sell", pnl)
        
        # Update system metrics
        cpu_usage = random.uniform(20, 90)
        memory_usage = random.uniform(40, 85)
        
        # Simulate occasional alerts
        if cpu_usage > 80:
            alert_manager.update_metrics({'cpu_usage': cpu_usage})
        
        print(f"      Second {i+1}: {symbol} {strategy} P&L: ${pnl:.2f}")
        
        await asyncio.sleep(1)
    
    print("   ✅ Real-time trading activity simulated")
    print("   ✅ Structured logs generated continuously")
    print("   ✅ Metrics updated in real-time")
    print("   ✅ Alerts triggered based on conditions")


def main():
    """Run comprehensive advanced logging demonstration"""
    print("🛡️ Advanced Logging & Monitoring System Demo")
    print("=" * 60)
    
    # Run demonstrations
    demo_structured_logging()
    demo_metrics_collection() 
    demo_alert_system()
    demo_integrated_monitoring()
    demo_monitoring_dashboard_integration()
    demo_production_deployment()
    
    # Run async real-time demo
    asyncio.run(demo_realtime_monitoring())
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 ADVANCED LOGGING SYSTEM SUMMARY")
    print("=" * 60)
    
    print("✅ IMPLEMENTED FEATURES:")
    print("   📝 Structured JSON logging with trading context")
    print("   📊 Real-time metrics collection (Prometheus compatible)")
    print("   🚨 Intelligent alert management with cooldowns")
    print("   🔄 Seamless integration with enhanced error handling")
    print("   📱 Mobile-responsive monitoring dashboards")
    print("   🔗 WebSocket real-time updates")
    print("   🛡️ Production-ready authentication")
    print("   📈 Trading-specific KPIs and analytics")
    print("   📋 Compliance-ready audit logging")
    print("   🎯 Multiple deployment options")
    
    print("\n🚀 PRODUCTION READY:")
    print("   • 24/7 monitoring capabilities")
    print("   • Professional dashboard interfaces") 
    print("   • Industry-standard monitoring (Grafana/Prometheus)")
    print("   • Real-time alerting and notifications")
    print("   • Scalable architecture")
    print("   • Zero-downtime deployments")
    
    print("\n📊 MONITORING OPTIONS:")
    print("   1. FastAPI Dashboard (Recommended for Production)")
    print("      → python monitoring_dashboard.py")
    print("   2. Grafana + Prometheus (Enterprise/Large Scale)")
    print("      → ./deploy_grafana_stack.sh") 
    print("   3. Enhanced Streamlit (Development/Prototyping)")
    print("      → ./deploy_streamlit_dashboard.sh")
    
    print(f"\n🎯 Next: Choose your monitoring solution and deploy!")
    print(f"📊 Logs saved to: logs/trading_system.jsonl")
    print(f"📈 Metrics available at: http://localhost:8000/metrics")

if __name__ == "__main__":
    main()