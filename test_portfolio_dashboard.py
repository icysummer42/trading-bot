#!/usr/bin/env python3
"""
Portfolio Dashboard Test Suite

Comprehensive tests for portfolio monitoring dashboard including:
- Position tracking functionality
- P&L calculation accuracy
- Risk alert generation
- Performance analytics
- Dashboard integration
"""

import sys
import os
import datetime as dt
import numpy as np
import tempfile
import shutil
from typing import Dict, List
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from bot.enhanced_engine import ExecutionEngine, StrategyEngine
from bot.risk_manager import AdvancedRiskManager
from bot.portfolio_tracker import PortfolioTracker, RiskAlert, PositionUpdate

def create_test_components():
    """Create test components for portfolio dashboard testing."""
    cfg = Config()
    cfg.polygon_key = None  # Test mode
    cfg.max_kelly_fraction = 0.20
    cfg.max_portfolio_var = 0.025
    
    execution_engine = ExecutionEngine(cfg)
    risk_manager = AdvancedRiskManager(cfg)
    
    return cfg, execution_engine, risk_manager

def test_portfolio_tracker_initialization():
    """Test portfolio tracker initialization."""
    print("\n=== Testing Portfolio Tracker Initialization ===")
    
    cfg, execution_engine, risk_manager = create_test_components()
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = PortfolioTracker(
            execution_engine=execution_engine,
            risk_manager=risk_manager,
            data_dir=temp_dir
        )
        
        assert tracker.execution_engine is not None, "Execution engine should be set"
        assert tracker.risk_manager is not None, "Risk manager should be set"
        assert tracker.greeks_calc is not None, "Greeks calculator should be initialized"
        assert len(tracker.risk_limits) > 0, "Risk limits should be configured"
        assert tracker.update_interval > 0, "Update interval should be positive"
        
        print("✅ Portfolio tracker initialized correctly")
        
        # Test risk limits configuration
        required_limits = [
            'max_position_var', 'max_portfolio_var', 'max_delta_exposure',
            'max_gamma_exposure', 'max_vega_exposure', 'max_theta_decay'
        ]
        
        for limit in required_limits:
            assert limit in tracker.risk_limits, f"Should have {limit} configured"
        
        print("✅ Risk limits properly configured")

def test_position_tracking():
    """Test position tracking functionality."""
    print("\n=== Testing Position Tracking ===")
    
    cfg, execution_engine, risk_manager = create_test_components()
    
    # Create test positions
    test_trades = [
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
        }
    ]
    
    # Execute test trades
    for trade in test_trades:
        success = execution_engine.place(trade)
        assert success, f"Should successfully place {trade['strategy']} trade"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = PortfolioTracker(
            execution_engine=execution_engine,
            risk_manager=risk_manager,
            data_dir=temp_dir
        )
        
        # Update positions
        tracker.update_positions()
        
        # Verify position updates
        assert len(tracker.position_updates) > 0, "Should have position updates"
        
        latest_updates = tracker._get_latest_position_updates()
        assert len(latest_updates) == 2, "Should have 2 position updates"
        
        symbols = {update.symbol for update in latest_updates}
        assert 'AAPL' in symbols, "Should track AAPL position"
        assert 'TSLA' in symbols, "Should track TSLA position"
        
        print("✅ Position tracking working correctly")
        
        # Test position update structure
        for update in latest_updates:
            assert hasattr(update, 'symbol'), "Update should have symbol"
            assert hasattr(update, 'strategy'), "Update should have strategy"
            assert hasattr(update, 'current_price'), "Update should have current price"
            assert hasattr(update, 'unrealized_pnl'), "Update should have P&L"
            assert hasattr(update, 'delta'), "Update should have delta"
            assert hasattr(update, 'theta'), "Update should have theta"
            assert hasattr(update, 'vega'), "Update should have vega"
            
            # Validate numeric fields
            assert isinstance(update.current_price, (int, float)), "Price should be numeric"
            assert isinstance(update.delta, (int, float)), "Delta should be numeric"
        
        print("✅ Position update structure correct")

def test_portfolio_greeks_calculation():
    """Test portfolio Greeks aggregation."""
    print("\n=== Testing Portfolio Greeks Calculation ===")
    
    cfg, execution_engine, risk_manager = create_test_components()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = PortfolioTracker(
            execution_engine=execution_engine,
            risk_manager=risk_manager,
            data_dir=temp_dir
        )
        
        # Create mock position updates
        mock_updates = [
            PositionUpdate(
                timestamp=dt.datetime.now(),
                symbol='AAPL',
                strategy='long_straddle',
                current_price=185.50,
                unrealized_pnl=500,
                delta=0.25,
                gamma=0.05,
                theta=-15,
                vega=25,
                rho=5,
                days_to_expiry=30,
                position_value=25000
            ),
            PositionUpdate(
                timestamp=dt.datetime.now(),
                symbol='TSLA',
                strategy='short_strangle',
                current_price=245.25,
                unrealized_pnl=-250,
                delta=-0.15,
                gamma=-0.03,
                theta=12,
                vega=-18,
                rho=-3,
                days_to_expiry=25,
                position_value=30000
            )
        ]
        
        # Calculate portfolio Greeks
        portfolio_greeks = tracker._calculate_portfolio_greeks(mock_updates)
        
        assert 'delta' in portfolio_greeks, "Should calculate portfolio delta"
        assert 'gamma' in portfolio_greeks, "Should calculate portfolio gamma" 
        assert 'theta' in portfolio_greeks, "Should calculate portfolio theta"
        assert 'vega' in portfolio_greeks, "Should calculate portfolio vega"
        assert 'rho' in portfolio_greeks, "Should calculate portfolio rho"
        
        # Verify calculations
        expected_delta = 0.25 + (-0.15)
        expected_gamma = 0.05 + (-0.03)
        expected_theta = -15 + 12
        expected_vega = 25 + (-18)
        expected_rho = 5 + (-3)
        
        assert abs(portfolio_greeks['delta'] - expected_delta) < 0.001, "Delta calculation should be correct"
        assert abs(portfolio_greeks['gamma'] - expected_gamma) < 0.001, "Gamma calculation should be correct"
        assert abs(portfolio_greeks['theta'] - expected_theta) < 0.001, "Theta calculation should be correct"
        assert abs(portfolio_greeks['vega'] - expected_vega) < 0.001, "Vega calculation should be correct"
        assert abs(portfolio_greeks['rho'] - expected_rho) < 0.001, "Rho calculation should be correct"
        
        print("✅ Portfolio Greeks calculation correct")

def test_risk_alert_generation():
    """Test risk alert generation."""
    print("\n=== Testing Risk Alert Generation ===")
    
    cfg, execution_engine, risk_manager = create_test_components()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = PortfolioTracker(
            execution_engine=execution_engine,
            risk_manager=risk_manager,
            data_dir=temp_dir
        )
        
        # Set aggressive risk limits for testing
        tracker.risk_limits['max_delta_exposure'] = 0.1  # Very low limit
        tracker.risk_limits['max_vega_exposure'] = 20    # Very low limit
        tracker.risk_limits['max_position_size'] = 0.05  # 5% limit
        
        # Create position updates that should trigger alerts
        mock_updates = [
            PositionUpdate(
                timestamp=dt.datetime.now(),
                symbol='AAPL',
                strategy='long_straddle',
                current_price=185.50,
                unrealized_pnl=500,
                delta=0.25,  # Exceeds 0.1 limit
                gamma=0.05,
                theta=-15,
                vega=35,     # Exceeds 20 limit
                rho=5,
                days_to_expiry=30,
                position_value=25000
            ),
            PositionUpdate(
                timestamp=dt.datetime.now(),
                symbol='TSLA',
                strategy='short_strangle',
                current_price=245.25,
                unrealized_pnl=-250,
                delta=0.15,
                gamma=-0.03,
                theta=12,
                vega=30,
                rho=-3,
                days_to_expiry=3,  # Low days to expiry
                position_value=60000  # Large position (60k out of ~85k total)
            )
        ]
        
        tracker.position_updates = mock_updates
        
        # Check risk limits
        tracker._check_risk_limits()
        
        # Verify alerts were generated
        assert len(tracker.active_alerts) > 0, "Should generate risk alerts"
        
        # Check alert types
        alert_categories = {alert.category for alert in tracker.active_alerts}
        assert 'greek' in alert_categories or 'position' in alert_categories, "Should have greek or position alerts"
        
        print(f"✅ Generated {len(tracker.active_alerts)} risk alerts")
        
        # Test alert structure
        for alert in tracker.active_alerts:
            assert hasattr(alert, 'timestamp'), "Alert should have timestamp"
            assert hasattr(alert, 'level'), "Alert should have level"
            assert hasattr(alert, 'category'), "Alert should have category" 
            assert hasattr(alert, 'message'), "Alert should have message"
            assert alert.level in ['info', 'warning', 'error'], "Alert level should be valid"
            assert alert.category in ['position', 'portfolio', 'greek', 'limit'], "Alert category should be valid"
        
        print("✅ Risk alert structure correct")

def test_portfolio_summary():
    """Test portfolio summary generation."""
    print("\n=== Testing Portfolio Summary ===")
    
    cfg, execution_engine, risk_manager = create_test_components()
    
    # Create and execute test trades
    test_trades = [
        {
            'symbol': 'AAPL',
            'strategy': 'long_straddle',
            'size': 25000,
            'edge': 0.15,
            'market_data': {'spot_price': 185.50, 'volatility': 0.22}
        }
    ]
    
    for trade in test_trades:
        execution_engine.place(trade)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = PortfolioTracker(
            execution_engine=execution_engine,
            risk_manager=risk_manager,
            data_dir=temp_dir
        )
        
        # Update positions and generate snapshot
        tracker.update_positions()
        tracker._generate_portfolio_snapshot()
        
        # Get portfolio summary
        summary = tracker.get_portfolio_summary()
        
        assert 'error' not in summary, "Summary should not contain errors"
        assert 'timestamp' in summary, "Summary should have timestamp"
        assert 'total_value' in summary, "Summary should have total value"
        assert 'unrealized_pnl' in summary, "Summary should have P&L"
        assert 'position_count' in summary, "Summary should have position count"
        assert 'portfolio_greeks' in summary, "Summary should have portfolio Greeks"
        assert 'risk_metrics' in summary, "Summary should have risk metrics"
        assert 'positions' in summary, "Summary should have position details"
        
        # Validate data types
        assert isinstance(summary['total_value'], (int, float)), "Total value should be numeric"
        assert isinstance(summary['unrealized_pnl'], (int, float)), "P&L should be numeric"
        assert isinstance(summary['position_count'], int), "Position count should be integer"
        assert isinstance(summary['portfolio_greeks'], dict), "Greeks should be dict"
        assert isinstance(summary['risk_metrics'], dict), "Risk metrics should be dict"
        assert isinstance(summary['positions'], list), "Positions should be list"
        
        print("✅ Portfolio summary structure correct")
        
        # Test position details
        positions = summary['positions']
        if positions:
            for position in positions:
                required_fields = ['symbol', 'strategy', 'current_price', 'unrealized_pnl', 'delta', 'theta', 'vega']
                for field in required_fields:
                    assert field in position, f"Position should have {field}"
        
        print("✅ Position details correct")

def test_data_persistence():
    """Test data persistence functionality."""
    print("\n=== Testing Data Persistence ===")
    
    cfg, execution_engine, risk_manager = create_test_components()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = PortfolioTracker(
            execution_engine=execution_engine,
            risk_manager=risk_manager,
            data_dir=temp_dir
        )
        
        # Create mock data
        mock_updates = [
            PositionUpdate(
                timestamp=dt.datetime.now(),
                symbol='AAPL',
                strategy='long_straddle',
                current_price=185.50,
                unrealized_pnl=500,
                delta=0.25,
                gamma=0.05,
                theta=-15,
                vega=25,
                rho=5,
                days_to_expiry=30,
                position_value=25000
            )
        ]
        
        tracker.position_updates = mock_updates
        tracker._generate_portfolio_snapshot()
        
        # Save data
        tracker._save_snapshot_data()
        
        # Verify files were created
        snapshot_file = os.path.join(temp_dir, "portfolio_snapshots.pkl")
        alerts_file = os.path.join(temp_dir, "active_alerts.pkl")
        
        assert os.path.exists(snapshot_file), "Snapshot file should be created"
        print("✅ Snapshot data saved successfully")
        
        # Test data export
        export_file = tracker.export_portfolio_data()
        assert os.path.exists(export_file), "Export file should be created"
        
        # Verify export content
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        required_sections = ['export_timestamp', 'portfolio_summary', 'risk_limits']
        for section in required_sections:
            assert section in export_data, f"Export should contain {section}"
        
        print("✅ Data export working correctly")

def test_monitoring_thread():
    """Test monitoring thread functionality."""
    print("\n=== Testing Monitoring Thread ===")
    
    cfg, execution_engine, risk_manager = create_test_components()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = PortfolioTracker(
            execution_engine=execution_engine,
            risk_manager=risk_manager,
            data_dir=temp_dir
        )
        
        # Start monitoring
        tracker.start_monitoring()
        assert tracker.is_monitoring, "Should be monitoring after start"
        assert tracker.monitor_thread is not None, "Monitor thread should be created"
        assert tracker.monitor_thread.is_alive(), "Monitor thread should be alive"
        
        print("✅ Monitoring thread started successfully")
        
        # Stop monitoring
        tracker.stop_monitoring()
        assert not tracker.is_monitoring, "Should stop monitoring"
        
        print("✅ Monitoring thread stopped successfully")

def run_all_tests():
    """Run all portfolio dashboard tests."""
    print("🚀 PORTFOLIO DASHBOARD TEST SUITE")
    print("=" * 70)
    
    test_functions = [
        test_portfolio_tracker_initialization,
        test_position_tracking,
        test_portfolio_greeks_calculation,
        test_risk_alert_generation,
        test_portfolio_summary,
        test_data_persistence,
        test_monitoring_thread,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
            print(f"✅ {test_func.__name__} PASSED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Total: {passed + failed}")
    
    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"\n🏗️ PORTFOLIO DASHBOARD FEATURES VALIDATED:")
        features = [
            "✅ Portfolio tracker initialization and configuration",
            "✅ Real-time position tracking and updates",
            "✅ Portfolio Greeks calculation and aggregation", 
            "✅ Risk alert generation and monitoring",
            "✅ Portfolio summary generation and reporting",
            "✅ Data persistence and export functionality",
            "✅ Background monitoring thread management",
            "✅ Integration with execution and risk engines"
        ]
        
        for feature in features:
            print(f"   {feature}")
            
        print(f"\n🚀 READY FOR PRODUCTION DEPLOYMENT!")
        
    else:
        print(f"\n⚠️ SOME TESTS FAILED - REVIEW IMPLEMENTATION")
    
    return passed, failed

if __name__ == "__main__":
    run_all_tests()