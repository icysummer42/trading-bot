#!/usr/bin/env python3
"""
Portfolio Monitoring Dashboard

Comprehensive real-time portfolio monitoring with:
- Position tracking and P&L analysis
- Greeks exposure monitoring  
- Risk limit alerts and notifications
- Interactive visualizations
- Performance analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from typing import Dict, List, Optional, Any
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from bot.enhanced_engine import ExecutionEngine, StrategyEngine
from bot.risk_manager import AdvancedRiskManager, Position
from bot.greeks import GreeksCalculator
from logger import get_logger

# Configure Streamlit page
st.set_page_config(
    page_title="QuantBot Portfolio Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .profit { color: #00C851; }
    .loss { color: #ff4444; }
    .neutral { color: #666; }
    .alert { 
        background-color: #ffebee; 
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

logger = get_logger("portfolio_dashboard")

class PortfolioDashboard:
    """Main portfolio monitoring dashboard class."""
    
    def __init__(self):
        self.cfg = self._load_config()
        self.execution_engine = ExecutionEngine(self.cfg)
        self.strategy_engine = StrategyEngine(self.cfg)
        self.risk_manager = AdvancedRiskManager(self.cfg)
        self.greeks_calc = GreeksCalculator()
        
        # Mock data for demonstration
        self._initialize_mock_portfolio()
    
    def _load_config(self) -> Config:
        """Load configuration."""
        try:
            cfg = Config()
            return cfg
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            return Config()  # Fallback to defaults
    
    def _initialize_mock_portfolio(self):
        """Initialize with mock portfolio data for demonstration."""
        # Create mock positions
        mock_positions = [
            {
                'symbol': 'AAPL',
                'strategy': 'Long Straddle',
                'size': 25000,
                'entry_date': dt.date.today() - dt.timedelta(days=5),
                'expiry': dt.date.today() + dt.timedelta(days=25),
                'entry_price': 185.50,
                'current_price': 187.25,
                'delta': 0.15,
                'gamma': 0.08,
                'theta': -12.5,
                'vega': 45.2,
                'rho': 8.1,
                'unrealized_pnl': 875
            },
            {
                'symbol': 'TSLA',
                'strategy': 'Short Strangle',
                'size': 30000,
                'entry_date': dt.date.today() - dt.timedelta(days=3),
                'expiry': dt.date.today() + dt.timedelta(days=27),
                'entry_price': 245.25,
                'current_price': 248.75,
                'delta': -0.25,
                'gamma': -0.12,
                'theta': 18.7,
                'vega': -52.1,
                'rho': -12.3,
                'unrealized_pnl': -1250
            },
            {
                'symbol': 'NVDA',
                'strategy': 'Bull Call Spread',
                'size': 40000,
                'entry_date': dt.date.today() - dt.timedelta(days=8),
                'expiry': dt.date.today() + dt.timedelta(days=22),
                'entry_price': 875.30,
                'current_price': 882.15,
                'delta': 0.42,
                'gamma': 0.03,
                'theta': -8.9,
                'vega': 28.4,
                'rho': 15.6,
                'unrealized_pnl': 2450
            },
            {
                'symbol': 'SPY',
                'strategy': 'Iron Condor',
                'size': 20000,
                'entry_date': dt.date.today() - dt.timedelta(days=12),
                'expiry': dt.date.today() + dt.timedelta(days=18),
                'entry_price': 485.75,
                'current_price': 487.10,
                'delta': 0.02,
                'gamma': -0.01,
                'theta': 15.2,
                'vega': -18.5,
                'rho': 3.2,
                'unrealized_pnl': 650
            },
            {
                'symbol': 'QQQ',
                'strategy': 'Long Call Butterfly',
                'size': 15000,
                'entry_date': dt.date.today() - dt.timedelta(days=6),
                'expiry': dt.date.today() + dt.timedelta(days=24),
                'entry_price': 378.90,
                'current_price': 380.45,
                'delta': 0.08,
                'gamma': -0.05,
                'theta': 8.3,
                'vega': -15.7,
                'rho': 4.8,
                'unrealized_pnl': 325
            }
        ]
        
        self.mock_positions = pd.DataFrame(mock_positions)
        
        # Historical P&L data
        dates = pd.date_range(start=dt.date.today() - dt.timedelta(days=30), end=dt.date.today(), freq='D')
        np.random.seed(42)  # For consistent demo data
        
        self.historical_pnl = pd.DataFrame({
            'date': dates,
            'daily_pnl': np.cumsum(np.random.normal(50, 200, len(dates))),
            'portfolio_value': 1000000 + np.cumsum(np.random.normal(50, 200, len(dates)))
        })
    
    def render_dashboard(self):
        """Render the main dashboard."""
        st.title("📊 QuantBot Portfolio Monitor")
        st.markdown("*Real-time portfolio tracking and risk management*")
        
        # Sidebar controls
        self.render_sidebar()
        
        # Main dashboard sections
        self.render_portfolio_summary()
        self.render_positions_table()
        
        # Create columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_pnl_chart()
            self.render_greeks_exposure()
        
        with col2:
            self.render_portfolio_allocation()
            self.render_risk_metrics()
        
        # Additional sections
        self.render_risk_alerts()
        self.render_performance_analytics()
    
    def render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("📋 Portfolio Controls")
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        # Time range selector
        st.sidebar.subheader("Time Range")
        time_range = st.sidebar.selectbox(
            "Select Range",
            options=["1D", "1W", "1M", "3M", "1Y"],
            index=2
        )
        
        # Portfolio filters
        st.sidebar.subheader("Filters")
        
        # Strategy filter
        strategies = self.mock_positions['strategy'].unique().tolist() if not self.mock_positions.empty else []
        selected_strategies = st.sidebar.multiselect(
            "Strategies",
            options=strategies,
            default=strategies
        )
        
        # Symbol filter
        symbols = self.mock_positions['symbol'].unique().tolist() if not self.mock_positions.empty else []
        selected_symbols = st.sidebar.multiselect(
            "Symbols", 
            options=symbols,
            default=symbols
        )
        
        # Risk settings
        st.sidebar.subheader("⚠️ Risk Limits")
        max_position_size = st.sidebar.slider("Max Position Size (%)", 1, 20, 8)
        max_portfolio_var = st.sidebar.slider("Max Portfolio VaR (%)", 1, 10, 3)
        max_drawdown = st.sidebar.slider("Max Drawdown (%)", 5, 25, 15)
        
        # Store selections in session state
        st.session_state.update({
            'time_range': time_range,
            'selected_strategies': selected_strategies,
            'selected_symbols': selected_symbols,
            'max_position_size': max_position_size,
            'max_portfolio_var': max_portfolio_var,
            'max_drawdown': max_drawdown
        })
        
        # Trading controls
        st.sidebar.subheader("🚀 Trading")
        if st.sidebar.button("New Position"):
            st.sidebar.info("Position entry form would open here")
        
        if st.sidebar.button("Close All Positions"):
            st.sidebar.warning("This would close all open positions")
    
    def render_portfolio_summary(self):
        """Render portfolio summary metrics."""
        st.subheader("💼 Portfolio Summary")
        
        # Calculate summary metrics
        total_value = 1000000  # Mock portfolio value
        total_pnl = self.mock_positions['unrealized_pnl'].sum()
        pnl_pct = (total_pnl / total_value) * 100
        total_positions = len(self.mock_positions)
        
        # Calculate Greeks
        total_delta = self.mock_positions['delta'].sum()
        total_gamma = self.mock_positions['gamma'].sum()
        total_theta = self.mock_positions['theta'].sum()
        total_vega = self.mock_positions['vega'].sum()
        
        # Create metrics columns
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                label="Portfolio Value",
                value=f"${total_value:,.0f}",
                delta=f"${total_pnl:+,.0f}"
            )
        
        with col2:
            color = "normal" if abs(pnl_pct) < 1 else "inverse" if pnl_pct < 0 else "normal"
            st.metric(
                label="P&L %",
                value=f"{pnl_pct:+.2f}%",
                delta=f"${total_pnl:+,.0f}"
            )
        
        with col3:
            st.metric(
                label="Positions",
                value=f"{total_positions}",
                delta=None
            )
        
        with col4:
            delta_color = "🔴" if total_delta < -0.5 else "🟢" if total_delta > 0.5 else "🟡"
            st.metric(
                label="Portfolio Delta",
                value=f"{total_delta:+.2f}",
                delta=f"{delta_color}"
            )
        
        with col5:
            theta_color = "🔴" if total_theta < -50 else "🟢" if total_theta > 20 else "🟡"
            st.metric(
                label="Daily Theta",
                value=f"${total_theta:+.0f}",
                delta=f"{theta_color}"
            )
        
        with col6:
            vega_color = "🔴" if abs(total_vega) > 100 else "🟢" if abs(total_vega) < 50 else "🟡"
            st.metric(
                label="Portfolio Vega",
                value=f"${total_vega:+.0f}",
                delta=f"{vega_color}"
            )
    
    def render_positions_table(self):
        """Render positions table."""
        st.subheader("📋 Current Positions")
        
        if self.mock_positions.empty:
            st.info("No open positions")
            return
        
        # Prepare display data
        display_df = self.mock_positions.copy()
        display_df['P&L'] = display_df['unrealized_pnl'].apply(lambda x: f"${x:+,.0f}")
        display_df['Size'] = display_df['size'].apply(lambda x: f"${x:,.0f}")
        display_df['Entry'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
        display_df['Current'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
        display_df['Days to Expiry'] = (display_df['expiry'] - dt.date.today()).dt.days
        
        # Style the dataframe
        styled_df = display_df[[
            'symbol', 'strategy', 'Size', 'Entry', 'Current', 'P&L', 
            'Days to Expiry', 'delta', 'gamma', 'theta', 'vega'
        ]].style.format({
            'delta': '{:+.3f}',
            'gamma': '{:+.3f}', 
            'theta': '${:+.1f}',
            'vega': '${:+.1f}'
        }).applymap(
            lambda x: 'color: green' if '+' in str(x) and '$' in str(x) else 
                     'color: red' if '-' in str(x) and '$' in str(x) else '',
            subset=['P&L']
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Position actions
        st.subheader("Position Actions")
        selected_position = st.selectbox(
            "Select Position",
            options=self.mock_positions['symbol'].tolist(),
            format_func=lambda x: f"{x} ({self.mock_positions[self.mock_positions['symbol']==x]['strategy'].iloc[0]})"
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📊 Analyze"):
                st.info(f"Would show detailed analysis for {selected_position}")
        with col2:
            if st.button("⚡ Adjust"):
                st.info(f"Would show adjustment options for {selected_position}")
        with col3:
            if st.button("🔒 Hedge"):
                st.info(f"Would show hedging options for {selected_position}")
        with col4:
            if st.button("❌ Close"):
                st.warning(f"Would close position in {selected_position}")
    
    def render_pnl_chart(self):
        """Render P&L chart."""
        st.subheader("📈 P&L Performance")
        
        # Create P&L chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.historical_pnl['date'],
            y=self.historical_pnl['daily_pnl'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#1f77b4', width=2),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))
        
        fig.update_layout(
            title="30-Day P&L Trend",
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            hovermode='x unified',
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # P&L statistics
        current_pnl = self.historical_pnl['daily_pnl'].iloc[-1]
        max_pnl = self.historical_pnl['daily_pnl'].max()
        min_pnl = self.historical_pnl['daily_pnl'].min()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current P&L", f"${current_pnl:+,.0f}")
        with col2:
            st.metric("Max P&L", f"${max_pnl:+,.0f}")
        with col3:
            st.metric("Max Drawdown", f"${min_pnl:+,.0f}")
    
    def render_greeks_exposure(self):
        """Render Greeks exposure chart."""
        st.subheader("🔢 Greeks Exposure")
        
        # Aggregate Greeks by position
        greeks_data = {
            'Delta': self.mock_positions['delta'].sum(),
            'Gamma': self.mock_positions['gamma'].sum() * 100,  # Scale for visibility
            'Theta': self.mock_positions['theta'].sum(),
            'Vega': self.mock_positions['vega'].sum(),
        }
        
        # Create radar chart
        fig = go.Figure()
        
        categories = list(greeks_data.keys())
        values = list(greeks_data.values())
        
        # Normalize values for radar chart
        max_abs = max(abs(v) for v in values)
        normalized_values = [v / max_abs * 100 for v in values]
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=categories,
            fill='toself',
            name='Greeks Exposure',
            line=dict(color='#ff7f0e')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-100, 100]
                )),
            showlegend=False,
            title="Portfolio Greeks Profile",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Greeks details
        for greek, value in greeks_data.items():
            if greek == 'Gamma':
                value /= 100  # Unscale
            
            color = "🔴" if value < -10 else "🟢" if value > 10 else "🟡"
            st.write(f"{color} **{greek}**: {value:+.1f}")
    
    def render_portfolio_allocation(self):
        """Render portfolio allocation charts."""
        st.subheader("🥧 Portfolio Allocation")
        
        # By Strategy
        strategy_allocation = self.mock_positions.groupby('strategy')['size'].sum().reset_index()
        
        fig = px.pie(
            strategy_allocation,
            values='size',
            names='strategy',
            title="Allocation by Strategy",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(height=250, showlegend=True)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # By Symbol
        symbol_allocation = self.mock_positions.groupby('symbol')['size'].sum().reset_index()
        
        fig2 = px.bar(
            symbol_allocation,
            x='symbol',
            y='size',
            title="Allocation by Symbol",
            color='symbol',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig2.update_layout(height=250, showlegend=False)
        fig2.update_yaxis(title="Position Size ($)")
        
        st.plotly_chart(fig2, use_container_width=True)
    
    def render_risk_metrics(self):
        """Render risk metrics dashboard."""
        st.subheader("⚠️ Risk Metrics")
        
        # Mock risk calculations
        portfolio_var = 2.1  # 2.1% VaR
        max_drawdown = 8.5   # 8.5% max drawdown
        sharpe_ratio = 1.85
        concentration_risk = 32.5  # 32.5% in largest position
        
        # Risk gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = portfolio_var,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Portfolio VaR (95%)"},
            delta = {'reference': 3.0},
            gauge = {
                'axis': {'range': [None, 5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgray"},
                    {'range': [2, 3], 'color': "yellow"},
                    {'range': [3, 5], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 3.0}}
        ))
        
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics table
        risk_metrics = pd.DataFrame({
            'Metric': ['Portfolio VaR (95%)', 'Max Drawdown', 'Sharpe Ratio', 'Concentration Risk'],
            'Current': [f"{portfolio_var:.1f}%", f"{max_drawdown:.1f}%", f"{sharpe_ratio:.2f}", f"{concentration_risk:.1f}%"],
            'Limit': ["3.0%", "15.0%", ">1.0", "<40.0%"],
            'Status': ["✅ OK", "✅ OK", "✅ OK", "✅ OK"]
        })
        
        st.dataframe(risk_metrics, use_container_width=True, hide_index=True)
    
    def render_risk_alerts(self):
        """Render risk alerts section."""
        st.subheader("🚨 Risk Alerts")
        
        # Mock alerts
        alerts = [
            {
                'level': 'warning',
                'message': 'TSLA position showing high vega exposure (-52.1)',
                'timestamp': dt.datetime.now() - dt.timedelta(minutes=15)
            },
            {
                'level': 'info', 
                'message': 'Portfolio theta decay accelerating (5 positions < 20 DTE)',
                'timestamp': dt.datetime.now() - dt.timedelta(hours=2)
            }
        ]
        
        for alert in alerts:
            if alert['level'] == 'warning':
                st.warning(f"⚠️ **{alert['timestamp'].strftime('%H:%M')}**: {alert['message']}")
            elif alert['level'] == 'error':
                st.error(f"🚨 **{alert['timestamp'].strftime('%H:%M')}**: {alert['message']}")
            else:
                st.info(f"ℹ️ **{alert['timestamp'].strftime('%H:%M')}**: {alert['message']}")
        
        if not alerts:
            st.success("✅ No active risk alerts")
    
    def render_performance_analytics(self):
        """Render performance analytics section."""
        st.subheader("📊 Performance Analytics")
        
        # Create tabs for different analytics
        tab1, tab2, tab3 = st.tabs(["📈 Returns", "📉 Drawdown", "🎯 Strategy Performance"])
        
        with tab1:
            self.render_returns_analysis()
        
        with tab2:
            self.render_drawdown_analysis()
        
        with tab3:
            self.render_strategy_performance()
    
    def render_returns_analysis(self):
        """Render returns analysis."""
        # Mock returns data
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        cumulative_returns = np.cumprod(1 + returns) - 1
        
        # Returns distribution
        fig = px.histogram(
            x=returns * 100,
            nbins=30,
            title="Daily Returns Distribution",
            labels={'x': 'Daily Return (%)', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ann. Return", f"{np.mean(returns) * 252 * 100:.1f}%")
        with col2:
            st.metric("Ann. Volatility", f"{np.std(returns) * np.sqrt(252) * 100:.1f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{(np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)):.2f}")
        with col4:
            st.metric("Max Daily Loss", f"{np.min(returns) * 100:.2f}%")
    
    def render_drawdown_analysis(self):
        """Render drawdown analysis."""
        # Mock drawdown data
        portfolio_values = 1000000 + np.cumsum(np.random.normal(100, 500, 252))
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='red'),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown Over Time",
            yaxis_title="Drawdown (%)",
            xaxis_title="Days"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Drawdown", f"{drawdown[-1]:.1f}%")
        with col2:
            st.metric("Max Drawdown", f"{np.min(drawdown):.1f}%")
        with col3:
            st.metric("Recovery Time", "12 days")
    
    def render_strategy_performance(self):
        """Render strategy performance comparison."""
        # Mock strategy performance data
        strategy_perf = pd.DataFrame({
            'Strategy': ['Long Straddle', 'Short Strangle', 'Bull Call Spread', 'Iron Condor', 'Long Call Butterfly'],
            'Total P&L': [875, -1250, 2450, 650, 325],
            'Win Rate': [0.65, 0.45, 0.78, 0.58, 0.62],
            'Avg Winner': [1250, 800, 950, 420, 380],
            'Avg Loser': [-680, -1100, -550, -380, -290],
            'Sharpe': [1.45, 0.85, 2.1, 1.2, 1.1]
        })
        
        # Performance comparison chart
        fig = px.scatter(
            strategy_perf,
            x='Win Rate',
            y='Total P&L',
            size='Sharpe',
            color='Strategy',
            title="Strategy Performance Overview",
            labels={'Win Rate': 'Win Rate', 'Total P&L': 'Total P&L ($)'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy performance table
        st.dataframe(
            strategy_perf.style.format({
                'Total P&L': '${:+,.0f}',
                'Win Rate': '{:.1%}',
                'Avg Winner': '${:,.0f}',
                'Avg Loser': '${:+,.0f}',
                'Sharpe': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )


def main():
    """Main dashboard application."""
    dashboard = PortfolioDashboard()
    dashboard.render_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "© 2024 QuantBot Research | "
        f"Last Updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "📧 Alerts: Enabled | "
        "🔄 Auto-refresh: Every 30s"
    )


if __name__ == "__main__":
    main()