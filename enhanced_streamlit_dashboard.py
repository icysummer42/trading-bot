#!/usr/bin/env python3
"""
Enhanced Streamlit Dashboard for Production Trading
=================================================

Improved version of the Streamlit dashboard with:
- Real-time auto-refresh
- Mobile-responsive layout
- Better error handling
- Professional styling
- Alert management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

from advanced_logging import metrics_collector, get_trading_logger
from error_handling import get_system_health, get_error_statistics

# Configure page
st.set_page_config(
    page_title="Trading System Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stMetric { background-color: #1E1E1E; padding: 1rem; border-radius: 0.5rem; }
    .metric-success { color: #00FF88 !important; }
    .metric-warning { color: #FF9900 !important; }
    .metric-critical { color: #FF3366 !important; }
    .sidebar .sidebar-content { background-color: #0E1117; }
</style>
""", unsafe_allow_html=True)

# Initialize logger
logger = get_trading_logger(__name__)

@st.cache_data(ttl=5)  # Cache for 5 seconds
def get_dashboard_data():
    """Get fresh dashboard data with caching"""
    return {
        'system_health': get_system_health(),
        'error_stats': get_error_statistics(),
        'timestamp': datetime.now()
    }

def main():
    # Header
    st.title("🎯 Trading System Monitor")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.selectbox("Refresh Interval", [1, 5, 10, 30], index=1)
        
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        st.markdown("---")
        st.subheader("📊 System Status")
        
        # Get current data
        data = get_dashboard_data()
        health = data['system_health']
        
        status = health.get('overall_status', 'unknown')
        if status == 'healthy':
            st.success(f"✅ {status.title()}")
        elif status in ['warning', 'degraded']:
            st.warning(f"⚠️ {status.title()}")
        else:
            st.error(f"🚨 {status.title()}")
    
    # Main dashboard
    data = get_dashboard_data()
    health = data['system_health']
    errors = data['error_stats']
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "System Health", 
            health.get('overall_status', 'Unknown').title(),
            delta=None
        )
    
    with col2:
        error_count = errors.get('total_errors', 0)
        st.metric(
            "Total Errors",
            error_count,
            delta=f"+{errors.get('recent_errors_count', 0)} recent"
        )
    
    with col3:
        sources = health.get('sources', {})
        healthy_sources = sum(1 for s in sources.values() if s.get('status') == 'healthy')
        st.metric(
            "Data Sources",
            f"{healthy_sources}/{len(sources)}",
            delta="Active"
        )
    
    with col4:
        st.metric(
            "Last Update",
            data['timestamp'].strftime("%H:%M:%S"),
            delta=None
        )
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Error Statistics")
        
        if errors.get('total_errors', 0) > 0:
            # Error categories pie chart
            categories = errors.get('errors_by_category', {})
            if categories:
                fig = px.pie(
                    values=list(categories.values()),
                    names=list(categories.keys()),
                    title="Errors by Category"
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No error categories to display")
        else:
            st.success("🎉 No errors recorded!")
    
    with col2:
        st.subheader("🌐 API Status")
        
        sources = health.get('sources', {})
        for name, status in sources.items():
            status_text = status.get('status', 'unknown')
            if status_text == 'healthy':
                st.success(f"✅ {name.upper()}: {status_text}")
            elif status_text in ['degraded', 'warning']:
                st.warning(f"⚠️ {name.upper()}: {status_text}")
            else:
                st.error(f"❌ {name.upper()}: {status_text}")
    
    # Detailed Information
    st.markdown("---")
    st.subheader("📋 Detailed Information")
    
    tab1, tab2, tab3 = st.tabs(["System Health", "Error Details", "Raw Data"])
    
    with tab1:
        st.json(health)
    
    with tab2:
        if errors.get('total_errors', 0) > 0:
            st.json(errors)
        else:
            st.info("No errors to display")
    
    with tab3:
        st.json(data)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
