#!/usr/bin/env python3
"""
Monitoring Setup & Integration Guide
===================================

This script sets up different monitoring solutions and provides integration
examples for the quantitative options trading bot.

Monitoring Options:
1. FastAPI Dashboard (Built-in) - Custom web dashboard
2. Grafana + Prometheus (Recommended) - Industry standard
3. Enhanced Streamlit (Improved) - Better version of existing

Choose the best option for your deployment needs.
"""

import os
import subprocess
import sys
from pathlib import Path
import json
import yaml

def create_monitoring_configs():
    """Create configuration files for different monitoring solutions"""
    
    # Create monitoring directory
    monitoring_dir = Path("monitoring")
    monitoring_dir.mkdir(exist_ok=True)
    
    # 1. Prometheus configuration
    prometheus_config = {
        'global': {
            'scrape_interval': '15s',
            'evaluation_interval': '15s'
        },
        'scrape_configs': [
            {
                'job_name': 'trading-bot',
                'static_configs': [
                    {'targets': ['localhost:8000']}  # Metrics server
                ]
            }
        ]
    }
    
    with open(monitoring_dir / "prometheus.yml", "w") as f:
        yaml.dump(prometheus_config, f)
    
    # 2. Grafana dashboard configuration
    grafana_dashboard = {
        "dashboard": {
            "id": None,
            "title": "Quantitative Options Trading Bot",
            "description": "Real-time monitoring dashboard for options trading system",
            "tags": ["trading", "options", "quant"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "System Health",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "system_health_score",
                            "legendFormat": "Health Score"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                },
                {
                    "id": 2,
                    "title": "API Request Rate",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(api_requests_total[5m])",
                            "legendFormat": "{{endpoint}}"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                },
                {
                    "id": 3,
                    "title": "Trading Performance",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "trading_portfolio_value",
                            "legendFormat": "Portfolio Value"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                },
                {
                    "id": 4,
                    "title": "Error Rate",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(system_errors_total[5m])",
                            "legendFormat": "{{component}}"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
                },
                {
                    "id": 5,
                    "title": "System Resources",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "system_cpu_usage_percent",
                            "legendFormat": "CPU %"
                        },
                        {
                            "expr": "system_memory_usage_mb",
                            "legendFormat": "Memory MB"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
                }
            ],
            "time": {"from": "now-1h", "to": "now"},
            "refresh": "5s"
        }
    }
    
    with open(monitoring_dir / "grafana-dashboard.json", "w") as f:
        json.dump(grafana_dashboard, f, indent=2)
    
    # 3. Docker Compose for monitoring stack
    docker_compose = {
        'version': '3.8',
        'services': {
            'prometheus': {
                'image': 'prom/prometheus:latest',
                'container_name': 'prometheus',
                'ports': ['9090:9090'],
                'volumes': [
                    './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml'
                ],
                'command': [
                    '--config.file=/etc/prometheus/prometheus.yml',
                    '--storage.tsdb.path=/prometheus',
                    '--web.console.libraries=/etc/prometheus/console_libraries',
                    '--web.console.templates=/etc/prometheus/consoles',
                    '--storage.tsdb.retention.time=200h',
                    '--web.enable-lifecycle'
                ]
            },
            'grafana': {
                'image': 'grafana/grafana:latest',
                'container_name': 'grafana',
                'ports': ['3000:3000'],
                'environment': [
                    'GF_SECURITY_ADMIN_USER=admin',
                    'GF_SECURITY_ADMIN_PASSWORD=trading123'
                ],
                'volumes': [
                    'grafana-storage:/var/lib/grafana'
                ]
            },
            'trading-dashboard': {
                'build': '.',
                'container_name': 'trading-dashboard',
                'ports': ['8080:8080'],
                'environment': [
                    'MONITOR_USERNAME=admin',
                    'MONITOR_PASSWORD=trading123'
                ],
                'depends_on': ['prometheus']
            }
        },
        'volumes': {
            'grafana-storage': {}
        }
    }
    
    with open(monitoring_dir / "docker-compose.yml", "w") as f:
        yaml.dump(docker_compose, f)
    
    # 4. Alert rules for Prometheus
    alert_rules = {
        'groups': [
            {
                'name': 'trading_system_alerts',
                'rules': [
                    {
                        'alert': 'HighErrorRate',
                        'expr': 'rate(system_errors_total[5m]) > 0.1',
                        'for': '2m',
                        'labels': {'severity': 'warning'},
                        'annotations': {
                            'summary': 'High error rate detected',
                            'description': 'Error rate is {{ $value }} errors/second'
                        }
                    },
                    {
                        'alert': 'SystemUnhealthy',
                        'expr': 'system_health_score < 50',
                        'for': '1m',
                        'labels': {'severity': 'critical'},
                        'annotations': {
                            'summary': 'Trading system unhealthy',
                            'description': 'System health score is {{ $value }}%'
                        }
                    },
                    {
                        'alert': 'HighCPUUsage',
                        'expr': 'system_cpu_usage_percent > 80',
                        'for': '5m',
                        'labels': {'severity': 'warning'},
                        'annotations': {
                            'summary': 'High CPU usage',
                            'description': 'CPU usage is {{ $value }}%'
                        }
                    },
                    {
                        'alert': 'APIDown',
                        'expr': 'up{job="trading-bot"} == 0',
                        'for': '1m',
                        'labels': {'severity': 'critical'},
                        'annotations': {
                            'summary': 'Trading bot API is down',
                            'description': 'The trading bot metrics endpoint is not responding'
                        }
                    }
                ]
            }
        ]
    }
    
    with open(monitoring_dir / "alert_rules.yml", "w") as f:
        yaml.dump(alert_rules, f)
    
    print("✅ Monitoring configuration files created in ./monitoring/")


def install_monitoring_dependencies():
    """Install required dependencies for monitoring"""
    dependencies = [
        "fastapi>=0.68.0",
        "uvicorn[standard]>=0.15.0",
        "websockets>=10.0",
        "jinja2>=3.0.0",
        "python-multipart>=0.0.5",
        "prometheus-client>=0.12.0",
        "psutil>=5.8.0",
        "pyyaml>=6.0"
    ]
    
    print("📦 Installing monitoring dependencies...")
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {dep}")


def create_enhanced_streamlit_dashboard():
    """Create an enhanced Streamlit dashboard with better features"""
    
    streamlit_code = '''#!/usr/bin/env python3
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
'''
    
    with open("enhanced_streamlit_dashboard.py", "w") as f:
        f.write(streamlit_code)
    
    print("✅ Enhanced Streamlit dashboard created: enhanced_streamlit_dashboard.py")


def create_deployment_scripts():
    """Create deployment scripts for different monitoring solutions"""
    
    # FastAPI Dashboard deployment
    fastapi_script = '''#!/bin/bash
# FastAPI Dashboard Deployment Script

echo "🚀 Starting FastAPI Monitoring Dashboard..."

# Set environment variables
export MONITOR_USERNAME=${MONITOR_USERNAME:-admin}
export MONITOR_PASSWORD=${MONITOR_PASSWORD:-trading123}

# Install dependencies
pip install fastapi uvicorn websockets jinja2 prometheus-client psutil

# Start the dashboard
python monitoring_dashboard.py
'''
    
    with open("deploy_fastapi_dashboard.sh", "w") as f:
        f.write(fastapi_script)
    os.chmod("deploy_fastapi_dashboard.sh", 0o755)
    
    # Grafana + Prometheus deployment
    grafana_script = '''#!/bin/bash
# Grafana + Prometheus Deployment Script

echo "🚀 Starting Grafana + Prometheus Monitoring Stack..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed" 
    exit 1
fi

# Start the monitoring stack
cd monitoring
docker-compose up -d

echo "✅ Monitoring stack started!"
echo "📊 Grafana: http://localhost:3000 (admin/trading123)"
echo "📈 Prometheus: http://localhost:9090"
echo "🎯 Trading Dashboard: http://localhost:8080"
'''
    
    with open("deploy_grafana_stack.sh", "w") as f:
        f.write(grafana_script)
    os.chmod("deploy_grafana_stack.sh", 0o755)
    
    # Enhanced Streamlit deployment  
    streamlit_script = '''#!/bin/bash
# Enhanced Streamlit Dashboard Deployment Script

echo "🚀 Starting Enhanced Streamlit Dashboard..."

# Install Streamlit if not installed
pip install streamlit plotly

# Start the enhanced dashboard
streamlit run enhanced_streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
'''
    
    with open("deploy_streamlit_dashboard.sh", "w") as f:
        f.write(streamlit_script)
    os.chmod("deploy_streamlit_dashboard.sh", 0o755)
    
    print("✅ Deployment scripts created:")
    print("   - deploy_fastapi_dashboard.sh")
    print("   - deploy_grafana_stack.sh") 
    print("   - deploy_streamlit_dashboard.sh")


def display_recommendations():
    """Display monitoring solution recommendations"""
    
    recommendations = """
🎯 MONITORING SOLUTION RECOMMENDATIONS
=====================================

🏆 OPTION 1: FastAPI + Custom Dashboard (RECOMMENDED FOR PRODUCTION)
   Pros:
   ✅ Real-time WebSocket updates
   ✅ Mobile-responsive design
   ✅ Built-in authentication
   ✅ No external dependencies
   ✅ Customizable for trading workflows
   ✅ Professional appearance
   
   Best for: Production trading operations, custom requirements
   
   Usage: python monitoring_dashboard.py
   Access: http://localhost:8080 (admin/trading123)

🥈 OPTION 2: Grafana + Prometheus (INDUSTRY STANDARD)
   Pros:
   ✅ Professional monitoring platform
   ✅ Advanced alerting (email, Slack, SMS)
   ✅ Extensive plugin ecosystem
   ✅ High availability & clustering
   ✅ Industry standard for trading systems
   
   Cons:
   ⚠️ Requires Docker setup
   ⚠️ More complex configuration
   
   Best for: Large-scale deployments, enterprise environments
   
   Usage: ./deploy_grafana_stack.sh
   Access: http://localhost:3000 (admin/trading123)

🥉 OPTION 3: Enhanced Streamlit (IMPROVED EXISTING)
   Pros:
   ✅ Builds on existing knowledge
   ✅ Python-native development
   ✅ Auto-refresh capabilities
   ✅ Professional styling
   
   Cons:
   ⚠️ Limited scalability
   ⚠️ Not ideal for 24/7 operations
   
   Best for: Development, prototyping, small-scale trading
   
   Usage: ./deploy_streamlit_dashboard.sh
   Access: http://localhost:8501

💡 RECOMMENDATION SUMMARY:

For PRODUCTION TRADING:     Use FastAPI Dashboard (#1)
For ENTERPRISE DEPLOYMENT:  Use Grafana + Prometheus (#2)
For DEVELOPMENT/PROTOTYPING: Use Enhanced Streamlit (#3)

All solutions integrate with the enhanced logging system and provide:
- Real-time system monitoring
- Error tracking and alerting
- Trading performance analytics
- Mobile-friendly interfaces
"""
    
    print(recommendations)


def main():
    """Main setup function"""
    print("🛡️ Advanced Logging & Monitoring Setup")
    print("=" * 50)
    
    # Create monitoring configs
    create_monitoring_configs()
    
    # Install dependencies
    install_monitoring_dependencies()
    
    # Create enhanced Streamlit dashboard
    create_enhanced_streamlit_dashboard()
    
    # Create deployment scripts
    create_deployment_scripts()
    
    # Display recommendations
    display_recommendations()
    
    print("\n✅ Monitoring system setup complete!")
    print("\nNext steps:")
    print("1. Choose your monitoring solution from the recommendations above")
    print("2. Run the appropriate deployment script")
    print("3. Configure alerts and notifications as needed")
    print("4. Integrate with your trading system")

if __name__ == "__main__":
    main()