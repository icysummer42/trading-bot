#!/bin/bash
# Enhanced Streamlit Dashboard Deployment Script

echo "🚀 Starting Enhanced Streamlit Dashboard..."

# Install Streamlit if not installed
pip install streamlit plotly

# Start the enhanced dashboard
streamlit run enhanced_streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
