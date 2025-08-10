#!/bin/bash
# FastAPI Dashboard Deployment Script

echo "🚀 Starting FastAPI Monitoring Dashboard..."

# Set environment variables
export MONITOR_USERNAME=${MONITOR_USERNAME:-admin}
export MONITOR_PASSWORD=${MONITOR_PASSWORD:-trading123}

# Install dependencies
pip install fastapi uvicorn websockets jinja2 prometheus-client psutil

# Start the dashboard
python monitoring_dashboard.py
