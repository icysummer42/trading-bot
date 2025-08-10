#!/bin/bash
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
