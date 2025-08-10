#!/usr/bin/env python3
"""
Production Trading Monitoring Dashboard
=====================================

FastAPI-based real-time monitoring dashboard for quantitative options trading.
Designed for 24/7 operations with WebSocket updates, mobile responsiveness,
and professional monitoring capabilities.

Features:
- Real-time WebSocket data streaming
- Mobile-responsive design
- User authentication
- REST API for data access
- Alert management interface
- System health monitoring
- Trading performance analytics
- Integration with logging system

Alternative to Streamlit for production use.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import secrets
import threading
import time
import os
from pathlib import Path

from advanced_logging import metrics_collector, alert_manager, get_trading_logger
from error_handling import get_system_health, get_error_statistics

# Initialize components
app = FastAPI(title="Trading System Monitor", version="1.0.0")
security = HTTPBasic()
logger = get_trading_logger(__name__)

# Simple authentication (can be enhanced)
ADMIN_USERNAME = os.getenv("MONITOR_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("MONITOR_PASSWORD", "trading123")

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify user credentials"""
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials.username

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected", client_count=len(self.active_connections))
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected", client_count=len(self.active_connections))
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection is broken, remove it
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Data cache for dashboard
class DashboardDataCache:
    def __init__(self):
        self.data = {
            'system_health': {},
            'error_stats': {},
            'performance_metrics': {},
            'trading_metrics': {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'portfolio_value': 100000.0,  # Default
                'positions': []
            },
            'alerts': [],
            'api_status': {}
        }
        self.last_update = datetime.now()
        
        # Start background data collection
        self.start_data_collection()
    
    def start_data_collection(self):
        """Start background thread to collect and cache data"""
        def collect_data():
            while True:
                try:
                    self.update_data()
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    logger.error(f"Error updating dashboard data: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=collect_data, daemon=True)
        thread.start()
    
    def update_data(self):
        """Update cached dashboard data"""
        # System health
        self.data['system_health'] = get_system_health()
        
        # Error statistics
        self.data['error_stats'] = get_error_statistics()
        
        # Performance metrics (mock for now - would integrate with real system)
        import psutil
        self.data['performance_metrics'] = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
        }
        
        # Trading metrics (mock - would integrate with real portfolio)
        self.data['trading_metrics'].update({
            'last_update': datetime.now().isoformat(),
            'uptime_hours': (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).total_seconds() / 3600
        })
        
        # API status check
        self.data['api_status'] = {
            'polygon': 'healthy',  # Would check actual API status
            'yfinance': 'degraded',
            'openai': 'healthy'
        }
        
        self.last_update = datetime.now()

dashboard_cache = DashboardDataCache()

# API Endpoints
@app.get("/")
async def dashboard(request: Request, username: str = Depends(verify_credentials)):
    """Main dashboard page"""
    return HTMLResponse(DASHBOARD_HTML)

@app.get("/api/health")
async def get_health(username: str = Depends(verify_credentials)):
    """Get system health status"""
    return dashboard_cache.data['system_health']

@app.get("/api/metrics")
async def get_metrics(username: str = Depends(verify_credentials)):
    """Get performance metrics"""
    return {
        'system_health': dashboard_cache.data['system_health'],
        'performance': dashboard_cache.data['performance_metrics'],
        'trading': dashboard_cache.data['trading_metrics'],
        'errors': dashboard_cache.data['error_stats'],
        'api_status': dashboard_cache.data['api_status'],
        'last_update': dashboard_cache.last_update.isoformat()
    }

@app.get("/api/alerts")
async def get_alerts(username: str = Depends(verify_credentials)):
    """Get current alerts"""
    # Get recent alerts from alert manager
    alerts = []
    for rule_name, rule in alert_manager.rules.items():
        if rule.last_triggered and (datetime.now() - rule.last_triggered).total_seconds() < 3600:
            alerts.append({
                'name': rule_name,
                'severity': rule.severity,
                'message': rule.message,
                'triggered_at': rule.last_triggered.isoformat()
            })
    
    return {'alerts': alerts}

@app.post("/api/alerts/{alert_name}/acknowledge")
async def acknowledge_alert(alert_name: str, username: str = Depends(verify_credentials)):
    """Acknowledge an alert"""
    if alert_name in alert_manager.rules:
        alert_manager.rules[alert_name].last_triggered = None
        logger.info(f"Alert {alert_name} acknowledged by {username}")
        return {"status": "acknowledged"}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send real-time updates
            data = {
                'type': 'update',
                'timestamp': datetime.now().isoformat(),
                'data': dashboard_cache.data
            }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(1)  # Send updates every second
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# HTML Template for Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading System Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f0f0f;
            color: #ffffff;
            line-height: 1.6;
        }
        
        .header {
            background: #1a1a1a;
            padding: 1rem 2rem;
            border-bottom: 2px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            color: #00ff88;
            font-size: 1.5rem;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 2s infinite;
        }
        
        .status-dot.warning { background: #ff9900; }
        .status-dot.critical { background: #ff3366; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            padding: 1rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 1.5rem;
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            border-color: #555;
        }
        
        .card h3 {
            color: #00ff88;
            margin-bottom: 1rem;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid #333;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-value {
            font-weight: bold;
            color: #00ff88;
        }
        
        .metric-value.warning { color: #ff9900; }
        .metric-value.critical { color: #ff3366; }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        
        .progress-fill {
            height: 100%;
            background: #00ff88;
            transition: width 0.3s ease;
        }
        
        .progress-fill.warning { background: #ff9900; }
        .progress-fill.critical { background: #ff3366; }
        
        .alert {
            background: #2a1a1a;
            border-left: 4px solid #ff3366;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-radius: 4px;
        }
        
        .alert.warning {
            border-left-color: #ff9900;
        }
        
        .alert-time {
            font-size: 0.8rem;
            color: #888;
        }
        
        .api-status {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        .api-badge {
            background: #333;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .api-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #00ff88;
        }
        
        .api-dot.degraded { background: #ff9900; }
        .api-dot.down { background: #ff3366; }
        
        .timestamp {
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            background: #1a1a1a;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-size: 0.8rem;
            color: #888;
        }
        
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
                padding: 0.5rem;
            }
            
            .header {
                padding: 1rem;
                flex-direction: column;
                gap: 0.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Trading System Monitor</h1>
        <div class="status-indicator">
            <div class="status-dot" id="systemStatus"></div>
            <span id="systemStatusText">Initializing...</span>
        </div>
    </div>
    
    <div class="container">
        <!-- System Health Card -->
        <div class="card">
            <h3>🛡️ System Health</h3>
            <div class="metric">
                <span>Overall Status</span>
                <span class="metric-value" id="overallStatus">Loading...</span>
            </div>
            <div class="metric">
                <span>Error Handling</span>
                <span class="metric-value" id="errorHandling">Loading...</span>
            </div>
            <div class="metric">
                <span>Data Sources</span>
                <span class="metric-value" id="dataSources">Loading...</span>
            </div>
            <div class="metric">
                <span>Uptime</span>
                <span class="metric-value" id="uptime">Loading...</span>
            </div>
        </div>
        
        <!-- Performance Metrics Card -->
        <div class="card">
            <h3>⚡ Performance</h3>
            <div class="metric">
                <span>CPU Usage</span>
                <span class="metric-value" id="cpuUsage">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="cpuBar" style="width: 0%"></div>
            </div>
            
            <div class="metric">
                <span>Memory Usage</span>
                <span class="metric-value" id="memoryUsage">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="memoryBar" style="width: 0%"></div>
            </div>
            
            <div class="metric">
                <span>Disk Usage</span>
                <span class="metric-value" id="diskUsage">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="diskBar" style="width: 0%"></div>
            </div>
        </div>
        
        <!-- Trading Metrics Card -->
        <div class="card">
            <h3>📈 Trading Performance</h3>
            <div class="metric">
                <span>Portfolio Value</span>
                <span class="metric-value" id="portfolioValue">$0.00</span>
            </div>
            <div class="metric">
                <span>Total P&L</span>
                <span class="metric-value" id="totalPnl">$0.00</span>
            </div>
            <div class="metric">
                <span>Total Trades</span>
                <span class="metric-value" id="totalTrades">0</span>
            </div>
            <div class="metric">
                <span>Win Rate</span>
                <span class="metric-value" id="winRate">0%</span>
            </div>
            <div class="metric">
                <span>Open Positions</span>
                <span class="metric-value" id="openPositions">0</span>
            </div>
        </div>
        
        <!-- Error Statistics Card -->
        <div class="card">
            <h3>⚠️ Error Statistics</h3>
            <div class="metric">
                <span>Total Errors</span>
                <span class="metric-value" id="totalErrors">0</span>
            </div>
            <div class="metric">
                <span>Recent Errors</span>
                <span class="metric-value" id="recentErrors">0</span>
            </div>
            <div class="metric">
                <span>Critical Errors</span>
                <span class="metric-value" id="criticalErrors">0</span>
            </div>
            <div class="metric">
                <span>Last Error</span>
                <span class="metric-value" id="lastError">None</span>
            </div>
        </div>
        
        <!-- API Status Card -->
        <div class="card">
            <h3>🌐 API Status</h3>
            <div class="api-status" id="apiStatus">
                <!-- API badges will be populated by JavaScript -->
            </div>
        </div>
        
        <!-- Recent Alerts Card -->
        <div class="card">
            <h3>🚨 Recent Alerts</h3>
            <div id="alertsList">
                <p style="color: #888;">No recent alerts</p>
            </div>
        </div>
    </div>
    
    <div class="timestamp" id="lastUpdate">Last updated: Never</div>
    
    <script>
        // WebSocket connection for real-time updates
        let ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            if (message.type === 'update') {
                updateDashboard(message.data);
            }
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            document.getElementById('systemStatusText').textContent = 'Connection Error';
            document.getElementById('systemStatus').className = 'status-dot critical';
        };
        
        function updateDashboard(data) {
            // Update system health
            const health = data.system_health || {};
            document.getElementById('overallStatus').textContent = health.overall_status || 'Unknown';
            document.getElementById('errorHandling').textContent = health.error_handling?.status || 'Unknown';
            document.getElementById('dataSources').textContent = Object.keys(health.sources || {}).length;
            
            // Update system status indicator
            const statusDot = document.getElementById('systemStatus');
            const statusText = document.getElementById('systemStatusText');
            
            if (health.overall_status === 'healthy') {
                statusDot.className = 'status-dot';
                statusText.textContent = 'System Healthy';
            } else if (health.overall_status === 'degraded') {
                statusDot.className = 'status-dot warning';
                statusText.textContent = 'System Degraded';
            } else {
                statusDot.className = 'status-dot critical';
                statusText.textContent = 'System Critical';
            }
            
            // Update performance metrics
            const perf = data.performance_metrics || {};
            updateMetricBar('cpuUsage', 'cpuBar', perf.cpu_usage || 0);
            updateMetricBar('memoryUsage', 'memoryBar', perf.memory_usage || 0);
            updateMetricBar('diskUsage', 'diskBar', perf.disk_usage || 0);
            
            // Update trading metrics
            const trading = data.trading_metrics || {};
            document.getElementById('portfolioValue').textContent = 
                '$' + (trading.portfolio_value || 0).toLocaleString('en-US', {minimumFractionDigits: 2});
            document.getElementById('totalPnl').textContent = 
                '$' + (trading.total_pnl || 0).toLocaleString('en-US', {minimumFractionDigits: 2});
            document.getElementById('totalTrades').textContent = trading.total_trades || 0;
            
            const winRate = trading.total_trades > 0 ? 
                (trading.winning_trades / trading.total_trades * 100).toFixed(1) : 0;
            document.getElementById('winRate').textContent = winRate + '%';
            document.getElementById('openPositions').textContent = trading.positions?.length || 0;
            
            // Update error statistics
            const errors = data.error_stats || {};
            document.getElementById('totalErrors').textContent = errors.total_errors || 0;
            document.getElementById('recentErrors').textContent = errors.recent_errors_count || 0;
            document.getElementById('criticalErrors').textContent = errors.critical_errors || 0;
            document.getElementById('lastError').textContent = 
                errors.last_error ? new Date(errors.last_error).toLocaleString() : 'None';
            
            // Update API status
            const apiStatus = data.api_status || {};
            const apiContainer = document.getElementById('apiStatus');
            apiContainer.innerHTML = '';
            
            Object.entries(apiStatus).forEach(([api, status]) => {
                const badge = document.createElement('div');
                badge.className = 'api-badge';
                badge.innerHTML = `
                    <div class="api-dot ${status}"></div>
                    <span>${api.toUpperCase()}</span>
                `;
                apiContainer.appendChild(badge);
            });
            
            // Update timestamp
            document.getElementById('lastUpdate').textContent = 
                'Last updated: ' + new Date().toLocaleString();
        }
        
        function updateMetricBar(valueId, barId, percentage) {
            const valueElement = document.getElementById(valueId);
            const barElement = document.getElementById(barId);
            
            valueElement.textContent = percentage.toFixed(1) + '%';
            barElement.style.width = percentage + '%';
            
            // Update colors based on thresholds
            const className = percentage > 80 ? 'critical' : percentage > 60 ? 'warning' : '';
            valueElement.className = 'metric-value ' + className;
            barElement.className = 'progress-fill ' + className;
        }
        
        // Fallback: Fetch data via REST API if WebSocket fails
        async function fetchData() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Failed to fetch data:', error);
            }
        }
        
        // Fallback polling every 10 seconds
        setInterval(fetchData, 10000);
    </script>
</body>
</html>
"""

def start_monitoring_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the monitoring dashboard server"""
    logger.info(f"Starting monitoring dashboard on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    start_monitoring_server()