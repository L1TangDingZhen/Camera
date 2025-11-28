#!/bin/bash
# Life Tracker - Stop Script for Jetson
# Stops all running Life Tracker processes

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "Stopping Life Tracker..."
echo ""

# Stop main application
if [ -f /tmp/life_tracker_main.pid ]; then
    MAIN_PID=$(cat /tmp/life_tracker_main.pid)
    if kill -0 $MAIN_PID 2>/dev/null; then
        kill $MAIN_PID
        echo -e "${GREEN}✅ Main application stopped (PID: $MAIN_PID)${NC}"
    else
        echo -e "${YELLOW}⚠️  Main application not running${NC}"
    fi
    rm /tmp/life_tracker_main.pid
else
    echo -e "${YELLOW}⚠️  No PID file found for main application${NC}"
fi

# Stop dashboard
if [ -f /tmp/life_tracker_dashboard.pid ]; then
    DASHBOARD_PID=$(cat /tmp/life_tracker_dashboard.pid)
    if kill -0 $DASHBOARD_PID 2>/dev/null; then
        kill $DASHBOARD_PID
        echo -e "${GREEN}✅ Dashboard stopped (PID: $DASHBOARD_PID)${NC}"
    else
        echo -e "${YELLOW}⚠️  Dashboard not running${NC}"
    fi
    rm /tmp/life_tracker_dashboard.pid
else
    echo -e "${YELLOW}⚠️  No PID file found for dashboard${NC}"
fi

echo ""
echo -e "${GREEN}Done.${NC}"
echo ""
