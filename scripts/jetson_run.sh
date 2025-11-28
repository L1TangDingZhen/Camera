#!/bin/bash
# Life Tracker - Jetson Convenience Run Script
# Easily start the system with different configuration modes
# Usage: ./run_jetson.sh [lite|balanced|performance] [options]

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Default configuration
MODE="balanced"
VENV_PATH="Camera"
DAEMON=false
WITH_DASHBOARD=false

# Print usage
usage() {
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  lite          - Power saving mode (7W, 45+ FPS, YOLOv8n + 720p)"
    echo "  balanced      - Balanced mode (15W, 30-35 FPS, YOLOv8s + 720p) [DEFAULT]"
    echo "  performance   - High performance (25W, 20-25 FPS, YOLOv8m + 1080p)"
    echo ""
    echo "Options:"
    echo "  -d, --daemon       Run in background (nohup)"
    echo "  -w, --with-dashboard  Also start web dashboard"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 balanced                  # Run in balanced mode (foreground)"
    echo "  $0 lite -d                   # Run in lite mode (background)"
    echo "  $0 performance -w            # Run in performance mode with dashboard"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        lite|balanced|performance)
            MODE="$1"
            shift
            ;;
        -d|--daemon)
            DAEMON=true
            shift
            ;;
        -w|--with-dashboard)
            WITH_DASHBOARD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Select configuration file
case $MODE in
    lite)
        CONFIG="config/config_jetson_lite.yaml"
        MODE_DESC="Lite (7W, 45+ FPS)"
        ;;
    balanced)
        CONFIG="config/config_jetson_balanced.yaml"
        MODE_DESC="Balanced (15W, 30-35 FPS) ⭐ Recommended"
        ;;
    performance)
        CONFIG="config/config_jetson_performance.yaml"
        MODE_DESC="Performance (25W, 20-25 FPS)"
        ;;
    *)
        echo -e "${RED}Invalid mode: $MODE${NC}"
        usage
        exit 1
        ;;
esac

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Configuration file not found: $CONFIG${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Virtual environment not found: $VENV_PATH${NC}"
    echo -e "${YELLOW}Please run setup_jetson.sh first${NC}"
    exit 1
fi

# Print banner
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║           Life Tracker - Jetson Launcher              ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo -e "${BLUE}Mode:${NC} $MODE_DESC"
echo -e "${BLUE}Config:${NC} $CONFIG"
echo -e "${BLUE}Virtual Environment:${NC} $VENV_PATH"
echo -e "${BLUE}Daemon:${NC} $DAEMON"
echo -e "${BLUE}Dashboard:${NC} $WITH_DASHBOARD"
echo ""

# Activate virtual environment
echo -e "${GREEN}[1/3]${NC} Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo -e "${RED}main.py not found in current directory${NC}"
    exit 1
fi

# Start web dashboard if requested
if [ "$WITH_DASHBOARD" = true ]; then
    echo -e "${GREEN}[2/3]${NC} Starting web dashboard..."

    if [ "$DAEMON" = true ]; then
        nohup python web_dashboard.py > logs/dashboard.log 2>&1 &
        DASHBOARD_PID=$!
        echo -e "${BLUE}Dashboard PID:${NC} $DASHBOARD_PID"
        echo $DASHBOARD_PID > /tmp/life_tracker_dashboard.pid
    else
        # Start dashboard in background
        python web_dashboard.py > logs/dashboard.log 2>&1 &
        DASHBOARD_PID=$!
        echo -e "${BLUE}Dashboard PID:${NC} $DASHBOARD_PID"
        echo $DASHBOARD_PID > /tmp/life_tracker_dashboard.pid
    fi

    echo -e "${GREEN}Web dashboard started at http://localhost:5000${NC}"
    sleep 2
fi

# Start main application
echo -e "${GREEN}[3/3]${NC} Starting Life Tracker..."
echo ""

if [ "$DAEMON" = true ]; then
    # Run in background
    nohup python main.py --config "$CONFIG" > logs/app.log 2>&1 &
    MAIN_PID=$!
    echo $MAIN_PID > /tmp/life_tracker_main.pid

    echo -e "${GREEN}✅ Life Tracker started in background${NC}"
    echo -e "${BLUE}Main PID:${NC} $MAIN_PID"
    echo ""
    echo "Logs:"
    echo "  Main app: tail -f logs/app.log"
    if [ "$WITH_DASHBOARD" = true ]; then
        echo "  Dashboard: tail -f logs/dashboard.log"
    fi
    echo ""
    echo "Stop:"
    echo "  kill $MAIN_PID"
    if [ "$WITH_DASHBOARD" = true ]; then
        echo "  kill $DASHBOARD_PID"
    fi
    echo ""
    echo "Or use: ./stop_jetson.sh"
else
    # Run in foreground
    echo -e "${GREEN}✅ Starting Life Tracker (press Ctrl+C to stop)${NC}"
    echo ""

    # Trap Ctrl+C to cleanup
    trap cleanup SIGINT SIGTERM

    cleanup() {
        echo ""
        echo -e "${YELLOW}Stopping Life Tracker...${NC}"
        if [ "$WITH_DASHBOARD" = true ] && [ -f /tmp/life_tracker_dashboard.pid ]; then
            DASHBOARD_PID=$(cat /tmp/life_tracker_dashboard.pid)
            kill $DASHBOARD_PID 2>/dev/null || true
            rm /tmp/life_tracker_dashboard.pid
            echo -e "${GREEN}Dashboard stopped${NC}"
        fi
        echo -e "${GREEN}Life Tracker stopped${NC}"
        exit 0
    }

    # Run main application
    python main.py --config "$CONFIG"
fi
