#!/bin/bash
# Life Tracker - PC GPU Mode Run Script
# Runs the system with GPU acceleration (requires NVIDIA GPU + CUDA)
# Usage: ./scripts/pc_run_gpu.sh [--async] [--no-vis]

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Default settings
USE_ASYNC=false
NO_VIS=false
CONFIG="config/config_gpu.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --async)
            USE_ASYNC=true
            shift
            ;;
        --no-vis)
            NO_VIS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --async      Use asynchronous pipeline (4-thread, higher FPS)"
            echo "  --no-vis     Run without visualization window"
            echo "  -h, --help   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Default: synchronous mode with visualization"
            echo "  $0 --async            # Asynchronous mode (faster)"
            echo "  $0 --async --no-vis   # Async mode, no display (server mode)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Print banner
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Life Tracker - PC GPU Mode${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# Check if virtual environment exists, create if not
if [ ! -d "Camera" ]; then
    echo -e "${YELLOW}[SETUP]${NC} Virtual environment 'Camera' not found, creating..."
    python3 -m venv Camera
    source Camera/bin/activate
    echo -e "${BLUE}[SETUP]${NC} Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}[SETUP]${NC} Installation complete!"
    echo ""
else
    echo -e "${BLUE}[INFO]${NC} Activating virtual environment..."
    source Camera/bin/activate
fi

# Check GPU availability
echo -e "${BLUE}[INFO]${NC} Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo -e "${RED}[ERROR]${NC} nvidia-smi not found!"
    echo -e "${RED}[ERROR]${NC} Please ensure NVIDIA drivers are installed"
    exit 1
fi

# Check CUDA availability
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'CUDA Version: {torch.version.cuda}')"

# Select script based on mode
if [ "$USE_ASYNC" = true ]; then
    SCRIPT="main_async.py"
    MODE_NAME="Asynchronous (4-thread pipeline)"
else
    SCRIPT="main.py"
    MODE_NAME="Synchronous"
fi

# Build command
CMD="python3 $SCRIPT --config $CONFIG"

if [ "$NO_VIS" = true ]; then
    CMD="$CMD --no-vis"
fi

# Print configuration
echo -e "${BLUE}[INFO]${NC} Configuration:"
echo -e "  Mode:          ${MODE_NAME}"
echo -e "  Config:        ${CONFIG}"
echo -e "  Visualization: $([ "$NO_VIS" = true ] && echo "Disabled" || echo "Enabled")"
echo ""

# Run
echo -e "${GREEN}[START]${NC} Starting Life Tracker..."
echo -e "${YELLOW}[TIP]${NC} Press 'q' to quit"
echo ""

$CMD

# Exit message
echo ""
echo -e "${GREEN}[EXIT]${NC} Life Tracker stopped"
echo ""
