#!/bin/bash
# ============================================================
# Life Tracker - Jetson Orin Nano Setup Script
# ============================================================
# This script automates dependency installation on Jetson devices
#
# What it does:
#   1. Checks Jetson environment (JetPack, PyTorch, CUDA)
#   2. Creates virtual environment 'Camera'
#   3. Installs base dependencies (requirements_jetson.txt)
#   4. Optionally installs RTMPose (20-40 min compilation)
#   5. Downloads YOLO models
#
# Usage:
#   ./scripts/jetson_setup.sh
#   ./scripts/jetson_setup.sh --with-rtmpose    # Install RTMPose immediately
#   ./scripts/jetson_setup.sh --skip-rtmpose    # Skip RTMPose (use MediaPipe)
# ============================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo ""
    echo -e "${CYAN}==>${NC} $1"
}

# Parse command line arguments
INSTALL_RTMPOSE=""
for arg in "$@"; do
    case $arg in
        --with-rtmpose)
            INSTALL_RTMPOSE="yes"
            shift
            ;;
        --skip-rtmpose)
            INSTALL_RTMPOSE="no"
            shift
            ;;
    esac
done

# Check if running on Jetson
check_jetson() {
    print_step "Step 1: Checking Jetson Environment"

    if [ -f /etc/nv_tegra_release ]; then
        JETSON_VERSION=$(cat /etc/nv_tegra_release)
        print_success "Jetson device detected"
        print_info "$JETSON_VERSION"
    else
        print_warning "Not running on Jetson device"
        print_warning "This script is optimized for Jetson, but will continue anyway"
    fi
}

# Check PyTorch installation
check_pytorch() {
    print_info "Checking PyTorch installation..."

    if python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")

        print_success "PyTorch ${TORCH_VERSION} detected"
        print_info "CUDA Available: ${CUDA_AVAILABLE}"
        print_info "CUDA Version: ${CUDA_VERSION}"

        if [ "$CUDA_AVAILABLE" != "True" ]; then
            print_error "PyTorch is installed but CUDA is not available!"
            print_error "Please install PyTorch with CUDA support for Jetson"
            print_error "Visit: https://jetson-ai-lab.github.io/pytorch.html"
            exit 1
        fi
    else
        print_error "PyTorch not found!"
        print_error "Please install PyTorch for Jetson first:"
        print_error ""
        print_error "For JetPack 6.2.1 (current):"
        print_error "  pip install torch==2.8.0 torchvision==0.23.0 \\"
        print_error "    --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126"
        print_error ""
        print_error "Visit: https://jetson-ai-lab.github.io/pytorch.html"
        exit 1
    fi
}

# Check JetPack version
check_jetpack() {
    print_info "Checking JetPack version..."

    if [ -f /etc/nv_tegra_release ]; then
        JETPACK_R=$(cat /etc/nv_tegra_release | grep -oP 'R\d+\.\d+\.\d+' | head -1)
        print_info "JetPack Release: $JETPACK_R"
    else
        print_warning "Cannot detect JetPack version"
    fi
}

# Install system dependencies
install_system_deps() {
    print_step "Step 2: Installing System Dependencies"

    print_info "Updating package list..."
    sudo apt-get update -qq

    print_info "Installing essential build tools..."
    sudo apt-get install -y -qq \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        git \
        wget \
        curl \
        nano \
        htop

    print_success "System dependencies installed"
}

# Create virtual environment
create_venv() {
    print_step "Step 3: Creating Virtual Environment"

    if [ -d "Camera" ]; then
        print_warning "Virtual environment 'Camera' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing old virtual environment..."
            rm -rf Camera
            python3 -m venv Camera
            print_success "Virtual environment 'Camera' recreated"
        else
            print_info "Using existing virtual environment"
        fi
    else
        python3 -m venv Camera
        print_success "Virtual environment 'Camera' created"
    fi
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source Camera/bin/activate
    print_success "Virtual environment activated"
}

# Upgrade pip
upgrade_pip() {
    print_info "Upgrading pip..."
    pip install -q --upgrade pip setuptools wheel
    print_success "pip upgraded to $(pip --version | awk '{print $2}')"
}

# Install base Python dependencies
install_base_deps() {
    print_step "Step 4: Installing Base Python Dependencies"

    if [ ! -f "requirements_jetson.txt" ]; then
        print_error "requirements_jetson.txt not found!"
        exit 1
    fi

    print_info "Installing from requirements_jetson.txt..."
    print_warning "This may take 10-15 minutes..."

    pip install -r requirements_jetson.txt

    print_success "Base dependencies installed"
}

# Ask user about RTMPose installation
ask_rtmpose() {
    if [ -z "$INSTALL_RTMPOSE" ]; then
        echo ""
        print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_info "RTMPose GPU Acceleration (Optional)"
        print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "RTMPose provides GPU-accelerated pose estimation:"
        echo "  • Performance: ~12-15ms inference (vs ~80-100ms MediaPipe CPU)"
        echo "  • FPS: 20-25 FPS @ 720p (vs ~10-12 FPS MediaPipe)"
        echo "  • Compilation time: 20-40 minutes"
        echo ""
        echo "Without RTMPose, the system will use MediaPipe (CPU):"
        echo "  • Works immediately, no compilation needed"
        echo "  • Good accuracy, but slower"
        echo "  • You can install RTMPose later if needed"
        echo ""
        read -p "Do you want to install RTMPose now? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            INSTALL_RTMPOSE="yes"
        else
            INSTALL_RTMPOSE="no"
        fi
    fi
}

# Install RTMPose dependencies
install_rtmpose() {
    print_step "Step 5: Installing RTMPose Dependencies"

    if [ ! -f "requirements_rtmpose.txt" ]; then
        print_error "requirements_rtmpose.txt not found!"
        exit 1
    fi

    print_warning "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_warning "IMPORTANT: mmcv compilation takes 20-40 minutes"
    print_warning "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    print_info "This step will compile mmcv from source for Jetson"
    print_info "Progress indicators:"
    print_info "  • High CPU usage (100%) is normal"
    print_info "  • Fan will run at high speed"
    print_info "  • DO NOT interrupt this process"
    echo ""

    # Check available memory
    AVAILABLE_MEM=$(free -m | awk '/^Mem:/ {print $7}')
    print_info "Available memory: ${AVAILABLE_MEM}MB"

    if [ "$AVAILABLE_MEM" -lt 3000 ]; then
        print_warning "Low memory detected (< 3GB free)"
        print_warning "mmcv compilation may fail without swap"
        echo ""
        read -p "Enable 4GB swap file? (recommended) (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            print_info "Creating 4GB swap file..."
            sudo fallocate -l 4G /swapfile
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
            sudo swapon /swapfile
            print_success "Swap enabled: $(free -h | grep Swap)"
        fi
    fi

    print_info "Installing OpenMIM..."
    pip install openmim

    print_info "Installing mmengine..."
    mim install mmengine==0.8.0

    print_warning "Starting mmcv compilation (20-40 minutes)..."
    print_info "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    mim install mmcv==2.1.0
    print_success "mmcv compiled successfully!"

    print_info "Installing mmpose..."
    mim install mmpose==1.1.0

    print_success "RTMPose dependencies installed!"
    print_info "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
}

# Create necessary directories
create_directories() {
    print_step "Step 6: Creating Project Directories"

    mkdir -p data logs models config
    print_success "Directories created: data, logs, models, config"
}

# Download YOLO models
download_models() {
    print_step "Step 7: Downloading YOLO Models"

    cd models

    if [ ! -f "yolov8n.pt" ]; then
        print_info "Downloading YOLOv8n (6MB)..."
        wget -q --show-progress https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
        print_success "YOLOv8n downloaded"
    else
        print_info "YOLOv8n already exists"
    fi

    if [ ! -f "yolov8s.pt" ]; then
        print_info "Downloading YOLOv8s (22MB)..."
        wget -q --show-progress https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
        print_success "YOLOv8s downloaded"
    else
        print_info "YOLOv8s already exists"
    fi

    cd ..
    print_success "YOLO models ready"
}

# Set permissions for scripts
set_permissions() {
    print_info "Setting executable permissions for scripts..."
    chmod +x scripts/*.sh 2>/dev/null || true
    print_success "Script permissions set"
}

# Print final summary
print_summary() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                                                            ║"
    echo "║            ✅  Installation Complete!                     ║"
    echo "║                                                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    print_info "Environment Summary:"
    echo "  • Virtual environment: Camera"
    echo "  • Python: $(python --version 2>&1 | awk '{print $2}')"
    echo "  • PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    echo "  • CUDA: $(python -c 'import torch; print(torch.version.cuda)')"

    if [ "$INSTALL_RTMPOSE" = "yes" ]; then
        echo "  • Pose estimator: RTMPose (GPU)"
        echo "  • mmcv: $(python -c 'import mmcv; print(mmcv.__version__)')"
        echo "  • mmpose: $(python -c 'import mmpose; print(mmpose.__version__)')"
    else
        echo "  • Pose estimator: MediaPipe (CPU)"
    fi

    echo ""
    print_success "Next Steps:"
    echo ""
    echo "1. Activate virtual environment:"
    echo "   ${CYAN}source Camera/bin/activate${NC}"
    echo ""
    echo "2. Run Life Tracker:"
    if [ "$INSTALL_RTMPOSE" = "yes" ]; then
        echo "   ${CYAN}./scripts/jetson_run.sh balanced${NC}      # Recommended (720p, 20-25 FPS)"
        echo "   ${CYAN}./scripts/jetson_run.sh lite${NC}          # Power saving (480p, 25-30 FPS)"
        echo "   ${CYAN}./scripts/jetson_run.sh performance${NC}   # High quality (1080p, 15-20 FPS)"
    else
        echo "   ${CYAN}python main.py --config config/config_cpu.yaml${NC}"
    fi
    echo ""
    echo "3. Access web dashboard:"
    echo "   Open ${CYAN}http://localhost:5000${NC} in your browser"
    echo ""

    if [ "$INSTALL_RTMPOSE" = "no" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_info "To install RTMPose later for GPU acceleration:"
        echo "  ${CYAN}source Camera/bin/activate${NC}"
        echo "  ${CYAN}pip install openmim${NC}"
        echo "  ${CYAN}mim install mmengine==0.8.0${NC}"
        echo "  ${CYAN}mim install mmcv==2.1.0${NC}      # Takes 20-40 minutes"
        echo "  ${CYAN}mim install mmpose==1.1.0${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi

    echo ""
}

# Main installation flow
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                                                            ║"
    echo "║       Life Tracker - Jetson Setup Script v2.0.0           ║"
    echo "║                                                            ║"
    echo "║       Platform: Jetson Orin Nano Super (8GB)              ║"
    echo "║       Date: 2025-11-28                                     ║"
    echo "║                                                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # Check environment
    check_jetson
    check_pytorch
    check_jetpack

    # Install system dependencies
    install_system_deps

    # Setup Python environment
    create_venv
    activate_venv
    upgrade_pip

    # Install Python dependencies
    install_base_deps

    # Ask about RTMPose
    ask_rtmpose

    # Install RTMPose if requested
    if [ "$INSTALL_RTMPOSE" = "yes" ]; then
        install_rtmpose
    else
        print_info "Skipping RTMPose installation (using MediaPipe)"
    fi

    # Setup project structure
    create_directories
    download_models
    set_permissions

    # Print summary
    print_summary
}

# Run main installation
main
