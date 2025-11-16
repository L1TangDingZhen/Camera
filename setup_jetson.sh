#!/bin/bash
# Life Tracker - Jetson Orin Nano Super Setup Script
# This script automates the installation process on Jetson devices
# Virtual environment name: Camera

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
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

# Check if running on Jetson
check_jetson() {
    print_info "Checking if running on Jetson device..."
    if [ -f /etc/nv_tegra_release ]; then
        JETSON_VERSION=$(cat /etc/nv_tegra_release)
        print_success "Jetson device detected: $JETSON_VERSION"
    else
        print_warning "Not running on Jetson device. Continuing anyway..."
    fi
}

# Check JetPack version
check_jetpack() {
    print_info "Checking JetPack version..."
    if command -v jetson_release &> /dev/null; then
        jetson_release
    else
        print_warning "jetson_release not found. Install with: sudo apt install python3-jetson-stats"
    fi
}

# Install system dependencies
install_system_deps() {
    print_info "Installing system dependencies..."

    sudo apt-get update

    # Essential build tools
    sudo apt-get install -y \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        cmake \
        git \
        wget \
        curl

    # OpenCV dependencies
    sudo apt-get install -y \
        libopencv-dev \
        python3-opencv \
        libopenblas-dev \
        libjpeg-dev \
        zlib1g-dev \
        libpng-dev \
        libtiff-dev

    # Additional tools
    sudo apt-get install -y \
        htop \
        jtop \
        nano \
        tmux

    print_success "System dependencies installed"
}

# Create virtual environment
create_venv() {
    print_info "Creating virtual environment 'Camera'..."

    if [ -d "Camera" ]; then
        print_warning "Virtual environment 'Camera' already exists. Skipping creation."
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
    pip install --upgrade pip setuptools wheel
    print_success "pip upgraded"
}

# Install PyTorch for Jetson
install_pytorch() {
    print_info "Installing PyTorch for Jetson..."

    # Check if PyTorch is already installed
    if python -c "import torch" &> /dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        print_warning "PyTorch already installed: $TORCH_VERSION"
        return
    fi

    print_info "Downloading PyTorch wheel for Jetson..."
    print_warning "This may take several minutes..."

    # PyTorch 2.0 for JetPack 5.x
    # Note: Update URL for your specific JetPack version
    # Download from: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

    TORCH_WHEEL="torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
    TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v511/${TORCH_WHEEL}"

    if [ ! -f "$TORCH_WHEEL" ]; then
        wget "$TORCH_URL" || {
            print_error "Failed to download PyTorch. Please download manually from:"
            print_error "https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
            exit 1
        }
    fi

    pip install "$TORCH_WHEEL"
    print_success "PyTorch installed"
}

# Install torchvision for Jetson
install_torchvision() {
    print_info "Installing torchvision for Jetson..."

    # Install dependencies
    sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

    # Install torchvision from source (recommended for Jetson)
    print_info "Building torchvision from source..."

    git clone --branch v0.15.0 https://github.com/pytorch/vision torchvision_source
    cd torchvision_source

    export BUILD_VERSION=0.15.0
    python setup.py install

    cd ..
    rm -rf torchvision_source

    print_success "torchvision installed"
}

# Install Python dependencies
install_python_deps() {
    print_info "Installing Python dependencies..."

    # Check if requirements_jetson.txt exists
    if [ -f "requirements_jetson.txt" ]; then
        print_info "Using requirements_jetson.txt..."
        pip install -r requirements_jetson.txt
    else
        print_warning "requirements_jetson.txt not found, using requirements.txt..."
        pip install -r requirements.txt
    fi

    print_success "Python dependencies installed"
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."

    mkdir -p data
    mkdir -p logs
    mkdir -p models
    mkdir -p config

    print_success "Directories created"
}

# Download models
download_models() {
    print_info "Downloading YOLOv8 models..."

    cd models

    # Download YOLOv8 models
    if [ ! -f "yolov8s.pt" ]; then
        wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
        print_success "YOLOv8s downloaded"
    else
        print_warning "yolov8s.pt already exists"
    fi

    if [ ! -f "yolov8n.pt" ]; then
        wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
        print_success "YOLOv8n downloaded"
    else
        print_warning "yolov8n.pt already exists"
    fi

    if [ ! -f "yolov8m.pt" ]; then
        wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
        print_success "YOLOv8m downloaded"
    else
        print_warning "yolov8m.pt already exists"
    fi

    cd ..

    print_success "Models downloaded"
}

# Set permissions
set_permissions() {
    print_info "Setting permissions..."

    chmod +x run_jetson.sh || true
    chmod +x setup_jetson.sh || true

    print_success "Permissions set"
}

# Print summary
print_summary() {
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║                                                        ║"
    echo "║  ✅  Life Tracker Setup Complete!                     ║"
    echo "║                                                        ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""
    print_info "Virtual environment: Camera"
    print_info "Configuration files: config/config_jetson_*.yaml"
    echo ""
    print_success "Next steps:"
    echo "  1. Activate virtual environment:"
    echo "     $ source Camera/bin/activate"
    echo ""
    echo "  2. Run the system (choose one):"
    echo "     $ ./run_jetson.sh balanced      # Recommended (15W, 30-35 FPS)"
    echo "     $ ./run_jetson.sh lite          # Power saving (7W, 45+ FPS)"
    echo "     $ ./run_jetson.sh performance   # High performance (25W, 20-25 FPS)"
    echo ""
    echo "  3. Start web dashboard (in another terminal):"
    echo "     $ source Camera/bin/activate"
    echo "     $ python web_dashboard.py"
    echo "     Open http://localhost:5000"
    echo ""
    print_info "For more information, see DEPLOY_JETSON.md"
    echo ""
}

# Main installation flow
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║                                                        ║"
    echo "║      Life Tracker - Jetson Setup Script v1.0.1        ║"
    echo "║                                                        ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""

    # Check environment
    check_jetson
    check_jetpack

    # Install dependencies
    install_system_deps

    # Setup Python environment
    create_venv
    activate_venv
    upgrade_pip

    # Install frameworks
    install_pytorch
    install_torchvision
    install_python_deps

    # Setup project
    create_directories
    download_models
    set_permissions

    # Done
    print_summary
}

# Run main installation
main
