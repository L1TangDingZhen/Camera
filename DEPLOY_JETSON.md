# Life Tracker - Jetson Orin Nano Deployment Guide

> Complete guide for deploying Life Tracker on NVIDIA Jetson Orin Nano Super

---

## 📋 Prerequisites

### Hardware Requirements

- **NVIDIA Jetson Orin Nano Super** (8GB) or Jetson Orin Nano
- **64GB+ MicroSD Card** (128GB recommended)
- **USB Camera** or **CSI Camera**
- **Power Supply**: 15W adapter (or higher, 25W for performance mode)
- **Network**: Ethernet or WiFi for initial setup

### Software Requirements

**Current Tested Environment (2025-11-28)**:
- **JetPack 6.2.1** (R36.4.0)
- **Kernel**: 5.15.148-tegra
- **Ubuntu**: 22.04 LTS (aarch64)
- **Python**: 3.10.12
- **PyTorch**: 2.8.0 (from Jetson AI Lab PyPI)
- **CUDA**: 12.6
- **cuDNN**: Included in JetPack
- **TensorRT**: Included in JetPack

**Prerequisites**:
- JetPack 6.2.1+ must be installed
- PyTorch 2.8.0+ with CUDA support must be installed **before** running setup

---

## 🚀 Deployment Methods

We provide **three deployment methods**:

1. **Manual Installation** - Step-by-step installation (recommended for understanding the process)
2. **Automated Setup Script** - One-command installation (recommended for quick start)
3. **Docker Deployment** - Containerized deployment (recommended for production)

---

## 📦 Method 1: Manual Installation

### Understanding the Two Requirements Files

This project uses **two separate requirements files** for Jetson:

1. **`requirements_jetson.txt`** - Base dependencies (10-15 min)
   - YOLOv8, MediaPipe, Flask, pandas, scikit-learn, etc.
   - System works immediately with MediaPipe CPU pose estimation

2. **`requirements_rtmpose.txt`** - RTMPose GPU acceleration (20-40 min)
   - mmcv 2.1.0, mmengine 0.8.0, mmpose 1.1.0
   - Compiles from source on Jetson
   - Optional but **recommended for production**
   - Provides **2-3x faster** pose estimation vs MediaPipe

**Why separate?** This allows you to get started quickly with MediaPipe, then optionally add GPU acceleration later.

---

### Step 1: Flash JetPack 6.2.1

1. Download **NVIDIA SDK Manager** on your Ubuntu PC:
   ```bash
   https://developer.nvidia.com/sdk-manager
   ```

2. Flash JetPack 6.2.1+ to Jetson:
   - Connect Jetson to PC via USB-C
   - Put Jetson in recovery mode
   - Follow SDK Manager instructions
   - Choose JetPack 6.2.1 or later

3. Verify installation:
   ```bash
   # Check JetPack version
   cat /etc/nv_tegra_release
   # Should show: R36 (release), REVISION: 4.0

   # Monitor system resources
   jtop
   ```

### Step 2: Install PyTorch (Prerequisite)

**IMPORTANT**: PyTorch must be installed **before** running the setup script.

```bash
# Check if PyTorch is already installed
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# If not installed or CUDA is False, install PyTorch 2.8.0 for JetPack 6.2.1:
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

# Verify CUDA is available (should show True)
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
```

**Reference**: https://jetson-ai-lab.github.io/pytorch.html

### Step 3: Clone Repository

```bash
# Clone the repository
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera
```

### Step 4: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv Camera

# Activate virtual environment
source Camera/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 5: Install Base Dependencies

```bash
# Install base dependencies (10-15 minutes)
pip install -r requirements_jetson.txt

# This installs:
# - YOLOv8 (ultralytics)
# - MediaPipe (CPU pose estimation)
# - Flask (web dashboard)
# - pandas, numpy, opencv-python
# - All other base dependencies
```

**Your system is now ready to run with MediaPipe!**

### Step 6: (Optional) Install RTMPose for GPU Acceleration

If you want GPU-accelerated pose estimation (recommended):

**Option A: Using the requirements file reference**

Check `requirements_rtmpose.txt` for detailed instructions, then run:

```bash
# Install OpenMIM (OpenMMLab package manager)
pip install openmim

# Install RTMPose dependencies (20-40 minutes, compiles from source)
mim install mmengine==0.8.0
mim install mmcv==2.1.0      # Takes 20-40 min, high CPU usage is normal
mim install mmpose==1.1.0

# Verify installation
python3 -c "import mmcv; print(f'mmcv: {mmcv.__version__}')"
python3 -c "import mmengine; print(f'mmengine: {mmengine.__version__}')"
python3 -c "import mmpose; print(f'mmpose: {mmpose.__version__}')"
```

**Option B: Let the setup script handle it (see Method 2)**

**Note**: mmcv compilation takes 20-40 minutes. High CPU usage is normal. Do NOT interrupt the process.

If you encounter memory errors during compilation, enable swap:
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h  # Verify swap is enabled
```

### Step 7: Download Models

```bash
# Create models directory
mkdir -p models

# Download YOLO models
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
cd ..

# (Optional) Download RTMPose models if you installed RTMPose
mim download mmpose --config rtmpose-s_8xb256-420e_coco-256x192 --dest models/rtmpose/
```

### Step 8: Configure Camera

Check available cameras:
```bash
# List video devices
ls /dev/video*

# Test camera
v4l2-ctl --list-devices
```

Update camera configuration in `config/config_jetson_balanced.yaml`:
```yaml
camera:
  source: 0  # Change to your camera ID (/dev/video0)
```

For CSI camera, use GStreamer pipeline (see Advanced Configuration section).

### Step 9: Run the System

```bash
# Activate virtual environment (if not already activated)
source Camera/bin/activate

# Run with balanced mode (recommended)
./scripts/jetson_run.sh balanced

# Or run other modes
./scripts/jetson_run.sh lite          # Power saving (640x480, 25-30 FPS)
./scripts/jetson_run.sh performance   # High quality (1920x1080, 15-20 FPS)
```

### Step 10: Access Web Dashboard

Open browser and navigate to:
```
http://<jetson-ip>:5000
```

Find Jetson IP:
```bash
hostname -I
```

### Step 11: Stop the System

```bash
# If running in foreground: Press Ctrl+C

# If running in background:
./scripts/jetson_stop.sh
```

---

## 🤖 Method 2: Automated Setup Script (Recommended)

The automated setup script handles everything for you.

### Step 1: Install PyTorch (Prerequisite)

```bash
# Install PyTorch 2.8.0 for JetPack 6.2.1
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

# Verify
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Step 2: Clone Repository

```bash
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera
```

### Step 3: Run Setup Script

**Option A: Interactive mode (asks about RTMPose)**
```bash
./scripts/jetson_setup.sh
```

**Option B: Auto-install RTMPose**
```bash
./scripts/jetson_setup.sh --with-rtmpose
```

**Option C: Skip RTMPose (MediaPipe only)**
```bash
./scripts/jetson_setup.sh --skip-rtmpose
```

**What the script does**:
- ✅ Checks Jetson environment (JetPack, PyTorch, CUDA)
- ✅ Creates virtual environment named **Camera**
- ✅ Installs base dependencies (requirements_jetson.txt)
- ✅ Optionally installs RTMPose (20-40 min compilation)
- ✅ Downloads YOLO models
- ✅ Creates necessary directories
- ✅ Sets permissions for shell scripts

**Estimated time**:
- Without RTMPose: 10-15 minutes
- With RTMPose: 30-50 minutes

### Step 4: Run the System

```bash
# Activate virtual environment
source Camera/bin/activate

# Run with balanced mode (recommended)
./scripts/jetson_run.sh balanced
```

---

## 🐳 Method 3: Docker Deployment

### Prerequisites

Install NVIDIA Container Runtime:

```bash
# Install Docker (if not already installed)
sudo apt-get update
sudo apt-get install -y docker.io

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify installation
docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r36.4.0 nvidia-smi
```

### Step 1: Clone Repository

```bash
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera
```

### Step 2: Build Docker Image

```bash
# Build image (this takes 30-50 minutes due to mmcv compilation)
docker build -f Dockerfile.jetson -t life-tracker:jetson .
```

**The Dockerfile includes**:
- Base image: `nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.8-py3`
- PyTorch 2.8.0 and TorchVision 0.23.0 (pre-installed)
- RTMPose dependencies (mmcv 2.1.0, mmengine 0.8.0, mmpose 1.1.0)
- All application dependencies
- YOLO and RTMPose models

### Step 3: Run with Docker Compose

Choose your performance mode:

#### Balanced Mode (Recommended, 720p, 20-25 FPS, ~15W)
```bash
docker-compose -f docker-compose.jetson.yml --profile balanced up -d
```

#### Lite Mode (Power Saving, 480p, 25-30 FPS, ~10W)
```bash
docker-compose -f docker-compose.jetson.yml --profile lite up -d
```

#### Performance Mode (High Quality, 1080p, 15-20 FPS, ~20-25W)
```bash
docker-compose -f docker-compose.jetson.yml --profile performance up -d
```

### Step 4: Check Status

```bash
# View logs
docker-compose -f docker-compose.jetson.yml logs -f

# Check running containers
docker ps

# Monitor resources
jtop
```

### Step 5: Stop Containers

```bash
docker-compose -f docker-compose.jetson.yml down
```

---

## ⚙️ Performance Modes

| Mode | Resolution | FPS | Power | Model | Use Case |
|------|-----------|-----|-------|-------|----------|
| **Lite** | 640x480 | 25-30 | ~10W | YOLOv8n | Power saving, battery |
| **Balanced** | 1280x720 | 20-25 | ~15W | YOLOv8s | **Recommended**, 24/7 monitoring |
| **Performance** | 1920x1080 | 15-20 | ~20-25W | YOLOv8m | Maximum quality |

**Configuration files**:
- Lite: `config/config_jetson_lite.yaml`
- Balanced: `config/config_jetson_balanced.yaml`
- Performance: `config/config_jetson_performance.yaml`

---

## 🔧 Advanced Configuration

### Adjust Power Mode

```bash
# Check current power mode
sudo nvpmodel -q

# Set power mode
sudo nvpmodel -m 0  # Maximum performance (25W)
sudo nvpmodel -m 1  # Balanced (15W)
sudo nvpmodel -m 2  # Power saving (10W)

# Enable maximum clocks
sudo jetson_clocks
```

### Switch Between MediaPipe and RTMPose

Edit your config file (e.g., `config/config_jetson_balanced.yaml`):

**MediaPipe (CPU, no RTMPose installation needed)**:
```yaml
models:
  pose:
    backend: mediapipe
    complexity: 1
    device: cpu
```

**RTMPose (GPU, requires RTMPose installation)**:
```yaml
models:
  pose:
    backend: rtmpose
    model: rtmpose-s
    device: cuda:0
```

### Camera Configuration

**For USB camera (default)**:
```yaml
camera:
  source: 0  # /dev/video0
```

**For CSI camera**:
```yaml
camera:
  source: "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
```

**For network camera (RTSP)**:
```yaml
camera:
  source: "rtsp://username:password@192.168.1.100:554/stream"
```

---

## 📊 Performance Monitoring

### Using jtop

```bash
# Install jetson-stats (if not already installed)
sudo apt install python3-jetson-stats

# Run jtop (interactive monitor)
jtop

# jtop shows:
# - GPU/CPU usage
# - Memory usage
# - Power consumption
# - Temperature
# - Clock frequencies
```

### Log Monitoring

```bash
# Application logs
tail -f logs/app.log

# Performance metrics
tail -f logs/performance.log

# Docker logs (if using Docker)
docker logs -f life-tracker-balanced
```

### Performance Metrics

**Expected performance with RTMPose (Balanced mode, 720p)**:
- **Overall FPS**: 20-25 FPS
- **Person Detection**: ~8-10ms (YOLOv8s)
- **Pose Estimation**: ~12-15ms (RTMPose-s)
- **State Machine**: ~1-2ms
- **Total Latency**: ~40-50ms
- **Power**: ~15W

**Expected performance with MediaPipe (Balanced mode, 720p)**:
- **Overall FPS**: 10-12 FPS
- **Person Detection**: ~8-10ms (YOLOv8s)
- **Pose Estimation**: ~80-100ms (MediaPipe CPU)
- **State Machine**: ~1-2ms
- **Total Latency**: ~90-110ms
- **Power**: ~12W

---

## 🐛 Troubleshooting

### Issue: PyTorch not found or CUDA not available

```bash
# Check PyTorch installation
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

# If False or error, install PyTorch for JetPack 6.2.1:
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

### Issue: Camera Not Detected

```bash
# Check camera devices
ls -l /dev/video*

# Test USB camera
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Test CSI camera
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink

# Check camera permissions
sudo usermod -aG video $USER
# Log out and log back in
```

### Issue: mmcv Compilation Fails

**Symptoms**: Build hangs or fails during `mim install mmcv==2.1.0`

**Solutions**:

1. **Enable swap** (most common fix):
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   free -h  # Verify swap is active
   ```

2. **Close other applications** to free memory

3. **Don't interrupt the process** - compilation takes 20-40 minutes

4. **Monitor progress**:
   ```bash
   watch -n 1 'ps aux | grep python'
   ```

### Issue: CUDA Out of Memory

```bash
# Use lite mode (lower resolution)
./scripts/jetson_run.sh lite

# Or reduce resolution in config
# Edit config/config_jetson_balanced.yaml:
camera:
  resolution: [640, 480]  # Lower resolution
```

### Issue: Low FPS

```bash
# Check power mode
sudo nvpmodel -q

# Set to maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor with jtop
jtop
```

### Issue: Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Restart docker service
sudo systemctl restart docker
```

### Issue: Import Error for mmcv/mmpose

```bash
# Version mismatch, uninstall and reinstall
pip uninstall mmcv mmengine mmpose -y

# Reinstall with correct versions
pip install openmim
mim install mmengine==0.8.0
mim install mmcv==2.1.0
mim install mmpose==1.1.0
```

---

## 📝 Version History

### Current Environment (2025-11-28)

**Hardware**:
- Device: Jetson Orin Nano Super (8GB)
- JetPack: 6.2.1 (R36.4.0)
- Kernel: 5.15.148-tegra

**Software**:
- Python: 3.10.12
- PyTorch: 2.8.0
- CUDA: 12.6
- cuDNN: Included in JetPack
- TensorRT: Included in JetPack

**Dependencies**:
- mmcv: 2.1.0 (upgraded from 2.0.0)
- mmengine: 0.8.0
- mmpose: 1.1.0 (upgraded from 1.0.0)
- ultralytics: 8.3.229
- mediapipe: 0.10.18

**Performance**:
- RTMPose (Balanced): 20-25 FPS @ 720p, ~15W
- MediaPipe (Balanced): 10-12 FPS @ 720p, ~12W

---

## 🔗 Additional Resources

- **Main Documentation**: `README.md`
- **Docker Guide**: `DOCKER.md`
- **RTMPose Installation**: `INSTALL_RTMPOSE.md`
- **User Guide**: `USER_GUIDE.md`
- **Technical Guide**: `DL_RL_TECHNICAL_GUIDE.md`

**External Resources**:
- PyTorch for Jetson: https://jetson-ai-lab.github.io/pytorch.html
- JetPack Documentation: https://developer.nvidia.com/embedded/jetpack
- Jetson Forums: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/

---

## 🆘 Support

- **GitHub Issues**: https://github.com/L1TangDingZhen/Camera/issues
- **Jetson Forums**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/

---

**Document Version**: 2.0.0
**Last Updated**: 2025-11-28
**Target Platform**: Jetson Orin Nano Super (8GB), JetPack 6.2.1
