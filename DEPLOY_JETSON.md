# Life Tracker - Jetson Orin Nano Deployment Guide

> Complete guide for deploying Life Tracker on NVIDIA Jetson Orin Nano Super

---

## 📋 Prerequisites

### Hardware Requirements

- **NVIDIA Jetson Orin Nano Super** (or Jetson Orin Nano)
- **8GB RAM** (minimum)
- **64GB+ MicroSD Card** (128GB recommended)
- **USB Camera** or **CSI Camera**
- **Power Supply**: 15W adapter (or higher)
- **Network**: Ethernet or WiFi for initial setup

### Software Requirements

- **JetPack 5.1.2** or later (Ubuntu 20.04 base)
- **CUDA 11.4+**
- **cuDNN 8.6+**
- **TensorRT 8.5+**
- **Python 3.8+**

---

## 🚀 Deployment Methods

We provide **two deployment methods**:

1. **Manual Installation** - Direct installation on Jetson (recommended for development)
2. **Docker Deployment** - Containerized deployment (recommended for production)

---

## 📦 Method 1: Manual Installation

### Step 1: Flash JetPack

1. Download **NVIDIA SDK Manager** on your Ubuntu PC:
   ```bash
   https://developer.nvidia.com/sdk-manager
   ```

2. Flash JetPack 5.1.2+ to Jetson:
   - Connect Jetson to PC via USB-C
   - Put Jetson in recovery mode
   - Follow SDK Manager instructions

3. Verify installation:
   ```bash
   jetson_release  # Check JetPack version
   jtop            # Monitor system resources
   ```

### Step 2: Clone Repository

```bash
# Clone the repository
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera

# Checkout specific version (optional)
git checkout v1.0.1-jetson-deploy
```

### Step 3: Run Setup Script

```bash
# Make script executable
chmod +x setup_jetson.sh

# Run automated setup
./setup_jetson.sh
```

**What the script does**:
- ✅ Installs system dependencies (apt packages)
- ✅ Creates virtual environment named **Camera**
- ✅ Installs PyTorch for Jetson (prebuilt wheel)
- ✅ Installs torchvision from source
- ✅ Installs Python dependencies
- ✅ Downloads YOLOv8 models (n/s/m)
- ✅ Creates necessary directories

**Estimated time**: 30-60 minutes (depending on network speed)

### Step 4: Configure Camera

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
  source: 0  # Change to your camera ID
```

### Step 5: Run the System

```bash
# Activate virtual environment
source Camera/bin/activate

# Run with balanced mode (recommended)
./run_jetson.sh balanced

# Or run other modes
./run_jetson.sh lite          # Power saving
./run_jetson.sh performance   # High accuracy
```

**Run options**:
```bash
# Run in background (daemon)
./run_jetson.sh balanced -d

# Run with web dashboard
./run_jetson.sh balanced -w

# Both
./run_jetson.sh balanced -d -w
```

### Step 6: Access Web Dashboard

Open browser and navigate to:
```
http://<jetson-ip>:5000
```

Find Jetson IP:
```bash
hostname -I
```

### Step 7: Stop the System

```bash
# If running in foreground: Press Ctrl+C

# If running in background:
./stop_jetson.sh
```

---

## 🐳 Method 2: Docker Deployment

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
docker run --rm --runtime=nvidia nvcr.io/nvidia/l4t-base:r35.2.1 nvidia-smi
```

### Step 1: Clone Repository

```bash
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera
git checkout v1.0.1-jetson-deploy
```

### Step 2: Build Docker Image

```bash
# Build image (this takes 20-40 minutes)
docker build -f Dockerfile.jetson -t life-tracker:jetson-mediapipe .
```

### Step 3: Run with Docker Compose

Choose your mode:

#### Balanced Mode (Recommended, 15W, 30-35 FPS)
```bash
docker-compose -f docker-compose.jetson.yml --profile balanced up -d
```

#### Lite Mode (Power Saving, 7W, 45+ FPS)
```bash
docker-compose -f docker-compose.jetson.yml --profile lite up -d
```

#### Performance Mode (High Accuracy, 25W, 20-25 FPS)
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

## ⚙️ Configuration Modes

### Lite Mode (Power Saving)

```yaml
# config/config_jetson_lite.yaml
Power: 7W
FPS: 45+
Model: YOLOv8n
Resolution: 720p
Frame Skipping: Every 2 frames
```

**Use case**: Battery-powered, demo scenarios

### Balanced Mode (Recommended ⭐)

```yaml
# config/config_jetson_balanced.yaml
Power: 15W
FPS: 30-35
Model: YOLOv8s
Resolution: 720p
Frame Skipping: None
```

**Use case**: 24/7 daily activity monitoring

### Performance Mode (High Accuracy)

```yaml
# config/config_jetson_performance.yaml
Power: 25W
FPS: 20-25
Model: YOLOv8m
Resolution: 1080p
Frame Skipping: None
```

**Use case**: Best accuracy for detailed analysis

---

## 🔧 Advanced Configuration

### Adjust Power Mode

```bash
# Check current mode
sudo /usr/sbin/nvpmodel -q

# Set 15W mode (recommended)
sudo /usr/sbin/nvpmodel -m 0

# Set 7W mode
sudo /usr/sbin/nvpmodel -m 1
```

### Enable TensorRT Optimization

Already enabled in all Jetson configs:
```yaml
tensorrt:
  enabled: true
  fp16_mode: true
  workspace_size: 2048  # MB
```

### Camera Configuration

For CSI camera:
```yaml
camera:
  source: "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
```

For USB camera (default):
```yaml
camera:
  source: 0  # /dev/video0
```

---

## 📊 Performance Monitoring

### Using jtop

```bash
# Install (if not already installed)
sudo apt install python3-jetson-stats

# Run jtop
jtop
```

### Log Monitoring

```bash
# Application logs
tail -f logs/app.log

# Dashboard logs
tail -f logs/dashboard.log

# Docker logs
docker logs -f life-tracker-balanced
```

---

## 🐛 Troubleshooting

### Issue: Camera Not Detected

```bash
# Check camera
ls -l /dev/video*

# Test camera
v4l2-ctl --list-devices
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink  # For CSI camera
```

### Issue: CUDA Out of Memory

```yaml
# Reduce resolution in config
camera:
  resolution: [640, 480]  # Lower resolution

# Or use lite mode
./run_jetson.sh lite
```

### Issue: Low FPS

```bash
# Check power mode
sudo nvpmodel -q

# Set to MAXN mode (25W)
sudo nvpmodel -m 0

# Monitor GPU usage
jtop
```

### Issue: Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Restart docker
sudo systemctl restart docker
```

---

## 🔄 Upgrading to RTMPose (Stage 2)

**Note**: Current release uses MediaPipe. RTMPose upgrade coming in v1.1.

To prepare for RTMPose:

1. Install MMPose:
   ```bash
   source Camera/bin/activate
   pip install openmim
   mim install mmcv-full
   mim install mmpose
   ```

2. Download RTMPose model:
   ```bash
   mim download mmpose --config rtmpose-s_8xb256-420e_coco-256x192 --dest models/
   ```

3. Update config:
   ```yaml
   models:
     pose:
       backend: rtmpose
       model: models/rtmpose-s_8xb256-420e_coco-256x192.pth
       device: cuda:0
   ```

---

## 📝 Version Management

### Current Release

- **v1.0.1-jetson-deploy** - MediaPipe baseline with Jetson deployment scripts

### Upcoming Releases

- **v1.1-rtmpose** - RTMPose GPU acceleration (30-40 FPS expected)
- **v1.2-tensorrt** - Full TensorRT optimization (INT8)

---

## 🆘 Support

- **Issues**: https://github.com/L1TangDingZhen/Camera/issues
- **Jetson Forums**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/
- **Documentation**: See other .md files in repository

---

## 📜 License

MIT License - See LICENSE file for details
