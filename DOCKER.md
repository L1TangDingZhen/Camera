# Docker Deployment Guide

**Life Tracker** supports Docker deployment on both PC and Jetson platforms.

---

## Quick Start

### PC - CPU Version (YOLO + MediaPipe)
```bash
# Build
docker build -f Dockerfile.cpu -t life-tracker:cpu .

# Run
docker run --rm -it \
  --device /dev/video0:/dev/video0 \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  life-tracker:cpu

# Or use docker-compose
docker-compose --profile cpu up
```

### PC - GPU Version (YOLO + RTMPose)
```bash
# Build
docker build -f Dockerfile.gpu -t life-tracker:gpu .

# Run
docker run --rm -it \
  --gpus all \
  --device /dev/video0:/dev/video0 \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  life-tracker:gpu

# Or use docker-compose
docker-compose --profile gpu up
```

### Jetson - Balanced Mode (YOLO + RTMPose)
```bash
# Build
docker build -f Dockerfile.jetson -t life-tracker:jetson .

# Run
docker run --rm -it \
  --runtime nvidia \
  --device /dev/video0:/dev/video0 \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  life-tracker:jetson

# Or use docker-compose
docker-compose -f docker-compose.jetson.yml --profile balanced up
```

---

## Dockerfile Overview

### 1. Dockerfile.cpu (PC - CPU Only)
**Platform**: PC (x86_64/amd64)
**Inference**: YOLOv8 + MediaPipe
**GPU Required**: No

**Environment**:
- **Python**: 3.10
- **Base Image**: python:3.10-slim

**Features**:
- Lightweight, no CUDA dependencies
- CPU-only inference
- MediaPipe pose estimation
- ~8-12 FPS on modern CPUs

**Use Cases**:
- Development and testing
- Systems without NVIDIA GPU
- Low-power deployment

---

### 2. Dockerfile.gpu (PC - GPU Accelerated)
**Platform**: PC (x86_64/amd64) with NVIDIA GPU
**Inference**: YOLOv8 + RTMPose
**GPU Required**: Yes (CUDA 12.2+)

**Environment**:
- **Python**: 3.10
- **CUDA**: 12.2
- **mmcv**: 2.0.0
- **mmengine**: 0.8.0
- **mmpose**: 1.0.0

**Features**:
- NVIDIA CUDA 12.2 runtime
- RTMPose GPU acceleration with mmcv/mmpose
- Automatic RTMPose verification during build
- ~25-30 FPS on RTX GPUs

**Use Cases**:
- Production deployment
- High-performance requirements
- Development with RTMPose

**Requirements**:
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime (`nvidia-docker2`)
- 4GB+ GPU memory recommended

---

### 3. Dockerfile.jetson (Jetson Orin Nano)
**Platform**: NVIDIA Jetson Orin Nano Super (8GB)
**Inference**: YOLOv8 + RTMPose
**GPU Required**: Yes (integrated GPU)

**Environment**:
- **JetPack**: 6.2.1 (R36.4.0)
- **Kernel**: 5.15.148-tegra
- **Python**: 3.10.12
- **PyTorch**: 2.8.0 (Jetson AI Lab PyPI)
- **CUDA**: 12.6
- **cuDNN**: Included in base image
- **TensorRT**: Included in base image

**Features**:
- Based on `nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.8-py3`
- PyTorch and TorchVision pre-installed
- RTMPose with mmcv compilation (20-40 min build time)
- Optimized for Jetson hardware

**Performance Modes**:
- **Lite**: 640x480, 25-30 FPS, ~10W
- **Balanced**: 1280x720, 20-25 FPS, ~15W (default)
- **Performance**: 1920x1080, 15-20 FPS, ~20-25W

**Use Cases**:
- 24/7 monitoring on edge device
- Low-power deployment
- Production IoT applications

---

## Docker Compose Usage

### PC Deployment

**CPU Version**:
```bash
docker-compose --profile cpu up
```

**GPU Version**:
```bash
docker-compose --profile gpu up
```

**Dashboard Only**:
```bash
docker-compose --profile dashboard up
```

### Jetson Deployment

**Lite Mode** (Power Saving):
```bash
docker-compose -f docker-compose.jetson.yml --profile lite up
```

**Balanced Mode** (Recommended):
```bash
docker-compose -f docker-compose.jetson.yml --profile balanced up
```

**Performance Mode** (High Accuracy):
```bash
docker-compose -f docker-compose.jetson.yml --profile performance up
```

**Dashboard Only**:
```bash
docker-compose -f docker-compose.jetson.yml --profile dashboard up
```

---

## Volume Mapping

All Dockerfiles mount these directories:

```yaml
volumes:
  - ./data:/app/data        # Database and session data
  - ./logs:/app/logs        # Application logs
  - ./models:/app/models    # ML models (YOLO, RTMPose)
  - ./config:/app/config    # Configuration files
```

**Data Persistence**:
- Database: `data/database.db`
- Logs: `logs/app.log`
- Models: `models/*.pt`, `models/*.pth`

---

## Port Mapping

Default port: `5000` (Web Dashboard)

```bash
# Change port mapping
docker run -p 8080:5000 life-tracker:cpu  # Access at http://localhost:8080
```

---

## Camera Access

### Linux (Direct Device Mapping)
```bash
docker run --device /dev/video0:/dev/video0 life-tracker:cpu
```

### Windows/Mac (Use Video File)
Edit `config/config_*.yaml`:
```yaml
camera:
  source: /app/data/test_video.mp4  # Use video file instead of camera
```

### Network Camera (RTSP)
```yaml
camera:
  source: rtsp://username:password@192.168.1.100:554/stream
```

---

## Environment Variables

### Common Variables
```bash
CONFIG_FILE=config/config_gpu.yaml  # Configuration file
PYTHONUNBUFFERED=1                   # Disable Python buffering
```

### GPU-Specific (GPU/Jetson only)
```bash
NVIDIA_VISIBLE_DEVICES=all                    # Which GPUs to use
NVIDIA_DRIVER_CAPABILITIES=compute,utility,video  # Driver capabilities
```

---

## Health Checks

All Dockerfiles include health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health')" || exit 1
```

**Check container health**:
```bash
docker ps  # Look for "healthy" status
```

---

## Build Arguments

### Custom Base Image (Advanced)
```bash
# GPU version with specific CUDA version
docker build -f Dockerfile.gpu \
  --build-arg CUDA_VERSION=12.4.0 \
  -t life-tracker:gpu-cuda124 .
```

---

## Troubleshooting

### Issue: GPU Not Detected in Container

**PC (NVIDIA Docker)**:
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi

# If fails, install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

**Jetson**:
```bash
# Check runtime
docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r36.2.0 nvidia-smi

# If fails, check /etc/docker/daemon.json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

### Issue: Camera Not Accessible

**Linux**:
```bash
# Check camera permissions
ls -l /dev/video0
sudo chmod 666 /dev/video0  # Temporary fix

# Permanent fix: Add user to video group
sudo usermod -aG video $USER
```

**Windows/Mac**:
Use video file or network stream instead of `/dev/video0`.

### Issue: Out of Memory (Jetson)

**Enable swap**:
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Use lite mode**:
```bash
docker-compose -f docker-compose.jetson.yml --profile lite up
```

### Issue: mmcv Compilation Fails (Jetson)

**Symptoms**: Build hangs or fails during mmcv installation

**Solutions**:
1. Ensure 4GB+ RAM available (enable swap if needed)
2. Don't interrupt build process (takes 20-40 minutes)
3. Use pre-built image if available
4. Fall back to MediaPipe (no mmcv needed)

---

## Performance Expectations

### PC - CPU Version (YOLO + MediaPipe)
- **FPS**: 8-12 (depends on CPU)
- **Latency**: ~80-100ms
- **CPU Usage**: 60-80%
- **RAM**: ~2GB

### PC - GPU Version (YOLO + RTMPose)
- **FPS**: 25-30 (RTX 4070, 1080p)
- **Latency**: ~30-40ms
- **GPU Usage**: 40-60%
- **VRAM**: ~2-3GB
- **Power**: ~150-180W

### Jetson - Balanced Mode (YOLO + RTMPose, 720p)
- **FPS**: 20-25
- **Latency**: ~45-50ms
- **Power**: ~15W
- **RAM**: ~3GB

### Jetson - Lite Mode (YOLO + RTMPose, 480p)
- **FPS**: 25-30
- **Latency**: ~35-40ms
- **Power**: ~10W
- **RAM**: ~2.5GB

### Jetson - Performance Mode (YOLO + RTMPose, 1080p)
- **FPS**: 15-20
- **Latency**: ~50-70ms
- **Power**: ~20-25W
- **RAM**: ~3.5GB

---

## Best Practices

1. **Use docker-compose** for easier management
2. **Mount volumes** for data persistence
3. **Set resource limits** in production:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4'
         memory: 8G
   ```
4. **Configure logging** to prevent disk filling
5. **Use specific tags** instead of `latest`
6. **Health check** your containers regularly

---

## Production Deployment

### Recommended Setup (Jetson)

```bash
# 1. Clone repository
git clone https://github.com/your-repo/life-tracker.git
cd life-tracker

# 2. Build image
docker build -f Dockerfile.jetson -t life-tracker:jetson .

# 3. Test in foreground
docker-compose -f docker-compose.jetson.yml --profile balanced up

# 4. Deploy in background
docker-compose -f docker-compose.jetson.yml --profile balanced up -d

# 5. Check logs
docker logs -f life-tracker-balanced

# 6. Monitor health
watch -n 5 docker ps
```

### Auto-Restart on Boot

Add to `/etc/systemd/system/life-tracker.service`:
```ini
[Unit]
Description=Life Tracker Docker Container
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=/path/to/life-tracker
ExecStart=/usr/bin/docker-compose -f docker-compose.jetson.yml --profile balanced up
ExecStop=/usr/bin/docker-compose -f docker-compose.jetson.yml --profile balanced down
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable life-tracker
sudo systemctl start life-tracker
```

---

## Additional Resources

- **Main Documentation**: `README.md`
- **User Guide**: `USER_GUIDE.md`
- **Deployment Guide**: `DEPLOY.md`
- **Jetson Deployment**: `DEPLOY_JETSON.md`
- **RTMPose Installation**: `INSTALL_RTMPOSE.md`

---

**Document Version**: 2.0.0
**Last Updated**: 2025-11-28
**Supported Platforms**: PC (CPU/GPU), Jetson Orin Nano
