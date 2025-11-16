# Local Deployment Guide

## 📋 Prerequisites

- Python 3.8+
- Camera (USB or built-in)
- (Optional) NVIDIA GPU + CUDA 11.0+

---

## 🚀 Option 1: Virtual Environment (Recommended for Beginners)

### 1. Clone Project

```bash
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera
git checkout claude/three-stage-deployment-roadmap-011CUrFSWFN5rH8EACYAZGjD
```

### 2. Create Virtual Environment

```bash
# Create
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Seeing (venv) prefix indicates success
```

### 3. Install Dependencies

```bash
# Method A: One-click install (may fail)
pip install -r requirements.txt

# Method B: Step-by-step install (more stable)
pip install --upgrade pip
pip install numpy opencv-python pyyaml
pip install torch torchvision  # CPU version, auto-selected
pip install ultralytics
pip install mediapipe
pip install pandas scipy matplotlib psutil tqdm loguru
pip install flask flask-cors
```

**Common Issues:**

- **Slow torch installation?** Use Tsinghua mirror:
  ```bash
  pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- **Have NVIDIA GPU?** Install CUDA version:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### 4. Test Installation

```bash
# Quick test (no camera needed)
python test_quick.py

# Seeing "All component tests passed" indicates success
```

### 5. Calibrate ROI Area

```bash
# Run after connecting camera
python scripts/calibrate_roi.py --device pc

# Operations:
# - Click mouse to mark area vertices
# - Press 'c' to complete current area
# - Press 's' to save configuration
# - Press 'q' to exit
```

### 6. Official Run

```bash
# PC development mode (GPU)
python main.py --device pc

# X390 verification mode (CPU)
python main.py --device x390

# No window display (background run)
python main.py --device pc --no-vis
```

### 7. Exit Environment

```bash
deactivate
```

---

## 🐳 Option 2: Docker (Complete Isolation)

### 1. Install Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER  # Add user to docker group
# Logout and log back in

# Mac: Download Docker Desktop
# Windows: Download Docker Desktop
```

### 2. Build Image

```bash
cd Camera

# Build
docker build -t life-tracker .

# Or use docker-compose
docker-compose build
```

### 3. Run Container

```bash
# Method 1: docker command
docker run -it --rm \
  --device=/dev/video0 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config:/app/config \
  life-tracker

# Method 2: docker-compose (recommended)
docker-compose up -d  # Run in background
docker-compose logs -f  # View logs
docker-compose down  # Stop
```

### 4. Enter Container for Debugging

```bash
docker exec -it life-tracker bash
```

---

## 🔧 Option 3: Conda Environment (Recommended for Researchers)

### 1. Install Conda

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 2. Create Environment

```bash
cd Camera

# Create environment
conda create -n life-tracker python=3.10 -y
conda activate life-tracker

# Install dependencies
pip install -r requirements.txt
```

### 3. Run

```bash
conda activate life-tracker
python test_quick.py
python main.py --device pc
```

### 4. Exit

```bash
conda deactivate
```

---

## 📦 Testing Without Camera

### Using Video File

```bash
# 1. Prepare test video
# Download or record a video with people, place in data/test_video.mp4

# 2. Modify configuration
# Edit config/config_pc.yaml
camera:
  source: "data/test_video.mp4"  # Change to video path
  fps: 30
  resolution: [640, 480]

# 3. Run
python main.py --device pc
```

### Using Test Script

```bash
# Test without camera
python test_quick.py

# Performance test
python test_quick.py --perf
```

---

## 🐛 Common Issue Troubleshooting

### 1. ImportError: No module named 'cv2'

```bash
pip install opencv-python
```

### 2. ImportError: No module named 'ultralytics'

```bash
pip install ultralytics
```

### 3. Camera Cannot Open

```bash
# Check devices
ls /dev/video*

# Try different IDs
# Modify camera.source in config/config_pc.yaml to 1 or 2
```

### 4. CUDA out of memory

```yaml
# Modify config/config_pc.yaml
models:
  person:
    device: cpu  # Change to CPU
  pose:
    backend: mediapipe  # Use CPU-friendly backend
    device: cpu
```

### 5. YOLOv8 Model Download Failed

```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
mv yolov8s.pt models/
```

### 6. MediaPipe Initialization Failed

```bash
# Uninstall and reinstall
pip uninstall mediapipe -y
pip install mediapipe --no-cache-dir
```

---

## 📊 Verify Successful Run

After running, you should see:

```
============================================================
  Life Tracker - PC Development
  Device: cuda:0
============================================================

[Initialization] Loading person detector...
[PersonDetector] Loading model: yolov8s.pt
[PersonDetector] Device: cuda:0
[PersonDetector] Model loaded successfully

[Initialization] Loading pose estimator...
[MediaPipePose] Initialization complete, complexity=1

[Initialization] Loading ROI manager...
[ROIManager] Loaded 0 areas: []

[Initialization] Creating state machine...
[BehaviorStateMachine] Initialization complete

[Initialization] Creating event logger...
[EventLogger] Initialization complete

[Initialization] Opening camera...

[Initialization] All components loaded!

[Running] Starting monitoring...
```

Window display:
- Real-time footage
- FPS display
- Current state
- Area information

---

## 🎯 Next Steps

1. **Calibrate ROI area**: `python scripts/calibrate_roi.py --device pc`
2. **Model comparison test**: `python scripts/compare_models.py --device pc`
3. **View Web interface**: (To be implemented) Visit http://localhost:5000
4. **Migrate to X390**: Follow Phase 2 steps in README.md

---

## 💡 Performance Reference

| Environment | Configuration | FPS | Notes |
|------|------|-----|------|
| PC GPU | i5-12 + RTX 4070 | 280+ | YOLOv8s + RTMPose |
| PC CPU | i5-12 | 6-8 | YOLOv8s + MediaPipe |
| X390 CPU | i5-8 | 6-8 | YOLOv8s + MediaPipe |
| Jetson | Orin Nano | 38-45 | YOLOv8s-TRT + RTMPose-TRT |

---

## 📞 Get Help

Encountering issues?

1. Check FAQ section in `README.md`
2. Run `python test_quick.py` to locate issues
3. Check `logs/app.log` log file
4. Submit GitHub Issue
