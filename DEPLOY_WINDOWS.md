# Windows Deployment Guide

## 🪟 Windows Special Instructions

Deploying Life Tracker on Windows has some special considerations.

---

## ✅ Recommended Option: Virtual Environment (No Docker)

### 1. Install Python

```powershell
# Download and install Python 3.10+
# https://www.python.org/downloads/

# Verify installation
python --version
```

### 2. Clone Code

```powershell
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera
```

### 3. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Seeing (venv) prefix indicates success
```

### 4. Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Step-by-step install (recommended)
pip install numpy opencv-python pyyaml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
pip install mediapipe
pip install pandas scipy matplotlib psutil tqdm loguru
pip install flask flask-cors

# Or one-click install (may be slow)
pip install -r requirements.txt
```

### 5. Test Run

```powershell
# Test without camera
python test_quick.py

# If camera is available
python main.py --device pc
```

---

## 🐳 Using Docker (Advanced Users)

### Special Docker Configuration on Windows

#### Method 1: Using headless Version (Recommended)

```powershell
# Use headless Dockerfile (no GUI, suitable for background running)
docker build -f Dockerfile.headless -t life-tracker:headless .

# Run (no visualization)
docker run -d ^
  --name life-tracker ^
  -v %cd%\data:/app/data ^
  -v %cd%\logs:/app/logs ^
  -v %cd%\config:/app/config ^
  -p 5000:5000 ^
  life-tracker:headless
```

#### Method 2: Using WSL2 Camera

If you're using WSL2, you can access the camera:

```powershell
# Run in WSL2
wsl

# Then deploy as in Linux
cd /mnt/c/Users/your_username/Desktop/code/Camera
docker-compose up -d
```

#### Method 3: Using Video File for Testing

1. Prepare test video
```powershell
# Download or place video file in data directory
# Example: data\test_video.mp4
```

2. Modify configuration
```yaml
# Edit config/config_pc.yaml
camera:
  source: "data/test_video.mp4"  # Use video file
  fps: 30
  resolution: [640, 480]
```

3. Build and run
```powershell
docker build -t life-tracker .
docker run -d ^
  --name life-tracker ^
  -v %cd%\data:/app/data ^
  -v %cd%\logs:/app/logs ^
  -v %cd%\config:/app/config ^
  life-tracker
```

---

## 📸 Windows Camera Support

### Check Camera

```powershell
# Test after installing dependencies
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# If output is True, camera is available
```

### Common Camera IDs

Camera IDs on Windows may be:
- `0` - Default camera
- `1` - Second camera
- `"video=USB Camera"` - Specify device name

Modify `config/config_pc.yaml`:

```yaml
camera:
  source: 0  # Or 1, 2, etc.
```

---

## 🔧 Common Issues

### 1. torch Installation Failed

```powershell
# Use CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# If you have NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. OpenCV Cannot Open Window

When running in Docker, Windows doesn't support X11 forwarding, solutions:

**Option A**: Use no-visualization mode
```powershell
python main.py --device pc --no-vis
```

**Option B**: Use virtual environment (no Docker)
```powershell
# Run locally on Windows
venv\Scripts\activate
python main.py --device pc
```

### 3. Docker Build Failed

If encountering package installation errors:

```powershell
# Use headless version
docker build -f Dockerfile.headless -t life-tracker .
```

### 4. Permission Issues

```powershell
# Run PowerShell as administrator
# Or use Docker Desktop's integrated terminal
```

### 5. Path Issues

Windows uses backslashes, needs escaping in Python:

```python
# Wrong
camera.source = "C:\videos\test.mp4"

# Correct
camera.source = "C:/videos/test.mp4"
# Or
camera.source = r"C:\videos\test.mp4"
```

---

## 🎯 Quick Start (Windows Recommended Process)

### Option 1: Without Docker (Simplest)

```powershell
# 1. Install Python 3.10+
# 2. Clone code
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Test
python test_quick.py

# 6. Run (if camera available)
python main.py --device pc
```

### Option 2: Using Docker (Background Running)

```powershell
# 1. Make sure Docker Desktop is installed
# 2. Clone code
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera

# 3. Prepare test video (optional)
# Place video in data\test_video.mp4

# 4. Modify configuration to use video
# Edit config\config_pc.yaml
# camera.source: "data/test_video.mp4"

# 5. Build and run
docker build -f Dockerfile.headless -t life-tracker .
docker run -d --name life-tracker -v %cd%\data:/app/data life-tracker

# 6. View logs
docker logs -f life-tracker
```

---

## 💡 Performance Optimization

### CPU Optimization

If no NVIDIA GPU on Windows, force use of CPU:

```yaml
# config/config_pc.yaml
device: cpu

models:
  person:
    device: cpu
  pose:
    backend: mediapipe  # CPU-friendly
    device: cpu
```

### Reduce Resource Usage

```yaml
camera:
  fps: 10  # Lower frame rate
  resolution: [320, 240]  # Lower resolution
```

---

## 📊 Expected Performance (Windows)

| Configuration | FPS | Notes |
|------|-----|------|
| i5-12 + CPU | 5-8 | YOLOv8s + MediaPipe |
| i7-12 + CPU | 8-12 | YOLOv8s + MediaPipe |
| i5-12 + RTX 4070 | 280+ | YOLOv8s + RTMPose |

---

## 🆘 Get Help

If encountering issues:

1. Check `logs\app.log` log
2. Run `python test_quick.py` for diagnostics
3. Check `DEPLOY.md` general deployment guide
4. Submit GitHub Issue

---

## ✅ Verify Successful Installation

Running test script should show:

```
============================================================
  Life Tracker Component Test
============================================================

[1/5] Testing configuration loading...
  ✓ Configuration file loaded successfully

[2/5] Testing YOLOv8 detector...
  ✓ Detector normal
  ✓ FPS: 8.5

[3/5] Testing MediaPipe pose estimation...
  ✓ Pose estimation normal, keypoints: 17

[4/5] Testing ROI manager...
  ✓ ROI manager initialized successfully

[5/5] Testing state machine...
  ✓ State machine initialized successfully

[6/6] Testing database...
  ✓ Database initialized successfully

============================================================
  ✓ All component tests passed!
============================================================
```

---

## 🎉 Complete

Now you can run Life Tracker on Windows!

Recommend using **virtual environment method** (without Docker), simpler and more stable.
