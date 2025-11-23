# 📹 Life Tracker - AI-Powered Daily Activity Analysis System

> Computer vision and AI-powered system for analyzing daily activities. Automatically recognizes sitting/standing/lying postures, learns your daily routines, and provides personalized health insights.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📚 Documentation

Life Tracker provides comprehensive documentation for users at all levels:

### 📘 For New Users
- **[User Guide](USER_GUIDE.md)** - Quick start, basic usage, and common workflows
  - What is Life Tracker?
  - Default configuration explained
  - Step-by-step setup
  - Troubleshooting basics

### 📗 For Advanced Users
- **[Technical Guide](DL_RL_TECHNICAL_GUIDE.md)** - Model training, advanced configuration, and optimization
  - Complete solution comparison (15 configurations)
  - How to switch between models
  - Training deep learning classifiers
  - RL Ensemble and Decision integration
  - Performance benchmarks

### 📙 Specialized Guides
- **[RTMPose Installation](INSTALL_RTMPOSE.md)** - GPU-accelerated pose estimation setup
  - Linux/Jetson installation (recommended)
  - Windows installation guide
  - Common issues and solutions

- **[RTMPose Technical Comparison](RTMPOSE_TECHNICAL_COMPARISON.md)** - Performance analysis
  - MediaPipe vs RTMPose benchmarks
  - Model selection guide (tiny/s/m/l)
  - TensorRT optimization options

### 🎯 Quick Navigation

| I want to... | Read this |
|--------------|-----------|
| Get started quickly | [User Guide](USER_GUIDE.md) → Quick Start |
| Switch to GPU acceleration | [Technical Guide](DL_RL_TECHNICAL_GUIDE.md) → Solution 2 |
| Train my own models | [Technical Guide](DL_RL_TECHNICAL_GUIDE.md) → Training Workflow |
| Achieve highest accuracy | [Technical Guide](DL_RL_TECHNICAL_GUIDE.md) → Solution 7 (RL Full) |
| Deploy to Jetson | [RTMPose Installation](INSTALL_RTMPOSE.md) |

---

## 🎯 Core Features

### ✨ Implemented Functionality

#### 1. Real-time Posture Recognition 🎯
- ✅ **YOLOv8 Person Detection** - Fast and accurate human detection
- ✅ **MediaPipe Pose Estimation** - Extract 17 COCO keypoints with 3D world coordinates
- ✅ **Multiple Classifier Options** - SVM / Deep Learning / RL Ensemble
- ✅ **Performance**: 15-20 FPS (CPU) / 30-40 FPS (GPU expected)

#### 2. Activity Duration Tracking 📊
- ✅ **SessionTracker** - Automatically logs duration of each activity
- ✅ **Prolonged Activity Detection** - Alerts after 30 minutes of continuous sitting
- ✅ **Daily/Weekly Statistics** - Track sitting, standing, lying time
- ✅ **SQLite Persistence** - All data automatically saved

#### 3. Intelligent Behavior Prediction 🧠
- ✅ **Pattern Learning** - Analyze 14-day data to identify routines
- ✅ **State Prediction** - Predict expected activity at current time
- ✅ **Smart Suggestions** - Compare prediction vs actual, provide personalized reminders
- ✅ **Example**: "Based on your habits, you're usually standing now (75% probability), consider moving around"

#### 4. Web Visualization Dashboard 🌐
- ✅ **Real-time Monitoring** - Current state, duration, location
- ✅ **Interactive Charts** - Pie charts (activity distribution) + Line graphs (weekly trends)
- ✅ **Prediction Cards** - Display predicted vs actual state
- ✅ **Daily Timeline** - Visualize typical schedule
- ✅ **Auto-refresh** - Updates every 30 seconds

#### 5. ROI Zone Management 📍
- ✅ **Multi-zone Support** - Monitor bed/door/desk/bathroom areas
- ✅ **Event Triggers** - Enter/exit events
- ✅ **Calibration Tool** - Easy ROI setup

---

## 🚀 Deployment Roadmap

```
Stage 1: PC Development (RTX 4070)
  ✅ Rapid development + feature validation
  ✅ MediaPipe CPU/GPU mode
  ✅ Current status

Stage 2: Optimization (PC)
  ⏳ RTMPose GPU acceleration
  ⏳ TensorRT optimization
  ⏳ Performance tuning

Stage 3: Edge Deployment (Jetson Orin Nano Super)
  ⏳ Production environment
  ⏳ 24/7 monitoring
  ⏳ TensorRT INT8 inference
```

## 📋 Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ RAM
- Webcam or RTSP camera

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera

# Create virtual environment
python3 -m venv Camera
source Camera/bin/activate  # Linux/Mac
# Camera\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# YOLOv8 models (auto-download on first run)
# MediaPipe pose models (auto-download on first run)

# For RTMPose/ViTPose (optional, requires mmpose)
# See: https://github.com/open-mmlab/mmpose
```

### 3. Calibrate ROI Zones (Optional)

```bash
# Run ROI calibration tool
python scripts/calibrate_roi.py

# Follow prompts to mark zones (bed/desk/door)
# Saves to config/*.yaml
```

### 4. Run the System

```bash
# GPU mode (recommended)
python main.py --mode gpu

# CPU mode
python main.py --mode cpu

# Custom config
python main.py --config config/my_config.yaml
```

### 5. View Dashboard

```bash
# Start web dashboard
python web_dashboard.py

# Open http://localhost:5000
```

## 📁 Project Structure

```
Camera/
├── config/                      # Configuration files
│   ├── config_gpu.yaml         # GPU mode config
│   ├── config_cpu.yaml         # CPU mode config
│   └── config_jetson_*.yaml    # Jetson configs (lite/balanced/performance)
├── src/
│   ├── detectors/              # Detection modules
│   │   ├── base.py            # Base classes and interfaces
│   │   ├── person_detector.py # YOLOv8 person detection
│   │   └── pose_estimator.py  # Pose estimation (multi-backend)
│   ├── classifiers/            # Pose classifiers
│   │   ├── pose_classifier.py          # SVM classifier
│   │   ├── pose_classifier_dl.py       # Deep learning classifier
│   │   └── pose_classifier_ensemble.py # RL ensemble
│   ├── state/                  # State management
│   │   ├── roi_manager.py     # ROI zone management
│   │   ├── behavior_state.py  # Behavior state machine
│   │   └── rl_state_decision.py # RL decision agent
│   ├── analytics/              # Analytics modules
│   │   ├── session_tracker.py # Duration tracking
│   │   ├── predictor.py       # Sitting predictor
│   │   └── behavior_predictor.py # Behavior pattern learning
│   ├── storage/                # Data storage
│   │   ├── database.py        # SQLite database
│   │   └── event_logger.py    # Event logging
│   └── utils/                  # Utilities
├── scripts/                     # Utility scripts
│   ├── calibrate_roi.py       # ROI calibration
│   ├── collect_training_data.py # Data collection
│   ├── train_svm.py           # Train SVM classifier
│   ├── train_dl_classifier.py # Train DL classifier
│   └── compare_models.py      # Model comparison
├── models/                      # Model files
├── data/                        # Data directory
│   └── database.db            # SQLite database
├── main.py                      # Main entry point
├── web_dashboard.py             # Web dashboard
├── query_stats.py               # Query statistics
└── requirements.txt             # Python dependencies
```

## 🔧 Configuration Guide

### Main Configuration Options

```yaml
# config/config_gpu.yaml

device: cuda:0  # Device: cuda:0, cpu

models:
  person:
    model: yolov8m.pt           # Person detection: yolov8n/s/m/l/x
    confidence: 0.5
    device: cuda:0

  pose:
    backend: mediapipe          # Backend: mediapipe, rtmpose, vitpose
    complexity: 1               # MediaPipe: 0=Lite, 1=Full, 2=Heavy

camera:
  source: 0                      # Camera ID or RTSP URL
  fps: 30
  resolution: [1920, 1080]

behavior:
  classifier:
    type: svm                    # Classifier: svm, deep_learning, rl_ensemble
    model_path: models/pose_classifier_svm.pkl

roi:
  zones:
    bed:
      enabled: true
      points: []                # Use calibrate_roi.py to set

tensorrt:
  enabled: false                # Enable on Jetson for acceleration
  fp16_mode: true
```

## 🛠️ Utility Scripts

### ROI Calibration Tool

```bash
python scripts/calibrate_roi.py

# Controls:
# 1. Click to mark polygon vertices
# 2. Press 'c' to complete current zone
# 3. Press 's' to save config
# 4. Press 'q' to quit
```

### Data Collection for Training

```bash
# Collect training data for classifier
python scripts/collect_training_data.py --mode gpu

# Train SVM classifier
python train_svm.py

# Train deep learning classifier (optional)
python scripts/train_dl_classifier.py --model_type lstm
```

### Model Comparison

```bash
python scripts/compare_models.py --config config/config_gpu.yaml

# Automatically tests all model combinations
```

## 📈 Query Statistics

```bash
# View today's statistics
python query_stats.py

# Query specific date range
python query_stats.py --start 2024-01-01 --end 2024-01-31
```

## 🐛 Troubleshooting

### Q1: YOLOv8 model download failed?

```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
mv yolov8m.pt models/
```

### Q2: MediaPipe initialization failed?

```bash
# Reinstall
pip uninstall mediapipe
pip install mediapipe --no-cache-dir
```

### Q3: Camera not opening?

```bash
# Check devices (Linux)
ls /dev/video*

# Try different camera ID
# Modify camera.source in config/*.yaml to 1 or 2
```

### Q4: CUDA out of memory?

```yaml
# Reduce resolution in config
camera:
  resolution: [1280, 720]  # Lower from 1920x1080
```

## 📚 Documentation

- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [WEB_DASHBOARD_GUIDE.md](WEB_DASHBOARD_GUIDE.md) - Dashboard usage
- [MODELS_ARCHITECTURE.md](MODELS_ARCHITECTURE.md) - Model architecture details
- [SESSION_TRACKER_IMPLEMENTATION.md](SESSION_TRACKER_IMPLEMENTATION.md) - SessionTracker technical docs
- [SMART_BEHAVIOR_PREDICTION.md](SMART_BEHAVIOR_PREDICTION.md) - Prediction system
- [JETSON_COMPATIBILITY_ANALYSIS.md](JETSON_COMPATIBILITY_ANALYSIS.md) - Jetson deployment analysis
- [RTMPOSE_TECHNICAL_COMPARISON.md](RTMPOSE_TECHNICAL_COMPARISON.md) - RTMPose vs MediaPipe

## 📝 Roadmap

- [x] Real-time posture detection (sitting/standing/lying)
- [x] Activity duration tracking
- [x] Behavior pattern learning and prediction
- [x] Web visualization dashboard
- [x] ROI zone management
- [ ] Multi-camera support
- [ ] Mobile app integration
- [ ] Cloud data sync
- [ ] Advanced action recognition (eating, exercising, etc.)

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://google.github.io/mediapipe/)
- [MMPose](https://github.com/open-mmlab/mmpose)
