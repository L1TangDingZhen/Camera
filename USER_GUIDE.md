# Life Tracker User Guide

Welcome to Life Tracker! This guide will help you get started quickly and understand the core features and basic usage.

## 📖 Table of Contents

- [Project Introduction](#project-introduction)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Current Default Configuration](#current-default-configuration)
- [Quick Start](#quick-start)
- [Basic Usage Workflow](#basic-usage-workflow)
- [Common Commands](#common-commands)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Future Development](#future-development)

---

## Project Introduction

**Life Tracker** is a computer vision-based daily activity analysis system that automatically recognizes and records your sitting, standing, and lying postures to help you understand your daily routines.

### Key Features

- ✅ **Automatic Recognition**: No manual logging needed, camera automatically detects postures
- ✅ **Privacy Protection**: All data processed locally, no cloud uploads
- ✅ **Lightweight & Efficient**: Supports CPU operation, GPU acceleration available
- ✅ **Ready to Use**: Default configuration works out of the box, no training required
- ✅ **Highly Extensible**: Supports multiple model combinations for different accuracy needs

### Use Cases

- 📊 **Health Monitoring**: Track prolonged sitting duration, remind to move
- 🛋️ **Routine Analysis**: Understand daily sitting/standing/lying time distribution
- 💤 **Sleep Tracking**: Automatically identify sleep periods
- 🏠 **Smart Home**: Automatically control lights, AC based on detected state

---

## Core Features

### 1. Posture Recognition

The system can recognize three basic postures:

| Posture | Description | Detection Criteria |
|---------|-------------|-------------------|
| 🪑 **Sitting** | Sitting on chair or bed | Moderate hip height, bent knees |
| 🧍 **Standing** | Standing upright | Upright body, straight knees |
| 🛌 **Lying** | Lying on bed or sofa | Small body-to-horizontal angle |

### 2. Event Tracking

- Enter State: Logged after posture remains stable for a duration
- Leave State: Logged when posture change detected
- Duration: Automatically calculates duration of each state

### 3. Data Storage

- Local SQLite database storage
- Automatic backup (configurable)
- Supports data export and analysis

### 4. Visualization Interface

- Real-time camera feed
- Current state display
- Keypoint visualization (optional)
- Web interface (optional)

---

## System Architecture

Life Tracker uses a **three-layer architecture**, each layer independently configurable:

```
┌──────────────────────────────────────────────┐
│           Decision Layer                      │
│  When to output / How to reduce false alarms │
│  Default: Simple debouncing                   │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│          Classifier Layer                     │
│  Determine current posture: sit/stand/lie    │
│  Default: SVM classifier                      │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│       Pose Estimation Layer                   │
│  Extract human keypoints from image           │
│  Default: MediaPipe                           │
└──────────────────────────────────────────────┘
```

---

## Current Default Configuration

Life Tracker's default configuration is optimized and **ready to use without additional training**:

### 🎯 Default Solution

| Layer | Model | Features | Hardware Requirements |
|-------|-------|----------|----------------------|
| Pose Estimation | **MediaPipe** | CPU-friendly, cross-platform | CPU |
| Classifier | **SVM** | Fast, stable, pre-trained | CPU |
| Decision Strategy | **Simple Debouncing** | Simple and effective | - |

### 📊 Default Performance

- **Accuracy**: 90-95% (based on standard environment)
- **Latency**: ~50ms/frame
- **Resource Usage**: Moderate (~300MB RAM)
- **Device Requirements**: Regular PC, no GPU needed

### ✨ Advantages of Default Solution

- ✅ **Plug and Play**: No training required, run immediately
- ✅ **Cross-platform**: Supports Windows/Linux/macOS
- ✅ **Stable & Reliable**: Extensively tested
- ✅ **Resource Friendly**: CPU-only, works on laptops

### ⚠️ Suitable Scenarios

Default configuration is suitable for:
- First-time use, quick experience
- Daily home use, moderate accuracy requirements
- CPU-only environment
- Standard lighting and angles

For higher accuracy or faster speed, see [Advanced Usage](#advanced-usage).

---

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Webcam** (built-in laptop or external USB webcam)
3. **Basic Dependencies** (auto-installed)

### Three Steps to Get Started

#### Step 1: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/Camera.git
cd Camera

# Install core dependencies
pip install -r requirements.txt
```

**What gets installed**:
- OpenCV (image processing)
- MediaPipe (pose estimation)
- PyTorch (machine learning framework)
- YOLOv8 (person detection)
- Other utility libraries

#### Step 2: Run the System

```bash
# Run with default config (GPU mode)
python main.py --config config/config_gpu.yaml

# Or use CPU mode (slower but more compatible)
python main.py --config config/config_cpu.yaml
```

#### Step 3: Observe Results

After system starts, you'll see:

```
[Init] Loading person detector...
[Init] Loading pose estimator...
[MediaPipe] Initialization complete ✓
[BehaviorStateMachine] Using SVM classifier
[BehaviorStateMachine] Using simple debouncing decision
[System] Started successfully! Press 'q' to quit
```

**Real-time display shows**:
- Camera video stream
- Currently detected posture (Sitting/Standing/Lying)
- Posture duration
- Person detection box and keypoints (optional)

---

## Basic Usage Workflow

### 1. First Run

For first-time run, we recommend:

```bash
# Use GPU config (if you have a graphics card)
python main.py --config config/config_gpu.yaml

# System will automatically:
# 1. Open webcam
# 2. Detect person
# 3. Recognize posture
# 4. Display real-time results
```

**Confirm system is working**:
- ✅ Camera feed displays normally
- ✅ Person detected (green box)
- ✅ Posture recognition accurate (try sitting, standing, lying)

### 2. Collect Training Data (Optional)

If default model accuracy is insufficient, collect your own data:

```bash
# Collect sitting data (60 seconds)
python collect_data.py --label sitting --duration 60

# Collect standing data
python collect_data.py --label standing --duration 60

# Collect lying data
python collect_data.py --label lying --duration 60
```

**Collection tips**:
- Collect at least 30-60 seconds per posture
- Include different angles and lighting
- Simulate real usage scenarios

After collection, data is saved in `training_data/` directory.

### 3. Train Model (Optional)

If you collected your own data, you can retrain:

```bash
# Train SVM model
python train_svm.py --data training_data

# New model saved to: models/pose_classifier_svm.pkl
# Next run will automatically use new model
```

### 4. View Analysis Results

During system operation, all data is saved to database:

```bash
# View database
sqlite3 data/database.db

# Query today's activity records
SELECT * FROM events WHERE date(timestamp) = date('now');

# Calculate today's sitting duration
SELECT SUM(duration) FROM events
WHERE state = 'sitting' AND date(timestamp) = date('now');
```

---

## Common Commands

### Run System

```bash
# GPU mode (recommended if you have GPU)
python main.py --config config/config_gpu.yaml

# CPU mode (better compatibility)
python main.py --config config/config_cpu.yaml

# Specify camera device
python main.py --config config/config_gpu.yaml --camera 1
```

### Data Collection

```bash
# Collect training data
python collect_data.py --label [sitting|standing|lying] --duration 60

# View collected data
ls training_data/
cat training_data/sitting.json | jq length  # Check sample count
```

### Model Training

```bash
# Train SVM (default classifier)
python train_svm.py --data training_data

# View trained models
ls models/
```

### Testing and Debugging

```bash
# Quick test
python test_quick.py

# Robustness test (test model accuracy)
python test_robustness.py --label sitting --duration 30

# View logs
tail -f logs/app.log
```

### Configuration and Maintenance

```bash
# View config file
cat config/config_gpu.yaml

# Backup database
cp data/database.db data/database_backup.db

# Clean logs
rm logs/*.log
```

---

## Advanced Usage

Default configuration meets most needs, but if you want:

- 🚀 **Faster Speed** (GPU acceleration)
- 📈 **Higher Accuracy** (deep learning models)
- 🎯 **Fewer False Alarms** (intelligent decision)
- 🔧 **Custom Configuration** (adapt to special environments)

### Other Available Solutions

Life Tracker supports multiple model combination solutions:

| Solution | Pose Estimation | Classifier | Accuracy Gain | Speed Gain | Needs GPU | Needs Training |
|----------|----------------|------------|---------------|-----------|-----------|----------------|
| **Default** | MediaPipe | SVM | Baseline | Baseline | ❌ | ❌ |
| **GPU Accelerated** | RTMPose | SVM | - | **4x** ⚡ | ✅ | ❌ |
| **High Accuracy** | RTMPose | MLP | **+5%** | 4x | ✅ | ✅ |
| **Highest Accuracy** | RTMPose | Ensemble | **+8%** | 3x | ✅ | ✅ |

### 📚 Detailed Technical Documentation

Want to learn how to switch solutions, train models, tune parameters?

➡️ **See**: [DL_RL_TECHNICAL_GUIDE.md](DL_RL_TECHNICAL_GUIDE.md)

Technical documentation includes:
- Detailed comparison of all solutions
- Step-by-step switching guide
- Complete training workflow
- Model performance benchmarks
- Advanced configuration options

---

## Troubleshooting

### Problem 1: Cannot Open Camera

**Symptoms**:
```
Error: Cannot open camera
```

**Solutions**:
```bash
# Check camera devices
ls /dev/video*  # Linux
# Or check in Windows Device Manager

# Try other device IDs
python main.py --config config/config_gpu.yaml --camera 1
python main.py --config config/config_gpu.yaml --camera 2
```

### Problem 2: Display Too Small/Blurry

**Symptoms**: Window very small, text hard to read

**Solutions**:
```yaml
# Modify config file: config/config_gpu.yaml
camera:
  resolution: [1280, 720]  # Change to higher resolution
  fps: 30  # Increase frame rate
```

### Problem 3: Inaccurate Recognition

**Possible Causes**:
- Poor lighting
- Wrong camera angle
- Atypical posture

**Solutions**:

**Solution A**: Adjust Environment
- Improve lighting
- Adjust camera position (recommended: front view, 2-3 meters distance)

**Solution B**: Collect Training Data
```bash
# Collect data in your actual environment
python collect_data.py --label sitting --duration 120
python collect_data.py --label standing --duration 120
python collect_data.py --label lying --duration 120

# Retrain
python train_svm.py --data training_data
```

**Solution C**: Switch to Higher Accuracy Model
See [DL_RL_TECHNICAL_GUIDE.md](DL_RL_TECHNICAL_GUIDE.md)

### Problem 4: Slow/Laggy

**Symptoms**: Low FPS, choppy video

**Solutions**:

```yaml
# Method 1: Reduce resolution
camera:
  resolution: [640, 480]  # From 1280x720 to 640x480
  fps: 15  # Reduce frame rate

# Method 2: Skip frames
inference:
  skip_frames: 2  # Process every 2 frames
  detection_interval: 3  # Detect person every 3 frames
```

Or switch to CPU config:
```bash
python main.py --config config/config_cpu.yaml
```

### Problem 5: Training Data Not Found

**Symptoms**:
```
FileNotFoundError: training_data/sitting.json not found
```

**Solutions**:
```bash
# Check data files
ls training_data/

# If none, collect data first
python collect_data.py --label sitting --duration 60
```

### More Questions?

- Check logs: `cat logs/app.log`
- Check Issues: https://github.com/yourusername/Camera/issues
- Read technical docs: [DL_RL_TECHNICAL_GUIDE.md](DL_RL_TECHNICAL_GUIDE.md)

---

## Future Development

Life Tracker is continuously improving, future plans:

### Short-term Plans

- ✅ **RTMPose Integration** (GPU-accelerated pose estimation) - Completed
- ✅ **Deep Learning Classifiers** (MLP/LSTM/Transformer) - Completed
- ✅ **RL Integration** (Ensemble fusion + adaptive decision) - Completed
- 🔄 **Jetson Optimization** (edge device deployment) - In Progress
- 📱 **Mobile App** (Android/iOS) - Planned

### Long-term Vision

- 🎯 **More Posture Recognition** (working, resting, exercising, etc.)
- 🏠 **Smart Home Integration** (Home Assistant, HomeKit)
- 📊 **Advanced Data Analysis** (trend prediction, health recommendations)
- 🌐 **Multi-person Tracking** (independent tracking for family members)
- 🤖 **AI Assistant Integration** (voice interaction, smart reminders)

### Contributing

Contributions welcome - code, suggestions, or issue reports:
- GitHub: https://github.com/yourusername/Camera
- Issues: https://github.com/yourusername/Camera/issues
- Discussions: https://github.com/yourusername/Camera/discussions

---

## Documentation Navigation

- 📘 **User Guide** (Current): Quick start and basic usage
- 📗 **Technical Guide**: [DL_RL_TECHNICAL_GUIDE.md](DL_RL_TECHNICAL_GUIDE.md) - Model training and advanced configuration
- 📙 **RTMPose Installation**: [INSTALL_RTMPOSE.md](INSTALL_RTMPOSE.md) - GPU-accelerated pose estimation
- 📕 **Technical Comparison**: [RTMPOSE_TECHNICAL_COMPARISON.md](RTMPOSE_TECHNICAL_COMPARISON.md) - Performance benchmarks

---

**Start using Life Tracker and understand your every day!** 🚀
