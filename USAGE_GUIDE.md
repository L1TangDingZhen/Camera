# Life Tracker - Usage Guide

## 📁 Project Structure

```
Life Tracker/
├── main.py                    # Single-camera synchronous mode (for debugging)
├── main_async.py              # Unified entry point (supports single/multi-camera async mode)
├── config/
│   ├── config_gpu.yaml        # GPU configuration (default)
│   └── config_cpu.yaml        # CPU-specific configuration
├── src/
│   ├── multi_camera/          # Multi-camera module
│   ├── detectors/             # Detectors (YOLO, RTMPose)
│   ├── state/                 # State management
│   └── storage/               # Data storage
└── models/                    # Model files
```

## 🚀 Quick Start

### 1. Single Camera Mode

```bash
# Synchronous mode (simple, for debugging)
python main.py

# Asynchronous mode (production, recommended)
python main_async.py

# CPU mode
python main_async.py --config config/config_cpu.yaml
```

### 2. Multi-Camera Mode

#### Method 1: Auto-Detect (Recommended)

```bash
# Use default configuration (auto-detect all cameras)
python main_async.py

# Or explicitly specify config file
python main_async.py --config config/config_gpu.yaml
```

Edit `config/config_gpu.yaml` to enable auto-detection (uncomment Option 2):
```yaml
# Option 2: Multi-Camera Auto-Detect - Only works with main_async.py
cameras:
  auto_detect: true
  default_resolution: [1920, 1080]
  default_fps: 30
  max_device_id: 10  # Scan /dev/video0 to /dev/video9
```

#### Method 2: Manual Configuration

Edit `config/config_gpu.yaml`, comment out Option 1, and enable Option 3:

```yaml
# Comment out single camera
# camera:
#   source: 0
#   ...

# Enable manual multi-camera
cameras:
  - id: 0
    name: "Living Room"
    source: 0
    resolution: [1920, 1080]
    fps: 30
  - id: 1
    name: "Bedroom"
    source: 2
    resolution: [1920, 1080]
    fps: 30
```

Then run:
```bash
python main_async.py
```

## 📋 Configuration Files

### `config/config_gpu.yaml` - GPU Configuration

- **Device**: CUDA GPU
- **Camera**: Default single camera (can switch to multi-camera)
- **Models**: YOLOv8n (TensorRT) + RTMPose-s (TensorRT)
- **Classifier**: Transformer + RL Decision Agent
- **Use Cases**: GPU environment (PC/Jetson)

**Switch to multi-camera:**
In the file, comment out the `camera:` configuration and enable the `cameras:` configuration (Option 2 or Option 3)

### `config/config_cpu.yaml` - CPU Configuration

- **Device**: CPU
- **Camera**: Single camera only
- **Models**: YOLOv8n (PyTorch) + MediaPipe
- **Classifier**: SVM
- **Use Cases**: No GPU environment, lightweight deployment

## 🎯 How It Works

### Automatic Mode Detection

`main_async.py` automatically detects camera configuration from the config file:

1. **Single Camera**: Uses `AsyncLifeTracker` (4-thread async pipeline)
2. **Multi-Camera**: Uses `MultiCameraManager` (independent pipeline per camera)

Detection logic:
```python
if cameras.auto_detect == true:
    → Multi-camera mode
elif len(cameras) > 1:
    → Multi-camera mode
else:
    → Single camera mode
```

## ⌨️ Keyboard Shortcuts

### Runtime Controls

- **q** - Quit the application
- **f** - Toggle fullscreen/windowed mode (multi-camera)
- **m** - Toggle split-screen/separate windows mode (multi-camera)

## 🔧 Common Issues

### 1. Camera Not Detected

```bash
# Test camera auto-detection
python test_auto_detect_cameras.py
```

### 2. Camera Initialization Failed

The system will automatically skip failed cameras and continue running the others.

### 3. Blurry Display

The system automatically adapts to screen resolution. Press **f** to toggle fullscreen mode for best display quality.

### 4. GPU Out of Memory

Reduce the number of cameras running simultaneously, or disable multi-person tracking in the config file:
```yaml
models:
  person:
    enable_tracking: false
```

## 📊 Performance Optimization

### Single Camera (AsyncLifeTracker)
- 4-thread async pipeline
- Theoretical FPS: 17-25
- GPU Usage: Medium

### Multi-Camera (MultiCameraManager)
- Independent 4-thread pipeline per camera
- Theoretical FPS: 15-20 (per camera)
- GPU Usage: High (increases with camera count)
- Fullscreen auto-adapts to desktop environment

## 🎨 Visualization Configuration

### Multi-Camera Display Mode

Configure in `config/config_gpu.yaml`:

```yaml
visualization_mode: split_screen  # Split-screen display (default)
# visualization_mode: separate_windows  # Separate windows
```

### Debug Information Display

```yaml
debug:
  show_keypoints: true   # Show keypoints
  show_skeleton: true    # Show skeleton
  verbose: true          # Verbose console output
```

## 📝 Logs and Data

- **Event logs**: `logs/events.log`
- **Database**: `data/database.db`
- **Performance metrics**: Enable with `logging.performance_metrics: true`

## 🚀 Next Steps

1. Adjust camera configuration as needed
2. Calibrate ROI zones (use `scripts/calibrate_roi.py`)
3. Train custom classifiers (use `train_*.py`)
4. Launch Web Dashboard to view statistics

---

**Tip**: First-time users should try `--config config/config_gpu.yaml` to experience auto-detection!
