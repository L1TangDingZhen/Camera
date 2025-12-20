# Multi-Camera System Guide

## 🎥 Overview

The Life Tracker multi-camera system supports simultaneous tracking across multiple camera feeds with:
- Independent person detection and tracking per camera
- Unified split-screen visualization
- Per-camera event logging with `camera_id` tagging
- Shared model weights for memory efficiency
- Full multi-person support on each camera

## 📋 Requirements

### Hardware
- **2 USB cameras** (or more, up to 4 supported)
- **Jetson Orin Nano with 8GB RAM** (or equivalent)
- **GPU memory**: Each camera uses ~1.5-2GB GPU RAM
  - 2 cameras: ~3-4GB GPU RAM
  - 3 cameras: ~5-6GB GPU RAM

### Software
- All dependencies from main Life Tracker installation
- No additional packages required

## 🚀 Quick Start

### 1. Check Available Cameras

```bash
# List all video devices
ls /dev/video*

# Output example:
# /dev/video0  /dev/video1  /dev/video2  /dev/video3
#
# Note: USB cameras usually appear at even numbers (0, 2, 4...)
# video1, video3, etc. are often metadata devices
```

### 2. Test Camera Access

```bash
# Test camera 0
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera 0:', cap.isOpened()); cap.release()"

# Test camera 2
python -c "import cv2; cap = cv2.VideoCapture(2); print('Camera 2:', cap.isOpened()); cap.release()"
```

### 3. Configure Cameras

Edit `config/config_multi_camera.yaml`:

```yaml
cameras:
  - id: 0
    name: "Living Room"
    source: 0  # Your first camera device
    resolution: [1280, 720]
    fps: 30

  - id: 1
    name: "Bedroom"
    source: 2  # Your second camera device
    resolution: [1280, 720]
    fps: 30
```

### 4. Run Multi-Camera System

```bash
# Default configuration
python main_multi_camera.py

# Custom configuration
python main_multi_camera.py --config config/my_multi_camera.yaml
```

### 5. Controls

- **Press 'q'**: Quit the application
- **Press 'm'**: Toggle between split-screen and separate windows mode

## 📐 Architecture

```
┌────────────────────────────────────────────────────────┐
│            MultiCameraManager                          │
│         (Shared EventLogger + Database)                │
└────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌──────────────────┐           ┌──────────────────┐
│  CameraInstance  │           │  CameraInstance  │
│   Camera ID: 0   │           │   Camera ID: 1   │
│                  │           │                  │
│  - PersonDetect  │           │  - PersonDetect  │
│  - PoseEstimate  │           │  - PoseEstimate  │
│  - MultiPerson   │           │  - MultiPerson   │
│  - StateMachine  │           │  - StateMachine  │
│                  │           │                  │
│  Thread Pool:    │           │  Thread Pool:    │
│  └─ Main Loop    │           │  └─ Main Loop    │
└──────────────────┘           └──────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        ▼
            ┌─────────────────────┐
            │  Split-Screen View  │
            │   [Cam 0 | Cam 1]   │
            └─────────────────────┘
```

## 🔧 Configuration Details

### Camera-Specific Settings

Each camera can have independent settings:

```yaml
cameras:
  - id: 0
    name: "Main Camera"
    source: 0
    resolution: [1920, 1080]  # Camera 0: Full HD
    fps: 30

    # Optional: Camera-specific ROI zones
    roi:
      enabled: true
      zones:
        bed:
          type: polygon
          points: [[100, 200], [500, 200], [500, 600], [100, 600]]

  - id: 1
    name: "Side Camera"
    source: 2
    resolution: [1280, 720]  # Camera 1: HD
    fps: 30

    # Different ROI zones for this camera
    roi:
      enabled: true
      zones:
        desk:
          type: polygon
          points: [[50, 100], [400, 100], [400, 500], [50, 500]]
```

### Shared Model Configuration

All cameras share the same model configuration to save memory:

```yaml
models:
  person:
    model: models/yolov8n.engine
    enable_tracking: true
    max_persons: 3  # Per camera, so 2 cameras × 3 persons = 6 total

  pose:
    backend: mediapipe
    device: cuda:0
```

### Visualization Modes

```yaml
visualization_mode: split_screen  # Options: split_screen, separate_windows
```

- **split_screen**: All cameras in one window (recommended for 2-4 cameras)
  - 2 cameras: Side-by-side layout
  - 3 cameras: Top row (2) + bottom row (1 full width)
  - 4 cameras: 2×2 grid

- **separate_windows**: Each camera in its own window

## 📊 Database Schema

Events from multi-camera system include `camera_id` field:

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    event_type TEXT,
    timestamp REAL,
    state TEXT,
    zone TEXT,
    tracking_id INTEGER,    -- Person ID within camera
    camera_id INTEGER,      -- Camera ID (0, 1, 2, ...)
    metadata TEXT,
    created_at TIMESTAMP
);
```

### Querying Multi-Camera Data

```python
from src.storage import Database

db = Database('data/database.db')

# Get events from specific camera
camera_0_events = db.get_events(camera_id=0)

# Get all events with camera info
all_events = db.get_events()
for event in all_events:
    print(f"Camera {event['camera_id']}: Person {event['tracking_id']} - {event['state']}")
```

## ⚡ Performance Optimization

### Resolution vs Performance

| Resolution | FPS (per camera) | GPU RAM | Total FPS (2 cameras) |
|------------|------------------|---------|----------------------|
| 640×480    | ~60 FPS          | ~1.2GB  | ~120 FPS             |
| 1280×720   | ~40 FPS          | ~1.8GB  | ~80 FPS              |
| 1920×1080  | ~25 FPS          | ~2.5GB  | ~50 FPS              |

**Recommendation**: Use 1280×720 for best balance

### Detection Interval Optimization

```yaml
inference:
  detection_interval: 3  # Detect every 3 frames
  # Higher value = less GPU usage, but slower tracking response
  # Recommended: 2-5 for real-time, 5-10 for recording
```

### Max Persons Per Camera

```yaml
models:
  person:
    max_persons: 3  # Reduce if experiencing performance issues
    # Memory usage increases with more persons tracked
```

## 🐛 Troubleshooting

### Issue: "Cannot open camera"

**Solution**: Check camera device numbers
```bash
# Find available cameras
v4l2-ctl --list-devices

# Test each device
python -c "import cv2; cap = cv2.VideoCapture(X); print(cap.isOpened())"
```

### Issue: "Out of GPU memory"

**Solutions**:
1. Reduce camera resolution (e.g., 1920×1080 → 1280×720)
2. Reduce max_persons per camera
3. Increase detection_interval
4. Use fewer cameras

### Issue: Low FPS

**Solutions**:
1. Increase `detection_interval` (e.g., 3 → 5)
2. Reduce camera resolution
3. Disable pose estimation on some cameras
4. Use MJPEG encoding: `self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))`

### Issue: Cameras out of sync

This is normal - each camera runs independently. Events are timestamped for accurate correlation.

## 📈 Future Enhancements

### Phase 1 (Current)
- ✅ Multiple camera support
- ✅ Per-camera tracking
- ✅ Split-screen visualization
- ✅ Database with camera_id

### Phase 2 (Planned)
- [ ] Cross-camera person tracking (Re-ID)
- [ ] Global person ID across cameras
- [ ] Activity heatmaps per camera
- [ ] Multi-camera timeline view

### Phase 3 (Future)
- [ ] Face recognition integration
- [ ] Person identity binding across cameras
- [ ] Multi-camera 3D pose reconstruction
- [ ] Automatic camera calibration

## 🔗 Related Features

This multi-camera system works seamlessly with:
- **Multi-person tracking** (v1.2.0) - Each camera tracks multiple persons
- **Face recognition** (upcoming v1.4.0) - Identify persons across cameras
- **ROI zones** - Define different zones per camera
- **Behavior analysis** - Independent analysis per camera

## 📞 Support

For issues specific to multi-camera setup:
1. Check camera device numbers: `ls /dev/video*`
2. Verify GPU memory: `nvidia-smi`
3. Test cameras individually first
4. Review logs: `logs/events_multi_camera.log`

## 🎯 Example Use Cases

### Home Monitoring
- Camera 0: Living room (track daily activity)
- Camera 1: Bedroom (track sleep patterns)

### Office Setup
- Camera 0: Desk area (track sitting duration)
- Camera 1: Break area (track standing/walking)

### Care Facility
- Camera 0: Patient room
- Camera 1: Common area
- Track patient movement patterns across zones

---

**Version**: v1.3.0-MultiCamera
**Last Updated**: 2025-12-19
