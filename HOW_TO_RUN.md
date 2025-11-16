# How to Run the Prolonged Sitting Reminder System

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

**Notes:**
- Python version requires **3.8+**
- Some GPU packages (mmcv-full/mmpose) may fail to install on Windows, which doesn't affect CPU execution
- If ultralytics installation fails, install separately: `pip install ultralytics`

### 2. Prepare Camera

Ensure your computer has an available camera:
- **Laptop built-in camera**: Usually `/dev/video0` (Linux) or `0` (Windows)
- **External USB camera**: May be `/dev/video1` or `1`

### 3. Run the Application

```bash
# Method 1: Use GPU configuration (if you have a CUDA GPU)
python main.py --config config/config_gpu.yaml

# Method 2: Use CPU configuration (universal, but slower)
python main.py --config config/config_cpu.yaml

# Method 3: Enable debug mode (show skeleton points, angles, etc.)
python main.py --config config/config_gpu.yaml --debug

# Method 4: No visualization window (run in background only)
python main.py --config config/config_gpu.yaml --no-vis
```

### 4. Check SessionTracker Effect

After running, you will see:

**Startup Information:**
```
[Initialization] Loading configuration...
[Initialization] Loading pose estimator...
[Initialization] Loading ROI manager...
[Initialization] Creating event logger...
[Initialization] Creating state machine...
[BehaviorStateMachine] SessionTracker enabled  ← Seeing this line means SessionTracker is working
[Initialization] Opening camera...
```

**During Runtime:**
- Camera window will display real-time footage
- Upper left corner shows current state (Sitting/Standing/Lying)
- SessionTracker automatically records duration of each state in background

**Data Storage Location:**
```
data/database.db  ← SQLite database, contains all session records
```

---

## View SessionTracker Recorded Data

### Method 1: Use Python Script Query

Create a simple query script:

```python
# query_stats.py
from src.storage.database import Database
from src.analytics.session_tracker import SessionTracker

# Connect to database
db = Database('data/database.db')
tracker = SessionTracker(database=db)

# View today's statistics
stats = tracker.get_today_statistics()
print(f"📊 Today's Statistics ({stats['date']}):")
print(f"  Sitting: {stats['sitting_duration']/3600:.2f} hours")
print(f"  Standing: {stats['standing_duration']/3600:.2f} hours")
print(f"  Lying: {stats['lying_duration']/3600:.2f} hours")
print(f"  Total sessions: {stats['total_sessions']}")

# View detailed sitting statistics
sitting_stats = tracker.get_sitting_statistics()
print(f"\n💺 Detailed Sitting Statistics:")
print(f"  Total duration: {sitting_stats['total_duration_minutes']:.1f} minutes")
print(f"  Session count: {sitting_stats['session_count']}")
print(f"  Average per session: {sitting_stats['average_session_duration']/60:.1f} minutes")
print(f"  Longest session: {sitting_stats['longest_session']/60:.1f} minutes")

# Check for prolonged sitting
if tracker.check_prolonged_sitting(threshold_minutes=30):
    current_duration = tracker.get_current_duration() / 60
    print(f"\n⚠️  Prolonged Sitting Warning: Continuous sitting for {current_duration:.0f} minutes!")
```

Run:
```bash
python query_stats.py
```

### Method 2: Query Database Directly

```bash
# View last 10 session records
sqlite3 data/database.db "SELECT datetime(timestamp, 'unixepoch', 'localtime') as time, state, duration/60 as duration_min, zone FROM state_history ORDER BY id DESC LIMIT 10;"
```

### Method 3: Use Python REPL

```python
python3

>>> from src.storage.database import Database
>>> from src.analytics.session_tracker import SessionTracker
>>>
>>> db = Database('data/database.db')
>>> tracker = SessionTracker(database=db)
>>>
>>> # View today's statistics
>>> tracker.get_today_statistics()
>>>
>>> # View weekly statistics
>>> tracker.get_weekly_statistics()
```

---

## Configuration File Description

### `config/config_gpu.yaml` (Recommended)
- Uses GPU acceleration
- YOLOv8 person detection + MediaPipe pose estimation
- FPS: ~15-20 (sufficient for detecting sit/stand/lie)

### `config/config_cpu.yaml` (Backup)
- Pure CPU execution
- MediaPipe pose estimation
- FPS: ~10-15 (may be slower)

You can modify configuration as needed:

```yaml
camera:
  source: 0  # Change to your camera number
  resolution: [1920, 1080]  # Change to your desired resolution

behavior:
  thresholds:
    min_confidence: 0.5  # Pose estimation confidence threshold
    standing_hip_knee_angle: 150  # Standing determination angle

# SessionTracker configuration (can be added in the future)
session_tracking:
  prolonged_sitting_threshold: 30  # Prolonged sitting threshold (minutes)
  auto_save_interval: 60  # Auto-save interval (seconds)
```

---

## Common Issues

### 1. Camera Won't Open
```
RuntimeError: Unable to open camera: 0
```

**Solutions:**
- Linux: Check camera device `ls /dev/video*`, modify `camera.source` in config
- Windows: Try changing to `1` or `2` (if multiple cameras)
- Check camera permissions

### 2. Dependency Installation Failed

**mmcv-full/mmpose installation failed (common on Windows):**
```bash
# These packages are only used in GPU mode, not needed for CPU mode
# Can comment out these two lines in requirements.txt
```

**PyTorch installation failed:**
```bash
# Visit https://pytorch.org for installation command suitable for your system
# For example (CPU version):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Running Slow/Laggy

**Solutions:**
- Lower resolution: Modify `camera.resolution` in config to `[1280, 720]` or `[640, 480]`
- Use CPU configuration: `--config config/config_cpu.yaml`
- Disable debug mode: Don't use `--debug` parameter

### 4. SessionTracker Not Enabled

**Check if startup log contains:**
```
[BehaviorStateMachine] SessionTracker enabled
```

**If not:**
- Check if `src/analytics/session_tracker.py` exists
- Check if import is successful (should not have ImportError)

### 5. View Real-time Statistics

**Currently SessionTracker runs in background, not displayed in interface.**

**View real-time data:**
```python
# Run in another terminal
python query_stats.py
```

**Next development steps:** We will add:
- Display current session duration on screen
- Display today's statistics panel
- Red warning when sitting exceeds 30 minutes

---

## Keyboard Controls

Supported keys during runtime:

- **q**: Exit program
- **Space**: Pause/resume
- **s**: Screenshot save
- **d**: Toggle debug mode
- **r**: Reset ROI area

---

## Data Storage

### Database Location
```
data/database.db
```

### Table Structure
```sql
state_history:
  - id: Record ID
  - timestamp: End timestamp
  - state: State (sitting/standing/lying/sleeping)
  - zone: Zone (bed/chair/desk, etc.)
  - duration: Duration (seconds)
  - created_at: Record creation time
```

### Backup Data
```bash
# Backup entire database
cp data/database.db data/database_backup_$(date +%Y%m%d).db

# Export to CSV
sqlite3 -header -csv data/database.db "SELECT * FROM state_history;" > sessions.csv
```

---

## Complete Startup Process Example

```bash
# 1. Clone project (if not already done)
cd ~/Camera

# 2. Install dependencies
pip install -r requirements.txt

# 3. Confirm camera is available
# Linux:
ls /dev/video*
# Windows: Open "Camera" app to test

# 4. Run (GPU mode)
python main.py --config config/config_gpu.yaml

# 5. Observe startup log, confirm SessionTracker is enabled
# [BehaviorStateMachine] SessionTracker enabled  ← This line is important

# 6. Sit/stand in front of camera, let system detect your posture

# 7. Open another terminal to view statistics
python query_stats.py

# 8. Press 'q' key to exit
```

---

## Next Steps

SessionTracker is now working in the background, recording your activity data.

**Features to be added:**
1. ✅ SessionTracker background running (completed)
2. ⏳ Display current session duration on screen (in development)
3. ⏳ Display today's statistics panel (in development)
4. ⏳ Prolonged sitting warning prompt (in development)
5. ⏳ Weekly/monthly report generation (planned)
6. ⏳ Export Excel reports (planned)

Start running now and let SessionTracker begin recording your activity data! 📊
