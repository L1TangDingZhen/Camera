# Prolonged Sitting Reminder System - Quick Start Guide

## ✅ Completed Features

### 1. SessionTracker (Duration Statistics) ✅
- Automatic tracking of sitting/standing/lying duration
- Today/weekly statistics
- Prolonged sitting detection (>30 minutes)
- Data persistence to SQLite

### 2. Web Dashboard (Data Visualization) ✅
- Real-time status monitoring
- Interactive charts (pie charts, line charts)
- Responsive design (mobile/tablet/desktop)
- Auto-refresh (every 30 seconds)

### 3. AI Prediction (Smart Prediction) ✅
- Predict next sitting duration
- Recommend optimal reminder interval
- Anomaly behavior detection
- Personalized suggestions

---

## 🚀 Quick Start in 3 Steps

### Step 1: Run Main Program (Record Data)

```bash
# Install dependencies
pip install -r requirements.txt

# Start camera monitoring
python main.py --config config/config_gpu.yaml

# Let it run for a while to record your activities
# Recommend at least 30 minutes to 1 hour for sufficient data
```

**What does this step do?**
- Opens camera
- Detects your posture (sitting/standing/lying)
- Automatically records duration of each state
- Saves data to `data/database.db`

---

### Step 2: View Statistics (Command Line)

```bash
# Run in another terminal
python query_stats.py
```

**You will see:**
```
📊 SessionTracker Statistics
============================================================

【Today's Statistics】
Date: 2025-11-08
Total sessions: 15

  🪑 Sitting: 2h 30m (2.50h)
  🧍 Standing: 1h 15m (1.25h)
  🛏️  Lying: 0h 20m (0.33h)

【Current Session】
  Status: sitting
  Duration: 5m
  Zone: chair

【Detailed Sitting Statistics】
  Total duration: 150.0 minutes (2.50h)
  Session count: 8
  Average per session: 18.8 minutes
  Longest session: 45.0 minutes

【Weekly Statistics】
  Period: 2025-11-04 to 2025-11-10
  ...
```

---

### Step 3: Launch Web Dashboard (Visualization Interface)

```bash
# Start web server
python web_dashboard.py

# Open browser and visit
# http://127.0.0.1:5000
```

**You will see:**
- 📍 **Current Status**: Real-time display of sitting/standing/lying
- 📊 **Today's Statistics**: Four cards showing sitting, standing, lying duration and session count
- 💺 **Detailed Sitting**: Total duration, session count, average duration, longest duration
- 📊 **Activity Pie Chart**: Visualize today's time allocation
- 📈 **Weekly Trend**: Line chart showing changes over the past 7 days
- 🔮 **Smart Prediction**: AI predicts next sitting duration and suggested reminder interval
- 🔍 **Anomaly Detection**: Automatically identify abnormal sitting behavior today

**Prolonged Sitting Warning:**
- When you sit for more than 30 minutes, a red warning banner will appear at the top!

---

## 📂 Project File Structure

```
Camera/
├── main.py                          # Main program (camera monitoring)
├── query_stats.py                   # Command line statistics query
├── web_dashboard.py                 # Web visualization server
├── config/
│   ├── config_gpu.yaml              # GPU configuration (recommended)
│   └── config_cpu.yaml              # CPU configuration
├── data/
│   └── database.db                  # SQLite database (auto-created)
├── src/
│   ├── analytics/
│   │   ├── session_tracker.py       # Duration statistics module
│   │   └── predictor.py             # AI prediction module
│   ├── state/
│   │   └── behavior_state.py        # State machine (integrated SessionTracker)
│   └── storage/
│       ├── database.py              # Database operations
│       └── event_logger.py          # Event logging
├── templates/
│   └── dashboard.html               # Web interface HTML
├── static/
│   ├── css/style.css                # Styles
│   └── js/dashboard.js              # Frontend logic
├── HOW_TO_RUN.md                    # Detailed running guide
├── WEB_DASHBOARD_GUIDE.md           # Web feature description
└── SESSION_TRACKER_IMPLEMENTATION.md # Technical documentation
```

---

## 🎯 Use Cases

### Scenario 1: Daily Monitoring
```bash
# Start after booting up in the morning
python main.py --config config/config_gpu.yaml

# Run all day, automatically record your activities
# Open http://127.0.0.1:5000 anytime to view data
```

### Scenario 2: Daily Review
```bash
# Check today's statistics after work in the evening
python query_stats.py

# Or open Web Dashboard to view detailed charts and predictions
```

### Scenario 3: Prolonged Sitting Reminder
```bash
# Start main program
python main.py --config config/config_gpu.yaml

# Also start Web Dashboard
python web_dashboard.py

# Open browser, warnings will automatically appear when sitting exceeds 30 minutes
```

---

## 🔧 Quick Troubleshooting

### Q1: First time running query_stats.py shows all data as 0?
**A**: Normal! Need to run `main.py` first to record some activity data.

### Q2: Web Dashboard prediction shows "Predictions will be available after accumulating more data"?
**A**: Prediction requires at least 3 sitting records. Recommend checking the prediction feature after 3-7 days of use.

### Q3: main.py error "Unable to open camera"?
**A**: Check:
1. Whether the camera is occupied by other programs
2. Modify `camera.source` in `config/config_gpu.yaml` (0, 1, 2...)

### Q4: Cannot access Web Dashboard?
**A**: Confirm:
1. `python web_dashboard.py` is running
2. Browser visits http://127.0.0.1:5000 (note the port number)

### Q5: Want to view data on mobile?
**A**:
```bash
# Allow LAN access when starting
python web_dashboard.py --host 0.0.0.0

# Visit in mobile browser
http://[your_computer_IP]:5000
# For example: http://192.168.1.100:5000
```

---

## 📊 API Usage Examples

### Python Call
```python
from src.storage.database import Database
from src.analytics.session_tracker import SessionTracker
from src.analytics.predictor import SittingPredictor

# Create instances
db = Database('data/database.db')
tracker = SessionTracker(database=db)
predictor = SittingPredictor(database=db)

# Get today's statistics
stats = tracker.get_today_statistics()
print(f"Today's sitting: {stats['sitting_duration']/3600:.1f} hours")

# Predict next sitting duration
prediction = predictor.predict_next_sitting_duration()
print(f"Predicted next sitting: {prediction['predicted_duration_minutes']} minutes")
print(f"Confidence: {prediction['confidence']*100:.0f}%")

# Anomaly detection
anomaly = predictor.detect_anomaly()
if anomaly['is_anomaly']:
    print(f"⚠️ {anomaly['message']}")
```

### JavaScript Call (Browser)
```javascript
// Get today's statistics
fetch('http://127.0.0.1:5000/api/stats/today')
  .then(res => res.json())
  .then(data => {
    console.log('Today sitting:', data.data.sitting_duration/3600, 'hours');
  });

// Get prediction
fetch('http://127.0.0.1:5000/api/prediction/next_sitting')
  .then(res => res.json())
  .then(data => {
    console.log('Predicted duration:', data.data.predicted_duration_minutes, 'minutes');
  });
```

### curl Command Line Call
```bash
# Today's statistics
curl http://127.0.0.1:5000/api/stats/today | jq

# Prediction
curl http://127.0.0.1:5000/api/prediction/next_sitting | jq

# Anomaly detection
curl http://127.0.0.1:5000/api/prediction/anomaly | jq
```

---

## 🎨 Screenshot Preview

### Web Dashboard Interface

```
┌─────────────────────────────────────────────────────────────┐
│  🪑 Prolonged Sitting Reminder System - Data Dashboard       │
│  Real-time monitoring of your activity data, maintain       │
│  healthy lifestyle                                           │
└─────────────────────────────────────────────────────────────┘

⚠️ Prolonged Sitting Warning
   You have been sitting for 35 minutes, recommend standing up!

┌─────────────────────────────────────────────────────────────┐
│ 📍 Current Status                                            │
│ Current: 🪑 Sitting   Duration: 15 minutes   Zone: chair     │
└─────────────────────────────────────────────────────────────┘

┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ 🪑       │ │ 🧍       │ │ 🛏️       │ │ 📊       │
│ Sitting  │ │ Standing │ │ Lying    │ │ Sessions │
│ 2.5h     │ │ 1.2h     │ │ 0.3h     │ │ 15       │
└──────────┘ └──────────┘ └──────────┘ └──────────┘

┌─────────────────────────────────────────────────────────────┐
│ 💺 Detailed Sitting Statistics                               │
│ Total: 2.5h  Sessions: 8  Average: 18.8m  Longest: 45.0m    │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐  ┌──────────────────────────────────────┐
│ 📊 Today's       │  │ 📈 Weekly Activity Trend              │
│    Activity      │  │  [Line chart]                        │
│    Distribution  │  │                                      │
│  [Pie chart]     │  │                                      │
└──────────────────┘  └──────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 🔮 Smart Prediction                                          │
│ 🎯 Predicted next sitting: 45 min  ⏰ Suggested interval: 25 min │
│ Confidence: 75%                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 🔍 Today's Anomaly Detection                                 │
│ ✅ Today's sitting duration is normal                        │
│ Today: 2.5h  Historical avg: 2.8h  Deviation: -10.7%        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔥 Advanced Features

### 1. Customize Reminder Threshold
Edit `config/config_gpu.yaml`:
```yaml
session_tracking:
  prolonged_sitting_threshold: 25  # Change to 25 minutes
```

### 2. Export Data to CSV
```bash
sqlite3 -header -csv data/database.db \
  "SELECT * FROM state_history;" > my_activity_data.csv
```

### 3. LAN Multi-device Access
```bash
# Server side
python web_dashboard.py --host 0.0.0.0 --port 5000

# Other devices on the same WiFi
# http://[computer_IP]:5000
```

### 4. Production Environment Deployment
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_dashboard:app
```

---

## 📚 Related Documentation

- **HOW_TO_RUN.md** - Complete installation and running guide
- **WEB_DASHBOARD_GUIDE.md** - Detailed web feature description
- **SESSION_TRACKER_IMPLEMENTATION.md** - Technical implementation documentation

---

## ✨ Summary

You now have a **complete prolonged sitting reminder system**:

1. ✅ **Data Collection** - main.py automatically records activities
2. ✅ **Data Storage** - SQLite persistence
3. ✅ **Data Query** - query_stats.py command line query
4. ✅ **Data Visualization** - Web Dashboard chart display
5. ✅ **Smart Prediction** - AI prediction and suggestions
6. ✅ **Anomaly Detection** - Automatically identify abnormal behavior
7. ✅ **Prolonged Sitting Warning** - Real-time reminders

**Start using it now!** 🎉

```bash
# Terminal 1: Start monitoring
python main.py --config config/config_gpu.yaml

# Terminal 2: Start Web Dashboard
python web_dashboard.py

# Browser: Open browser
# http://127.0.0.1:5000
```

**Wishing you healthy work habits!** 💪
