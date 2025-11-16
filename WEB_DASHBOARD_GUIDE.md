# Web Dashboard User Guide

## 🌐 Feature Overview

The Web Dashboard is the **data visualization interface** for the prolonged sitting reminder system, providing the following features:

1. **Real-time Data Monitoring** - Current activity status, duration
2. **Statistical Charts** - Today's activity distribution, weekly trends
3. **Smart Prediction** - Predict next sitting duration, optimal reminder time
4. **Anomaly Detection** - Identify abnormal prolonged sitting behavior
5. **Prolonged Sitting Warning** - Real-time warning alerts

---

## 🚀 Quick Start

### 1. Start Web Server

```bash
# Basic startup (default http://127.0.0.1:5000)
python web_dashboard.py

# Specify port
python web_dashboard.py --port 8080

# Allow external access (LAN access)
python web_dashboard.py --host 0.0.0.0 --port 5000

# Enable debug mode
python web_dashboard.py --debug
```

### 2. Access Dashboard

Open browser and visit:
```
http://127.0.0.1:5000
```

For accessing from other devices on LAN:
```
http://[your_computer_IP]:5000
```

### 3. View Real-time Data

The dashboard will **auto-refresh every 30 seconds**, no manual operation needed.

---

## 📊 Feature Details

### 1. Current Status Card

Displays current activity information:
- **Current Status**: Sitting 🪑 / Standing 🧍 / Lying 🛏️
- **Duration**: How long current status has lasted
- **Zone**: Detected zone (bed/chair, etc.)

**Example**:
```
Current Status: 🪑 Sitting
Duration: 15 minutes
Zone: chair
```

---

### 2. Prolonged Sitting Warning Banner

When continuously sitting exceeds **30 minutes**, a red warning bar will appear at the top:

```
⚠️ Prolonged Sitting Warning
You have been sitting for 35 minutes, recommend standing up!
```

**Features**:
- Auto-triggered (no manual checking needed)
- Prominent red banner
- Shake animation effect

---

### 3. Today's Statistics Cards

Four statistic cards showing today's data:

| Icon | Metric | Description |
|------|------|------|
| 🪑 | Sitting Duration | Today's cumulative sitting time |
| 🧍 | Standing Duration | Today's cumulative standing time |
| 🛏️ | Lying Duration | Today's cumulative lying time |
| 📊 | Session Count | Today's total activity sessions |

---

### 4. Detailed Sitting Statistics

More detailed sitting analysis:
- **Total Duration**: Today's total sitting time (hours)
- **Session Count**: How many times sat today
- **Average Duration**: Average duration per sitting session
- **Longest Session**: Longest sitting session duration

**Purpose**: Understand your sitting habits

---

### 5. Today's Activity Distribution (Pie Chart)

Visualize today's time allocation:
- 🟥 Red = Sitting
- 🟦 Blue = Standing
- 🟨 Yellow = Lying

**Interaction**:
- Hover to view specific values
- Auto-calculate percentages

---

### 6. Weekly Activity Trend (Line Chart)

Shows activity trends over the past 7 days:
- 📈 Red line = Sitting duration
- 📈 Blue line = Standing duration
- 📈 Yellow line = Lying duration

**Purpose**:
- Discover weekly patterns (e.g., whether weekend sitting is longer)
- Track improvement progress

---

### 7. Smart Prediction 🔮

**AI prediction features** based on historical data:

#### 7.1 Predict Next Sitting Duration

```
🎯 Predicted Next Sitting Duration
45 minutes
Confidence: 75%
Suggestion: Expected prolonged sitting, recommend setting 25-minute reminder
```

**Algorithm**:
1. Analyze sitting records from the past 30 days
2. Calculate average based on current time period (±2 hours)
3. Use median to smooth outliers
4. Provide confidence score

**Cold Start**: Requires at least 3 sitting records for prediction

#### 7.2 Suggested Reminder Interval

```
⏰ Suggested Reminder Interval
25 minutes
Your sitting duration is long, recommend frequent reminders. High-risk periods: 9-12AM, 2-5PM
```

**Algorithm**:
1. Analyze sitting patterns from past 14 days
2. Identify periods prone to prolonged sitting
3. Recommend reminder interval based on average sitting duration:
   - Average >45 minutes → 25-minute reminder (frequent)
   - Average 30-45 minutes → 30-minute reminder (standard)
   - Average <30 minutes → 40-minute reminder (relaxed)

---

### 8. Today's Anomaly Detection 🔍

Intelligently detect if today's behavior is abnormal:

#### Example 1: Normal
```
✅ Today's sitting duration is normal

Today's sitting: 2.5 hours
Historical average: 2.8 hours
Deviation: -10.7%
```

#### Example 2: Moderate Anomaly
```
🟡 ⚠️ Today's sitting duration significantly higher than average (+35%), please pay attention to more movement

Today's sitting: 4.2 hours
Historical average: 3.1 hours
Deviation: +35.5%
```

#### Example 3: High Anomaly
```
🔴 ⚠️ Today's sitting duration significantly higher than average (+67%), please pay attention to more movement

Today's sitting: 6.8 hours
Historical average: 4.1 hours
Deviation: +65.9%
```

**Judgment Criteria**:
- Deviation <20% → Normal (green)
- Deviation 20-50% → Moderate anomaly (yellow)
- Deviation >50% → High anomaly (red)

---

## 🔌 API Interface

Web Dashboard provides RESTful API that can be integrated into other applications:

### Basic Statistics

```bash
# Today's statistics
curl http://127.0.0.1:5000/api/stats/today

# Sitting statistics
curl http://127.0.0.1:5000/api/stats/sitting

# Weekly statistics
curl http://127.0.0.1:5000/api/stats/weekly

# Current session
curl http://127.0.0.1:5000/api/stats/current

# Historical records (last 7 days)
curl http://127.0.0.1:5000/api/stats/history?days=7
```

### Prediction Interface

```bash
# Predict next sitting duration
curl http://127.0.0.1:5000/api/prediction/next_sitting

# Predict optimal reminder time
curl http://127.0.0.1:5000/api/prediction/optimal_reminder

# Anomaly detection
curl http://127.0.0.1:5000/api/prediction/anomaly
```

### Alert Interface

```bash
# Check prolonged sitting warning (default 30 minutes)
curl http://127.0.0.1:5000/api/alert/prolonged_sitting

# Custom threshold (25 minutes)
curl http://127.0.0.1:5000/api/alert/prolonged_sitting?threshold=25
```

---

## 🎨 Responsive Design

Web Dashboard supports multiple devices:

- **Desktop Browser** (recommended): Full features, best experience
- **Tablet**: Adaptive layout
- **Mobile**: Single column display, retains all features

**Recommended Resolution**: 1280x720 and above

---

## ⚙️ Configuration Options

### Modify Auto-refresh Interval

Edit `static/js/dashboard.js`:

```javascript
// Modify refresh interval (default 30 seconds)
refreshInterval = setInterval(refreshAllData, 30000);  // Change to 60000 = 1 minute
```

### Modify Prolonged Sitting Warning Threshold

Edit `web_dashboard.py`:

```python
# Modify default threshold (default 30 minutes)
def check_prolonged_sitting():
    threshold = int(request.args.get('threshold', 30))  # Change to 25
```

### Modify Chart Colors

Edit `static/css/style.css`, find corresponding color codes and modify.

---

## 🐛 Common Issues

### 1. Page shows "Database file does not exist"

**Reason**: Haven't run `main.py` to create database yet

**Solution**:
```bash
# First run main program to create database
python main.py --config config/config_gpu.yaml

# Wait a few minutes for it to record data

# Then start Web Dashboard
python web_dashboard.py
```

### 2. All statistics show 0

**Reason**: No data in database yet

**Solution**:
1. Run `main.py` to let system record your activities
2. Sit/stand/lie a few times to generate some data
3. Refresh Web Dashboard

### 3. Prediction shows "Predictions will be available after accumulating more data"

**Reason**: Insufficient historical data (requires at least 3 sitting records)

**Solution**:
- Continue using the system to accumulate more data
- Recommend checking prediction feature after at least 3-7 days of use

### 4. Charts not displaying

**Check**:
1. Whether browser console has errors (F12)
2. Confirm Chart.js loaded successfully (requires network connection)
3. Clear browser cache and retry

### 5. Cannot access from LAN

**Check**:
1. Confirm started with `--host 0.0.0.0` parameter
2. Check firewall settings (open port 5000)
3. Confirm IP address is correct

---

## 📱 Mobile Access

### Add to Home Screen (iOS/Android)

1. Open `http://[IP]:5000` in browser
2. Click "Add to Home Screen"
3. Use like a native app

### Recommended Settings

- **iOS Safari**: Add to home screen
- **Android Chrome**: Install PWA (future feature)

---

## 🔐 Security Notes

⚠️ **Important**: Web Dashboard has no authentication mechanism by default

**Recommendations**:
1. Use only on local or trusted LAN
2. Do not expose to public internet
3. For public access, recommend configuring reverse proxy + HTTPS

**Advanced Configuration** (using Nginx reverse proxy):
```nginx
location / {
    proxy_pass http://127.0.0.1:5000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

---

## 🚀 Performance Optimization

### Server Side

```bash
# Use production server (Gunicorn)
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 web_dashboard:app
```

### Client Side

- Charts use Canvas rendering (high performance)
- Auto-cache today's statistics
- Asynchronous data loading

---

## 📊 Data Privacy

All data stored in **local SQLite database** (`data/database.db`):
- ✅ No data uploaded to cloud
- ✅ Works completely offline
- ✅ You own all data

---

## 🎯 Next Steps

- [ ] Weekly/monthly report auto-generation (PDF/email)
- [ ] Export data to Excel
- [ ] Custom reminder rules
- [ ] Multi-user support
- [ ] Desktop notification integration
- [ ] PWA support (offline use)
- [ ] Data comparison (this week vs last week)

---

## 📞 Technical Support

Encountering issues?

1. Check "Common Issues" section in this document
2. View browser console error messages
3. View server terminal output
4. Run `python query_stats.py` to check if data is normal

---

## Summary

Web Dashboard provides **complete data visualization and prediction features**:

✅ **Real-time Monitoring** - Current status, duration, zone
✅ **Statistical Analysis** - Today/weekly data, chart display
✅ **Smart Prediction** - Predict next sitting duration, optimal reminder time
✅ **Anomaly Detection** - Automatically identify abnormal behavior
✅ **Prolonged Sitting Warning** - Auto-reminder after exceeding 30 minutes
✅ **API Interface** - Can integrate into other applications
✅ **Responsive Design** - Supports mobile/tablet/desktop

**Start using it now!** 🎉
