# 🧠 Smart Behavior Prediction System - Detailed Guide

## 💡 This is True "Smart Prediction"!

### Previous Prediction vs Current Prediction

| Comparison | Previous Prediction (predictor.py) | Current Prediction (behavior_predictor.py) |
|--------|---------------------------|-------------------------------------|
| **Prediction Goal** | How long will next sitting session be? | **What should you be doing at current time?** |
| **Based On** | Average sitting duration | Hourly historical behavior patterns |
| **Output** | 45 minutes | 🧍 Standing (78% confidence) |
| **Use Case** | Set reminder interval | **Real-time behavior guidance** |
| **Intelligence Level** | 📊 Simple statistics | 🧠 **Pattern recognition + behavior learning** |

---

## 🎯 Core Function: Predict What You Should Be Doing Based on History

### Example Scenarios

#### Scenario 1: Afternoon Tea Time (Matches Habit) ✅

**Time**: 3 PM
**Historical Pattern**: Over the past 14 days, you've been standing and moving (getting water, walking around) 80% of the time at 3 PM each day

**System Prediction**:
```
🧠 Smart Behavior Prediction
┌──────────────────────────┐
│ 📊 Based on History      │
│ 🧍 Standing              │
│ Confidence: 80%          │
│ Based on your past       │
│ weekday afternoon data,  │
│ you are standing 80% of  │
│ the time at this hour    │
└──────────────────────────┘
┌──────────────────────────┐
│ 🎯 Current Actual State  │
│ 🧍 Standing              │
│ ✅ Matches habit         │
└──────────────────────────┘

✅ Your current state matches your daily habit, keep it up!
```

---

#### Scenario 2: Forgot to Rest (Doesn't Match Habit) ⚠️

**Time**: 3 PM
**Historical Pattern**: Over the past 14 days, you've been standing and moving 80% of the time at 3 PM each day
**Current Situation**: You're still sitting and working

**System Prediction**:
```
🧠 Smart Behavior Prediction
┌──────────────────────────┐
│ 📊 Based on History      │
│ 🧍 Standing              │
│ Confidence: 80%          │
│ Based on your past       │
│ weekday afternoon data,  │
│ you are standing 80% of  │
│ the time at this hour    │
└──────────────────────────┘
┌──────────────────────────┐
│ 🎯 Current Actual State  │
│ 🪑 Sitting               │
│ 💡 Doesn't match habit   │
└──────────────────────────┘

💡 Based on your habits, this is usually time for standing activity (80% probability), recommend standing and moving around
```

**This is true intelligence!** The system learns your daily routine and proactively reminds you when you deviate from your habits.

---

## 🔬 Technical Principles

### 1. Behavior Pattern Learning

**Algorithm Flow**:
```python
# Analyze past 14 days of data
for each_day in past_14_days:
    for hour in 0-23:
        # Count time spent in each state during this hour
        sitting_time[hour] += get_sitting_duration(day, hour)
        standing_time[hour] += get_standing_duration(day, hour)
        lying_time[hour] += get_lying_duration(day, hour)

# Calculate probability of each state per hour
for hour in 0-23:
    total_time = sitting_time[hour] + standing_time[hour] + lying_time[hour]
    sitting_probability[hour] = sitting_time[hour] / total_time
    standing_probability[hour] = standing_time[hour] / total_time
    lying_probability[hour] = lying_time[hour] / total_time

    # Find most common state
    most_common_state[hour] = max(sitting, standing, lying)
    confidence[hour] = max(probabilities)
```

**Data Structure**:
```python
hourly_patterns = {
    0: {'sitting': 0.05, 'standing': 0.0, 'lying': 0.95, 'most_common': 'lying', 'confidence': 0.95},
    1: {'sitting': 0.0, 'standing': 0.0, 'lying': 1.0, 'most_common': 'lying', 'confidence': 1.0},
    ...
    9: {'sitting': 0.85, 'standing': 0.10, 'lying': 0.05, 'most_common': 'sitting', 'confidence': 0.85},
    ...
    14: {'sitting': 0.60, 'standing': 0.35, 'lying': 0.05, 'most_common': 'sitting', 'confidence': 0.60},
    15: {'sitting': 0.20, 'standing': 0.75, 'lying': 0.05, 'most_common': 'standing', 'confidence': 0.75},
    ...
    23: {'sitting': 0.10, 'standing': 0.05, 'lying': 0.85, 'most_common': 'lying', 'confidence': 0.85}
}
```

---

### 2. Current State Prediction

**Prediction Logic**:
```python
def predict_current_state():
    current_hour = datetime.now().hour  # Example: 15 (3 PM)

    # Look up historical pattern for this hour
    pattern = hourly_patterns[current_hour]

    # Return most likely state
    return {
        'predicted_state': pattern['most_common'],  # 'standing'
        'confidence': pattern['confidence'],         # 0.75
        'explanation': generate_explanation(...)
    }
```

**Explanation Generation**:
```python
def generate_explanation(state, hour, confidence):
    # Based on time period
    if hour in [0,1,2,3,4,5]:
        time_desc = 'early morning'
    elif hour in [6,7,8]:
        time_desc = 'morning'
    elif hour in [9,10,11]:
        time_desc = 'late morning'
    elif hour in [12,13]:
        time_desc = 'noon'
    elif hour in [14,15,16,17]:
        time_desc = 'afternoon'
    elif hour in [18,19,20,21]:
        time_desc = 'evening'
    else:
        time_desc = 'late night'

    # Generate human-friendly explanation
    return f'Based on your past {day_type} {time_desc} data, you are in {state} state {confidence*100}% of the time at this hour'
```

---

### 3. Smart Suggestion Generation

**Comparison Algorithm**:
```python
def get_smart_suggestion(current_state):
    # 1. Predict what you should be doing
    prediction = predict_current_state()
    predicted = prediction['predicted_state']

    # 2. Get actual state
    actual = current_state

    # 3. Compare
    if predicted == actual:
        return '✅ Your current state matches your daily habit, keep it up!'
    else:
        # State mismatch, provide suggestion
        if predicted == 'standing' and actual == 'sitting':
            return f'💡 Based on your habits, this is usually time for standing activity ({confidence}% probability), recommend standing and moving around'
        elif predicted == 'sitting' and actual == 'standing':
            return f'💡 Based on your habits, this is usually time for sitting work ({confidence}% probability)'
        # ... more scenarios
```

---

## 📊 Typical Daily Routine Summary

The system also generates your typical daily routine:

```
📅 Your Typical Daily Routine
┌─────────────────────────────────────┐
│ 0-6AM    Sleeping   ████████████ 95% │
│ 7-9AM    Sitting    ████████     80% │
│ 9AM-12PM Sitting    █████████    85% │
│ 12-1PM   Standing   ███████      70% │
│ 1-6PM    Sitting    ████████     80% │
│ 6-7PM    Standing   ████████     75% │
│ 7-10PM   Lying      ███████      72% │
│ 10PM-12AM Sleeping  █████████    88% │
└─────────────────────────────────────┘

Your typical routine:
0-6AM sleeping, 7AM-12PM sitting work, 12-1PM standing activity,
1-6PM sitting work, 6-7PM standing activity, 7-10PM lying rest,
10PM-12AM sleeping

Based on analysis of past 14 days
```

---

## 🚀 How to Use

### Method 1: Web Dashboard (Recommended)

```bash
# Start Web Dashboard
python web_dashboard.py

# Open browser
http://127.0.0.1:5000
```

On the page you will see:
- **🧠 Smart Behavior Prediction** card (blue gradient)
  - Left side: What you should be doing based on historical patterns
  - Right side: What you're actually doing
  - Bottom: Smart suggestions

- **📅 Your Typical Daily Routine** card
  - Timeline display
  - Typical activity for each time period
  - Confidence bar chart

### Method 2: API Call

```bash
# Predict what you should be doing at current time
curl http://127.0.0.1:5000/api/behavior/predict_current_state | jq

# Get smart suggestions
curl http://127.0.0.1:5000/api/behavior/smart_suggestion?current_state=sitting | jq

# View typical daily routine
curl http://127.0.0.1:5000/api/behavior/daily_routine | jq

# View detailed hourly patterns
curl http://127.0.0.1:5000/api/behavior/hourly_patterns?days=14 | jq
```

### Method 3: Python Code

```python
from src.storage.database import Database
from src.analytics.behavior_predictor import SmartBehaviorSuggestion

# Initialize
db = Database('data/database.db')
behavior = SmartBehaviorSuggestion(database=db)

# Analyze behavior patterns
patterns = behavior.analyzer.analyze_hourly_patterns(days=14)
print("Hourly patterns:", patterns)

# Predict what you should be doing now
prediction = behavior.predictor.predict_current_state()
print(f"Prediction: {prediction['predicted_state']}")
print(f"Confidence: {prediction['confidence']}")
print(f"Explanation: {prediction['explanation']}")

# Get smart suggestion (assuming currently sitting)
suggestion = behavior.get_smart_suggestion(current_state='sitting')
print(f"Suggestion: {suggestion['message']}")
print(f"Priority: {suggestion['priority']}")

# View typical daily routine
routine = behavior.get_daily_routine_summary()
print(f"Routine summary: {routine['summary']}")
```

---

## 📈 Data Requirements

| Data Amount | Prediction Quality | Description |
|--------|---------|------|
| **0-1 day** | ❌ Not available | Insufficient data, cannot identify patterns |
| **2-3 days** | ⚠️ Low | Can see preliminary patterns, but unreliable |
| **7 days** | ✅ Medium | One week of data, patterns basically stable |
| **14 days** | ✅✅ Good | **Recommended!** Sufficient to identify weekday/weekend differences |
| **30+ days** | ✅✅✅ Excellent | High confidence, can identify monthly patterns |

---

## 🎯 Actual Usage Effects

### Scenario A: Weekday Morning (High Confidence)

```
Current time: Wednesday 10:30 AM
Historical data: Past 14 days, every weekday at 10 AM, you've been sitting and working

Prediction result:
  Expected: 🪑 Sitting (Confidence: 92%)
  Actual: 🪑 Sitting
  Suggestion: ✅ Your current state matches your daily habit, keep it up!
```

### Scenario B: Lunch Break (Medium Confidence)

```
Current time: Wednesday 12:15 PM
Historical data: Past 14 days, during lunch break you sometimes sit eating (40%), sometimes stand moving (35%), sometimes lie resting (25%)

Prediction result:
  Expected: 🪑 Sitting (Confidence: 40%)  # Most common but not absolute
  Actual: 🧍 Standing
  Suggestion: ✅ Your current state is close to your habits  # Because confidence is not high, don't strongly suggest changing
```

### Scenario C: Afternoon Break Time (High Confidence Mismatch)

```
Current time: Wednesday 3:00 PM
Historical data: Past 14 days, every day at 3 PM you stand and move for 15 minutes

Prediction result:
  Expected: 🧍 Standing (Confidence: 85%)
  Actual: 🪑 Sitting
  Suggestion: 💡 Based on your habits, this is usually time for standing activity (85% probability), recommend standing and moving around
  Priority: HIGH  # Because confidence is high and state doesn't match
```

---

## 🆚 Difference from Simple Prolonged Sitting Reminder

### Traditional Prolonged Sitting Reminder
```
Sat for 30 minutes → Remind to stand up
Sat for 30 minutes → Remind to stand up
Sat for 30 minutes → Remind to stand up
(Regardless of time, just remind after sitting for 30 minutes)
```

### Smart Behavior Prediction
```
10 AM (work time):
  History: Usually sitting and working
  Current: Sitting
  Suggestion: ✅ Matches habit, continue working

3 PM (break time):
  History: Usually standing and moving
  Current: Still sitting
  Suggestion: 💡 Time to stand up and move! (even if only sat for 20 minutes)

8 PM (leisure time):
  History: Usually lying and watching videos
  Current: Still sitting
  Suggestion: 💡 Time to lie down and rest
```

**Key Difference**: Not mechanically reminding based on time, but **learning your habits and only reminding when you deviate from them**!

---

## 🔮 Future Enhancement Directions

### 1. Machine Learning Models
```python
# Use Prophet for time series forecasting
from fbprophet import Prophet

model = Prophet()
model.fit(historical_data)
forecast = model.predict(future_dates)
```

### 2. Context Awareness
```python
# Integrate calendar
if calendar.has_meeting(current_time):
    # During meetings, more likely to be sitting
    predicted_state = 'sitting'
```

### 3. Adaptive Learning
```python
# Recent data has higher weight
weight = lambda days_ago: 1.0 / (1 + days_ago * 0.1)
```

### 4. Activity Recommendations
```python
if predicted == 'standing' and actual == 'sitting':
    activities = [
        'Get a glass of water from the dispenser',
        'Walk to the window and look far away',
        'Do 5 minutes of stretching',
        'Go to the bathroom and wash your face'
    ]
    suggest_random_activity()
```

---

## ✨ Summary

**This is true "Smart Prediction"!**

Not simple statistical averages, but:
- ✅ Learning your daily routine patterns
- ✅ Identifying typical behavior for each time period
- ✅ Predicting what you should be doing at current time
- ✅ Comparing prediction and actual, providing smart suggestions
- ✅ Adjusting suggestion priority based on confidence

**Core Philosophy**: Let AI learn your habits and become your **personalized health assistant**, not a mechanical timer!

---

## 📞 Usage Help

### Quick Test

1. Run main program to record data (at least 3-7 days)
```bash
python main.py --config config/config_gpu.yaml
```

2. Start Web Dashboard
```bash
python web_dashboard.py
```

3. Open browser to view
```
http://127.0.0.1:5000
```

4. Check the **🧠 Smart Behavior Prediction** card to see if the system has learned your habits!

---

**Try it now and see if AI can accurately predict what you should be doing right now!** 🎯
