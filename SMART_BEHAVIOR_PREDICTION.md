# 🧠 智能行为预测系统 - 详细说明

## 💡 这才是真正的"智能预测"！

### 之前的预测 vs 现在的预测

| 对比项 | 之前的预测 (predictor.py) | 现在的预测 (behavior_predictor.py) |
|--------|---------------------------|-------------------------------------|
| **预测目标** | 下次坐姿会坐多久？ | **当前时间应该做什么？** |
| **基于数据** | 平均坐姿时长 | 每小时的历史行为模式 |
| **输出结果** | 45分钟 | 🧍 站立（78%置信度） |
| **使用场景** | 设置提醒间隔 | **实时行为指导** |
| **智能程度** | 📊 简单统计 | 🧠 **模式识别+行为学习** |

---

## 🎯 核心功能：根据历史预测当前应该做什么

### 示例场景

#### 场景1: 下午茶时间（符合习惯）✅

**时间**: 下午3点
**历史规律**: 你过去14天，每天下午3点，80%的时间在站立活动（喝水、走动）

**系统预测**:
```
🧠 智能行为预测
┌──────────────────────────┐
│ 📊 根据历史规律          │
│ 🧍 站立                  │
│ 置信度: 80%              │
│ 根据您过去的工作日下午   │
│ 数据，此时您有80%的时间  │
│ 处于站立状态             │
└──────────────────────────┘
┌──────────────────────────┐
│ 🎯 当前实际状态          │
│ 🧍 站立                  │
│ ✅ 符合习惯              │
└──────────────────────────┘

✅ 您的当前状态符合日常习惯，保持良好！
```

---

#### 场景2: 忘记休息（不符合习惯）⚠️

**时间**: 下午3点
**历史规律**: 你过去14天，每天下午3点，80%的时间在站立活动
**实际情况**: 你还在坐着工作

**系统预测**:
```
🧠 智能行为预测
┌──────────────────────────┐
│ 📊 根据历史规律          │
│ 🧍 站立                  │
│ 置信度: 80%              │
│ 根据您过去的工作日下午   │
│ 数据，此时您有80%的时间  │
│ 处于站立状态             │
└──────────────────────────┘
┌──────────────────────────┐
│ 🎯 当前实际状态          │
│ 🪑 坐姿                  │
│ 💡 与习惯不符            │
└──────────────────────────┘

💡 根据您的习惯，现在通常是站立活动的时间（80%概率），建议起身走动一下
```

**这就是真正的智能！** 系统学会了你的作息规律，当你偏离习惯时主动提醒。

---

## 🔬 技术原理

### 1. 行为模式学习

**算法流程**:
```python
# 分析过去14天的数据
for each_day in past_14_days:
    for hour in 0-23:
        # 统计这个小时处于各状态的时长
        sitting_time[hour] += get_sitting_duration(day, hour)
        standing_time[hour] += get_standing_duration(day, hour)
        lying_time[hour] += get_lying_duration(day, hour)

# 计算每小时各状态的概率
for hour in 0-23:
    total_time = sitting_time[hour] + standing_time[hour] + lying_time[hour]
    sitting_probability[hour] = sitting_time[hour] / total_time
    standing_probability[hour] = standing_time[hour] / total_time
    lying_probability[hour] = lying_time[hour] / total_time

    # 找出最常见的状态
    most_common_state[hour] = max(sitting, standing, lying)
    confidence[hour] = max(probabilities)
```

**数据结构**:
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

### 2. 当前状态预测

**预测逻辑**:
```python
def predict_current_state():
    current_hour = datetime.now().hour  # 例如: 15 (下午3点)

    # 查找这个小时的历史模式
    pattern = hourly_patterns[current_hour]

    # 返回最可能的状态
    return {
        'predicted_state': pattern['most_common'],  # 'standing'
        'confidence': pattern['confidence'],         # 0.75
        'explanation': generate_explanation(...)
    }
```

**解释生成**:
```python
def generate_explanation(state, hour, confidence):
    # 根据时间段
    if hour in [0,1,2,3,4,5]:
        time_desc = '凌晨'
    elif hour in [6,7,8]:
        time_desc = '早晨'
    elif hour in [9,10,11]:
        time_desc = '上午'
    elif hour in [12,13]:
        time_desc = '中午'
    elif hour in [14,15,16,17]:
        time_desc = '下午'
    elif hour in [18,19,20,21]:
        time_desc = '晚上'
    else:
        time_desc = '深夜'

    # 生成人性化解释
    return f'根据您过去的{day_type}{time_desc}数据，此时您有{confidence*100}%的时间处于{state}状态'
```

---

### 3. 智能建议生成

**对比算法**:
```python
def get_smart_suggestion(current_state):
    # 1. 预测当前应该做什么
    prediction = predict_current_state()
    predicted = prediction['predicted_state']

    # 2. 获取实际状态
    actual = current_state

    # 3. 对比
    if predicted == actual:
        return '✅ 您的当前状态符合日常习惯，保持良好！'
    else:
        # 状态不匹配，给出建议
        if predicted == 'standing' and actual == 'sitting':
            return f'💡 根据您的习惯，现在通常是站立活动的时间（{confidence}%概率），建议起身走动一下'
        elif predicted == 'sitting' and actual == 'standing':
            return f'💡 根据您的习惯，现在通常是坐姿工作的时间（{confidence}%概率）'
        # ... 更多场景
```

---

## 📊 典型作息总结

系统还会生成你的典型作息：

```
📅 您的典型作息
┌─────────────────────────────────────┐
│ 0-6时    睡眠      ████████████ 95% │
│ 7-9时    坐姿工作  ████████     80% │
│ 9-12时   坐姿工作  █████████    85% │
│ 12-13时  站立活动  ███████      70% │
│ 13-18时  坐姿工作  ████████     80% │
│ 18-19时  站立活动  ████████     75% │
│ 19-22时  躺卧休息  ███████      72% │
│ 22-24时  睡眠      █████████    88% │
└─────────────────────────────────────┘

您的典型作息：
0-6时睡眠，7-12时坐姿工作，12-13时站立活动，
13-18时坐姿工作，18-19时站立活动，19-22时躺卧休息，
22-24时睡眠

基于过去14天的数据分析
```

---

## 🚀 如何使用

### 方式1: Web Dashboard（推荐）

```bash
# 启动Web Dashboard
python web_dashboard.py

# 浏览器打开
http://127.0.0.1:5000
```

在页面上会看到：
- **🧠 智能行为预测** 卡片（蓝色渐变）
  - 左侧：根据历史规律，你应该做什么
  - 右侧：你实际在做什么
  - 底部：智能建议

- **📅 您的典型作息** 卡片
  - 时间线展示
  - 每个时段的典型活动
  - 置信度条形图

### 方式2: API调用

```bash
# 预测当前时间应该做什么
curl http://127.0.0.1:5000/api/behavior/predict_current_state | jq

# 获取智能建议
curl http://127.0.0.1:5000/api/behavior/smart_suggestion?current_state=sitting | jq

# 查看典型作息
curl http://127.0.0.1:5000/api/behavior/daily_routine | jq

# 查看每小时详细模式
curl http://127.0.0.1:5000/api/behavior/hourly_patterns?days=14 | jq
```

### 方式3: Python代码

```python
from src.storage.database import Database
from src.analytics.behavior_predictor import SmartBehaviorSuggestion

# 初始化
db = Database('data/database.db')
behavior = SmartBehaviorSuggestion(database=db)

# 分析行为模式
patterns = behavior.analyzer.analyze_hourly_patterns(days=14)
print("每小时模式:", patterns)

# 预测当前应该做什么
prediction = behavior.predictor.predict_current_state()
print(f"预测: {prediction['predicted_state']}")
print(f"置信度: {prediction['confidence']}")
print(f"解释: {prediction['explanation']}")

# 获取智能建议（假设当前在坐着）
suggestion = behavior.get_smart_suggestion(current_state='sitting')
print(f"建议: {suggestion['message']}")
print(f"优先级: {suggestion['priority']}")

# 查看典型作息
routine = behavior.get_daily_routine_summary()
print(f"作息总结: {routine['summary']}")
```

---

## 📈 数据需求

| 数据量 | 预测质量 | 说明 |
|--------|---------|------|
| **0-1天** | ❌ 不可用 | 数据不足，无法识别模式 |
| **2-3天** | ⚠️ 低 | 可以看到初步模式，但不可靠 |
| **7天** | ✅ 中等 | 一周数据，模式基本稳定 |
| **14天** | ✅✅ 良好 | **推荐！**足够识别工作日/周末差异 |
| **30天+** | ✅✅✅ 优秀 | 高置信度，能识别月度规律 |

---

## 🎯 实际使用效果

### 场景A: 工作日上午（高置信度）

```
当前时间: 周三上午10:30
历史数据: 过去14天，每个工作日上午10点，你都在坐姿工作

预测结果:
  应该: 🪑 坐姿 (置信度: 92%)
  实际: 🪑 坐姿
  建议: ✅ 您的当前状态符合日常习惯，保持良好！
```

### 场景B: 午休时间（中等置信度）

```
当前时间: 周三中午12:15
历史数据: 过去14天，午休时间你有时坐着吃饭(40%)，有时站着活动(35%)，有时躺着休息(25%)

预测结果:
  应该: 🪑 坐姿 (置信度: 40%)  # 最常见但不绝对
  实际: 🧍 站立
  建议: ✅ 您的当前状态与习惯相近  # 因为置信度不高，不强烈建议改变
```

### 场景C: 下午休息时间（高置信度不匹配）

```
当前时间: 周三下午3:00
历史数据: 过去14天，每天下午3点你都会站立活动15分钟

预测结果:
  应该: 🧍 站立 (置信度: 85%)
  实际: 🪑 坐姿
  建议: 💡 根据您的习惯，现在通常是站立活动的时间（85%概率），建议起身走动一下
  优先级: HIGH  # 因为置信度高且状态不匹配
```

---

## 🆚 与简单久坐提醒的区别

### 传统久坐提醒
```
坐了30分钟 → 提醒起身
坐了30分钟 → 提醒起身
坐了30分钟 → 提醒起身
（无论什么时间，只要坐够30分钟就提醒）
```

### 智能行为预测
```
上午10点（工作时间）:
  历史: 通常坐着工作
  当前: 坐着
  建议: ✅ 符合习惯，继续工作

下午3点（休息时间）:
  历史: 通常站立活动
  当前: 还在坐着
  建议: 💡 该站起来活动了！(即使只坐了20分钟)

晚上8点（休闲时间）:
  历史: 通常躺着看视频
  当前: 还在坐着
  建议: 💡 该躺下休息了
```

**关键差异**: 不是机械地按时间提醒，而是**学习你的习惯，在你偏离习惯时才提醒**！

---

## 🔮 未来增强方向

### 1. 机器学习模型
```python
# 使用Prophet进行时间序列预测
from fbprophet import Prophet

model = Prophet()
model.fit(historical_data)
forecast = model.predict(future_dates)
```

### 2. 上下文感知
```python
# 整合日历
if calendar.has_meeting(current_time):
    # 会议期间，预测更可能是坐姿
    predicted_state = 'sitting'
```

### 3. 自适应学习
```python
# 最近数据权重更高
weight = lambda days_ago: 1.0 / (1 + days_ago * 0.1)
```

### 4. 活动推荐
```python
if predicted == 'standing' and actual == 'sitting':
    activities = [
        '到饮水机接杯水',
        '走到窗边看看远方',
        '做5分钟拉伸运动',
        '去洗手间洗把脸'
    ]
    suggest_random_activity()
```

---

## ✨ 总结

**这才是真正的"智能预测"！**

不是简单的统计平均值，而是：
- ✅ 学习你的日常作息规律
- ✅ 识别每个时段的典型行为
- ✅ 预测当前时间应该做什么
- ✅ 对比预测和实际，给出智能建议
- ✅ 根据置信度调整建议优先级

**核心理念**: 让AI学会你的习惯，成为你的**个性化健康助手**，而不是机械的定时提醒器！

---

## 📞 使用帮助

### 快速测试

1. 运行主程序记录数据（至少3-7天）
```bash
python main.py --config config/config_gpu.yaml
```

2. 启动Web Dashboard
```bash
python web_dashboard.py
```

3. 打开浏览器查看
```
http://127.0.0.1:5000
```

4. 查看 **🧠 智能行为预测** 卡片，看系统是否学会了你的习惯！

---

**现在就试试，看看AI能否准确预测你此刻应该做什么！** 🎯
