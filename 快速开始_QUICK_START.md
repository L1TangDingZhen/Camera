# 久坐提醒系统 - 快速开始指南

## ✅ 已完成的功能

### 1. SessionTracker (时长统计) ✅
- 自动追踪坐/站/躺时长
- 今日/本周统计
- 久坐检测（>30分钟）
- 数据持久化到SQLite

### 2. Web Dashboard (数据可视化) ✅
- 实时状态监控
- 交互式图表（饼图、折线图）
- 响应式设计（手机/平板/桌面）
- 自动刷新（每30秒）

### 3. AI Prediction (智能预测) ✅
- 预测下次坐姿时长
- 推荐最佳提醒间隔
- 异常行为检测
- 个性化建议

---

## 🚀 三步快速开始

### Step 1: 运行主程序（记录数据）

```bash
# 安装依赖
pip install -r requirements.txt

# 启动摄像头监测
python main.py --config config/config_gpu.yaml

# 让它运行一段时间，记录你的活动
# 建议至少运行30分钟-1小时，以便有足够数据
```

**这一步会做什么？**
- 打开摄像头
- 检测你的姿态（坐/站/躺）
- 自动记录每个状态的时长
- 保存数据到 `data/database.db`

---

### Step 2: 查看统计数据（命令行）

```bash
# 在另一个终端运行
python query_stats.py
```

**你会看到：**
```
📊 SessionTracker 统计数据
============================================================

【今日统计】
日期: 2025-11-08
总会话数: 15

  🪑 坐姿: 2h 30m (2.50h)
  🧍 站立: 1h 15m (1.25h)
  🛏️  躺卧: 0h 20m (0.33h)

【当前会话】
  状态: sitting
  时长: 5m
  区域: chair

【坐姿详细统计】
  总时长: 150.0 分钟 (2.50h)
  会话次数: 8
  平均每次: 18.8 分钟
  最长一次: 45.0 分钟

【本周统计】
  周期: 2025-11-04 到 2025-11-10
  ...
```

---

### Step 3: 启动Web Dashboard（可视化界面）

```bash
# 启动Web服务器
python web_dashboard.py

# 打开浏览器访问
# http://127.0.0.1:5000
```

**你会看到：**
- 📍 **当前状态**: 实时显示正在坐/站/躺
- 📊 **今日统计**: 四个卡片显示坐姿、站立、躺卧时长和会话数
- 💺 **坐姿详细**: 总时长、会话次数、平均时长、最长时长
- 📊 **活动饼图**: 可视化今日时间分配
- 📈 **本周趋势**: 折线图显示过去7天的变化
- 🔮 **智能预测**: AI预测下次坐姿时长和建议提醒间隔
- 🔍 **异常检测**: 自动识别今日坐姿是否异常

**久坐警告：**
- 当你坐姿超过30分钟，顶部会显示红色警告横幅！

---

## 📂 项目文件说明

```
Camera/
├── main.py                          # 主程序（摄像头监测）
├── query_stats.py                   # 命令行查询统计
├── web_dashboard.py                 # Web可视化服务器
├── config/
│   ├── config_gpu.yaml              # GPU配置（推荐）
│   └── config_cpu.yaml              # CPU配置
├── data/
│   └── database.db                  # SQLite数据库（自动创建）
├── src/
│   ├── analytics/
│   │   ├── session_tracker.py       # 时长统计模块
│   │   └── predictor.py             # AI预测模块
│   ├── state/
│   │   └── behavior_state.py        # 状态机（集成SessionTracker）
│   └── storage/
│       ├── database.py              # 数据库操作
│       └── event_logger.py          # 事件记录
├── templates/
│   └── dashboard.html               # Web界面HTML
├── static/
│   ├── css/style.css                # 样式
│   └── js/dashboard.js              # 前端逻辑
├── 如何运行.md                      # 详细运行指南
├── WEB_DASHBOARD_使用指南.md        # Web功能说明
└── SESSION_TRACKER_IMPLEMENTATION.md # 技术文档
```

---

## 🎯 使用场景

### 场景1: 日常监测
```bash
# 早上开机后启动
python main.py --config config/config_gpu.yaml

# 全天运行，自动记录你的活动
# 随时打开 http://127.0.0.1:5000 查看数据
```

### 场景2: 每日回顾
```bash
# 晚上下班后查看今日统计
python query_stats.py

# 或打开Web Dashboard查看详细图表和预测
```

### 场景3: 久坐提醒
```bash
# 启动主程序
python main.py --config config/config_gpu.yaml

# 同时启动Web Dashboard
python web_dashboard.py

# 打开浏览器，当坐姿超过30分钟会自动显示警告
```

---

## 🔧 常见问题快速解决

### Q1: 第一次运行query_stats.py显示所有数据为0？
**A**: 正常！需要先运行 `main.py` 记录一些活动数据。

### Q2: Web Dashboard预测显示"累积更多数据后将提供预测"？
**A**: 预测需要至少3次坐姿记录。建议使用3-7天后再查看预测功能。

### Q3: main.py报错 "无法打开摄像头"？
**A**: 检查：
1. 摄像头是否被其他程序占用
2. 修改 `config/config_gpu.yaml` 中的 `camera.source` (0, 1, 2...)

### Q4: Web Dashboard无法访问？
**A**: 确认：
1. `python web_dashboard.py` 正在运行
2. 浏览器访问 http://127.0.0.1:5000 (注意端口号)

### Q5: 想在手机上查看数据？
**A**:
```bash
# 启动时允许局域网访问
python web_dashboard.py --host 0.0.0.0

# 手机浏览器访问
http://[你的电脑IP]:5000
# 例如: http://192.168.1.100:5000
```

---

## 📊 API使用示例

### Python调用
```python
from src.storage.database import Database
from src.analytics.session_tracker import SessionTracker
from src.analytics.predictor import SittingPredictor

# 创建实例
db = Database('data/database.db')
tracker = SessionTracker(database=db)
predictor = SittingPredictor(database=db)

# 获取今日统计
stats = tracker.get_today_statistics()
print(f"今日坐姿: {stats['sitting_duration']/3600:.1f}小时")

# 预测下次坐姿时长
prediction = predictor.predict_next_sitting_duration()
print(f"预测下次坐姿: {prediction['predicted_duration_minutes']}分钟")
print(f"置信度: {prediction['confidence']*100:.0f}%")

# 异常检测
anomaly = predictor.detect_anomaly()
if anomaly['is_anomaly']:
    print(f"⚠️ {anomaly['message']}")
```

### JavaScript调用（浏览器）
```javascript
// 获取今日统计
fetch('http://127.0.0.1:5000/api/stats/today')
  .then(res => res.json())
  .then(data => {
    console.log('今日坐姿:', data.data.sitting_duration/3600, '小时');
  });

// 获取预测
fetch('http://127.0.0.1:5000/api/prediction/next_sitting')
  .then(res => res.json())
  .then(data => {
    console.log('预测时长:', data.data.predicted_duration_minutes, '分钟');
  });
```

### curl命令行调用
```bash
# 今日统计
curl http://127.0.0.1:5000/api/stats/today | jq

# 预测
curl http://127.0.0.1:5000/api/prediction/next_sitting | jq

# 异常检测
curl http://127.0.0.1:5000/api/prediction/anomaly | jq
```

---

## 🎨 截图预览

### Web Dashboard界面

```
┌─────────────────────────────────────────────────────────────┐
│  🪑 久坐提醒系统 - 数据仪表盘                                  │
│  实时监测您的活动数据，保持健康生活方式                         │
└─────────────────────────────────────────────────────────────┘

⚠️ 久坐警告
   您已持续坐姿35分钟，建议起身活动！

┌─────────────────────────────────────────────────────────────┐
│ 📍 当前状态                                                   │
│ 当前状态: 🪑 坐姿    持续时长: 15分钟    所在区域: chair        │
└─────────────────────────────────────────────────────────────┘

┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ 🪑       │ │ 🧍       │ │ 🛏️       │ │ 📊       │
│ 坐姿时长  │ │ 站立时长  │ │ 躺卧时长  │ │ 会话次数  │
│ 2.5小时  │ │ 1.2小时  │ │ 0.3小时  │ │ 15       │
└──────────┘ └──────────┘ └──────────┘ └──────────┘

┌─────────────────────────────────────────────────────────────┐
│ 💺 坐姿详细统计                                               │
│ 总时长: 2.5h  会话次数: 8  平均时长: 18.8m  最长一次: 45.0m   │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐  ┌──────────────────────────────────────┐
│ 📊 今日活动分布   │  │ 📈 本周活动趋势                       │
│  [饼图]          │  │  [折线图]                            │
└──────────────────┘  └──────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 🔮 智能预测                                                   │
│ 🎯 预测下次坐姿时长: 45分钟  ⏰ 建议提醒间隔: 25分钟           │
│ 置信度: 75%                                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 🔍 今日异常检测                                               │
│ ✅ 今日坐姿时长正常                                           │
│ 今日: 2.5h  历史平均: 2.8h  偏差: -10.7%                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔥 高级功能

### 1. 自定义提醒阈值
编辑 `config/config_gpu.yaml`:
```yaml
session_tracking:
  prolonged_sitting_threshold: 25  # 改为25分钟
```

### 2. 导出数据到CSV
```bash
sqlite3 -header -csv data/database.db \
  "SELECT * FROM state_history;" > my_activity_data.csv
```

### 3. 局域网多设备访问
```bash
# 服务器端
python web_dashboard.py --host 0.0.0.0 --port 5000

# 同一WiFi下的其他设备访问
# http://[电脑IP]:5000
```

### 4. 生产环境部署
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_dashboard:app
```

---

## 📚 相关文档

- **如何运行.md** - 完整的安装和运行指南
- **WEB_DASHBOARD_使用指南.md** - Web功能详细说明
- **SESSION_TRACKER_IMPLEMENTATION.md** - 技术实现文档

---

## ✨ 总结

你现在拥有一个**完整的久坐提醒系统**：

1. ✅ **数据采集** - main.py 自动记录活动
2. ✅ **数据存储** - SQLite 持久化
3. ✅ **数据查询** - query_stats.py 命令行查询
4. ✅ **数据可视化** - Web Dashboard 图表展示
5. ✅ **智能预测** - AI 预测和建议
6. ✅ **异常检测** - 自动识别异常行为
7. ✅ **久坐警告** - 实时提醒

**现在就开始使用吧！** 🎉

```bash
# Terminal 1: 启动监测
python main.py --config config/config_gpu.yaml

# Terminal 2: 启动Web Dashboard
python web_dashboard.py

# Browser: 打开浏览器
# http://127.0.0.1:5000
```

**祝你拥有健康的工作习惯！** 💪
