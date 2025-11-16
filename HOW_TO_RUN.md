# 如何运行久坐提醒系统

## 快速开始

### 1. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt
```

**注意事项：**
- Python版本需要 **3.8+**
- Windows下某些GPU包（mmcv-full/mmpose）可能安装失败，不影响CPU运行
- 如果安装ultralytics失败，可以单独安装: `pip install ultralytics`

### 2. 准备摄像头

确保你的电脑有可用的摄像头：
- **笔记本内置摄像头**: 通常是 `/dev/video0` (Linux) 或 `0` (Windows)
- **外接USB摄像头**: 可能是 `/dev/video1` 或 `1`

### 3. 运行应用

```bash
# 方式1: 使用GPU配置（如果有CUDA GPU）
python main.py --config config/config_gpu.yaml

# 方式2: 使用CPU配置（通用，但较慢）
python main.py --config config/config_cpu.yaml

# 方式3: 启用调试模式（显示骨骼点、角度等）
python main.py --config config/config_gpu.yaml --debug

# 方式4: 不显示可视化窗口（仅后台运行）
python main.py --config config/config_gpu.yaml --no-vis
```

### 4. 查看SessionTracker效果

运行后，你会看到：

**启动信息：**
```
[初始化] 加载配置...
[初始化] 加载姿态估计器...
[初始化] 加载ROI管理器...
[初始化] 创建事件记录器...
[初始化] 创建状态机...
[BehaviorStateMachine] SessionTracker已启用  ← 看到这行说明SessionTracker已工作
[初始化] 打开摄像头...
```

**运行时：**
- 摄像头窗口会显示实时画面
- 左上角显示当前状态（Sitting/Standing/Lying）
- SessionTracker在后台自动记录每个状态的持续时间

**数据存储位置：**
```
data/database.db  ← SQLite数据库，包含所有会话记录
```

---

## 查看SessionTracker记录的数据

### 方式1: 使用Python脚本查询

创建一个简单的查询脚本：

```python
# query_stats.py
from src.storage.database import Database
from src.analytics.session_tracker import SessionTracker

# 连接数据库
db = Database('data/database.db')
tracker = SessionTracker(database=db)

# 查看今日统计
stats = tracker.get_today_statistics()
print(f"📊 今日统计 ({stats['date']}):")
print(f"  坐姿: {stats['sitting_duration']/3600:.2f} 小时")
print(f"  站立: {stats['standing_duration']/3600:.2f} 小时")
print(f"  躺卧: {stats['lying_duration']/3600:.2f} 小时")
print(f"  总会话数: {stats['total_sessions']}")

# 查看坐姿详细统计
sitting_stats = tracker.get_sitting_statistics()
print(f"\n💺 坐姿详细统计:")
print(f"  总时长: {sitting_stats['total_duration_minutes']:.1f} 分钟")
print(f"  会话次数: {sitting_stats['session_count']}")
print(f"  平均每次: {sitting_stats['average_session_duration']/60:.1f} 分钟")
print(f"  最长一次: {sitting_stats['longest_session']/60:.1f} 分钟")

# 检查是否久坐
if tracker.check_prolonged_sitting(threshold_minutes=30):
    current_duration = tracker.get_current_duration() / 60
    print(f"\n⚠️  久坐警告: 已持续坐姿 {current_duration:.0f} 分钟!")
```

运行：
```bash
python query_stats.py
```

### 方式2: 直接查询数据库

```bash
# 查看最近10条会话记录
sqlite3 data/database.db "SELECT datetime(timestamp, 'unixepoch', 'localtime') as time, state, duration/60 as duration_min, zone FROM state_history ORDER BY id DESC LIMIT 10;"
```

### 方式3: 使用Python REPL

```python
python3

>>> from src.storage.database import Database
>>> from src.analytics.session_tracker import SessionTracker
>>>
>>> db = Database('data/database.db')
>>> tracker = SessionTracker(database=db)
>>>
>>> # 查看今日统计
>>> tracker.get_today_statistics()
>>>
>>> # 查看本周统计
>>> tracker.get_weekly_statistics()
```

---

## 配置文件说明

### `config/config_gpu.yaml` (推荐)
- 使用GPU加速
- YOLOv8人体检测 + MediaPipe姿态估计
- FPS: ~15-20（足够检测坐/站/躺）

### `config/config_cpu.yaml` (备用)
- 纯CPU运行
- MediaPipe姿态估计
- FPS: ~10-15（可能较慢）

你可以根据需要修改配置：

```yaml
camera:
  source: 0  # 修改为你的摄像头编号
  resolution: [1920, 1080]  # 修改为你想要的分辨率

behavior:
  thresholds:
    min_confidence: 0.5  # 姿态估计置信度阈值
    standing_hip_knee_angle: 150  # 站立判定角度

# SessionTracker配置（将来可以添加）
session_tracking:
  prolonged_sitting_threshold: 30  # 久坐阈值（分钟）
  auto_save_interval: 60  # 自动保存间隔（秒）
```

---

## 常见问题

### 1. 摄像头打不开
```
RuntimeError: 无法打开摄像头: 0
```

**解决方案：**
- Linux: 检查摄像头设备 `ls /dev/video*`，修改config中的`camera.source`
- Windows: 尝试修改为 `1` 或 `2`（如果有多个摄像头）
- 检查摄像头权限

### 2. 依赖安装失败

**mmcv-full/mmpose安装失败（Windows常见）：**
```bash
# 这些包只在GPU模式下用到，CPU模式不需要
# 可以注释掉requirements.txt中的这两行
```

**PyTorch安装失败：**
```bash
# 访问 https://pytorch.org 获取适合你系统的安装命令
# 例如（CPU版本）:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. 运行很慢/卡顿

**解决方案：**
- 降低分辨率: 修改config中的 `camera.resolution` 为 `[1280, 720]` 或 `[640, 480]`
- 使用CPU配置: `--config config/config_cpu.yaml`
- 关闭调试模式: 不要使用 `--debug` 参数

### 4. SessionTracker没有启用

**检查启动日志中是否有：**
```
[BehaviorStateMachine] SessionTracker已启用
```

**如果没有：**
- 检查 `src/analytics/session_tracker.py` 是否存在
- 检查导入是否成功（不应该有ImportError）

### 5. 查看实时统计

**目前SessionTracker在后台运行，界面上还没有显示。**

**查看实时数据：**
```python
# 在另一个终端运行
python query_stats.py
```

**下一步开发：**我们将添加：
- 屏幕上显示当前会话时长
- 显示今日统计面板
- 久坐超过30分钟时红色警告

---

## 键盘控制

运行时支持的按键：

- **q**: 退出程序
- **空格**: 暂停/继续
- **s**: 截图保存
- **d**: 切换调试模式
- **r**: 重置ROI区域

---

## 数据存储

### 数据库位置
```
data/database.db
```

### 表结构
```sql
state_history:
  - id: 记录ID
  - timestamp: 结束时间戳
  - state: 状态 (sitting/standing/lying/sleeping)
  - zone: 区域 (bed/chair/desk等)
  - duration: 持续时长（秒）
  - created_at: 记录创建时间
```

### 备份数据
```bash
# 备份整个数据库
cp data/database.db data/database_backup_$(date +%Y%m%d).db

# 导出为CSV
sqlite3 -header -csv data/database.db "SELECT * FROM state_history;" > sessions.csv
```

---

## 完整启动流程示例

```bash
# 1. 克隆项目（如果还没有）
cd ~/Camera

# 2. 安装依赖
pip install -r requirements.txt

# 3. 确认摄像头可用
# Linux:
ls /dev/video*
# Windows: 打开"相机"应用测试

# 4. 运行（GPU模式）
python main.py --config config/config_gpu.yaml

# 5. 观察启动日志，确认SessionTracker已启用
# [BehaviorStateMachine] SessionTracker已启用  ← 这行很重要

# 6. 在摄像头前坐下/站立，让系统检测你的姿态

# 7. 另开一个终端，查看统计数据
python query_stats.py

# 8. 按'q'键退出
```

---

## 下一步计划

当前SessionTracker已在后台工作，正在记录你的活动数据。

**即将添加的功能：**
1. ✅ SessionTracker后台运行 (已完成)
2. ⏳ 屏幕显示当前会话时长 (开发中)
3. ⏳ 显示今日统计面板 (开发中)
4. ⏳ 久坐警告提示 (开发中)
5. ⏳ 周报/月报生成 (计划中)
6. ⏳ 导出Excel报表 (计划中)

现在就开始运行，让SessionTracker开始为你记录活动数据吧！📊
