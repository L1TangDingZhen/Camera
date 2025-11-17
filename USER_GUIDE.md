# Life Tracker 用户手册

欢迎使用Life Tracker！本手册将帮助你快速上手，了解系统的核心功能和基本使用方法。

## 📖 目录

- [项目介绍](#项目介绍)
- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [当前默认配置](#当前默认配置)
- [快速开始](#快速开始)
- [基础使用流程](#基础使用流程)
- [常用命令清单](#常用命令清单)
- [进阶使用](#进阶使用)
- [故障排查](#故障排查)
- [未来发展方向](#未来发展方向)

---

## 项目介绍

**Life Tracker** 是一个基于计算机视觉的日常活动分析系统，能够自动识别和记录你的坐、站、躺等姿态状态，帮助你了解自己的作息规律。

### 主要特点

- ✅ **自动识别**：无需手动记录，摄像头自动识别姿态
- ✅ **隐私保护**：所有数据本地处理，不上传云端
- ✅ **轻量高效**：支持CPU运行，也可以使用GPU加速
- ✅ **开箱即用**：默认配置无需训练，立即可用
- ✅ **高度可扩展**：支持多种模型组合，满足不同精度需求

### 适用场景

- 📊 **健康监测**：记录久坐时长，提醒适时运动
- 🛋️ **作息分析**：了解每天坐、站、躺的时间分布
- 💤 **睡眠追踪**：自动识别睡眠时段
- 🏠 **智能家居**：根据状态自动控制灯光、空调等设备

---

## 核心功能

### 1. 姿态识别

系统可以识别以下三种基本姿态：

| 姿态 | 描述 | 判断标准 |
|------|------|---------|
| 🪑 **坐** (Sitting) | 坐在椅子或床上 | 臀部高度适中、膝盖弯曲 |
| 🧍 **站** (Standing) | 站立状态 | 身体直立、膝盖伸直 |
| 🛌 **躺** (Lying) | 躺在床上或沙发 | 身体与水平面角度小 |

### 2. 事件追踪

- 进入状态：当姿态稳定持续一定时间后记录
- 离开状态：检测到姿态变化时记录
- 持续时长：自动计算每个状态的持续时间

### 3. 数据存储

- 本地SQLite数据库存储
- 自动备份（可配置）
- 支持数据导出和分析

### 4. 可视化界面

- 实时摄像头画面
- 当前状态显示
- 关键点可视化（可选）
- Web界面（可选）

---

## 系统架构

Life Tracker采用**三层架构**，每一层都可以独立配置：

```
┌──────────────────────────────────────────────┐
│           决策层 (Decision Layer)             │
│  何时输出结果 / 如何减少误报                    │
│  默认：Simple防抖                              │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│          分类层 (Classifier Layer)            │
│  判断当前姿态：坐/站/躺                         │
│  默认：SVM分类器                               │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│       姿态估计层 (Pose Estimation Layer)      │
│  从图像中提取人体关键点                         │
│  默认：MediaPipe                               │
└──────────────────────────────────────────────┘
```

---

## 当前默认配置

Life Tracker的默认配置经过优化，**开箱即用，无需额外训练**：

### 🎯 默认方案

| 层级 | 使用模型 | 特点 | 硬件要求 |
|------|---------|------|---------|
| 姿态估计 | **MediaPipe** | CPU友好、跨平台 | CPU |
| 分类器 | **SVM** | 快速、稳定、已训练 | CPU |
| 决策策略 | **Simple防抖** | 简单有效 | - |

### 📊 默认性能

- **精度**：90-95%（基于标准环境）
- **延迟**：约50ms/帧
- **资源占用**：中等（约300MB内存）
- **设备要求**：普通PC即可，无需GPU

### ✨ 默认方案的优点

- ✅ **即插即用**：无需训练，直接运行
- ✅ **跨平台**：Windows/Linux/macOS都支持
- ✅ **稳定可靠**：经过大量测试
- ✅ **资源友好**：CPU运行，笔记本也能用

### ⚠️ 适用场景

默认配置适合：
- 初次使用，快速体验
- 日常家用，精度要求不高
- CPU-only环境
- 标准光照和角度

如果你需要更高精度或更快速度，请参考 [进阶使用](#进阶使用)。

---

## 快速开始

### 前置要求

1. **Python 3.8+**
2. **摄像头**（笔记本内置或外接USB摄像头）
3. **基础依赖**（自动安装）

### 三步上手

#### 步骤1：安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/Camera.git
cd Camera

# 安装核心依赖
pip install -r requirements.txt
```

**安装内容**：
- OpenCV（图像处理）
- MediaPipe（姿态估计）
- PyTorch（机器学习框架）
- YOLOv8（人体检测）
- 其他工具库

#### 步骤2：运行系统

```bash
# 使用默认配置运行（GPU模式）
python main.py --config config/config_gpu.yaml

# 或使用CPU模式（更慢但兼容性好）
python main.py --config config/config_cpu.yaml
```

#### 步骤3：观察结果

系统启动后，你会看到：

```
[初始化] 加载人体检测器...
[初始化] 加载姿态估计器...
[MediaPipe] 初始化完成 ✓
[BehaviorStateMachine] 使用SVM分类器
[BehaviorStateMachine] 使用简单防抖决策
[系统] 启动成功！按 'q' 退出
```

**实时画面**显示：
- 摄像头视频流
- 当前检测到的姿态（Sitting/Standing/Lying）
- 姿态持续时长
- 人体检测框和关键点（可选）

---

## 基础使用流程

### 1. 首次运行

第一次运行系统，建议：

```bash
# 使用GPU配置（如果有显卡）
python main.py --config config/config_gpu.yaml

# 系统会自动：
# 1. 打开摄像头
# 2. 检测人体
# 3. 识别姿态
# 4. 显示实时结果
```

**确认系统正常工作**：
- ✅ 摄像头画面正常显示
- ✅ 能检测到人体（绿色框）
- ✅ 姿态识别准确（尝试坐下、站立、躺下）

### 2. 收集训练数据（可选）

如果默认模型精度不够，可以收集自己的数据：

```bash
# 收集坐姿数据（60秒）
python collect_data.py --label sitting --duration 60

# 收集站姿数据
python collect_data.py --label standing --duration 60

# 收集躺姿数据
python collect_data.py --label lying --duration 60
```

**收集建议**：
- 每个姿态至少收集30-60秒
- 包含不同角度、光照
- 模拟真实使用场景

收集完成后，数据保存在 `training_data/` 目录。

### 3. 训练模型（可选）

如果收集了自己的数据，可以重新训练：

```bash
# 训练SVM模型
python train_svm.py --data training_data

# 新模型会保存到：models/pose_classifier_svm.pkl
# 下次运行自动使用新模型
```

### 4. 查看分析结果

系统运行期间，所有数据保存在数据库：

```bash
# 查看数据库
sqlite3 data/database.db

# 查询今天的活动记录
SELECT * FROM events WHERE date(timestamp) = date('now');

# 统计今天坐了多久
SELECT SUM(duration) FROM events
WHERE state = 'sitting' AND date(timestamp) = date('now');
```

---

## 常用命令清单

### 运行系统

```bash
# GPU模式（推荐，如果有显卡）
python main.py --config config/config_gpu.yaml

# CPU模式（兼容性好）
python main.py --config config/config_cpu.yaml

# 指定摄像头设备
python main.py --config config/config_gpu.yaml --camera 1
```

### 数据收集

```bash
# 收集训练数据
python collect_data.py --label [sitting|standing|lying] --duration 60

# 查看已收集数据
ls training_data/
cat training_data/sitting.json | jq length  # 查看样本数量
```

### 模型训练

```bash
# 训练SVM（默认分类器）
python train_svm.py --data training_data

# 查看训练好的模型
ls models/
```

### 测试和调试

```bash
# 快速测试
python test_quick.py

# 鲁棒性测试（测试模型准确率）
python test_robustness.py --label sitting --duration 30

# 查看日志
tail -f logs/app.log
```

### 配置和维护

```bash
# 查看配置文件
cat config/config_gpu.yaml

# 备份数据库
cp data/database.db data/database_backup.db

# 清理日志
rm logs/*.log
```

---

## 进阶使用

默认配置已经能满足大多数需求，但如果你想要：

- 🚀 **更快的速度**（GPU加速）
- 📈 **更高的精度**（深度学习模型）
- 🎯 **更少的误报**（智能决策）
- 🔧 **自定义配置**（适应特殊环境）

### 其他可用方案

Life Tracker支持多种模型组合方案：

| 方案 | 姿态估计 | 分类器 | 精度提升 | 速度提升 | 需要GPU | 需要训练 |
|------|---------|--------|---------|---------|---------|---------|
| **默认** | MediaPipe | SVM | 基线 | 基线 | ❌ | ❌ |
| **GPU加速** | RTMPose | SVM | - | **4x** ⚡ | ✅ | ❌ |
| **高精度** | RTMPose | MLP | **+5%** | 4x | ✅ | ✅ |
| **最高精度** | RTMPose | Ensemble | **+8%** | 3x | ✅ | ✅ |

### 📚 详细技术文档

想要了解如何切换方案、训练模型、调优参数？

➡️ **请参阅**：[DL_RL_TECHNICAL_GUIDE.md](DL_RL_TECHNICAL_GUIDE.md)

技术文档包含：
- 所有方案的详细对比
- 逐步切换指南
- 完整训练流程
- 模型性能基准
- 高级配置选项

---

## 故障排查

### 问题1：摄像头打不开

**症状**：
```
Error: Cannot open camera
```

**解决方案**：
```bash
# 检查摄像头设备
ls /dev/video*  # Linux
# 或在Windows设备管理器中查看

# 尝试其他设备ID
python main.py --config config/config_gpu.yaml --camera 1
python main.py --config config/config_gpu.yaml --camera 2
```

### 问题2：画面显示太小/模糊

**症状**：窗口很小，字体看不清

**解决方案**：
```yaml
# 修改配置文件：config/config_gpu.yaml
camera:
  resolution: [1280, 720]  # 改成更高分辨率
  fps: 30  # 提高帧率
```

### 问题3：识别不准确

**可能原因**：
- 光照不好
- 摄像头角度不对
- 姿态不典型

**解决方案**：

**方案A**：调整环境
- 改善光照
- 调整摄像头位置（建议正面、距离2-3米）

**方案B**：收集训练数据
```bash
# 在你的实际环境中收集数据
python collect_data.py --label sitting --duration 120
python collect_data.py --label standing --duration 120
python collect_data.py --label lying --duration 120

# 重新训练
python train_svm.py --data training_data
```

**方案C**：切换到更高精度模型
参考 [DL_RL_TECHNICAL_GUIDE.md](DL_RL_TECHNICAL_GUIDE.md)

### 问题4：程序运行慢/卡顿

**症状**：FPS很低，画面卡顿

**解决方案**：

```yaml
# 方法1：降低分辨率
camera:
  resolution: [640, 480]  # 从1280x720降到640x480
  fps: 15  # 降低帧率

# 方法2：跳帧处理
inference:
  skip_frames: 2  # 每2帧处理一次
  detection_interval: 3  # 每3帧检测一次人体
```

或者切换到CPU配置：
```bash
python main.py --config config/config_cpu.yaml
```

### 问题5：找不到训练数据

**症状**：
```
FileNotFoundError: training_data/sitting.json not found
```

**解决方案**：
```bash
# 检查数据文件
ls training_data/

# 如果没有，先收集数据
python collect_data.py --label sitting --duration 60
```

### 更多问题？

- 查看日志：`cat logs/app.log`
- 查看Issues：https://github.com/yourusername/Camera/issues
- 阅读技术文档：[DL_RL_TECHNICAL_GUIDE.md](DL_RL_TECHNICAL_GUIDE.md)

---

## 未来发展方向

Life Tracker正在持续改进，未来计划：

### 短期计划

- ✅ **RTMPose集成**（GPU加速姿态估计）- 已完成
- ✅ **深度学习分类器**（MLP/LSTM/Transformer）- 已完成
- ✅ **RL集成**（Ensemble融合 + 自适应决策）- 已完成
- 🔄 **Jetson优化**（边缘设备部署）- 进行中
- 📱 **移动端App**（Android/iOS）- 计划中

### 长期愿景

- 🎯 **更多姿态识别**（工作、休息、运动等）
- 🏠 **智能家居集成**（Home Assistant、HomeKit）
- 📊 **高级数据分析**（趋势预测、健康建议）
- 🌐 **多人追踪**（家庭成员独立追踪）
- 🤖 **AI助手集成**（语音交互、智能提醒）

### 贡献

欢迎贡献代码、提出建议或报告问题：
- GitHub: https://github.com/yourusername/Camera
- Issues: https://github.com/yourusername/Camera/issues
- Discussions: https://github.com/yourusername/Camera/discussions

---

## 文档导航

- 📘 **用户手册**（当前）：快速上手和基础使用
- 📗 **技术文档**：[DL_RL_TECHNICAL_GUIDE.md](DL_RL_TECHNICAL_GUIDE.md) - 模型训练和高级配置
- 📙 **RTMPose安装**：[INSTALL_RTMPOSE.md](INSTALL_RTMPOSE.md) - GPU加速姿态估计
- 📕 **技术对比**：[RTMPOSE_TECHNICAL_COMPARISON.md](RTMPOSE_TECHNICAL_COMPARISON.md) - 性能基准测试

---

**开始使用Life Tracker，了解你的每一天！** 🚀
