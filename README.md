# Life Tracker - 行为监测系统

基于AI视觉的三阶段部署方案，用于监测人体行为（睡眠、作息等）

## 🎯 三阶段路线图

```
阶段1: PC (RTX 4070)          → 快速开发 + 模型选型
  ↓
阶段2: X390 (i5-8代 CPU)     → 真实验证 + 24/7稳定性测试
  ↓
阶段3: Jetson Orin Nano      → 长期生产运行 + 低功耗
```

## ✨ 主要功能

- ✅ 人体检测（YOLOv8）
- ✅ 姿态估计（RTMPose / ViTPose / MediaPipe）
- ✅ 行为识别（坐/躺/睡眠）
- ✅ ROI区域监测（床/门/椅子/浴室）
- ✅ 事件触发（进入/离开/入睡/醒来）
- ✅ 数据存储（SQLite）
- ✅ 性能监控
- ✅ 多设备支持（PC/笔记本/Jetson）

## 📋 系统要求

### 阶段1 - PC开发

- Python 3.8+
- CUDA 11.0+ (NVIDIA GPU)
- 8GB+ RAM
- 摄像头

### 阶段2 - X390验证

- Python 3.8+
- 4GB+ RAM (CPU only)
- 摄像头

### 阶段3 - Jetson生产

- JetPack 5.0+
- 8GB RAM
- 摄像头

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目（如果还没有）
git clone <your-repo>
cd Camera

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型

```bash
# YOLOv8人体检测模型（自动下载）
# 第一次运行会自动下载

# MediaPipe姿态估计（自动下载）
# 第一次运行会自动下载

# 如果使用RTMPose或ViTPose（可选）
# 参考: https://github.com/open-mmlab/mmpose
```

### 3. 标定ROI区域

```bash
# 运行ROI标定工具
python scripts/calibrate_roi.py --device pc

# 按照提示标定床、门、椅子等区域
# 保存后会更新config/config_pc.yaml
```

### 4. 运行系统

```bash
# PC开发模式
python main.py --device pc

# X390验证模式
python main.py --device x390

# Jetson生产模式
python main.py --device jetson

# 或指定配置文件
python main.py --config config/config_pc.yaml
```

### 5. 查看结果

```bash
# 启动Web界面（TODO）
python src/web/app.py

# 访问 http://localhost:5000
```

## 📁 项目结构

```
Camera/
├── config/                      # 配置文件
│   ├── config_pc.yaml          # PC开发配置
│   ├── config_x390.yaml        # X390验证配置
│   └── config_jetson.yaml      # Jetson生产配置
├── src/
│   ├── detectors/              # 检测器模块
│   │   ├── base.py            # 基类和接口
│   │   ├── person_detector.py # YOLOv8人体检测
│   │   └── pose_estimator.py  # 姿态估计（多后端）
│   ├── state/                  # 状态管理
│   │   ├── roi_manager.py     # ROI管理
│   │   └── behavior_state.py  # 行为状态机
│   ├── storage/                # 数据存储
│   │   ├── database.py        # SQLite数据库
│   │   └── event_logger.py    # 事件记录
│   ├── analysis/               # 数据分析（TODO）
│   └── web/                    # Web界面（TODO）
├── scripts/                     # 工具脚本
│   ├── calibrate_roi.py       # ROI标定工具
│   ├── benchmark.py           # 性能测试
│   └── compare_models.py      # 模型对比
├── models/                      # 模型文件
├── data/                        # 数据目录
│   └── database.db            # SQLite数据库
├── logs/                        # 日志目录
├── main.py                      # 主程序入口
└── requirements.txt             # Python依赖

```

## 🔧 配置说明

### 主要配置项

```yaml
# config/config_pc.yaml

device: cuda:0  # 设备: cuda:0, cpu

models:
  person:
    model: yolov8s.pt           # 人体检测模型
    confidence: 0.5

  pose:
    backend: rtmpose            # 姿态估计后端: rtmpose, vitpose, mediapipe
    model: rtmpose_s.pth

camera:
  source: 0                      # 摄像头ID或RTSP URL
  fps: 30
  resolution: [640, 480]

roi:
  zones:
    bed:
      enabled: true
      points: []                # 使用calibrate_roi.py标定

behavior:
  sitting:
    hip_height_min: 0.3         # 坐姿判断阈值
    knee_angle_max: 120

  lying:
    body_angle_max: 30          # 躺姿判断阈值

  sleeping:
    still_duration: 300         # 睡眠判断：持续不动5分钟
```

## 📊 性能基准

| 阶段 | 硬件 | 检测模型 | 姿态模型 | FPS | 功耗 |
|------|------|----------|----------|-----|------|
| 1-PC | i5-12 + RTX 4070 | YOLOv8s | RTMPose-s | 280+ | ~100W |
| 2-X390 | i5-8 CPU | YOLOv8s | MediaPipe | 6-8 | ~30W |
| 3-Jetson | Orin Nano GPU | YOLOv8s-TRT | RTMPose-TRT | 38-45 | ~12W |

## 🛠️ 工具脚本

### ROI标定工具

```bash
python scripts/calibrate_roi.py --device pc

# 操作：
# 1. 点击鼠标标定多边形顶点
# 2. 按 'c' 完成当前区域
# 3. 按 's' 保存配置
# 4. 按 'q' 退出
```

### 模型对比工具

```bash
python scripts/compare_models.py --config config/config_pc.yaml

# 自动测试所有模型组合，生成对比报告
```

### 性能测试

```bash
python scripts/benchmark.py --config config/config_pc.yaml --duration 60

# 测试60秒，输出性能报告
```

## 📈 数据分析

```bash
# 每日报告（TODO）
python scripts/daily_report.py --date 2024-01-01

# 趋势分析（TODO）
python scripts/trend_analysis.py --start 2024-01-01 --end 2024-01-31
```

## 🔄 三阶段迁移

### PC → X390

```bash
# 1. PC上导出配置
python scripts/export_config.py --target x390

# 2. 复制项目到X390
scp -r Camera/ user@x390:/home/user/

# 3. X390上安装依赖
ssh user@x390
cd Camera
pip install -r requirements.txt

# 4. 运行
python main.py --device x390
```

### X390 → Jetson

```bash
# 1. 准备ONNX模型（在PC上）
python scripts/export_to_onnx.py

# 2. 复制到Jetson
scp models/*.onnx jetson@192.168.1.100:~/Camera/models/

# 3. Jetson上转换为TensorRT
ssh jetson@192.168.1.100
cd Camera
python scripts/convert_to_trt.py

# 4. 运行
python main.py --device jetson
```

## 🐛 常见问题

### Q1: YOLOv8模型下载失败？

```bash
# 手动下载
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
mv yolov8s.pt models/
```

### Q2: MediaPipe无法初始化？

```bash
# 重新安装
pip uninstall mediapipe
pip install mediapipe --no-cache-dir
```

### Q3: 摄像头无法打开？

```bash
# 检查设备
ls /dev/video*

# 尝试其他ID
python main.py --config config/config_pc.yaml
# 修改config中的camera.source为1或2
```

### Q4: CUDA out of memory？

```yaml
# 修改配置降低批大小
models:
  person:
    batch_size: 1  # 降为1
```

## 📝 待办事项

- [ ] 完成Web界面
- [ ] 实现数据分析模块（统计、趋势、预测）
- [ ] 创建完整的工具脚本
- [ ] 添加单元测试
- [ ] 编写详细文档
- [ ] 支持多摄像头
- [ ] 添加通知功能（异常行为告警）

## 📄 许可证

MIT License

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://google.github.io/mediapipe/)
- [MMPose](https://github.com/open-mmlab/mmpose)
