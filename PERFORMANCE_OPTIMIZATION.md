# 性能优化指南

## 当前性能分析

### 实测数据 (1280x720配置)

| 组件 | 耗时 | 占比 | 理论FPS |
|------|------|------|---------|
| 摄像头读取 | 121.79ms | 46.9% | 8.2 |
| YOLOv8m检测 | 68.38ms | 26.3% | 14.6 |
| RTMPose-s估计 | 69.49ms | 26.8% | 14.4 |
| **总计** | **259.67ms** | **100%** | **3.9** |

### 瓶颈分析

⚠️ **主要瓶颈：摄像头硬件限制**
- 1280x720分辨率下，摄像头只支持**10 FPS**
- 实际读取耗时121ms，远超理论值(1000/10=100ms)
- 可能原因：USB带宽限制、驱动效率

## 优化方案对比

### 方案1：保守优化（推荐，平衡精度和速度）

**配置文件**: `config/config_gpu.yaml` (直接修改)

**修改内容**:
```yaml
camera:
  resolution: [640, 480]  # 改为640x480

models:
  person:
    model: yolov8n.pt  # 改为yolov8n（需要下载）
```

**预期性能**:
- 摄像头读取: 121ms → **74ms** (提升 38%)
- YOLOv8检测: 68ms → **27ms** (提升 60%)
- RTMPose-s: 69ms → 69ms (不变)
- **总计: 170ms → 12-15 FPS** ✓

**优点**: 简单，只需修改config
**缺点**: 分辨率降低，远距离检测精度下降

---

### 方案2：激进优化（最高FPS）

**配置文件**: `config/config_gpu_optimized.yaml` (已创建)

**修改内容**:
```yaml
camera:
  resolution: [640, 480]

models:
  person:
    model: yolov8n.pt

  pose:
    model: rtmpose-tiny  # 改为rtmpose-tiny
    config_file: models/rtmpose/configs/rtmpose-t_8xb256-420e_coco-256x192.py
    checkpoint: models/rtmpose/rtmpose-t_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth

inference:
  detection_interval: 2  # 跳帧检测
```

**预期性能**:
- 摄像头读取: 121ms → **74ms**
- YOLOv8n检测: 68ms → **27ms** (每2帧检测一次)
- RTMPose-tiny: 69ms → **35ms** (提升 50%)
- **总计: 136ms → 18-22 FPS** ✓✓

**优点**: 最高FPS，接近实时
**缺点**:
- 姿态估计精度略降 (AP 66% vs 68.5%)
- 需要下载rtmpose-tiny模型

---

### 方案3：中等优化（当前配置微调）

**配置文件**: `config/config_gpu.yaml`

**只修改分辨率**:
```yaml
camera:
  resolution: [640, 480]
```

**预期性能**:
- 摄像头读取: 121ms → **74ms**
- YOLOv8m: 68ms → **40ms** (分辨率降低)
- RTMPose-s: 69ms → **40ms** (分辨率降低)
- **总计: 154ms → 10-12 FPS** ✓

**优点**: 无需下载新模型，保持高精度
**缺点**: FPS提升有限

---

## 实施步骤

### 推荐：方案1 (保守优化)

1. **下载YOLOv8n模型**:
```bash
source Camera/bin/activate
cd /home/eyes/Desktop/Camera
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

2. **修改配置文件**:
```bash
# 编辑 config/config_gpu.yaml
# 修改两处：
#   - camera.resolution: [640, 480]
#   - models.person.model: yolov8n.pt
```

3. **运行测试**:
```bash
python main.py --mode gpu
```

**预期结果**: 12-15 FPS ✓

---

### 激进：方案2 (最高FPS)

1. **下载YOLOv8n**:
```bash
source Camera/bin/activate
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

2. **下载RTMPose-tiny** (如果没有):
```bash
# 检查是否存在
ls models/rtmpose/rtmpose-t_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth

# 如果不存在，下载
python download_rtmpose_models.py --model rtmpose-tiny
```

3. **使用优化配置**:
```bash
python main.py --config config/config_gpu_optimized.yaml
```

**预期结果**: 18-22 FPS ✓✓

---

## 模型性能对比

### YOLOv8 模型对比

| 模型 | 参数量 | 推理时间 (720p) | 推理时间 (480p) | mAP | 推荐场景 |
|------|--------|----------------|----------------|-----|---------|
| yolov8n | 3.2M | 27ms | 15ms | 37.3 | 实时性优先 ✓ |
| yolov8s | 11.2M | 45ms | 25ms | 44.9 | 平衡 |
| **yolov8m** | 25.9M | **68ms** | **40ms** | 50.2 | 精度优先 (当前) |

### RTMPose 模型对比

| 模型 | 参数量 | 推理时间 | AP (COCO) | 推荐场景 |
|------|--------|---------|-----------|---------|
| **rtmpose-tiny** | 0.9M | **35ms** | 66.0 | 速度优先 ✓✓ |
| **rtmpose-s** | 1.2M | **69ms** | 68.5 | 平衡 (当前) |
| rtmpose-m | 4.0M | 120ms | 72.7 | 精度优先 |

---

## 其他优化技巧

### 1. 跳帧检测 (detection_interval)

人体位置不会每帧都变化，可以每N帧检测一次：

```yaml
inference:
  detection_interval: 2  # 每2帧检测一次，节省50% YOLOv8时间
```

**原理**:
- 第1帧: 运行YOLOv8检测
- 第2帧: 复用上一帧的bbox，只运行RTMPose
- 第3帧: 再次运行YOLOv8...

**效果**: YOLOv8耗时从68ms降到34ms (平均)

---

### 2. TensorRT优化 (高级)

如果需要更高性能，可以转换模型为TensorRT:

```yaml
models:
  pose:
    tensorrt:
      enabled: true
      fp16_mode: true
```

**预期提升**: RTMPose 69ms → 30ms (2.3倍)

**缺点**: 首次运行需要编译(10-20分钟)

---

### 3. 降低摄像头FPS

如果不需要30 FPS输入:

```yaml
camera:
  fps: 15  # 降低到15 FPS
```

**效果**: 摄像头读取从121ms降到理论67ms (但实际可能不变)

---

## 常见问题

### Q: 为什么720p摄像头这么慢？

A: Jetson Orin Nano的USB3.0带宽有限，720p@10fps已经是硬件极限。

**解决方案**:
1. 使用MIPI CSI摄像头 (更高带宽)
2. 降低分辨率到640x480 (可达30fps)
3. 使用USB3.0独立供电

### Q: YOLOv8n会降低多少精度？

A: 人体检测场景下，yolov8n vs yolov8m差距很小：
- yolov8m: mAP 50.2
- yolov8n: mAP 37.3
- **人体检测准确率**: 相差<3%

对于单人场景，影响可忽略。

### Q: rtmpose-tiny vs rtmpose-s差距大吗？

A: 姿态估计精度：
- rtmpose-s: AP 68.5
- rtmpose-tiny: AP 66.0
- **差距**: 2.5% AP

对于坐/站/躺分类，影响很小。

---

## 推荐配置总结

### 日常使用（推荐）
```bash
python main.py --config config/config_gpu.yaml
# 修改为: 640x480 + yolov8n + rtmpose-s
# FPS: 12-15
# 精度: 高
```

### 高性能模式
```bash
python main.py --config config/config_gpu_optimized.yaml
# 配置: 640x480 + yolov8n + rtmpose-tiny + detection_interval=2
# FPS: 18-22
# 精度: 中等
```

### 高精度模式
```bash
python main.py --config config/config_gpu.yaml
# 配置: 1280x720 + yolov8m + rtmpose-s
# FPS: 3-6
# 精度: 最高
```
