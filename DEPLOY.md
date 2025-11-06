# 本地部署指南

## 📋 前置要求

- Python 3.8+
- 摄像头（USB或内置）
- （可选）NVIDIA GPU + CUDA 11.0+

---

## 🚀 方案一：虚拟环境（推荐新手）

### 1. 克隆项目

```bash
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera
git checkout claude/three-stage-deployment-roadmap-011CUrFSWFN5rH8EACYAZGjD
```

### 2. 创建虚拟环境

```bash
# 创建
python3 -m venv venv

# 激活（Linux/Mac）
source venv/bin/activate

# 激活（Windows）
venv\Scripts\activate

# 看到 (venv) 前缀表示成功
```

### 3. 安装依赖

```bash
# 方法A: 一键安装（可能失败）
pip install -r requirements.txt

# 方法B: 分步安装（更稳定）
pip install --upgrade pip
pip install numpy opencv-python pyyaml
pip install torch torchvision  # CPU版本，自动选择
pip install ultralytics
pip install mediapipe
pip install pandas scipy matplotlib psutil tqdm loguru
pip install flask flask-cors
```

**常见问题：**

- **torch安装慢？** 使用清华镜像：
  ```bash
  pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- **有NVIDIA GPU？** 安装CUDA版本：
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### 4. 测试安装

```bash
# 快速测试（不需要摄像头）
python test_quick.py

# 看到 "所有组件测试通过" 表示成功
```

### 5. 标定ROI区域

```bash
# 连接摄像头后运行
python scripts/calibrate_roi.py --device pc

# 操作：
# - 鼠标点击标记区域顶点
# - 按 'c' 完成当前区域
# - 按 's' 保存配置
# - 按 'q' 退出
```

### 6. 正式运行

```bash
# PC开发模式（GPU）
python main.py --device pc

# X390验证模式（CPU）
python main.py --device x390

# 不显示窗口（后台运行）
python main.py --device pc --no-vis
```

### 7. 退出环境

```bash
deactivate
```

---

## 🐳 方案二：Docker（完全隔离）

### 1. 安装Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER  # 添加用户到docker组
# 注销重新登录

# Mac: 下载 Docker Desktop
# Windows: 下载 Docker Desktop
```

### 2. 构建镜像

```bash
cd Camera

# 构建
docker build -t life-tracker .

# 或使用docker-compose
docker-compose build
```

### 3. 运行容器

```bash
# 方式1: docker命令
docker run -it --rm \
  --device=/dev/video0 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config:/app/config \
  life-tracker

# 方式2: docker-compose（推荐）
docker-compose up -d  # 后台运行
docker-compose logs -f  # 查看日志
docker-compose down  # 停止
```

### 4. 进入容器调试

```bash
docker exec -it life-tracker bash
```

---

## 🔧 方案三：Conda环境（推荐科研人员）

### 1. 安装Conda

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 2. 创建环境

```bash
cd Camera

# 创建环境
conda create -n life-tracker python=3.10 -y
conda activate life-tracker

# 安装依赖
pip install -r requirements.txt
```

### 3. 运行

```bash
conda activate life-tracker
python test_quick.py
python main.py --device pc
```

### 4. 退出

```bash
conda deactivate
```

---

## 📦 无摄像头测试方案

### 使用视频文件

```bash
# 1. 准备测试视频
# 下载或录制一段包含人的视频，放到 data/test_video.mp4

# 2. 修改配置
# 编辑 config/config_pc.yaml
camera:
  source: "data/test_video.mp4"  # 改为视频路径
  fps: 30
  resolution: [640, 480]

# 3. 运行
python main.py --device pc
```

### 使用测试脚本

```bash
# 不需要摄像头的测试
python test_quick.py

# 性能测试
python test_quick.py --perf
```

---

## 🐛 常见问题排查

### 1. ImportError: No module named 'cv2'

```bash
pip install opencv-python
```

### 2. ImportError: No module named 'ultralytics'

```bash
pip install ultralytics
```

### 3. 摄像头无法打开

```bash
# 检查设备
ls /dev/video*

# 尝试不同ID
# 修改 config/config_pc.yaml 中的 camera.source 为 1 或 2
```

### 4. CUDA out of memory

```yaml
# 修改 config/config_pc.yaml
models:
  person:
    device: cpu  # 改为CPU
  pose:
    backend: mediapipe  # 使用CPU友好的后端
    device: cpu
```

### 5. YOLOv8模型下载失败

```bash
# 手动下载
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
mv yolov8s.pt models/
```

### 6. MediaPipe初始化失败

```bash
# 卸载重装
pip uninstall mediapipe -y
pip install mediapipe --no-cache-dir
```

---

## 📊 验证运行成功

运行后应该看到：

```
============================================================
  Life Tracker - PC Development
  设备: cuda:0
============================================================

[初始化] 加载人体检测器...
[PersonDetector] 加载模型: yolov8s.pt
[PersonDetector] 设备: cuda:0
[PersonDetector] 模型加载成功

[初始化] 加载姿态估计器...
[MediaPipePose] 初始化完成，complexity=1

[初始化] 加载ROI管理器...
[ROIManager] 加载了 0 个区域: []

[初始化] 创建状态机...
[BehaviorStateMachine] 初始化完成

[初始化] 创建事件记录器...
[EventLogger] 初始化完成

[初始化] 打开摄像头...

[初始化] 所有组件加载完成!

[运行] 开始监测...
```

窗口显示：
- 实时画面
- FPS显示
- 当前状态
- 区域信息

---

## 🎯 下一步

1. **标定ROI区域**: `python scripts/calibrate_roi.py --device pc`
2. **模型对比测试**: `python scripts/compare_models.py --device pc`
3. **查看Web界面**: （待实现）访问 http://localhost:5000
4. **迁移到X390**: 按照 README.md 中的阶段2步骤

---

## 💡 性能参考

| 环境 | 配置 | FPS | 说明 |
|------|------|-----|------|
| PC GPU | i5-12 + RTX 4070 | 280+ | YOLOv8s + RTMPose |
| PC CPU | i5-12 | 6-8 | YOLOv8s + MediaPipe |
| X390 CPU | i5-8 | 6-8 | YOLOv8s + MediaPipe |
| Jetson | Orin Nano | 38-45 | YOLOv8s-TRT + RTMPose-TRT |

---

## 📞 获取帮助

遇到问题？

1. 查看 `README.md` 的常见问题章节
2. 运行 `python test_quick.py` 定位问题
3. 查看 `logs/app.log` 日志文件
4. 提交 GitHub Issue
