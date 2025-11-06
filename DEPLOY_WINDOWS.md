# Windows部署指南

## 🪟 Windows特殊说明

Windows上部署Life Tracker有一些特殊注意事项。

---

## ✅ 推荐方案：虚拟环境（不用Docker）

### 1. 安装Python

```powershell
# 下载并安装 Python 3.10+
# https://www.python.org/downloads/

# 验证安装
python --version
```

### 2. 克隆代码

```powershell
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera
```

### 3. 创建虚拟环境

```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 看到 (venv) 前缀表示成功
```

### 4. 安装依赖

```powershell
# 升级pip
python -m pip install --upgrade pip

# 分步安装（推荐）
pip install numpy opencv-python pyyaml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
pip install mediapipe
pip install pandas scipy matplotlib psutil tqdm loguru
pip install flask flask-cors

# 或一键安装（可能较慢）
pip install -r requirements.txt
```

### 5. 测试运行

```powershell
# 不需要摄像头的测试
python test_quick.py

# 如果有摄像头
python main.py --device pc
```

---

## 🐳 使用Docker（高级用户）

### Windows上Docker的特殊配置

#### 方法1: 使用headless版本（推荐）

```powershell
# 使用headless Dockerfile（无GUI，适合后台运行）
docker build -f Dockerfile.headless -t life-tracker:headless .

# 运行（无可视化）
docker run -d ^
  --name life-tracker ^
  -v %cd%\data:/app/data ^
  -v %cd%\logs:/app/logs ^
  -v %cd%\config:/app/config ^
  -p 5000:5000 ^
  life-tracker:headless
```

#### 方法2: 使用WSL2摄像头

如果你使用WSL2，可以访问摄像头：

```powershell
# 在WSL2中运行
wsl

# 然后按照Linux方式部署
cd /mnt/c/Users/你的用户名/Desktop/code/Camera
docker-compose up -d
```

#### 方法3: 使用视频文件测试

1. 准备测试视频
```powershell
# 下载或放置视频文件到 data 目录
# 例如: data\test_video.mp4
```

2. 修改配置
```yaml
# 编辑 config/config_pc.yaml
camera:
  source: "data/test_video.mp4"  # 使用视频文件
  fps: 30
  resolution: [640, 480]
```

3. 构建并运行
```powershell
docker build -t life-tracker .
docker run -d ^
  --name life-tracker ^
  -v %cd%\data:/app/data ^
  -v %cd%\logs:/app/logs ^
  -v %cd%\config:/app/config ^
  life-tracker
```

---

## 📸 Windows摄像头支持

### 检查摄像头

```powershell
# 安装依赖后测试
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# 如果输出 True，说明摄像头可用
```

### 常见摄像头ID

Windows上摄像头ID可能是：
- `0` - 默认摄像头
- `1` - 第二个摄像头
- `"video=USB Camera"` - 指定设备名

修改 `config/config_pc.yaml`：

```yaml
camera:
  source: 0  # 或 1, 2 等
```

---

## 🔧 常见问题

### 1. torch安装失败

```powershell
# 使用CPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 如果有NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. OpenCV无法打开窗口

在Docker中运行时，Windows不支持X11转发，需要：

**方案A**: 使用无可视化模式
```powershell
python main.py --device pc --no-vis
```

**方案B**: 使用虚拟环境（不用Docker）
```powershell
# 在Windows本地运行
venv\Scripts\activate
python main.py --device pc
```

### 3. Docker build失败

如果遇到包安装错误：

```powershell
# 使用headless版本
docker build -f Dockerfile.headless -t life-tracker .
```

### 4. 权限问题

```powershell
# 以管理员身份运行PowerShell
# 或使用 Docker Desktop 的集成终端
```

### 5. 路径问题

Windows使用反斜杠，Python中需要转义：

```python
# 错误
camera.source = "C:\videos\test.mp4"

# 正确
camera.source = "C:/videos/test.mp4"
# 或
camera.source = r"C:\videos\test.mp4"
```

---

## 🎯 快速开始（Windows推荐流程）

### 选项1: 不使用Docker（最简单）

```powershell
# 1. 安装Python 3.10+
# 2. 克隆代码
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera

# 3. 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 测试
python test_quick.py

# 6. 运行（如果有摄像头）
python main.py --device pc
```

### 选项2: 使用Docker（后台运行）

```powershell
# 1. 确保安装了 Docker Desktop
# 2. 克隆代码
git clone https://github.com/L1TangDingZhen/Camera.git
cd Camera

# 3. 准备测试视频（可选）
# 放置视频到 data\test_video.mp4

# 4. 修改配置使用视频
# 编辑 config\config_pc.yaml
# camera.source: "data/test_video.mp4"

# 5. 构建并运行
docker build -f Dockerfile.headless -t life-tracker .
docker run -d --name life-tracker -v %cd%\data:/app/data life-tracker

# 6. 查看日志
docker logs -f life-tracker
```

---

## 💡 性能优化

### CPU优化

Windows上如果没有NVIDIA GPU，强制使用CPU：

```yaml
# config/config_pc.yaml
device: cpu

models:
  person:
    device: cpu
  pose:
    backend: mediapipe  # CPU友好
    device: cpu
```

### 降低资源占用

```yaml
camera:
  fps: 10  # 降低帧率
  resolution: [320, 240]  # 降低分辨率
```

---

## 📊 期望性能（Windows）

| 配置 | FPS | 说明 |
|------|-----|------|
| i5-12 + CPU | 5-8 | YOLOv8s + MediaPipe |
| i7-12 + CPU | 8-12 | YOLOv8s + MediaPipe |
| i5-12 + RTX 4070 | 280+ | YOLOv8s + RTMPose |

---

## 🆘 获取帮助

如果遇到问题：

1. 查看 `logs\app.log` 日志
2. 运行 `python test_quick.py` 诊断
3. 查看 `DEPLOY.md` 通用部署指南
4. 提交GitHub Issue

---

## ✅ 验证安装成功

运行测试脚本应该看到：

```
============================================================
  Life Tracker 组件测试
============================================================

[1/5] 测试配置加载...
  ✓ 配置文件加载成功

[2/5] 测试YOLOv8检测器...
  ✓ 检测器正常
  ✓ FPS: 8.5

[3/5] 测试MediaPipe姿态估计...
  ✓ 姿态估计正常，关键点数: 17

[4/5] 测试ROI管理器...
  ✓ ROI管理器初始化成功

[5/5] 测试状态机...
  ✓ 状态机初始化成功

[6/6] 测试数据库...
  ✓ 数据库初始化成功

============================================================
  ✓ 所有组件测试通过！
============================================================
```

---

## 🎉 完成

现在你可以在Windows上运行Life Tracker了！

推荐使用**虚拟环境方式**（不用Docker），更简单稳定。
