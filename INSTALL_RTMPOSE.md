# RTMPose 安装指南

本指南提供RTMPose在不同平台的详细安装说明。

## 📋 目录

- [为什么使用RTMPose](#为什么使用rtmpose)
- [快速开始](#快速开始)
- [Linux/Jetson安装（推荐）](#linuxjetson安装推荐)
- [Windows安装（高级）](#windows安装高级)
- [验证安装](#验证安装)
- [常见问题](#常见问题)

---

## 为什么使用RTMPose

| 对比项 | MediaPipe (默认) | RTMPose |
|--------|-----------------|---------|
| **设备支持** | ❌ CPU only | ✅ CPU + GPU |
| **速度** | ~50ms (CPU) | ~12ms (GPU FP16) |
| **精度** | AP ~67% | AP ~68.5% |
| **平台支持** | ✅ 全平台 | ⚠️ Linux优先 |
| **安装难度** | ✅ 简单 | ⚠️ 中等-困难 |

**推荐场景**：
- ✅ 开发测试 → 使用 **MediaPipe** (简单稳定)
- ✅ 生产部署 → 使用 **RTMPose** (GPU加速)

---

## 快速开始

### 选项A：继续使用MediaPipe（最简单）

如果你只是想开发测试，**不需要安装RTMPose**！

```yaml
# config/config_gpu.yaml
models:
  pose:
    backend: mediapipe  # 使用默认的MediaPipe
    complexity: 1
    device: cpu
```

MediaPipe已经足够好用（AP 67%，50ms），安装简单，跨平台兼容性好。

### 选项B：安装RTMPose（高性能）

如果你需要：
- 🚀 更快的速度（12ms vs 50ms）
- 📈 更高的精度（AP 68.5% vs 67%）
- 🎯 生产部署到Jetson

那么请继续阅读下面的安装指南。

---

## Linux/Jetson安装（推荐）

### 环境要求

- Python 3.8+
- PyTorch 1.8+ with CUDA 11.1+
- GCC 5.4+
- CUDA Toolkit 11.1+

### 步骤1：安装OpenMIM

```bash
pip install openmim
```

OpenMIM是OpenMMLab的包管理工具，会自动处理版本兼容性。

### 步骤2：安装MMPose依赖

```bash
# 安装mmcv（会自动选择与PyTorch兼容的版本）
mim install mmcv==2.0.0

# 安装mmengine
mim install mmengine==0.8.0

# 安装mmpose
mim install mmpose==1.0.0
```

**注意**：这些命令会自动下载预编译的wheel包，避免编译C++/CUDA代码。

### 步骤3：下载RTMPose模型

```bash
# 下载推荐模型（rtmpose-s）
python download_rtmpose_models.py --model rtmpose-s

# 或下载所有模型
python download_rtmpose_models.py --all
```

### 步骤4：验证安装

```bash
# 检查依赖版本
python -c "import mmcv; print(f'mmcv: {mmcv.__version__}')"
python -c "import mmpose; print(f'mmpose: {mmpose.__version__}')"
python -c "import mmengine; print(f'mmengine: {mmengine.__version__}')"

# 运行测试
python test_quick.py --backend rtmpose
```

### Jetson特殊说明

Jetson设备可以使用相同的安装方法。如果遇到编译问题，可以尝试：

```bash
# 使用Jetson预编译的wheel（如果可用）
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/jetpack/index.html
```

---

## Windows安装（高级）

### ⚠️ 警告

**MMPose在Windows上安装非常复杂**，主要问题：

1. **编译依赖**：需要Visual Studio 2019+和CUDA Toolkit
2. **版本冲突**：mmcv与PyTorch/CUDA版本严格绑定
3. **编译时间长**：首次安装可能需要30分钟+
4. **失败率高**：编译错误频繁，难以调试

### Windows用户的三个选择

#### 选项1：继续使用MediaPipe（推荐）

**最简单的方案**，无需任何额外安装：

```yaml
# config/config_gpu.yaml
models:
  pose:
    backend: mediapipe
```

MediaPipe在Windows上工作完美，性能也足够日常开发使用。

#### 选项2：使用WSL2（推荐给高级用户）

在Windows上使用Linux子系统，享受Linux的便利：

```bash
# 在PowerShell（管理员）中启用WSL2
wsl --install

# 重启后，在WSL2中安装
wsl
sudo apt update
sudo apt install python3-pip

# 然后按照Linux安装步骤进行
pip install openmim
mim install mmcv==2.0.0 mmpose==1.0.0
```

**优点**：
- ✅ 安装简单（与Linux相同）
- ✅ 可以访问Windows文件系统
- ✅ GPU加速可用（需要WSL2 + CUDA支持）

#### 选项3：硬核安装（仅供参考，不推荐）

如果你坚持要在原生Windows上安装：

**前置要求**：
1. Visual Studio 2019或2022（需要C++桌面开发工具）
2. CUDA Toolkit 11.8（需要与PyTorch版本匹配）
3. PyTorch 2.0+（CUDA版本）

```bash
# 1. 安装编译工具
# 下载并安装 Visual Studio Build Tools
# https://visualstudio.microsoft.com/downloads/

# 2. 安装PyTorch（CUDA版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 尝试安装mmcv（可能失败）
pip install openmim
mim install mmcv==2.0.0  # 可能需要编译30分钟+

# 4. 如果编译失败，尝试预编译版本
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# 5. 安装mmpose
mim install mmpose==1.0.0
```

**常见错误**：

```
ERROR: Cannot build mmcv
  → 解决：检查Visual Studio是否正确安装

ERROR: CUDA version mismatch
  → 解决：确保CUDA、PyTorch、mmcv版本匹配

ERROR: cl.exe not found
  → 解决：运行vcvars64.bat初始化编译环境
```

**如果遇到问题**，强烈建议：
1. 使用WSL2
2. 或继续使用MediaPipe
3. 或直接部署到Jetson设备

---

## 验证安装

### 检查依赖

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import mmcv; print(f'mmcv: {mmcv.__version__}')"
python -c "import mmpose; print(f'mmpose: {mmpose.__version__}')"
```

**期望输出**：
```
PyTorch: 2.0.0+cu118
CUDA available: True
mmcv: 2.0.0
mmpose: 1.0.0
```

### 测试RTMPose

```bash
# 快速测试
python test_quick.py --backend rtmpose

# 完整测试
python main.py --config config/config_gpu.yaml
```

**期望结果**：
- ✅ 模型加载成功
- ✅ 推理时间 <20ms
- ✅ 无错误信息

---

## 常见问题

### Q1: mmcv安装失败，报编译错误

**A**: 使用mim安装而不是pip：

```bash
pip install openmim
mim install mmcv==2.0.0  # mim会自动下载预编译包
```

如果仍然失败（Windows），建议使用WSL2或MediaPipe。

### Q2: 提示"CUDA version mismatch"

**A**: 确保PyTorch、CUDA Toolkit、mmcv版本匹配：

| PyTorch | CUDA Toolkit | mmcv |
|---------|--------------|------|
| 2.0.x | 11.8 | 2.0.0+cu118 |
| 1.13.x | 11.7 | 1.7.1+cu117 |

重新安装匹配版本：
```bash
# 例如：PyTorch 2.0 + CUDA 11.8
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
mim install mmcv==2.0.0
```

### Q3: 模型文件下载失败

**A**: 尝试以下方法：

```bash
# 方法1：使用脚本下载
python download_rtmpose_models.py --model rtmpose-s

# 方法2：手动下载
# 1. 访问 https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
# 2. 下载配置文件和权重文件
# 3. 放到 models/rtmpose/ 目录

# 方法3：使用mim下载
mim download mmpose --config rtmpose-s_8xb256-420e_coco-256x192 --dest models/rtmpose/
```

### Q4: 运行时提示"config file not found"

**A**: 检查配置文件路径：

```yaml
# config/config_gpu.yaml
models:
  pose:
    backend: rtmpose
    model: rtmpose-s
    config_file: models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py
    checkpoint: models/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth
    device: cuda:0
```

确保文件存在：
```bash
ls models/rtmpose/configs/
ls models/rtmpose/*.pth
```

### Q5: TensorRT优化不生效

**A**: TensorRT完整集成需要MMDeploy：

```bash
# 安装MMDeploy（可选）
mim install mmdeploy

# 或者只使用FP16优化（不需要MMDeploy）
# 配置文件中：
tensorrt:
  enabled: true
  fp16_mode: true  # 自动使用PyTorch的half精度
```

当前实现会自动使用FP16加速（如果启用），无需完整的TensorRT转换。

### Q6: Jetson上性能不如预期

**A**: 确保启用TensorRT优化：

```yaml
# config/config_jetson.yaml
models:
  pose:
    backend: rtmpose
    model: rtmpose-s
    device: cuda:0

tensorrt:
  enabled: true
  fp16_mode: true  # Jetson推荐FP16
  workspace_size: 2048
```

检查功耗模式：
```bash
# 查看当前功耗模式
sudo nvpmodel -q

# 切换到最高性能（25W）
sudo nvpmodel -m 0
```

---

## 总结

### 推荐方案

**开发测试**（Windows/macOS）：
```
使用 MediaPipe (backend: mediapipe)
→ 安装简单，跨平台兼容
```

**生产部署**（Linux/Jetson）：
```
使用 RTMPose (backend: rtmpose)
→ GPU加速，性能更好
```

### 快速决策

```
你的目标是什么？
│
├─ 快速开发测试
│  └─ 使用 MediaPipe ✅（无需额外安装）
│
├─ 高性能部署
│  ├─ Linux/Jetson → 安装 RTMPose ✅
│  └─ Windows → 使用 WSL2 + RTMPose
│
└─ Windows原生开发
   ├─ 轻度使用 → MediaPipe ✅（推荐）
   └─ 必须GPU → WSL2 + RTMPose
```

如有其他问题，请提交Issue或查看[MMPose官方文档](https://mmpose.readthedocs.io/)。
