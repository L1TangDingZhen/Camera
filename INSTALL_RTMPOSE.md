# RTMPose Installation Guide

This guide provides detailed installation instructions for RTMPose on different platforms.

## 📋 Table of Contents

- [Why Use RTMPose](#why-use-rtmpose)
- [Quick Start](#quick-start)
- [Linux/Jetson Installation (Recommended)](#linuxjetson-installation-recommended)
- [Windows Installation (Advanced)](#windows-installation-advanced)
- [Verify Installation](#verify-installation)
- [FAQ](#faq)

---

## Why Use RTMPose

| Comparison | MediaPipe (Default) | RTMPose |
|-----------|-------------------|---------|
| **Device Support** | ❌ CPU only | ✅ CPU + GPU |
| **Speed** | ~50ms (CPU) | ~12ms (GPU FP16) |
| **Accuracy** | AP ~67% | AP ~68.5% |
| **Platform Support** | ✅ All platforms | ⚠️ Linux preferred |
| **Installation Difficulty** | ✅ Simple | ⚠️ Medium-Hard |

**Recommended Scenarios**:
- ✅ Development/Testing → Use **MediaPipe** (simple and stable)
- ✅ Production Deployment → Use **RTMPose** (GPU acceleration)

---

## Quick Start

### Option A: Continue Using MediaPipe (Simplest)

If you just want to develop and test, **you don't need to install RTMPose**!

```yaml
# config/config_gpu.yaml
models:
  pose:
    backend: mediapipe  # Use default MediaPipe
    complexity: 1
    device: cpu
```

MediaPipe is already good enough (AP 67%, 50ms), with simple installation and excellent cross-platform compatibility.

### Option B: Install RTMPose (High Performance)

If you need:
- 🚀 Faster speed (12ms vs 50ms)
- 📈 Higher accuracy (AP 68.5% vs 67%)
- 🎯 Production deployment to Jetson

Then continue reading the installation guide below.

---

## Linux/Jetson Installation (Recommended)

### Environment Requirements

- Python 3.8+
- PyTorch 1.8+ with CUDA 11.1+
- GCC 5.4+
- CUDA Toolkit 11.1+

### Step 1: Install OpenMIM

```bash
pip install openmim
```

OpenMIM is OpenMMLab's package management tool that automatically handles version compatibility.

### Step 2: Install MMPose Dependencies

```bash
# Install mmengine (core dependency)
mim install mmengine==0.8.0

# Install mmcv (automatically selects version compatible with PyTorch)
mim install mmcv==2.1.0

# Install mmpose
mim install mmpose==1.1.0
```

**Note**: These commands will automatically download pre-compiled wheel packages, avoiding C++/CUDA code compilation.

### Step 3: Download RTMPose Models

```bash
# Download RTMPose-s model (recommended)
mim download mmpose --config rtmpose-s_8xb256-420e_coco-256x192 --dest models/rtmpose/

# Or download RTMPose-m model (higher accuracy)
mim download mmpose --config rtmpose-m_8xb256-420e_coco-256x192 --dest models/rtmpose/

# Or download RTMPose-l model (best accuracy)
mim download mmpose --config rtmpose-l_8xb256-420e_coco-256x192 --dest models/rtmpose/
```

### Step 4: Verify Installation

```bash
# Check dependency versions
python -c "import mmcv; print(f'mmcv: {mmcv.__version__}')"
python -c "import mmpose; print(f'mmpose: {mmpose.__version__}')"
python -c "import mmengine; print(f'mmengine: {mmengine.__version__}')"

# Run test
python test_quick.py --backend rtmpose
```

### Jetson Special Notes

Jetson devices can use the same installation method. If you encounter compilation issues, try:

```bash
# Use pre-compiled wheels for Jetson (if available)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/jetpack/index.html
```

---

## Windows Installation (Advanced)

### ⚠️ Warning

**MMPose installation on Windows is very complex**, main issues:

1. **Compilation Dependencies**: Requires Visual Studio 2019+ and CUDA Toolkit
2. **Version Conflicts**: mmcv is strictly bound to PyTorch/CUDA versions
3. **Long Compilation Time**: First installation may take 30+ minutes
4. **High Failure Rate**: Frequent compilation errors, difficult to debug

### Three Options for Windows Users

#### Option 1: Continue Using MediaPipe (Recommended)

**Simplest solution**, no additional installation needed:

```yaml
# config/config_gpu.yaml
models:
  pose:
    backend: mediapipe
```

MediaPipe works perfectly on Windows with sufficient performance for daily development.

#### Option 2: Use WSL2 (Recommended for Advanced Users)

Use Linux subsystem on Windows, enjoy Linux convenience:

```bash
# Enable WSL2 in PowerShell (Administrator)
wsl --install

# After reboot, install in WSL2
wsl
sudo apt update
sudo apt install python3-pip

# Then follow Linux installation steps
pip install openmim
mim install mmcv==2.1.0 mmpose==1.1.0
```

**Advantages**:
- ✅ Simple installation (same as Linux)
- ✅ Access to Windows file system
- ✅ GPU acceleration available (requires WSL2 + CUDA support)

#### Option 3: Native Installation (For Reference Only, Not Recommended)

If you insist on installing on native Windows:

**Prerequisites**:
1. Visual Studio 2019 or 2022 (requires C++ desktop development tools)
2. CUDA Toolkit 11.8 (must match PyTorch version)
3. PyTorch 2.0+ (CUDA version)

```bash
# 1. Install compilation tools
# Download and install Visual Studio Build Tools
# https://visualstudio.microsoft.com/downloads/

# 2. Install PyTorch (CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Try installing mmcv (may fail)
pip install openmim
mim install mmcv==2.1.0  # May require 30+ minutes compilation

# 4. If compilation fails, try pre-compiled version
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# 5. Install mmpose
mim install mmpose==1.1.0
```

**Common Errors**:

```
ERROR: Cannot build mmcv
  → Solution: Check if Visual Studio is correctly installed

ERROR: CUDA version mismatch
  → Solution: Ensure CUDA, PyTorch, mmcv versions match

ERROR: cl.exe not found
  → Solution: Run vcvars64.bat to initialize compilation environment
```

**If you encounter problems**, strongly recommended:
1. Use WSL2
2. Or continue using MediaPipe
3. Or deploy directly to Jetson device

---

## Verify Installation

### Check Dependencies

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import mmcv; print(f'mmcv: {mmcv.__version__}')"
python -c "import mmpose; print(f'mmpose: {mmpose.__version__}')"
```

**Expected Output**:
```
PyTorch: 2.0.0+cu118
CUDA available: True
mmcv: 2.0.0
mmpose: 1.0.0
```

### Test RTMPose

```bash
# Quick test
python test_quick.py --backend rtmpose

# Full test
python main.py --config config/config_gpu.yaml
```

**Expected Results**:
- ✅ Model loads successfully
- ✅ Inference time <20ms
- ✅ No error messages

---

## FAQ

### Q1: mmcv installation fails with compilation errors

**A**: Use mim install instead of pip:

```bash
pip install openmim
mim install mmcv==2.1.0  # mim automatically downloads pre-compiled packages
```

If still failing (Windows), recommend using WSL2 or MediaPipe.

### Q2: "CUDA version mismatch" error

**A**: Ensure PyTorch, CUDA Toolkit, mmcv versions match:

| PyTorch | CUDA Toolkit | mmcv |
|---------|--------------|------|
| 2.0.x | 11.8 | 2.0.0+cu118 |
| 1.13.x | 11.7 | 1.7.1+cu117 |

Reinstall matching versions:
```bash
# Example: PyTorch 2.0 + CUDA 11.8
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
mim install mmcv==2.1.0
```

### Q3: Model file download fails

**A**: Try the following methods:

```bash
# Method 1: Use mim download (recommended)
mim download mmpose --config rtmpose-s_8xb256-420e_coco-256x192 --dest models/rtmpose/

# Method 2: Manual download
# 1. Visit https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
# 2. Download config file and weight file
# 3. Place in models/rtmpose/ directory

# Method 3: Use mim download
mim download mmpose --config rtmpose-s_8xb256-420e_coco-256x192 --dest models/rtmpose/
```

### Q4: Runtime error "config file not found"

**A**: Check configuration file path:

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

Ensure files exist:
```bash
ls models/rtmpose/configs/
ls models/rtmpose/*.pth
```

### Q5: TensorRT optimization not working

**A**: Full TensorRT integration requires MMDeploy:

```bash
# Install MMDeploy (optional)
mim install mmdeploy

# Or just use FP16 optimization (doesn't require MMDeploy)
# In config file:
tensorrt:
  enabled: true
  fp16_mode: true  # Automatically uses PyTorch half precision
```

Current implementation automatically uses FP16 acceleration (if enabled), without requiring full TensorRT conversion.

### Q6: Performance on Jetson not as expected

**A**: Ensure TensorRT optimization is enabled:

```yaml
# config/config_jetson.yaml
models:
  pose:
    backend: rtmpose
    model: rtmpose-s
    device: cuda:0

tensorrt:
  enabled: true
  fp16_mode: true  # Recommended FP16 for Jetson
  workspace_size: 2048
```

Check power mode:
```bash
# Check current power mode
sudo nvpmodel -q

# Switch to maximum performance (25W)
sudo nvpmodel -m 0
```

---

## Summary

### Recommended Solutions

**Development/Testing** (Windows/macOS):
```
Use MediaPipe (backend: mediapipe)
→ Simple installation, cross-platform compatible
```

**Production Deployment** (Linux/Jetson):
```
Use RTMPose (backend: rtmpose)
→ GPU acceleration, better performance
```

### Quick Decision

```
What is your goal?
│
├─ Quick development and testing
│  └─ Use MediaPipe ✅ (no additional installation needed)
│
├─ High performance deployment
│  ├─ Linux/Jetson → Install RTMPose ✅
│  └─ Windows → Use WSL2 + RTMPose
│
└─ Native Windows development
   ├─ Light usage → MediaPipe ✅ (recommended)
   └─ Must have GPU → WSL2 + RTMPose
```

For other questions, please submit an Issue or check the [MMPose official documentation](https://mmpose.readthedocs.io/).
