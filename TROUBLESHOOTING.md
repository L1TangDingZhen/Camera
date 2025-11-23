# Troubleshooting Guide

This document records solutions to common issues encountered during development and deployment.

## Table of Contents
- [RTMPose Installation Issues](#rtmpose-installation-issues)
- [RTMPose Runtime Issues](#rtmpose-runtime-issues)
- [PyTorch Compatibility Issues](#pytorch-compatibility-issues)
- [Camera Issues](#camera-issues)
- [Performance Issues](#performance-issues)

---

## RTMPose Installation Issues

### Issue 1: PyTorch 2.8.0 Cannot Load RTMPose Model

**Error Message:**
```
_pickle.UnpicklingError: Weights only load failed...
WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct
```

**Root Cause:**
- PyTorch 2.6+ changed default `weights_only` parameter in `torch.load()` from `False` to `True`
- RTMPose model files (trained in 2023) contain numpy objects
- PyTorch 2.8.0 security check rejects these legacy checkpoint files

**Solution:**
Modify mmengine's checkpoint loading code to disable `weights_only` check:

**File:** `Camera/lib/python3.10/site-packages/mmengine/runner/checkpoint.py`

**Line 347-348:**

```python
# BEFORE (will fail with PyTorch 2.6+)
checkpoint = torch.load(filename, map_location=map_location)

# AFTER (add weights_only=False parameter)
# Fix for PyTorch 2.6+ compatibility: disable weights_only for legacy checkpoints
checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
```

**Verification:**
```bash
source Camera/bin/activate
python main.py --mode gpu --no-vis
# Should see: [RTMPose] 模型加载成功 ✓
```

**Important Notes:**
- This is a virtual environment modification, safe to make
- If you recreate the virtual environment, you'll need to apply this fix again
- Only use this with trusted checkpoint files (RTMPose official models are safe)
- Future mmengine versions may fix this automatically

**Alternative Solutions (not recommended):**
1. Downgrade PyTorch to 2.5.0 (breaks other dependencies)
2. Wait for mmengine update (uncertain timeline)
3. Use MediaPipe instead (but loses RTMPose performance benefits)

---

### Issue 2: mmcv Version Compatibility

**Error Message:**
```
AssertionError: MMCV==2.2.0 is used but incompatible.
Please install mmcv>=2.0.0rc4, <=2.1.0.
```

**Root Cause:**
- Running `mim install mmcv` without version number installs latest version (2.2.0)
- mmpose 1.1.0 requires mmcv between 2.0.0rc4 and 2.1.0
- Version 2.2.0 is too new

**Solution:**
```bash
source Camera/bin/activate

# Uninstall wrong version
pip uninstall mmcv -y

# Install compatible version (MUST specify version number!)
mim install mmcv==2.1.0
# or
mim install mmcv==2.0.0  # More stable, recommended

# Verify
python -c "import mmcv; print(f'mmcv: {mmcv.__version__}')"
```

**Lesson Learned:**
- Always specify exact version when using `mim install`
- Never run `mim install mmcv` without `==X.X.X`

---

### Issue 3: Missing mmdet Dependency

**Error Message:**
```
KeyError: 'CSPNeXt is not in the model registry'
mmdet is not installed
```

**Root Cause:**
- RTMPose uses CSPNeXt backbone from MMDetection
- CSPNeXt model class must be registered before loading RTMPose

**Solution:**
```bash
source Camera/bin/activate
mim install mmdet
```

**Note:** You don't need mmdet for object detection (YOLOv8 handles that), but mmpose needs it for model registry.

---

## RTMPose Runtime Issues

### Issue: Broadcast Error During Pose Estimation ✅ RESOLVED

**Error Message:**
```
[RTMPose] 姿态估计失败: operands could not be broadcast together with shapes (1,2) (1,2) (1,4)
```

**Root Cause:**
- mmpose's `bbox_xyxy2cs()` function doesn't handle 5-element bboxes (with confidence score)
- When bbox is `[x1, y1, x2, y2, score]` (shape `(1, 5)`), `np.hsplit(bbox, [1, 2, 3])` incorrectly splits as:
  - `x1 = bbox[:, :1]` → `(1, 1)` ✓
  - `y1 = bbox[:, 1:2]` → `(1, 1)` ✓
  - `x2 = bbox[:, 2:3]` → `(1, 1)` ✓
  - `y2 = bbox[:, 3:]` → `(1, 2)` ✗ **包含了score！**
- This causes `bbox_scale` to have shape `(1, 3)` instead of `(1, 2)`
- Later `np.hstack([h*aspect_ratio, h])` = `hstack([(1,2), (1,2)])` = `(1, 4)` causing broadcast error

**Status:** ✅ **RESOLVED**

**Solution:**
修改mmpose源码，在`bbox_xyxy2cs`中只取前4列：

**File:** `/home/eyes/Desktop/mmpose/mmpose/structures/bbox/transforms.py`
**Line 66-67:**
```python
# FIX: 只取前4列，忽略可能存在的第5列（score）
bbox = bbox[:, :4]

x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
```

**Verification:**
```bash
source Camera/bin/activate
python main.py --mode gpu
# Should work without broadcast errors
```

**Note:** This fix is essential for RTMPose to work with YOLOv8 detections that include confidence scores.

---

## PyTorch Compatibility Issues

### Issue: JetPack 6.2.1 + PyTorch Installation

**Environment:**
- Jetson Orin Nano
- JetPack 6.2.1 (R36.4.7)
- CUDA 12.6

**Correct Installation:**
```bash
source Camera/bin/activate

# Install from Jetson AI Lab PyPI (NOT pip default!)
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

# Verify
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
# Should print: CUDA Available: True
```

**Wrong Way:**
```bash
# DON'T DO THIS - installs CPU-only version
pip install torch torchvision
```

---

## Camera Issues

### Issue 1: Camera Index Changes After Replug

**Problem:**
- USB camera device index changes (video0 → video2 → video1)
- Program fails with "无法打开摄像头: 0"

**Solution:**
Automatic camera search has been implemented in `main.py` (lines 68-84).

**Behavior:**
1. Try configured camera (e.g., `source: 2`)
2. If failed, automatically search `/dev/video0-9`
3. Use first available camera
4. Print: `[成功] 找到可用摄像头: /dev/videoX`

**Manual Check:**
```bash
source Camera/bin/activate
python -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'/dev/video{i}: {frame.shape[1]}x{frame.shape[0]}')
        cap.release()
"
```

### Issue 2: Camera Permission Denied

**Error:**
```
VIDEOIO(V4L2): can't open camera by index
```

**Solution:**
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again, then verify
groups | grep video
```

---

## Performance Issues

### Issue: Low FPS (4-5 FPS) in GPU Mode

**Symptoms:**
```
读取帧:     118ms  (59%)  ← Bottleneck
人体检测:    83ms  (41%)
总耗时:     202ms
理论FPS:     4.9
```

**Root Causes:**
1. High camera resolution (1920x1080 or higher)
2. Large YOLOv8 model (yolov8m)
3. High detection frequency (every frame)

**Solutions (in order of effectiveness):**

#### Solution 1: Use Correct Camera (fastest, 28 FPS)
Ensure using low-resolution USB camera (640x360):
```yaml
# config/config_gpu.yaml
camera:
  source: 2  # 640x360 camera
  resolution: [640, 360]
```

**Expected Performance:**
```
读取帧:     0.2ms  ← Fixed!
人体检测:    35ms
理论FPS:     28.4
```

#### Solution 2: Switch to Smaller Model (2-3x faster)
```yaml
# config/config_gpu.yaml
models:
  person:
    model: yolov8n.pt  # Change from yolov8m.pt
```

**Effect:**
- Detection: 83ms → 35ms
- FPS: 5 → 10-12

#### Solution 3: Reduce Detection Frequency (3x faster)
```yaml
# config/config_gpu.yaml
inference:
  detection_interval: 3  # Detect every 3rd frame (was 1)
```

**Effect:**
- Detection overhead reduced by 66%
- FPS: 5 → 15-20

#### Solution 4: Lower Camera Resolution
```yaml
# config/config_gpu.yaml
camera:
  resolution: [640, 480]  # Lower from 1920x1080
```

---

## Installation Summary for Jetson Orin Nano

**Complete working setup:**

```bash
# 1. Create virtual environment
python3 -m venv Camera
source Camera/bin/activate

# 2. Install PyTorch (Jetson-specific)
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

# 3. Install basic requirements
pip install -r requirements_jetson.txt

# 4. Install RTMPose dependencies
pip install openmim
mim install mmengine==0.8.0
mim install mmcv==2.1.0    # or 2.0.0
mim install mmdet

# Note: mmpose installed from local git repo at /home/eyes/Desktop/mmpose

# 5. Fix PyTorch 2.8 compatibility
# Edit Camera/lib/python3.10/site-packages/mmengine/runner/checkpoint.py
# Line 347: Add weights_only=False to torch.load()

# 6. Download RTMPose models
python download_rtmpose_models.py --model rtmpose-s

# 7. Verify installation
python -c "import torch, mmcv, mmpose; print('All OK')"

# 8. Run
python main.py --mode gpu --no-vis
```

**Tested Configuration:**
- Hardware: Jetson Orin Nano 8GB
- JetPack: 6.2.1 (R36.4.7)
- CUDA: 12.6
- Python: 3.10.12
- PyTorch: 2.8.0
- mmcv: 2.1.0
- mmengine: 0.8.0
- mmpose: 1.1.0
- mmdet: 3.3.0

---

## Quick Reference Commands

**Check versions:**
```bash
source Camera/bin/activate
python -c "
import torch, mmcv, mmengine, mmpose
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'mmcv: {mmcv.__version__}')
print(f'mmengine: {mmengine.__version__}')
print(f'mmpose: {mmpose.__version__}')
"
```

**Check cameras:**
```bash
ls -l /dev/video*
v4l2-ctl --list-devices  # If installed
```

**Check GPU:**
```bash
nvidia-smi
```

**Kill stuck processes:**
```bash
pkill -f "python main.py"
```

---

## Contact & Updates

If you encounter new issues:
1. Check this document first
2. Review logs in `logs/app.log`
3. Enable debug mode: `python main.py --mode gpu --debug`
4. Check GitHub issues for similar problems

Last Updated: 2025-11-20
Environment: Jetson Orin Nano + JetPack 6.2.1
