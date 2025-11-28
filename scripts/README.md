# Scripts Directory

Quick-start scripts for running Life Tracker on different platforms.

---

## 🖥️ PC Scripts

### GPU Mode (Recommended)
```bash
# Run with GPU acceleration (requires NVIDIA GPU + CUDA)
./scripts/pc_run_gpu.sh

# Asynchronous mode (higher FPS)
./scripts/pc_run_gpu.sh --async

# Server mode (no visualization)
./scripts/pc_run_gpu.sh --async --no-vis
```

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA 11.0+ installed
- PyTorch with CUDA

**Expected Performance:**
- Synchronous: 20-25 FPS
- Asynchronous: 25-30 FPS

---

### CPU Mode
```bash
# Run with CPU only (no GPU required)
./scripts/pc_run_cpu.sh

# Asynchronous mode
./scripts/pc_run_cpu.sh --async

# Server mode
./scripts/pc_run_cpu.sh --async --no-vis
```

**Requirements:**
- Python 3.8+
- No GPU required

**Expected Performance:**
- Synchronous: 10-15 FPS
- Asynchronous: 15-20 FPS

---

## 🤖 Jetson Scripts

### Setup (First Time Only)
```bash
# Complete setup on Jetson Orin Nano
./scripts/jetson_setup.sh

# Interactive mode (asks about RTMPose installation)
./scripts/jetson_setup.sh

# Auto-install RTMPose (GPU acceleration)
./scripts/jetson_setup.sh --with-rtmpose

# Skip RTMPose (MediaPipe only, faster setup)
./scripts/jetson_setup.sh --skip-rtmpose

# What the script does:
# - Checks Jetson environment (JetPack, PyTorch, CUDA)
# - Creates virtual environment named 'Camera'
# - Installs base dependencies (requirements_jetson.txt)
# - Optionally installs RTMPose (20-40 min compilation)
# - Downloads YOLO models
# - Creates necessary directories

# IMPORTANT: PyTorch 2.8.0+ must be installed BEFORE running this script
# Install PyTorch: pip install torch==2.8.0 torchvision==0.23.0 \
#                  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

### Run
```bash
# Balanced mode (recommended)
./scripts/jetson_run.sh balanced

# Lite mode (lower resource usage)
./scripts/jetson_run.sh lite

# Performance mode (maximum performance)
./scripts/jetson_run.sh performance

# Run in background (daemon mode)
./scripts/jetson_run.sh balanced --daemon

# With web dashboard
./scripts/jetson_run.sh balanced --with-dashboard
```

### Stop
```bash
# Stop all running processes
./scripts/jetson_stop.sh
```

**Configuration Modes:**
- `lite`: 640x480, optimized for low power
- `balanced`: 1280x720, best balance (default)
- `performance`: 1920x1080, maximum quality

**Expected Performance:**
- Lite: 25-30 FPS @ 640x480
- Balanced: 20-25 FPS @ 1280x720
- Performance: 15-20 FPS @ 1920x1080

---

## 📋 Script Summary

| Script | Platform | Purpose |
|--------|----------|---------|
| `pc_run_gpu.sh` | PC with GPU | Run with GPU acceleration |
| `pc_run_cpu.sh` | PC (any) | Run with CPU only |
| `jetson_setup.sh` | Jetson | Complete installation |
| `jetson_run.sh` | Jetson | Run with different modes |
| `jetson_stop.sh` | Jetson | Stop running processes |

---

## 💡 Tips

1. **First time users**: Start with `pc_run_cpu.sh` or `pc_run_gpu.sh` to verify everything works
2. **Best performance**: Use `--async` flag for asynchronous pipeline (4-thread)
3. **Server deployment**: Use `--no-vis` to run without display window
4. **Jetson users**: Always run `jetson_setup.sh` first before using `jetson_run.sh`

---

## 🐛 Troubleshooting

### GPU not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA in Python
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Permission denied
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### Virtual environment not found
```bash
# PC: Create virtual environment
python3 -m venv Camera
source Camera/bin/activate
pip install -r requirements.txt

# Jetson: Run setup script
./scripts/jetson_setup.sh
```
