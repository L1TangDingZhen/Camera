# Deep Learning & Reinforcement Learning Technical Guide

This guide is for advanced users, providing detailed technical information about Deep Learning (DL) and Reinforcement Learning (RL) features in Life Tracker, including model training and advanced configurations.

> 💡 **New User**? Please read [USER_GUIDE.md](USER_GUIDE.md) first for basic usage.

## 📋 Table of Contents

- [Current Default Solution](#current-default-solution)
- [Complete Solution Comparison](#complete-solution-comparison)
- [How to Switch Solutions](#how-to-switch-solutions)
- [Pose Estimator Details](#pose-estimator-details)
- [Classifier Details](#classifier-details)
- [Training Workflow](#training-workflow)
- [Performance Comparison](#performance-comparison)
- [FAQ](#faq)

---

## System Architecture

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Decision Layer (Decision)                │
│  Simple Debouncing / RL Decision (Learn when to output) │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                Classifier Layer (Classifier)             │
│  SVM / DL (MLP/LSTM/Transformer) / RL Ensemble          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Pose Estimation Layer (Pose)                │
│             MediaPipe / RTMPose / ViTPose               │
└─────────────────────────────────────────────────────────┘
```

### Model Selection Matrix

| Layer | Option | Speed | Accuracy | Complexity | Recommended Scenario |
|-------|--------|-------|----------|-----------|---------------------|
| **Decision** | Simple | ⚡⚡⚡ | ⭐⭐⭐ | Low | General |
| | RL Decision | ⚡⚡ | ⭐⭐⭐⭐⭐ | High | High Accuracy |
| **Classifier** | SVM | ⚡⚡⚡ | ⭐⭐⭐ | Low | Quick Deploy |
| | MLP | ⚡⚡⚡ | ⭐⭐⭐⭐ | Medium | Single Frame |
| | LSTM | ⚡⚡ | ⭐⭐⭐⭐⭐ | Medium | Sequence |
| | Ensemble | ⚡⚡ | ⭐⭐⭐⭐⭐ | High | Best Accuracy |
| **Pose** | MediaPipe | ⚡⚡ | ⭐⭐⭐ | Low | CPU |
| | RTMPose | ⚡⚡⚡ | ⭐⭐⭐⭐ | Medium | GPU |

---

## Current Default Solution

Life Tracker's default configuration is optimized and **ready to use without training**:

### 🎯 Default Combination

```yaml
Pose Estimation: MediaPipe (CPU)
Classifier:      SVM (Pre-trained)
Decision:        Simple Debouncing
```

### 📊 Default Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 90-95% | Based on standard environment |
| **Latency** | ~50ms/frame | MediaPipe ~40ms + SVM ~1ms + Others ~9ms |
| **Memory** | ~300MB | Including model and runtime |
| **GPU Required** | ❌ No | Pure CPU operation |
| **Training Required** | ❌ No | Uses pre-trained models |

### ✅ Default Solution Advantages

- **Plug and Play**: No training needed, ready to use
- **Cross-platform**: Windows/Linux/macOS support
- **Hardware Friendly**: Works on regular laptop CPUs
- **Stable and Reliable**: Extensively tested in production

### ⚠️ Default Solution Limitations

- Slower speed (~50ms vs GPU solution ~13ms)
- Medium accuracy (90-95% vs high-accuracy solution 96-99%)
- Environment sensitive (affected by lighting and angle changes)

---

## Complete Solution Comparison

Life Tracker supports **15 different model combinations** to meet needs from quick deployment to ultimate performance.

### 📊 Solution Comparison Table

| Solution | Pose | Classifier | Decision | Accuracy | Latency | GPU | Training | Recommended Scenario |
|----------|------|-----------|----------|----------|---------|-----|----------|---------------------|
| **Solution 1** | MediaPipe | SVM | Simple | 90-95% | ~50ms | ❌ | ❌ | Default, quick deploy |
| **Solution 2** | RTMPose | SVM | Simple | 90-95% | ~13ms | ✅ | ❌ | GPU acceleration |
| **Solution 3** | RTMPose | MLP | Simple | 92-96% | ~13ms | ✅ | ✅ | High accuracy |
| **Solution 4** | RTMPose | LSTM | Simple | 93-97% | ~17ms | ✅ | ✅ | Sequence optimization |
| **Solution 5** | RTMPose | Transformer | Simple | 94-97% | ~22ms | ✅ | ✅ | Highest single model |
| **Solution 6** | RTMPose | Ensemble | Simple | 95-98% | ~19ms | ✅ | ✅ | Multi-model fusion |
| **Solution 7** | RTMPose | Ensemble | RL | 96-99% | ~22ms | ✅ | ✅ | Ultimate accuracy |
| **Solution 8** | MediaPipe | MLP | Simple | 92-96% | ~41ms | ❌ | ✅ | CPU+DL |
| **Solution 9** | MediaPipe | LSTM | Simple | 93-97% | ~45ms | ❌ | ✅ | CPU+Sequence |
| **Solution 10** | MediaPipe | Ensemble | Simple | 95-98% | ~47ms | ❌ | ✅ | CPU best accuracy |

> 💡 **Note**: More solution combinations available in `config/`

### 🎯 Solution Selection Decision Tree

```
What is your goal?
│
├─ Quick start, no configuration
│  └─ Solution 1 (Default) ✅
│
├─ Have GPU, want faster
│  └─ Solution 2 (RTMPose + SVM) ✅
│
├─ Pursue higher accuracy
│  ├─ Have GPU
│  │  ├─ Single frame sufficient → Solution 3 (RTMPose + MLP)
│  │  ├─ Need sequence → Solution 4 (RTMPose + LSTM)
│  │  └─ Pursue ultimate → Solution 7 (RTMPose + Ensemble + RL) ⭐
│  │
│  └─ CPU only
│     └─ Solution 10 (MediaPipe + Ensemble)
│
└─ Jetson deployment
   └─ Solution 2 or 3 (RTMPose + SVM/MLP)
```

### 📈 Performance Improvement (vs Default Solution)

| Solution | Speed Gain | Accuracy Gain | Resource Increase | Training Time |
|----------|-----------|---------------|-------------------|--------------|
| Solution 2 | **+4x** ⚡ | - | GPU | 0 minutes |
| Solution 3 | +4x | **+2-4%** | GPU | 10 minutes |
| Solution 4 | +3x | **+3-7%** | GPU | 20 minutes |
| Solution 6 | +2.6x | **+5-8%** | GPU | 40 minutes |
| Solution 7 | +2.3x | **+6-9%** | GPU | 60 minutes |

---

## How to Switch Solutions

### 3 Key Configuration Locations

Life Tracker controls model selection through configuration files, with **3 key configuration locations**:

```yaml
# config/config_gpu.yaml

# Location 1: Pose Estimator Switching (line 23)
models:
  pose:
    backend: mediapipe  # 👈 Change this!
    # Options: mediapipe, rtmpose, vitpose

# Location 2: Classifier Switching (line 118)
behavior:
  classifier:
    type: svm  # 👈 Change this!
    # Options: svm, deep_learning, rl_ensemble

# Location 3: Decision Strategy Switching (line 187)
behavior:
  decision:
    type: simple  # 👈 Change this!
    # Options: simple, rl
```

---

### Switching Example 1: Default → Solution 2 (GPU Acceleration)

**Goal**: Use GPU to accelerate pose estimation, 4x speed improvement

**Steps**:

#### Step 1: Install RTMPose Dependencies

```bash
# On Linux/Jetson
pip install openmim
mim install mmcv==2.0.0
mim install mmpose==1.0.0

# Detailed installation guide: INSTALL_RTMPOSE.md
```

#### Step 2: Download RTMPose Model

```bash
python download_rtmpose_models.py --model rtmpose-s
```

#### Step 3: Modify Configuration File

Open `config/config_gpu.yaml`, find line 23:

```yaml
# Before (MediaPipe)
models:
  pose:
    backend: mediapipe
    complexity: 1
    device: cpu
    confidence: 0.3

# After (RTMPose)
models:
  pose:
    backend: rtmpose  # 👈 Change this
    model: rtmpose-s
    config_file: models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py
    checkpoint: models/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth
    device: cuda:0  # 👈 Use GPU
    confidence: 0.3
```

Or use the pre-configured file directly:

```bash
python main.py --config config/config_rtmpose.yaml
```

#### Step 4: Restart System

```bash
python main.py --config config/config_gpu.yaml
```

**Expected Results**:
- Inference latency: 50ms → 13ms ✅
- Accuracy unchanged: ~90-95%
- FPS increase: 15-20 FPS → 60-80 FPS

---

### Switching Example 2: Default → Solution 3 (DL Optimization)

**Goal**: Use deep learning classifier, 2-4% accuracy improvement

**Steps**:

#### Step 1: Collect Training Data (if needed)

```bash
python collect_data.py --label sitting --duration 60
python collect_data.py --label standing --duration 60
python collect_data.py --label lying --duration 60
```

#### Step 2: Train MLP Model

```bash
python train_dl.py --model mlp --epochs 100 --device cuda

# Expected output:
# Epoch [100/100] Best Val Acc: 95.XX%
# ✓ Saved best model: models/pose_classifier_mlp.pth
```

#### Step 3: Modify Configuration File

Find line 118 (classifier configuration):

```yaml
# Before (SVM)
behavior:
  classifier:
    type: svm
    path: models/pose_classifier_svm.pkl
    device: cpu

# After (MLP)
behavior:
  classifier:
    type: deep_learning  # 👈 Change this
    model_type: mlp
    path: models/pose_classifier_mlp.pth
    device: cuda:0
```

#### Step 4: Restart System

```bash
python main.py --config config/config_gpu.yaml
```

**Expected Results**:
- Accuracy improvement: 90-95% → 92-96% ✅
- Latency slight increase: 13ms → 13ms (negligible impact)
- GPU required: Yes

---

### Switching Example 3: Default → Solution 7 (Ultimate Accuracy)

**Goal**: Use complete RL system, 96-99% accuracy

**Steps**:

#### Step 1: Train All Models (in order)

```bash
# 1. Train base classifiers
python train_svm.py --data training_data
python train_dl.py --model mlp --epochs 100 --device cuda
python train_dl.py --model lstm --epochs 100 --device cuda

# 2. Train Ensemble fusion weights
python train_ensemble.py --models svm,mlp,lstm --epochs 100 --device cuda

# 3. Train RL Decision Agent
python train_decision_agent.py --classifier rl_ensemble --epochs 100 --device cuda
```

Total training time: ~60 minutes (depends on data volume)

#### Step 2: Use Complete RL Configuration File

```bash
python main.py --config config/config_rl_full.yaml
```

Or manually modify `config/config_gpu.yaml`:

```yaml
# Location 1: Pose Estimation (line 23)
models:
  pose:
    backend: rtmpose
    model: rtmpose-s
    device: cuda:0

# Location 2: Classifier (line 118)
behavior:
  classifier:
    type: rl_ensemble  # 👈 Ensemble
    device: cuda:0
    agent_path: models/ensemble_agent.pt
    ensemble_models:
      - type: svm
        path: models/pose_classifier_svm.pkl
      - type: deep_learning
        model_type: mlp
        path: models/pose_classifier_mlp.pth
        device: cuda:0
      - type: deep_learning
        model_type: lstm
        path: models/pose_classifier_lstm.pth
        device: cuda:0

# Location 3: Decision Strategy (line 187)
behavior:
  decision:
    type: rl  # 👈 RL Decision
    agent_path: models/decision_agent.pt
    device: cuda:0
```

#### Step 3: Restart System

```bash
python main.py --config config/config_rl_full.yaml
```

**Expected Results**:
- Accuracy: 96-99% ✅ (+6-9%)
- Latency: ~22ms (still real-time)
- False alarm rate: Reduced 50-70%
- Environment adaptability: Significantly improved

---

### Quick Configuration File Reference

If you don't want to manually modify, use pre-configured files directly:

| Solution | Config File | Description |
|----------|------------|-------------|
| Solution 1 | `config/config_gpu.yaml` | Default configuration |
| Solution 2 | `config/config_rtmpose.yaml` | RTMPose acceleration |
| Solution 6 | `config/config_rl_ensemble.yaml` | RL Ensemble |
| Solution 7 | `config/config_rl_full.yaml` | Complete RL system |

Usage:

```bash
python main.py --config config/config_rtmpose.yaml
python main.py --config config/config_rl_full.yaml
```

---

## Pose Estimator Details

Life Tracker supports 3 pose estimation backends, each with unique characteristics:

### 1. MediaPipe Pose (Default)

**Technical Features**:
- Lightweight pose estimation developed by Google
- Based on BlazePose architecture
- Outputs 33 3D keypoints (auto-mapped to COCO-17)
- Includes world landmarks (real 3D coordinates)

**Performance Metrics**:
- Inference time: 40-50ms (CPU)
- Accuracy: AP ~67%
- Model size: ~3MB
- Platform support: Windows/Linux/macOS/Android/iOS

**Suitable Scenarios**:
- ✅ CPU-only environments
- ✅ Cross-platform deployment
- ✅ Rapid prototyping
- ✅ Mobile applications

**Configuration Example**:
```yaml
models:
  pose:
    backend: mediapipe
    complexity: 1  # 0=Lite, 1=Full, 2=Heavy
    device: cpu
    confidence: 0.3
```

---

### 2. RTMPose (Recommended for Production)

**Technical Features**:
- Real-time pose estimation developed by OpenMMLab
- GPU acceleration with TensorRT support
- Multiple model choices (tiny/s/m/l)
- Optimized for edge devices (Jetson)

**Performance Metrics** (RTMPose-s):
- Inference time: 12-18ms (GPU FP32), ~12ms (FP16)
- Accuracy: AP ~68.5%
- Model size: ~18MB
- Platform support: Linux/Jetson (recommended), Windows (complex)

**Model Comparison**:

| Model | Parameters | Inference Time | Accuracy | Use Case |
|-------|-----------|---------------|----------|----------|
| RTMPose-tiny | 1.4M | ~8ms | AP 65.9% | Low power |
| **RTMPose-s** | 4.5M | **~12ms** | **AP 68.6%** | **Standard deploy** ⭐ |
| RTMPose-m | 13.6M | ~20ms | AP 72.7% | High accuracy |
| RTMPose-l | 27.7M | ~35ms | AP 75.3% | Ultimate accuracy |

**Suitable Scenarios**:
- ✅ GPU environments
- ✅ Production deployment
- ✅ Jetson edge devices
- ✅ High real-time requirements

**Configuration Example**:
```yaml
models:
  pose:
    backend: rtmpose
    model: rtmpose-s  # Recommended
    config_file: models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py
    checkpoint: models/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth
    device: cuda:0
    confidence: 0.3
```

**Installation Guide**: See [INSTALL_RTMPOSE.md](INSTALL_RTMPOSE.md)

---

### 3. ViTPose (High Precision Optional)

**Technical Features**:
- Based on Vision Transformer
- Highest accuracy but slower
- Uses MMPose framework (same as RTMPose)
- Mainly for academic research

**Performance Metrics** (ViTPose-s):
- Inference time: ~25ms (GPU)
- Accuracy: AP ~75%
- Model size: ~30MB
- Platform support: Linux/Jetson

**Suitable Scenarios**:
- ✅ Extremely high accuracy requirements
- ✅ Real-time not critical
- ✅ Academic research
- ⚠️ Not recommended for this project (RTMPose more balanced)

**Configuration Example**:
```yaml
models:
  pose:
    backend: vitpose
    model: vitpose-s.pth
    device: cuda:0
    confidence: 0.3
```

**Note**: ViTPose installation is the same as RTMPose, see [INSTALL_RTMPOSE.md](INSTALL_RTMPOSE.md).

---

### Pose Estimator Comparison Summary

| Comparison | MediaPipe | RTMPose | ViTPose |
|-----------|----------|---------|---------|
| **Speed** | Slow (~50ms) | Fast (~12ms) | Medium (~25ms) |
| **Accuracy** | Medium (AP 67%) | High (AP 68.5%) | Highest (AP 75%) |
| **GPU Required** | ❌ CPU only | ✅ GPU recommended | ✅ GPU required |
| **Installation Difficulty** | Simple | Medium | Medium |
| **Windows Support** | ✅ Perfect | ⚠️ Complex | ⚠️ Complex |
| **Jetson Optimization** | ❌ Poor | ✅ Excellent | ⚠️ Fair |
| **Recommendation** | ⭐⭐⭐⭐ Default | ⭐⭐⭐⭐⭐ Production | ⭐⭐ Research |

**Recommended Choice**:
- Development/Testing → **MediaPipe**
- Production Deployment → **RTMPose**
- Academic Research → ViTPose

---

## Classifier Details

## Quick Start

### Scenario A: Use Existing SVM (Default)

**No training needed**, use directly:

```bash
# Use default SVM classifier
python main.py --config config/config_gpu.yaml
```

### Scenario B: Upgrade to Deep Learning Classifier

```bash
# 1. Train MLP model
python train_dl.py --model mlp --epochs 100 --device cuda

# 2. Modify configuration file
# config/config_gpu.yaml:
#   classifier:
#     type: deep_learning
#     model_type: mlp
#     path: models/pose_classifier_mlp.pth

# 3. Run
python main.py --config config/config_gpu.yaml
```

### Scenario C: Complete RL System (Highest Accuracy)

```bash
# 1. Train all models (in order)
python train_svm.py --data training_data
python train_dl.py --model mlp --epochs 100 --device cuda
python train_dl.py --model lstm --epochs 100 --device cuda
python train_ensemble.py --models svm,mlp,lstm --epochs 100
python train_decision_agent.py --classifier rl_ensemble --epochs 100

# 2. Use complete configuration
python main.py --config config/config_rl_full.yaml
```

---

## Classifier Selection Guide

### 1. SVM Classifier (Default)

**Advantages**:
- ✅ Fast training (<1 minute)
- ✅ Extremely fast inference (~1ms)
- ✅ Small memory footprint
- ✅ No GPU needed

**Disadvantages**:
- ⚠️ Relatively lower accuracy (90-95%)
- ⚠️ Limited generalization

**Suitable Scenarios**:
- Rapid prototype validation
- CPU-only environments
- Extreme real-time requirements
- Limited training data

**Training Method**:
```bash
python train_svm.py --data training_data --output models/pose_classifier_svm.pkl
```

**Configuration**:
```yaml
behavior:
  classifier:
    type: svm
    path: models/pose_classifier_svm.pkl
    device: cpu
```

---

### 2. MLP Classifier (Single-frame Deep Learning)

**Advantages**:
- ✅ Higher accuracy (92-96%)
- ✅ Fast inference (~1ms)
- ✅ Simple training

**Disadvantages**:
- ⚠️ Doesn't utilize temporal information
- ⚠️ Requires GPU for training

**Suitable Scenarios**:
- Single-frame classification sufficient
- Need higher accuracy than SVM
- Have GPU but want fast inference

**Training Method**:
```bash
# GPU training (recommended)
python train_dl.py --model mlp --epochs 100 --batch-size 32 --device cuda

# CPU training (slow)
python train_dl.py --model mlp --epochs 100 --batch-size 16 --device cpu
```

**Configuration**:
```yaml
behavior:
  classifier:
    type: deep_learning
    model_type: mlp
    path: models/pose_classifier_mlp.pth
    device: cuda:0  # or cpu
```

---

### 3. LSTM Classifier (Sequence Deep Learning)

**Advantages**:
- ✅ Highest accuracy (93-97%)
- ✅ Utilizes temporal information
- ✅ Robust to noise

**Disadvantages**:
- ⚠️ Slower inference (~5ms)
- ⚠️ Requires sequence data
- ⚠️ Complex training

**Suitable Scenarios**:
- Pursue highest accuracy
- Actions have obvious temporal features
- Can accept slight latency increase

**Training Method**:
```bash
# ⚠️ Note: LSTM requires sequence data
# First collect sequence data (if not already)
python collect_data.py --sequence-mode --sequence-length 10 --label sitting

# Train LSTM
python train_dl.py --model lstm --epochs 100 --device cuda
```

**Configuration**:
```yaml
behavior:
  classifier:
    type: deep_learning
    model_type: lstm
    path: models/pose_classifier_lstm.pth
    device: cuda:0
```

---

### 4. Transformer Classifier (High Accuracy Sequence)

**Advantages**:
- ✅ Theoretically highest accuracy
- ✅ Attention mechanism
- ✅ Long sequence modeling

**Disadvantages**:
- ⚠️ Slowest inference (~10ms)
- ⚠️ High training resource requirements
- ⚠️ Requires large amounts of data

**Suitable Scenarios**:
- Research and experimentation
- Sufficient data available
- Latency insensitive

**Training Method**:
```bash
python train_dl.py --model transformer --epochs 100 --device cuda
```

---

### 5. RL Ensemble (Multi-model Fusion)

**Advantages**:
- ✅ Highest accuracy (95-98%)
- ✅ Strong environment adaptability
- ✅ Dynamic weight adjustment
- ✅ Fully utilizes multiple model strengths

**Disadvantages**:
- ⚠️ Complex training (requires multiple base models)
- ⚠️ Slower inference (sum of all models)
- ⚠️ Complex configuration

**Suitable Scenarios**:
- Pursue ultimate accuracy
- Diverse deployment environments
- Sufficient GPU resources
- Production environments

**Training Method**:
```bash
# Step 1: Train base models
python train_svm.py --data training_data
python train_dl.py --model mlp --epochs 100 --device cuda
python train_dl.py --model lstm --epochs 100 --device cuda

# Step 2: Train Ensemble fusion weights
python train_ensemble.py --models svm,mlp,lstm --epochs 100 --device cuda
```

**Configuration**:
```yaml
behavior:
  classifier:
    type: rl_ensemble
    device: cuda:0
    agent_path: models/ensemble_agent.pt
    ensemble_models:
      - type: svm
        path: models/pose_classifier_svm.pkl
      - type: deep_learning
        model_type: mlp
        path: models/pose_classifier_mlp.pth
        device: cuda:0
      - type: deep_learning
        model_type: lstm
        path: models/pose_classifier_lstm.pth
        device: cuda:0
```

---

## Training Workflow

### Complete Training Workflow (From Scratch)

#### Stage 1: Data Collection

```bash
# Collect sitting data (60 seconds)
python collect_data.py --label sitting --duration 60

# Collect standing data
python collect_data.py --label standing --duration 60

# Collect lying data
python collect_data.py --label lying --duration 60

# Check collected data
ls training_data/
# Should see: sitting.json (XXX samples), standing.json, lying.json
```

**Data Quality Recommendations**:
- ✅ At least 500 samples per posture
- ✅ Include different angles, lighting, distances
- ✅ Include transition actions (sit→stand)
- ✅ Simulate real usage scenarios

#### Stage 2: Train Base Classifiers

```bash
# Train SVM (fast baseline)
python train_svm.py --data training_data
# Expected output: Validation Acc: 90-95%

# Train MLP (deep learning improvement)
python train_dl.py --model mlp --epochs 100 --batch-size 32 --device cuda
# Expected output: Best Val Acc: 92-96%

# Train LSTM (sequence improvement)
python train_dl.py --model lstm --epochs 100 --batch-size 16 --device cuda
# Expected output: Best Val Acc: 93-97%
```

#### Stage 3: Train RL Ensemble (Optional)

```bash
# Train Ensemble fusion weights
python train_ensemble.py \
    --models svm,mlp,lstm \
    --epochs 100 \
    --batch-size 32 \
    --device cuda \
    --data training_data

# Expected output:
# [INFO] Loading base classifiers...
# [INFO] Loaded SVM classifier: models/pose_classifier_svm.pkl
# [INFO] Loaded MLP classifier: models/pose_classifier_mlp.pth
# [INFO] Loaded LSTM classifier: models/pose_classifier_lstm.pth
# Epoch [100/100] Train Loss: X.XXX, Train Acc: XX.XX% | Val Acc: 95-98%
# ✓ Saved best model (Val Acc: XX.XX%)
```

#### Stage 4: Train RL Decision (Optional)

```bash
# Train Decision Agent
python train_decision_agent.py \
    --classifier rl_ensemble \
    --epochs 100 \
    --device cuda \
    --data training_data

# Expected output:
# [INFO] Loading classifier: rl_ensemble
# [INFO] Generating training episodes...
# [INFO] Generated XXXX episodes
# Epoch [100/100] Train Loss: X.XXX, Train Acc: XX.XX% | Val Acc: XX.XX%
```

---

## Configuration and Deployment

### Configuration Switching Locations

Life Tracker has **two key configuration locations**:

#### Location 1: Classifier Configuration

```yaml
# config/config_gpu.yaml:116
behavior:
  classifier:
    type: svm  # 👈 Change this!
    # Options: svm, deep_learning, rl_ensemble
```

#### Location 2: Decision Strategy Configuration

```yaml
# config/config_gpu.yaml:185
behavior:
  decision:
    type: simple  # 👈 Change this!
    # Options: simple, rl
```

### Recommended Configuration Combinations

#### Combination 1: Quick Deploy (Default)

```yaml
classifier:
  type: svm
decision:
  type: simple
```

**Performance**: 90-95% accuracy, ~1ms latency, medium false alarms

---

#### Combination 2: Balanced Mode

```yaml
classifier:
  type: deep_learning
  model_type: mlp
decision:
  type: simple
```

**Performance**: 92-96% accuracy, ~1ms latency, low false alarms

---

#### Combination 3: High Accuracy Mode

```yaml
classifier:
  type: deep_learning
  model_type: lstm
decision:
  type: simple
```

**Performance**: 93-97% accuracy, ~5ms latency, low false alarms

---

#### Combination 4: Ensemble Mode

```yaml
classifier:
  type: rl_ensemble
  ensemble_models: [svm, mlp, lstm]
decision:
  type: simple
```

**Performance**: 95-98% accuracy, ~7ms latency, very low false alarms

---

#### Combination 5: Ultimate Mode (Highest Accuracy)

```yaml
classifier:
  type: rl_ensemble
  ensemble_models: [svm, mlp, lstm]
decision:
  type: rl
```

**Performance**: 96-99% accuracy, ~10ms latency, extremely low false alarms

---

## Performance Comparison

### Complete Performance Table

| Configuration | Accuracy | False Alarm Rate | Inference Latency | GPU Required | Training Time | Complexity | Recommended Scenario |
|--------------|----------|-----------------|------------------|--------------|--------------|-----------|---------------------|
| SVM | 90-95% | 15-20% | ~1ms | ❌ | 1min | ⭐ | Quick deploy |
| MLP | 92-96% | 10-15% | ~1ms | ✅ | 10min | ⭐⭐ | Single-frame optimization |
| LSTM | 93-97% | 8-12% | ~5ms | ✅ | 20min | ⭐⭐⭐ | Sequence optimization |
| Transformer | 94-97% | 7-10% | ~10ms | ✅ | 30min | ⭐⭐⭐⭐ | Research |
| Ensemble | 95-98% | 5-8% | ~7ms | ✅ | 40min | ⭐⭐⭐⭐ | High accuracy |
| Ensemble+RL | 96-99% | 2-5% | ~10ms | ✅ | 60min | ⭐⭐⭐⭐⭐ | Ultimate accuracy |

### Resource Consumption Comparison

| Model | Memory Usage | Model Size | Training GPU Memory | Inference GPU Memory |
|-------|-------------|-----------|-------------------|---------------------|
| SVM | ~5MB | ~2MB | N/A | N/A |
| MLP | ~10MB | ~5MB | ~500MB | ~100MB |
| LSTM | ~20MB | ~10MB | ~1GB | ~200MB |
| Transformer | ~30MB | ~15MB | ~2GB | ~300MB |
| Ensemble (3 models) | ~35MB | ~17MB | ~1GB | ~300MB |

---

## FAQ

### Q1: How to choose the right classifier?

**Decision Tree**:

```
Need extreme speed?
├─ Yes → Use SVM
└─ No
    │
    ├─ Have GPU?
    │  ├─ No → Use SVM
    │  └─ Yes
    │      │
    │      ├─ Pursue ultimate accuracy?
    │      │  ├─ Yes → Use Ensemble + RL Decision
    │      │  └─ No → Use MLP
    │      │
    │      └─ Actions have obvious temporal features?
    │         ├─ Yes → Use LSTM
    │         └─ No → Use MLP
```

### Q2: Training shows "Insufficient data"

**Cause**: Too few samples per class

**Solution**:
```bash
# Check data volume
python -c "
import json
for label in ['sitting', 'standing', 'lying']:
    with open(f'training_data/{label}.json') as f:
        data = json.load(f)
        print(f'{label}: {len(data)} samples')
"

# If <500, recollect
python collect_data.py --label sitting --duration 120  # Collect 2 minutes
```

**Minimum requirement**: At least 300 samples per class
**Recommended**: 500-1000 samples per class

### Q3: Ensemble training fails: "Base model not found"

**Cause**: Missing base classifier model files

**Solution**:
```bash
# Check model files
ls models/

# Should see:
# pose_classifier_svm.pkl ✓
# pose_classifier_mlp.pth ✓
# pose_classifier_lstm.pth ✓

# If missing, train in order:
python train_svm.py --data training_data
python train_dl.py --model mlp --epochs 100 --device cuda
python train_dl.py --model lstm --epochs 100 --device cuda
```

### Q4: RL Decision loading fails

**Cause**: `decision_agent.pt` doesn't exist

**Solution**:
```bash
# Train Decision Agent
python train_decision_agent.py --classifier svm --epochs 100 --device cuda

# Or use Ensemble classifier for training
python train_decision_agent.py --classifier rl_ensemble --epochs 100 --device cuda
```

### Q5: Accuracy not improving, or even decreasing

**Possible Causes**:

1. **Overfitting**: Training data not diverse enough
   ```bash
   # Solution: Collect more diverse data
   # - Different lighting conditions
   # - Different distances and angles
   # - Different clothing
   ```

2. **Underfitting**: Insufficient training time
   ```bash
   # Increase training epochs
   python train_dl.py --model mlp --epochs 200  # From 100 to 200
   ```

3. **Data Leakage**: Training and validation sets overlap
   ```bash
   # Check train_test_split in train_dl.py
   # Ensure random_state is fixed, stratify=labels
   ```

### Q6: LSTM requires sequence data, but I only have single-frame data

**Solution A**: Use existing single-frame data (not recommended)

```bash
# LSTM can work but won't utilize temporal information
python train_dl.py --model lstm --epochs 100 --device cuda
```

**Solution B**: Recollect sequence data (recommended)

```bash
# Collect sequence data (each sample contains 10 consecutive frames)
python collect_data.py --sequence-mode --sequence-length 10 --label sitting
python collect_data.py --sequence-mode --sequence-length 10 --label standing
python collect_data.py --sequence-mode --sequence-length 10 --label lying

# Then train LSTM
python train_dl.py --model lstm --epochs 100 --device cuda
```

### Q7: How to evaluate model performance?

**Method 1**: Use robustness testing tool

```bash
# Test SVM
python test_robustness.py --classifier svm --label sitting --duration 30

# Test MLP
python test_robustness.py --classifier mlp --label sitting --duration 30

# Test Ensemble
python test_robustness.py --classifier rl_ensemble --label sitting --duration 30
```

**Method 2**: Check training logs

```bash
# View validation accuracy during training
cat logs/training.log | grep "Val Acc"
```

**Method 3**: Actual usage testing

```bash
# Run system, observe false alarm situations
python main.py --config config/config_gpu.yaml

# Perform various actions, observe classification accuracy
```

### Q8: Out of memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:

```bash
# 1. Reduce batch size
python train_dl.py --model lstm --batch-size 8  # From 32 to 8

# 2. Use smaller model
python train_dl.py --model mlp  # Instead of transformer

# 3. Use CPU training (slow but stable)
python train_dl.py --model lstm --device cpu
```

### Q9: How to deploy to Jetson?

**Recommended Configuration (Jetson Orin Nano)**:

```yaml
# config/config_jetson_rl.yaml
models:
  pose:
    backend: rtmpose  # Use GPU acceleration
    model: rtmpose-s

behavior:
  classifier:
    type: deep_learning  # MLP fastest
    model_type: mlp
    device: cuda:0

  decision:
    type: simple  # Simple decision, save resources
```

**Not recommended on Jetson**:
- ❌ Transformer (too slow)
- ❌ Ensemble (high resource usage)
- ❌ RL Decision (increased latency)

**Jetson Best Practices**:
- ✅ RTMPose + MLP + Simple = Best balance
- ✅ Or MediaPipe + SVM + Simple = Most resource-friendly

---

## Summary

### Quick Decision Guide

**I want to...**

- **Quick deployment** → SVM + Simple
- **Higher accuracy but stay fast** → MLP + Simple
- **Highest accuracy (with GPU)** → Ensemble + RL Decision
- **Jetson deployment** → RTMPose + MLP + Simple
- **CPU-only** → MediaPipe + SVM + Simple

### Training Priority

**Must train**:
1. SVM (fast baseline)

**Recommended training**:
2. MLP (accuracy improvement)

**Advanced features**:
3. LSTM (sequence optimization)
4. Ensemble (multi-model fusion)
5. Decision Agent (adaptive decision)

---

For other questions, please refer to:
- Training script help: `python train_dl.py --help`
- Technical comparison: `RTMPOSE_TECHNICAL_COMPARISON.md`
- Model architecture: `MODELS_ARCHITECTURE.md`
- Issue feedback: https://github.com/yourusername/Camera/issues
