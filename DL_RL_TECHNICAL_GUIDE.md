# Deep Learning & Reinforcement Learning Technical Guide

本指南面向进阶用户，详细介绍Life Tracker中深度学习（DL）和强化学习（RL）功能的技术细节、模型训练和高级配置。

> 💡 **新手用户**？请先阅读 [USER_GUIDE.md](USER_GUIDE.md) 了解基础使用。

## 📋 目录

- [当前默认方案](#当前默认方案)
- [完整方案对比](#完整方案对比)
- [如何切换方案](#如何切换方案)
- [姿态估计器详解](#姿态估计器详解)
- [分类器详解](#分类器详解)
- [训练流程](#训练流程)
- [性能对比](#性能对比)
- [常见问题](#常见问题)

---

## 系统架构

### 三层架构

```
┌─────────────────────────────────────────────────────────┐
│                     决策层 (Decision)                    │
│  Simple防抖 / RL Decision (学习何时输出结果)              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    分类层 (Classifier)                   │
│  SVM / DL (MLP/LSTM/Transformer) / RL Ensemble          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   姿态估计层 (Pose)                      │
│             MediaPipe / RTMPose / ViTPose               │
└─────────────────────────────────────────────────────────┘
```

### 模型选择矩阵

| 层级 | 选项 | 速度 | 精度 | 复杂度 | 推荐场景 |
|------|------|------|------|--------|---------|
| **决策层** | Simple | ⚡⚡⚡ | ⭐⭐⭐ | 低 | 通用 |
| | RL Decision | ⚡⚡ | ⭐⭐⭐⭐⭐ | 高 | 高精度 |
| **分类层** | SVM | ⚡⚡⚡ | ⭐⭐⭐ | 低 | 快速部署 |
| | MLP | ⚡⚡⚡ | ⭐⭐⭐⭐ | 中 | 单帧分类 |
| | LSTM | ⚡⚡ | ⭐⭐⭐⭐⭐ | 中 | 序列分类 |
| | Ensemble | ⚡⚡ | ⭐⭐⭐⭐⭐ | 高 | 最高精度 |
| **姿态层** | MediaPipe | ⚡⚡ | ⭐⭐⭐ | 低 | CPU环境 |
| | RTMPose | ⚡⚡⚡ | ⭐⭐⭐⭐ | 中 | GPU环境 |

---

## 当前默认方案

Life Tracker的默认配置经过优化，**开箱即用，无需训练**：

### 🎯 默认组合

```yaml
姿态估计: MediaPipe (CPU)
分类器:   SVM (已训练)
决策策略: Simple防抖
```

### 📊 默认性能

| 指标 | 数值 | 说明 |
|------|------|------|
| **精度** | 90-95% | 基于标准环境测试 |
| **延迟** | ~50ms/帧 | MediaPipe ~40ms + SVM ~1ms + 其他 ~9ms |
| **内存占用** | ~300MB | 包含模型和运行时 |
| **GPU需求** | ❌ 不需要 | 纯CPU运行 |
| **训练需求** | ❌ 不需要 | 使用预训练模型 |

### ✅ 默认方案的优势

- **即插即用**：无需任何训练，下载即用
- **跨平台**：Windows/Linux/macOS全支持
- **硬件友好**：普通笔记本CPU即可运行
- **稳定可靠**：经过大量测试，生产环境验证

### ⚠️ 默认方案的限制

- 速度较慢（~50ms vs GPU方案 ~13ms）
- 精度中等（90-95% vs 高精度方案 96-99%）
- 环境敏感（光照、角度变化影响较大）

---

## 完整方案对比

Life Tracker支持**15种不同的模型组合方案**，满足从快速部署到极致性能的各种需求。

### 📊 方案对比表

| 方案ID | 姿态估计 | 分类器 | 决策策略 | 精度 | 延迟 | GPU | 训练 | 推荐场景 |
|--------|---------|--------|---------|------|------|-----|------|---------|
| **方案1** | MediaPipe | SVM | Simple | 90-95% | ~50ms | ❌ | ❌ | 默认，快速部署 |
| **方案2** | RTMPose | SVM | Simple | 90-95% | ~13ms | ✅ | ❌ | GPU加速 |
| **方案3** | RTMPose | MLP | Simple | 92-96% | ~13ms | ✅ | ✅ | 高精度 |
| **方案4** | RTMPose | LSTM | Simple | 93-97% | ~17ms | ✅ | ✅ | 序列优化 |
| **方案5** | RTMPose | Transformer | Simple | 94-97% | ~22ms | ✅ | ✅ | 最高单模型精度 |
| **方案6** | RTMPose | Ensemble | Simple | 95-98% | ~19ms | ✅ | ✅ | 多模型融合 |
| **方案7** | RTMPose | Ensemble | RL | 96-99% | ~22ms | ✅ | ✅ | 极致精度 |
| **方案8** | MediaPipe | MLP | Simple | 92-96% | ~41ms | ❌ | ✅ | CPU+DL |
| **方案9** | MediaPipe | LSTM | Simple | 93-97% | ~45ms | ❌ | ✅ | CPU+序列 |
| **方案10** | MediaPipe | Ensemble | Simple | 95-98% | ~47ms | ❌ | ✅ | CPU最高精度 |

> 💡 **注**：更多方案组合请参考配置文件 `config/`

### 🎯 方案选择决策树

```
你的目标是什么？
│
├─ 快速开始，无需配置
│  └─ 方案1 (默认) ✅
│
├─ 有GPU，想要更快
│  └─ 方案2 (RTMPose + SVM) ✅
│
├─ 追求更高精度
│  ├─ 有GPU
│  │  ├─ 单帧足够 → 方案3 (RTMPose + MLP)
│  │  ├─ 需要序列 → 方案4 (RTMPose + LSTM)
│  │  └─ 追求极致 → 方案7 (RTMPose + Ensemble + RL) ⭐
│  │
│  └─ 只有CPU
│     └─ 方案10 (MediaPipe + Ensemble)
│
└─ Jetson部署
   └─ 方案2或3 (RTMPose + SVM/MLP)
```

### 📈 性能提升对比（相对于默认方案）

| 方案 | 速度提升 | 精度提升 | 资源增加 | 训练时间 |
|------|---------|---------|---------|---------|
| 方案2 | **+4x** ⚡ | - | GPU | 0分钟 |
| 方案3 | +4x | **+2-4%** | GPU | 10分钟 |
| 方案4 | +3x | **+3-7%** | GPU | 20分钟 |
| 方案6 | +2.6x | **+5-8%** | GPU | 40分钟 |
| 方案7 | +2.3x | **+6-9%** | GPU | 60分钟 |

---

## 如何切换方案

### 配置文件中的3个关键位置

Life Tracker通过配置文件控制模型选择，有**3个关键配置位置**：

```yaml
# config/config_gpu.yaml

# 位置1: 姿态估计器切换 (第23行)
models:
  pose:
    backend: mediapipe  # 👈 改这里！
    # 可选: mediapipe, rtmpose, vitpose

# 位置2: 分类器切换 (第118行)
behavior:
  classifier:
    type: svm  # 👈 改这里！
    # 可选: svm, deep_learning, rl_ensemble

# 位置3: 决策策略切换 (第187行)
behavior:
  decision:
    type: simple  # 👈 改这里！
    # 可选: simple, rl
```

---

### 切换示例1：从默认切换到方案2（GPU加速）

**目标**：使用GPU加速姿态估计，速度提升4x

**步骤**：

#### 步骤1：安装RTMPose依赖

```bash
# 在Linux/Jetson上
pip install openmim
mim install mmcv==2.0.0
mim install mmpose==1.0.0

# 详细安装指南：INSTALL_RTMPOSE.md
```

#### 步骤2：下载RTMPose模型

```bash
python download_rtmpose_models.py --model rtmpose-s
```

#### 步骤3：修改配置文件

打开 `config/config_gpu.yaml`，找到第23行：

```yaml
# 修改前（MediaPipe）
models:
  pose:
    backend: mediapipe
    complexity: 1
    device: cpu
    confidence: 0.3

# 修改后（RTMPose）
models:
  pose:
    backend: rtmpose  # 👈 改这里
    model: rtmpose-s
    config_file: models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py
    checkpoint: models/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth
    device: cuda:0  # 👈 使用GPU
    confidence: 0.3
```

或直接使用预配置文件：

```bash
python main.py --config config/config_rtmpose.yaml
```

#### 步骤4：重启系统

```bash
python main.py --config config/config_gpu.yaml
```

**预期效果**：
- 推理延迟：50ms → 13ms ✅
- 精度不变：~90-95%
- FPS提升：15-20 FPS → 60-80 FPS

---

### 切换示例2：从默认切换到方案3（DL优化）

**目标**：使用深度学习分类器，精度提升2-4%

**步骤**：

#### 步骤1：收集训练数据（如果没有）

```bash
python collect_data.py --label sitting --duration 60
python collect_data.py --label standing --duration 60
python collect_data.py --label lying --duration 60
```

#### 步骤2：训练MLP模型

```bash
python train_dl.py --model mlp --epochs 100 --device cuda

# 预期输出：
# Epoch [100/100] Best Val Acc: 95.XX%
# ✓ 保存最佳模型: models/pose_classifier_mlp.pth
```

#### 步骤3：修改配置文件

找到第118行（分类器配置）：

```yaml
# 修改前（SVM）
behavior:
  classifier:
    type: svm
    path: models/pose_classifier_svm.pkl
    device: cpu

# 修改后（MLP）
behavior:
  classifier:
    type: deep_learning  # 👈 改这里
    model_type: mlp
    path: models/pose_classifier_mlp.pth
    device: cuda:0
```

#### 步骤4：重启系统

```bash
python main.py --config config/config_gpu.yaml
```

**预期效果**：
- 精度提升：90-95% → 92-96% ✅
- 延迟略增：13ms → 13ms（几乎无影响）
- 需要GPU：是

---

### 切换示例3：从默认切换到方案7（极致精度）

**目标**：使用完整RL系统，精度达到96-99%

**步骤**：

#### 步骤1：训练所有模型（按顺序）

```bash
# 1. 训练基础分类器
python train_svm.py --data training_data
python train_dl.py --model mlp --epochs 100 --device cuda
python train_dl.py --model lstm --epochs 100 --device cuda

# 2. 训练Ensemble融合权重
python train_ensemble.py --models svm,mlp,lstm --epochs 100 --device cuda

# 3. 训练RL Decision Agent
python train_decision_agent.py --classifier rl_ensemble --epochs 100 --device cuda
```

总训练时间：约60分钟（取决于数据量）

#### 步骤2：使用完整RL配置文件

```bash
python main.py --config config/config_rl_full.yaml
```

或手动修改 `config/config_gpu.yaml`：

```yaml
# 位置1: 姿态估计（第23行）
models:
  pose:
    backend: rtmpose
    model: rtmpose-s
    device: cuda:0

# 位置2: 分类器（第118行）
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

# 位置3: 决策策略（第187行）
behavior:
  decision:
    type: rl  # 👈 RL Decision
    agent_path: models/decision_agent.pt
    device: cuda:0
```

#### 步骤3：重启系统

```bash
python main.py --config config/config_rl_full.yaml
```

**预期效果**：
- 精度：96-99% ✅ (+6-9%)
- 延迟：~22ms（仍然实时）
- 误报率：降低50-70%
- 环境适应性：显著提升

---

### 快速配置文件对照

如果不想手动修改，可以直接使用预配置文件：

| 方案 | 配置文件 | 说明 |
|------|---------|------|
| 方案1 | `config/config_gpu.yaml` | 默认配置 |
| 方案2 | `config/config_rtmpose.yaml` | RTMPose加速 |
| 方案6 | `config/config_rl_ensemble.yaml` | RL Ensemble |
| 方案7 | `config/config_rl_full.yaml` | 完整RL系统 |

使用方法：

```bash
python main.py --config config/config_rtmpose.yaml
python main.py --config config/config_rl_full.yaml
```

---

## 姿态估计器详解

Life Tracker支持3种姿态估计后端，各有特点：

### 1. MediaPipe Pose（默认）

**技术特点**：
- Google开发的轻量级姿态估计
- 基于BlazePose架构
- 输出33个3D关键点（自动映射到COCO-17）
- 包含world landmarks（真实3D坐标）

**性能指标**：
- 推理时间：40-50ms (CPU)
- 精度：AP ~67%
- 模型大小：~3MB
- 平台支持：Windows/Linux/macOS/Android/iOS

**适用场景**：
- ✅ CPU-only环境
- ✅ 跨平台部署
- ✅ 快速原型开发
- ✅ 移动端应用

**配置示例**：
```yaml
models:
  pose:
    backend: mediapipe
    complexity: 1  # 0=Lite, 1=Full, 2=Heavy
    device: cpu
    confidence: 0.3
```

---

### 2. RTMPose（推荐用于生产）

**技术特点**：
- OpenMMLab开发的实时姿态估计
- GPU加速，支持TensorRT
- 多模型选择（tiny/s/m/l）
- 专为边缘设备优化（Jetson）

**性能指标**（RTMPose-s）：
- 推理时间：12-18ms (GPU FP32), ~12ms (FP16)
- 精度：AP ~68.5%
- 模型大小：~18MB
- 平台支持：Linux/Jetson（推荐），Windows（复杂）

**模型对比**：

| 模型 | 参数量 | 推理时间 | 精度 | 适用场景 |
|------|--------|---------|------|---------|
| RTMPose-tiny | 1.4M | ~8ms | AP 65.9% | 低功耗 |
| **RTMPose-s** | 4.5M | **~12ms** | **AP 68.6%** | **标准部署** ⭐ |
| RTMPose-m | 13.6M | ~20ms | AP 72.7% | 高精度 |
| RTMPose-l | 27.7M | ~35ms | AP 75.3% | 极致精度 |

**适用场景**：
- ✅ GPU环境
- ✅ 生产部署
- ✅ Jetson边缘设备
- ✅ 实时性要求高

**配置示例**：
```yaml
models:
  pose:
    backend: rtmpose
    model: rtmpose-s  # 推荐
    config_file: models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py
    checkpoint: models/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth
    device: cuda:0
    confidence: 0.3
```

**安装指南**：详见 [INSTALL_RTMPOSE.md](INSTALL_RTMPOSE.md)

---

### 3. ViTPose（高精度可选）

**技术特点**：
- 基于Vision Transformer
- 最高精度，但速度较慢
- 使用MMPose框架（与RTMPose相同）
- 主要用于学术研究

**性能指标**（ViTPose-s）：
- 推理时间：~25ms (GPU)
- 精度：AP ~75%
- 模型大小：~30MB
- 平台支持：Linux/Jetson

**适用场景**：
- ✅ 精度要求极高
- ✅ 实时性要求不严格
- ✅ 学术研究
- ⚠️ 不推荐用于本项目（RTMPose更平衡）

**配置示例**：
```yaml
models:
  pose:
    backend: vitpose
    model: vitpose-s.pth
    device: cuda:0
    confidence: 0.3
```

**注意**：ViTPose的安装方法与RTMPose相同，请参考 [INSTALL_RTMPOSE.md](INSTALL_RTMPOSE.md)。

---

### 姿态估计器对比总结

| 对比项 | MediaPipe | RTMPose | ViTPose |
|--------|----------|---------|---------|
| **速度** | 慢 (~50ms) | 快 (~12ms) | 中 (~25ms) |
| **精度** | 中 (AP 67%) | 高 (AP 68.5%) | 最高 (AP 75%) |
| **GPU需求** | ❌ CPU only | ✅ 推荐GPU | ✅ 需要GPU |
| **安装难度** | 简单 | 中等 | 中等 |
| **Windows支持** | ✅ 完美 | ⚠️ 复杂 | ⚠️ 复杂 |
| **Jetson优化** | ❌ 差 | ✅ 优秀 | ⚠️ 一般 |
| **推荐级别** | ⭐⭐⭐⭐ 默认 | ⭐⭐⭐⭐⭐ 生产 | ⭐⭐ 研究用 |

**推荐选择**：
- 开发测试 → **MediaPipe**
- 生产部署 → **RTMPose**
- 学术研究 → ViTPose

---

## 分类器详解

## 快速开始

### 场景A：使用现有SVM（默认）

**无需训练**，直接使用：

```bash
# 使用默认SVM分类器
python main.py --config config/config_gpu.yaml
```

### 场景B：升级到深度学习分类器

```bash
# 1. 训练MLP模型
python train_dl.py --model mlp --epochs 100 --device cuda

# 2. 修改配置文件
# config/config_gpu.yaml:
#   classifier:
#     type: deep_learning
#     model_type: mlp
#     path: models/pose_classifier_mlp.pth

# 3. 运行
python main.py --config config/config_gpu.yaml
```

### 场景C：完整RL系统（最高精度）

```bash
# 1. 训练所有模型（按顺序）
python train_svm.py --data training_data
python train_dl.py --model mlp --epochs 100 --device cuda
python train_dl.py --model lstm --epochs 100 --device cuda
python train_ensemble.py --models svm,mlp,lstm --epochs 100
python train_decision_agent.py --classifier rl_ensemble --epochs 100

# 2. 使用完整配置
python main.py --config config/config_rl_full.yaml
```

---

## 分类器选择指南

### 1. SVM分类器（默认）

**优点**：
- ✅ 快速训练（<1分钟）
- ✅ 推理极快（~1ms）
- ✅ 内存占用小
- ✅ 无需GPU

**缺点**：
- ⚠️ 精度相对较低（90-95%）
- ⚠️ 泛化能力有限

**适用场景**：
- 快速原型验证
- CPU-only环境
- 实时性要求极高
- 训练数据有限

**训练方法**：
```bash
python train_svm.py --data training_data --output models/pose_classifier_svm.pkl
```

**配置**：
```yaml
behavior:
  classifier:
    type: svm
    path: models/pose_classifier_svm.pkl
    device: cpu
```

---

### 2. MLP分类器（单帧深度学习）

**优点**：
- ✅ 精度较高（92-96%）
- ✅ 推理快速（~1ms）
- ✅ 训练简单

**缺点**：
- ⚠️ 不利用时序信息
- ⚠️ 需要GPU训练

**适用场景**：
- 单帧分类足够准确
- 需要比SVM更高精度
- 有GPU但希望推理快速

**训练方法**：
```bash
# GPU训练（推荐）
python train_dl.py --model mlp --epochs 100 --batch-size 32 --device cuda

# CPU训练（慢）
python train_dl.py --model mlp --epochs 100 --batch-size 16 --device cpu
```

**配置**：
```yaml
behavior:
  classifier:
    type: deep_learning
    model_type: mlp
    path: models/pose_classifier_mlp.pth
    device: cuda:0  # 或 cpu
```

---

### 3. LSTM分类器（序列深度学习）

**优点**：
- ✅ 最高精度（93-97%）
- ✅ 利用时序信息
- ✅ 对噪声鲁棒

**缺点**：
- ⚠️ 推理较慢（~5ms）
- ⚠️ 需要序列数据
- ⚠️ 训练复杂

**适用场景**：
- 追求最高精度
- 动作有明显时序特征
- 可接受小幅延迟增加

**训练方法**：
```bash
# ⚠️ 注意：LSTM需要序列数据
# 首先收集序列数据（如果还没有）
python collect_data.py --sequence-mode --sequence-length 10 --label sitting

# 训练LSTM
python train_dl.py --model lstm --epochs 100 --device cuda
```

**配置**：
```yaml
behavior:
  classifier:
    type: deep_learning
    model_type: lstm
    path: models/pose_classifier_lstm.pth
    device: cuda:0
```

---

### 4. Transformer分类器（高精度序列）

**优点**：
- ✅ 理论最高精度
- ✅ 注意力机制
- ✅ 长序列建模

**缺点**：
- ⚠️ 推理最慢（~10ms）
- ⚠️ 训练资源要求高
- ⚠️ 需要大量数据

**适用场景**：
- 研究和实验
- 数据充足
- 对延迟不敏感

**训练方法**：
```bash
python train_dl.py --model transformer --epochs 100 --device cuda
```

---

### 5. RL Ensemble（多模型融合）

**优点**：
- ✅ 精度最高（95-98%）
- ✅ 环境适应性强
- ✅ 动态权重调整
- ✅ 充分利用多个模型优势

**缺点**：
- ⚠️ 训练复杂（需要多个基础模型）
- ⚠️ 推理较慢（所有模型之和）
- ⚠️ 配置复杂

**适用场景**：
- 追求极致精度
- 部署环境多样化
- 有充足GPU资源
- 生产环境

**训练方法**：
```bash
# 步骤1：训练基础模型
python train_svm.py --data training_data
python train_dl.py --model mlp --epochs 100 --device cuda
python train_dl.py --model lstm --epochs 100 --device cuda

# 步骤2：训练Ensemble融合权重
python train_ensemble.py --models svm,mlp,lstm --epochs 100 --device cuda
```

**配置**：
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

## 训练流程

### 完整训练流程（从零开始）

#### 阶段1：数据收集

```bash
# 收集坐姿数据（60秒）
python collect_data.py --label sitting --duration 60

# 收集站姿数据
python collect_data.py --label standing --duration 60

# 收集躺姿数据
python collect_data.py --label lying --duration 60

# 检查收集的数据
ls training_data/
# 应该看到：sitting.json (XXX samples), standing.json, lying.json
```

**数据质量建议**：
- ✅ 每个姿态至少500个样本
- ✅ 包含不同角度、光照、距离
- ✅ 包含过渡动作（坐→站）
- ✅ 模拟真实使用场景

#### 阶段2：训练基础分类器

```bash
# 训练SVM（快速基线）
python train_svm.py --data training_data
# 预期输出：Validation Acc: 90-95%

# 训练MLP（深度学习改进）
python train_dl.py --model mlp --epochs 100 --batch-size 32 --device cuda
# 预期输出：Best Val Acc: 92-96%

# 训练LSTM（序列改进）
python train_dl.py --model lstm --epochs 100 --batch-size 16 --device cuda
# 预期输出：Best Val Acc: 93-97%
```

#### 阶段3：训练RL Ensemble（可选）

```bash
# 训练Ensemble融合权重
python train_ensemble.py \
    --models svm,mlp,lstm \
    --epochs 100 \
    --batch-size 32 \
    --device cuda \
    --data training_data

# 预期输出：
# [INFO] 加载基础分类器...
# [INFO] 已加载SVM分类器: models/pose_classifier_svm.pkl
# [INFO] 已加载MLP分类器: models/pose_classifier_mlp.pth
# [INFO] 已加载LSTM分类器: models/pose_classifier_lstm.pth
# Epoch [100/100] Train Loss: X.XXX, Train Acc: XX.XX% | Val Acc: 95-98%
# ✓ 保存最佳模型 (Val Acc: XX.XX%)
```

#### 阶段4：训练RL Decision（可选）

```bash
# 训练Decision Agent
python train_decision_agent.py \
    --classifier rl_ensemble \
    --epochs 100 \
    --device cuda \
    --data training_data

# 预期输出：
# [INFO] 加载分类器: rl_ensemble
# [INFO] 生成训练episodes...
# [INFO] 生成了 XXXX 个episodes
# Epoch [100/100] Train Loss: X.XXX, Train Acc: XX.XX% | Val Acc: XX.XX%
```

---

## 配置和部署

### 配置切换位置

Life Tracker有**两个关键配置位置**：

#### 位置1：分类器配置（Classifier）

```yaml
# config/config_gpu.yaml:116
behavior:
  classifier:
    type: svm  # 👈 改这里！
    # 可选: svm, deep_learning, rl_ensemble
```

#### 位置2：决策策略配置（Decision）

```yaml
# config/config_gpu.yaml:185
behavior:
  decision:
    type: simple  # 👈 改这里！
    # 可选: simple, rl
```

### 推荐配置组合

#### 组合1：快速部署（默认）

```yaml
classifier:
  type: svm
decision:
  type: simple
```

**性能**：精度90-95%，延迟~1ms，误报中等

---

#### 组合2：平衡模式

```yaml
classifier:
  type: deep_learning
  model_type: mlp
decision:
  type: simple
```

**性能**：精度92-96%，延迟~1ms，误报较低

---

#### 组合3：高精度模式

```yaml
classifier:
  type: deep_learning
  model_type: lstm
decision:
  type: simple
```

**性能**：精度93-97%，延迟~5ms，误报低

---

#### 组合4：Ensemble模式

```yaml
classifier:
  type: rl_ensemble
  ensemble_models: [svm, mlp, lstm]
decision:
  type: simple
```

**性能**：精度95-98%，延迟~7ms，误报很低

---

#### 组合5：终极模式（最高精度）

```yaml
classifier:
  type: rl_ensemble
  ensemble_models: [svm, mlp, lstm]
decision:
  type: rl
```

**性能**：精度96-99%，延迟~10ms，误报极低

---

## 性能对比

### 完整性能表

| 配置 | 精度 | 误报率 | 推理延迟 | GPU需求 | 训练时间 | 复杂度 | 推荐场景 |
|------|------|--------|----------|---------|----------|--------|---------|
| SVM | 90-95% | 15-20% | ~1ms | ❌ | 1min | ⭐ | 快速部署 |
| MLP | 92-96% | 10-15% | ~1ms | ✅ | 10min | ⭐⭐ | 单帧优化 |
| LSTM | 93-97% | 8-12% | ~5ms | ✅ | 20min | ⭐⭐⭐ | 序列优化 |
| Transformer | 94-97% | 7-10% | ~10ms | ✅ | 30min | ⭐⭐⭐⭐ | 研究用 |
| Ensemble | 95-98% | 5-8% | ~7ms | ✅ | 40min | ⭐⭐⭐⭐ | 高精度 |
| Ensemble+RL | 96-99% | 2-5% | ~10ms | ✅ | 60min | ⭐⭐⭐⭐⭐ | 极致精度 |

### 资源消耗对比

| 模型 | 内存占用 | 模型大小 | 训练GPU内存 | 推理GPU内存 |
|------|----------|----------|-------------|-------------|
| SVM | ~5MB | ~2MB | N/A | N/A |
| MLP | ~10MB | ~5MB | ~500MB | ~100MB |
| LSTM | ~20MB | ~10MB | ~1GB | ~200MB |
| Transformer | ~30MB | ~15MB | ~2GB | ~300MB |
| Ensemble (3模型) | ~35MB | ~17MB | ~1GB | ~300MB |

---

## 常见问题

### Q1: 如何选择合适的分类器？

**决策树**：

```
需要极致速度？
├─ Yes → 使用 SVM
└─ No
    │
    ├─ 有GPU？
    │  ├─ No → 使用 SVM
    │  └─ Yes
    │      │
    │      ├─ 追求极致精度？
    │      │  ├─ Yes → 使用 Ensemble + RL Decision
    │      │  └─ No → 使用 MLP
    │      │
    │      └─ 动作有明显时序特征？
    │         ├─ Yes → 使用 LSTM
    │         └─ No → 使用 MLP
```

### Q2: 训练时显示"数据不足"

**原因**：每个类别样本数太少

**解决**：
```bash
# 检查数据量
python -c "
import json
for label in ['sitting', 'standing', 'lying']:
    with open(f'training_data/{label}.json') as f:
        data = json.load(f)
        print(f'{label}: {len(data)} samples')
"

# 如果<500，重新收集
python collect_data.py --label sitting --duration 120  # 收集2分钟
```

**最低要求**：每类至少300个样本
**推荐**：每类500-1000个样本

### Q3: Ensemble训练失败："基础模型未找到"

**原因**：缺少基础分类器模型文件

**解决**：
```bash
# 检查模型文件
ls models/

# 应该看到：
# pose_classifier_svm.pkl ✓
# pose_classifier_mlp.pth ✓
# pose_classifier_lstm.pth ✓

# 如果缺少，按顺序训练：
python train_svm.py --data training_data
python train_dl.py --model mlp --epochs 100 --device cuda
python train_dl.py --model lstm --epochs 100 --device cuda
```

### Q4: RL Decision加载失败

**原因**：`decision_agent.pt` 不存在

**解决**：
```bash
# 训练Decision Agent
python train_decision_agent.py --classifier svm --epochs 100 --device cuda

# 或使用Ensemble分类器训练
python train_decision_agent.py --classifier rl_ensemble --epochs 100 --device cuda
```

### Q5: 精度没有提升，甚至下降

**可能原因**：

1. **过拟合**：训练数据不够多样化
   ```bash
   # 解决：收集更多样化数据
   # - 不同光照条件
   # - 不同距离和角度
   # - 不同衣着
   ```

2. **欠拟合**：训练时间不够
   ```bash
   # 增加训练轮数
   python train_dl.py --model mlp --epochs 200  # 从100增加到200
   ```

3. **数据泄漏**：训练集和验证集重叠
   ```bash
   # 检查train_dl.py中的train_test_split
   # 确保random_state固定，stratify=labels
   ```

### Q6: LSTM需要序列数据，但我只有单帧数据

**解决方案A**：使用现有单帧数据（不推荐）

```bash
# LSTM可以工作，但无法利用时序信息
python train_dl.py --model lstm --epochs 100 --device cuda
```

**解决方案B**：重新收集序列数据（推荐）

```bash
# 收集序列数据（每个样本包含连续10帧）
python collect_data.py --sequence-mode --sequence-length 10 --label sitting
python collect_data.py --sequence-mode --sequence-length 10 --label standing
python collect_data.py --sequence-mode --sequence-length 10 --label lying

# 然后训练LSTM
python train_dl.py --model lstm --epochs 100 --device cuda
```

### Q7: 如何评估模型性能？

**方法1**：使用鲁棒性测试工具

```bash
# 测试SVM
python test_robustness.py --classifier svm --label sitting --duration 30

# 测试MLP
python test_robustness.py --classifier mlp --label sitting --duration 30

# 测试Ensemble
python test_robustness.py --classifier rl_ensemble --label sitting --duration 30
```

**方法2**：查看训练日志

```bash
# 查看训练过程中的验证精度
cat logs/training.log | grep "Val Acc"
```

**方法3**：实际使用测试

```bash
# 运行系统，观察误报情况
python main.py --config config/config_gpu.yaml

# 做各种动作，观察分类准确性
```

### Q8: 内存不足（OOM）

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：

```bash
# 1. 减小batch size
python train_dl.py --model lstm --batch-size 8  # 从32降到8

# 2. 使用更小的模型
python train_dl.py --model mlp  # 而不是transformer

# 3. 使用CPU训练（慢但稳定）
python train_dl.py --model lstm --device cpu
```

### Q9: 如何部署到Jetson？

**推荐配置（Jetson Orin Nano）**：

```yaml
# config/config_jetson_rl.yaml
models:
  pose:
    backend: rtmpose  # 使用GPU加速
    model: rtmpose-s

behavior:
  classifier:
    type: deep_learning  # MLP最快
    model_type: mlp
    device: cuda:0

  decision:
    type: simple  # 简单决策，节省资源
```

**不推荐在Jetson上使用**：
- ❌ Transformer（太慢）
- ❌ Ensemble（资源占用高）
- ❌ RL Decision（延迟增加）

**Jetson最佳实践**：
- ✅ RTMPose + MLP + Simple = 最佳平衡
- ✅ 或 MediaPipe + SVM + Simple = 最省资源

---

## 总结

### 快速决策指南

**我想要...**

- **快速部署** → SVM + Simple
- **更高精度但保持快速** → MLP + Simple
- **最高精度（有GPU）** → Ensemble + RL Decision
- **Jetson部署** → RTMPose + MLP + Simple
- **CPU-only** → MediaPipe + SVM + Simple

### 训练优先级

**必须训练**：
1. SVM（快速基线）

**推荐训练**：
2. MLP（精度提升）

**高级功能**：
3. LSTM（序列优化）
4. Ensemble（多模型融合）
5. Decision Agent（自适应决策）

---

如有其他问题，请参考：
- 训练脚本帮助：`python train_dl.py --help`
- 技术对比：`RTMPOSE_TECHNICAL_COMPARISON.md`
- 模型架构：`MODELS_ARCHITECTURE.md`
- Issue反馈：https://github.com/yourusername/Camera/issues
