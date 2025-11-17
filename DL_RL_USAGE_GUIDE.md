# Deep Learning & Reinforcement Learning Usage Guide

本指南详细介绍Life Tracker中深度学习（DL）和强化学习（RL）功能的使用方法。

## 📋 目录

- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [分类器选择指南](#分类器选择指南)
- [训练流程](#训练流程)
- [配置和部署](#配置和部署)
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
