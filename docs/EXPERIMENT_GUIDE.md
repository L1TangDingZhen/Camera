# 深度学习实验指南

## 🎯 目标

让你能在**1-2天内**完成一次完整的DL实验，对比SVM vs LSTM的效果。

---

## 📋 实验流程（3步走）

### Step 1: 收集训练数据（1-2小时）

#### 方案A: 手动标注（推荐，质量高）

```bash
# 启动数据收集器
python scripts/collect_training_data.py \
    --duration 600 \
    --output data/my_training.npz

# 操作说明：
# - 按 's' = 坐姿
# - 按 't' = 站姿
# - 按 'l' = 躺姿
# - 按 'q' = 退出

# 收集建议：
# 1. 每种姿态至少2分钟
# 2. 包含各种变化：
#    - 坐：正常坐、前倾、后仰、侧身
#    - 站：正常站、半蹲、伸懒腰
#    - 躺：平躺、侧躺
# 3. 包含过渡过程（站→坐→躺）
```

#### 方案B: 自动标注（快速，但依赖SVM质量）

```bash
# 使用现有SVM自动标注
python scripts/collect_training_data.py \
    --auto-label \
    --duration 600 \
    --output data/auto_labeled.npz

# 适合：SVM准确率已经80%+，快速扩展数据集
```

#### 数据质量检查

```python
# 检查收集的数据
python -c "
import numpy as np
data = np.load('data/my_training.npz')
print('样本数:', len(data['labels']))
print('分布:', np.bincount(data['labels']))
# 期望: 每个类别至少300样本
"
```

---

### Step 2: 训练模型（30分钟 - 2小时）

#### 快速验证：MLP（30分钟）

```bash
# 训练最简单的MLP，快速验证pipeline
python scripts/train_dl_classifier.py \
    --model mlp \
    --data data/my_training.npz \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.001 \
    --device cuda  # 如果有GPU
```

**期望输出：**
```
[INFO] 训练集: 800 样本
[INFO] 验证集: 200 样本
Epoch 30/30
Training: 100%|██████| loss: 0.1234, acc: 92.5%
Validation: 100%|██████| loss: 0.1567, acc: 88.0%

✓ 最佳模型已保存: models/pose_classifier_mlp.pth (Val Acc: 88.0%)

Classification Report:
              precision    recall  f1-score
Sitting          0.90      0.92      0.91
Standing         0.85      0.83      0.84
Lying            0.89      0.88      0.88
```

#### 完整训练：LSTM（1-2小时）

```bash
# 训练LSTM，时序建模
python scripts/train_dl_classifier.py \
    --model lstm \
    --data data/my_training.npz \
    --epochs 50 \
    --sequence-length 10 \
    --batch-size 16 \
    --lr 0.0005 \
    --device cuda

# 会生成：
# - models/pose_classifier_lstm.pth (模型)
# - results/lstm_YYYYMMDD_HHMMSS/ (训练曲线、混淆矩阵)
```

**训练监控：**
- 看 `results/*/training_history.png` 检查是否过拟合
- 如果 val_loss 上升 → 减少epochs或增加数据
- 如果 train_acc 很高但 val_acc 很低 → 过拟合，增加dropout

---

### Step 3: 测试对比（30分钟）

#### 实时对比测试

```bash
# 同时运行DL和SVM，实时对比
python scripts/test_dl_model.py \
    --model models/pose_classifier_lstm.pth \
    --type lstm \
    --compare

# 会显示：
# - 左侧：DL预测 + 概率条
# - 右侧：SVM预测 + 概率条
# - 底部：是否一致
# - FPS统计
```

**观察重点：**
1. **一致率**：DL和SVM预测一致的比例
   - 80%+ → 基本相似
   - 60-80% → 有明显差异，重点关注不一致的case
   - <60% → 可能训练有问题

2. **稳定性**：状态切换频率
   - LSTM应该更平滑，抖动更少
   - 如果DL反而更抖动 → 可能sequence_length太短

3. **边界case**：
   - 前倾工作
   - 非标准坐姿（葛优躺）
   - 站坐过渡

#### 数据集评估

```bash
# 在测试集上定量评估
python scripts/test_dl_model.py \
    --model models/pose_classifier_lstm.pth \
    --type lstm \
    --test-data data/test_set.npz

# 输出：
# - 准确率
# - 每个类别的精确率/召回率
# - 混淆矩阵
```

---

## 📊 如何判断实验成功？

### 成功指标

| 指标 | SVM基线 | DL目标 | 说明 |
|------|---------|--------|------|
| **准确率** | 80-90% | +3-5% | 整体提升 |
| **过渡识别** | 60-70% | +15-20% | 关键提升点 |
| **稳定性** | 基准 | -30%抖动 | 状态切换次数减少 |
| **推理速度** | 0.8ms | <5ms | 不影响实时性 |

### 对比checklist

做完实验后，填这个表：

```
实验日期: __________
数据量: __________ 样本

                    SVM       LSTM      提升
-------------------------------------------------
整体准确率          ___%      ___%     +___%
sitting准确率       ___%      ___%     +___%
standing准确率      ___%      ___%     +___%
lying准确率         ___%      ___%     +___%

前倾工作识别        ___%      ___%     +___%
葛优躺识别          ___%      ___%     +___%
站坐过渡            ___%      ___%     +___%

推理时间            ___ms     ___ms    ___ms
完整FPS             ___       ___      ___

一致率: ___%
状态切换次数: SVM=___, LSTM=___

结论: □ LSTM更好  □ 差不多  □ SVM更好
```

---

## 🐛 常见问题

### Q1: 训练准确率很高，但验证准确率很低

**原因**: 过拟合

**解决**:
```bash
# 方法1: 增加数据
python scripts/collect_training_data.py --duration 1200

# 方法2: 数据增强
python scripts/train_dl_classifier.py --augment

# 方法3: 减少epochs
--epochs 30  # 而不是50
```

### Q2: 训练很慢

**原因**: 数据量大或没用GPU

**解决**:
```bash
# 检查是否用了GPU
python -c "import torch; print(torch.cuda.is_available())"

# 如果False，要么装CUDA，要么：
--device cpu --batch-size 8  # 用CPU但减小batch
```

### Q3: 模型预测全是一个类别

**原因**: 数据不平衡

**解决**:
```python
# 检查数据分布
data = np.load('data/my_training.npz')
print(np.bincount(data['labels']))

# 如果 [800, 100, 100] 这样极度不平衡
# → 重新收集，确保每个类别至少30%
```

### Q4: LSTM效果不如MLP

**原因**: sequence_length不合适或数据时序性不强

**解决**:
```bash
# 尝试不同序列长度
--sequence-length 5   # 更短
--sequence-length 20  # 更长

# 或者就用MLP（有时简单更好）
```

### Q5: 实时测试时FPS很低

**原因**: LSTM推理慢

**解决**:
```python
# 方法1: 用更小的模型
--model mlp  # 最快

# 方法2: 减少序列长度
--sequence-length 5

# 方法3: 优化推理（TODO: ONNX）
```

---

## 🎓 进阶实验

### 实验1: 序列长度对比

```bash
# 训练不同序列长度的LSTM
for seq_len in 5 10 15 20 30; do
    python scripts/train_dl_classifier.py \
        --model lstm \
        --sequence-length $seq_len \
        --output models/lstm_seq${seq_len}.pth
done

# 对比效果
```

### 实验2: 数据量影响

```bash
# 用不同数据量训练
for samples in 500 1000 2000 5000; do
    # 截取前N个样本
    python scripts/train_dl_classifier.py \
        --data data/my_training.npz \
        --max-samples $samples
done

# 画学习曲线
```

### 实验3: Transformer

```bash
# 尝试最先进的Transformer
python scripts/train_dl_classifier.py \
    --model transformer \
    --epochs 100 \
    --sequence-length 30

# 期望：准确率最高，但慢
```

---

## 📝 实验报告模板

```markdown
# 姿态识别DL实验报告

## 实验目标
对比SVM和LSTM在姿态识别任务上的效果

## 数据集
- 训练集: _____ 样本
- 验证集: _____ 样本
- 数据分布: sitting ___%, standing ___%, lying ___%

## 实验设置
- 模型: LSTM
- 超参数:
  - sequence_length: 10
  - hidden_dim: 128
  - epochs: 50
  - batch_size: 16
  - learning_rate: 0.0005

## 结果

### 定量结果
| 模型 | 准确率 | Sitting F1 | Standing F1 | Lying F1 |
|------|--------|-----------|------------|----------|
| SVM  | 87.3%  | 0.88      | 0.85       | 0.89     |
| LSTM | 92.1%  | 0.93      | 0.89       | 0.94     |

### 定性观察
1. LSTM在过渡阶段更稳定
2. 葛优躺识别提升明显
3. 前倾工作姿态LSTM更准确
4. 推理速度影响可忽略（21ms → 23ms）

## 结论
LSTM在准确率和稳定性上都优于SVM，推荐替换。

## 附件
- 训练曲线: results/lstm_*/training_history.png
- 混淆矩阵: results/lstm_*/confusion_matrix.png
- 实时测试视频: ...
```

---

## ✅ 实验Checklist

做实验前，确保：

- [ ] 已有SVM模型并了解其性能
- [ ] 安装了PyTorch (`pip install torch`)
- [ ] 摄像头可用
- [ ] 有GPU或愿意等CPU慢慢训练
- [ ] 预留1-2天时间

开始实验：

- [ ] Step 1: 收集数据（至少1000样本）
- [ ] 检查数据平衡性
- [ ] Step 2: 训练MLP快速验证
- [ ] MLP能work后，训练LSTM
- [ ] Step 3: 实时对比测试
- [ ] 填写对比checklist
- [ ] 决定是否替换

完成后：

- [ ] 保存模型和结果
- [ ] 写实验报告
- [ ] 如果LSTM更好，修改config使用它

---

## 🚀 快速命令参考

```bash
# === 数据收集 ===
# 手动标注10分钟
python scripts/collect_training_data.py --duration 600

# 自动标注5分钟
python scripts/collect_training_data.py --auto-label --duration 300


# === 模型训练 ===
# MLP快速验证
python scripts/train_dl_classifier.py --model mlp --epochs 30

# LSTM完整训练
python scripts/train_dl_classifier.py --model lstm --epochs 50 --sequence-length 10


# === 测试对比 ===
# 实时对比SVM vs LSTM
python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --type lstm --compare

# 数据集评估
python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --test-data data/test.npz


# === 数据检查 ===
# 查看数据统计
python -c "import numpy as np; d=np.load('data/my_training.npz'); print('Samples:', len(d['labels']), 'Distribution:', np.bincount(d['labels']))"
```

---

**祝实验顺利！有问题随时查这个文档 📚**
