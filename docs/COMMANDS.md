# Camera 项目命令速查手册

所有可用命令的完整参考，按使用场景分类。

---

## 📋 目录

1. [系统运行](#系统运行)
2. [数据收集](#数据收集)
3. [模型训练](#模型训练)
4. [模型测试](#模型测试)
5. [数据库工具](#数据库工具)
6. [配置和标定](#配置和标定)
7. [常见工作流程](#常见工作流程)

---

## 🚀 系统运行

### 运行久坐提醒系统（主程序）

```bash
# CPU模式（默认，适合笔记本）
python main.py

# CPU模式（显式指定）
python main.py --mode cpu

# GPU模式（需要NVIDIA GPU）
python main.py --mode gpu

# 调试模式（显示关键点、骨架、判断信息）
python main.py --mode gpu --debug

# 不显示可视化窗口（后台运行）
python main.py --no-vis

# 使用自定义配置文件
python main.py --config config/my_custom.yaml
```

**说明**：
- 主程序会持续运行，实时监测行为状态
- 数据保存到 `data/database.db`
- 按 `q` 退出

---

## 📸 数据收集

### 收集训练数据（手动标注）

```bash
# 基础收集（默认60秒）
python scripts/collect_training_data.py

# 指定时长和输出文件
python scripts/collect_training_data.py --duration 600 --output data/my_training.npz

# 使用GPU配置（加快检测速度）
python scripts/collect_training_data.py --duration 600 --config config/config_gpu.yaml

# 自动标注（使用现有SVM预测）
python scripts/collect_training_data.py --duration 300 --auto-label
```

**操作说明**：
- 运行后摄像头会打开
- 按键标注：
  - `s` - 坐姿 (sitting)
  - `t` - 站姿 (standing)
  - `l` - 躺姿 (lying)
  - `q` - 提前退出
- 建议：每个姿态至少200个样本，总共600+样本

**输出**：
- `data/my_training.npz` 包含：
  - `landmarks`: (N, 17, 4) 关键点坐标
  - `labels`: (N,) 标签 (0=sitting, 1=standing, 2=lying)
  - `timestamps`: (N,) 时间戳

---

## 🎓 模型训练

### 训练深度学习分类器

```bash
# 训练MLP（最简单，快速测试）
python scripts/train_dl_classifier.py --model mlp --data data/my_training.npz --epochs 50

# 训练LSTM（推荐，考虑时间序列）
python scripts/train_dl_classifier.py --model lstm --data data/my_training.npz --epochs 100 --sequence-length 10

# 训练Transformer（最复杂，需要更多数据）
python scripts/train_dl_classifier.py --model transformer --data data/my_training.npz --epochs 150 --sequence-length 10

# 使用数据增强
python scripts/train_dl_classifier.py --model lstm --data data/my_training.npz --epochs 100 --augment

# 指定设备
python scripts/train_dl_classifier.py --model lstm --data data/my_training.npz --device cuda

# 恢复训练
python scripts/train_dl_classifier.py --model lstm --data data/my_training.npz --resume models/checkpoint.pth

# 自定义学习率和批次大小
python scripts/train_dl_classifier.py --model lstm --data data/my_training.npz --lr 0.001 --batch-size 32
```

**参数说明**：
- `--model`: 模型类型 (mlp, lstm, transformer)
- `--data`: 训练数据路径
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小（默认32）
- `--lr`: 学习率（默认0.001）
- `--sequence-length`: LSTM/Transformer序列长度（默认10）
- `--device`: 设备 (cuda/cpu)
- `--augment`: 启用数据增强
- `--resume`: 恢复训练的checkpoint路径

**输出**：
- 模型文件：`models/pose_classifier_{model}.pth`
- 训练曲线：`results/{model}_YYYYMMDD_HHMMSS/training_history.png`
- 混淆矩阵：`results/{model}_YYYYMMDD_HHMMSS/confusion_matrix.png`
- 配置文件：`results/{model}_YYYYMMDD_HHMMSS/config.json`

---

## 🧪 模型测试

### 实时测试模型

```bash
# 测试单个模型
python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --type lstm

# 对比DL vs SVM
python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --type lstm --compare

# 测试MLP模型
python scripts/test_dl_model.py --model models/pose_classifier_mlp.pth --type mlp

# 测试Transformer模型
python scripts/test_dl_model.py --model models/pose_classifier_transformer.pth --type transformer

# 使用GPU配置
python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --type lstm --config config/config_gpu.yaml
```

**操作说明**：
- 打开摄像头实时测试
- 显示预测结果和置信度
- 对比模式下显示DL和SVM的一致性
- 按 `q` 退出

### 在测试集上评估

```bash
# 在独立测试集上评估
python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --type lstm --test-data data/test.npz
```

**输出**：
- 准确率
- 分类报告（Precision, Recall, F1-score）
- 混淆矩阵

### 对比多个模型

```bash
# 对比不同模型性能
python scripts/compare_models.py --config config/config_cpu.yaml
```

---

## 🗄️ 数据库工具

### 检查数据库内容

```bash
# 查看数据库表结构和内容
python scripts/check_database.py
```

**输出**：
- 所有表的名称
- 每个表的记录数
- 表结构（列名）
- 示例数据

### 导出数据库数据（如果需要）

```bash
# Python方式导出
python -c "
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/database.db')
df = pd.read_sql_query('SELECT * FROM events', conn)
df.to_csv('data/events_export.csv', index=False)
print(f'导出 {len(df)} 条记录到 events_export.csv')
"
```

---

## ⚙️ 配置和标定

### ROI区域标定

```bash
# 标定ROI区域（床、椅子、门等）
python scripts/calibrate_roi.py --config config/config_cpu.yaml

# 使用GPU配置标定
python scripts/calibrate_roi.py --config config/config_gpu.yaml
```

**操作说明**：
- 摄像头打开后，用鼠标点击区域的四个角点
- 标定完成后保存到配置文件

---

## 📊 常见工作流程

### 工作流程1：首次使用（从零开始）

```bash
# 步骤1: 收集训练数据（10分钟）
python scripts/collect_training_data.py --duration 600 --output data/my_training.npz
# → 操作：按 s/t/l 标注坐/站/躺姿态

# 步骤2: 检查数据质量
python -c "
import numpy as np
data = np.load('data/my_training.npz')
print(f'样本数: {len(data[\"landmarks\"])}')
print(f'标签分布: {np.bincount(data[\"labels\"])}')
print(f'sitting: {np.sum(data[\"labels\"]==0)}')
print(f'standing: {np.sum(data[\"labels\"]==1)}')
print(f'lying: {np.sum(data[\"labels\"]==2)}')
"

# 步骤3: 训练LSTM模型（1-2小时）
python scripts/train_dl_classifier.py --model lstm --data data/my_training.npz --epochs 100

# 步骤4: 测试模型准确率
python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --type lstm --compare

# 步骤5: 修改配置文件使用DL模型
# 编辑 config/config_cpu.yaml 或 config/config_gpu.yaml
# 在 behavior: 部分添加：
#   classifier:
#     type: 'deep_learning'
#     model_type: 'lstm'
#     model_path: 'models/pose_classifier_lstm.pth'

# 步骤6: 运行系统
python main.py --mode gpu --debug
```

---

### 工作流程2：日常使用（已有模型）

```bash
# 直接运行系统
python main.py --mode gpu

# 或者调试模式
python main.py --mode gpu --debug

# 查看今天的记录
python scripts/check_database.py
```

---

### 工作流程3：改进模型（补充数据）

```bash
# 步骤1: 收集更多数据
python scripts/collect_training_data.py --duration 300 --output data/additional_training.npz

# 步骤2: 合并数据集
python -c "
import numpy as np

# 加载旧数据
old_data = np.load('data/my_training.npz')
old_landmarks = old_data['landmarks']
old_labels = old_data['labels']

# 加载新数据
new_data = np.load('data/additional_training.npz')
new_landmarks = new_data['landmarks']
new_labels = new_data['labels']

# 合并
combined_landmarks = np.concatenate([old_landmarks, new_landmarks])
combined_labels = np.concatenate([old_labels, new_labels])

# 保存
np.savez('data/combined_training.npz',
         landmarks=combined_landmarks,
         labels=combined_labels)

print(f'合并完成: {len(combined_landmarks)} 个样本')
print(f'标签分布: {np.bincount(combined_labels)}')
"

# 步骤3: 重新训练
python scripts/train_dl_classifier.py --model lstm --data data/combined_training.npz --epochs 100

# 步骤4: 对比新旧模型
python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --type lstm --compare
```

---

### 工作流程4：实验不同模型

```bash
# 训练多个模型
python scripts/train_dl_classifier.py --model mlp --data data/my_training.npz --epochs 50
python scripts/train_dl_classifier.py --model lstm --data data/my_training.npz --epochs 100
python scripts/train_dl_classifier.py --model transformer --data data/my_training.npz --epochs 150

# 分别测试
python scripts/test_dl_model.py --model models/pose_classifier_mlp.pth --type mlp
python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --type lstm
python scripts/test_dl_model.py --model models/pose_classifier_transformer.pth --type transformer

# 选择最好的模型更新配置
```

---

## 🔧 快速命令（常用）

```bash
# 收集数据
python scripts/collect_training_data.py --duration 600 --output data/my_training.npz

# 训练模型
python scripts/train_dl_classifier.py --model lstm --data data/my_training.npz --epochs 100

# 测试模型
python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --type lstm --compare

# 运行系统
python main.py --mode gpu --debug

# 检查数据库
python scripts/check_database.py
```

---

## 📝 Git 操作

```bash
# 查看状态
git status

# 拉取最新代码
git pull origin claude/three-stage-deployment-roadmap-011CUrFSWFN5rH8EACYAZGjD

# 提交更改
git add .
git commit -m "描述你的更改"
git push origin claude/three-stage-deployment-roadmap-011CUrFSWFN5rH8EACYAZGjD
```

---

## ❓ 常见问题

### Q: 如何查看训练数据的质量？

```bash
python -c "
import numpy as np
data = np.load('data/my_training.npz')
print(f'总样本: {len(data[\"landmarks\"])}')
print(f'sitting: {np.sum(data[\"labels\"]==0)}')
print(f'standing: {np.sum(data[\"labels\"]==1)}')
print(f'lying: {np.sum(data[\"labels\"]==2)}')
"
```

### Q: 如何切换CPU/GPU模式？

修改命令行参数：
- CPU: `python main.py --mode cpu`
- GPU: `python main.py --mode gpu`

或修改配置文件（编辑 `config/config_cpu.yaml` 或 `config/config_gpu.yaml`）

### Q: 训练很慢怎么办？

- 减少epochs: `--epochs 50`
- 减少batch size: `--batch-size 16`
- 使用GPU: `--device cuda`
- 使用更小的模型: `--model mlp`

### Q: 模型准确率低怎么办？

1. 收集更多数据（每类至少500个样本）
2. 数据多样化（不同姿势、角度、光线）
3. 使用数据增强: `--augment`
4. 增加训练轮数: `--epochs 150`
5. 尝试不同模型: LSTM通常比MLP好

---

## 📌 注意事项

1. **Windows用户**：所有命令在 `(camera)` 虚拟环境中运行
2. **数据文件**：`.npz` 文件可能很大（几十MB），不要提交到Git
3. **模型文件**：`.pth` 文件也很大（几MB到几十MB），建议添加到 `.gitignore`
4. **GPU内存**：训练时如果显存不足，减少 `--batch-size`
5. **摄像头占用**：同时只能有一个程序使用摄像头

---

## 🎯 推荐配置

### 笔记本用户（无GPU）
```bash
# 收集数据
python scripts/collect_training_data.py --duration 600 --output data/my_training.npz

# 训练（会自动用CPU）
python scripts/train_dl_classifier.py --model mlp --data data/my_training.npz --epochs 50

# 运行
python main.py --mode cpu
```

### 台式机用户（有NVIDIA GPU）
```bash
# 收集数据（可用GPU加速检测）
python scripts/collect_training_data.py --duration 600 --output data/my_training.npz --config config/config_gpu.yaml

# 训练（自动用GPU）
python scripts/train_dl_classifier.py --model lstm --data data/my_training.npz --epochs 100

# 运行
python main.py --mode gpu --debug
```

---

## 📚 更多文档

- [架构重构说明](ARCHITECTURE_REFACTORED.md) - 分类层和决策层详解
- [README.md](../README.md) - 项目整体介绍
- [配置文件示例](../config/) - CPU/GPU配置模板

---

**最后更新**: 2025-11-11
**维护者**: Camera项目团队
