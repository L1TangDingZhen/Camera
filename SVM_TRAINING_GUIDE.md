# SVM姿态分类器 - 使用指南

本指南介绍如何使用SVM机器学习模型来提高姿态识别的准确率。

## 概述

系统现在支持两种姿态分类方式：

1. **基于规则的分类**（默认）：使用固定阈值判断坐/站/躺
2. **SVM机器学习分类**（推荐）：根据你录制的数据训练个性化模型，输出概率分布

### 为什么使用SVM？

✅ **自动学习最优阈值** - 不需要手动调参
✅ **输出概率分布** - 知道模型的置信度
✅ **相对特征** - 摄像头移动影响小
✅ **个性化** - 适应你的身体特征和环境
✅ **轻量快速** - CPU运行，<1ms延迟

---

## 快速开始

### 步骤1：录制训练数据（5分钟）

运行数据录制工具：

```bash
python collect_data.py
```

**操作流程：**

1. 程序打开摄像头，显示实时画面
2. **录制坐姿** (建议30秒以上):
   - 站在摄像头外面
   - 按 `s` 键
   - 倒计时5秒后自动开始录制
   - 坐在椅子上，做各种坐姿变化：
     - 正坐（面对摄像头）
     - 前倾（模拟写字）
     - 靠背（休息姿势）
     - 左转、右转（侧身）
   - 录够30秒后，按 `q` 停止

3. **录制站姿** (建议30秒以上):
   - 按 `t` 键
   - 倒计时5秒后开始录制
   - 站立并做各种动作：
     - 正面站立
     - 侧身站立（左、右）
     - 靠近镜头（小腿不可见）
     - 远离镜头
   - 按 `q` 停止

4. **录制躺姿** (建议20秒以上):
   - 按 `l` 键
   - 倒计时5秒后开始录制
   - 躺在床上：
     - 仰卧
     - 侧卧（左、右）
   - 按 `q` 停止

5. 按 `ESC` 退出程序

**参数调整**（可选）：

```bash
# 修改倒计时时间（默认5秒）
python collect_data.py --countdown 3

# 修改建议录制时长（默认30秒）
python collect_data.py --min-duration 20
```

**数据保存位置**：`training_data/` 目录下的JSON文件

---

### 步骤2：训练SVM模型（1分钟）

运行训练脚本：

```bash
python train_svm.py
```

程序会：
- 自动加载 `training_data/` 下的所有样本
- 使用网格搜索寻找最优参数
- 输出准确率报告和混淆矩阵
- 保存模型到 `models/pose_classifier_svm.pkl`

**预期输出示例：**

```
[INFO] 加载 sitting: 450 个样本
[INFO] 加载 standing: 380 个样本
[INFO] 加载 lying: 320 个样本

[INFO] 总样本数: 1150
[INFO] 特征维度: 57

[INFO] 训练集: 920 样本
[INFO] 测试集: 230 样本

[INFO] 使用网格搜索寻找最优参数...
[INFO] 最优参数: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
[INFO] 交叉验证准确率: 0.9457

训练集准确率: 0.9783 (97.83%)
测试集准确率: 0.9348 (93.48%)  ← 目标：>90%

测试集分类报告:
              precision    recall  f1-score   support
     sitting       0.95      0.94      0.94        89
    standing       0.93      0.95      0.94        77
       lying       0.93      0.92      0.92        64
```

**如果准确率<90%**：需要补录更多数据（重新运行 `collect_data.py`）

**参数说明**：

```bash
# 指定数据目录
python train_svm.py --data-dir my_training_data

# 指定输出路径
python train_svm.py --output my_models/svm.pkl

# 调整测试集比例（默认20%）
python train_svm.py --test-size 0.3

# 禁用网格搜索（更快但精度可能降低）
python train_svm.py --no-grid-search
```

---

### 步骤3：运行程序并查看效果

```bash
python main.py --config config/config_gpu.yaml
```

**SVM模型会自动加载** - 无需额外配置！

程序会在屏幕上显示：

```
[INFO] SVM分类器已加载: models/pose_classifier_svm.pkl
[INFO] 支持类别: ['sitting', 'standing', 'lying']
```

**在调试模式下**，会显示SVM概率分布：

```
SVM Probabilities:
  Sitting: 0.85   [绿色进度条]
  Standing: 0.12  [灰色进度条]
  Lying: 0.03     [灰色进度条]
```

---

## 高级使用

### 补录数据

如果某个姿态识别不准，可以补录该姿态的数据：

```bash
python collect_data.py
# 只录制识别不准的姿态（例如只按 's' 补录坐姿）
# 退出后重新训练
python train_svm.py
```

新数据会自动合并到已有数据中。

---

### 查看已录制的数据

```bash
ls -lh training_data/
# sitting_samples.json
# standing_samples.json
# lying_samples.json
```

每个文件包含该姿态的所有样本。

---

### 降级到规则分类

如果想暂时不使用SVM，删除或重命名模型文件即可：

```bash
mv models/pose_classifier_svm.pkl models/pose_classifier_svm.pkl.bak
```

程序会自动降级到基于规则的分类方法。

---

### 在配置文件中指定模型路径

编辑 `config/config_gpu.yaml` 或 `config/config_cpu.yaml`：

```yaml
behavior:
  svm_model_path: "custom_path/my_model.pkl"  # 自定义模型路径
```

---

## 常见问题

### Q1: 录制数据时，程序显示"已采集: 0 帧"？

**原因**：MediaPipe无法检测到姿态关键点

**解决**：
- 确保光线充足
- 身体尽量完整出现在画面中
- 离摄像头不要太远（2-4米最佳）

---

### Q2: 训练时提示"没有找到任何训练数据"？

**原因**：`training_data/` 目录为空或文件损坏

**解决**：
- 检查是否成功运行 `collect_data.py`
- 确认 `training_data/` 目录下有 `.json` 文件
- 重新录制数据

---

### Q3: 测试集准确率只有70-80%？

**原因**：数据量不足或数据质量差

**解决**：
- 每个姿态至少录制30秒
- 确保姿态变化多样（不要保持静止）
- 光线充足，关键点检测准确

---

### Q4: CPU占用高，运行变慢？

**回答**：SVM分类器非常轻量（<1ms），不会影响性能。

瓶颈仍然是MediaPipe姿态估计（CPU，约30-70ms）。

---

### Q5: 摄像头移动后，准确率下降？

**说明**：SVM使用相对特征，理论上摄像头移动影响小。

但如果角度变化非常大（例如从正面变侧面），建议：

1. 重新录制10-20秒该角度的数据
2. 重新训练模型

---

### Q6: 能否导出/导入模型？

**可以！** 模型文件 `models/pose_classifier_svm.pkl` 是标准的sklearn格式。

```bash
# 备份模型
cp models/pose_classifier_svm.pkl backup/

# 恢复模型
cp backup/pose_classifier_svm.pkl models/
```

---

## 性能对比

| 方案 | CPU占用 | 准确率 | 摄像头移动 | 需要配置 |
|------|---------|--------|-----------|---------|
| 基于规则 | 0% | 80-85% | ⚠️ 需重新调参 | ✅ 无 |
| **SVM (推荐)** | **<1%** | **90-95%** | **✅ 自动适应** | **5分钟录制** |

---

## 技术细节

### 特征向量（57维）

1. **3D归一化坐标** (51维)：17个关键点 × 3D坐标 (x, y, z)，除以躯干长度归一化
2. **几何特征** (6维)：
   - 躯干角度（相对垂直方向）
   - 髋膝Z轴差（深度差）
   - 髋膝3D距离
   - 髋部高度
   - 肩膀宽度
   - 关键点可见性统计

### 为什么用相对特征？

- **身高无关**：除以躯干长度，自动适应不同身高
- **摄像头角度无关**：使用3D world landmarks（MediaPipe提供）
- **比例不变**：摄像头移动时，比例关系保持稳定

---

## 下一步

如果SVM模型达到90%+准确率，恭喜！你已经完成了个性化姿态分类器的训练 🎉

现在系统会：
- ✅ 实时输出概率分布
- ✅ 自动选择最高概率的姿态
- ✅ 降级到规则分类（如果SVM失败）

**享受更准确的久坐提醒吧！** 😊
