# 重构后的清晰架构说明

## 🎯 重构目标

将混淆的RL代码分离为清晰的两层架构：
- **分类层**：负责从关键点预测概率分布
- **决策层**：负责决定何时输出最终状态

---

## 📁 新的文件结构

```
src/
├── classifiers/ (分类层)
│   ├── pose_classifier.py              # SVM分类器
│   ├── pose_classifier_dl.py           # DL分类器 (MLP/LSTM/Transformer)
│   └── pose_classifier_ensemble.py     # RL Ensemble分类器 ⭐ 新
│
└── state/ (决策层)
    ├── behavior_state.py               # 状态机主文件 (已修改)
    ├── rl_state_decision.py            # RL决策agent ⭐ 新
    └── roi_manager.py                  # ROI管理
```

**删除**：`pose_classifier_rl.py`（职责混乱，已拆分）

---

## 🏗️ 清晰的架构

### 完整流程

```
┌─────────────────────────────────────────────────────────┐
│                   输入：关键点 (17, 4)                   │
└──────────────────────┬──────────────────────────────────┘
                       ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  分类层 (Classifiers)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                       ↓
         ┌─────────────┼─────────────┐
         ↓             ↓             ↓
     选项1: SVM    选项2: DL    选项3: Ensemble
         ↓             ↓             ↓
   手工特征59维   自动学习68维   RL融合多模型
         ↓             ↓             ↓
     SVM分类       LSTM分类      动态权重融合
         ↓             ↓             ↓
         └─────────────┼─────────────┘
                       ↓
              概率分布 {'sitting': 0.78, ...}
                       ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  决策层 (State Machine)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                       ↓
         ┌─────────────┼─────────────┐
         ↓                           ↓
    选项1: Simple                 选项2: RL
         ↓                           ↓
    if prob > 0.5              RL Agent决策
    + 防抖2秒                  (何时输出)
         ↓                           ↓
         └─────────────┼─────────────┘
                       ↓
              最终状态 'sitting' / None
                       ↓
┌─────────────────────┴─────────────────────────────────┐
│                输出：状态 / 等待                       │
└───────────────────────────────────────────────────────┘
```

---

## 📊 三层对比表

| 层级 | 职责 | 输入 | 输出 | 文件 |
|------|------|------|------|------|
| **分类层** | 预测概率 | 关键点 | 概率分布 | `classifiers/` |
| **决策层** | 决定输出 | 概率+历史 | 状态/等待 | `state/` |
| **应用层** | 业务逻辑 | 状态 | 提醒/统计 | `analytics/` |

---

## 🎯 分类层详解

### 文件1: `pose_classifier.py` (原有)

```python
class PoseClassifierSVM:
    """SVM分类器 - 手工特征"""

    def extract_features(self, landmarks):
        # 提取59维手工特征
        return features

    def predict_proba(self, landmarks):
        features = self.extract_features(landmarks)
        probs = self.svm.predict_proba(features)
        return {'sitting': 0.7, 'standing': 0.2, 'lying': 0.1}
```

**特点**：
- ✅ 手工特征工程
- ✅ 轻量级，快速
- ✅ 数据需求少

---

### 文件2: `pose_classifier_dl.py` (原有)

```python
class PoseClassifierDL:
    """深度学习分类器 - 自动特征学习"""

    def __init__(self, model_type='lstm'):
        if model_type == 'mlp':
            self.model = PoseClassifierMLP()
        elif model_type == 'lstm':
            self.model = PoseClassifierLSTM()
        elif model_type == 'transformer':
            self.model = PoseClassifierTransformer()

    def predict_proba(self, landmarks):
        # 直接输入68维原始关键点
        probs = self.model(landmarks)
        return {'sitting': 0.85, 'standing': 0.10, 'lying': 0.05}
```

**特点**：
- ✅ 自动特征学习
- ✅ 时序建模（LSTM）
- ✅ 准确率更高

---

### 文件3: `pose_classifier_ensemble.py` ⭐ 新

```python
class RLEnsembleClassifier:
    """RL Ensemble分类器 - 动态融合多个分类器

    职责：只负责分类，不负责决策！
    """

    def __init__(self, base_classifiers):
        # base_classifiers = [SVM, MLP, LSTM]
        self.base_classifiers = base_classifiers
        self.ensemble_agent = EnsembleWeightingAgent()  # RL学习权重

    def predict_proba(self, landmarks):
        # 1. 获取所有分类器预测
        svm_prob = self.base_classifiers[0].predict_proba(landmarks)
        mlp_prob = self.base_classifiers[1].predict_proba(landmarks)
        lstm_prob = self.base_classifiers[2].predict_proba(landmarks)

        # 2. RL决定权重
        context = self._get_context(landmarks)
        weights = self.ensemble_agent.get_weights(context)

        # 场景1：白天工作 → weights = [0.2, 0.2, 0.6]  # 更信任LSTM
        # 场景2：晚上躺床 → weights = [0.6, 0.2, 0.2]  # 更信任SVM

        # 3. 加权融合
        final_prob = np.average([svm_prob, mlp_prob, lstm_prob], weights=weights)

        # 返回概率，不做决策！
        return {'sitting': 0.78, 'standing': 0.18, 'lying': 0.04}
```

**特点**：
- ✅ 融合多个分类器
- ✅ RL学习场景自适应权重
- ✅ 输出仍然是概率（不是最终决策）

---

## 🎯 决策层详解

### 文件1: `behavior_state.py` (已修改)

```python
class BehaviorStateMachine:
    """状态机 - 支持多种分类器和决策策略"""

    def __init__(self, config):
        # === 分类器选择 ===
        self._init_classifier(config)
        # 可选：SVM / DL / Ensemble

        # === 决策策略选择 ===
        self._init_decision_strategy(config)
        # 可选：Simple / RL

    def _classify_3d(self, landmarks):
        # 1. 分类层：获取概率
        probs = self.classifier.predict_proba(landmarks)

        # 2. 决策层：决定是否输出
        if self.use_rl_decision:
            # RL决策：可能输出，也可能等待
            state, action, meta = self.rl_decision_agent.decide(probs, context)
            if state is None:
                return BehaviorState.UNKNOWN  # 等待
            return state
        else:
            # 简单决策：直接取最高概率
            return max(probs, key=probs.get)
```

---

### 文件2: `rl_state_decision.py` ⭐ 新

```python
class RLDecisionAgent:
    """RL决策Agent - 学习何时输出状态

    职责：只负责决策，不负责分类！
    """

    def __init__(self):
        self.dqn = DecisionDQN()  # DQN网络
        self.history = deque(maxlen=30)

    def decide(self, probs, context):
        """决定是否输出状态

        Args:
            probs: 分类器输出的概率 {'sitting': 0.65, ...}
            context: 上下文（时间、运动量、历史等）

        Returns:
            (state, action, metadata)
            state: 'sitting' / None (如果等待)
            action: 0=classify, 1=wait, 2=verify, 3=reject
        """
        # 1. 更新历史
        self.history.append(probs)

        # 2. 编码状态
        state_vector = self._encode_state(probs, self.history, context)

        # 3. RL选择动作
        action = self.dqn.select_action(state_vector)

        # 4. 执行动作
        if action == 0:  # classify_now
            return max(probs, key=probs.get), action, {}
        elif action == 1:  # wait
            return None, action, {}  # 不输出，等待下一帧
        elif action == 2:  # request_verify
            return None, action, {'needs_verification': True}
        else:  # reject
            return None, action, {'rejected': True}
```

**特点**：
- ✅ 学习何时相信预测
- ✅ 高置信度 → 立即输出
- ✅ 低置信度 → 等待观察
- ✅ 历史矛盾 → 拒绝分类

---

## ⚙️ 配置文件示例

```yaml
# config/config_cpu.yaml

behavior:
  # ========== 分类器选择 ==========
  classifier:
    type: 'svm'  # 'svm' / 'deep_learning' / 'rl_ensemble'

    # === SVM配置 ===
    model_path: 'models/pose_classifier_svm.pkl'

    # === DL配置 ===
    # type: 'deep_learning'
    # model_type: 'lstm'
    # model_path: 'models/pose_classifier_lstm.pth'
    # device: 'cuda'

    # === Ensemble配置 ===
    # type: 'rl_ensemble'
    # ensemble_models:
    #   - type: 'svm'
    #     path: 'models/pose_classifier_svm.pkl'
    #   - type: 'deep_learning'
    #     model_type: 'mlp'
    #     path: 'models/pose_classifier_mlp.pth'
    #   - type: 'deep_learning'
    #     model_type: 'lstm'
    #     path: 'models/pose_classifier_lstm.pth'
    # agent_path: 'models/ensemble_agent.pth'

  # ========== 决策策略选择 ==========
  decision:
    type: 'simple'  # 'simple' / 'rl'

    # === Simple决策配置 ===
    enter_duration: 2  # 持续2秒才切换状态

    # === RL决策配置 ===
    # type: 'rl'
    # agent_path: 'models/rl_decision_agent.pth'
```

---

## 🔄 使用组合

你可以自由组合分类器和决策策略：

### 组合1: SVM + Simple (当前默认)
```yaml
classifier:
  type: 'svm'
decision:
  type: 'simple'
```
**适合**：快速原型，资源受限

---

### 组合2: LSTM + Simple (推荐)
```yaml
classifier:
  type: 'deep_learning'
  model_type: 'lstm'
decision:
  type: 'simple'
```
**适合**：生产环境，提升准确率

---

### 组合3: Ensemble + Simple
```yaml
classifier:
  type: 'rl_ensemble'
decision:
  type: 'simple'
```
**适合**：融合多个模型，提升鲁棒性

---

### 组合4: LSTM + RL Decision
```yaml
classifier:
  type: 'deep_learning'
  model_type: 'lstm'
decision:
  type: 'rl'
```
**适合**：准确率 + 稳定性双优化

---

### 组合5: Ensemble + RL Decision (最强)
```yaml
classifier:
  type: 'rl_ensemble'
decision:
  type: 'rl'
```
**适合**：科研，追求极致性能

---

## 📊 重构前后对比

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| **文件数量** | 1个混乱文件 | 2个清晰文件 |
| **职责分离** | ❌ 混在一起 | ✅ 清晰分离 |
| **可维护性** | 难 | 易 |
| **可扩展性** | 困难 | 简单 |
| **可替换性** | 困难 | 独立替换 |
| **代码行数** | 383行混乱代码 | 200行(Ensemble) + 200行(Decision) |

---

## ✅ 重构带来的好处

### 1. 职责清晰
- **分类层**：我只预测概率，不管决策
- **决策层**：我只管何时输出，不管分类

### 2. 独立替换
```python
# 只想换分类器？改classifier配置
classifier:
  type: 'lstm'  # SVM → LSTM

# 只想换决策？改decision配置
decision:
  type: 'rl'  # Simple → RL

# 互不影响！
```

### 3. 易于理解
```
新手：我只看 pose_classifier.py，学习SVM分类
进阶：我看 pose_classifier_dl.py，学习深度学习
高级：我看 pose_classifier_ensemble.py，学习RL融合

分层学习，不会混淆！
```

### 4. 未来扩展
```
想加新分类器？
→ 只需在 classifiers/ 添加新文件

想加新决策策略？
→ 只需在 state/ 添加新文件

不影响其他部分！
```

---

## 🎓 总结

**重构解决的核心问题**：
- ❌ 旧的 `pose_classifier_rl.py` 混合了分类和决策，职责不清
- ✅ 新架构分离为两层，每层职责单一

**记住**：
- **分类层**（`classifiers/`）：输入关键点，输出概率
- **决策层**（`state/`）：输入概率，输出状态/等待

**配置灵活**：
- 可以只用SVM（简单）
- 可以只换DL（提升准确率）
- 可以用Ensemble（融合多模型）
- 可以加RL决策（提升稳定性）
- 可以任意组合（5种组合）

**这是一个清晰、可扩展、易维护的架构！** ✨
