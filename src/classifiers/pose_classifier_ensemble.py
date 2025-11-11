#!/usr/bin/env python3
"""RL Ensemble分类器 - 融合多个分类器

职责：
- 输入：3D关键点 (17, 4)
- 调用多个基础分类器（SVM, MLP, LSTM等）
- RL动态学习权重
- 输出：融合后的概率分布

注意：只负责分类，不做决策！
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class EnsembleWeightingAgent(nn.Module):
    """RL Agent: 学习动态融合多个分类器的权重

    在不同场景下，不同分类器的表现不同：
    - 白天工作：LSTM更准（时序稳定）
    - 晚上躺床：SVM更准（简单姿态）
    - 运动过渡：LSTM更准（时序建模）

    RL学习在不同上下文给不同模型分配权重
    """

    def __init__(self, num_classifiers: int = 3, context_dim: int = 10):
        """
        Args:
            num_classifiers: 基础分类器数量（如3个：SVM, MLP, LSTM）
            context_dim: 上下文特征维度
        """
        super().__init__()

        # 输入：每个分类器的预测概率 (N × 3) + 上下文特征
        input_dim = num_classifiers * 3 + context_dim

        # 网络：学习权重
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classifiers),
            nn.Softmax(dim=-1)  # 输出归一化权重
        )

    def forward(self, classifier_probs, context):
        """
        Args:
            classifier_probs: (batch, num_classifiers, 3) - 各分类器预测
            context: (batch, context_dim) - 上下文特征

        Returns:
            weights: (batch, num_classifiers) - 每个分类器的权重
        """
        batch_size = classifier_probs.shape[0]
        probs_flat = classifier_probs.view(batch_size, -1)
        x = torch.cat([probs_flat, context], dim=-1)
        weights = self.weight_network(x)
        return weights


class RLEnsembleClassifier:
    """RL Ensemble分类器

    使用多个基础分类器，RL学习最优融合权重。

    Example:
        >>> svm = PoseClassifierSVM()
        >>> lstm = PoseClassifierDL(model_type='lstm')
        >>> ensemble = RLEnsembleClassifier([svm, lstm])
        >>> probs = ensemble.predict_proba(world_landmarks)
        >>> # {'sitting': 0.78, 'standing': 0.18, 'lying': 0.04}
    """

    def __init__(self,
                 base_classifiers: List,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 agent_path: Optional[str] = None):
        """
        Args:
            base_classifiers: 基础分类器列表（需实现predict_proba方法）
            device: 设备
            agent_path: 预训练的RL agent路径
        """
        if len(base_classifiers) < 2:
            raise ValueError("Ensemble至少需要2个分类器")

        self.base_classifiers = base_classifiers
        self.device = torch.device(device)
        self.num_classifiers = len(base_classifiers)

        # 初始化RL agent
        self.ensemble_agent = EnsembleWeightingAgent(
            num_classifiers=self.num_classifiers,
            context_dim=10
        ).to(self.device)

        # 加载预训练权重（如果有）
        if agent_path:
            self._load_agent(agent_path)

        self.is_loaded = True
        print(f"[RLEnsembleClassifier] 已加载 {self.num_classifiers} 个基础分类器")

    def _load_agent(self, agent_path: str):
        """加载预训练的RL agent"""
        try:
            checkpoint = torch.load(agent_path, map_location=self.device)
            self.ensemble_agent.load_state_dict(checkpoint['model_state_dict'])
            self.ensemble_agent.eval()
            print(f"[RLEnsembleClassifier] RL agent已加载: {agent_path}")
        except Exception as e:
            print(f"[WARN] 加载RL agent失败: {e}, 使用未训练的agent")

    def _get_context(self, world_landmarks: np.ndarray) -> torch.Tensor:
        """提取上下文特征

        上下文帮助RL决定哪个分类器更可信：
        - 关键点可见性：低可见性时，简单模型可能更稳定
        - 运动量：运动时，时序模型更准
        - 时间：不同时段不同姿态分布
        """
        from datetime import datetime

        # 1. 关键点可见性统计
        visibility = world_landmarks[:, 3]
        vis_mean = np.mean(visibility)
        vis_min = np.min(visibility)

        # 2. 时间上下文
        hour = datetime.now().hour / 24.0  # 归一化

        # 3. 其他特征（未来可扩展）
        context = [
            vis_mean,      # 平均可见性
            vis_min,       # 最小可见性
            hour,          # 当前时间
            0, 0, 0, 0, 0, 0, 0  # 预留
        ]

        return torch.tensor(context[:10], dtype=torch.float32)

    def predict_proba(self, world_landmarks: np.ndarray) -> Optional[Dict[str, float]]:
        """预测概率分布（兼容分类器接口）

        Args:
            world_landmarks: (17, 4) [x, y, z, visibility]

        Returns:
            概率分布 {'sitting': 0.78, 'standing': 0.18, 'lying': 0.04}
            如果所有分类器都失败，返回None
        """
        # 1. 收集所有基础分类器的预测
        classifier_probs_list = []
        valid_classifiers = []

        for i, clf in enumerate(self.base_classifiers):
            probs = clf.predict_proba(world_landmarks)
            if probs is not None:
                # 确保有三个类别
                prob_array = [
                    probs.get('sitting', 0),
                    probs.get('standing', 0),
                    probs.get('lying', 0)
                ]
                classifier_probs_list.append(prob_array)
                valid_classifiers.append(i)
            else:
                # 分类器失败，使用均匀分布
                classifier_probs_list.append([1/3, 1/3, 1/3])

        if len(valid_classifiers) == 0:
            return None

        # 2. 转换为tensor
        classifier_probs = torch.tensor(
            classifier_probs_list,
            dtype=torch.float32
        ).unsqueeze(0)  # (1, num_classifiers, 3)

        # 3. 获取上下文
        context = self._get_context(world_landmarks).unsqueeze(0)  # (1, 10)

        # 4. RL决定权重
        with torch.no_grad():
            weights = self.ensemble_agent(
                classifier_probs.to(self.device),
                context.to(self.device)
            )
            weights = weights.cpu().numpy()[0]  # (num_classifiers,)

        # 5. 加权融合
        weighted_probs = np.zeros(3)
        for i, probs in enumerate(classifier_probs_list):
            weighted_probs += weights[i] * np.array(probs)

        # 6. 归一化（防止数值误差）
        weighted_probs = weighted_probs / weighted_probs.sum()

        return {
            'sitting': float(weighted_probs[0]),
            'standing': float(weighted_probs[1]),
            'lying': float(weighted_probs[2])
        }

    def predict(self, world_landmarks: np.ndarray) -> Optional[str]:
        """预测类别（兼容分类器接口）"""
        probs = self.predict_proba(world_landmarks)
        if probs is None:
            return None
        return max(probs, key=probs.get)

    def get_weights_info(self) -> Dict:
        """获取最近一次的权重信息（用于调试）"""
        # TODO: 实现权重历史追踪
        return {}


# ============ 训练相关（未来实现）============

def train_ensemble_agent(base_classifiers: List,
                        training_data,
                        val_data,
                        epochs: int = 100):
    """训练Ensemble RL agent

    训练过程：
    1. 收集所有分类器的预测
    2. 比较不同权重组合的准确率
    3. 强化学习更新权重网络

    奖励函数：
    - 正确分类：+1
    - 错误分类：-1
    """
    # TODO: 实现训练逻辑
    print("[TODO] Ensemble agent训练功能待实现")
    pass


if __name__ == '__main__':
    # 测试代码
    print("Testing RLEnsembleClassifier...")

    # 创建dummy分类器
    class DummyClassifier:
        def predict_proba(self, landmarks):
            return {
                'sitting': np.random.rand(),
                'standing': np.random.rand(),
                'lying': np.random.rand()
            }

    # 测试ensemble
    base_clfs = [DummyClassifier() for _ in range(3)]
    ensemble = RLEnsembleClassifier(base_clfs)

    # 测试预测
    dummy_landmarks = np.random.rand(17, 4).astype(np.float32)
    dummy_landmarks[:, 3] = 0.9  # 高可见性

    probs = ensemble.predict_proba(dummy_landmarks)
    pred = ensemble.predict(dummy_landmarks)

    print(f"Probabilities: {probs}")
    print(f"Prediction: {pred}")
    print("✓ Test passed")
