"""基于强化学习的自适应姿态分类器

RL不直接做分类，而是学习：
1. 如何融合多个分类器的预测（ensemble）
2. 何时相信当前预测，何时等待更多帧
3. 动态调整分类阈值以平衡准确率和延迟
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from collections import deque
import random


class AdaptiveClassifierDQN(nn.Module):
    """DQN网络：学习最优分类策略

    状态: [当前帧概率, 历史帧统计, 置信度, 时序特征]
    动作: [立即分类, 等待下一帧, 请求更多证据]
    """

    def __init__(self, state_dim: int = 20, action_dim: int = 4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.network(state)


class EnsembleWeightingAgent(nn.Module):
    """学习动态融合多个分类器的权重

    场景：你有SVM、MLP、LSTM三个分类器
    RL Agent学习在不同场景下如何加权平均
    """

    def __init__(self, num_classifiers: int = 3, context_dim: int = 10):
        super().__init__()

        # 输入：每个分类器的预测概率 + 上下文特征
        input_dim = num_classifiers * 3 + context_dim  # 3类 * N个分类器 + context

        self.weight_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classifiers),
            nn.Softmax(dim=-1)  # 归一化为权重
        )

    def forward(self, classifier_probs, context):
        """
        Args:
            classifier_probs: (batch, num_classifiers, 3) - 每个分类器的预测
            context: (batch, context_dim) - 上下文特征
        Returns:
            weights: (batch, num_classifiers) - 每个分类器的权重
        """
        batch_size = classifier_probs.shape[0]
        probs_flat = classifier_probs.view(batch_size, -1)
        x = torch.cat([probs_flat, context], dim=-1)
        weights = self.weight_network(x)
        return weights


class RLEnhancedClassifier:
    """强化学习增强的姿态分类器

    核心思想：
    - 使用现有分类器（SVM/DL）作为基础
    - RL agent学习如何利用时序信息做出更稳定、准确的决策
    - 奖励函数：准确分类 +10，错误分类 -10，延迟决策 -0.5/frame
    """

    def __init__(self,
                 base_classifiers: List,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            base_classifiers: List of classifiers (SVM, DL, etc.)
            device: torch device
        """
        self.base_classifiers = base_classifiers
        self.device = torch.device(device)

        # RL components
        self.state_dim = 20  # 根据特征定义
        self.action_dim = 4  # [classify_now, wait, request_verify, reject]
        self.dqn = AdaptiveClassifierDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_dqn = AdaptiveClassifierDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 0.3  # exploration rate

        # Ensemble weighting agent
        self.ensemble_agent = EnsembleWeightingAgent(
            num_classifiers=len(base_classifiers)
        ).to(self.device)

        # History buffer
        self.prediction_history = deque(maxlen=30)  # 最近30帧
        self.confidence_history = deque(maxlen=30)

        # Metrics
        self.total_decisions = 0
        self.correct_decisions = 0
        self.avg_decision_delay = 0

    def get_state(self, current_probs: Dict[str, float],
                  history: List[Dict[str, float]],
                  context: Dict) -> np.ndarray:
        """编码当前状态

        Args:
            current_probs: 当前帧的预测概率
            history: 历史帧的预测
            context: 上下文信息（时间、运动量等）

        Returns:
            state: (state_dim,) numpy array
        """
        state = []

        # 1. 当前帧概率 (3维)
        state.extend([
            current_probs.get('sitting', 0),
            current_probs.get('standing', 0),
            current_probs.get('lying', 0)
        ])

        # 2. 当前帧置信度 (1维)
        max_prob = max(current_probs.values())
        state.append(max_prob)

        # 3. 历史统计 (6维)
        if len(history) > 0:
            # 最近5帧的平均概率
            recent_avg = {
                'sitting': np.mean([h.get('sitting', 0) for h in history[-5:]]),
                'standing': np.mean([h.get('standing', 0) for h in history[-5:]]),
                'lying': np.mean([h.get('lying', 0) for h in history[-5:]])
            }
            state.extend(recent_avg.values())

            # 方差（稳定性指标）
            sitting_var = np.var([h.get('sitting', 0) for h in history[-5:]])
            standing_var = np.var([h.get('standing', 0) for h in history[-5:]])
            lying_var = np.var([h.get('lying', 0) for h in history[-5:]])
            state.extend([sitting_var, standing_var, lying_var])
        else:
            state.extend([0] * 6)

        # 4. 时序特征 (4维)
        state.append(len(history))  # 历史长度
        state.append(context.get('motion', 0))  # 运动量
        state.append(context.get('time_since_last_change', 0))  # 距上次状态变化
        state.append(context.get('keypoint_visibility', 1.0))  # 关键点可见性

        # 5. 上下文 (6维)
        state.append(context.get('hour_of_day', 12) / 24.0)  # 时间归一化
        state.append(context.get('is_working_hours', 1))
        state.append(context.get('recent_error_rate', 0))  # 最近的错误率
        state.append(context.get('avg_state_duration', 0))  # 平均状态持续时间
        state.extend([0, 0])  # 预留

        return np.array(state[:self.state_dim], dtype=np.float32)

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        """选择动作

        Actions:
        0: classify_now - 立即输出分类结果
        1: wait - 等待更多帧（积累更多证据）
        2: request_verify - 请求人工验证（用于主动学习）
        3: reject - 拒绝分类（置信度太低）

        Returns:
            action: 0-3
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.dqn(state_tensor)
            action = q_values.argmax().item()

        return action

    def ensemble_predict(self, world_landmarks: np.ndarray,
                        context: Dict) -> Dict[str, float]:
        """集成多个分类器并动态加权

        Args:
            world_landmarks: (17, 4)
            context: 上下文信息

        Returns:
            weighted_probs: {'sitting': 0.7, ...}
        """
        # 收集所有分类器的预测
        classifier_probs = []

        for clf in self.base_classifiers:
            probs = clf.predict_proba(world_landmarks)
            if probs is not None:
                classifier_probs.append([
                    probs.get('sitting', 0),
                    probs.get('standing', 0),
                    probs.get('lying', 0)
                ])
            else:
                classifier_probs.append([1/3, 1/3, 1/3])  # 均匀分布

        # 转换为tensor
        probs_tensor = torch.tensor(classifier_probs, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)

        # 编码上下文
        context_features = torch.tensor([
            context.get('keypoint_visibility', 1.0),
            context.get('motion', 0),
            context.get('hour_of_day', 12) / 24.0,
            # ... 更多特征
            0, 0, 0, 0, 0, 0, 0  # 填充到10维
        ], dtype=torch.float32).unsqueeze(0)  # (1, 10)

        # RL agent决定权重
        with torch.no_grad():
            weights = self.ensemble_agent(probs_tensor.to(self.device),
                                         context_features.to(self.device))
            weights = weights.cpu().numpy()[0]

        # 加权平均
        weighted_probs_array = np.zeros(3)
        for i, probs in enumerate(classifier_probs):
            weighted_probs_array += weights[i] * np.array(probs)

        return {
            'sitting': float(weighted_probs_array[0]),
            'standing': float(weighted_probs_array[1]),
            'lying': float(weighted_probs_array[2]),
            'weights': weights.tolist()  # debug信息
        }

    def predict_with_rl(self, world_landmarks: np.ndarray,
                       context: Dict,
                       training: bool = False) -> Tuple[Optional[str], Dict]:
        """使用RL策略进行预测

        Returns:
            (prediction, metadata)
            prediction: 'sitting'/'standing'/'lying'/None (如果决定等待)
            metadata: 包含决策信息
        """
        # 1. 获取ensemble预测
        current_probs = self.ensemble_predict(world_landmarks, context)

        # 2. 更新历史
        self.prediction_history.append(current_probs)

        # 3. 编码状态
        state = self.get_state(current_probs, list(self.prediction_history), context)

        # 4. RL决策
        action = self.select_action(state, training)

        metadata = {
            'probs': current_probs,
            'action': action,
            'action_name': ['classify', 'wait', 'verify', 'reject'][action],
            'ensemble_weights': current_probs.get('weights', [])
        }

        # 5. 执行动作
        if action == 0:  # classify_now
            prediction = max(current_probs, key=lambda k: current_probs[k]
                           if k in ['sitting', 'standing', 'lying'] else 0)
            return prediction, metadata

        elif action == 1:  # wait
            return None, metadata  # 返回None表示需要更多帧

        elif action == 2:  # request_verify
            metadata['needs_verification'] = True
            return None, metadata

        else:  # reject
            metadata['rejected'] = True
            return None, metadata

    def update(self, state, action, reward, next_state, done):
        """更新RL agent（训练）

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        # Store transition
        self.memory.append((state, action, reward, next_state, done))

        # Train if enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q-learning update
        current_q = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_dqn(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = F.mse_loss(current_q, target_q)

        # Backprop
        self.dqn.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)

        # Update
        optimizer = torch.optim.Adam(self.dqn.parameters(), lr=0.0001)
        optimizer.step()

    def sync_target_network(self):
        """同步target网络"""
        self.target_dqn.load_state_dict(self.dqn.state_dict())


# 使用示例
if __name__ == '__main__':
    print("Testing RL-Enhanced Classifier...")

    # 假设你有多个基础分类器
    class DummyClassifier:
        def predict_proba(self, landmarks):
            return {
                'sitting': np.random.rand(),
                'standing': np.random.rand(),
                'lying': np.random.rand()
            }

    base_classifiers = [DummyClassifier() for _ in range(3)]

    # 创建RL分类器
    rl_classifier = RLEnhancedClassifier(base_classifiers)

    # 测试预测
    dummy_landmarks = np.random.rand(17, 4).astype(np.float32)
    context = {
        'motion': 0.1,
        'hour_of_day': 14,
        'keypoint_visibility': 0.9
    }

    prediction, metadata = rl_classifier.predict_with_rl(dummy_landmarks, context)
    print(f"Prediction: {prediction}")
    print(f"Metadata: {metadata}")
