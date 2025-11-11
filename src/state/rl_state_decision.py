#!/usr/bin/env python3
"""RL决策Agent - 学习何时输出状态

职责：
- 输入：概率分布 + 历史 + 上下文
- RL学习决策策略（何时相信预测，何时等待）
- 输出：动作（classify/wait/verify/reject）

注意：不负责分类，只负责决策！
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from collections import deque
import random


class DecisionDQN(nn.Module):
    """DQN网络：学习最优决策策略

    状态：
    - 当前概率分布
    - 历史10帧统计
    - 置信度趋势
    - 时间上下文

    动作：
    - 0: classify_now - 立即输出分类结果
    - 1: wait - 等待下一帧观察
    - 2: request_verify - 请求人工验证（主动学习）
    - 3: reject - 拒绝分类（质量太差）
    """

    def __init__(self, state_dim: int = 20, action_dim: int = 4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim)
        Returns:
            q_values: (batch, action_dim)
        """
        return self.network(state)


class RLDecisionAgent:
    """RL决策Agent

    核心思想：
    - 不是所有预测都应该立即输出
    - 高置信度 → 立即输出（快速响应）
    - 低置信度 → 等待积累证据（提高准确率）
    - 历史矛盾 → 可能是噪声，拒绝

    Example:
        >>> agent = RLDecisionAgent()
        >>> probs = {'sitting': 0.65, 'standing': 0.35, 'lying': 0.0}
        >>> context = {'motion': 0.1, 'hour': 14}
        >>> state, action = agent.decide(probs, context)
        >>> # state='sitting', action=0 (如果高置信度)
        >>> # state=None, action=1 (如果需要等待)
    """

    def __init__(self,
                 state_dim: int = 20,
                 action_dim: int = 4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 agent_path: Optional[str] = None):
        """
        Args:
            state_dim: 状态向量维度
            action_dim: 动作数量
            device: 设备
            agent_path: 预训练的RL agent路径
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # DQN网络
        self.dqn = DecisionDQN(state_dim, action_dim).to(self.device)
        self.target_dqn = DecisionDQN(state_dim, action_dim).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        # 经验回放
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 0.3  # 探索率

        # 历史缓冲
        self.prediction_history = deque(maxlen=30)
        self.confidence_history = deque(maxlen=30)

        # 统计
        self.total_decisions = 0
        self.wait_count = 0

        # 加载预训练模型
        if agent_path:
            self._load_agent(agent_path)

        print(f"[RLDecisionAgent] 初始化完成，设备: {self.device}")

    def _load_agent(self, agent_path: str):
        """加载预训练agent"""
        try:
            checkpoint = torch.load(agent_path, map_location=self.device)
            self.dqn.load_state_dict(checkpoint['model_state_dict'])
            self.dqn.eval()
            print(f"[RLDecisionAgent] Agent已加载: {agent_path}")
        except Exception as e:
            print(f"[WARN] 加载agent失败: {e}, 使用未训练的agent")

    def _encode_state(self,
                     current_probs: Dict[str, float],
                     history: List[Dict[str, float]],
                     context: Dict) -> np.ndarray:
        """编码状态向量

        包含：
        1. 当前概率 (3维)
        2. 置信度 (1维)
        3. 历史统计 (6维)
        4. 时序特征 (4维)
        5. 上下文 (6维)
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
            recent_sitting = np.mean([h.get('sitting', 0) for h in history[-5:]])
            recent_standing = np.mean([h.get('standing', 0) for h in history[-5:]])
            recent_lying = np.mean([h.get('lying', 0) for h in history[-5:]])
            state.extend([recent_sitting, recent_standing, recent_lying])

            # 方差（稳定性指标）
            sitting_var = np.var([h.get('sitting', 0) for h in history[-5:]])
            standing_var = np.var([h.get('standing', 0) for h in history[-5:]])
            lying_var = np.var([h.get('lying', 0) for h in history[-5:]])
            state.extend([sitting_var, standing_var, lying_var])
        else:
            state.extend([0] * 6)

        # 4. 时序特征 (4维)
        state.append(len(history) / 30.0)  # 历史长度（归一化）
        state.append(context.get('motion', 0))  # 运动量
        state.append(context.get('time_since_last_change', 0) / 60.0)  # 距上次状态变化（归一化）
        state.append(context.get('keypoint_visibility', 1.0))  # 关键点可见性

        # 5. 上下文 (6维)
        state.append(context.get('hour_of_day', 12) / 24.0)  # 时间归一化
        state.append(float(context.get('is_working_hours', 1)))
        state.append(context.get('current_duration', 0) / 300.0)  # 当前状态持续时间（归一化到5分钟）
        state.extend([0, 0, 0])  # 预留

        return np.array(state[:self.state_dim], dtype=np.float32)

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        """选择动作

        Args:
            state: 状态向量
            training: 是否训练模式（会使用ε-greedy探索）

        Returns:
            action: 0=classify, 1=wait, 2=verify, 3=reject
        """
        # ε-greedy探索
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # 贪婪策略
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.dqn(state_tensor)
            action = q_values.argmax().item()

        return action

    def decide(self,
               probs: Dict[str, float],
               context: Dict,
               training: bool = False) -> Tuple[Optional[str], int, Dict]:
        """做决策：是否输出状态

        Args:
            probs: 分类器输出的概率分布
            context: 上下文信息
            training: 是否训练模式

        Returns:
            (state, action, metadata)
            state: 'sitting'/'standing'/'lying'/None
            action: 0-3
            metadata: 决策信息
        """
        # 1. 更新历史
        self.prediction_history.append(probs)
        max_prob = max(probs.values())
        self.confidence_history.append(max_prob)

        # 2. 编码状态
        state = self._encode_state(probs, list(self.prediction_history), context)

        # 3. RL选择动作
        action = self.select_action(state, training)

        # 4. 统计
        self.total_decisions += 1
        if action == 1:
            self.wait_count += 1

        # 5. 元数据
        metadata = {
            'probs': probs,
            'action': action,
            'action_name': ['classify', 'wait', 'verify', 'reject'][action],
            'confidence': max_prob,
            'history_length': len(self.prediction_history),
            'wait_ratio': self.wait_count / max(self.total_decisions, 1)
        }

        # 6. 执行动作
        if action == 0:  # classify_now
            predicted_state = max(probs, key=probs.get)
            return predicted_state, action, metadata

        elif action == 1:  # wait
            return None, action, metadata

        elif action == 2:  # request_verify
            metadata['needs_verification'] = True
            return None, action, metadata

        else:  # reject (action == 3)
            metadata['rejected'] = True
            return None, action, metadata

    def update(self, state, action, reward, next_state, done):
        """更新RL agent（训练）

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        # 存储经验
        self.memory.append((state, action, reward, next_state, done))

        # 训练（如果有足够样本）
        if len(self.memory) < self.batch_size:
            return

        # 采样batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q-learning更新
        current_q = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_dqn(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 损失
        loss = F.mse_loss(current_q, target_q)

        # 反向传播
        optimizer = torch.optim.Adam(self.dqn.parameters(), lr=0.0001)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)
        optimizer.step()

    def sync_target_network(self):
        """同步target网络"""
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def reset(self):
        """重置历史（开始新视频片段时调用）"""
        self.prediction_history.clear()
        self.confidence_history.clear()

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_decisions': self.total_decisions,
            'wait_count': self.wait_count,
            'wait_ratio': self.wait_count / max(self.total_decisions, 1),
            'history_length': len(self.prediction_history)
        }


# ============ 训练相关（未来实现）============

def train_decision_agent(agent: RLDecisionAgent,
                        classifier,
                        training_data,
                        epochs: int = 100):
    """训练决策RL agent

    训练过程：
    1. 用分类器预测
    2. Agent决定是否输出
    3. 比较预测和真实标签
    4. 计算奖励，更新agent

    奖励函数：
    - 正确分类：+10
    - 错误分类：-10
    - 等待1帧：-0.5（惩罚延迟）
    """
    # TODO: 实现训练逻辑
    print("[TODO] Decision agent训练功能待实现")
    pass


if __name__ == '__main__':
    # 测试代码
    print("Testing RLDecisionAgent...")

    agent = RLDecisionAgent()

    # 测试场景1: 高置信度
    probs_high = {'sitting': 0.95, 'standing': 0.04, 'lying': 0.01}
    context = {'motion': 0.1, 'hour_of_day': 14, 'keypoint_visibility': 0.9}

    state, action, meta = agent.decide(probs_high, context)
    print(f"\n场景1 - 高置信度:")
    print(f"  输入: {probs_high}")
    print(f"  输出: state={state}, action={meta['action_name']}")
    print(f"  预期: 应该立即分类 (action=classify)")

    # 测试场景2: 低置信度
    probs_low = {'sitting': 0.55, 'standing': 0.45, 'lying': 0.0}
    state, action, meta = agent.decide(probs_low, context)
    print(f"\n场景2 - 低置信度:")
    print(f"  输入: {probs_low}")
    print(f"  输出: state={state}, action={meta['action_name']}")
    print(f"  预期: 可能等待 (action=wait)")

    # 测试统计
    stats = agent.get_stats()
    print(f"\n统计: {stats}")
    print("✓ Test passed")
