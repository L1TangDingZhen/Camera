#!/usr/bin/env python3
"""
RL决策Agent训练脚本

训练强化学习Agent来学习何时输出分类结果（提高准确率）

使用方法:
    python train_decision_agent.py --classifier svm --data training_data/ --epochs 100
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Tuple
from collections import deque
import random

from src.classifiers.pose_classifier import PoseClassifierSVM
from src.classifiers.pose_classifier_dl import PoseClassifierDL
from src.state.rl_state_decision import DecisionDQN


def load_classifier(classifier_type: str, model_path: str):
    """加载分类器

    Args:
        classifier_type: 分类器类型 ('svm', 'mlp', 'lstm', 'transformer')
        model_path: 模型路径

    Returns:
        分类器实例
    """
    if not Path(model_path).exists():
        raise ValueError(f"模型不存在: {model_path}")

    if classifier_type == 'svm':
        return PoseClassifierSVM(model_path)
    elif classifier_type in ['mlp', 'lstm', 'transformer']:
        return PoseClassifierDL(
            model_path=model_path,
            model_type=classifier_type,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        raise ValueError(f"未知的分类器类型: {classifier_type}")


def load_data(data_dir: str) -> List[Tuple[np.ndarray, str]]:
    """加载数据

    Args:
        data_dir: 数据目录

    Returns:
        [(landmarks, label), ...]
    """
    all_data = []

    for pose_label in ['sitting', 'standing', 'lying']:
        filepath = os.path.join(data_dir, f"{pose_label}_samples.json")

        if not os.path.exists(filepath):
            print(f"[WARN] 未找到 {pose_label} 数据文件: {filepath}")
            continue

        with open(filepath, 'r') as f:
            samples = json.load(f)

        print(f"[INFO] 加载 {pose_label}: {len(samples)} 个样本")

        for sample in samples:
            # 提取关键点
            if 'keypoints_sequence' in sample and len(sample['keypoints_sequence']) > 0:
                keypoints = np.array(sample['keypoints_sequence'][-1])
            elif 'keypoints' in sample:
                keypoints = np.array(sample['keypoints'])
            elif 'features' in sample:
                # 兼容 collect_data.py 生成的格式
                keypoints = np.array(sample['features'])
            else:
                continue

            all_data.append((keypoints.astype(np.float32), pose_label))

    return all_data


def generate_training_episodes(classifier, data: List[Tuple], episode_length: int = 30):
    """生成训练episodes

    模拟连续观察过程，生成状态-动作-奖励序列

    Args:
        classifier: 分类器
        data: [(landmarks, label), ...]
        episode_length: 每个episode长度

    Returns:
        episodes: List of (states, actions, rewards)
    """
    episodes = []
    label_mapping = {'sitting': 0, 'standing': 1, 'lying': 2}

    print(f"\n[INFO] 生成训练episodes...")

    # 按类别分组
    data_by_label = {'sitting': [], 'standing': [], 'lying': []}
    for landmarks, label in data:
        data_by_label[label].append(landmarks)

    # 生成episodes（每个episode是同一类别的连续观察）
    for label, landmarks_list in data_by_label.items():
        if len(landmarks_list) < episode_length:
            continue

        # 每个类别生成多个episodes
        num_episodes = len(landmarks_list) // episode_length

        for ep_idx in range(num_episodes):
            start_idx = ep_idx * episode_length
            end_idx = start_idx + episode_length

            episode_landmarks = landmarks_list[start_idx:end_idx]
            true_label = label_mapping[label]

            # 模拟连续观察
            states = []
            actions = []
            rewards = []

            prediction_history = []

            for t, landmarks in enumerate(episode_landmarks):
                # 分类器预测
                # landmarks 是特征向量（58维），使用 predict_proba_from_features
                if hasattr(classifier, 'predict_proba_from_features'):
                    probs = classifier.predict_proba_from_features(landmarks)
                else:
                    probs = classifier.predict_proba(landmarks)

                if probs is None:
                    continue

                prediction_history.append(probs)

                # 构造状态
                state = encode_state(probs, prediction_history, t)
                states.append(state)

                # 计算最优动作和奖励
                max_prob = max(probs.values())
                predicted_label = max(probs, key=probs.get)

                # 最优动作策略（启发式）：
                if max_prob > 0.9:
                    # 高置信度 → 立即分类
                    optimal_action = 0  # classify
                    is_correct = (label_mapping[predicted_label] == true_label)
                    reward = 10.0 if is_correct else -10.0

                elif max_prob > 0.7 and len(prediction_history) > 5:
                    # 中等置信度且已观察足够 → 分类
                    optimal_action = 0
                    is_correct = (label_mapping[predicted_label] == true_label)
                    reward = 5.0 if is_correct else -8.0

                elif max_prob < 0.5:
                    # 低置信度 → 拒绝
                    optimal_action = 3  # reject
                    reward = 1.0

                else:
                    # 中等置信度 → 等待
                    optimal_action = 1  # wait
                    reward = -0.5  # 轻微惩罚延迟

                actions.append(optimal_action)
                rewards.append(reward)

            if len(states) > 0:
                episodes.append((states, actions, rewards))

    print(f"[INFO] 生成了 {len(episodes)} 个episodes")
    return episodes


def encode_state(probs: Dict[str, float], history: List[Dict], timestep: int) -> np.ndarray:
    """编码状态向量

    Args:
        probs: 当前概率 {'sitting': 0.8, ...}
        history: 历史预测列表
        timestep: 当前时间步

    Returns:
        state: (20,) vector
    """
    state = []

    # 1. 当前概率 (3维)
    state.extend([
        probs.get('sitting', 0),
        probs.get('standing', 0),
        probs.get('lying', 0)
    ])

    # 2. 置信度 (1维)
    max_prob = max(probs.values())
    state.append(max_prob)

    # 3. 历史统计 (6维)
    if len(history) > 0:
        recent = history[-5:]
        recent_sitting = np.mean([h.get('sitting', 0) for h in recent])
        recent_standing = np.mean([h.get('standing', 0) for h in recent])
        recent_lying = np.mean([h.get('lying', 0) for h in recent])
        state.extend([recent_sitting, recent_standing, recent_lying])

        sitting_var = np.var([h.get('sitting', 0) for h in recent])
        standing_var = np.var([h.get('standing', 0) for h in recent])
        lying_var = np.var([h.get('lying', 0) for h in recent])
        state.extend([sitting_var, standing_var, lying_var])
    else:
        state.extend([0] * 6)

    # 4. 时序特征 (4维)
    state.append(len(history) / 30.0)  # 历史长度
    state.append(0)  # motion（这里简化为0）
    state.append(0)  # time_since_last_change
    state.append(1.0)  # keypoint_visibility

    # 5. 上下文 (6维)
    state.append(0.5)  # hour_of_day
    state.append(1.0)  # is_working_hours
    state.extend([0, 0, 0, 0])  # 预留

    return np.array(state[:20], dtype=np.float32)


def train_decision_agent(agent: DecisionDQN,
                         episodes: List,
                         epochs: int = 100,
                         batch_size: int = 32,
                         lr: float = 0.0001,
                         device: str = 'cuda'):
    """训练决策Agent（监督学习方式）

    Args:
        agent: DecisionDQN
        episodes: [(states, actions, rewards), ...]
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        device: 设备
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    agent.to(device)

    # 优化器
    optimizer = optim.AdamW(agent.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # 损失函数（使用交叉熵，将最优动作作为标签）
    criterion = nn.CrossEntropyLoss()

    # 展平所有episodes为(state, action)对
    all_states = []
    all_actions = []

    for states, actions, rewards in episodes:
        all_states.extend(states)
        all_actions.extend(actions)

    states_tensor = torch.tensor(np.array(all_states), dtype=torch.float32)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)

    # 划分训练集和验证集
    n_samples = len(all_states)
    n_train = int(n_samples * 0.8)
    indices = np.random.permutation(n_samples)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"  开始训练 RL决策Agent")
    print(f"{'='*60}\n")
    print(f"训练样本: {n_train}, 验证样本: {len(val_indices)}")
    print(f"设备: {device}\n")

    for epoch in range(epochs):
        # ===== 训练阶段 =====
        agent.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Mini-batch训练
        np.random.shuffle(train_indices)

        for i in range(0, len(train_indices), batch_size):
            batch_idx = train_indices[i:i+batch_size]

            batch_states = states_tensor[batch_idx].to(device)
            batch_actions = actions_tensor[batch_idx].to(device)

            # 前向传播
            optimizer.zero_grad()
            q_values = agent(batch_states)
            loss = criterion(q_values, batch_actions)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = q_values.max(1)
            train_total += batch_actions.size(0)
            train_correct += predicted.eq(batch_actions).sum().item()

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / (len(train_indices) // batch_size + 1)

        # ===== 验证阶段 =====
        agent.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_states = states_tensor[val_indices].to(device)
            val_actions = actions_tensor[val_indices].to(device)

            q_values = agent(val_states)
            _, predicted = q_values.max(1)
            val_total = val_actions.size(0)
            val_correct = predicted.eq(val_actions).sum().item()

        val_acc = 100.0 * val_correct / val_total

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': agent.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }, 'models/decision_agent.pt')
            print(f"  ✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"  训练完成！")
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  模型已保存: models/decision_agent.pt")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='训练RL决策Agent')

    parser.add_argument('--classifier', type=str, required=True,
                       choices=['svm', 'mlp', 'lstm', 'transformer'],
                       help='使用的分类器类型')
    parser.add_argument('--data', type=str, default='training_data',
                       help='训练数据目录')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='学习率')
    parser.add_argument('--episode-length', type=int, default=30,
                       help='每个episode长度')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备')

    args = parser.parse_args()

    # 确定模型路径
    if args.classifier == 'svm':
        model_path = 'models/pose_classifier_svm.pkl'
    else:
        model_path = f'models/pose_classifier_{args.classifier}.pth'

    # 加载分类器
    print(f"\n[INFO] 加载分类器: {args.classifier}")
    classifier = load_classifier(args.classifier, model_path)
    print(f"[INFO] 分类器加载成功\n")

    # 加载数据
    print(f"[INFO] 从 {args.data} 加载数据...")
    data = load_data(args.data)

    if len(data) == 0:
        print("[ERROR] 没有找到有效的数据！")
        return

    # 生成训练episodes
    episodes = generate_training_episodes(classifier, data, args.episode_length)

    if len(episodes) == 0:
        print("[ERROR] 没有生成有效的episodes！")
        return

    # 创建决策Agent
    print(f"\n[INFO] 创建决策Agent...")
    agent = DecisionDQN(state_dim=20, action_dim=4)

    # 创建模型目录
    os.makedirs('models', exist_ok=True)

    # 训练
    train_decision_agent(
        agent=agent,
        episodes=episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )


if __name__ == '__main__':
    main()
