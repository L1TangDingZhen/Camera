#!/usr/bin/env python3
"""
RL Ensemble权重训练脚本

训练强化学习Agent来学习如何动态融合多个分类器

使用方法:
    python train_ensemble.py --models svm,mlp,lstm --data training_data/ --epochs 100
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
from collections import defaultdict

from src.classifiers.pose_classifier import PoseClassifierSVM
from src.classifiers.pose_classifier_dl import PoseClassifierDL
from src.classifiers.pose_classifier_ensemble import EnsembleWeightingAgent


def load_base_classifiers(model_configs: List[Dict]) -> List:
    """加载基础分类器

    Args:
        model_configs: 模型配置列表
            [{'type': 'svm', 'path': 'models/pose_classifier_svm.pkl'},
             {'type': 'mlp', 'path': 'models/pose_classifier_mlp.pth'}]

    Returns:
        基础分类器列表
    """
    classifiers = []

    for config in model_configs:
        model_type = config['type']
        model_path = config['path']

        if not Path(model_path).exists():
            print(f"[ERROR] 模型不存在: {model_path}")
            continue

        try:
            if model_type == 'svm':
                clf = PoseClassifierSVM(model_path)
                classifiers.append(clf)
                print(f"[INFO] 已加载SVM分类器: {model_path}")

            elif model_type in ['mlp', 'lstm', 'transformer']:
                clf = PoseClassifierDL(
                    model_path=model_path,
                    model_type=model_type,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                if clf.is_loaded:
                    classifiers.append(clf)
                    print(f"[INFO] 已加载{model_type.upper()}分类器: {model_path}")
            else:
                print(f"[WARN] 未知的模型类型: {model_type}")

        except Exception as e:
            print(f"[ERROR] 加载模型失败 {model_path}: {e}")

    return classifiers


def load_validation_data(data_dir: str) -> List[Tuple[np.ndarray, str]]:
    """加载验证数据

    Args:
        data_dir: 数据目录

    Returns:
        [(world_landmarks, label), ...]
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
            # 尝试提取3D landmarks
            if 'keypoints_sequence' in sample and len(sample['keypoints_sequence']) > 0:
                # 取序列的最后一帧
                keypoints = np.array(sample['keypoints_sequence'][-1])  # (17, 4)
            elif 'keypoints' in sample:
                keypoints = np.array(sample['keypoints'])  # (17, 4)
            else:
                # 如果没有关键点，跳过（SVM只有特征向量）
                continue

            all_data.append((keypoints.astype(np.float32), pose_label))

    return all_data


def collect_predictions(classifiers: List, data: List[Tuple[np.ndarray, str]]) -> Tuple[List, List, List]:
    """收集所有分类器的预测

    Args:
        classifiers: 分类器列表
        data: [(landmarks, label), ...]

    Returns:
        all_predictions: List of (num_classifiers, 3) arrays
        all_contexts: List of context vectors
        all_labels: List of true labels
    """
    all_predictions = []
    all_contexts = []
    all_labels = []

    label_mapping = {'sitting': 0, 'standing': 1, 'lying': 2}

    print(f"\n[INFO] 收集 {len(classifiers)} 个分类器的预测...")

    for idx, (landmarks, true_label) in enumerate(data):
        if (idx + 1) % 100 == 0:
            print(f"  进度: {idx+1}/{len(data)}")

        # 收集每个分类器的预测概率
        predictions = []
        valid = True

        for clf in classifiers:
            probs = clf.predict_proba(landmarks)

            if probs is None:
                valid = False
                break

            # 确保3个类别的顺序
            prob_array = [
                probs.get('sitting', 0),
                probs.get('standing', 0),
                probs.get('lying', 0)
            ]
            predictions.append(prob_array)

        if not valid:
            continue

        # 提取上下文特征
        context = extract_context(landmarks)

        all_predictions.append(np.array(predictions, dtype=np.float32))  # (num_classifiers, 3)
        all_contexts.append(context)
        all_labels.append(label_mapping[true_label])

    print(f"[INFO] 收集完成，有效样本: {len(all_predictions)}")

    return all_predictions, all_contexts, all_labels


def extract_context(landmarks: np.ndarray) -> np.ndarray:
    """提取上下文特征（简化版）

    Args:
        landmarks: (17, 4) [x, y, z, visibility]

    Returns:
        context: (10,) vector
    """
    from datetime import datetime

    # 1. 关键点可见性
    visibility = landmarks[:, 3]
    vis_mean = np.mean(visibility)
    vis_min = np.min(visibility)

    # 2. 时间上下文（这里用固定值，实际应该从系统时间获取）
    hour = datetime.now().hour / 24.0

    # 3. 预留特征
    context = [
        vis_mean,
        vis_min,
        hour,
        0, 0, 0, 0, 0, 0, 0
    ]

    return np.array(context[:10], dtype=np.float32)


def train_ensemble_agent(agent: EnsembleWeightingAgent,
                         predictions: List[np.ndarray],
                         contexts: List[np.ndarray],
                         labels: List[int],
                         epochs: int = 100,
                         batch_size: int = 32,
                         lr: float = 0.001,
                         device: str = 'cuda'):
    """训练Ensemble权重Agent（监督学习方式）

    Args:
        agent: EnsembleWeightingAgent
        predictions: List of (num_classifiers, 3) arrays
        contexts: List of (10,) arrays
        labels: List of true labels (0/1/2)
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        device: 设备
    """

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    agent.to(device)

    # 转换为tensor
    predictions_tensor = torch.tensor(np.array(predictions), dtype=torch.float32)  # (N, num_clf, 3)
    contexts_tensor = torch.tensor(np.array(contexts), dtype=torch.float32)  # (N, 10)
    labels_tensor = torch.tensor(labels, dtype=torch.long)  # (N,)

    # 优化器
    optimizer = optim.AdamW(agent.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 划分训练集和验证集
    n_samples = len(predictions)
    n_train = int(n_samples * 0.8)
    indices = np.random.permutation(n_samples)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"  开始训练 Ensemble 权重Agent")
    print(f"{'='*60}\n")
    print(f"训练样本: {n_train}, 验证样本: {len(val_indices)}")
    print(f"分类器数量: {predictions_tensor.shape[1]}")
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

            batch_preds = predictions_tensor[batch_idx].to(device)  # (B, num_clf, 3)
            batch_context = contexts_tensor[batch_idx].to(device)  # (B, 10)
            batch_labels = labels_tensor[batch_idx].to(device)  # (B,)

            # 前向传播
            optimizer.zero_grad()

            # 获取权重
            weights = agent(batch_preds, batch_context)  # (B, num_clf)

            # 加权融合
            # batch_preds: (B, num_clf, 3)
            # weights: (B, num_clf) → (B, num_clf, 1)
            weighted_probs = (batch_preds * weights.unsqueeze(-1)).sum(dim=1)  # (B, 3)

            # 计算损失（使用交叉熵，需要logits）
            # 这里weighted_probs是概率，我们需要转回logits或直接用NLL loss
            loss = criterion(torch.log(weighted_probs + 1e-8), batch_labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = weighted_probs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / (len(train_indices) // batch_size + 1)

        # ===== 验证阶段 =====
        agent.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_preds = predictions_tensor[val_indices].to(device)
            val_context = contexts_tensor[val_indices].to(device)
            val_labels = labels_tensor[val_indices].to(device)

            weights = agent(val_preds, val_context)
            weighted_probs = (val_preds * weights.unsqueeze(-1)).sum(dim=1)

            _, predicted = weighted_probs.max(1)
            val_total = val_labels.size(0)
            val_correct = predicted.eq(val_labels).sum().item()

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
                'num_classifiers': predictions_tensor.shape[1],
            }, 'models/ensemble_agent.pt')
            print(f"  ✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"  训练完成！")
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  模型已保存: models/ensemble_agent.pt")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='训练RL Ensemble权重Agent')

    parser.add_argument('--models', type=str, required=True,
                       help='基础模型列表，逗号分隔（如: svm,mlp,lstm）')
    parser.add_argument('--data', type=str, default='training_data',
                       help='验证数据目录')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备')

    args = parser.parse_args()

    # 解析模型列表
    model_types = args.models.split(',')
    model_configs = []

    for mtype in model_types:
        mtype = mtype.strip()
        if mtype == 'svm':
            model_configs.append({
                'type': 'svm',
                'path': 'models/pose_classifier_svm.pkl'
            })
        elif mtype in ['mlp', 'lstm', 'transformer']:
            model_configs.append({
                'type': mtype,
                'path': f'models/pose_classifier_{mtype}.pth'
            })
        else:
            print(f"[WARN] 未知的模型类型: {mtype}")

    if len(model_configs) < 2:
        print("[ERROR] Ensemble需要至少2个分类器！")
        return

    # 加载基础分类器
    print(f"\n[INFO] 加载基础分类器...")
    classifiers = load_base_classifiers(model_configs)

    if len(classifiers) < 2:
        print("[ERROR] 至少需要2个有效的分类器！")
        return

    print(f"[INFO] 成功加载 {len(classifiers)} 个分类器\n")

    # 加载验证数据
    print(f"[INFO] 从 {args.data} 加载数据...")
    data = load_validation_data(args.data)

    if len(data) == 0:
        print("[ERROR] 没有找到有效的数据！")
        return

    # 收集预测
    predictions, contexts, labels = collect_predictions(classifiers, data)

    if len(predictions) == 0:
        print("[ERROR] 没有收集到有效的预测！")
        return

    # 创建Ensemble Agent
    print(f"\n[INFO] 创建 Ensemble Agent...")
    agent = EnsembleWeightingAgent(
        num_classifiers=len(classifiers),
        context_dim=10
    )

    # 创建模型目录
    os.makedirs('models', exist_ok=True)

    # 训练
    train_ensemble_agent(
        agent=agent,
        predictions=predictions,
        contexts=contexts,
        labels=labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )


if __name__ == '__main__':
    main()
