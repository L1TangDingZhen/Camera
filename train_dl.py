#!/usr/bin/env python3
"""
深度学习姿态分类器训练脚本

支持训练：MLP, LSTM, Transformer

使用方法:
    python train_dl.py --model mlp --data training_data/ --epochs 100
    python train_dl.py --model lstm --data training_data/ --epochs 100
    python train_dl.py --model transformer --data training_data/ --epochs 50
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Tuple

from src.classifiers.pose_classifier_dl import (
    PoseClassifierMLP,
    PoseClassifierLSTM,
    PoseClassifierTransformer
)


class PoseDataset(Dataset):
    """姿态数据集（单帧，用于MLP）"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PoseSequenceDataset(Dataset):
    """姿态序列数据集（用于LSTM/Transformer）"""

    def __init__(self, sequences: List[np.ndarray], labels: List[int], seq_len: int = 10):
        self.sequences = sequences
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # 如果序列长度不足，用最后一帧填充
        if len(seq) < self.seq_len:
            padding = [seq[-1]] * (self.seq_len - len(seq))
            seq = np.vstack([padding, seq])
        # 如果太长，截取最后seq_len帧
        elif len(seq) > self.seq_len:
            seq = seq[-self.seq_len:]

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def load_data_from_json(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """从JSON文件加载数据（用于MLP）

    Args:
        data_dir: training_data目录

    Returns:
        features: (N, 68) array
        labels: (N,) array
    """
    all_features = []
    all_labels = []

    label_mapping = {'sitting': 0, 'standing': 1, 'lying': 2}

    for pose_label in ['sitting', 'standing', 'lying']:
        filepath = os.path.join(data_dir, f"{pose_label}_samples.json")

        if not os.path.exists(filepath):
            print(f"[WARN] 未找到 {pose_label} 数据文件: {filepath}")
            continue

        with open(filepath, 'r') as f:
            samples = json.load(f)

        print(f"[INFO] 加载 {pose_label}: {len(samples)} 个样本")

        for sample in samples:
            # 假设每个sample有'features'字段（57维）
            # 或者有'keypoints'字段（17, 4）
            if 'features' in sample:
                features = np.array(sample['features'])
                # 如果是57维，需要扩展到68维
                if len(features) == 57:
                    # 补0到68维（这是临时方案，理想情况应该重新提取特征）
                    features = np.pad(features, (0, 11), mode='constant')
            elif 'keypoints_sequence' in sample:
                # 取序列的最后一帧
                keypoints = np.array(sample['keypoints_sequence'][-1])  # (17, 4)
                features = keypoints.flatten()  # (68,)
            else:
                continue

            all_features.append(features)
            all_labels.append(label_mapping[pose_label])

    if len(all_features) == 0:
        raise ValueError("没有找到任何训练数据！")

    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int64)


def train_model(model, train_loader, val_loader, epochs, lr, device, model_type, save_path):
    """训练模型"""

    model.to(device)

    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"  开始训练 {model_type.upper()} 模型")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'epoch': epoch,
                'val_acc': val_acc,
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"  ✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"  训练完成！")
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  模型已保存: {save_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='训练深度学习姿态分类器')

    parser.add_argument('--model', type=str, required=True,
                       choices=['mlp', 'lstm', 'transformer'],
                       help='模型类型')
    parser.add_argument('--data', type=str, default='training_data',
                       help='训练数据目录')
    parser.add_argument('--output', type=str, default=None,
                       help='输出模型路径（默认: models/pose_classifier_{model}.pth）')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--seq-len', type=int, default=10,
                       help='序列长度（用于LSTM/Transformer）')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    # 加载数据
    print(f"\n[INFO] 从 {args.data} 加载数据...")
    features, labels = load_data_from_json(args.data)

    print(f"[INFO] 总样本数: {len(features)}")
    print(f"[INFO] 特征维度: {features.shape[1]}")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=args.test_size, random_state=42, stratify=labels
    )

    print(f"[INFO] 训练集: {len(X_train)} 样本")
    print(f"[INFO] 验证集: {len(X_val)} 样本")

    # 创建数据集和数据加载器
    if args.model == 'mlp':
        train_dataset = PoseDataset(X_train, y_train)
        val_dataset = PoseDataset(X_val, y_val)
        model = PoseClassifierMLP(input_dim=68, hidden_dims=[128, 64, 32], num_classes=3)

    else:
        # LSTM/Transformer需要序列数据
        # 这里简化处理：将单帧数据复制成序列
        print(f"[WARN] LSTM/Transformer需要序列数据，当前使用单帧重复填充")
        print(f"[WARN] 建议使用 collect_data.py --sequence-mode 收集序列数据")

        train_sequences = [np.tile(X_train[i], (args.seq_len, 1)) for i in range(len(X_train))]
        val_sequences = [np.tile(X_val[i], (args.seq_len, 1)) for i in range(len(X_val))]

        train_dataset = PoseSequenceDataset(train_sequences, y_train.tolist(), args.seq_len)
        val_dataset = PoseSequenceDataset(val_sequences, y_val.tolist(), args.seq_len)

        if args.model == 'lstm':
            model = PoseClassifierLSTM(input_dim=68, hidden_dim=128, num_layers=2, num_classes=3)
        else:  # transformer
            model = PoseClassifierTransformer(input_dim=68, d_model=128, nhead=4,
                                             num_layers=2, num_classes=3, max_seq_len=args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 输出路径
    if args.output is None:
        os.makedirs('models', exist_ok=True)
        save_path = f'models/pose_classifier_{args.model}.pth'
    else:
        save_path = args.output

    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        model_type=args.model,
        save_path=save_path
    )


if __name__ == '__main__':
    main()
