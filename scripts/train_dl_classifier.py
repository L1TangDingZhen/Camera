#!/usr/bin/env python3
"""训练深度学习姿态分类器

用法:
    # 训练MLP
    python scripts/train_dl_classifier.py --model mlp --data data/training_data.npz

    # 训练LSTM
    python scripts/train_dl_classifier.py --model lstm --sequence-length 10

    # 训练Transformer
    python scripts/train_dl_classifier.py --model transformer --epochs 150
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.classifiers.pose_classifier_dl import (
    PoseClassifierMLP,
    PoseClassifierLSTM,
    PoseClassifierTransformer
)


class PoseDataset(Dataset):
    """姿态数据集"""

    def __init__(self, landmarks, labels, sequence_length=None, augment=False):
        """
        Args:
            landmarks: (N, 17, 4) numpy array
            labels: (N,) numpy array (0=sitting, 1=standing, 2=lying)
            sequence_length: 如果指定，返回序列数据用于LSTM/Transformer
            augment: 是否使用数据增强
        """
        self.landmarks = landmarks
        self.labels = labels
        self.sequence_length = sequence_length
        self.augment = augment

        # 预处理：归一化
        self.normalized_landmarks = self._normalize_landmarks(landmarks)

    def _normalize_landmarks(self, landmarks):
        """归一化关键点坐标"""
        normalized = []

        for lm in landmarks:
            # 计算躯干长度
            left_shoulder = lm[5][:3]
            right_shoulder = lm[6][:3]
            left_hip = lm[11][:3]
            right_hip = lm[12][:3]

            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            torso_length = np.linalg.norm(shoulder_center - hip_center)

            # 归一化
            lm_norm = lm.copy()
            if torso_length > 0.1:
                for i in range(17):
                    lm_norm[i, :3] /= torso_length

            normalized.append(lm_norm)

        return np.array(normalized)

    def _augment(self, landmarks):
        """数据增强"""
        # 1. 随机旋转 (±10度)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-10, 10)
            # TODO: 实现3D旋转

        # 2. 随机缩放
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.95, 1.05)
            landmarks = landmarks * scale

        # 3. 随机遮挡 (10%概率)
        if np.random.rand() > 0.9:
            mask_idx = np.random.randint(0, 17)
            landmarks[mask_idx, 3] = 0  # visibility = 0

        return landmarks

    def __len__(self):
        if self.sequence_length:
            return len(self.landmarks) - self.sequence_length + 1
        return len(self.landmarks)

    def __getitem__(self, idx):
        if self.sequence_length:
            # 返回序列
            sequence = self.normalized_landmarks[idx:idx+self.sequence_length]
            label = self.labels[idx + self.sequence_length - 1]  # 最后一帧的标签

            # Flatten: (seq_len, 17, 4) → (seq_len, 68)
            sequence_flat = sequence.reshape(self.sequence_length, -1)

            return torch.tensor(sequence_flat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        else:
            # 返回单帧
            landmarks = self.normalized_landmarks[idx]

            if self.augment:
                landmarks = self._augment(landmarks)

            # Flatten: (17, 4) → (68,)
            landmarks_flat = landmarks.flatten()

            return torch.tensor(landmarks_flat, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def load_data(data_path, val_split=0.2):
    """加载训练数据

    Args:
        data_path: .npz文件路径
        val_split: 验证集比例

    Returns:
        train_data, val_data
    """
    print(f"[INFO] 加载数据: {data_path}")

    data = np.load(data_path)
    landmarks = data['landmarks']  # (N, 17, 4)
    labels = data['labels']  # (N,)

    print(f"[INFO] 总样本数: {len(landmarks)}")
    print(f"[INFO] 类别分布: {np.bincount(labels)}")

    # 划分训练/验证集
    n_val = int(len(landmarks) * val_split)
    indices = np.random.permutation(len(landmarks))

    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_data = (landmarks[train_indices], labels[train_indices])
    val_data = (landmarks[val_indices], labels[val_indices])

    print(f"[INFO] 训练集: {len(train_data[0])} 样本")
    print(f"[INFO] 验证集: {len(val_data[0])} 样本")

    return train_data, val_data


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels


def plot_training_history(history, save_path):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] 训练曲线已保存: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Sitting', 'Standing', 'Lying']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # 标注数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] 混淆矩阵已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='训练深度学习姿态分类器')
    parser.add_argument('--model', type=str, required=True,
                       choices=['mlp', 'lstm', 'transformer'],
                       help='模型类型')
    parser.add_argument('--data', type=str, default='data/training_data.npz',
                       help='训练数据路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出模型路径 (默认: models/pose_classifier_{model}.pth)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='序列长度 (LSTM/Transformer)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--augment', action='store_true',
                       help='使用数据增强')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练 (checkpoint路径)')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    # 加载数据
    train_data, val_data = load_data(args.data)

    # 创建数据集
    sequence_length = args.sequence_length if args.model in ['lstm', 'transformer'] else None

    train_dataset = PoseDataset(
        train_data[0], train_data[1],
        sequence_length=sequence_length,
        augment=args.augment
    )
    val_dataset = PoseDataset(
        val_data[0], val_data[1],
        sequence_length=sequence_length,
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    # 创建模型
    print(f"[INFO] 创建模型: {args.model}")

    if args.model == 'mlp':
        model = PoseClassifierMLP(input_dim=68, num_classes=3)
    elif args.model == 'lstm':
        model = PoseClassifierLSTM(input_dim=68, num_classes=3)
    elif args.model == 'transformer':
        model = PoseClassifierTransformer(
            input_dim=68, num_classes=3,
            max_seq_len=args.sequence_length
        )

    model = model.to(device)

    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # 恢复训练
    start_epoch = 0
    best_val_acc = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    if args.resume:
        print(f"[INFO] 恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0)
        history = checkpoint.get('history', history)

    # 训练循环
    print(f"\n[INFO] 开始训练...")
    print(f"[INFO] 模型: {args.model}, Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print(f"[INFO] 学习率: {args.lr}, 设备: {device}\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = args.output or f'models/pose_classifier_{args.model}.pth'
            Path(output_path).parent.mkdir(exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'model_type': args.model,
                'history': history,
                'config': vars(args)
            }, output_path)

            print(f"✓ 最佳模型已保存: {output_path} (Val Acc: {val_acc:.2f}%)")

    # 最终评估
    print("\n" + "="*60)
    print("训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print("="*60)

    # 分类报告
    print("\nClassification Report:")
    print(classification_report(
        val_labels, val_preds,
        target_names=['Sitting', 'Standing', 'Lying']
    ))

    # 保存结果
    results_dir = Path(f'results/{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    results_dir.mkdir(parents=True, exist_ok=True)

    # 训练曲线
    plot_training_history(history, results_dir / 'training_history.png')

    # 混淆矩阵
    plot_confusion_matrix(val_labels, val_preds, results_dir / 'confusion_matrix.png')

    # 保存配置和结果
    with open(results_dir / 'config.json', 'w') as f:
        json.dump({
            'args': vars(args),
            'best_val_acc': best_val_acc,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1]
        }, f, indent=2)

    print(f"\n[INFO] 结果已保存到: {results_dir}")


if __name__ == '__main__':
    main()
