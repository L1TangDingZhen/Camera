"""基于深度学习的姿态分类器
支持多种架构：MLP, LSTM, Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Literal
from pathlib import Path


class PoseClassifierMLP(nn.Module):
    """简单的多层感知机分类器

    适合: 单帧分类，快速推理
    输入: 3D keypoints (17, 4) → flatten → (68,)
    """

    def __init__(self, input_dim: int = 68, hidden_dims: list = [128, 64, 32],
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, 68) - flattened keypoints [x,y,z,vis] * 17
        Returns:
            logits: (batch, num_classes)
        """
        return self.network(x)


class PoseClassifierLSTM(nn.Module):
    """LSTM时序分类器

    适合: 考虑时序信息，更稳定的分类
    输入: 连续N帧的关键点序列
    """

    def __init__(self, input_dim: int = 68, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 68) - 连续帧序列
        Returns:
            logits: (batch, num_classes)
        """
        # LSTM处理序列
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 分类
        logits = self.fc(last_output)
        return logits


class PoseClassifierTransformer(nn.Module):
    """Transformer分类器

    适合: 捕捉长距离依赖，最先进的时序建模
    输入: 连续N帧序列
    """

    def __init__(self, input_dim: int = 68, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2,
                 num_classes: int = 3, dropout: float = 0.1,
                 max_seq_len: int = 30):
        super().__init__()

        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 68)
        Returns:
            logits: (batch, num_classes)
        """
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(x)
        return logits


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PoseClassifierDL:
    """深度学习姿态分类器的统一接口

    兼容原有的SVM接口，方便替换
    """

    def __init__(self,
                 model_path: str = "models/pose_classifier_dl.pth",
                 model_type: Literal['mlp', 'lstm', 'transformer'] = 'mlp',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 sequence_length: int = 10):
        """
        Args:
            model_path: 模型权重路径
            model_type: 模型类型 ('mlp', 'lstm', 'transformer')
            device: 设备
            sequence_length: LSTM/Transformer需要的序列长度
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device(device)
        self.sequence_length = sequence_length

        # 标签映射
        self.label_mapping = {'sitting': 0, 'standing': 1, 'lying': 2}
        self.reverse_mapping = {0: 'sitting', 1: 'standing', 2: 'lying'}

        # 序列缓冲（用于LSTM/Transformer）
        self.sequence_buffer = []

        # 加载模型
        self.model = None
        self.is_loaded = False

        if Path(model_path).exists():
            self.load_model()
        else:
            print(f"[WARN] 深度学习模型不存在: {model_path}")
            print(f"[WARN] 请先训练模型")

    def load_model(self):
        """加载模型"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 根据类型创建模型
            if self.model_type == 'mlp':
                self.model = PoseClassifierMLP(
                    input_dim=68,
                    hidden_dims=[128, 64, 32],
                    num_classes=3
                )
            elif self.model_type == 'lstm':
                self.model = PoseClassifierLSTM(
                    input_dim=68,
                    hidden_dim=128,
                    num_layers=2,
                    num_classes=3
                )
            elif self.model_type == 'transformer':
                self.model = PoseClassifierTransformer(
                    input_dim=68,
                    d_model=128,
                    nhead=4,
                    num_layers=2,
                    num_classes=3,
                    max_seq_len=self.sequence_length
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True

            print(f"[INFO] 深度学习分类器已加载: {self.model_type}")
            print(f"[INFO] 设备: {self.device}")

        except Exception as e:
            print(f"[ERROR] 加载模型失败: {e}")
            self.is_loaded = False

    def preprocess(self, world_landmarks: np.ndarray) -> Optional[torch.Tensor]:
        """预处理3D landmarks

        Args:
            world_landmarks: (17, 4) [x, y, z, visibility]

        Returns:
            tensor: (68,) flattened and normalized
        """
        # 检查可见性
        required_indices = [5, 6, 11, 12]  # 肩膀和臀部
        for idx in required_indices:
            if world_landmarks[idx][3] < 0.3:
                return None

        # Flatten: (17, 4) → (68,)
        features = world_landmarks.flatten()

        # 归一化：躯干长度归一化
        left_shoulder = world_landmarks[5][:3]
        right_shoulder = world_landmarks[6][:3]
        left_hip = world_landmarks[11][:3]
        right_hip = world_landmarks[12][:3]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center)

        if torso_length < 0.1:
            return None

        # 归一化坐标维度（每4个元素的前3个是xyz）
        for i in range(17):
            features[i*4:i*4+3] /= torso_length

        return torch.tensor(features, dtype=torch.float32)

    def predict_proba(self, world_landmarks: np.ndarray) -> Optional[Dict[str, float]]:
        """预测概率分布（兼容SVM接口）

        Args:
            world_landmarks: (17, 4) [x, y, z, visibility]

        Returns:
            {'sitting': 0.75, 'standing': 0.20, 'lying': 0.05}
        """
        if not self.is_loaded:
            return None

        # 预处理
        features = self.preprocess(world_landmarks)
        if features is None:
            return None

        with torch.no_grad():
            if self.model_type == 'mlp':
                # MLP: 单帧推理
                features = features.unsqueeze(0).to(self.device)  # (1, 68)
                logits = self.model(features)

            else:
                # LSTM/Transformer: 需要序列
                self.sequence_buffer.append(features)
                if len(self.sequence_buffer) > self.sequence_length:
                    self.sequence_buffer.pop(0)

                # 如果序列不够，填充
                if len(self.sequence_buffer) < self.sequence_length:
                    # 用当前帧重复填充
                    padded = [features] * (self.sequence_length - len(self.sequence_buffer))
                    sequence = padded + self.sequence_buffer
                else:
                    sequence = self.sequence_buffer

                # (seq_len, 68) → (1, seq_len, 68)
                sequence_tensor = torch.stack(sequence).unsqueeze(0).to(self.device)
                logits = self.model(sequence_tensor)

            # Softmax得到概率
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # 转换为字典
        prob_dict = {
            self.reverse_mapping[i]: float(probs[i])
            for i in range(len(probs))
        }

        return prob_dict

    def predict(self, world_landmarks: np.ndarray) -> Optional[str]:
        """预测类别"""
        probs = self.predict_proba(world_landmarks)
        if probs is None:
            return None
        return max(probs, key=probs.get)

    def predict_proba_from_features(self, features: np.ndarray) -> Optional[Dict[str, float]]:
        """从特征向量直接预测概率（用于训练决策层）

        Args:
            features: (58,) 特征向量

        Returns:
            {'sitting': 0.75, 'standing': 0.20, 'lying': 0.05}
        """
        if not self.is_loaded:
            return None

        # 扩展到68维（DL模型期望68维）
        if len(features) == 58:
            features = np.pad(features, (0, 10), mode='constant')

        features_tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            if self.model_type == 'mlp':
                # MLP: 单帧
                features_tensor = features_tensor.unsqueeze(0).to(self.device)
                logits = self.model(features_tensor)
            else:
                # LSTM/Transformer: 重复成序列
                sequence = [features_tensor] * self.sequence_length
                sequence_tensor = torch.stack(sequence).unsqueeze(0).to(self.device)
                logits = self.model(sequence_tensor)

            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        return {
            self.reverse_mapping[i]: float(probs[i])
            for i in range(len(probs))
        }

    def reset_sequence(self):
        """重置序列缓冲（用于新的视频片段）"""
        self.sequence_buffer.clear()


# 训练脚本示例
def train_pose_classifier(train_data, val_data, model_type='mlp',
                         epochs=50, batch_size=32, lr=0.001):
    """训练深度学习姿态分类器

    Args:
        train_data: List of (world_landmarks, label) tuples
        val_data: List of (world_landmarks, label) tuples
        model_type: 'mlp', 'lstm', or 'transformer'
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    if model_type == 'mlp':
        model = PoseClassifierMLP(input_dim=68, num_classes=3)
    elif model_type == 'lstm':
        model = PoseClassifierLSTM(input_dim=68, num_classes=3)
    elif model_type == 'transformer':
        model = PoseClassifierTransformer(input_dim=68, num_classes=3)

    model.to(device)

    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # TODO: 实现完整训练循环
    # 这里省略数据加载和训练代码

    print(f"[INFO] 训练 {model_type} 模型...")
    print(f"[INFO] 数据集: {len(train_data)} train, {len(val_data)} val")

    # 训练循环
    for epoch in range(epochs):
        model.train()
        # ... 训练逻辑

        model.eval()
        # ... 验证逻辑

        scheduler.step()

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'epoch': epochs,
    }, f'models/pose_classifier_{model_type}.pth')

    print(f"[INFO] 模型已保存")


if __name__ == '__main__':
    # 测试代码
    print("Testing PoseClassifierDL...")

    # 创建dummy数据
    dummy_landmarks = np.random.rand(17, 4).astype(np.float32)
    dummy_landmarks[:, 3] = 0.9  # 高可见性

    # 测试MLP
    classifier = PoseClassifierDL(model_type='mlp')
    if classifier.is_loaded:
        probs = classifier.predict_proba(dummy_landmarks)
        print(f"MLP Predictions: {probs}")
