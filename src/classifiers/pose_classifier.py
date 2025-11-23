"""基于SVM的姿态分类器"""

import os
import pickle
import numpy as np
from typing import Optional, Dict


class PoseClassifierSVM:
    """SVM姿态分类器

    使用训练好的SVM模型对姿态进行分类，输出概率分布
    """

    def __init__(self, model_path: str = "models/pose_classifier_svm.pkl"):
        """
        Args:
            model_path: 训练好的模型文件路径
        """
        self.model_path = model_path
        self.clf = None
        self.scaler = None
        self.label_mapping = None
        self.reverse_mapping = None
        self.is_loaded = False

        # 尝试加载模型
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"[WARN] SVM模型文件不存在: {model_path}")
            print(f"[WARN] 请先运行 collect_data.py 和 train_svm.py 训练模型")
            print(f"[WARN] 将使用基于规则的分类方法作为降级方案")

    def load_model(self):
        """加载训练好的模型"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.clf = model_data['classifier']
            self.scaler = model_data['scaler']
            self.label_mapping = model_data['label_mapping']
            self.reverse_mapping = model_data['reverse_mapping']
            self.is_loaded = True

            print(f"[INFO] SVM分类器已加载: {self.model_path}")
            print(f"[INFO] 支持类别: {list(self.label_mapping.keys())}")

        except Exception as e:
            print(f"[ERROR] 加载模型失败: {e}")
            self.is_loaded = False

    def extract_features(self, world_landmarks: np.ndarray) -> Optional[np.ndarray]:
        """Extract feature vector from 3D world landmarks

        Args:
            world_landmarks: Either (17, 4) [x, y, z, visibility] or 68-dim flattened array

        Returns:
            features: Feature vector (normalized)
        """
        # Support both (17,4) and 68-dim 1D array inputs
        if world_landmarks.ndim == 1 and len(world_landmarks) == 68:
            world_landmarks = world_landmarks.reshape(17, 4)

        # Check keypoint visibility
        required_indices = [5, 6, 11, 12]  # 肩膀和臀部
        for idx in required_indices:
            if world_landmarks[idx][3] < 0.3:  # visibility < 0.3
                return None

        # 计算躯干长度（用于归一化）
        left_shoulder = world_landmarks[5][:3]
        right_shoulder = world_landmarks[6][:3]
        left_hip = world_landmarks[11][:3]
        right_hip = world_landmarks[12][:3]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center)

        if torso_length < 0.1:  # 躯干长度异常
            return None

        features = []

        # 1. 所有关键点的归一化3D坐标 (17 × 3 = 51维)
        for i in range(17):
            if world_landmarks[i][3] > 0:  # 可见
                features.extend(world_landmarks[i][:3] / torso_length)
            else:
                features.extend([0.0, 0.0, 0.0])  # 不可见用0填充

        # 2. 额外的几何特征

        # 2.1 躯干角度
        torso_vec = hip_center - shoulder_center
        vertical = np.array([0, -1, 0])
        torso_angle = np.degrees(np.arccos(
            np.clip(np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) + 1e-6), -1, 1)
        ))
        features.append(torso_angle / 90.0)  # 归一化到 [0, 2]

        # 2.2 髋膝Z轴差（如果膝盖可见）
        if world_landmarks[13][3] > 0.3 and world_landmarks[14][3] > 0.3:
            left_knee = world_landmarks[13][:3]
            right_knee = world_landmarks[14][:3]
            knee_center = (left_knee + right_knee) / 2
            hip_knee_z_diff = (hip_center[2] - knee_center[2]) / torso_length
            features.append(hip_knee_z_diff)
        else:
            features.append(0.0)

        # 2.3 髋膝距离
        if world_landmarks[13][3] > 0.3 and world_landmarks[14][3] > 0.3:
            left_knee = world_landmarks[13][:3]
            right_knee = world_landmarks[14][:3]
            knee_center = (left_knee + right_knee) / 2
            hip_knee_dist = np.linalg.norm(hip_center - knee_center) / torso_length
            features.append(hip_knee_dist)
        else:
            features.append(0.0)

        # 2.4 髋部高度（相对）
        hip_height = hip_center[1] / torso_length
        features.append(hip_height)

        # 2.5 肩膀宽度
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder) / torso_length
        features.append(shoulder_width)

        # 2.6 关键点可见性统计
        visibility_scores = [world_landmarks[i][3] for i in range(17)]
        features.append(np.mean(visibility_scores))
        features.append(np.min(visibility_scores))

        return np.array(features, dtype=np.float32)

    def predict_proba(self, world_landmarks: np.ndarray) -> Optional[Dict[str, float]]:
        """预测姿态概率分布

        Args:
            world_landmarks: (17, 4) [x, y, z, visibility]

        Returns:
            probabilities: {'sitting': 0.75, 'standing': 0.20, 'lying': 0.05}
                          如果模型未加载或特征提取失败，返回None
        """
        if not self.is_loaded:
            return None

        # 提取特征
        features = self.extract_features(world_landmarks)
        if features is None:
            return None

        # 标准化
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # 预测概率
        probs = self.clf.predict_proba(features_scaled)[0]

        # 转换为字典
        prob_dict = {}
        for label_idx, prob in enumerate(probs):
            label_name = self.reverse_mapping[label_idx]
            prob_dict[label_name] = float(prob)

        return prob_dict

    def predict_proba_from_features(self, features: np.ndarray) -> Optional[Dict[str, float]]:
        """Predict probability from feature vector (skip feature extraction if already 58-dim)

        Args:
            features: Either (58,) hand-crafted features or (68,) flattened keypoints

        Returns:
            probabilities: {'sitting': 0.75, 'standing': 0.20, 'lying': 0.05}
        """
        if not self.is_loaded:
            return None

        # If 68-dim raw keypoints, extract 58-dim hand-crafted features
        if len(features) == 68:
            features = self.extract_features(features)
            if features is None:
                return None

        # Normalize
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict probabilities
        probs = self.clf.predict_proba(features_scaled)[0]

        # Convert to dictionary
        prob_dict = {}
        for label_idx, prob in enumerate(probs):
            label_name = self.reverse_mapping[label_idx]
            prob_dict[label_name] = float(prob)

        return prob_dict

    def predict(self, world_landmarks: np.ndarray) -> Optional[str]:
        """预测姿态类别

        Args:
            world_landmarks: (17, 4) [x, y, z, visibility]

        Returns:
            label: 'sitting', 'standing', 'lying' 或 None
        """
        probs = self.predict_proba(world_landmarks)
        if probs is None:
            return None

        return max(probs, key=probs.get)
