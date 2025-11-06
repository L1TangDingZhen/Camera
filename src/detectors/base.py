"""
检测器基类和接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import numpy as np


class DetectorInterface(ABC):
    """人体检测器接口"""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cpu')

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        检测画面中的人体

        Args:
            frame: 输入图像 (H, W, 3)

        Returns:
            bbox: 边界框 [x1, y1, x2, y2, confidence] 或 None
        """
        pass

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        pass


class PoseEstimatorInterface(ABC):
    """姿态估计器接口"""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cpu')

    @abstractmethod
    def estimate(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        估计人体姿态

        Args:
            frame: 输入图像 (H, W, 3)
            bbox: 人体边界框 [x1, y1, x2, y2, confidence]，如果None则检测全图

        Returns:
            keypoints: 关键点坐标和置信度 (N, 3) [x, y, confidence]
                      N为关键点数量（不同模型不同）
                      None表示检测失败
        """
        pass

    @abstractmethod
    def get_keypoint_names(self) -> List[str]:
        """获取关键点名称列表"""
        pass

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        pass


class Keypoint:
    """关键点标准定义（COCO-17格式）"""

    # COCO-17 关键点索引
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    @staticmethod
    def get_connections() -> List[Tuple[int, int]]:
        """获取骨架连接关系"""
        return [
            (0, 1), (0, 2),  # 头部
            (1, 3), (2, 4),  # 脸部
            (5, 6),  # 肩膀
            (5, 7), (7, 9),  # 左臂
            (6, 8), (8, 10),  # 右臂
            (5, 11), (6, 12),  # 躯干
            (11, 12),  # 臀部
            (11, 13), (13, 15),  # 左腿
            (12, 14), (14, 16),  # 右腿
        ]


class PoseUtils:
    """姿态计算工具类"""

    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        计算三点之间的角度

        Args:
            p1, p2, p3: 点坐标 (x, y)

        Returns:
            角度（度）
        """
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        return np.degrees(angle)

    @staticmethod
    def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """计算两点之间的欧氏距离"""
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def get_body_orientation(keypoints: np.ndarray) -> float:
        """
        计算身体朝向角度（与水平面的角度）

        Args:
            keypoints: 关键点 (N, 3)

        Returns:
            角度（度），0度为水平，90度为垂直
        """
        # 使用肩膀和臀部的中点
        left_shoulder = keypoints[Keypoint.LEFT_SHOULDER][:2]
        right_shoulder = keypoints[Keypoint.RIGHT_SHOULDER][:2]
        left_hip = keypoints[Keypoint.LEFT_HIP][:2]
        right_hip = keypoints[Keypoint.RIGHT_HIP][:2]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        # 计算向量与水平线的角度
        vector = shoulder_center - hip_center
        angle = np.arctan2(vector[1], vector[0])

        return abs(np.degrees(angle))

    @staticmethod
    def get_body_height(keypoints: np.ndarray) -> float:
        """
        计算身体高度（像素）

        Args:
            keypoints: 关键点 (N, 3)

        Returns:
            身体高度（像素）
        """
        # 从头顶到脚底
        nose = keypoints[Keypoint.NOSE][:2]
        left_ankle = keypoints[Keypoint.LEFT_ANKLE][:2]
        right_ankle = keypoints[Keypoint.RIGHT_ANKLE][:2]

        ankle_center = (left_ankle + right_ankle) / 2

        return PoseUtils.calculate_distance(nose, ankle_center)
