"""
检测器模块
包含人体检测和姿态估计的多种后端实现
"""

from .base import DetectorInterface, PoseEstimatorInterface
from .person_detector import PersonDetector
from .pose_estimator import PoseEstimatorFactory

__all__ = [
    'DetectorInterface',
    'PoseEstimatorInterface',
    'PersonDetector',
    'PoseEstimatorFactory',
]
