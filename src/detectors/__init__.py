"""
检测器模块
包含人体检测和姿态估计的多种后端实现

姿态估计后端：
- MediaPipe: CPU友好，开发测试推荐
- RTMPose: GPU加速，生产部署推荐
- ViTPose: 高精度，特殊场景使用
"""

from .base import DetectorInterface, PoseEstimatorInterface
from .person_detector import PersonDetector
from .pose_estimator_base import PoseEstimatorFactory, Keypoint, PoseUtils

# 可选导入（避免缺少依赖时报错）
try:
    from .pose_estimator_mediapipe import MediaPipePoseEstimator
except ImportError:
    MediaPipePoseEstimator = None

try:
    from .pose_estimator_rtmpose import RTMPoseEstimator
except ImportError:
    RTMPoseEstimator = None

try:
    from .pose_estimator_vitpose import ViTPoseEstimator
except ImportError:
    ViTPoseEstimator = None

__all__ = [
    'DetectorInterface',
    'PoseEstimatorInterface',
    'PersonDetector',
    'PoseEstimatorFactory',
    'Keypoint',
    'PoseUtils',
    'MediaPipePoseEstimator',
    'RTMPoseEstimator',
    'ViTPoseEstimator',
]
