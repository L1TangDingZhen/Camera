"""
MediaPipe Pose 姿态估计器

特点：
- CPU友好，无需GPU
- 安装简单（pip install mediapipe）
- 跨平台支持好（Windows/Linux/macOS）
- 速度中等（~40-50ms on CPU）
- 精度良好（AP ~67%）
- 支持3D world landmarks

适用场景：
- 开发测试环境
- CPU-only环境
- Windows开发机
- 快速原型验证
"""

import time
from typing import Optional, Dict, List
import numpy as np
import cv2

from .pose_estimator_base import PoseEstimatorInterface, Keypoint


class MediaPipePoseEstimator(PoseEstimatorInterface):
    """MediaPipe姿态估计器（CPU友好）"""

    def __init__(self, config: dict):
        super().__init__(config)

        # 检查MediaPipe是否安装
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "MediaPipe未安装！\n"
                "安装方法：pip install mediapipe\n"
                "推荐版本：mediapipe>=0.10.0"
            )

        self.mp_pose = mp.solutions.pose
        self.complexity = config.get('complexity', 1)  # 0=Lite, 1=Full, 2=Heavy
        self.confidence = config.get('confidence', 0.5)

        # 初始化MediaPipe Pose
        print(f"[MediaPipe] 初始化姿态估计器...")
        print(f"[MediaPipe]   模型复杂度: {self.complexity} (0=Lite, 1=Full, 2=Heavy)")
        print(f"[MediaPipe]   置信度阈值: {self.confidence}")
        print(f"[MediaPipe]   设备: CPU (MediaPipe不支持GPU)")

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.complexity,
            smooth_landmarks=True,
            min_detection_confidence=self.confidence,
            min_tracking_confidence=self.confidence
        )

        # MediaPipe 33关键点到COCO 17关键点的映射
        self.mediapipe_to_coco = {
            0: 0,   # nose
            2: 1,   # left_eye (inner)
            5: 2,   # right_eye (inner)
            7: 3,   # left_ear
            8: 4,   # right_ear
            11: 5,  # left_shoulder
            12: 6,  # right_shoulder
            13: 7,  # left_elbow
            14: 8,  # right_elbow
            15: 9,  # left_wrist
            16: 10, # right_wrist
            23: 11, # left_hip
            24: 12, # right_hip
            25: 13, # left_knee
            26: 14, # right_knee
            27: 15, # left_ankle
            28: 16, # right_ankle
        }

        self.inference_times = []

        # 存储最近的3D world landmarks
        self.last_world_landmarks = None

        print(f"[MediaPipe] 初始化完成 ✓")

    def estimate(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        估计姿态

        Args:
            frame: 输入图像 (H, W, 3) BGR格式
            bbox: 可选的人体边界框 [x1, y1, x2, y2, confidence]

        Returns:
            keypoints: (17, 3) [x, y, confidence] COCO-17格式
                      None 如果检测失败
        """
        start_time = time.time()

        try:
            # 如果有bbox，裁剪图像以提高性能
            if bbox is not None:
                x1, y1, x2, y2 = bbox[:4].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                roi = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
            else:
                roi = frame
                offset_x, offset_y = 0, 0

            # BGR -> RGB (MediaPipe要求RGB输入)
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # 推理
            results = self.pose.process(rgb)

            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            # 检查是否检测到姿态
            if results.pose_landmarks is None:
                self.last_world_landmarks = None
                return None

            # 提取3D world landmarks (单位：米，以髋部为原点)
            if results.pose_world_landmarks is not None:
                world_landmarks_3d = np.zeros((17, 4), dtype=np.float32)  # (x, y, z, visibility)

                for mp_idx, coco_idx in self.mediapipe_to_coco.items():
                    wl = results.pose_world_landmarks.landmark[mp_idx]
                    world_landmarks_3d[coco_idx] = [
                        wl.x,  # 米
                        wl.y,  # 米
                        wl.z,  # 米
                        wl.visibility
                    ]

                self.last_world_landmarks = world_landmarks_3d
            else:
                self.last_world_landmarks = None

            # 转换为COCO-17格式（2D图像坐标）
            h, w = roi.shape[:2]
            keypoints = np.zeros((17, 3), dtype=np.float32)

            for mp_idx, coco_idx in self.mediapipe_to_coco.items():
                landmark = results.pose_landmarks.landmark[mp_idx]
                keypoints[coco_idx] = [
                    landmark.x * w + offset_x,  # 转回原图坐标
                    landmark.y * h + offset_y,
                    landmark.visibility
                ]

            return keypoints

        except Exception as e:
            print(f"[MediaPipe] 姿态估计失败: {e}")
            return None

    def get_world_landmarks(self) -> Optional[np.ndarray]:
        """
        获取3D world landmarks（真实空间坐标）

        MediaPipe特有功能，提供相对于髋部的3D坐标

        Returns:
            world_landmarks: (17, 4) [x, y, z, visibility]
                           单位：米，以人体髋部为原点
                           None 如果最近一次估计失败
        """
        return self.last_world_landmarks

    def get_keypoint_names(self) -> List[str]:
        """获取关键点名称列表"""
        return Keypoint.NAMES

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        获取性能指标

        Returns:
            metrics: 包含以下指标的字典
                - avg_inference_time: 平均推理时间（秒）
                - fps: 理论最大FPS
                - min_time: 最小推理时间
                - max_time: 最大推理时间
        """
        if not self.inference_times:
            return {'avg_inference_time': 0.0, 'fps': 0.0}

        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0

        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'min_time': np.min(self.inference_times),
            'max_time': np.max(self.inference_times),
        }

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'pose'):
            self.pose.close()
