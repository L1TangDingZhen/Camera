"""
姿态估计器 - 支持多种后端
1. MediaPipe Pose (CPU友好)
2. RTMPose (GPU高性能)
3. ViTPose (高精度)
"""

import time
from typing import Optional, Dict, List
import numpy as np
import cv2

from .base import PoseEstimatorInterface, Keypoint


# ==================== MediaPipe Backend ====================

class MediaPipePoseEstimator(PoseEstimatorInterface):
    """MediaPipe姿态估计器（CPU友好）"""

    def __init__(self, config: dict):
        super().__init__(config)

        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError("请安装 mediapipe: pip install mediapipe")

        self.mp_pose = mp.solutions.pose
        self.complexity = config.get('complexity', 1)  # 0=Lite, 1=Full, 2=Heavy
        self.confidence = config.get('confidence', 0.5)

        # 初始化MediaPipe Pose
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

        print(f"[MediaPipePose] 初始化完成，complexity={self.complexity}")

    def estimate(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """估计姿态"""
        start_time = time.time()

        try:
            # 如果有bbox，裁剪图像
            if bbox is not None:
                x1, y1, x2, y2 = bbox[:4].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                roi = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
            else:
                roi = frame
                offset_x, offset_y = 0, 0

            # BGR -> RGB
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # 推理
            results = self.pose.process(rgb)

            # 记录时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            # 提取关键点
            if results.pose_landmarks is None:
                return None

            # 转换为COCO-17格式
            h, w = roi.shape[:2]
            keypoints = np.zeros((17, 3), dtype=np.float32)

            for mp_idx, coco_idx in self.mediapipe_to_coco.items():
                landmark = results.pose_landmarks.landmark[mp_idx]
                keypoints[coco_idx] = [
                    landmark.x * w + offset_x,
                    landmark.y * h + offset_y,
                    landmark.visibility
                ]

            return keypoints

        except Exception as e:
            print(f"[MediaPipePose] 估计失败: {e}")
            return None

    def get_keypoint_names(self) -> List[str]:
        return Keypoint.NAMES

    def get_performance_metrics(self) -> Dict[str, float]:
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


# ==================== RTMPose Backend ====================

class RTMPoseEstimator(PoseEstimatorInterface):
    """RTMPose姿态估计器（GPU高性能）"""

    def __init__(self, config: dict):
        super().__init__(config)

        try:
            from mmpose.apis import init_model, inference_topdown
            from mmpose.structures import merge_data_samples
        except ImportError:
            raise ImportError(
                "请安装 MMPose:\n"
                "pip install openmim\n"
                "mim install mmcv-full\n"
                "mim install mmpose"
            )

        self.init_model = init_model
        self.inference_topdown = inference_topdown
        self.merge_data_samples = merge_data_samples

        self.model_path = config.get('model', 'rtmpose_s.pth')
        self.device = config.get('device', 'cuda:0')
        self.confidence = config.get('confidence', 0.3)

        # RTMPose配置文件（需要根据实际模型调整）
        # 这里使用预定义配置
        self.config_file = self._get_config_file()

        # 加载模型
        print(f"[RTMPose] 加载模型: {self.model_path}")
        try:
            self.model = self.init_model(
                self.config_file,
                self.model_path,
                device=self.device
            )
            print(f"[RTMPose] 模型加载成功")
        except Exception as e:
            print(f"[RTMPose] 模型加载失败: {e}")
            print("[RTMPose] 提示: 请确保已安装MMPose并下载模型文件")
            raise

        self.inference_times = []

    def _get_config_file(self) -> str:
        """获取配置文件路径"""
        # 这里简化处理，实际使用时需要提供正确的配置文件
        # 或者从mmpose的model zoo下载
        return 'rtmpose-s_8xb256-420e_coco-256x192.py'

    def estimate(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """估计姿态"""
        start_time = time.time()

        try:
            # 准备bbox
            if bbox is not None:
                bboxes = np.array([[bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]]])
            else:
                # 全图检测
                h, w = frame.shape[:2]
                bboxes = np.array([[0, 0, w, h, 1.0]])

            # 推理
            results = self.inference_topdown(
                self.model,
                frame,
                bboxes=bboxes
            )

            # 记录时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            # 提取关键点
            if not results or len(results) == 0:
                return None

            result = results[0]
            keypoints = result.pred_instances.keypoints[0]  # (17, 2)
            scores = result.pred_instances.keypoint_scores[0]  # (17,)

            # 合并为 (17, 3)
            keypoints_with_scores = np.concatenate([
                keypoints,
                scores[:, np.newaxis]
            ], axis=1)

            return keypoints_with_scores.astype(np.float32)

        except Exception as e:
            print(f"[RTMPose] 估计失败: {e}")
            return None

    def get_keypoint_names(self) -> List[str]:
        return Keypoint.NAMES

    def get_performance_metrics(self) -> Dict[str, float]:
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


# ==================== ViTPose Backend ====================

class ViTPoseEstimator(PoseEstimatorInterface):
    """ViTPose姿态估计器（高精度）"""

    def __init__(self, config: dict):
        super().__init__(config)

        # ViTPose也使用MMPose框架
        try:
            from mmpose.apis import init_model, inference_topdown
        except ImportError:
            raise ImportError("请安装 MMPose")

        self.init_model = init_model
        self.inference_topdown = inference_topdown

        self.model_path = config.get('model', 'vitpose_s.pth')
        self.device = config.get('device', 'cuda:0')
        self.confidence = config.get('confidence', 0.3)

        self.config_file = 'vitpose-s_coco-256x192.py'

        # 加载模型
        print(f"[ViTPose] 加载模型: {self.model_path}")
        try:
            self.model = self.init_model(
                self.config_file,
                self.model_path,
                device=self.device
            )
            print(f"[ViTPose] 模型加载成功")
        except Exception as e:
            print(f"[ViTPose] 模型加载失败: {e}")
            raise

        self.inference_times = []

    def estimate(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """估计姿态（与RTMPose实现类似）"""
        start_time = time.time()

        try:
            if bbox is not None:
                bboxes = np.array([[bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]]])
            else:
                h, w = frame.shape[:2]
                bboxes = np.array([[0, 0, w, h, 1.0]])

            results = self.inference_topdown(self.model, frame, bboxes=bboxes)

            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            if not results or len(results) == 0:
                return None

            result = results[0]
            keypoints = result.pred_instances.keypoints[0]
            scores = result.pred_instances.keypoint_scores[0]

            keypoints_with_scores = np.concatenate([
                keypoints,
                scores[:, np.newaxis]
            ], axis=1)

            return keypoints_with_scores.astype(np.float32)

        except Exception as e:
            print(f"[ViTPose] 估计失败: {e}")
            return None

    def get_keypoint_names(self) -> List[str]:
        return Keypoint.NAMES

    def get_performance_metrics(self) -> Dict[str, float]:
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


# ==================== Factory ====================

class PoseEstimatorFactory:
    """姿态估计器工厂"""

    @staticmethod
    def create(config: dict) -> PoseEstimatorInterface:
        """
        创建姿态估计器

        Args:
            config: 配置字典，必须包含 'backend' 字段

        Returns:
            姿态估计器实例
        """
        backend = config.get('backend', 'mediapipe').lower()

        if backend == 'mediapipe':
            return MediaPipePoseEstimator(config)
        elif backend == 'rtmpose':
            return RTMPoseEstimator(config)
        elif backend == 'vitpose':
            return ViTPoseEstimator(config)
        else:
            raise ValueError(
                f"不支持的后端: {backend}\n"
                f"支持的后端: mediapipe, rtmpose, vitpose"
            )
