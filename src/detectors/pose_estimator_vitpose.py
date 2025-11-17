"""
ViTPose 姿态估计器

特点：
- 基于Vision Transformer
- 高精度（AP ~75%）
- 速度较慢（~25ms on GPU）
- 适合精度优先场景

性能对比：
- MediaPipe: AP 67%, 50ms (CPU)
- RTMPose-s: AP 68.5%, 12ms (GPU)
- ViTPose-s: AP 75%, 25ms (GPU)

适用场景：
- 精度要求极高
- 实时性要求不严格
- GPU资源充足

⚠️ 注意：本项目主要使用RTMPose，ViTPose作为备选
"""

import time
from typing import Optional, Dict, List
from pathlib import Path
import numpy as np

from .pose_estimator_base import PoseEstimatorInterface, Keypoint


class ViTPoseEstimator(PoseEstimatorInterface):
    """ViTPose姿态估计器（高精度）"""

    def __init__(self, config: dict):
        super().__init__(config)

        # 检查依赖（ViTPose也使用MMPose框架）
        try:
            from mmpose.apis import init_model, inference_topdown
        except ImportError:
            raise ImportError(
                "MMPose未安装！\n"
                "请参考 INSTALL_RTMPOSE.md 安装MMPose"
            )

        self.init_model = init_model
        self.inference_topdown = inference_topdown

        self.model_path = config.get('model', 'vitpose-s.pth')
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
            print(f"[ViTPose] 模型加载成功 ✓")
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
            print(f"[ViTPose] 姿态估计失败: {e}")
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
