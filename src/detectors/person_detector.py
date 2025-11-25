"""
基于YOLOv8的人体检测器
支持PyTorch (.pt) 和 TensorRT (.engine) 模型
"""

import time
from typing import Optional, Dict
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .base import DetectorInterface


class PersonDetector(DetectorInterface):
    """YOLOv8人体检测器"""

    def __init__(self, config: dict):
        super().__init__(config)

        if YOLO is None:
            raise ImportError("请安装 ultralytics: pip install ultralytics")

        self.model_path = config.get('model', 'yolov8s.pt')
        self.confidence = config.get('confidence', 0.5)
        self.iou = config.get('iou', 0.45)
        self.device = config.get('device', 'cpu')
        # 可选的 bbox 平滑与扩张配置
        self.smooth_alpha = config.get('smooth_alpha', 0.0)  # 0 表示不平滑
        self.expand_ratio = config.get('expand_ratio', 1.0)  # 1 表示不扩张
        self._last_bbox = None

        # 性能统计
        self.inference_times = []

        # 加载模型
        print(f"[PersonDetector] 加载模型: {self.model_path}")
        print(f"[PersonDetector] 设备: {self.device}")

        try:
            self.model = YOLO(self.model_path)

            # Check if model is TensorRT engine or PyTorch model
            self.is_tensorrt = self.model_path.endswith('.engine')

            # Only call .to() for PyTorch models, not for TensorRT engines
            if not self.is_tensorrt:
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)

            # Warmup
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_frame, verbose=False, device=self.device)

            print(f"[PersonDetector] 模型加载成功")

        except Exception as e:
            print(f"[PersonDetector] 模型加载失败: {e}")
            raise

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        检测画面中的人体（只检测第一个人）

        Args:
            frame: 输入图像 (H, W, 3)

        Returns:
            bbox: [x1, y1, x2, y2, confidence] 或 None
        """
        start_time = time.time()

        try:
            # YOLO推理
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou,
                classes=[0],  # 0 = person in COCO dataset
                verbose=False,
                device=self.device
            )

            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            # 提取结果
            if len(results) == 0:
                return None

            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                return None

            # 只取第一个检测结果（单人场景）
            box = boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()

            bbox = np.array([x1, y1, x2, y2, confidence], dtype=np.float32)

            # 平滑 bbox（指数滑动）
            if self.smooth_alpha > 0 and self._last_bbox is not None:
                bbox[:4] = self.smooth_alpha * bbox[:4] + (1 - self.smooth_alpha) * self._last_bbox[:4]
                bbox[4] = self.smooth_alpha * bbox[4] + (1 - self.smooth_alpha) * self._last_bbox[4]

            # 扩张 bbox
            if self.expand_ratio != 1.0:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                w = (bbox[2] - bbox[0]) * self.expand_ratio
                h = (bbox[3] - bbox[1]) * self.expand_ratio
                bbox = np.array([
                    max(0.0, cx - w / 2),
                    max(0.0, cy - h / 2),
                    cx + w / 2,
                    cy + h / 2,
                    bbox[4]
                ], dtype=np.float32)

            self._last_bbox = bbox
            return bbox

        except Exception as e:
            print(f"[PersonDetector] 检测失败: {e}")
            return None

    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        if not self.inference_times:
            return {
                'avg_inference_time': 0.0,
                'fps': 0.0
            }

        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0

        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'min_time': np.min(self.inference_times),
            'max_time': np.max(self.inference_times),
        }


class PersonDetectorFactory:
    """人体检测器工厂"""

    @staticmethod
    def create(config: dict) -> DetectorInterface:
        """
        创建人体检测器

        Args:
            config: 配置字典

        Returns:
            检测器实例
        """
        # 目前只支持YOLO
        # 未来可以扩展支持其他检测器（如 MediaPipe, MoveNet等）
        return PersonDetector(config)
