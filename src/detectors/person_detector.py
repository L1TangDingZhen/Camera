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

        # 性能统计
        self.inference_times = []

        # 加载模型
        print(f"[PersonDetector] 加载模型: {self.model_path}")
        print(f"[PersonDetector] 设备: {self.device}")

        try:
            self.model = YOLO(self.model_path)
            # 设置模型参数
            if hasattr(self.model, 'to'):
                self.model.to(self.device)

            # 预热
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_frame, verbose=False)

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

            return np.array([x1, y1, x2, y2, confidence], dtype=np.float32)

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
