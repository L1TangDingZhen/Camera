"""
YOLOv8-based person detector
Supports PyTorch (.pt) and TensorRT (.engine) models
"""

import time
from typing import Optional, Dict, List
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .base import DetectorInterface


class PersonDetector(DetectorInterface):
    """YOLOv8 Person Detector"""

    def __init__(self, config: dict):
        super().__init__(config)

        if YOLO is None:
            raise ImportError("Please install ultralytics: pip install ultralytics")

        self.model_path = config.get('model', 'yolov8s.pt')
        self.confidence = config.get('confidence', 0.5)
        self.iou = config.get('iou', 0.45)
        self.device = config.get('device', 'cpu')

        # Single-person detection config
        self.smooth_alpha = config.get('smooth_alpha', 0.0)  # 0 = no smoothing
        self.expand_ratio = config.get('expand_ratio', 1.0)  # 1 = no expansion
        self._last_bbox = None

        # Multi-person detection and tracking config
        self.enable_tracking = config.get('enable_tracking', False)
        self.max_persons = config.get('max_persons', 5)  # Max 5 persons to track
        self._last_bboxes = {}  # tracking_id -> bbox (for smoothing)

        # Performance statistics
        self.inference_times = []

        # Batch size configuration (for batch TensorRT engines)
        self.fixed_batch_size = config.get('fixed_batch_size', None)  # e.g., 4 for batch=4 engine

        # Load model
        print(f"[PersonDetector] Loading model: {self.model_path}")
        print(f"[PersonDetector] Device: {self.device}")

        try:
            # Explicitly specify task='detect' to eliminate warnings
            self.model = YOLO(self.model_path, task='detect')

            # Check if model is TensorRT engine or PyTorch model
            self.is_tensorrt = self.model_path.endswith('.engine')

            # Only call .to() for PyTorch models, not for TensorRT engines
            if not self.is_tensorrt:
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)

            # Warmup with correct batch size for fixed-batch engines
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            if self.fixed_batch_size:
                # For fixed batch engines, warmup with full batch
                dummy_batch = [dummy_frame] * self.fixed_batch_size
                _ = self.model(dummy_batch, verbose=False, device=self.device)
            else:
                _ = self.model(dummy_frame, verbose=False, device=self.device)

            print(f"[PersonDetector] Model loaded successfully (batch={self.fixed_batch_size or 'dynamic'})")

        except Exception as e:
            print(f"[PersonDetector] Model loading failed: {e}")
            raise

    def detect_batch(self, frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Batch detection - detect person in multiple frames at once

        Args:
            frames: List of input images [(H, W, 3), ...]

        Returns:
            List of bboxes: [[x1, y1, x2, y2, confidence] or None, ...]
        """
        if len(frames) == 0:
            return []

        start_time = time.time()
        original_count = len(frames)

        try:
            # Handle fixed batch size engines (pad if needed)
            if self.fixed_batch_size and len(frames) < self.fixed_batch_size:
                # Pad with first frame to reach fixed batch size
                padding_count = self.fixed_batch_size - len(frames)
                frames = frames + [frames[0]] * padding_count

            # YOLO batch inference
            results = self.model(
                frames,
                conf=self.confidence,
                iou=self.iou,
                classes=[0],
                verbose=False,
                device=self.device
            )

            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            # Extract bboxes for each frame (only original frames, not padding)
            bboxes = []
            for i, result in enumerate(results):
                if i >= original_count:
                    break  # Skip padded results
                if result.boxes is None or len(result.boxes) == 0:
                    bboxes.append(None)
                else:
                    box = result.boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    bbox = np.array([x1, y1, x2, y2, confidence], dtype=np.float32)
                    bboxes.append(bbox)

            return bboxes

        except Exception as e:
            print(f"[PersonDetector] Batch detection failed: {e}")
            return [None] * original_count

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect person in frame (only first person)

        Args:
            frame: Input image (H, W, 3)

        Returns:
            bbox: [x1, y1, x2, y2, confidence] or None
        """
        start_time = time.time()

        try:
            # YOLO inference
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou,
                classes=[0],  # 0 = person in COCO dataset
                verbose=False,
                device=self.device
            )

            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            # Extract results
            if len(results) == 0:
                return None

            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                return None

            # Only take first detection (single-person scenario)
            box = boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()

            bbox = np.array([x1, y1, x2, y2, confidence], dtype=np.float32)

            # Smooth bbox (exponential moving average)
            if self.smooth_alpha > 0 and self._last_bbox is not None:
                bbox[:4] = self.smooth_alpha * bbox[:4] + (1 - self.smooth_alpha) * self._last_bbox[:4]
                bbox[4] = self.smooth_alpha * bbox[4] + (1 - self.smooth_alpha) * self._last_bbox[4]

            # Expand bbox
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
            print(f"[PersonDetector] Detection failed: {e}")
            return None

    def detect_multi(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect multiple persons in frame and track them

        Args:
            frame: Input image (H, W, 3)

        Returns:
            List[Dict]: Detection results list, each dict contains:
                {
                    'bbox': np.ndarray [x1, y1, x2, y2, confidence],
                    'tracking_id': int,
                    'confidence': float
                }
                Returns empty list if no persons detected
        """
        start_time = time.time()

        try:
            # Use YOLO's track feature for multi-object tracking
            results = self.model.track(
                frame,
                conf=self.confidence,
                iou=self.iou,
                classes=[0],  # 0 = person in COCO dataset
                verbose=False,
                device=self.device,
                persist=True,  # Persist tracker state
                tracker="bytetrack.yaml"  # Use ByteTrack
            )

            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            # Extract results
            if len(results) == 0:
                return []

            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                return []

            # Extract all detected persons
            detections = []

            for i, box in enumerate(boxes):
                # Skip if exceeds max tracking persons
                if i >= self.max_persons:
                    break

                # Extract bbox coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())

                # Extract tracking ID (if available)
                if hasattr(box, 'id') and box.id is not None:
                    tracking_id = int(box.id[0].cpu().numpy())
                else:
                    # If no ID, use index as temporary ID
                    tracking_id = i

                bbox = np.array([x1, y1, x2, y2, confidence], dtype=np.float32)

                # Smooth bbox (per tracking_id)
                if self.smooth_alpha > 0 and tracking_id in self._last_bboxes:
                    last_bbox = self._last_bboxes[tracking_id]
                    bbox[:4] = self.smooth_alpha * bbox[:4] + (1 - self.smooth_alpha) * last_bbox[:4]
                    bbox[4] = self.smooth_alpha * bbox[4] + (1 - self.smooth_alpha) * last_bbox[4]

                # Expand bbox
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

                # Save current bbox for next frame smoothing
                self._last_bboxes[tracking_id] = bbox.copy()

                # Add to detection list
                detections.append({
                    'bbox': bbox,
                    'tracking_id': tracking_id,
                    'confidence': confidence
                })

            return detections

        except Exception as e:
            print(f"[PersonDetector] Multi-person detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
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

    def cleanup(self):
        """Clean up GPU resources - call before program exit"""
        # Clear model reference
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None

        # Clear cached bboxes
        self._last_bbox = None
        self._last_bboxes = {}

        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


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
