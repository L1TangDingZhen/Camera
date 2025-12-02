"""
TensorRT Engine Wrapper for RTMPose

This module provides a简化的TensorRT推理接口，用于加载和运行RTMPose TensorRT引擎。
"""

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import torch
from typing import Tuple, Optional

# DON'T import pycuda.autoinit - it creates a conflicting CUDA context!
# Instead, we'll use PyTorch's CUDA context when needed

# Initialize CUDA driver manually
cuda.init()

# We'll create context on-demand using PyTorch's context


class TensorRTEngine:
    """TensorRT引擎包装器"""

    def __init__(self, engine_path: str):
        """
        Initialize TensorRT engine

        Args:
            engine_path: Path to the .engine file
        """
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Get PyCUDA context from PyTorch (avoid creating conflicting context)
        if torch.cuda.is_available():
            # Make PyTorch create CUDA context first
            torch.cuda.current_device()

            # Use primary context (shared with PyTorch, no conflict)
            self.cuda_ctx = cuda.Device(0).retain_primary_context()
            self.cuda_ctx.push()
        else:
            self.cuda_ctx = None

        # Load engine
        print(f"[TensorRT] Loading engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()

        # Create CUDA stream
        self.stream = cuda.Stream()

        # Get input/output information
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

            print(f"[TensorRT]   Tensor: {name}, Shape: {shape}, Dtype: {dtype}")

        # Allocate device memory
        self._allocate_buffers()

        print(f"[TensorRT] Engine loaded successfully")
        print(f"[TensorRT]   Inputs: {self.input_names}")
        print(f"[TensorRT]   Outputs: {self.output_names}")

    def _allocate_buffers(self):
        """Allocate GPU memory for inputs and outputs (TensorRT 10.x API)"""
        self.inputs = {}
        self.outputs = {}

        for name in self.input_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            # Allocate device memory
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.inputs[name] = {
                'shape': shape,
                'dtype': dtype,
                'device': device_mem,
                'host': None  # Will allocate on demand
            }

        for name in self.output_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            # Allocate device and host memory
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            host_mem = np.empty(shape, dtype=dtype)

            self.outputs[name] = {
                'shape': shape,
                'dtype': dtype,
                'device': device_mem,
                'host': host_mem
            }

    def infer(self, input_data: np.ndarray) -> dict:
        """
        Run inference (TensorRT 10.x API)

        Args:
            input_data: Input numpy array (should match engine input shape)

        Returns:
            dict: Dictionary of output tensors {name: numpy_array}
        """
        # Assume single input for now
        input_name = self.input_names[0]

        # Verify input shape
        expected_shape = self.inputs[input_name]['shape']
        if input_data.shape != tuple(expected_shape):
            raise ValueError(
                f"Input shape mismatch: expected {expected_shape}, got {input_data.shape}"
            )

        # Copy input to device (async)
        cuda.memcpy_htod_async(
            self.inputs[input_name]['device'],
            input_data,
            self.stream
        )

        # Set tensor addresses (TensorRT 10.x API)
        for name in self.input_names:
            self.context.set_tensor_address(name, int(self.inputs[name]['device']))

        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.outputs[name]['device']))

        # Run inference
        success = self.context.execute_async_v3(self.stream.handle)
        if not success:
            raise RuntimeError("TensorRT inference execution failed")

        # Synchronize stream
        self.stream.synchronize()

        # Copy outputs to host
        results = {}
        for name in self.output_names:
            cuda.memcpy_dtoh_async(
                self.outputs[name]['host'],
                self.outputs[name]['device'],
                self.stream
            )

        # Final synchronization
        self.stream.synchronize()

        # Copy results
        for name in self.output_names:
            results[name] = self.outputs[name]['host'].copy()

        return results

    def __del__(self):
        """Cleanup"""
        # Pop CUDA context if we created one
        if hasattr(self, 'cuda_ctx') and self.cuda_ctx is not None:
            try:
                self.cuda_ctx.pop()
            except:
                pass  # Context may already be destroyed
        # PyCUDA will handle memory cleanup automatically


class TensorRTRTMPose:
    """
    RTMPose TensorRT推理器

    这个类封装了RTMPose的TensorRT引擎，提供与MMPose类似的接口
    """

    def __init__(self, engine_path: str, device: str = 'cuda:0'):
        """
        Initialize RTMPose TensorRT engine

        Args:
            engine_path: Path to RTMPose TensorRT engine file
            device: CUDA device (currently only supports cuda:0)
        """
        self.engine = TensorRTEngine(engine_path)
        self.device = device

        # Expected input shape (from RTMPose training)
        self.input_size = (256, 192)  # H, W
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def _expand_bbox(self, bbox: np.ndarray, img_shape, padding: float = 1.25):
        """
        Expand bbox with padding (MMPose style)

        Args:
            bbox: [x1, y1, x2, y2, score]
            img_shape: (H, W)
            padding: Bbox padding factor

        Returns:
            Expanded bbox [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox[:4]
        img_h, img_w = img_shape

        # Calculate center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Calculate scale with padding
        w = (x2 - x1) * padding
        h = (y2 - y1) * padding

        # Expand bbox
        x1_new = max(0, center_x - w / 2)
        y1_new = max(0, center_y - h / 2)
        x2_new = min(img_w, center_x + w / 2)
        y2_new = min(img_h, center_y + h / 2)

        return np.array([x1_new, y1_new, x2_new, y2_new])

    def _get_warp_matrix(self, center, scale, rot, output_size, inv=False):
        """
        Get affine transformation matrix matching MMPose's get_warp_matrix exactly.

        Args:
            center: bbox center (x, y)
            scale: bbox scale (w, h) after padding
            rot: rotation angle in degrees
            output_size: target size (w, h)
            inv: if True, return inverse transform (dst->src)

        Returns:
            2x3 affine transformation matrix
        """
        import cv2
        import math

        def _rotate_point(pt, angle_rad):
            """Rotate a point by an angle."""
            sn, cs = np.sin(angle_rad), np.cos(angle_rad)
            new_x = pt[0] * cs - pt[1] * sn
            new_y = pt[0] * sn + pt[1] * cs
            return np.array([new_x, new_y], dtype=np.float32)

        def _get_3rd_point(a, b):
            """Get 3rd point for affine transform (perpendicular)."""
            direction = a - b
            return b + np.array([-direction[1], direction[0]], dtype=np.float32)

        src_w = scale[0]
        dst_w, dst_h = output_size

        rot_rad = np.deg2rad(rot)
        src_dir = _rotate_point(np.array([0., src_w * -0.5], dtype=np.float32), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        src[2, :] = _get_3rd_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            warp_mat = cv2.getAffineTransform(dst.astype(np.float32), src.astype(np.float32))
        else:
            warp_mat = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))

        return warp_mat

    def _get_3rd_point(self, a, b):
        """Get 3rd point for affine transform"""
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def preprocess(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Preprocess image for RTMPose inference (完全匹配MMPose的预处理)

        Args:
            img: Input image (H, W, 3) in BGR format
            bbox: Bounding box [x1, y1, x2, y2, score]

        Returns:
            Preprocessed tensor (1, 3, 256, 192) in FP32
        """
        import cv2

        # 从bbox计算center和scale（匹配MMPose的bbox_xyxy2cs）
        x1, y1, x2, y2 = bbox[:4]
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

        # Scale = bbox dimensions * padding
        padding = 1.25
        w = (x2 - x1) * padding
        h = (y2 - y1) * padding

        # 调整aspect ratio（匹配MMPose的TopdownAffine._fix_aspect_ratio）
        aspect_ratio = self.input_size[1] / self.input_size[0]  # 192/256 = 0.75
        if w > h * aspect_ratio:
            scale = np.array([w, w / aspect_ratio], dtype=np.float32)
        else:
            scale = np.array([h * aspect_ratio, h], dtype=np.float32)

        # 保存center和scale用于后处理
        self._last_center = center
        self._last_scale = scale

        # 获取仿射变换矩阵（使用MMPose的get_warp_matrix）
        output_size = (self.input_size[1], self.input_size[0])  # (W, H) = (192, 256)
        trans = self._get_warp_matrix(center, scale, 0, output_size, inv=False)

        # 使用仿射变换裁剪和缩放图像
        warped = cv2.warpAffine(
            img,
            trans,
            output_size,
            flags=cv2.INTER_LINEAR
        )

        # 转换BGR到RGB（ImageNet标准）
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

        # Normalize (ImageNet RGB mean/std)
        normalized = (warped_rgb.astype(np.float32) - self.mean) / self.std

        # HWC -> CHW
        transposed = normalized.transpose(2, 0, 1)

        # Add batch dimension
        batched = transposed[np.newaxis, ...]

        # Ensure contiguous memory layout (critical for TensorRT)
        return np.ascontiguousarray(batched, dtype=np.float32)

    def postprocess(self, outputs: dict, bbox: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess RTMPose outputs (使用MMPose标准的仿射变换逆矩阵)

        Args:
            outputs: TensorRT engine outputs
            bbox: Original bounding box [x1, y1, x2, y2, score]
            img_shape: Original image shape (H, W)

        Returns:
            Keypoints in COCO format (17, 3) [x, y, confidence]
        """
        import cv2

        # RTMPose outputs: SimCC format (X and Y distributions)
        # 新版engine: 'simcc_x' / 'simcc_y'
        # 旧版engine: 'output' / '501'

        # 尝试新版输出名称
        if 'simcc_x' in outputs:
            simcc_x = outputs['simcc_x']  # (1, 17, 384) - x coordinates
            simcc_y = outputs['simcc_y']  # (1, 17, 512) - y coordinates
        # 尝试旧版输出名称
        elif 'output' in outputs:
            simcc_x = outputs['output']  # (1, 17, 384) - x coordinates
            simcc_y = outputs.get('501', outputs['output'])  # (1, 17, 512) - y coordinates
        # 通用fallback
        else:
            output_list = list(outputs.values())
            simcc_x = output_list[0]
            simcc_y = output_list[1] if len(output_list) > 1 else output_list[0]

        # Decode SimCC predictions
        batch_size, num_keypoints, _ = simcc_x.shape
        x_logits = simcc_x[0]
        y_logits = simcc_y[0]

        # Argmax decode
        x_coords = np.argmax(x_logits, axis=-1)
        y_coords = np.argmax(y_logits, axis=-1)

        # Confidence from max logits
        x_scores = np.max(x_logits, axis=-1)
        y_scores = np.max(y_logits, axis=-1)
        scores = np.sqrt(x_scores * y_scores)

        # SimCC坐标映射到模型输入图像坐标
        # RTMPose的SimCC: simcc_split_ratio = 2.0
        simcc_split_ratio = 2.0
        x_coords_img = x_coords.astype(np.float32) / simcc_split_ratio
        y_coords_img = y_coords.astype(np.float32) / simcc_split_ratio

        # 获取仿射变换逆矩阵（使用MMPose的get_warp_matrix，inv=True）
        center = self._last_center
        scale = self._last_scale
        output_size = (self.input_size[1], self.input_size[0])  # (W, H) = (192, 256)

        # 直接获取逆矩阵，避免数值误差
        trans_inv = self._get_warp_matrix(center, scale, 0, output_size, inv=True)

        # 将每个关键点映射回原图坐标
        keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)
        for i in range(num_keypoints):
            pt = np.array([x_coords_img[i], y_coords_img[i], 1.0])
            pt_transformed = trans_inv @ pt
            keypoints[i, 0] = pt_transformed[0]
            keypoints[i, 1] = pt_transformed[1]
            keypoints[i, 2] = scores[i]

        return keypoints

    def __call__(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Run inference on a single person

        Args:
            img: Input image (H, W, 3) in BGR format
            bbox: Bounding box [x1, y1, x2, y2, score]

        Returns:
            Keypoints (17, 3) [x, y, confidence]
        """
        # Preprocess
        input_tensor = self.preprocess(img, bbox)

        # Inference
        outputs = self.engine.infer(input_tensor)

        # Postprocess
        img_shape = img.shape[:2]
        keypoints = self.postprocess(outputs, bbox, img_shape)

        return keypoints
