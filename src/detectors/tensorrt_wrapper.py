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

    def _get_affine_transform(self, center, scale, rot, output_size):
        """
        Get affine transformation matrix (from MMPose)
        """
        import cv2

        shift = np.array([0, 0], dtype=np.float32)
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = np.array([0, src_w * -0.5], dtype=np.float32)
        dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    def _get_3rd_point(self, a, b):
        """Get 3rd point for affine transform"""
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def preprocess(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Preprocess image for RTMPose inference (simplified, compatible)

        Args:
            img: Input image (H, W, 3) in BGR format
            bbox: Bounding box [x1, y1, x2, y2, score]

        Returns:
            Preprocessed tensor (1, 3, 256, 192) in FP32
        """
        import cv2

        # Expand bbox with 1.25x padding (like MMPose)
        expanded_bbox = self._expand_bbox(bbox, img.shape[:2], padding=1.25)
        x1, y1, x2, y2 = [int(v) for v in expanded_bbox]

        # Crop expanded region
        person_img = img[y1:y2, x1:x2]

        if person_img.size == 0:
            # Return zero tensor if crop failed
            return np.zeros((1, 3, *self.input_size), dtype=np.float32)

        # Resize to input size (W, H)
        resized = cv2.resize(person_img, (self.input_size[1], self.input_size[0]))

        # Keep BGR format (MMPose expects BGR, mean/std are in BGR order)
        # DO NOT convert to RGB!

        # Normalize (input is BGR, mean/std are in BGR order)
        normalized = (resized.astype(np.float32) - self.mean) / self.std

        # Store bbox for postprocessing
        self._last_bbox_expanded = expanded_bbox

        # HWC -> CHW
        transposed = normalized.transpose(2, 0, 1)

        # Add batch dimension
        batched = transposed[np.newaxis, ...]

        # Ensure contiguous memory layout (critical for TensorRT)
        return np.ascontiguousarray(batched, dtype=np.float32)

    def postprocess(self, outputs: dict, bbox: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess RTMPose outputs (simplified)

        Args:
            outputs: TensorRT engine outputs
            bbox: Original bounding box [x1, y1, x2, y2, score]
            img_shape: Original image shape (H, W)

        Returns:
            Keypoints in COCO format (17, 3) [x, y, confidence]
        """
        # RTMPose outputs: 'output' and potentially '501' (intermediate features)
        # We need the final output which should be shape (1, 17, 384) - SimCC format

        if 'output' in outputs:
            simcc_x = outputs['output']  # (1, 17, 384) - x coordinates
        else:
            # Use first output
            simcc_x = list(outputs.values())[0]

        if '501' in outputs:
            simcc_y = outputs['501']  # (1, 17, 512) - y coordinates
        else:
            # If only one output, duplicate for y (fallback)
            simcc_y = simcc_x

        # Decode SimCC predictions with softmax (temperature to sharpen peaks)
        batch_size, num_keypoints, _ = simcc_x.shape
        x_logits = simcc_x[0]
        y_logits = simcc_y[0]

        # Argmax decode (consistent with原始实现)
        x_coords = np.argmax(x_logits, axis=-1)
        y_coords = np.argmax(y_logits, axis=-1)

        # Confidence from max logits
        x_scores = np.max(x_logits, axis=-1)
        y_scores = np.max(y_logits, axis=-1)
        scores = np.sqrt(x_scores * y_scores)

        # Convert from SimCC coordinates to pixel coordinates in resized image
        # SimCC uses 384 bins for x, 512 bins for y (for 192x256 input)
        x_scale = self.input_size[1] / simcc_x.shape[-1]  # 192 / 384 = 0.5
        y_scale = self.input_size[0] / simcc_y.shape[-1]  # 256 / 512 = 0.5

        x_pixels = x_coords * x_scale  # In resized image (192 width)
        y_pixels = y_coords * y_scale  # In resized image (256 height)

        # Map back to expanded bbox coordinates
        expanded_bbox = self._last_bbox_expanded
        x1, y1, x2, y2 = expanded_bbox

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Scale from resized image to expanded bbox
        x_original = x1 + (x_pixels / self.input_size[1]) * bbox_w
        y_original = y1 + (y_pixels / self.input_size[0]) * bbox_h

        # Combine into (17, 3) format
        keypoints = np.stack([x_original, y_original, scores], axis=-1)

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
