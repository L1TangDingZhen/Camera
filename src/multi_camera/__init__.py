"""
Multi-Camera Module
Supports simultaneous tracking across multiple camera feeds

Architecture:
- SharedPipelineManager: Optimized pipeline with shared inference
  - Standard Mode: ceil(N/2) inference services
  - Batch Mode: 1 BatchInferenceService (1.76x speedup, recommended)
- MultiCameraManager: Legacy pipeline with independent inference per camera

Resource allocation (Standard Mode):
- 1-2 cameras: 1 inference service, 1.5GB GPU memory
- 3-4 cameras: 2 inference services, 3.0GB GPU memory
- N cameras: ceil(N/2) inference services, ceil(N/2)*1.5GB GPU memory

Resource allocation (Batch Mode - Recommended):
- N cameras: 1 batch inference service, ~1.5GB GPU memory
- Performance: 1.76x speedup at batch=8 compared to sequential
"""

from .camera_manager import MultiCameraManager, CameraInstance
from .camera_reader import CameraReader
from .inference_service import InferenceService, FramePacket, ResultPacket
from .batch_inference_service import BatchInferenceService
from .shared_pipeline import SharedPipelineManager

__all__ = [
    # Optimized pipelines (recommended)
    'SharedPipelineManager',
    'InferenceService',
    'BatchInferenceService',
    'CameraReader',
    'FramePacket',
    'ResultPacket',

    # Legacy pipeline (for compatibility)
    'MultiCameraManager',
    'CameraInstance',
]
