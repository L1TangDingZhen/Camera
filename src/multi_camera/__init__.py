"""
Multi-Camera Module
Supports simultaneous tracking across multiple camera feeds
"""

from .camera_manager import MultiCameraManager, CameraInstance

__all__ = [
    'MultiCameraManager',
    'CameraInstance',
]
