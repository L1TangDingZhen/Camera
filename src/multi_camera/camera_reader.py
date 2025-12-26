"""
Camera Reader
Lightweight camera reader that only handles frame capture

Architecture:
- Single responsibility: read frames from camera
- Put frames into queue for InferenceService
- No detection, no pose estimation, no state management
"""

import time
import cv2
import numpy as np
from typing import Optional, Tuple
from threading import Thread, Event
from queue import Queue, Full

from .inference_service import FramePacket


class CameraReader:
    """
    Lightweight camera reader

    Only responsibilities:
    - Open camera device
    - Read frames continuously
    - Put frames into queue with metadata
    """

    def __init__(
        self,
        camera_id: int,
        camera_config: dict,
        output_queue: Queue,
        verbose: bool = False
    ):
        """
        Initialize camera reader

        Args:
            camera_id: Unique camera identifier
            camera_config: Camera configuration (source, resolution, fps)
            output_queue: Queue to put frames into
            verbose: Enable verbose logging
        """
        self.camera_id = camera_id
        self.camera_name = camera_config.get('name', f'Camera {camera_id}')
        self.camera_config = camera_config
        self.output_queue = output_queue
        self.verbose = verbose

        # Running flag
        self.running = Event()
        self.running.set()

        # Frame counter
        self.frame_num = 0

        # Performance stats
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0

        # Camera capture
        self.cap = None
        self.camera_width = 0
        self.camera_height = 0

        # Initialize camera
        self._init_camera()

    def _init_camera(self):
        """Initialize camera capture with resolution fallback"""

        source = self.camera_config.get('source', self.camera_id)
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"[CameraReader {self.camera_id}] Cannot open camera source: {source}")

        # Set resolution with fallback
        requested_width, requested_height = self.camera_config.get('resolution', [1920, 1080])
        requested_fps = self.camera_config.get('fps', 30)

        actual_width, actual_height, actual_fps = self._set_camera_resolution(
            requested_width, requested_height, requested_fps
        )

        self.camera_width = actual_width
        self.camera_height = actual_height

        print(f"[CameraReader {self.camera_id}] '{self.camera_name}' initialized: {actual_width}x{actual_height} @ {actual_fps}fps")

    def _set_camera_resolution(
        self,
        requested_width: int,
        requested_height: int,
        requested_fps: int
    ) -> Tuple[int, int, int]:
        """
        Set camera resolution with automatic fallback

        Args:
            requested_width: Requested width
            requested_height: Requested height
            requested_fps: Requested FPS

        Returns:
            (actual_width, actual_height, actual_fps)
        """
        # Fallback resolutions
        resolution_fallbacks = [
            (requested_width, requested_height),
            (1920, 1080),
            (1280, 720),
            (640, 480),
            (320, 240),
        ]

        # Remove duplicates
        seen = set()
        unique_fallbacks = []
        for res in resolution_fallbacks:
            if res not in seen:
                seen.add(res)
                unique_fallbacks.append(res)

        for try_width, try_height in unique_fallbacks:
            # Set MJPG encoding
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # Set resolution and fps
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, try_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, try_height)
            self.cap.set(cv2.CAP_PROP_FPS, requested_fps)

            # Verify by reading a frame
            ret, frame = self.cap.read()

            if ret and frame is not None:
                real_width, real_height = frame.shape[1], frame.shape[0]
                actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

                if (real_width >= try_width * 0.9 and real_height >= try_height * 0.9):
                    if self.verbose:
                        if real_width == requested_width and real_height == requested_height:
                            print(f"[CameraReader {self.camera_id}] ✓ {real_width}x{real_height} @ {actual_fps}fps")
                        else:
                            print(f"[CameraReader {self.camera_id}] Using {real_width}x{real_height} (requested {requested_width}x{requested_height})")

                    return real_width, real_height, actual_fps

        # Fallback: use whatever the camera gives
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"[CameraReader {self.camera_id}] ⚠ Fallback: {actual_width}x{actual_height} @ {actual_fps}fps")
        return actual_width, actual_height, actual_fps

    def run(self):
        """Main loop - continuously read frames and put into queue"""

        print(f"[CameraReader {self.camera_id}] Starting capture loop...")

        while self.running.is_set():
            # Read frame
            ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.verbose:
                    print(f"[CameraReader {self.camera_id}] Failed to read frame")
                time.sleep(0.01)
                continue

            # Create frame packet
            self.frame_num += 1
            timestamp = time.time()

            packet = FramePacket(
                frame=frame,
                camera_id=self.camera_id,
                frame_num=self.frame_num,
                timestamp=timestamp
            )

            # Put into queue (non-blocking)
            try:
                self.output_queue.put_nowait(packet)
            except Full:
                # Queue full, drop oldest frame
                try:
                    self.output_queue.get_nowait()
                    self.output_queue.put_nowait(packet)
                except:
                    pass

            # Update FPS stats
            self._update_fps()

        print(f"[CameraReader {self.camera_id}] Capture loop stopped")

    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time

        if elapsed >= 1.0:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_time = current_time

    def stop(self):
        """Stop the camera reader"""
        self.running.clear()
        if self.cap is not None:
            self.cap.release()
        print(f"[CameraReader {self.camera_id}] Stopped")

    def get_resolution(self) -> Tuple[int, int]:
        """Get camera resolution"""
        return self.camera_width, self.camera_height

    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps

    def get_frame_count(self) -> int:
        """Get total frames read"""
        return self.frame_num
