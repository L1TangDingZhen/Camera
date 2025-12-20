"""
Multi-Camera Manager
Manages multiple camera instances and coordinates their operation
"""

import time
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from threading import Thread, Event
import signal
import tkinter as tk

from ..detectors import PersonDetector, PoseEstimatorFactory
from ..state import BehaviorStateMachine, ROIManager, MultiPersonManager
from ..storage import EventLogger


class CameraInstance:
    """
    Single camera instance with full async pipeline

    Each camera runs independently with its own:
    - Person detection (YOLOv8)
    - Pose estimation (RTMPose/MediaPipe)
    - Multi-person tracking (ByteTrack)
    - State management (BehaviorStateMachine)
    """

    def __init__(self, camera_id: int, camera_config: dict, shared_config: dict, event_logger: EventLogger):
        """
        Initialize camera instance

        Args:
            camera_id: Unique camera identifier
            camera_config: Camera-specific configuration
            shared_config: Shared model and system configuration
            event_logger: Shared event logger
        """
        self.camera_id = camera_id
        self.camera_name = camera_config.get('name', f'Camera {camera_id}')
        self.camera_config = camera_config
        self.shared_config = shared_config
        self.event_logger = event_logger

        # Running flag
        self.running = Event()
        self.running.set()

        # Latest frame for visualization
        self.latest_frame = None
        self.frame_lock = Event()
        self.frame_lock.set()

        # Performance stats
        self.fps = 0.0
        self.person_count = 0

        # Verbose mode from config
        self.verbose = shared_config.get('debug', {}).get('verbose', False)

        if self.verbose:
            print(f"[Camera {camera_id}] Initializing '{self.camera_name}'...")

        # Initialize components
        self._init_components()

        if self.verbose:
            print(f"[Camera {camera_id}] Initialization complete")

    def _init_components(self):
        """Initialize detection and tracking components"""

        # 1. Initialize camera capture
        source = self.camera_config['source']
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"[Camera {self.camera_id}] Cannot open camera source: {source}")

        # Set camera parameters with automatic fallback
        requested_width, requested_height = self.camera_config['resolution']
        requested_fps = self.camera_config['fps']

        # Try to set requested resolution with MJPG encoding
        actual_width, actual_height, actual_fps = self._set_camera_resolution(
            requested_width, requested_height, requested_fps
        )

        # Store camera resolution for display scaling
        self.camera_width = actual_width
        self.camera_height = actual_height

        print(f"[Camera {self.camera_id}] Resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")

        # 2. Create person detector (shared model, but independent instances)
        # Note: For memory efficiency, we could share the model weights
        # but keep separate instances for thread safety
        self.person_detector = PersonDetector(self.shared_config['models']['person'])

        # 3. Create pose estimator
        self.pose_estimator = PoseEstimatorFactory.create(self.shared_config['models']['pose'])

        # 4. Create ROI manager
        roi_config = self.camera_config.get('roi', self.shared_config.get('roi', {}))
        self.roi_manager = ROIManager(roi_config)

        # 5. Create multi-person manager or single state machine
        self.enable_multi_person = self.shared_config['models']['person'].get('enable_tracking', False)

        if self.enable_multi_person:
            self.multi_person_manager = MultiPersonManager(
                self.shared_config,
                self.roi_manager,
                self.event_logger
            )
            self.state_machine = None
        else:
            self.state_machine = BehaviorStateMachine(
                self.shared_config,
                self.roi_manager,
                database=self.event_logger.db
            )
            self.multi_person_manager = None

        # Detection interval optimization
        self.detection_interval = self.shared_config.get('inference', {}).get('detection_interval', 3)
        self.detection_counter = 0
        self.cached_data = None  # Cached detections/bbox

    def _set_camera_resolution(self, requested_width: int, requested_height: int, requested_fps: int):
        """
        Set camera resolution with automatic fallback to supported resolutions

        Args:
            requested_width: Requested width
            requested_height: Requested height
            requested_fps: Requested FPS

        Returns:
            (actual_width, actual_height, actual_fps): Actually achieved resolution
        """
        # Define fallback resolution list (from high to low)
        resolution_fallbacks = [
            (requested_width, requested_height),  # Try requested first
            (1920, 1080),  # FHD
            (1280, 720),   # HD
            (640, 480),    # VGA
            (320, 240),    # QVGA (last resort)
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_fallbacks = []
        for res in resolution_fallbacks:
            if res not in seen:
                seen.add(res)
                unique_fallbacks.append(res)

        if self.verbose:
            print(f"[Camera {self.camera_id}] Requested: {requested_width}x{requested_height} @ {requested_fps}fps")

        best_width, best_height = None, None
        best_fps = None

        for try_width, try_height in unique_fallbacks:
            # Set MJPG encoding for bandwidth efficiency
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, try_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, try_height)
            self.cap.set(cv2.CAP_PROP_FPS, requested_fps)

            # Read actual values
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            # Verify by reading a frame
            ret, frame = self.cap.read()

            if ret and frame is not None:
                real_width, real_height = frame.shape[1], frame.shape[0]

                # Check if we got what we asked for (or close enough)
                if (real_width == try_width and real_height == try_height) or \
                   (real_width >= try_width * 0.9 and real_height >= try_height * 0.9):
                    # Success!
                    best_width, best_height = real_width, real_height
                    best_fps = actual_fps

                    if real_width == requested_width and real_height == requested_height:
                        print(f"[Camera {self.camera_id}] ✅ {real_width}x{real_height} @ {actual_fps}fps")
                    else:
                        print(f"[Camera {self.camera_id}] ⚠️  Using {real_width}x{real_height} @ {actual_fps}fps (requested {requested_width}x{requested_height})")

                    break  # Got a working resolution, stop trying
                else:
                    # This resolution didn't work, try next
                    if self.verbose and try_width == requested_width and try_height == requested_height:
                        print(f"[Camera {self.camera_id}] ⚠️  Requested {try_width}x{try_height} not supported, trying lower...")
                    continue
            else:
                # Failed to read frame, try next resolution
                continue

        # If we didn't find any working resolution, use camera default
        if best_width is None:
            print(f"[Camera {self.camera_id}] ⚠️  No standard resolution worked, using camera default")
            ret, frame = self.cap.read()
            if ret:
                best_width, best_height = frame.shape[1], frame.shape[0]
                best_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                print(f"[Camera {self.camera_id}]     Using default: {best_width}x{best_height} @ {best_fps}fps")
            else:
                raise RuntimeError(f"[Camera {self.camera_id}] Cannot read any frame from camera")

        return best_width, best_height, best_fps

    def run(self):
        """Main processing loop for this camera"""
        if self.verbose:
            print(f"[Camera {self.camera_id}] Starting processing loop...")

        frame_count = 0
        fps_time = time.time()

        while self.running.is_set():
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print(f"[Camera {self.camera_id}] Failed to read frame")
                break

            current_time = time.time()
            self.detection_counter += 1

            # 1. Person detection (every N frames)
            if self.detection_counter % self.detection_interval == 0:
                if self.enable_multi_person:
                    self.cached_data = self.person_detector.detect_multi(frame)
                else:
                    self.cached_data = self.person_detector.detect(frame)

            # 2. Pose estimation and state update
            if self.enable_multi_person:
                # Multi-person mode
                detections = self.cached_data if self.cached_data else []

                events = self.multi_person_manager.update_multi(
                    detections,
                    frame,
                    self.pose_estimator,
                    current_time
                )

                # Add camera_id to events
                for event in events:
                    event.metadata['camera_id'] = self.camera_id
                    event.metadata['camera_name'] = self.camera_name

                # Log events (quiet mode: no console output)
                if events:
                    for event in events:
                        if self.verbose:
                            self.event_logger.log_event(event)
                        else:
                            # Silent logging (only to database and file)
                            self.event_logger._log_event(event)

                # Update person count
                self.person_count = len(detections)

                # 3. Visualization
                vis_frame = self._visualize_multi(frame, detections)

            else:
                # Single-person mode
                bbox = self.cached_data

                # Pose estimation
                keypoints = None
                world_landmarks = None
                if bbox is not None:
                    keypoints = self.pose_estimator.estimate(frame, bbox)
                    if hasattr(self.pose_estimator, 'get_world_landmarks'):
                        world_landmarks = self.pose_estimator.get_world_landmarks()

                # State update
                events = self.state_machine.update(bbox, keypoints, current_time, world_landmarks)

                # Add camera_id to events
                for event in events:
                    event.metadata['camera_id'] = self.camera_id
                    event.metadata['camera_name'] = self.camera_name

                # Log events
                if events:
                    for event in events:
                        if self.verbose:
                            self.event_logger.log_event(event)
                        else:
                            # Silent mode: only log to database and file, no console output
                            self.event_logger._log_event(event)

                # Update person count
                self.person_count = 1 if bbox is not None else 0

                # 3. Visualization
                vis_frame = self._visualize_single(frame, bbox, keypoints)

            # Update latest frame (thread-safe)
            self.frame_lock.wait()
            self.frame_lock.clear()
            self.latest_frame = vis_frame
            self.frame_lock.set()

            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                self.fps = frame_count
                frame_count = 0
                fps_time = time.time()

        # Cleanup
        self.cap.release()
        print(f"[Camera {self.camera_id}] Processing loop stopped")

    def _visualize_multi(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Visualize multi-person detection"""
        vis_frame = frame.copy()

        # Draw ROI zones
        vis_frame = self.roi_manager.draw_zones(vis_frame)

        # Color palette for different tracking_ids
        colors = {
            0: (255, 100, 100),   # Light blue
            1: (100, 255, 100),   # Light green
            2: (100, 100, 255),   # Light red
            3: (255, 255, 100),   # Light cyan
            4: (255, 100, 255),   # Light magenta
        }

        # Draw each person
        for detection in detections:
            tracking_id = detection['tracking_id']
            bbox = detection['bbox']
            color = colors.get(tracking_id % len(colors), (0, 255, 0))

            # Draw bbox
            x1, y1, x2, y2 = bbox[:4].astype(int)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw skeleton if available
            if self.multi_person_manager and tracking_id in self.multi_person_manager.person_keypoints:
                keypoints = self.multi_person_manager.person_keypoints[tracking_id]
                if keypoints is not None:
                    self._draw_skeleton(vis_frame, keypoints, color)

            # Get person state
            person_state = "unknown"
            if self.multi_person_manager:
                state = self.multi_person_manager.get_person_state(tracking_id)
                if state:
                    person_state = state

            # Draw label
            label = f"ID:{tracking_id} [{person_state}]"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw camera info
        self._draw_camera_info(vis_frame, len(detections))

        return vis_frame

    def _visualize_single(self, frame: np.ndarray, bbox: Optional[np.ndarray],
                         keypoints: Optional[np.ndarray]) -> np.ndarray:
        """Visualize single-person detection"""
        vis_frame = frame.copy()

        # Draw ROI zones
        vis_frame = self.roi_manager.draw_zones(vis_frame)

        # Draw person if detected
        if bbox is not None:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw skeleton
            if keypoints is not None:
                self._draw_skeleton(vis_frame, keypoints, (0, 255, 0))

            # Draw state
            if self.state_machine:
                state = self.state_machine.current_state.value
                cv2.putText(vis_frame, f"[{state}]", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw camera info
        person_count = 1 if bbox is not None else 0
        self._draw_camera_info(vis_frame, person_count)

        return vis_frame

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, color: tuple):
        """Draw skeleton on frame"""
        from ..detectors.base import Keypoint

        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)

        # Draw skeleton connections
        connections = Keypoint.get_connections()
        for idx1, idx2 in connections:
            if keypoints[idx1, 2] > 0.3 and keypoints[idx2, 2] > 0.3:
                x1, y1 = keypoints[idx1, :2].astype(int)
                x2, y2 = keypoints[idx2, :2].astype(int)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    def _draw_camera_info(self, frame: np.ndarray, person_count: int):
        """Draw camera information overlay"""
        # Camera name and ID
        cv2.putText(frame, f"{self.camera_name} (ID:{self.camera_id})", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Person count
        cv2.putText(frame, f"Persons: {person_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest visualization frame (thread-safe)"""
        self.frame_lock.wait()
        return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        """Stop camera processing"""
        print(f"[Camera {self.camera_id}] Stopping...")
        self.running.clear()


class MultiCameraManager:
    """
    Multi-Camera Manager

    Manages multiple camera instances and provides unified visualization
    """

    def __init__(self, config_path: str):
        """
        Initialize multi-camera manager

        Args:
            config_path: Path to multi-camera configuration file
        """
        print("\n" + "="*70)
        print("  Life Tracker - Multi-Camera System")
        print("="*70 + "\n")

        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Validate configuration
        if 'cameras' not in self.config:
            raise ValueError("Configuration must contain 'cameras' section")

        # Running flag
        self.running = True

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Auto-detect cameras if enabled
        camera_configs = self._get_camera_configs()

        # Create shared event logger
        print("[Init] Creating shared event logger...")
        self.event_logger = EventLogger(self.config)

        # Create camera instances
        self.cameras: List[CameraInstance] = []
        self.camera_threads: List[Thread] = []

        for cam_cfg in camera_configs:
            camera_id = cam_cfg['id']
            try:
                camera = CameraInstance(
                    camera_id=camera_id,
                    camera_config=cam_cfg,
                    shared_config=self.config,
                    event_logger=self.event_logger
                )
                self.cameras.append(camera)
            except Exception as e:
                print(f"[Init] ⚠️  Failed to initialize Camera {camera_id} (source: {cam_cfg['source']}): {e}")
                print(f"[Init] Skipping Camera {camera_id}...")
                continue

        if not self.cameras:
            raise RuntimeError("Failed to initialize any cameras!")

        print(f"\n[Init] {len(self.cameras)} camera(s) initialized successfully")
        print("="*70 + "\n")

        # Detect screen resolution
        self.screen_width, self.screen_height = self._get_screen_resolution()
        print(f"[Init] Screen resolution: {self.screen_width}x{self.screen_height}")

        # Visualization config
        self.show_visualization = not self.config.get('no_visualization', False)
        self.visualization_mode = self.config.get('visualization_mode', 'split_screen')  # 'split_screen' or 'separate_windows'

    def _get_camera_configs(self) -> List[Dict]:
        """
        Get camera configurations (auto-detect or manual)

        Returns:
            List of camera configuration dictionaries
        """
        cameras_config = self.config['cameras']

        # Check if auto-detect is enabled
        if isinstance(cameras_config, dict) and cameras_config.get('auto_detect', False):
            print("[Init] Auto-detecting cameras...")
            return self._auto_detect_cameras(cameras_config)
        elif isinstance(cameras_config, list):
            # Manual configuration
            print(f"[Init] Using manual camera configuration ({len(cameras_config)} cameras)")
            return cameras_config
        else:
            raise ValueError("Invalid 'cameras' configuration format")

    def _auto_detect_cameras(self, config: Dict) -> List[Dict]:
        """
        Automatically detect all available cameras

        Args:
            config: Auto-detect configuration with default settings

        Returns:
            List of detected camera configurations
        """
        detected_cameras = []
        default_resolution = config.get('default_resolution', [1920, 1080])
        default_fps = config.get('default_fps', 30)
        max_device_id = config.get('max_device_id', 10)

        print(f"[AutoDetect] Scanning camera devices 0-{max_device_id}...")

        for device_id in range(max_device_id):
            cap = cv2.VideoCapture(device_id)

            if cap.isOpened():
                # Try to read multiple frames to verify it's a stable camera
                stable = True
                for _ in range(3):  # Try 3 times
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        stable = False
                        break

                if stable:
                    camera_id = len(detected_cameras)
                    cam_config = {
                        'id': camera_id,
                        'name': f'Camera-{camera_id}',
                        'source': device_id,
                        'resolution': default_resolution,
                        'fps': default_fps,
                        'flip': False,
                        'rotation': 0
                    }
                    detected_cameras.append(cam_config)
                    print(f"[AutoDetect] ✅ Found camera at /dev/video{device_id} (Camera-{camera_id})")
                else:
                    print(f"[AutoDetect] ⚠️  /dev/video{device_id} unstable, skipping...")

                cap.release()

        if not detected_cameras:
            raise RuntimeError("No cameras detected! Please check camera connections.")

        print(f"[AutoDetect] Detected {len(detected_cameras)} camera(s)")
        return detected_cameras

    def _get_screen_resolution(self):
        """
        Get screen resolution dynamically

        Returns:
            Tuple[int, int]: (width, height) of screen
        """
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the window
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            return screen_width, screen_height
        except Exception as e:
            print(f"[Warning] Could not detect screen resolution: {e}")
            print(f"[Warning] Using default: 1920x1080")
            return 1920, 1080

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C signal"""
        print("\n[Exit] Received termination signal...")
        self.running = False
        for camera in self.cameras:
            camera.stop()

    def run(self):
        """Start all cameras and visualization"""
        print("[MultiCameraManager] Starting all cameras...")

        # Start camera threads
        for camera in self.cameras:
            thread = Thread(target=camera.run, name=f"Camera-{camera.camera_id}")
            thread.daemon = True
            thread.start()
            self.camera_threads.append(thread)
            print(f"[MultiCameraManager] Camera {camera.camera_id} thread started")

        # Wait for cameras to produce first frames
        time.sleep(2)

        # Visualization loop
        if self.show_visualization:
            self._visualization_loop()
        else:
            # No visualization, just wait for threads
            try:
                for thread in self.camera_threads:
                    thread.join()
            except KeyboardInterrupt:
                pass

        # Cleanup
        self._cleanup()

    def _visualization_loop(self):
        """Main visualization loop with split-screen display"""
        print("[Visualization] Starting split-screen display...")

        # Create window with WINDOW_NORMAL flag (allows resizing and fullscreen)
        window_name = 'Life Tracker - Multi-Camera'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Set window to fullscreen mode (automatically handles desktop environment)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("[Visualization] Window set to fullscreen mode")
        print("[Visualization] Press 'f' to toggle fullscreen, 'q' to quit")

        window_created = True

        while self.running:
            # Collect frames from all cameras
            frames = []
            for camera in self.cameras:
                frame = camera.get_latest_frame()
                if frame is not None:
                    frames.append(frame)
                else:
                    # Create placeholder if camera not ready
                    h, w = 480, 640
                    placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"Camera {camera.camera_id} initializing...",
                               (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    frames.append(placeholder)

            if not frames:
                continue

            # Create split-screen view
            if self.visualization_mode == 'split_screen':
                combined = self._create_split_screen(frames)
                cv2.imshow(window_name, combined)
            else:
                # Separate windows
                if window_created:
                    cv2.destroyWindow(window_name)
                    window_created = False
                for i, frame in enumerate(frames):
                    cv2.imshow(f'Camera {self.cameras[i].camera_id} - {self.cameras[i].camera_name}', frame)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Visualization] User requested quit")
                self.running = False
                break
            elif key == ord('f'):
                # Toggle fullscreen mode
                current_mode = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if current_mode == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print("[Visualization] Switched to windowed mode")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("[Visualization] Switched to fullscreen mode")
            elif key == ord('m'):
                # Toggle visualization mode
                self.visualization_mode = 'separate_windows' if self.visualization_mode == 'split_screen' else 'split_screen'
                cv2.destroyAllWindows()
                window_created = False
                print(f"[Visualization] Switched to {self.visualization_mode} mode")

        cv2.destroyAllWindows()

    def _create_split_screen(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Create split-screen view from multiple frames with dynamic resolution

        智能缩放逻辑：
        1. 计算屏幕分配给每个摄像头的区域
        2. 比较摄像头原始分辨率和分配区域
        3. 使用较小的分辨率，并保持宽高比（防止变形）

        Layout for 1 camera:  [Camera 0 (full screen)]
        Layout for 2 cameras: [Camera 0 | Camera 1]
        Layout for 3 cameras: [Camera 0 | Camera 1]
                              [Camera 2 | Camera 2]  (Camera 2 spans full width)
        Layout for 4+ cameras: [Camera 0 | Camera 1]
                               [Camera 2 | Camera 3]
        """
        num_cameras = len(frames)

        if num_cameras == 0:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        # Calculate layout dimensions based on number of cameras
        if num_cameras == 1:
            # Single camera: use full screen
            rows, cols = 1, 1
        elif num_cameras == 2:
            # Two cameras: side by side
            rows, cols = 1, 2
        elif num_cameras == 3:
            # Three cameras: 2 on top, 1 on bottom
            rows, cols = 2, 2
        else:
            # Four or more cameras: 2x2 grid
            rows, cols = 2, 2

        # Calculate allocated resolution per camera (屏幕分配的区域)
        allocated_width = self.screen_width // cols
        allocated_height = self.screen_height // rows

        # Resize all frames with aspect ratio preservation
        resized_frames = []
        for i, (frame, camera) in enumerate(zip(frames, self.cameras[:num_cameras])):
            # Get camera's original resolution
            cam_width = camera.camera_width
            cam_height = camera.camera_height

            # 比较摄像头分辨率和分配区域，选择较小的
            # Calculate scale to fit allocated area while preserving aspect ratio
            scale_w = allocated_width / cam_width
            scale_h = allocated_height / cam_height
            scale = min(scale_w, scale_h)  # Use smaller scale to fit in allocated area

            # Calculate target size
            target_width = int(cam_width * scale)
            target_height = int(cam_height * scale)

            # 如果摄像头分辨率 < 分配区域，用摄像头分辨率
            # 如果摄像头分辨率 > 分配区域，缩放到分配区域
            if scale >= 1.0:
                # Camera resolution is smaller than allocated area, use original
                target_width = cam_width
                target_height = cam_height

            resized = cv2.resize(frame, (target_width, target_height))

            # Add black padding to match allocated area (to align all cameras)
            padded = np.zeros((allocated_height, allocated_width, 3), dtype=np.uint8)
            y_offset = (allocated_height - target_height) // 2
            x_offset = (allocated_width - target_width) // 2
            padded[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized

            resized_frames.append(padded)

        # Create layout based on number of cameras
        if num_cameras == 1:
            return resized_frames[0]

        elif num_cameras == 2:
            # Side by side: [0 | 1]
            return np.hstack(resized_frames)

        elif num_cameras == 3:
            # Top row: [0 | 1], Bottom row: [2 (full width)]
            top_row = np.hstack(resized_frames[:2])
            bottom_frame = resized_frames[2]
            # Stretch bottom frame to full width
            bottom_row = cv2.resize(bottom_frame, (allocated_width * 2, allocated_height))
            return np.vstack([top_row, bottom_row])

        else:  # 4 or more cameras
            # 2x2 grid (only show first 4 cameras)
            top_row = np.hstack(resized_frames[:2])
            bottom_row = np.hstack(resized_frames[2:4])
            return np.vstack([top_row, bottom_row])

    def _cleanup(self):
        """Cleanup resources"""
        print("\n[Cleanup] Stopping all cameras...")

        # Stop all cameras
        for camera in self.cameras:
            camera.stop()

        # Wait for threads to finish
        for thread in self.camera_threads:
            thread.join(timeout=2.0)

        # Close event logger
        self.event_logger.close()

        print("[Cleanup] All resources released")
        print("="*70)
