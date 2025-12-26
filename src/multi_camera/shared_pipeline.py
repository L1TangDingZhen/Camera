"""
Shared Pipeline Manager
Manages cameras and inference services with optimal resource allocation

Architecture (Standard Mode):
- N cameras → N CameraReader threads
- ceil(N/2) InferenceService threads (each handles up to 2 cameras)
- Total threads: N + ceil(N/2)
- GPU memory: ceil(N/2) × 1.5GB

Architecture (Batch Mode - Recommended):
- N cameras → N CameraReader threads
- 1 BatchInferenceService thread (handles ALL cameras)
- Total threads: N + 1
- GPU memory: ~1.5GB (single shared model)
- Performance: 1.76x speedup at batch=8

Thread allocation (Standard):
- 1 camera  → 2 threads (1 Camera + 1 Inference)
- 2 cameras → 3 threads (2 Camera + 1 Inference)
- 3 cameras → 5 threads (3 Camera + 2 Inference)
- 4 cameras → 6 threads (4 Camera + 2 Inference)

Thread allocation (Batch):
- N cameras → N + 1 threads (N Camera + 1 BatchInference)
"""

import math
import time
import cv2
import yaml
import numpy as np
from typing import List, Dict, Optional, Tuple
from threading import Thread, Event
from queue import Queue
import signal
import tkinter as tk

from .camera_reader import CameraReader
from .inference_service import InferenceService, ResultPacket
from .batch_inference_service import BatchInferenceService
from ..storage import EventLogger


class SharedPipelineManager:
    """
    Pipeline manager with shared inference resources

    Features:
    - Automatic resource allocation based on camera count
    - Shared YOLO + RTMPose models (per inference service)
    - Fair scheduling across cameras
    - Unified visualization
    - Batch mode for optimal GPU utilization (1.76x speedup)
    """

    def __init__(self, config_path: str):
        """
        Initialize pipeline manager

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path

        # Running flag
        self.running = Event()
        self.running.set()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Components
        self.camera_readers: List[CameraReader] = []
        self.inference_services: List[InferenceService] = []
        self.camera_threads: List[Thread] = []
        self.inference_threads: List[Thread] = []

        # Queues
        self.input_queues: Dict[int, Queue] = {}   # camera_id -> frame queue
        self.output_queues: Dict[int, Queue] = {}  # camera_id -> result queue

        # Shared resources
        self.database = None
        self.event_logger = None

        # Camera configurations
        self.camera_configs: List[dict] = []

        # Verbose mode
        self.verbose = self.config.get('debug', {}).get('verbose', False)

        # Batch mode configuration
        batch_config = self.config.get('batch', {})
        self.batch_mode = batch_config.get('enabled', False)

        # Visualization
        self.window_name = "Life Tracker - Shared Pipeline"
        self.fullscreen = False

        # Initialize
        self._init_cameras()
        if self.batch_mode:
            self._init_resources_batch()
        else:
            self._init_resources()
        self._print_allocation_info()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n[SharedPipeline] Received shutdown signal...")
        self.stop()

    def _init_cameras(self):
        """Parse camera configurations"""

        cameras_config = self.config.get('cameras', {})

        if isinstance(cameras_config, dict):
            if cameras_config.get('auto_detect', False):
                # Auto-detect cameras
                self.camera_configs = self._auto_detect_cameras(cameras_config)
            else:
                # Single camera from 'camera' config
                camera_config = self.config.get('camera', {})
                if camera_config:
                    self.camera_configs = [{
                        'id': 0,
                        'name': 'Camera-0',
                        'source': camera_config.get('source', 0),
                        'resolution': camera_config.get('resolution', [1920, 1080]),
                        'fps': camera_config.get('fps', 30),
                    }]
        elif isinstance(cameras_config, list):
            # Manual camera list
            self.camera_configs = cameras_config

        if not self.camera_configs:
            # Fallback to single camera
            camera_config = self.config.get('camera', {})
            self.camera_configs = [{
                'id': 0,
                'name': 'Camera-0',
                'source': camera_config.get('source', 0),
                'resolution': camera_config.get('resolution', [1920, 1080]),
                'fps': camera_config.get('fps', 30),
            }]

        print(f"[SharedPipeline] Found {len(self.camera_configs)} camera(s)")

    def _auto_detect_cameras(self, cameras_config: dict) -> List[dict]:
        """Auto-detect available cameras"""

        default_resolution = cameras_config.get('default_resolution', [1920, 1080])
        default_fps = cameras_config.get('default_fps', 30)
        max_device_id = cameras_config.get('max_device_id', 10)

        detected = []

        for device_id in range(max_device_id):
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    detected.append({
                        'id': len(detected),
                        'name': f'Camera-{len(detected)}',
                        'source': device_id,
                        'resolution': default_resolution,
                        'fps': default_fps,
                    })
                    print(f"[SharedPipeline] Detected camera at /dev/video{device_id}")
            cap.release()

        return detected

    def _init_resources(self):
        """Initialize shared resources and allocate threads"""

        num_cameras = len(self.camera_configs)

        # Calculate number of inference services needed
        num_inference_services = math.ceil(num_cameras / 2)

        print(f"[SharedPipeline] Allocating resources:")
        print(f"  - Cameras: {num_cameras}")
        print(f"  - Inference services: {num_inference_services}")
        print(f"  - Total threads: {num_cameras + num_inference_services}")
        print(f"  - Estimated GPU memory: {num_inference_services * 1.5:.1f} GB")

        # Initialize event logger (it creates its own database connection)
        self.event_logger = EventLogger(self.config)
        self.database = self.event_logger.db

        # Create queues for each camera (small size to reduce latency)
        queue_size = self.config.get('inference', {}).get('max_buffer_size', 2)
        for cam_cfg in self.camera_configs:
            cam_id = cam_cfg['id']
            self.input_queues[cam_id] = Queue(maxsize=queue_size)
            self.output_queues[cam_id] = Queue(maxsize=queue_size)

        # Create camera readers
        for cam_cfg in self.camera_configs:
            cam_id = cam_cfg['id']
            reader = CameraReader(
                camera_id=cam_id,
                camera_config=cam_cfg,
                output_queue=self.input_queues[cam_id],
                verbose=self.verbose
            )
            self.camera_readers.append(reader)

        # Distribute cameras to inference services (2 cameras per service)
        camera_ids = [cfg['id'] for cfg in self.camera_configs]

        for service_id in range(num_inference_services):
            # Get camera IDs for this service
            start_idx = service_id * 2
            end_idx = min(start_idx + 2, num_cameras)
            service_camera_ids = camera_ids[start_idx:end_idx]

            # Create input/output queue subsets for this service
            service_input_queues = {cid: self.input_queues[cid] for cid in service_camera_ids}
            service_output_queues = {cid: self.output_queues[cid] for cid in service_camera_ids}

            # Create inference service
            service = InferenceService(
                config=self.config,
                camera_ids=service_camera_ids,
                input_queues=service_input_queues,
                output_queues=service_output_queues,
                event_logger=self.event_logger,
                service_id=service_id
            )
            self.inference_services.append(service)

    def _init_resources_batch(self):
        """Initialize resources for batch mode - single BatchInferenceService for all cameras"""

        num_cameras = len(self.camera_configs)

        print(f"[SharedPipeline] Allocating resources (BATCH MODE):")
        print(f"  - Cameras: {num_cameras}")
        print(f"  - Batch inference service: 1 (handles all cameras)")
        print(f"  - Total threads: {num_cameras + 1}")
        print(f"  - Estimated GPU memory: ~1.5 GB")

        # Initialize event logger
        self.event_logger = EventLogger(self.config)
        self.database = self.event_logger.db

        # Create queues for each camera
        queue_size = self.config.get('inference', {}).get('max_buffer_size', 2)
        for cam_cfg in self.camera_configs:
            cam_id = cam_cfg['id']
            self.input_queues[cam_id] = Queue(maxsize=queue_size)
            self.output_queues[cam_id] = Queue(maxsize=queue_size)

        # Create camera readers
        for cam_cfg in self.camera_configs:
            cam_id = cam_cfg['id']
            reader = CameraReader(
                camera_id=cam_id,
                camera_config=cam_cfg,
                output_queue=self.input_queues[cam_id],
                verbose=self.verbose
            )
            self.camera_readers.append(reader)

        # Create single BatchInferenceService for ALL cameras
        camera_ids = [cfg['id'] for cfg in self.camera_configs]

        batch_service = BatchInferenceService(
            config=self.config,
            camera_ids=camera_ids,
            input_queues=self.input_queues,
            output_queues=self.output_queues,
            event_logger=self.event_logger,
            service_id=0
        )
        self.inference_services.append(batch_service)

    def _print_allocation_info(self):
        """Print resource allocation information"""

        print(f"\n{'='*60}")
        if self.batch_mode:
            print(f"  Shared Pipeline Resource Allocation (BATCH MODE)")
        else:
            print(f"  Shared Pipeline Resource Allocation")
        print(f"{'='*60}")

        for i, service in enumerate(self.inference_services):
            cams = ', '.join([f"Camera-{cid}" for cid in service.camera_ids])
            service_type = "BatchInferenceService" if self.batch_mode else "InferenceService"
            print(f"  {service_type} {i}: {cams}")

        print(f"{'='*60}\n")

    def run(self):
        """Start all threads and run visualization loop"""

        print("[SharedPipeline] Starting all threads...")

        # Start camera reader threads
        for reader in self.camera_readers:
            thread = Thread(target=reader.run, name=f"CameraReader-{reader.camera_id}")
            thread.daemon = True
            thread.start()
            self.camera_threads.append(thread)

        # Start inference service threads
        for service in self.inference_services:
            thread = Thread(target=service.run, name=f"InferenceService-{service.service_id}")
            thread.daemon = True
            thread.start()
            self.inference_threads.append(thread)

        print("[SharedPipeline] All threads started")

        # Wait for first frames
        time.sleep(0.5)

        # Run visualization loop
        self._visualization_loop()

    def _visualization_loop(self):
        """Main visualization loop - collects results and displays"""

        print("[SharedPipeline] Starting visualization loop...")

        # Get screen size for fullscreen
        try:
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except:
            screen_width, screen_height = 1920, 1080

        print(f"[SharedPipeline] Screen size: {screen_width}x{screen_height}")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        print("[SharedPipeline] Window created, waiting for frames...")

        last_results: Dict[int, ResultPacket] = {}
        fps_counter = 0
        fps_time = time.time()
        display_fps = 0.0
        wait_log_time = time.time()
        first_frame_received = False

        while self.running.is_set():
            # Collect latest results from all cameras
            for cam_id in self.output_queues:
                try:
                    result = self.output_queues[cam_id].get_nowait()
                    last_results[cam_id] = result
                    if not first_frame_received:
                        print(f"[SharedPipeline] First frame received from camera {cam_id}")
                        first_frame_received = True
                except:
                    pass

            # Check if we have any results
            if not last_results:
                # Log waiting status every 2 seconds
                current_time = time.time()
                if current_time - wait_log_time >= 2.0:
                    print("[SharedPipeline] Waiting for inference results...")
                    wait_log_time = current_time
                time.sleep(0.01)
                continue

            # Build visualization
            frames = []
            for cam_id in sorted(last_results.keys()):
                result = last_results[cam_id]
                frame = self._draw_result(result)
                frames.append(frame)

            # Combine frames
            if len(frames) == 1:
                combined = frames[0]
            elif len(frames) == 2:
                # Side by side
                combined = self._combine_frames_horizontal(frames)
            elif len(frames) <= 4:
                # 2x2 grid
                combined = self._combine_frames_grid(frames, 2, 2)
            else:
                # Dynamic grid
                cols = math.ceil(math.sqrt(len(frames)))
                rows = math.ceil(len(frames) / cols)
                combined = self._combine_frames_grid(frames, rows, cols)

            # Scale to fit screen if needed
            h, w = combined.shape[:2]
            if w > screen_width or h > screen_height:
                scale = min(screen_width / w, screen_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                combined = cv2.resize(combined, (new_w, new_h))

            # Add FPS overlay
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_time >= 1.0:
                display_fps = fps_counter / (current_time - fps_time)
                fps_counter = 0
                fps_time = current_time

            cv2.putText(
                combined,
                f"Display FPS: {display_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # Show
            cv2.imshow(self.window_name, combined)

            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[SharedPipeline] Quit requested")
                break
            elif key == ord('f'):
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        self.stop()

    def _draw_result(self, result: ResultPacket) -> np.ndarray:
        """Draw inference result on frame"""

        frame = result.frame.copy()

        # Color map for different persons
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 165, 255),  # Orange
        ]

        # Draw each person (bbox + keypoints + label)
        for i, state_info in enumerate(result.states):
            bbox = state_info.get('bbox')
            state = state_info.get('state', 'unknown')
            person_id = state_info.get('person_id', 0)
            keypoints = state_info.get('keypoints')

            # Get color for this person
            color = colors[person_id % len(colors)]

            if bbox is not None:
                if hasattr(bbox, '__iter__') and len(bbox) >= 4:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                    # Draw bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    state_str = state.value if hasattr(state, 'value') else str(state)
                    label = f"P{person_id}: {state_str}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw keypoints for this person
            if keypoints is not None:
                self._draw_keypoints(frame, keypoints, color)

        # Draw camera info
        cam_label = f"Camera {result.camera_id} | Persons: {result.person_count}"
        cv2.putText(frame, cam_label, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, color: tuple = (0, 255, 0)):
        """Draw pose keypoints on frame"""

        if keypoints is None or len(keypoints) == 0:
            return

        # COCO skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        # Draw keypoints (use brighter version of color)
        kp_color = tuple(min(255, int(c * 1.3)) for c in color)

        for i, kp in enumerate(keypoints):
            if len(kp) >= 2:
                x, y = int(kp[0]), int(kp[1])
                conf = kp[2] if len(kp) > 2 else 1.0

                if conf > 0.3 and x > 0 and y > 0:
                    cv2.circle(frame, (x, y), 4, kp_color, -1)

        # Draw skeleton
        for i, j in skeleton:
            if i < len(keypoints) and j < len(keypoints):
                kp1, kp2 = keypoints[i], keypoints[j]
                if len(kp1) >= 2 and len(kp2) >= 2:
                    x1, y1 = int(kp1[0]), int(kp1[1])
                    x2, y2 = int(kp2[0]), int(kp2[1])
                    conf1 = kp1[2] if len(kp1) > 2 else 1.0
                    conf2 = kp2[2] if len(kp2) > 2 else 1.0

                    if conf1 > 0.3 and conf2 > 0.3 and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    def _combine_frames_horizontal(self, frames: List[np.ndarray]) -> np.ndarray:
        """Combine frames horizontally"""

        # Resize to same height
        max_height = max(f.shape[0] for f in frames)
        resized = []
        for f in frames:
            if f.shape[0] != max_height:
                scale = max_height / f.shape[0]
                new_width = int(f.shape[1] * scale)
                f = cv2.resize(f, (new_width, max_height))
            resized.append(f)

        return np.hstack(resized)

    def _combine_frames_grid(self, frames: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
        """Combine frames in a grid layout"""

        # Calculate cell size
        max_height = max(f.shape[0] for f in frames)
        max_width = max(f.shape[1] for f in frames)

        # Create grid
        grid_height = max_height * rows
        grid_width = max_width * cols
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for idx, frame in enumerate(frames):
            row = idx // cols
            col = idx % cols

            # Resize frame to cell size
            resized = cv2.resize(frame, (max_width, max_height))

            # Place in grid
            y1 = row * max_height
            y2 = y1 + max_height
            x1 = col * max_width
            x2 = x1 + max_width

            grid[y1:y2, x1:x2] = resized

        return grid

    def stop(self):
        """Stop all threads and cleanup"""

        print("[SharedPipeline] Stopping...")
        self.running.clear()

        # Stop camera readers
        for reader in self.camera_readers:
            reader.stop()

        # Stop inference services (includes GPU cleanup)
        for service in self.inference_services:
            service.stop()

        # Final CUDA cache cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[SharedPipeline] CUDA cache cleared")
        except Exception as e:
            print(f"[SharedPipeline] CUDA cleanup warning: {e}")

        # Close visualization
        cv2.destroyAllWindows()

        # Close database
        if self.database:
            self.database.close()

        print("[SharedPipeline] All resources released")


def main():
    """Entry point for shared pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Life Tracker - Shared Pipeline')
    parser.add_argument('--config', type=str, default='config/config_gpu.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    manager = SharedPipelineManager(args.config)
    manager.run()


if __name__ == '__main__':
    main()
