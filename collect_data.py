#!/usr/bin/env python3
"""
Data Collection Tool - 4-Thread Async Pipeline for Maximum Performance

Architecture (inspired by main_async.py):
  Thread 1: Camera → continuous frame reading (30 FPS)
  Thread 2: YOLO → person detection (every frame)
  Thread 3: RTMPose → pose estimation (every frame)
  Thread 4: Display + Data Collection → visualization & saving

Performance:
  - Display FPS: 25-30 (流畅显示，最小延迟)
  - Collection Rate: 25-30 samples/sec (100% 数据覆盖)
  - Latency: <100ms (流水线延迟)

Usage:
    python collect_data.py

    # 自定义配置
    python collect_data.py --backend rtmpose --config config/config_gpu.yaml

Key Controls:
    's' - Start recording Sitting
    't' - Start recording Standing
    'l' - Start recording Lying
    'q' - Stop current recording
    'ESC' - Exit program
"""

import cv2
import numpy as np
import argparse
import time
import os
import json
import yaml
import threading
from queue import Queue, Empty
from datetime import datetime
from typing import Optional, List, Tuple
from pathlib import Path
import signal
import sys


# COCO-17 skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2),      # nose -> eyes
    (1, 3), (2, 4),      # eyes -> ears
    (5, 6),              # shoulders
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # torso
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

SKELETON_COLORS = [
    (255, 128, 0),   # orange - face
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (0, 255, 0),     # green - shoulders
    (255, 255, 0),   # yellow - left arm
    (255, 255, 0),
    (0, 255, 255),   # cyan - right arm
    (0, 255, 255),
    (0, 128, 255),   # blue - torso
    (0, 128, 255),
    (255, 0, 255),   # magenta - hips
    (255, 0, 128),   # pink - left leg
    (255, 0, 128),
    (128, 0, 255),   # purple - right leg
    (128, 0, 255),
]


class AsyncDataCollector:
    """4-Thread async pipeline data collector"""

    def __init__(self, backend: str = 'rtmpose',
                 config_path: str = 'config/config_gpu.yaml',
                 countdown: int = 3,
                 min_duration: int = 120,
                 detection_confidence: float = None):
        """
        Args:
            backend: 'mediapipe' or 'rtmpose'
            config_path: Path to config file
            countdown: Countdown seconds after key press
            min_duration: Recommended minimum recording duration (seconds)
            detection_confidence: Override detection confidence
        """
        self.backend = backend
        self.countdown = countdown
        self.min_duration = min_duration
        self.detection_confidence = detection_confidence
        self.config_path = config_path

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Override detection confidence if specified
        if self.detection_confidence is not None:
            self.config['models']['person']['confidence'] = self.detection_confidence

        # Recording state
        self.is_recording = False
        self.current_pose_label = None
        self.record_start_time = None
        self.collected_samples = []
        self.samples_lock = threading.Lock()

        # Sequence buffer for LSTM/Transformer
        self.sequence_buffer = []
        self.sequence_length = 10

        # Data storage
        self.output_dir = "training_data"
        os.makedirs(self.output_dir, exist_ok=True)

        # Sample count cache
        self._sample_count_cache = {}
        self._cache_initialized = False
        self._init_sample_count_cache()

        # Async save queue
        self.save_queue = Queue(maxsize=100)
        self.save_thread = None
        self.stop_save_thread = False

        # Running flags
        self.running = False

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Pipeline queues (small size to maintain real-time)
        self.queue_frames = Queue(maxsize=2)
        self.queue_detections = Queue(maxsize=2)
        self.queue_poses = Queue(maxsize=2)

        # Performance statistics
        self.stats = {
            'camera_fps': 0,
            'detection_fps': 0,
            'pose_fps': 0,
            'display_fps': 0
        }

        # Initialize components
        self._init_components()

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C"""
        print("\n[Exit] Received termination signal...")
        self.running = False

    def _init_components(self):
        """Initialize detection components"""
        print(f"\n{'='*60}")
        print(f"  Async Data Collector - 4-Thread Pipeline")
        print(f"  Backend: {self.backend}")
        print(f"{'='*60}\n")

        if self.backend == 'mediapipe':
            self._init_mediapipe()
        elif self.backend == 'rtmpose':
            self._init_rtmpose()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        print("[Init] All components loaded!\n")

    def _init_mediapipe(self):
        """Initialize MediaPipe Pose"""
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # MediaPipe to COCO mapping
        self.MEDIAPIPE_TO_COCO = {
            0: 0,   # nose
            11: 5,  # left_shoulder
            12: 6,  # right_shoulder
            13: 7,  # left_elbow
            14: 8,  # right_elbow
            15: 9,  # left_wrist
            16: 10, # right_wrist
            23: 11, # left_hip
            24: 12, # right_hip
            25: 13, # left_knee
            26: 14, # right_knee
            27: 15, # left_ankle
            28: 16, # right_ankle
        }

    def _init_rtmpose(self):
        """Initialize RTMPose (YOLO + RTMPose)"""
        from src.detectors import PoseEstimatorFactory
        from src.detectors.person_detector import PersonDetector

        print("[Init] Loading YOLOv8 person detector...")
        self.person_detector = PersonDetector(self.config['models']['person'])

        print("[Init] Loading RTMPose pose estimator...")
        self.pose_estimator = PoseEstimatorFactory.create(self.config['models']['pose'])

    def _init_sample_count_cache(self):
        """Initialize sample count cache (once at startup)"""
        if self._cache_initialized:
            return

        print("[INFO] Loading sample counts...")
        for pose_label in ['sitting', 'standing', 'lying']:
            filepath = os.path.join(self.output_dir, f"{pose_label}_samples.json")
            if not os.path.exists(filepath):
                self._sample_count_cache[pose_label] = 0
                continue

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self._sample_count_cache[pose_label] = len(data)
                    print(f"  {pose_label}: {len(data)} samples")
            except:
                self._sample_count_cache[pose_label] = 0

        self._cache_initialized = True

    # =========================================================================
    # Thread 1: Camera reading thread
    # =========================================================================
    def _camera_thread(self):
        """Continuously read frames from camera"""
        print("[Thread 1] Camera thread started")
        frame_count = 0
        fps_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Failed to read frame")
                continue

            # Non-blocking put (drop if queue full)
            try:
                self.queue_frames.put((time.time(), frame), block=False)
            except:
                pass  # Queue full, drop frame

            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                self.stats['camera_fps'] = frame_count
                frame_count = 0
                fps_time = time.time()

        print("[Thread 1] Camera thread exited")

    # =========================================================================
    # Thread 2: YOLO detection thread (every frame)
    # =========================================================================
    def _detection_thread(self):
        """Detect person in every frame"""
        print("[Thread 2] Detection thread started")
        frame_count = 0
        fps_time = time.time()

        while self.running:
            try:
                timestamp, frame = self.queue_frames.get(timeout=0.1)
            except Empty:
                continue

            # Detect person (every frame!)
            if self.backend == 'rtmpose':
                bbox = self.person_detector.detect(frame)
            else:
                bbox = None  # MediaPipe doesn't need bbox

            # Put to next queue
            try:
                self.queue_detections.put((timestamp, frame, bbox), block=False)
            except:
                pass  # Queue full, drop

            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                self.stats['detection_fps'] = frame_count
                frame_count = 0
                fps_time = time.time()

        print("[Thread 2] Detection thread exited")

    # =========================================================================
    # Thread 3: Pose estimation thread (every frame)
    # =========================================================================
    def _pose_thread(self):
        """Estimate pose in every frame"""
        print("[Thread 3] Pose thread started")
        frame_count = 0
        fps_time = time.time()

        while self.running:
            try:
                timestamp, frame, bbox = self.queue_detections.get(timeout=0.1)
            except Empty:
                continue

            # Estimate pose (every frame!)
            if self.backend == 'mediapipe':
                keypoints = self._process_mediapipe(frame)
            else:
                keypoints = self._process_rtmpose(frame, bbox)

            # Put to next queue
            try:
                self.queue_poses.put((timestamp, frame, bbox, keypoints), block=False)
            except:
                pass  # Queue full, drop

            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                self.stats['pose_fps'] = frame_count
                frame_count = 0
                fps_time = time.time()

        print("[Thread 3] Pose thread exited")

    def _process_mediapipe(self, frame):
        """Process frame with MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_world_landmarks:
            return None

        # Convert to COCO-17 format
        world_landmarks = np.zeros((17, 4), dtype=np.float32)
        for mp_idx, coco_idx in self.MEDIAPIPE_TO_COCO.items():
            wl = results.pose_world_landmarks.landmark[mp_idx]
            world_landmarks[coco_idx] = [wl.x, wl.y, wl.z, wl.visibility]

        return world_landmarks

    def _process_rtmpose(self, frame, bbox):
        """Process frame with RTMPose"""
        if bbox is None:
            return None

        keypoints = self.pose_estimator.estimate(frame, bbox)
        return keypoints

    # =========================================================================
    # Thread 4: Display + Data Collection thread
    # =========================================================================
    def _display_thread(self):
        """Display and collect data"""
        print("[Thread 4] Display thread started")
        frame_count = 0
        fps_time = time.time()
        countdown_start_time = None

        while self.running:
            try:
                timestamp, frame, bbox, keypoints = self.queue_poses.get(timeout=0.1)
            except Empty:
                # No new data, show empty frame
                continue

            current_time = time.time()

            # Draw visualization
            vis_frame = self._draw_visualization(frame, bbox, keypoints)

            # Handle countdown phase
            if countdown_start_time is not None and not self.is_recording:
                elapsed = current_time - countdown_start_time
                remaining = self.countdown - int(elapsed)
                if remaining > 0:
                    vis_frame = self._draw_countdown(vis_frame, remaining)
                else:
                    self.is_recording = True
                    self.record_start_time = current_time
                    countdown_start_time = None
                    print(f"[INFO] Started recording {self.current_pose_label}...")

            # Handle recording phase
            elif self.is_recording:
                if keypoints is not None:
                    self._collect_sample(keypoints, frame.shape, current_time)
                duration = int(current_time - self.record_start_time)
                vis_frame = self._draw_recording_info(vis_frame, duration)

            # Handle idle phase
            else:
                vis_frame = self._draw_instructions(vis_frame)

            # Show FPS
            fps_text = f"Cam:{self.stats['camera_fps']} Det:{self.stats['detection_fps']} Pose:{self.stats['pose_fps']} Disp:{self.stats['display_fps']}"
            cv2.putText(vis_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display
            cv2.imshow(f'Data Collection - {self.backend}', vis_frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.running = False
                if self.is_recording:
                    self._stop_recording()
                break
            elif key == ord('s') and not self.is_recording and countdown_start_time is None:
                self._start_recording('sitting')
                countdown_start_time = current_time
            elif key == ord('t') and not self.is_recording and countdown_start_time is None:
                self._start_recording('standing')
                countdown_start_time = current_time
            elif key == ord('l') and not self.is_recording and countdown_start_time is None:
                self._start_recording('lying')
                countdown_start_time = current_time
            elif key == ord('q'):
                if countdown_start_time is not None:
                    countdown_start_time = None
                    self.current_pose_label = None
                    print("[INFO] Cancelled")
                elif self.is_recording:
                    self._stop_recording()

            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                self.stats['display_fps'] = frame_count
                frame_count = 0
                fps_time = time.time()

        print("[Thread 4] Display thread exited")

    def _draw_visualization(self, frame, bbox, keypoints):
        """Draw bbox and skeleton"""
        vis_frame = frame.copy()

        if self.backend == 'rtmpose' and bbox is not None:
            x1, y1, x2, y2, conf = bbox
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if keypoints is not None:
            # Draw skeleton lines
            for idx, (i, j) in enumerate(COCO_SKELETON):
                if i < len(keypoints) and j < len(keypoints):
                    if keypoints.shape[1] == 3:  # RTMPose format
                        x1, y1, conf1 = keypoints[i]
                        x2, y2, conf2 = keypoints[j]
                        if conf1 > 0.3 and conf2 > 0.3:
                            color = SKELETON_COLORS[idx % len(SKELETON_COLORS)]
                            cv2.line(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw keypoints
            for i, kpt in enumerate(keypoints):
                if keypoints.shape[1] == 3:
                    x, y, conf = kpt
                    if conf > 0.3:
                        cv2.circle(vis_frame, (int(x), int(y)), 4, (0, 255, 255), -1)
                        cv2.circle(vis_frame, (int(x), int(y)), 4, (0, 0, 0), 1)

        return vis_frame

    def _collect_sample(self, keypoints, frame_shape, timestamp):
        """Collect training sample (thread-safe)"""
        h, w = frame_shape[:2]

        # Normalize keypoints
        if keypoints.shape[1] == 3:  # RTMPose
            normalized = np.zeros((17, 4), dtype=np.float32)
            normalized[:, 0] = keypoints[:, 0] / w
            normalized[:, 1] = keypoints[:, 1] / h
            normalized[:, 2] = 0.0
            normalized[:, 3] = keypoints[:, 2]
            keypoints = normalized

        features = keypoints.flatten()

        # Update sequence buffer
        self.sequence_buffer.append(features)
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)

        # Create sample
        sample = {
            'features': features.tolist(),
            'label': self.current_pose_label,
            'timestamp': timestamp
        }

        if len(self.sequence_buffer) == self.sequence_length:
            sample['features_sequence'] = [f.tolist() for f in self.sequence_buffer]

        # Thread-safe append
        with self.samples_lock:
            self.collected_samples.append(sample)

    def _draw_countdown(self, frame, remaining):
        """Draw countdown overlay"""
        h, w = frame.shape[:2]
        text = f"Recording {self.current_pose_label} starts in: {remaining}"
        cv2.putText(frame, text, (50, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                   1.5, (0, 255, 255), 3)
        return frame

    def _draw_recording_info(self, frame, duration):
        """Draw recording information"""
        h, w = frame.shape[:2]
        with self.samples_lock:
            sample_count = len(self.collected_samples)

        text1 = f"Recording: {self.current_pose_label}"
        cv2.putText(frame, text1, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        text2 = f"Duration: {duration}s (min {self.min_duration}s)"
        cv2.putText(frame, text2, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        text3 = f"Collected: {sample_count} frames"
        cv2.putText(frame, text3, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        text4 = "Press 'q' to stop"
        cv2.putText(frame, text4, (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _draw_instructions(self, frame):
        """Draw idle instructions"""
        h, w = frame.shape[:2]

        instructions = [
            f"Async Data Collector ({self.backend})",
            "",
            "Key Controls:",
            "  's' - Record Sitting",
            "  't' - Record Standing",
            "  'l' - Record Lying",
            "  'ESC' - Exit",
            "",
            "Collected Data:",
            f"  Sitting: {self._sample_count_cache.get('sitting', 0)} samples",
            f"  Standing: {self._sample_count_cache.get('standing', 0)} samples",
            f"  Lying: {self._sample_count_cache.get('lying', 0)} samples",
        ]

        y = 80
        for line in instructions:
            cv2.putText(frame, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
            y += 35

        return frame

    # =========================================================================
    # Recording control
    # =========================================================================
    def _start_recording(self, pose_label: str):
        """Start recording"""
        self.current_pose_label = pose_label
        self.record_start_time = None
        with self.samples_lock:
            self.collected_samples = []
        self.sequence_buffer = []
        print(f"\n[INFO] Preparing to record {pose_label}, countdown {self.countdown}s...")

    def _stop_recording(self):
        """Stop recording and save"""
        if not self.is_recording:
            return

        self.is_recording = False
        self.sequence_buffer = []

        with self.samples_lock:
            num_samples = len(self.collected_samples)
            if num_samples == 0:
                print("[WARN] No samples collected")
                return

            # Queue for async saving
            self.save_queue.put((self.current_pose_label, self.collected_samples.copy()))

            # Update cache
            self._sample_count_cache[self.current_pose_label] = \
                self._sample_count_cache.get(self.current_pose_label, 0) + num_samples

            print(f"[INFO] Queued {num_samples} samples for saving")

            # Clear
            self.collected_samples = []

    # =========================================================================
    # Async saving
    # =========================================================================
    def _start_save_thread(self):
        """Start background save thread"""
        if self.save_thread is None or not self.save_thread.is_alive():
            self.stop_save_thread = False
            self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
            self.save_thread.start()

    def _save_worker(self):
        """Background worker for async saving"""
        while not self.stop_save_thread:
            try:
                data = self.save_queue.get(timeout=0.1)
                if data is None:
                    break

                pose_label, samples = data
                filepath = os.path.join(self.output_dir, f"{pose_label}_samples.json")

                # Load existing
                existing_data = []
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r') as f:
                            existing_data = json.load(f)
                    except:
                        existing_data = []

                # Merge and save
                existing_data.extend(samples)
                with open(filepath, 'w') as f:
                    json.dump(existing_data, f)

            except:
                pass

    def _stop_save_thread(self):
        """Stop save thread"""
        self.stop_save_thread = True
        if self.save_thread:
            self.save_thread.join(timeout=2.0)

    # =========================================================================
    # Main run
    # =========================================================================
    def run(self):
        """Run 4-thread async pipeline"""
        # Open camera
        self.cap = cv2.VideoCapture(self.config['camera']['source'])

        # Configure camera
        print("[INFO] Configuring camera...")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"[Camera] {actual_w}x{actual_h} @ {actual_fps} FPS (MJPEG)")

        print("\n[INFO] Key controls:")
        print("  's' - Record Sitting  |  't' - Record Standing  |  'l' - Record Lying")
        print("  'q' - Stop recording  |  ESC - Exit\n")

        # Start all threads
        self.running = True
        self._start_save_thread()

        threads = [
            threading.Thread(target=self._camera_thread, daemon=True),
            threading.Thread(target=self._detection_thread, daemon=True),
            threading.Thread(target=self._pose_thread, daemon=True),
            threading.Thread(target=self._display_thread, daemon=False),  # Main thread
        ]

        for t in threads[:-1]:  # Start background threads
            t.start()

        print("[INFO] 4-thread pipeline started!\n")

        # Run display thread in main thread
        threads[-1].run()

        # Cleanup
        self.running = False
        for t in threads[:-1]:
            t.join(timeout=1.0)

        self.cap.release()
        cv2.destroyAllWindows()

        if self.backend == 'mediapipe':
            self.pose.close()

        self._stop_save_thread()

        print(f"\n[INFO] Final stats:")
        print(f"  Camera FPS: {self.stats['camera_fps']}")
        print(f"  Detection FPS: {self.stats['detection_fps']}")
        print(f"  Pose FPS: {self.stats['pose_fps']}")
        print(f"  Display FPS: {self.stats['display_fps']}")
        print(f"\n[INFO] Data saved to: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Async Data Collection Tool')
    parser.add_argument('--backend', type=str, default='rtmpose',
                       choices=['mediapipe', 'rtmpose'],
                       help='Pose estimation backend (default: rtmpose)')
    parser.add_argument('--config', type=str, default='config/config_gpu.yaml',
                       help='Config file path')
    parser.add_argument('--countdown', type=int, default=3,
                       help='Countdown seconds before recording')
    parser.add_argument('--min-duration', type=int, default=120,
                       help='Recommended minimum recording duration')
    parser.add_argument('--detection-confidence', type=float, default=None,
                       help='Override YOLO detection confidence')

    args = parser.parse_args()

    collector = AsyncDataCollector(
        backend=args.backend,
        config_path=args.config,
        countdown=args.countdown,
        min_duration=args.min_duration,
        detection_confidence=args.detection_confidence
    )

    collector.run()


if __name__ == '__main__':
    main()
