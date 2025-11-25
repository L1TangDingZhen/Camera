#!/usr/bin/env python3
"""
Life Tracker - Async Pipeline Version
4-Thread asynchronous pipeline architecture for maximum performance

Architecture:
  Thread 1: Camera → continuously read frames
  Thread 2: YOLO → detect person every N frames
  Thread 3: RTMPose → estimate pose
  Thread 4: State Machine + Visualization → update & display
"""

import argparse
import time
import yaml
import cv2
import numpy as np
from pathlib import Path
from threading import Thread
from queue import Queue, Empty
import signal
import sys

from src.detectors import PersonDetector, PoseEstimatorFactory
from src.state import BehaviorStateMachine, ROIManager
from src.storage import EventLogger


class AsyncLifeTracker:
    """Life Tracker with asynchronous pipeline"""

    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        print(f"\n{'='*60}")
        print(f"  Life Tracker - Async Pipeline")
        print(f"  Device: {self.config['device']}")
        print(f"{'='*60}\n")

        # Running flag
        self.running = True

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Initialize components
        self._init_components()

        # Create queues (limit size to avoid memory bloat)
        self.queue_frames = Queue(maxsize=2)       # Camera → YOLO
        self.queue_bboxes = Queue(maxsize=2)       # YOLO → RTMPose
        self.queue_keypoints = Queue(maxsize=2)    # RTMPose → StateMachine

        # Performance statistics
        self.stats = {
            'camera_fps': 0,
            'detection_fps': 0,
            'pose_fps': 0,
            'display_fps': 0
        }

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C signal"""
        print("\n[Exit] Received termination signal...")
        self.running = False

    def _init_components(self):
        """Initialize all components"""
        # 1. Create detectors
        print("[Init] Loading person detector...")
        self.person_detector = PersonDetector(self.config['models']['person'])

        print("[Init] Loading pose estimator...")
        self.pose_estimator = PoseEstimatorFactory.create(self.config['models']['pose'])

        # 2. Create ROI manager
        print("[Init] Loading ROI manager...")
        self.roi_manager = ROIManager(self.config.get('roi', {}))

        # 3. Create event logger
        print("[Init] Creating event logger...")
        self.event_logger = EventLogger(self.config)

        # 4. Create state machine
        print("[Init] Creating state machine...")
        self.state_machine = BehaviorStateMachine(
            self.config, self.roi_manager, database=self.event_logger.db
        )

        # 5. Initialize camera
        print("[Init] Opening camera...")
        camera_config = self.config['camera']
        self.cap = cv2.VideoCapture(camera_config['source'])

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {camera_config['source']}")

        # Set camera parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['resolution'][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['resolution'][1])
        self.cap.set(cv2.CAP_PROP_FPS, camera_config['fps'])

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"[Camera] Device: /dev/video{camera_config['source']}")
        print(f"[Camera] Actual resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")

        # Visualization config
        self.show_visualization = not self.config.get('no_visualization', False)
        self.show_keypoints = self.config.get('debug', {}).get('show_keypoints', False)
        self.show_skeleton = self.config.get('debug', {}).get('show_skeleton', False)
        self.show_state_info = self.config.get('debug', {}).get('show_state_info', False)

        print("[Init] All components loaded!\n")

    # =====================================================================
    # Thread 1: Camera reading thread
    # =====================================================================
    def _camera_thread(self):
        """Continuously read frames from camera and put into queue"""
        print("[Thread 1] Camera thread started")
        frame_count = 0
        fps_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[Camera] Failed to read frame")
                break

            # Non-blocking put (drop frame if queue is full to maintain real-time)
            try:
                self.queue_frames.put((time.time(), frame), block=False)
            except:
                # Queue full, skip this frame to maintain real-time performance
                pass

            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                self.stats['camera_fps'] = frame_count
                frame_count = 0
                fps_time = time.time()

        print("[Thread 1] Camera thread exited")

    # =====================================================================
    # Thread 2: YOLO detection thread (with detailed profiling)
    # =====================================================================
    def _detection_thread(self):
        """Detect person every N frames, reuse cached bbox for intermediate frames"""
        print("[Thread 2] YOLO detection thread started")
        detection_interval = self.config['inference'].get('detection_interval', 3)
        frame_count = 0
        cached_bbox = None
        fps_time = time.time()

        # Detailed timing
        timings = {
            'queue_get': [],
            'detect': [],
            'queue_put': [],
            'total': []
        }

        while self.running:
            try:
                t_start = time.time()

                # Get frame from queue (blocking, wait up to 0.1s)
                t0 = time.time()
                timestamp, frame = self.queue_frames.get(timeout=0.1)
                t1 = time.time()
                timings['queue_get'].append((t1 - t0) * 1000)

                frame_count += 1

                # Detect every N frames
                if frame_count % detection_interval == 0:
                    t0 = time.time()
                    bbox = self.person_detector.detect(frame)
                    t1 = time.time()
                    timings['detect'].append((t1 - t0) * 1000)
                    cached_bbox = bbox
                else:
                    # Reuse cached bbox for intermediate frames
                    bbox = cached_bbox

                # Put to next queue
                t0 = time.time()
                try:
                    self.queue_bboxes.put((timestamp, frame, bbox), block=False)
                except:
                    pass  # Queue full, drop frame
                t1 = time.time()
                timings['queue_put'].append((t1 - t0) * 1000)

                t_end = time.time()
                timings['total'].append((t_end - t_start) * 1000)

                # Calculate FPS and print detailed stats every second
                if time.time() - fps_time >= 1.0:
                    self.stats['detection_fps'] = frame_count

                    # Print detailed timing stats
                    if len(timings['detect']) > 0:
                        avg_detect = np.mean(timings['detect'])
                        max_detect = np.max(timings['detect'])
                        avg_total = np.mean(timings['total'])

                        print(f"\n[YOLO Thread] Avg: detect={avg_detect:.1f}ms "
                              f"(max={max_detect:.1f}ms), total={avg_total:.1f}ms")

                        # Clear timings
                        for key in timings:
                            timings[key].clear()

                    frame_count = 0
                    fps_time = time.time()

            except Empty:
                continue

        print("[Thread 2] YOLO detection thread exited")

    # =====================================================================
    # Thread 3: RTMPose pose estimation thread (with detailed profiling)
    # =====================================================================
    def _pose_thread(self):
        """Estimate pose from detected person bbox"""
        print("[Thread 3] RTMPose pose thread started")
        frame_count = 0
        fps_time = time.time()

        # Detailed timing
        timings = {
            'queue_get': [],
            'estimate': [],
            'queue_put': [],
            'total': []
        }

        while self.running:
            try:
                t_start = time.time()

                # Get data from queue
                t0 = time.time()
                timestamp, frame, bbox = self.queue_bboxes.get(timeout=0.1)
                t1 = time.time()
                timings['queue_get'].append((t1 - t0) * 1000)

                frame_count += 1

                # Estimate pose
                keypoints = None
                world_landmarks = None
                if bbox is not None:
                    t0 = time.time()
                    keypoints = self.pose_estimator.estimate(frame, bbox)
                    t1 = time.time()
                    timings['estimate'].append((t1 - t0) * 1000)

                    if hasattr(self.pose_estimator, 'get_world_landmarks'):
                        world_landmarks = self.pose_estimator.get_world_landmarks()

                # Put to next queue
                t0 = time.time()
                try:
                    self.queue_keypoints.put(
                        (timestamp, frame, bbox, keypoints, world_landmarks),
                        block=False
                    )
                except:
                    pass  # Queue full, drop frame
                t1 = time.time()
                timings['queue_put'].append((t1 - t0) * 1000)

                t_end = time.time()
                timings['total'].append((t_end - t_start) * 1000)

                # Calculate FPS and print detailed stats every second
                if time.time() - fps_time >= 1.0:
                    self.stats['pose_fps'] = frame_count

                    # Print detailed timing stats
                    if len(timings['estimate']) > 0:
                        avg_estimate = np.mean(timings['estimate'])
                        max_estimate = np.max(timings['estimate'])
                        avg_total = np.mean(timings['total'])

                        print(f"\n[Pose Thread] Avg: estimate={avg_estimate:.1f}ms "
                              f"(max={max_estimate:.1f}ms), total={avg_total:.1f}ms")

                        # Clear timings
                        for key in timings:
                            timings[key].clear()

                    frame_count = 0
                    fps_time = time.time()

            except Empty:
                continue

        print("[Thread 3] RTMPose pose thread exited")

    # =====================================================================
    # Thread 4: State machine + visualization thread (main thread)
    # =====================================================================
    def _display_thread(self):
        """Update state machine and display visualization"""
        print("[Thread 4] State machine + display thread started")

        frame_count = 0
        fps_time = time.time()
        fps = 0

        # Detailed timing
        timings = {
            'queue_get': [],
            'state_machine': [],
            'visualize': [],
            'imshow': [],
            'waitkey': [],
            'total': []
        }

        # Create window
        if self.show_visualization:
            cv2.namedWindow('Life Tracker - Async', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Life Tracker - Async',
                           self.config['camera']['resolution'][0],
                           self.config['camera']['resolution'][1])

        while self.running:
            try:
                t_start = time.time()

                # Get data from queue
                t0 = time.time()
                timestamp, frame, bbox, keypoints, world_landmarks = \
                    self.queue_keypoints.get(timeout=0.1)
                t1 = time.time()
                timings['queue_get'].append((t1 - t0) * 1000)

                frame_count += 1
                current_time = time.time()

                # Update state machine
                t0 = time.time()
                events = self.state_machine.update(bbox, keypoints, current_time, world_landmarks)
                t1 = time.time()
                timings['state_machine'].append((t1 - t0) * 1000)

                # Log events
                if events:
                    self.event_logger.log_events(events)

                # Visualization
                if self.show_visualization:
                    t0 = time.time()
                    vis_frame = self._visualize(frame, bbox, keypoints, fps)
                    t1 = time.time()
                    timings['visualize'].append((t1 - t0) * 1000)

                    t0 = time.time()
                    cv2.imshow('Life Tracker - Async', vis_frame)
                    t1 = time.time()
                    timings['imshow'].append((t1 - t0) * 1000)

                    t0 = time.time()
                    key = cv2.waitKey(1) & 0xFF
                    t1 = time.time()
                    timings['waitkey'].append((t1 - t0) * 1000)

                    if key == ord('q'):
                        print("\n[Exit] User pressed 'q'")
                        self.running = False
                        break

                t_end = time.time()
                timings['total'].append((t_end - t_start) * 1000)

                # Calculate FPS and print detailed stats every second
                if current_time - fps_time >= 1.0:
                    fps = frame_count
                    self.stats['display_fps'] = fps

                    # Print performance stats
                    print(f"\r[Perf] Camera:{self.stats['camera_fps']:2d}fps | "
                          f"YOLO:{self.stats['detection_fps']:2d}fps | "
                          f"Pose:{self.stats['pose_fps']:2d}fps | "
                          f"Display:{self.stats['display_fps']:2d}fps",
                          end='', flush=True)

                    # Print detailed timing stats
                    if len(timings['state_machine']) > 0:
                        avg_state = np.mean(timings['state_machine'])
                        avg_vis = np.mean(timings['visualize']) if timings['visualize'] else 0
                        avg_imshow = np.mean(timings['imshow']) if timings['imshow'] else 0
                        avg_waitkey = np.mean(timings['waitkey']) if timings['waitkey'] else 0
                        avg_total = np.mean(timings['total'])

                        print(f"\n[Display Thread] Avg: state={avg_state:.1f}ms, "
                              f"vis={avg_vis:.1f}ms, imshow={avg_imshow:.1f}ms, "
                              f"waitkey={avg_waitkey:.1f}ms, total={avg_total:.1f}ms")

                        # Clear timings
                        for key in timings:
                            timings[key].clear()

                    frame_count = 0
                    fps_time = current_time

            except Empty:
                continue

        print("\n[Thread 4] State machine + display thread exited")

    def _visualize(self, frame, bbox, keypoints, fps):
        """Draw visualization on frame"""
        vis_frame = frame.copy()

        # Draw FPS
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw bbox
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw keypoints
        if keypoints is not None and self.show_keypoints:
            for i, kp in enumerate(keypoints):
                if len(kp) >= 3:
                    x, y, conf = int(kp[0]), int(kp[1]), kp[2]
                    if conf > 0.3:
                        cv2.circle(vis_frame, (x, y), 3, (0, 255, 255), -1)

        # Draw skeleton
        if keypoints is not None and self.show_skeleton:
            self._draw_skeleton(vis_frame, keypoints)

        # Draw state info
        if self.show_state_info:
            state = self.state_machine.current_state
            cv2.putText(vis_frame, f"State: {state}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return vis_frame

    def _draw_skeleton(self, frame, keypoints):
        """Draw skeleton connections"""
        # COCO-17 skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        for i, j in skeleton:
            if i < len(keypoints) and j < len(keypoints):
                pt1 = keypoints[i]
                pt2 = keypoints[j]
                if len(pt1) >= 3 and len(pt2) >= 3:
                    if pt1[2] > 0.3 and pt2[2] > 0.3:
                        x1, y1 = int(pt1[0]), int(pt1[1])
                        x2, y2 = int(pt2[0]), int(pt2[1])
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # =====================================================================
    # Main run method
    # =====================================================================
    def run(self):
        """Start the async pipeline"""
        print("[Run] Starting async 4-thread pipeline...\n")

        # Create 4 threads
        threads = [
            Thread(target=self._camera_thread, name="CameraThread", daemon=True),
            Thread(target=self._detection_thread, name="DetectionThread", daemon=True),
            Thread(target=self._pose_thread, name="PoseThread", daemon=True),
        ]

        # Start threads
        for t in threads:
            t.start()

        # Main thread runs display
        try:
            self._display_thread()
        except KeyboardInterrupt:
            print("\n[Exit] Ctrl+C interrupt")
            self.running = False

        # Wait for all threads to exit
        print("[Run] Waiting for threads to exit...")
        for t in threads:
            t.join(timeout=2.0)

        # Cleanup
        self.cap.release()
        if self.show_visualization:
            cv2.destroyAllWindows()

        print("[Run] Program exited\n")


def main():
    parser = argparse.ArgumentParser(description='Life Tracker - Async Pipeline')
    parser.add_argument('--config', type=str, default='config/config_gpu.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['cpu', 'gpu'], default='gpu',
                       help='Run mode (cpu or gpu)')
    args = parser.parse_args()

    # Select config file based on mode
    if args.mode == 'cpu':
        config_path = 'config/config_cpu.yaml'
    else:
        config_path = args.config

    # Create and run tracker
    tracker = AsyncLifeTracker(config_path)
    tracker.run()


if __name__ == '__main__':
    main()
