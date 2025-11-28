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

        # Detailed timing logs for final summary (controlled by config)
        self.enable_profiling = self.config.get('debug', {}).get('performance_profiling', False)

        if self.enable_profiling:
            self.timing_history = {
                'camera_read': [],
                'detection': [],
                'pose_estimation': [],
                'state_machine': []
            }
        else:
            self.timing_history = None

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
        # Force MJPEG encoding (avoid YUYV bandwidth bottleneck, especially for FHD)
        # MJPEG: ~50-200KB/frame vs YUYV: ~4MB/frame (FHD)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
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

        # Pose estimation interval optimization
        self.pose_interval = self.config.get('inference', {}).get('pose_interval', 1)
        self.pose_frame_counter = 0
        self.cached_keypoints = None
        if self.pose_interval > 1:
            print(f"[Optimization] Pose interval: every {self.pose_interval} frames (reduces RTMPose load by {(1-1/self.pose_interval)*100:.0f}%)")

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
            t0 = time.time()
            ret, frame = self.cap.read()
            t1 = time.time()

            if not ret:
                print("[Camera] Failed to read frame")
                break

            # Record timing
            if self.enable_profiling:
                read_time_ms = (t1 - t0) * 1000
                self.timing_history['camera_read'].append(read_time_ms)

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
                    detect_time = (t1 - t0) * 1000
                    timings['detect'].append(detect_time)
                    if self.enable_profiling:
                        self.timing_history['detection'].append(detect_time)
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

                # Estimate pose with interval optimization
                keypoints = None
                world_landmarks = None
                if bbox is not None:
                    # Increment pose counter only when person is detected
                    self.pose_frame_counter += 1
                    # Check if we should run RTMPose or use cached result
                    should_estimate = (self.pose_frame_counter % self.pose_interval == 1)

                    if should_estimate:
                        # Run RTMPose estimation
                        t0 = time.time()
                        keypoints = self.pose_estimator.estimate(frame, bbox)
                        t1 = time.time()
                        pose_time = (t1 - t0) * 1000
                        timings['estimate'].append(pose_time)
                        if self.enable_profiling:
                            self.timing_history['pose_estimation'].append(pose_time)

                        # Cache the result
                        self.cached_keypoints = keypoints
                    else:
                        # Use cached keypoints
                        keypoints = self.cached_keypoints
                        # Record 0ms for cached frames
                        timings['estimate'].append(0.0)
                        if self.enable_profiling:
                            self.timing_history['pose_estimation'].append(0.0)

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
            # Display window size: 1280x720 for comfortable viewing (regardless of camera resolution)
            display_width = 1280
            display_height = 720
            cv2.resizeWindow('Life Tracker - Async', display_width, display_height)

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
                state_time = (t1 - t0) * 1000
                timings['state_machine'].append(state_time)
                if self.enable_profiling:
                    self.timing_history['state_machine'].append(state_time)

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

        # Draw state info (async original style - thinner font)
        if self.show_state_info:
            y_offset = 30
            line_height = 35
            font_scale = 0.7
            font_thickness = 2

            # FPS
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
            y_offset += line_height

            # State with color coding
            state = self.state_machine.current_state
            state_value = state.value if hasattr(state, 'value') else str(state)
            state_color = {
                'sitting': (0, 255, 255),  # Yellow
                'lying': (0, 0, 255),      # Red
                'standing': (0, 255, 0),   # Green
                'sleeping': (255, 0, 0),   # Blue
                'absent': (128, 128, 128), # Gray
            }.get(state_value, (255, 255, 255))

            cv2.putText(vis_frame, f"State: {state_value}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, state_color, font_thickness)
            y_offset += line_height

            # Zone
            zone = self.state_machine.current_zone or "None"
            cv2.putText(vis_frame, f"Zone: {zone}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness)
            y_offset += line_height

            # Duration
            duration = self.state_machine.get_state_duration(time.time())
            cv2.putText(vis_frame, f"Duration: {duration:.1f}s", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness)
            y_offset += line_height

        return vis_frame

    def _draw_skeleton(self, frame, keypoints):
        """Draw skeleton connections (same style as main.py)"""
        from src.detectors.base import Keypoint

        # Use Keypoint.get_connections() for consistency with main.py
        connections = Keypoint.get_connections()
        for idx1, idx2 in connections:
            if keypoints[idx1, 2] > 0.3 and keypoints[idx2, 2] > 0.3:
                x1, y1 = keypoints[idx1, :2].astype(int)
                x2, y2 = keypoints[idx2, :2].astype(int)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color like main.py

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

        # Print final performance summary
        self._print_performance_summary()

        # Cleanup
        self.cap.release()
        if self.show_visualization:
            cv2.destroyAllWindows()

        print("[Run] Program exited\n")

    def _print_performance_summary(self):
        """Print detailed performance summary at the end"""
        # Skip detailed profiling if disabled
        if not self.enable_profiling:
            print("\n[Performance] Profiling disabled (set debug.performance_profiling: true to enable)")
            return

        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)

        # Camera info
        camera_config = self.config['camera']
        print(f"\nCamera Configuration:")
        print(f"  Resolution: {camera_config['resolution'][0]}x{camera_config['resolution'][1]}")
        print(f"  Target FPS: {camera_config['fps']}")
        print(f"  Encoding: MJPEG (forced)")

        # Pipeline stats
        print(f"\nPipeline Configuration:")
        print(f"  Mode: Asynchronous 4-thread")
        print(f"  Detection Interval: {self.config['inference'].get('detection_interval', 3)}")

        # Detailed timing statistics
        print(f"\n{'Stage':<20} {'Count':<10} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 70)

        for stage_name, display_name in [
            ('camera_read', 'Camera Read'),
            ('detection', 'YOLO Detection'),
            ('pose_estimation', 'RTMPose Pose'),
            ('state_machine', 'State Machine')
        ]:
            timings = self.timing_history[stage_name]
            if len(timings) > 0:
                avg = np.mean(timings)
                min_val = np.min(timings)
                max_val = np.max(timings)
                count = len(timings)
                print(f"{display_name:<20} {count:<10} {avg:<12.2f} {min_val:<12.2f} {max_val:<12.2f}")
            else:
                print(f"{display_name:<20} {'0':<10} {'-':<12} {'-':<12} {'-':<12}")

        # Calculate theoretical FPS
        if len(self.timing_history['pose_estimation']) > 0:
            avg_camera = np.mean(self.timing_history['camera_read']) if self.timing_history['camera_read'] else 0
            avg_detection = np.mean(self.timing_history['detection']) if self.timing_history['detection'] else 0
            avg_pose = np.mean(self.timing_history['pose_estimation']) if self.timing_history['pose_estimation'] else 0
            avg_state = np.mean(self.timing_history['state_machine']) if self.timing_history['state_machine'] else 0

            # Bottleneck is the slowest stage
            bottleneck = max(avg_pose, avg_detection)
            theoretical_fps = 1000.0 / bottleneck if bottleneck > 0 else 0

            print(f"\n{'Metric':<30} {'Value':<15}")
            print("-" * 50)
            print(f"{'Bottleneck Stage':<30} {'RTMPose' if avg_pose > avg_detection else 'YOLO Detection':<15}")
            print(f"{'Bottleneck Time (ms)':<30} {bottleneck:<15.2f}")
            print(f"{'Theoretical Max FPS':<30} {theoretical_fps:<15.1f}")
            print(f"{'Actual Display FPS':<30} {self.stats['display_fps']:<15d}")

            # Performance breakdown
            total_processing = avg_camera + avg_detection + avg_pose + avg_state
            print(f"\n{'Stage':<30} {'% of Total':<15}")
            print("-" * 50)
            if total_processing > 0:
                print(f"{'Camera Read':<30} {(avg_camera/total_processing*100):<15.1f}")
                print(f"{'YOLO Detection':<30} {(avg_detection/total_processing*100):<15.1f}")
                print(f"{'RTMPose Pose':<30} {(avg_pose/total_processing*100):<15.1f}")
                print(f"{'State Machine':<30} {(avg_state/total_processing*100):<15.1f}")

        print("\n" + "="*70 + "\n")


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
