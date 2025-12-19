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
from src.state import BehaviorStateMachine, ROIManager, MultiPersonManager
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
        self.queue_detections = Queue(maxsize=2)   # YOLO → Pose (multi-person support)
        self.queue_results = Queue(maxsize=2)      # Pose → StateMachine (multi-person support)

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

        # 4. Create state machine or multi-person manager
        self.enable_multi_person = self.config['models']['person'].get('enable_tracking', False)

        if self.enable_multi_person:
            print("[Init] Creating multi-person manager...")
            self.multi_person_manager = MultiPersonManager(
                self.config, self.roi_manager, self.event_logger
            )
            self.state_machine = None
            print(f"[Init] Multi-person mode enabled (max: {self.config['models']['person'].get('max_persons', 5)} persons)")
        else:
            print("[Init] Creating state machine (single-person mode)...")
            self.state_machine = BehaviorStateMachine(
                self.config, self.roi_manager, database=self.event_logger.db
            )
            self.multi_person_manager = None

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
        """Detect person every N frames, reuse cached data for intermediate frames"""
        print("[Thread 2] YOLO detection thread started")
        detection_interval = self.config['inference'].get('detection_interval', 3)
        frame_count = 0
        cached_data = None  # Cache either bbox (single-person) or detections list (multi-person)
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
                    if self.enable_multi_person:
                        # Multi-person detection
                        detections = self.person_detector.detect_multi(frame)
                        cached_data = detections
                    else:
                        # Single-person detection
                        bbox = self.person_detector.detect(frame)
                        cached_data = bbox
                    t1 = time.time()
                    detect_time = (t1 - t0) * 1000
                    timings['detect'].append(detect_time)
                    if self.enable_profiling:
                        self.timing_history['detection'].append(detect_time)
                else:
                    # Reuse cached data for intermediate frames
                    pass  # cached_data stays the same

                # Put to next queue
                t0 = time.time()
                try:
                    self.queue_detections.put((timestamp, frame, cached_data), block=False)
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
        """Estimate pose from detected person(s)"""
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

                # Get data from queue (cached_data is either bbox or detections list)
                t0 = time.time()
                timestamp, frame, cached_data = self.queue_detections.get(timeout=0.1)
                t1 = time.time()
                timings['queue_get'].append((t1 - t0) * 1000)

                frame_count += 1

                if self.enable_multi_person:
                    # Multi-person mode: process each detection
                    detections = cached_data  # This is a list of detections
                    results = []

                    t0 = time.time()
                    if detections:
                        for detection in detections:
                            bbox = detection['bbox']
                            tracking_id = detection['tracking_id']

                            # Estimate pose for this person
                            keypoints = self.pose_estimator.estimate(frame, bbox)

                            # Add keypoints to detection dict
                            result = detection.copy()
                            result['keypoints'] = keypoints
                            results.append(result)
                    t1 = time.time()
                    pose_time = (t1 - t0) * 1000 if detections else 0.0
                    timings['estimate'].append(pose_time)
                    if self.enable_profiling:
                        self.timing_history['pose_estimation'].append(pose_time)

                    # Put results to next queue
                    t0 = time.time()
                    try:
                        self.queue_results.put((timestamp, frame, results), block=False)
                    except:
                        pass  # Queue full, drop frame
                    t1 = time.time()
                    timings['queue_put'].append((t1 - t0) * 1000)

                else:
                    # Single-person mode: original logic with caching
                    bbox = cached_data  # This is a single bbox
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
                        self.queue_results.put(
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

                # Get data from queue (different structure for multi vs single)
                t0 = time.time()
                if self.enable_multi_person:
                    # Multi-person: (timestamp, frame, results)
                    # results is list of {bbox, tracking_id, confidence, keypoints}
                    timestamp, frame, results = self.queue_results.get(timeout=0.1)
                else:
                    # Single-person: (timestamp, frame, bbox, keypoints, world_landmarks)
                    timestamp, frame, bbox, keypoints, world_landmarks = \
                        self.queue_results.get(timeout=0.1)
                t1 = time.time()
                timings['queue_get'].append((t1 - t0) * 1000)

                frame_count += 1
                current_time = time.time()

                # Update state machine
                t0 = time.time()
                if self.enable_multi_person:
                    # Multi-person mode: use MultiPersonManager
                    # Convert results back to detections for update_multi
                    detections = []
                    for result in results:
                        detections.append({
                            'bbox': result['bbox'],
                            'tracking_id': result['tracking_id'],
                            'confidence': result['confidence']
                        })

                    # Note: MultiPersonManager.update_multi does pose estimation internally
                    # So we need to pass the keypoints separately
                    # For now, we'll store keypoints in manager before calling update
                    for result in results:
                        tid = result['tracking_id']
                        self.multi_person_manager.person_keypoints[tid] = result['keypoints']
                        self.multi_person_manager.person_bboxes[tid] = result['bbox']

                    # Update states for all persons (without re-estimating pose)
                    events = []
                    for result in results:
                        tid = result['tracking_id']
                        bbox = result['bbox']
                        keypoints = result['keypoints']

                        # Create or get state machine for this person
                        if tid not in self.multi_person_manager.state_machines:
                            from src.state import BehaviorStateMachine
                            self.multi_person_manager.state_machines[tid] = BehaviorStateMachine(
                                self.config, self.roi_manager, person_id=tid
                            )

                        # Update state machine
                        self.multi_person_manager.last_update_times[tid] = current_time
                        person_events = self.multi_person_manager.state_machines[tid].update(
                            bbox, keypoints, current_time, None
                        )

                        # Add tracking_id to events
                        for event in person_events:
                            event.tracking_id = tid
                        events.extend(person_events)

                    # Cleanup inactive trackers
                    self.multi_person_manager._cleanup_inactive_trackers(
                        current_time, set(r['tracking_id'] for r in results)
                    )
                else:
                    # Single-person mode
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
                    if self.enable_multi_person:
                        vis_frame = self._visualize_multi(frame, results, fps)
                    else:
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

    def _visualize_multi(self, frame: np.ndarray, results: list, fps: float) -> np.ndarray:
        """
        Multi-person visualization

        Args:
            frame: Original frame
            results: List of detection results [{bbox, tracking_id, confidence, keypoints}, ...]
            fps: Frame rate

        Returns:
            Visualized frame
        """
        vis_frame = frame.copy()

        # Draw ROI zones
        if getattr(self, 'show_roi', True):
            vis_frame = self.roi_manager.draw_zones(vis_frame)

        # Define colors for different tracking_ids
        colors = {
            0: (255, 100, 100),   # Light blue
            1: (100, 255, 100),   # Light green
            2: (100, 100, 255),   # Light red
            3: (255, 255, 100),   # Light cyan
            4: (255, 100, 255),   # Light magenta
        }

        # Draw each person
        for result in results:
            tracking_id = result['tracking_id']
            bbox = result['bbox']
            keypoints = result.get('keypoints')

            # Get color for this tracking_id
            color = colors.get(tracking_id % len(colors), (0, 255, 0))

            # Draw bbox
            x1, y1, x2, y2 = bbox[:4].astype(int)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)

            # Draw skeleton if keypoints available
            if keypoints is not None:
                self._draw_skeleton_for_person(vis_frame, keypoints, bbox, color)

            # Get person state if available
            person_state = "unknown"
            if self.multi_person_manager:
                state = self.multi_person_manager.get_person_state(tracking_id)
                if state:
                    person_state = state

            # Draw label with tracking_id and state
            label = f"ID:{tracking_id} [{person_state}]"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Label background
            cv2.rectangle(
                vis_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                color,
                -1
            )

            # Label text
            cv2.putText(
                vis_frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Draw status (FPS, person count, etc.)
        self._draw_status_multi(vis_frame, fps, len(results))

        return vis_frame

    def _draw_skeleton_for_person(self, frame: np.ndarray, keypoints: np.ndarray,
                                   bbox: np.ndarray, color: tuple):
        """
        Draw skeleton for a single person in multi-person mode

        Args:
            frame: Frame to draw on
            keypoints: Person's keypoints (17x3 array)
            bbox: Person's bounding box
            color: Color to use for this person
        """
        from src.detectors.base import Keypoint

        # Define body part indices for confidence thresholds
        lower_body_indices = {11, 12, 13, 14, 15, 16}  # hips, knees, ankles

        # Confidence thresholds
        upper_body_high = 0.3
        upper_body_low = 0.25
        lower_body_high = 0.55
        lower_body_low = 0.4

        # Max segment length based on bbox height
        _, y1, _, y2, _ = bbox.astype(float)
        max_seg_len = (y2 - y1) * 1.2

        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            high = lower_body_high if i in lower_body_indices else upper_body_high
            if conf > high:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)

        # Draw skeleton connections
        connections = Keypoint.get_connections()
        for idx1, idx2 in connections:
            high1 = lower_body_high if idx1 in lower_body_indices else upper_body_high
            high2 = lower_body_high if idx2 in lower_body_indices else upper_body_high
            low1 = lower_body_low if idx1 in lower_body_indices else upper_body_low
            low2 = lower_body_low if idx2 in lower_body_indices else upper_body_low

            c1 = keypoints[idx1, 2]
            c2 = keypoints[idx2, 2]
            allow_line = ((c1 > high1 and c2 > low2) or (c2 > high2 and c1 > low1)) and (c1 > low1 and c2 > low2)

            if allow_line:
                x1, y1 = keypoints[idx1, :2].astype(int)
                x2, y2 = keypoints[idx2, :2].astype(int)

                # Skip overly long segments
                seg_len = np.hypot(x1 - x2, y1 - y2)
                if seg_len <= max_seg_len:
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    def _draw_status_multi(self, frame: np.ndarray, fps: float, person_count: int):
        """Draw status info for multi-person mode"""
        info_y = 30

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        info_y += 40
        # Person count
        cv2.putText(frame, f"Persons: {person_count}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Active tracking count
        if self.multi_person_manager:
            active_count = self.multi_person_manager.get_active_person_count()
            info_y += 40
            cv2.putText(frame, f"Tracked: {active_count}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # List all tracked persons and states
            all_states = self.multi_person_manager.get_all_states()
            for tracking_id, state in all_states.items():
                info_y += 35
                cv2.putText(frame, f"  ID {tracking_id}: {state}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
