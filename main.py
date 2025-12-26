#!/usr/bin/env python3
"""
Life Tracker

"""

import argparse
import time
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from src.detectors import PersonDetector, PoseEstimatorFactory
from src.state import BehaviorStateMachine, ROIManager, MultiPersonManager
from src.storage import EventLogger


class LifeTracker:
    """Life Tracker main class"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Configuration file path
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        print(f"\n{'='*60}")
        print(f"  Life Tracker - {self.config['name']}")
        print(f"  Device: {self.config['device']}")
        print(f"{'='*60}\n")

        # Initialize components
        self._init_components()

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
        print("[Init] Create event logger...")
        self.event_logger = EventLogger(self.config)

        # 4. Create state machine or multi-person manager
        self.enable_multi_person = self.config['models']['person'].get('enable_tracking', False)

        if self.enable_multi_person:
            print("[Init] Create multi-person manager...")
            self.multi_person_manager = MultiPersonManager(
                self.config,
                self.roi_manager,
                self.event_logger
            )
            self.state_machine = None  # Multi-person mode doesn't use single state machine
            print(f"[Init] Multi-person mode enabled (max: {self.config['models']['person'].get('max_persons', 5)} persons)")
        else:
            print("[Init] Create state machine (single-person mode)...")
            self.state_machine = BehaviorStateMachine(
                self.config,
                self.roi_manager,
                database=self.event_logger.db
            )
            self.multi_person_manager = None

        # 5. InitCamera
        print("[Init] Opening camera...")
        camera_config = self.config['camera']
        camera_source = camera_config['source']

        # If configured camera cannot open, auto-search for available cameras
        self.cap = cv2.VideoCapture(camera_source)

        if not self.cap.isOpened():
            print(f"[WARNING] Camera {camera_source} cannot open, searching for available cameras...")
            found = False
            for i in range(10):  # Try to search /dev/video0 to /dev/video9
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        print(f"[SUCCESS] Found available camera: /dev/video{i} (Resolution: {frame.shape[1]}x{frame.shape[0]})")
                        self.cap = test_cap
                        camera_source = i
                        found = True
                        break
                    test_cap.release()

            if not found:
                raise RuntimeError(f"Cannot find any available Camera (Tried /dev/video0-9)")

        # Set camera parameters
        # Force MJPEG encoding (avoid YUYV bandwidth bottleneck, especially for FHD resolution)
        # MJPEG: ~50-200KB/frame vs YUYV: ~4MB/frame (FHD)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['resolution'][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['resolution'][1])
        self.cap.set(cv2.CAP_PROP_FPS, camera_config['fps'])

        # Print actual camera parameters
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"[Camera] Device: /dev/video{camera_source}")
        print(f"[Camera] Actual resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")

        # Runtime parameters
        self.show_visualization = True
        self.running = True

        print("\n[Init] All components loaded successfully!\n")

    def run(self):
        """Main loop"""
        print("[Running] Start monitoring...\n")

        frame_count = 0
        fps_calc_time = time.time()
        fps = 0

        # Performance profiling
        enable_profiling = self.config.get('debug', {}).get('show_state_info', False)
        if enable_profiling:
            print("🔍 Performance profiling mode enabled\n")

        # Detection frequency control
        detection_interval = self.config.get('inference', {}).get('detection_interval', 1)
        detection_counter = 0  # Detection frame counter
        cached_bbox = None  # Cached bbox (single-person mode)
        cached_detections = []  # Cached detections (multi-person mode)
        if detection_interval > 1:
            print(f"⚡ Detection optimization: Every{detection_interval}frame detect once, reuse results for intermediate frames\n")

        profiling_data = {
            'read_frame': [],
            'detection': [],
            'pose': [],
            'state_machine': [],
            'visualization': [],
            'waitkey': [],
            'total_frame': []
        }

        # Create resizable window
        if self.show_visualization:
            cv2.namedWindow('Life Tracker', cv2.WINDOW_NORMAL)
            # Set default window size to camera resolution（Full HD）
            cam_width = self.config['camera']['resolution'][0]
            cam_height = self.config['camera']['resolution'][1]
            cv2.resizeWindow('Life Tracker', cam_width, cam_height)

        try:
            while self.running:
                frame_start = time.time()

                # Read frame
                t0 = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Cannot read camera frame")
                    break
                t1 = time.time()
                profiling_data['read_frame'].append((t1 - t0) * 1000)

                frame_count += 1
                current_time = time.time()

                # 1. Person detection（Detect every N frames, reuse for intermediate frames）
                detection_counter += 1
                t0 = time.time()

                if self.enable_multi_person:
                    # Multi-person detection and tracking
                    if detection_counter % detection_interval == 0:
                        detections = self.person_detector.detect_multi(frame)
                        cached_detections = detections
                        t1 = time.time()
                        profiling_data['detection'].append((t1 - t0) * 1000)
                    else:
                        detections = cached_detections
                        t1 = time.time()

                    # 2 & 3. Multi-person pose estimation and state update
                    t0 = time.time()
                    events = self.multi_person_manager.update_multi(
                        detections,
                        frame,
                        self.pose_estimator,
                        current_time
                    )
                    t1 = time.time()
                    profiling_data['state_machine'].append((t1 - t0) * 1000)

                    # Save for visualization (store all detections)
                    self._last_detections = detections
                    self._last_keypoints = None  # Will be handled in visualization

                else:
                    # Single-person mode (original logic)
                    if detection_counter % detection_interval == 0:
                        # Execute actual detection
                        bbox = self.person_detector.detect(frame)
                        cached_bbox = bbox  # Cache result
                        t1 = time.time()
                        profiling_data['detection'].append((t1 - t0) * 1000)
                    else:
                        # Reuse previous detection result
                        bbox = cached_bbox
                        t1 = time.time()

                    # 2. Pose estimation
                    t0 = time.time()
                    keypoints = None
                    world_landmarks = None
                    if bbox is not None:
                        keypoints = self.pose_estimator.estimate(frame, bbox)
                        # Get 3D world landmarks (if supported)
                        if hasattr(self.pose_estimator, 'get_world_landmarks'):
                            world_landmarks = self.pose_estimator.get_world_landmarks()
                    t1 = time.time()
                    profiling_data['pose'].append((t1 - t0) * 1000)

                    # Save keypoints for debug display
                    self._last_keypoints = keypoints

                    # 3. Update state machine（Use 3D coordinates）
                    t0 = time.time()
                    events = self.state_machine.update(bbox, keypoints, current_time, world_landmarks)
                    t1 = time.time()
                    profiling_data['state_machine'].append((t1 - t0) * 1000)

                # 4. Log events
                if events:
                    self.event_logger.log_events(events)

                # 5. Record performance metrics（Every 60 seconds）
                if frame_count % (self.config['camera']['fps'] * 60) == 0:
                    detector_metrics = self.person_detector.get_performance_metrics()
                    pose_metrics = self.pose_estimator.get_performance_metrics()
                    self.event_logger.log_performance(detector_metrics, pose_metrics)

                # 6. Visualization
                t0 = time.time()
                if self.show_visualization:
                    if self.enable_multi_person:
                        vis_frame = self._visualize_multi(frame, detections, fps)
                    else:
                        vis_frame = self._visualize(frame, bbox, keypoints, fps)
                    cv2.imshow('Life Tracker', vis_frame)
                t1 = time.time()
                profiling_data['visualization'].append((t1 - t0) * 1000)

                # 7. Handle key
                t0 = time.time()
                if self.show_visualization:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[Exit] User pressed'q'key")
                        break
                    elif key == ord('r'):
                        # Toggle ROI display
                        self.show_roi = not getattr(self, 'show_roi', True)
                t1 = time.time()
                profiling_data['waitkey'].append((t1 - t0) * 1000)

                # Record total frame time
                frame_end = time.time()
                profiling_data['total_frame'].append((frame_end - frame_start) * 1000)

                # 8. Calculate FPS
                if current_time - fps_calc_time >= 1.0:
                    fps = frame_count / (current_time - fps_calc_time)
                    frame_count = 0
                    fps_calc_time = current_time

                # 9. Every30frameOutput performance profiling（approx3once per second）
                if enable_profiling and len(profiling_data['total_frame']) >= 30:
                    self._print_profiling(profiling_data)
                    # Clear data
                    for key in profiling_data:
                        profiling_data[key] = []

        except KeyboardInterrupt:
            print("\n[Exit] User interrupted (Ctrl+C)")

        finally:
            self.cleanup()

    def _visualize(self, frame: np.ndarray, bbox: np.ndarray,
                   keypoints: np.ndarray, fps: float) -> np.ndarray:
        """
        Visualization

        Args:
            frame: Original frame
            bbox: Bounding box
            keypoints: Keypoints
            fps: Frame rate

        Returns:
            Visualized frame
        """
        vis_frame = frame.copy()

        # Draw ROI zones
        if getattr(self, 'show_roi', True):
            vis_frame = self.roi_manager.draw_zones(vis_frame)

        # Draw bbox
        if bbox is not None:
            x1, y1, x2, y2, conf = bbox.astype(int)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Person: {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw keypoints
        if keypoints is not None:
            self._draw_keypoints(vis_frame, keypoints, bbox)

        # Draw state info
        self._draw_status(vis_frame, fps)

        return vis_frame

    def _visualize_multi(self, frame: np.ndarray, detections: list, fps: float) -> np.ndarray:
        """
        Multi-person visualization

        Args:
            frame: Original frame
            detections: List of detections [{bbox, tracking_id, confidence}, ...]
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
        for detection in detections:
            tracking_id = detection['tracking_id']
            bbox = detection['bbox']
            confidence = detection['confidence']

            # Get color for this tracking_id
            color = colors.get(tracking_id % len(colors), (0, 255, 0))

            # Draw bbox
            x1, y1, x2, y2 = bbox[:4].astype(int)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)

            # Draw skeleton if keypoints available
            if self.multi_person_manager and tracking_id in self.multi_person_manager.person_keypoints:
                keypoints = self.multi_person_manager.person_keypoints[tracking_id]
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
        self._draw_status_multi(vis_frame, fps, len(detections))

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

        # Confidence thresholds (same as single-person mode)
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
                # Use person's assigned color for keypoints
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
                    # Use person's assigned color for skeleton lines
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    def _draw_status_multi(self, frame: np.ndarray, fps: float, person_count: int):
        """Draw status info for multi-person mode"""
        info_y = 30

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Person count
        info_y += 35
        cv2.putText(frame, f"Persons: {person_count}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Active tracking count
        if self.multi_person_manager:
            active_count = self.multi_person_manager.get_active_person_count()
            info_y += 35
            cv2.putText(frame, f"Tracked: {active_count}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # All states
            all_states = self.multi_person_manager.get_all_states()
            for tracking_id, state in all_states.items():
                info_y += 30
                cv2.putText(frame, f"  ID {tracking_id}: {state}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def _draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, bbox: Optional[np.ndarray] = None):
        """Draw keypointsand skeleton"""
        from src.detectors.base import Keypoint

        # init cache to hold recent reliable joints for brief occlusion
        if not hasattr(self, '_last_draw_keypoints'):
            self._last_draw_keypoints = None

        # 定义阈值（保持上半身原阈值，对下半身增加高/低双阈值，配合长度过滤防乱连、兼顾遮挡兜底）
        lower_body_indices = {11, 12, 13, 14, 15, 16}
        upper_body_high = 0.3
        upper_body_low = 0.25  # 稍宽松，用于轻微遮挡
        lower_body_high = 0.55
        lower_body_low = 0.4   # 稍宽松，用于半遮挡腿部

        # 基于bbox高度估算最大允许线段长度，过滤缺点导致的超长连线
        max_seg_len = None
        if bbox is not None:
            _, y1, _, y2, _ = bbox.astype(float)
            max_seg_len = (y2 - y1) * 1.2  # 留一定余量
        else:
            max_seg_len = frame.shape[0] * 0.6  # 无bbox时按画面高度约束

        # 构造可视化用的关键点，短暂持有上帧的高置信度位置，避免轻微遮挡时直接断线
        vis_keypoints = keypoints.copy()
        if self._last_draw_keypoints is not None and self._last_draw_keypoints.shape == keypoints.shape:
            for i in range(keypoints.shape[0]):
                high = lower_body_high if i in lower_body_indices else upper_body_high
                low = lower_body_low if i in lower_body_indices else upper_body_low
                if vis_keypoints[i, 2] < high:
                    prev_conf = self._last_draw_keypoints[i, 2]
                    if prev_conf > high or (vis_keypoints[i, 2] > low and prev_conf > low):
                        vis_keypoints[i, :2] = self._last_draw_keypoints[i, :2]
                        vis_keypoints[i, 2] = max(vis_keypoints[i, 2], prev_conf) * 0.9  # 轻微衰减，避免长期保留

        # Draw keypoints
        for i, (x, y, conf) in enumerate(vis_keypoints):
            high = lower_body_high if i in lower_body_indices else upper_body_high
            low = lower_body_low if i in lower_body_indices else upper_body_low
            # 点的绘制：高置信或（当前中等+上一帧高）时也画，缓和遮挡
            if conf > high or (conf > low and self._last_draw_keypoints is not None and self._last_draw_keypoints.shape == vis_keypoints.shape and self._last_draw_keypoints[i, 2] > high):
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        # Draw skeleton
        connections = Keypoint.get_connections()
        for idx1, idx2 in connections:
            high1 = lower_body_high if idx1 in lower_body_indices else upper_body_high
            high2 = lower_body_high if idx2 in lower_body_indices else upper_body_high
            low1 = lower_body_low if idx1 in lower_body_indices else upper_body_low
            low2 = lower_body_low if idx2 in lower_body_indices else upper_body_low

            # 线段绘制采用“至少一端高置信 + 双端不低于宽松阈值”的策略，兼顾稳定和遮挡兜底
            c1 = vis_keypoints[idx1, 2]
            c2 = vis_keypoints[idx2, 2]
            allow_line = ((c1 > high1 and c2 > low2) or (c2 > high2 and c1 > low1)) and (c1 > low1 and c2 > low2)

            if allow_line:
                x1, y1 = vis_keypoints[idx1, :2].astype(int)
                x2, y2 = vis_keypoints[idx2, :2].astype(int)

                # 跳过明显超长的线段以避免乱连
                seg_len = np.hypot(x1 - x2, y1 - y2)
                if max_seg_len is not None and seg_len > max_seg_len:
                    continue

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 保存本帧可视化关键点，用于下一帧短暂补偿
        self._last_draw_keypoints = vis_keypoints

    def _draw_status(self, frame: np.ndarray, fps: float):
        """Draw state info"""
        h, w = frame.shape[:2]

        # Debug mode: show more detailed information
        debug_mode = self.config.get('debug', {}).get('show_state_info', False)
        info_height = 150 if not debug_mode else 250

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Text information（Optimized for Full HD resolution）
        y_offset = 40
        line_height = 35
        font_scale = 1.0  # Increase font size（original0.6）
        font_thickness = 3  # Increase thickness（original2）

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        y_offset += line_height

        # Current state
        state = self.state_machine.get_current_state()
        state_color = {
            'sitting': (0, 255, 255),  # Yellow
            'lying': (255, 0, 255),    # Purple
            'standing': (0, 255, 0),   # Green
            'sleeping': (255, 0, 0),   # Blue
            'absent': (128, 128, 128), # Gray
        }.get(state.value, (255, 255, 255))

        cv2.putText(frame, f"State: {state.value}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, state_color, font_thickness)
        y_offset += line_height

        # Current zone
        zone = self.state_machine.current_zone or "None"
        cv2.putText(frame, f"Zone: {zone}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness)
        y_offset += line_height

        # State duration
        duration = self.state_machine.get_state_duration(time.time())
        cv2.putText(frame, f"Duration: {duration:.1f}s", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness)
        y_offset += line_height

        # Debug info（Optimized for Full HD）
        debug_font_scale = 0.7  # Increase debug text size（original0.4-0.5）
        debug_thickness = 2     # Increase thickness（original1）
        debug_line_height = 30  # Increase line spacing（original18-20）

        if debug_mode and hasattr(self, '_last_keypoints') and self._last_keypoints is not None:
            from src.detectors.base import Keypoint, PoseUtils
            kp = self._last_keypoints

            # Calculate key metrics
            try:
                # Body angle
                body_angle = PoseUtils.get_body_orientation(kp)
                cv2.putText(frame, f"Body Angle: {body_angle:.1f}deg", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, debug_font_scale, (0, 255, 255), debug_thickness)
                y_offset += debug_line_height

                # Knee angle
                if kp[Keypoint.LEFT_HIP, 2] > 0.3 and kp[Keypoint.LEFT_KNEE, 2] > 0.3 and kp[Keypoint.LEFT_ANKLE, 2] > 0.3:
                    knee_angle = PoseUtils.calculate_angle(
                        kp[Keypoint.LEFT_HIP, :2],
                        kp[Keypoint.LEFT_KNEE, :2],
                        kp[Keypoint.LEFT_ANKLE, :2]
                    )
                    cv2.putText(frame, f"Knee Angle: {knee_angle:.1f}deg", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, debug_font_scale, (0, 255, 255), debug_thickness)
                    y_offset += debug_line_height

                # Body height
                body_height = PoseUtils.get_body_height(kp)
                cv2.putText(frame, f"Height: {body_height:.0f}px", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, debug_font_scale, (0, 255, 255), debug_thickness)
                y_offset += debug_line_height

                # Diagnostic info（Show decision basis）
                diagnosis = self.state_machine.get_diagnosis()
                if diagnosis:
                    y_offset += 15  # Empty line
                    cv2.putText(frame, "=== Diagnosis ===", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, debug_font_scale, (255, 200, 0), debug_thickness)
                    y_offset += debug_line_height

                    mode = diagnosis.get('mode', 'N/A')
                    cv2.putText(frame, f"Mode: {mode}", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, debug_font_scale, (255, 255, 255), debug_thickness)
                    y_offset += debug_line_height

                    # Show different diagnostic info based on mode
                    if mode == '3d':
                        # 3D mode: show real 3D features
                        if 'torso_angle' in diagnosis:
                            torso_angle = diagnosis['torso_angle']
                            color = (0, 255, 255)  # Yellow
                            cv2.putText(frame, f"TorsoAngle: {torso_angle:.1f}deg (0=upright, 90=horizontal)",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, debug_font_scale, color, debug_thickness)
                            y_offset += debug_line_height

                        if 'hip_knee_z_diff' in diagnosis:
                            z_diff = diagnosis['hip_knee_z_diff']
                            color = (0, 255, 255)
                            cv2.putText(frame, f"Hip-Knee Z: {z_diff:.1f}cm (>0=sitting, ~0=standing)",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, debug_font_scale, color, debug_thickness)
                            y_offset += debug_line_height

                        if 'hip_knee_dist' in diagnosis:
                            dist = diagnosis['hip_knee_dist']
                            color = (0, 255, 255)
                            cv2.putText(frame, f"Hip-Knee Dist: {dist:.1f}cm (>35=extended)",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                        # Show judgment result
                        if 'lying_check' in diagnosis:
                            lying = diagnosis['lying_check']
                            color = (0, 255, 0) if lying else (128, 128, 128)
                            cv2.putText(frame, f"Lying: {'YES' if lying else 'NO'}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                        if 'standing_check' in diagnosis:
                            standing = diagnosis['standing_check']
                            color = (0, 255, 0) if standing else (128, 128, 128)
                            cv2.putText(frame, f"Standing: {'YES' if standing else 'NO'}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                        # Show SVM probability distribution (if available)
                        if hasattr(self.state_machine, 'last_probabilities') and self.state_machine.last_probabilities:
                            y_offset += 10  # Increase spacing
                            cv2.putText(frame, "SVM Probabilities:",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            y_offset += 20

                            probs = self.state_machine.last_probabilities
                            # Sort by probability descending
                            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

                            for label, prob in sorted_probs:
                                # Color: high probability use Green, low use Gray
                                if prob > 0.5:
                                    color = (0, 255, 0)  # Green
                                elif prob > 0.3:
                                    color = (0, 255, 255)  # Yellow
                                else:
                                    color = (128, 128, 128)  # Gray

                                # Draw probability bar
                                bar_width = int(prob * 150)  # max150pixels
                                cv2.rectangle(frame, (120, y_offset - 10), (120 + bar_width, y_offset + 5),
                                            color, -1)

                                # Show text
                                text = f"{label.capitalize()}: {prob:.2f}"
                                cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                          0.4, color, 1)
                                y_offset += 18

                    elif mode == 'upper_body':
                        # Upper body mode: show body_angle and shoulder_hip_ratio
                        if 'body_angle' in diagnosis:
                            angle = diagnosis['body_angle']
                            angle_range = diagnosis.get('body_angle_range', (0, 0))
                            angle_ok = diagnosis.get('body_angle_ok', False)
                            color = (0, 255, 0) if angle_ok else (0, 0, 255)
                            status = "OK" if angle_ok else "FAIL"
                            cv2.putText(frame, f"Angle: {angle:.1f} [{angle_range[0]}-{angle_range[1]}] {status}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                        if 'shoulder_hip_ratio' in diagnosis:
                            ratio = diagnosis['shoulder_hip_ratio']
                            ratio_range = diagnosis.get('ratio_range', (0, 0))
                            ratio_ok = diagnosis.get('ratio_ok', False)
                            color = (0, 255, 0) if ratio_ok else (0, 0, 255)
                            status = "OK" if ratio_ok else "FAIL"
                            cv2.putText(frame, f"Ratio: {ratio:.2f} [{ratio_range[0]}-{ratio_range[1]}] {status}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                    elif mode == 'full_body':
                        # Full body mode: show knee_angle and hip_height_ratio
                        if 'knee_angle' in diagnosis:
                            knee_angle = diagnosis['knee_angle']
                            knee_threshold = diagnosis.get('knee_angle_threshold', 120)
                            knee_ok = diagnosis.get('knee_angle_ok', False)
                            color = (0, 255, 0) if knee_ok else (0, 0, 255)
                            status = "OK" if knee_ok else "FAIL"
                            cv2.putText(frame, f"Knee: {knee_angle:.1f} [<{knee_threshold}] {status}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                        if 'hip_height_ratio' in diagnosis:
                            hip_ratio = diagnosis['hip_height_ratio']
                            hip_range = diagnosis.get('hip_ratio_range', (0, 0))
                            hip_ok = diagnosis.get('hip_ratio_ok', False)
                            color = (0, 255, 0) if hip_ok else (0, 0, 255)
                            status = "OK" if hip_ok else "FAIL"
                            cv2.putText(frame, f"HipRatio: {hip_ratio:.2f} [{hip_range[0]}-{hip_range[1]}] {status}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

            except:
                pass

        # Hint info
        tips = "Press: 'q'=quit"
        if debug_mode:
            tips += " | Debug ON"
        cv2.putText(frame, tips, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _print_profiling(self, profiling_data):
        """Print performance profiling report"""
        print("\n" + "="*70)
        print("  Performance profiling report (30frame average)")
        print("="*70)

        # Calculate average value for each stage
        total_avg = np.mean(profiling_data['total_frame'])

        stages = [
            ('Read frame', 'read_frame'),
            ('Person detection', 'detection'),
            ('Pose estimation', 'pose'),
            ('State machine update', 'state_machine'),
            ('Visualization rendering', 'visualization'),
            ('waitKey', 'waitkey'),
        ]

        print(f"{'Stage':<12} {'Avg Time':>10} {'Ratio':>8} {'Min':>10} {'Max':>10}")
        print("-"*70)

        detection_count = len(profiling_data['detection'])
        total_frames = len(profiling_data['total_frame'])

        for name, key in stages:
            if profiling_data[key]:
                avg = np.mean(profiling_data[key])
                min_val = np.min(profiling_data[key])
                max_val = np.max(profiling_data[key])
                percentage = (avg / total_avg * 100) if total_avg > 0 else 0

                # Show actual detection times for detection stage
                if key == 'detection' and detection_count < total_frames:
                    print(f"{name:<12} {avg:>8.2f}ms {percentage:>6.1f}% {min_val:>8.2f}ms {max_val:>8.2f}ms (only{detection_count}times)")
                else:
                    print(f"{name:<12} {avg:>8.2f}ms {percentage:>6.1f}% {min_val:>8.2f}ms {max_val:>8.2f}ms")

        print("-"*70)
        print(f"{'Total':<12} {total_avg:>8.2f}ms {'100.0%':>7}")
        print(f"{'Theoretical FPS':<12} {1000/total_avg:>8.1f}")

        # Show detection optimization info
        detection_interval = self.config.get('inference', {}).get('detection_interval', 1)
        if detection_interval > 1:
            print(f"{'Detection interval':<12} Every{detection_interval}frameDetection1times (reduce{(1-1/detection_interval)*100:.0f}%detection load)")

        print("="*70 + "\n")

    def cleanup(self):
        """Cleanup resources"""
        print("\n[Cleanup] Releasing resources...")

        # Cleanup GPU resources - RTMPose
        if hasattr(self, 'pose_estimator') and hasattr(self.pose_estimator, 'cleanup'):
            try:
                self.pose_estimator.cleanup()
                print("[Cleanup] RTMPose cleaned up")
            except Exception as e:
                print(f"[Cleanup] RTMPose warning: {e}")

        # Cleanup GPU resources - YOLO
        if hasattr(self, 'person_detector') and hasattr(self.person_detector, 'cleanup'):
            try:
                self.person_detector.cleanup()
                print("[Cleanup] YOLO cleaned up")
            except Exception as e:
                print(f"[Cleanup] YOLO warning: {e}")

        # Final CUDA cache cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[Cleanup] CUDA cache cleared")
        except:
            pass

        # Release camera
        if hasattr(self, 'cap'):
            self.cap.release()

        # Close windows
        cv2.destroyAllWindows()

        # Close event logger
        if hasattr(self, 'event_logger'):
            self.event_logger.close()

        print("[Cleanup] All resources released!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Life Tracker - Synchronous Mode (Single Camera)')

    parser.add_argument('--config', type=str, default='config/config_gpu.yaml',
                       help='Configuration file path (default: config_gpu.yaml)')
    parser.add_argument('--no-vis', action='store_true',
                       help='Do not show visualization window')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: show keypoints, skeleton and decision info')

    args = parser.parse_args()

    # Use specified config file
    config_path = args.config

    # Check configuration file
    if not Path(config_path).exists():
        print(f"ERROR: Configuration file does not exist: {config_path}")
        print(f"\nAvailable configuration files:")
        print(f"  config/config_cpu.yaml  - CPU mode（Laptop/X390）")
        print(f"  config/config_gpu.yaml  - GPU mode（PC/Jetson）")
        return

    # Create tracker
    tracker = LifeTracker(config_path)

    # Enable debug mode
    if args.debug:
        tracker.config['debug']['show_keypoints'] = True
        tracker.config['debug']['show_skeleton'] = True
        tracker.config['debug']['show_angles'] = True
        print("\n🔍 Debug mode enabled")

    if args.no_vis:
        tracker.show_visualization = False

    # Running
    tracker.run()


if __name__ == '__main__':
    main()
