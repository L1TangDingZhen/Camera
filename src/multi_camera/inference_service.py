"""
Inference Service
Shared YOLO + RTMPose inference service for multiple cameras

Architecture:
- Single YOLO instance shared by up to 2 cameras
- Single RTMPose instance shared by up to 2 cameras
- State machines maintained per camera
- Reduces GPU memory from N×1.5GB to ceil(N/2)×1.5GB
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from threading import Thread, Event
from queue import Queue, Empty
from dataclasses import dataclass

from ..detectors import PersonDetector, PoseEstimatorFactory
from ..state import BehaviorStateMachine, ROIManager, MultiPersonManager
from ..storage import EventLogger


@dataclass
class FramePacket:
    """Frame data packet for queue transmission"""
    frame: np.ndarray
    camera_id: int
    frame_num: int
    timestamp: float


@dataclass
class ResultPacket:
    """Inference result packet"""
    camera_id: int
    frame_num: int
    timestamp: float
    frame: np.ndarray
    bboxes: List[Any]
    keypoints: Optional[np.ndarray]
    world_landmarks: Optional[Any]
    states: List[Dict]
    person_count: int


class InferenceService:
    """
    Shared inference service for multiple cameras

    Handles:
    - YOLO detection (shared, with detection_interval)
    - RTMPose estimation (shared, with pose_interval)
    - State machine updates (per camera)
    - Result distribution (push to output queues)
    """

    def __init__(
        self,
        config: dict,
        camera_ids: List[int],
        input_queues: Dict[int, Queue],
        output_queues: Dict[int, Queue],
        event_logger: EventLogger,
        service_id: int = 0
    ):
        """
        Initialize inference service

        Args:
            config: Shared configuration
            camera_ids: List of camera IDs this service handles
            input_queues: Input queues per camera (camera_id -> Queue)
            output_queues: Output queues per camera (camera_id -> Queue)
            event_logger: Shared event logger
            service_id: ID of this inference service (for multi-service setup)
        """
        self.config = config
        self.camera_ids = camera_ids
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.event_logger = event_logger
        self.service_id = service_id

        # Running flag
        self.running = Event()
        self.running.set()

        # Interval control
        inference_config = config.get('inference', {})
        self.detection_interval = inference_config.get('detection_interval', 3)
        self.pose_interval = inference_config.get('pose_interval', 2)

        # Frame counters per camera
        self.frame_counters: Dict[int, int] = {cam_id: 0 for cam_id in camera_ids}

        # Cached detection results per camera (for skipped frames)
        self.cached_bboxes: Dict[int, List] = {cam_id: [] for cam_id in camera_ids}
        self.cached_keypoints: Dict[int, Optional[np.ndarray]] = {cam_id: None for cam_id in camera_ids}
        self.cached_world_landmarks: Dict[int, Any] = {cam_id: None for cam_id in camera_ids}

        # Verbose mode
        self.verbose = config.get('debug', {}).get('verbose', False)

        # Initialize components
        self._init_components()

        print(f"[InferenceService {service_id}] Initialized for cameras: {camera_ids}")
        print(f"[InferenceService {service_id}] detection_interval={self.detection_interval}, pose_interval={self.pose_interval}")

    def _init_components(self):
        """Initialize detection models and per-camera state machines"""

        # 1. Shared YOLO detector (no tracking, use detect instead of detect_multi)
        print(f"[InferenceService {self.service_id}] Loading YOLO...")
        self.person_detector = PersonDetector(self.config['models']['person'])

        # 2. Shared RTMPose estimator (disable ALL internal smoothing)
        print(f"[InferenceService {self.service_id}] Loading RTMPose...")
        # Create config copy with ALL smoothing disabled
        pose_config = self.config['models']['pose'].copy()
        if 'adaptive_smoother' in pose_config:
            pose_config['adaptive_smoother'] = pose_config['adaptive_smoother'].copy()
            pose_config['adaptive_smoother']['enabled'] = False
        # Also disable fallback smoothing
        pose_config['keypoint_smooth_alpha'] = 0.0
        self.pose_estimator = PoseEstimatorFactory.create(pose_config)

        # 3. Per-camera smoothers (lightweight, only stores arrays)
        from ..smoothers import AdaptiveSmoother
        smoother_config = self.config['models']['pose'].get('adaptive_smoother', {})
        self.smoothers: Dict[int, AdaptiveSmoother] = {}
        if smoother_config.get('enabled', False):
            print(f"[InferenceService {self.service_id}] Creating per-camera smoothers...")
            for cam_id in self.camera_ids:
                self.smoothers[cam_id] = AdaptiveSmoother(
                    conf_threshold=smoother_config.get('conf_threshold', 0.5),
                    conf_enabled=smoother_config.get('conf_enabled', True),
                    static_deadzone=smoother_config.get('static_deadzone', 4.0),
                    moving_deadzone=smoother_config.get('moving_deadzone', 1.5),
                    speed_threshold=smoother_config.get('speed_threshold', 2.0),
                    deadzone_enabled=smoother_config.get('deadzone_enabled', True),
                    static_alpha=smoother_config.get('static_alpha', 0.1),
                    moving_alpha=smoother_config.get('moving_alpha', 0.4),
                    ema_enabled=smoother_config.get('ema_enabled', True),
                    max_velocity=smoother_config.get('max_velocity', 50.0),
                    velocity_limit_enabled=smoother_config.get('velocity_limit_enabled', True),
                    debug=False
                )

        # 4. Per-camera state machines (single person mode - tracking via face recognition later)
        self.enable_multi_person = False  # Disabled, will use face recognition for tracking

        # ROI managers per camera (could be different per camera)
        self.roi_managers: Dict[int, ROIManager] = {}

        # State machines or multi-person managers per camera
        self.state_managers: Dict[int, Any] = {}

        for cam_id in self.camera_ids:
            # Create ROI manager
            roi_config = self.config.get('roi', {})
            self.roi_managers[cam_id] = ROIManager(roi_config)

            # Create state manager
            if self.enable_multi_person:
                self.state_managers[cam_id] = MultiPersonManager(
                    self.config,
                    self.roi_managers[cam_id],
                    self.event_logger
                )
            else:
                # Note: BehaviorStateMachine expects database, not event_logger
                self.state_managers[cam_id] = BehaviorStateMachine(
                    self.config,
                    self.roi_managers[cam_id],
                    self.event_logger.db  # Pass database instance
                )

        print(f"[InferenceService {self.service_id}] Components loaded (multi_person={self.enable_multi_person})")

    def run(self):
        """Main inference loop - polls input queues and processes frames"""

        print(f"[InferenceService {self.service_id}] Starting inference loop...")

        frame_count = 0
        last_log_time = time.time()

        while self.running.is_set():
            processed_any = False

            # Fair polling: iterate through all cameras
            for cam_id in self.camera_ids:
                try:
                    # Non-blocking get from this camera's queue
                    packet: FramePacket = self.input_queues[cam_id].get_nowait()
                    processed_any = True
                    frame_count += 1

                    # Process this frame
                    result = self._process_frame(packet)

                    # Push result to output queue
                    if result is not None:
                        try:
                            self.output_queues[cam_id].put_nowait(result)
                        except:
                            # Output queue full, drop oldest
                            try:
                                self.output_queues[cam_id].get_nowait()
                                self.output_queues[cam_id].put_nowait(result)
                            except:
                                pass

                    # Log progress every 5 seconds
                    current_time = time.time()
                    if current_time - last_log_time >= 5.0:
                        fps = frame_count / (current_time - last_log_time)
                        print(f"[InferenceService {self.service_id}] Processed {frame_count} frames, {fps:.1f} FPS")
                        frame_count = 0
                        last_log_time = current_time

                except Empty:
                    # No frame available from this camera
                    continue
                except Exception as e:
                    print(f"[InferenceService {self.service_id}] Error processing camera {cam_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # If no frames were processed, sleep briefly to avoid busy loop
            if not processed_any:
                time.sleep(0.001)

        print(f"[InferenceService {self.service_id}] Inference loop stopped")

    def _process_frame(self, packet: FramePacket) -> Optional[ResultPacket]:
        """
        Process a single frame

        Args:
            packet: Frame packet containing frame data and metadata

        Returns:
            ResultPacket with inference results
        """
        cam_id = packet.camera_id
        frame = packet.frame
        frame_num = packet.frame_num
        timestamp = packet.timestamp

        # Update frame counter
        self.frame_counters[cam_id] = frame_num

        # Determine what to run this frame
        # Use frame_num == 1 for first frame to ensure immediate detection
        run_detection = (frame_num == 1) or (frame_num % self.detection_interval == 0)
        run_pose = (frame_num == 1) or (frame_num % self.pose_interval == 0)

        bboxes = self.cached_bboxes[cam_id]
        keypoints = self.cached_keypoints[cam_id]
        world_landmarks = self.cached_world_landmarks[cam_id]

        # 1. YOLO Detection (shared detector, no tracking)
        if run_detection:
            bbox = self.person_detector.detect(frame)
            bboxes = [bbox] if bbox is not None else []
            self.cached_bboxes[cam_id] = bboxes

            # Clear keypoints cache when no person detected
            if len(bboxes) == 0:
                self.cached_keypoints[cam_id] = None
                self.cached_world_landmarks[cam_id] = None
                keypoints = None
                world_landmarks = None

        # 2. RTMPose Estimation (shared model + per-camera smoother)
        if run_pose and len(bboxes) > 0:
            primary_bbox = bboxes[0]

            if primary_bbox is not None:
                # Get bbox array
                if isinstance(primary_bbox, dict):
                    bbox_array = primary_bbox.get('bbox')
                elif isinstance(primary_bbox, (list, tuple, np.ndarray)):
                    bbox_array = np.array(primary_bbox)
                else:
                    bbox_array = None

                if bbox_array is not None:
                    # Shared RTMPose estimation
                    keypoints = self.pose_estimator.estimate(frame, bbox_array)

                    # Apply per-camera smoother
                    if keypoints is not None and cam_id in self.smoothers:
                        keypoints = self.smoothers[cam_id].process(keypoints)

                    if hasattr(self.pose_estimator, 'get_world_landmarks'):
                        world_landmarks = self.pose_estimator.get_world_landmarks()

                    self.cached_keypoints[cam_id] = keypoints
                    self.cached_world_landmarks[cam_id] = world_landmarks

        # 3. State Machine Update (single person mode only)
        states = []
        state_manager = self.state_managers[cam_id]

        primary_bbox = bboxes[0] if bboxes else None
        bbox_array = None

        if primary_bbox is not None:
            if isinstance(primary_bbox, dict):
                bbox_array = primary_bbox.get('bbox')
            elif isinstance(primary_bbox, (list, tuple, np.ndarray)):
                bbox_array = np.array(primary_bbox)

        events = state_manager.update(
            bbox_array,
            keypoints,
            timestamp,
            world_landmarks
        )

        states.append({
            'person_id': 0,
            'state': state_manager.current_state,
            'bbox': bbox_array,
            'keypoints': keypoints
        })

        # 4. Create result packet
        result = ResultPacket(
            camera_id=cam_id,
            frame_num=frame_num,
            timestamp=timestamp,
            frame=frame,
            bboxes=bboxes,
            keypoints=keypoints,
            world_landmarks=world_landmarks,
            states=states,
            person_count=len(bboxes)
        )

        return result

    def stop(self):
        """Stop the inference service"""
        self.running.clear()
        print(f"[InferenceService {self.service_id}] Stopping...")

        # Cleanup GPU resources - RTMPose
        if hasattr(self, 'pose_estimator') and self.pose_estimator is not None:
            if hasattr(self.pose_estimator, 'cleanup'):
                try:
                    self.pose_estimator.cleanup()
                    print(f"[InferenceService {self.service_id}] RTMPose cleaned up")
                except Exception as e:
                    print(f"[InferenceService {self.service_id}] RTMPose cleanup warning: {e}")

        # Cleanup GPU resources - YOLO
        if hasattr(self, 'person_detector') and self.person_detector is not None:
            if hasattr(self.person_detector, 'cleanup'):
                try:
                    self.person_detector.cleanup()
                    print(f"[InferenceService {self.service_id}] YOLO cleaned up")
                except Exception as e:
                    print(f"[InferenceService {self.service_id}] YOLO cleanup warning: {e}")

    def get_stats(self) -> Dict:
        """Get service statistics"""
        return {
            'service_id': self.service_id,
            'camera_ids': self.camera_ids,
            'frame_counters': self.frame_counters.copy(),
            'detection_interval': self.detection_interval,
            'pose_interval': self.pose_interval,
        }
