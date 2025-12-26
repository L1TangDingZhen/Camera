"""
Batch Inference Service
Collects frames from N cameras and processes them in batches for optimal GPU utilization

Architecture:
- Single service handles ALL cameras (not ceil(N/2) services)
- Collects frames from all cameras into a batch
- YOLO batch inference: process N frames at once
- RTMPose batch inference: process N person crops at once
- Per-camera state machines and smoothers maintained

Performance gains (from benchmarks):
- Batch=1: 20.4ms/frame
- Batch=4: 14.97ms/frame (1.36x speedup)
- Batch=8: 11.60ms/frame (1.76x speedup)
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


class BatchInferenceService:
    """
    Batch inference service for multiple cameras

    Key differences from InferenceService:
    1. Single service handles ALL cameras (not split)
    2. Collects frames into batches before inference
    3. Uses detect_batch() for YOLO batch inference
    4. Uses infer_batch() for RTMPose batch inference (if available)
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
        Initialize batch inference service

        Args:
            config: Shared configuration
            camera_ids: List of ALL camera IDs this service handles
            input_queues: Input queues per camera (camera_id -> Queue)
            output_queues: Output queues per camera (camera_id -> Queue)
            event_logger: Shared event logger
            service_id: ID of this service (usually 0 for single service)
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

        # Batch configuration
        batch_config = config.get('batch', {})
        self.batch_timeout_ms = batch_config.get('timeout_ms', 10)  # Max wait for batch collection

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

        # Performance tracking
        self.batch_sizes = []
        self.inference_times = []

        # Initialize components
        self._init_components()

        print(f"[BatchInferenceService] Initialized for {len(camera_ids)} cameras: {camera_ids}")
        print(f"[BatchInferenceService] timeout={self.batch_timeout_ms}ms")

    def _init_components(self):
        """Initialize detection models and per-camera state machines"""

        # 1. Shared YOLO detector with batch support
        print(f"[BatchInferenceService] Loading YOLO (batch mode)...")
        person_config = self.config['models']['person'].copy()
        self.person_detector = PersonDetector(person_config)

        # 2. Shared RTMPose estimator (disable internal smoothing)
        print(f"[BatchInferenceService] Loading RTMPose...")
        pose_config = self.config['models']['pose'].copy()
        if 'adaptive_smoother' in pose_config:
            pose_config['adaptive_smoother'] = pose_config['adaptive_smoother'].copy()
            pose_config['adaptive_smoother']['enabled'] = False
        pose_config['keypoint_smooth_alpha'] = 0.0
        self.pose_estimator = PoseEstimatorFactory.create(pose_config)

        # 3. Per-camera smoothers
        from ..smoothers import AdaptiveSmoother
        smoother_config = self.config['models']['pose'].get('adaptive_smoother', {})
        self.smoothers: Dict[int, AdaptiveSmoother] = {}
        if smoother_config.get('enabled', False):
            print(f"[BatchInferenceService] Creating per-camera smoothers...")
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

        # 4. Per-camera state machines
        self.roi_managers: Dict[int, ROIManager] = {}
        self.state_managers: Dict[int, Any] = {}

        for cam_id in self.camera_ids:
            roi_config = self.config.get('roi', {})
            self.roi_managers[cam_id] = ROIManager(roi_config)
            self.state_managers[cam_id] = BehaviorStateMachine(
                self.config,
                self.roi_managers[cam_id],
                self.event_logger.db
            )

        print(f"[BatchInferenceService] Components loaded")

    def run(self):
        """Main batch inference loop"""
        print(f"[BatchInferenceService] Starting batch inference loop...")

        total_frames = 0
        total_batches = 0
        last_log_time = time.time()

        while self.running.is_set():
            # Collect batch from all cameras
            batch_packets = self._collect_batch()

            if not batch_packets:
                time.sleep(0.001)
                continue

            # Process batch
            results = self._process_batch(batch_packets)

            # Distribute results
            for result in results:
                cam_id = result.camera_id
                try:
                    self.output_queues[cam_id].put_nowait(result)
                except:
                    try:
                        self.output_queues[cam_id].get_nowait()
                        self.output_queues[cam_id].put_nowait(result)
                    except:
                        pass

            total_frames += len(batch_packets)
            total_batches += 1

            # Log stats every 5 seconds
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                elapsed = current_time - last_log_time
                fps = total_frames / elapsed
                avg_batch = total_frames / total_batches if total_batches > 0 else 0
                print(f"[BatchInferenceService] {fps:.1f} FPS, avg batch size: {avg_batch:.1f}")
                total_frames = 0
                total_batches = 0
                last_log_time = current_time

        print(f"[BatchInferenceService] Loop stopped")

    def _collect_batch(self) -> List[FramePacket]:
        """
        Collect frames from all cameras into a batch

        Strategy:
        - Try to get one frame from each camera
        - Wait up to batch_timeout_ms for frames
        - Return whatever we have collected
        """
        batch = []
        start_time = time.time()
        timeout_sec = self.batch_timeout_ms / 1000.0

        # Try to collect from each camera
        for cam_id in self.camera_ids:
            try:
                # Calculate remaining timeout
                elapsed = time.time() - start_time
                remaining = max(0.001, timeout_sec - elapsed)

                packet = self.input_queues[cam_id].get(timeout=remaining)
                batch.append(packet)
            except Empty:
                continue
            except Exception as e:
                if self.verbose:
                    print(f"[BatchInferenceService] Error collecting from camera {cam_id}: {e}")
                continue

        return batch

    def _process_batch(self, packets: List[FramePacket]) -> List[ResultPacket]:
        """
        Process a batch of frames

        Args:
            packets: List of FramePacket from different cameras

        Returns:
            List of ResultPacket for each camera
        """
        if not packets:
            return []

        results = []

        # Separate packets by whether they need detection
        detection_packets = []
        pose_only_packets = []

        for packet in packets:
            cam_id = packet.camera_id
            frame_num = packet.frame_num
            self.frame_counters[cam_id] = frame_num

            run_detection = (frame_num == 1) or (frame_num % self.detection_interval == 0)

            if run_detection:
                detection_packets.append(packet)
            else:
                pose_only_packets.append(packet)

        # 1. Batch YOLO detection
        if detection_packets:
            frames = [p.frame for p in detection_packets]

            # Use batch detection
            if hasattr(self.person_detector, 'detect_batch'):
                bboxes_list = self.person_detector.detect_batch(frames)
            else:
                # Fallback to sequential
                bboxes_list = [self.person_detector.detect(f) for f in frames]

            # Update cache
            for packet, bbox in zip(detection_packets, bboxes_list):
                cam_id = packet.camera_id
                self.cached_bboxes[cam_id] = [bbox] if bbox is not None else []
                if bbox is None:
                    self.cached_keypoints[cam_id] = None
                    self.cached_world_landmarks[cam_id] = None

        # 2. Batch RTMPose estimation
        # Collect all frames that need pose estimation
        pose_packets = []
        pose_frames = []
        pose_bboxes = []

        for packet in packets:
            cam_id = packet.camera_id
            frame_num = packet.frame_num

            run_pose = (frame_num == 1) or (frame_num % self.pose_interval == 0)
            bboxes = self.cached_bboxes[cam_id]

            if run_pose and bboxes and bboxes[0] is not None:
                bbox = bboxes[0]
                if isinstance(bbox, dict):
                    bbox_array = bbox.get('bbox')
                elif isinstance(bbox, (list, tuple, np.ndarray)):
                    bbox_array = np.array(bbox)
                else:
                    bbox_array = None

                if bbox_array is not None:
                    pose_packets.append(packet)
                    pose_frames.append(packet.frame)
                    pose_bboxes.append(bbox_array)

        # Run batch pose estimation
        if pose_packets:
            if hasattr(self.pose_estimator, 'infer_batch'):
                keypoints_list = self.pose_estimator.infer_batch(pose_frames, pose_bboxes)
            else:
                # Fallback to sequential
                keypoints_list = []
                for frame, bbox in zip(pose_frames, pose_bboxes):
                    kp = self.pose_estimator.estimate(frame, bbox)
                    keypoints_list.append(kp)

            # Update cache and apply smoothers
            for packet, keypoints in zip(pose_packets, keypoints_list):
                cam_id = packet.camera_id

                # Apply per-camera smoother
                if keypoints is not None and cam_id in self.smoothers:
                    keypoints = self.smoothers[cam_id].process(keypoints)

                self.cached_keypoints[cam_id] = keypoints

                if hasattr(self.pose_estimator, 'get_world_landmarks'):
                    self.cached_world_landmarks[cam_id] = self.pose_estimator.get_world_landmarks()

        # 3. Update state machines and create results
        for packet in packets:
            cam_id = packet.camera_id
            frame = packet.frame
            frame_num = packet.frame_num
            timestamp = packet.timestamp

            bboxes = self.cached_bboxes[cam_id]
            keypoints = self.cached_keypoints[cam_id]
            world_landmarks = self.cached_world_landmarks[cam_id]

            # State machine update
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

            states = [{
                'person_id': 0,
                'state': state_manager.current_state,
                'bbox': bbox_array,
                'keypoints': keypoints
            }]

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
            results.append(result)

        return results

    def stop(self):
        """Stop the batch inference service"""
        self.running.clear()
        print(f"[BatchInferenceService] Stopping...")

        # Cleanup RTMPose
        if hasattr(self, 'pose_estimator') and self.pose_estimator is not None:
            if hasattr(self.pose_estimator, 'cleanup'):
                try:
                    self.pose_estimator.cleanup()
                    print(f"[BatchInferenceService] RTMPose cleaned up")
                except Exception as e:
                    print(f"[BatchInferenceService] RTMPose cleanup warning: {e}")

        # Cleanup YOLO
        if hasattr(self, 'person_detector') and self.person_detector is not None:
            if hasattr(self.person_detector, 'cleanup'):
                try:
                    self.person_detector.cleanup()
                    print(f"[BatchInferenceService] YOLO cleaned up")
                except Exception as e:
                    print(f"[BatchInferenceService] YOLO cleanup warning: {e}")

    def get_stats(self) -> Dict:
        """Get service statistics"""
        return {
            'service_id': self.service_id,
            'camera_ids': self.camera_ids,
            'frame_counters': self.frame_counters.copy(),
            'detection_interval': self.detection_interval,
            'pose_interval': self.pose_interval,
        }
