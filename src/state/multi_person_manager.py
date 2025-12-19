"""
Multi-Person Manager
Manages multiple state machine instances, coordinates multi-person detection and tracking
"""

from typing import List, Dict, Optional
import numpy as np
from .behavior_state import BehaviorStateMachine, BehaviorEvent
from .roi_manager import ROIManager


class MultiPersonManager:
    """
    Multi-Person Manager

    Maintains independent state machine instances for each tracked person
    """

    def __init__(self, config: dict, roi_manager: ROIManager, event_logger):
        """
        Initialize multi-person manager

        Args:
            config: Configuration dictionary
            roi_manager: ROI manager instance
            event_logger: Event logger
        """
        self.config = config
        self.roi_manager = roi_manager
        self.event_logger = event_logger

        # Store state machine instance for each person: {tracking_id: BehaviorStateMachine}
        self.state_machines: Dict[int, BehaviorStateMachine] = {}

        # Store keypoints for each person: {tracking_id: keypoints}
        self.person_keypoints: Dict[int, np.ndarray] = {}

        # Store bbox for each person: {tracking_id: bbox}
        self.person_bboxes: Dict[int, np.ndarray] = {}

        # Tracking timeout config (seconds)
        self.tracking_timeout = config.get('behavior', {}).get('tracking_timeout', 5.0)

        # Record last update time for each person: {tracking_id: timestamp}
        self.last_update_times: Dict[int, float] = {}

        print(f"[MultiPersonManager] Initialization complete")
        print(f"[MultiPersonManager]   Tracking timeout: {self.tracking_timeout}s")

    def update_multi(
        self,
        detections: List[Dict],
        frame: np.ndarray,
        pose_estimator,
        current_time: float
    ) -> List[BehaviorEvent]:
        """
        Update multi-person states

        Args:
            detections: Detection results list, each dict contains {bbox, tracking_id, confidence}
            frame: Current frame (for pose estimation)
            pose_estimator: Pose estimator instance
            current_time: Current timestamp

        Returns:
            Event list for all persons
        """
        all_events = []

        # Set of tracking_ids detected in current frame
        current_tracking_ids = set()

        # Process each detected person
        for detection in detections:
            tracking_id = detection['tracking_id']
            bbox = detection['bbox']

            current_tracking_ids.add(tracking_id)

            # Create new state machine if this person is first time appearing
            if tracking_id not in self.state_machines:
                print(f"[MultiPersonManager] New tracking object: ID={tracking_id}")
                self.state_machines[tracking_id] = BehaviorStateMachine(
                    self.config,
                    self.roi_manager,
                    person_id=tracking_id  # Pass person_id for identification
                )

            # Update this person's last update time
            self.last_update_times[tracking_id] = current_time

            # Perform pose estimation
            keypoints = pose_estimator.estimate(frame, bbox)

            # Save keypoints and bbox for visualization
            self.person_keypoints[tracking_id] = keypoints
            self.person_bboxes[tracking_id] = bbox

            # Get 3D world coordinates (if supported)
            world_landmarks = None
            if hasattr(pose_estimator, 'get_world_landmarks'):
                world_landmarks = pose_estimator.get_world_landmarks()

            # Update this person's state machine
            events = self.state_machines[tracking_id].update(
                bbox, keypoints, current_time, world_landmarks
            )

            # Add tracking_id info to each event
            for event in events:
                event.tracking_id = tracking_id

            all_events.extend(events)

        # Clean up persons not detected for long time (timeout)
        self._cleanup_inactive_trackers(current_time, current_tracking_ids)

        return all_events

    def _cleanup_inactive_trackers(
        self,
        current_time: float,
        current_tracking_ids: set
    ):
        """
        Clean up tracking objects not detected for long time

        Args:
            current_time: Current time
            current_tracking_ids: Set of tracking_ids detected in current frame
        """
        inactive_ids = []

        for tracking_id, last_time in self.last_update_times.items():
            # If this person not detected in current frame and exceeds timeout
            if tracking_id not in current_tracking_ids:
                if current_time - last_time > self.tracking_timeout:
                    inactive_ids.append(tracking_id)

        # Remove inactive tracking objects
        for tracking_id in inactive_ids:
            print(f"[MultiPersonManager] Removing inactive tracking object: ID={tracking_id}")
            del self.state_machines[tracking_id]
            del self.last_update_times[tracking_id]
            if tracking_id in self.person_keypoints:
                del self.person_keypoints[tracking_id]
            if tracking_id in self.person_bboxes:
                del self.person_bboxes[tracking_id]

    def get_active_person_count(self) -> int:
        """Get current number of active persons"""
        return len(self.state_machines)

    def get_person_state(self, tracking_id: int) -> Optional[str]:
        """
        Get current state of specified person

        Args:
            tracking_id: Tracking ID

        Returns:
            State string, or None if not exists
        """
        if tracking_id in self.state_machines:
            return self.state_machines[tracking_id].current_state.value
        return None

    def get_all_states(self) -> Dict[int, str]:
        """
        Get current states of all persons

        Returns:
            {tracking_id: state_string}
        """
        return {
            tracking_id: sm.current_state.value
            for tracking_id, sm in self.state_machines.items()
        }

    def reset(self):
        """Reset all state machines"""
        self.state_machines.clear()
        self.last_update_times.clear()
        self.person_keypoints.clear()
        self.person_bboxes.clear()
        print(f"[MultiPersonManager] All tracking objects reset")
