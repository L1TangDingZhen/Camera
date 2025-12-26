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
    Includes face recognition for person identification
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

        # Face recognition config
        face_config = config.get('face_recognition', {})
        self.face_recognition_enabled = face_config.get('enabled', False)

        if self.face_recognition_enabled:
            # Initialize face recognizer and database
            from src.recognition import FaceRecognizer, FaceDatabase

            self.face_recognizer = FaceRecognizer(face_config)
            self.face_database = FaceDatabase(face_config.get('database_path', 'data/faces.db'))

            # Mapping: tracking_id -> person_id (from face database)
            self.tracking_to_person: Dict[int, Optional[int]] = {}

            # Mapping: tracking_id -> person_name (for display)
            self.person_names: Dict[int, str] = {}

            # Record last face recognition time for each tracking_id
            self.last_face_recognition_time: Dict[int, float] = {}

            # Face recognition interval (seconds)
            self.recognition_interval = face_config.get('recognition_interval', 5.0)

            # Face matching threshold
            self.match_threshold = face_config.get('match_threshold', 0.5)

            print(f"[MultiPersonManager] Face recognition ENABLED")
            print(f"[MultiPersonManager]   Recognition interval: {self.recognition_interval}s")
            print(f"[MultiPersonManager]   Match threshold: {self.match_threshold}")

            # Print database statistics
            stats = self.face_database.get_statistics()
            print(f"[MultiPersonManager]   Database: {stats['total_persons']} persons "
                  f"({stats['registered_persons']} registered, {stats['strangers']} strangers)")
        else:
            self.face_recognizer = None
            self.face_database = None
            print(f"[MultiPersonManager] Face recognition DISABLED")

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
            is_new_tracking = tracking_id not in self.state_machines

            if is_new_tracking:
                print(f"[MultiPersonManager] New tracking object: ID={tracking_id}")
                self.state_machines[tracking_id] = BehaviorStateMachine(
                    self.config,
                    self.roi_manager,
                    person_id=tracking_id  # Pass person_id for identification
                )

            # Update this person's last update time
            self.last_update_times[tracking_id] = current_time

            # Face recognition (if enabled)
            if self.face_recognition_enabled:
                # Perform face recognition if:
                # 1. First time seeing this tracking_id, OR
                # 2. Enough time has passed since last recognition
                should_recognize = (
                    is_new_tracking or
                    tracking_id not in self.last_face_recognition_time or
                    current_time - self.last_face_recognition_time.get(tracking_id, 0) > self.recognition_interval
                )

                if should_recognize:
                    self._recognize_face(tracking_id, frame, bbox, current_time)

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

    def _recognize_face(self, tracking_id: int, frame: np.ndarray, bbox: np.ndarray, current_time: float):
        """
        Perform face recognition for a tracking object

        Args:
            tracking_id: Tracking ID
            frame: Current frame
            bbox: Person bounding box
            current_time: Current timestamp
        """
        # Extract face embedding
        result = self.face_recognizer.extract_face_embedding(frame, bbox)

        if result is None:
            # No face detected (e.g., person facing away, poor lighting)
            # Keep previous name if already recognized, otherwise use tracking ID
            if tracking_id not in self.person_names:
                self.person_names[tracking_id] = f"T{tracking_id}"
            # If already recognized, keep the existing name (don't overwrite)
            print(f"[MultiPersonManager] T{tracking_id}: No face detected, keeping name '{self.person_names.get(tracking_id, 'unknown')}'")
            return

        embedding, face_bbox, confidence = result

        # Get all known persons from database
        known_persons = self.face_database.get_all_persons()

        if len(known_persons) == 0:
            # No known persons, create new stranger
            person_id = self.face_database.add_stranger(embedding)
            self.tracking_to_person[tracking_id] = person_id
            person = self.face_database.get_person(person_id)
            self.person_names[tracking_id] = person['name']
            print(f"[MultiPersonManager] T{tracking_id} -> New person: {person['name']} (ID={person_id})")
        else:
            # Try to match against known persons
            known_embeddings = [p['embedding'] for p in known_persons]
            matched_idx, similarity = self.face_recognizer.match_face(
                embedding, known_embeddings, threshold=self.match_threshold
            )

            if matched_idx >= 0:
                # Matched existing person
                person = known_persons[matched_idx]
                person_id = person['person_id']
                self.tracking_to_person[tracking_id] = person_id
                self.person_names[tracking_id] = person['name']

                # Update last_seen in database
                self.face_database.update_last_seen(person_id, similarity, camera_id=0)

                print(f"[MultiPersonManager] T{tracking_id} -> Matched: {person['name']} "
                      f"(ID={person_id}, similarity={similarity:.3f})")
            else:
                # No match, create new stranger
                person_id = self.face_database.add_stranger(embedding)
                self.tracking_to_person[tracking_id] = person_id
                person = self.face_database.get_person(person_id)
                self.person_names[tracking_id] = person['name']
                print(f"[MultiPersonManager] T{tracking_id} -> New person: {person['name']} "
                      f"(ID={person_id}, best_similarity={similarity:.3f})")

        # Update last recognition time
        self.last_face_recognition_time[tracking_id] = current_time

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

            # Cleanup face recognition data
            if self.face_recognition_enabled:
                if tracking_id in self.tracking_to_person:
                    del self.tracking_to_person[tracking_id]
                if tracking_id in self.person_names:
                    del self.person_names[tracking_id]
                if tracking_id in self.last_face_recognition_time:
                    del self.last_face_recognition_time[tracking_id]

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

    def get_person_name(self, tracking_id: int) -> str:
        """
        Get person name for display

        Args:
            tracking_id: Tracking ID

        Returns:
            Person name string
        """
        if self.face_recognition_enabled and tracking_id in self.person_names:
            return self.person_names[tracking_id]
        else:
            return f"Person {tracking_id}"

    def get_person_info(self, tracking_id: int) -> Optional[Dict]:
        """
        Get full person information

        Args:
            tracking_id: Tracking ID

        Returns:
            Dict with keys: person_id, name, is_registered, or None if not found
        """
        if not self.face_recognition_enabled:
            return None

        person_id = self.tracking_to_person.get(tracking_id)
        if person_id is None:
            return None

        person = self.face_database.get_person(person_id)
        return person

    def reset(self):
        """Reset all state machines"""
        self.state_machines.clear()
        self.last_update_times.clear()
        self.person_keypoints.clear()
        self.person_bboxes.clear()

        if self.face_recognition_enabled:
            self.tracking_to_person.clear()
            self.person_names.clear()
            self.last_face_recognition_time.clear()

        print(f"[MultiPersonManager] All tracking objects reset")
