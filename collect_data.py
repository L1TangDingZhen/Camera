#!/usr/bin/env python3
"""
Data Collection Tool - Collect sitting/standing/lying pose training data

Supports both MediaPipe (CPU) and RTMPose (GPU) backends

Usage:
    # Use RTMPose (default, recommended)
    python collect_data.py

    # Use MediaPipe
    python collect_data.py --backend mediapipe

    # Custom countdown and duration
    python collect_data.py --countdown 5 --min-duration 120

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
from datetime import datetime
from typing import Optional, List, Tuple
from pathlib import Path


class DataCollector:
    """Pose data collector supporting both MediaPipe and RTMPose"""

    def __init__(self, backend: str = 'rtmpose',
                 config_path: str = 'config/config_gpu.yaml',
                 countdown: int = 3,
                 min_duration: int = 120):
        """
        Args:
            backend: 'mediapipe' or 'rtmpose'
            config_path: Path to config file (for RTMPose)
            countdown: Countdown seconds after key press
            min_duration: Recommended minimum recording duration (seconds)
        """
        self.backend = backend
        self.countdown = countdown
        self.min_duration = min_duration

        # Recording state
        self.is_recording = False
        self.current_pose_label = None
        self.record_start_time = None
        self.collected_samples = []

        # Sequence buffer for LSTM/Transformer
        self.sequence_buffer = []
        self.sequence_length = 10

        # Data storage
        self.output_dir = "training_data"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize pose estimator based on backend
        if backend == 'mediapipe':
            self._init_mediapipe()
        elif backend == 'rtmpose':
            self._init_rtmpose(config_path)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'mediapipe' or 'rtmpose'")

        print(f"[INFO] Using backend: {backend}")

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

    def _init_rtmpose(self, config_path: str):
        """Initialize RTMPose"""
        from src.detectors import PoseEstimatorFactory
        from src.detectors.person_detector import PersonDetector

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("[INFO] Initializing YOLOv8 person detector...")
        self.person_detector = PersonDetector(self.config)

        print("[INFO] Initializing RTMPose pose estimator...")
        self.pose_estimator = PoseEstimatorFactory.create(self.config['models']['pose'])

    def process_frame(self, frame):
        """Process frame and extract keypoints

        Returns:
            keypoints: (17, 4) [x, y, z, confidence] or None
            annotated_frame: Frame with keypoints drawn
        """
        if self.backend == 'mediapipe':
            return self._process_mediapipe(frame)
        else:
            return self._process_rtmpose(frame)

    def _process_mediapipe(self, frame):
        """Process frame with MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_world_landmarks:
            return None, frame

        # Convert to COCO-17 format
        world_landmarks = np.zeros((17, 4), dtype=np.float32)

        for mp_idx, coco_idx in self.MEDIAPIPE_TO_COCO.items():
            wl = results.pose_world_landmarks.landmark[mp_idx]
            world_landmarks[coco_idx] = [wl.x, wl.y, wl.z, wl.visibility]

        # Draw landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return world_landmarks, frame

    def _process_rtmpose(self, frame):
        """Process frame with RTMPose"""
        detections = self.person_detector.detect(frame)

        if len(detections) == 0:
            return None, frame

        # Get first person
        bbox = detections[0]
        keypoints = self.pose_estimator.estimate(frame, bbox)

        if keypoints is None or len(keypoints) == 0:
            return None, frame

        # Draw bbox
        x1, y1, x2, y2, conf = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw keypoints
        for i, kpt in enumerate(keypoints):
            x, y, z, confidence = kpt
            if confidence > 0.3:
                px = int(x * frame.shape[1])
                py = int(y * frame.shape[0])
                cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

        return keypoints, frame

    def run(self):
        """Run data collection"""
        cap = cv2.VideoCapture(0)
        # Force MJPEG encoding for better bandwidth efficiency
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\n[INFO] Data collection tool started")
        print(f"[INFO] Backend: {self.backend}")
        print("[INFO] Key controls:")
        print("  's' - Start recording Sitting")
        print("  't' - Start recording Standing")
        print("  'l' - Start recording Lying")
        print("  'q' - Stop current recording")
        print("  'ESC' - Exit program\n")

        countdown_start_time = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()

            # Countdown phase
            if countdown_start_time is not None and not self.is_recording:
                elapsed = current_time - countdown_start_time
                remaining = self.countdown - int(elapsed)

                if remaining > 0:
                    frame = self.draw_countdown(frame, remaining)
                else:
                    self.is_recording = True
                    self.record_start_time = current_time
                    countdown_start_time = None
                    print(f"[INFO] Started recording {self.current_pose_label}...")

            # Recording phase
            elif self.is_recording:
                keypoints, frame = self.process_frame(frame)

                if keypoints is not None:
                    # Save as 68-dim flattened features (17 x 4 = 68)
                    features = keypoints.flatten()

                    # Update sequence buffer
                    self.sequence_buffer.append(features)
                    if len(self.sequence_buffer) > self.sequence_length:
                        self.sequence_buffer.pop(0)

                    # Save sample
                    sample = {
                        'features': features.tolist(),  # 68-dim
                        'label': self.current_pose_label,
                        'timestamp': current_time
                    }

                    # Add sequence data if buffer is full
                    if len(self.sequence_buffer) == self.sequence_length:
                        sample['features_sequence'] = [f.tolist() for f in self.sequence_buffer]

                    self.collected_samples.append(sample)

                # Display recording status
                duration = int(current_time - self.record_start_time)
                frame = self.draw_recording_info(frame, duration)

            # Idle phase
            else:
                frame = self.draw_instructions(frame)

            # Display
            cv2.imshow(f'Data Collection Tool ({self.backend})', frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                if self.is_recording:
                    self.stop_recording()
                break

            elif key == ord('s') and not self.is_recording and countdown_start_time is None:
                self.start_recording('sitting')
                countdown_start_time = current_time

            elif key == ord('t') and not self.is_recording and countdown_start_time is None:
                self.start_recording('standing')
                countdown_start_time = current_time

            elif key == ord('l') and not self.is_recording and countdown_start_time is None:
                self.start_recording('lying')
                countdown_start_time = current_time

            elif key == ord('q'):
                if countdown_start_time is not None:
                    countdown_start_time = None
                    self.current_pose_label = None
                    print("[INFO] Recording cancelled")
                elif self.is_recording:
                    self.stop_recording()

        cap.release()
        cv2.destroyAllWindows()

        if self.backend == 'mediapipe':
            self.pose.close()

        print("\n[INFO] Data collection completed")
        print(f"[INFO] Data saved to: {self.output_dir}/")

    def start_recording(self, pose_label: str):
        """Start recording a pose"""
        self.current_pose_label = pose_label
        self.record_start_time = None
        self.collected_samples = []
        self.sequence_buffer = []
        print(f"\n[INFO] Preparing to record {pose_label}, countdown {self.countdown} seconds...")

    def stop_recording(self):
        """Stop recording and save"""
        if not self.is_recording:
            return

        self.is_recording = False
        self.sequence_buffer = []

        if len(self.collected_samples) == 0:
            print("[WARN] No valid samples collected")
            return

        # Save to file
        filepath = os.path.join(self.output_dir, f"{self.current_pose_label}_samples.json")

        # Load existing data
        existing_data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []

        # Merge data
        existing_data.extend(self.collected_samples)

        # Save
        with open(filepath, 'w') as f:
            json.dump(existing_data, f)

        print(f"[INFO] Saved {len(self.collected_samples)} samples to {filepath}")
        print(f"[INFO] Total {self.current_pose_label} samples: {len(existing_data)}")

    def draw_countdown(self, frame, remaining):
        """Draw countdown overlay"""
        h, w = frame.shape[:2]
        text = f"Recording {self.current_pose_label} starts in: {remaining}"
        cv2.putText(frame, text, (50, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                   1.5, (0, 255, 255), 3)
        return frame

    def draw_recording_info(self, frame, duration):
        """Draw recording information"""
        h, w = frame.shape[:2]

        text1 = f"Recording: {self.current_pose_label}"
        cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 255), 2)

        text2 = f"Duration: {duration}s (recommended {self.min_duration}s+)"
        cv2.putText(frame, text2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 0), 2)

        text3 = f"Collected: {len(self.collected_samples)} frames"
        cv2.putText(frame, text3, (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 0), 2)

        text4 = "Press 'q' to stop recording"
        cv2.putText(frame, text4, (20, h-30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        return frame

    def draw_instructions(self, frame):
        """Draw idle instructions"""
        h, w = frame.shape[:2]

        instructions = [
            f"Data Collection Tool ({self.backend})",
            "",
            "Key Controls:",
            "  's' - Record Sitting",
            "  't' - Record Standing",
            "  'l' - Record Lying",
            "  'ESC' - Exit",
            "",
            "Collected Data:",
            f"  Sitting: {self.get_sample_count('sitting')} samples",
            f"  Standing: {self.get_sample_count('standing')} samples",
            f"  Lying: {self.get_sample_count('lying')} samples",
        ]

        y = 50
        for line in instructions:
            cv2.putText(frame, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
            y += 35

        return frame

    def get_sample_count(self, pose_label: str) -> int:
        """Get sample count for a pose"""
        filepath = os.path.join(self.output_dir, f"{pose_label}_samples.json")
        if not os.path.exists(filepath):
            return 0

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return len(data)
        except:
            return 0


def main():
    parser = argparse.ArgumentParser(description='Pose Data Collection Tool')
    parser.add_argument('--backend', type=str, default='rtmpose',
                       choices=['mediapipe', 'rtmpose'],
                       help='Pose estimation backend (default: rtmpose)')
    parser.add_argument('--config', type=str, default='config/config_gpu.yaml',
                       help='Config file path (for RTMPose)')
    parser.add_argument('--countdown', type=int, default=3,
                       help='Countdown seconds before recording')
    parser.add_argument('--min-duration', type=int, default=120,
                       help='Recommended minimum recording duration (seconds)')

    args = parser.parse_args()

    collector = DataCollector(
        backend=args.backend,
        config_path=args.config,
        countdown=args.countdown,
        min_duration=args.min_duration
    )

    collector.run()


if __name__ == '__main__':
    main()
