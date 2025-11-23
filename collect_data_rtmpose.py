#!/usr/bin/env python3
"""
Data Collection Tool - Using RTMPose to collect sitting/standing/lying training data

Usage:
    python collect_data_rtmpose.py --countdown 5 --min-duration 30

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

# Import RTMPose and detector
from src.detectors import PoseEstimatorFactory
from src.detectors.person_detector import PersonDetector


class DataCollectorRTMPose:
    """Pose data collector using RTMPose"""

    def __init__(self, config_path: str = "config/config_gpu.yaml",
                 countdown: int = 5, min_duration: int = 30):
        """
        Args:
            config_path: Path to config file
            countdown: Countdown seconds after key press
            min_duration: Recommended minimum recording duration (seconds)
        """
        self.countdown = countdown
        self.min_duration = min_duration

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize person detector
        print("[INFO] Initializing YOLOv8 person detector...")
        self.person_detector = PersonDetector(self.config)

        # Initialize RTMPose
        print("[INFO] Initializing RTMPose pose estimator...")
        self.pose_estimator = PoseEstimatorFactory.create(self.config['models']['pose'])

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

    def run(self):
        """Run data collection"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\n[INFO] Data collection tool started")
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
                # Detect person
                detections = self.person_detector.detect(frame)

                if len(detections) > 0:
                    # Get first person
                    bbox = detections[0]

                    # Estimate pose
                    keypoints = self.pose_estimator.estimate(frame, bbox)

                    if keypoints is not None and len(keypoints) > 0:
                        # RTMPose outputs COCO-17 format 3D coordinates
                        world_landmarks = keypoints  # (17, 4) [x, y, z, confidence]

                        # Save as 68-dim flattened features
                        features = world_landmarks.flatten()  # 17 x 4 = 68 dim

                        if features is not None:
                            # Update sequence buffer
                            self.sequence_buffer.append(features)
                            if len(self.sequence_buffer) > self.sequence_length:
                                self.sequence_buffer.pop(0)

                            # Save sample (both single frame and sequence)
                            sample = {
                                'features': features.tolist(),  # Single frame 68-dim
                                'label': self.current_pose_label,
                                'timestamp': current_time
                            }

                            # Add sequence data if buffer is full
                            if len(self.sequence_buffer) == self.sequence_length:
                                sample['features_sequence'] = [f.tolist() for f in self.sequence_buffer]

                            self.collected_samples.append(sample)

                        # Draw keypoints
                        frame = self.draw_keypoints(frame, keypoints, bbox)

                # Display recording status
                duration = int(current_time - self.record_start_time)
                frame = self.draw_recording_info(frame, duration)

            # Idle phase
            else:
                frame = self.draw_instructions(frame)

            # Display
            cv2.imshow('Data Collection Tool (RTMPose)', frame)

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

    def draw_keypoints(self, frame, keypoints, bbox):
        """Draw keypoints on frame"""
        x1, y1, x2, y2, conf = bbox

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw keypoints
        for i, kpt in enumerate(keypoints):
            x, y, z, confidence = kpt
            if confidence > 0.3:
                # Convert to image coordinates
                px = int(x * frame.shape[1])
                py = int(y * frame.shape[0])
                cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

        return frame

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

        # Recording status
        text1 = f"Recording: {self.current_pose_label}"
        cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 255), 2)

        # Duration
        text2 = f"Duration: {duration}s (recommended {self.min_duration}s+)"
        cv2.putText(frame, text2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 0), 2)

        # Sample count
        text3 = f"Collected: {len(self.collected_samples)} frames"
        cv2.putText(frame, text3, (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 0), 2)

        # Stop instruction
        text4 = "Press 'q' to stop recording"
        cv2.putText(frame, text4, (20, h-30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        return frame

    def draw_instructions(self, frame):
        """Draw idle instructions"""
        h, w = frame.shape[:2]

        instructions = [
            "Data Collection Tool (RTMPose)",
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
    parser = argparse.ArgumentParser(description='Pose Data Collection Tool (RTMPose)')
    parser.add_argument('--config', type=str, default='config/config_gpu.yaml',
                       help='Path to config file')
    parser.add_argument('--countdown', type=int, default=3,
                       help='Countdown seconds before recording')
    parser.add_argument('--min-duration', type=int, default=120,
                       help='Recommended minimum recording duration (seconds)')

    args = parser.parse_args()

    collector = DataCollectorRTMPose(
        config_path=args.config,
        countdown=args.countdown,
        min_duration=args.min_duration
    )

    collector.run()


if __name__ == '__main__':
    main()
