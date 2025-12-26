#!/usr/bin/env python3
"""
Face Registration Tool
Manually register known persons into the face database
"""

import cv2
import numpy as np
import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recognition import FaceRecognizer, FaceDatabase


class FaceRegistrationTool:
    """
    Face Registration Tool

    Supports two modes:
    1. Webcam mode: Capture face from webcam
    2. Image mode: Load face from image file
    """

    def __init__(self, config_path: str = "config/config_gpu.yaml"):
        """
        Initialize registration tool

        Args:
            config_path: Path to configuration file
        """
        print("=" * 60)
        print("Face Registration Tool")
        print("=" * 60)

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        face_config = config.get('face_recognition', {})

        # Initialize face recognizer
        print("\n[1/3] Initializing face recognizer...")
        self.face_recognizer = FaceRecognizer(face_config)

        # Initialize face database
        print("\n[2/3] Loading face database...")
        db_path = face_config.get('database_path', 'data/faces.db')
        self.face_database = FaceDatabase(db_path)

        # Print database statistics
        stats = self.face_database.get_statistics()
        print(f"\n[3/3] Database statistics:")
        print(f"  Total persons: {stats['total_persons']}")
        print(f"  Registered: {stats['registered_persons']}")
        print(f"  Strangers: {stats['strangers']}")

        print("\n" + "=" * 60)
        print("Initialization complete ✓")
        print("=" * 60 + "\n")

    def register_from_webcam(self, name: str, notes: str = ""):
        """
        Register face from webcam capture

        Args:
            name: Person's name
            notes: Optional notes
        """
        print(f"\n[Register] Name: '{name}'")
        print("[Webcam] Opening camera...")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[Error] Failed to open webcam!")
            return

        print("\n" + "=" * 60)
        print("WEBCAM CAPTURE MODE")
        print("=" * 60)
        print("Instructions:")
        print("  1. Position your face in the center of the frame")
        print("  2. Ensure good lighting (face clearly visible)")
        print("  3. Press SPACE to capture")
        print("  4. Press ESC to cancel")
        print("=" * 60 + "\n")

        embedding = None
        captured_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to read frame from webcam!")
                break

            # Display frame
            display_frame = frame.copy()

            # Try to detect face in real-time
            result = self.face_recognizer.extract_face_embedding(frame)

            if result is not None:
                _, face_bbox, confidence = result

                # Draw face bbox
                x1, y1, x2, y2 = face_bbox.astype(int)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display confidence
                text = f"Confidence: {confidence:.2f}"
                cv2.putText(display_frame, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display instruction
                cv2.putText(display_frame, "Press SPACE to capture", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # No face detected
                cv2.putText(display_frame, "No face detected", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Face Registration - Webcam", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("\n[Cancel] Registration cancelled")
                break
            elif key == ord(' '):  # SPACE
                # Capture frame
                print("\n[Capture] Extracting face embedding...")
                result = self.face_recognizer.extract_face_embedding(frame)

                if result is None:
                    print("[Error] No face detected in captured frame!")
                    print("[Retry] Please try again")
                    continue

                embedding, face_bbox, confidence = result
                captured_frame = frame.copy()

                print(f"[Success] Face detected (confidence: {confidence:.3f})")
                print(f"[Embedding] Shape: {embedding.shape}, Norm: {np.linalg.norm(embedding):.3f}")

                # Draw bbox on captured frame
                x1, y1, x2, y2 = face_bbox.astype(int)
                cv2.rectangle(captured_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(captured_frame, f"{name}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Show captured frame
                cv2.imshow("Captured Face", captured_frame)
                print("\n[Confirm] Is this face correct?")
                print("  Press ENTER to confirm and save")
                print("  Press ESC to retry")

                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == 13:  # ENTER
                        break
                    elif key2 == 27:  # ESC
                        embedding = None
                        cv2.destroyWindow("Captured Face")
                        print("[Retry] Capture cancelled, please try again")
                        break

                if embedding is not None:
                    break

        cap.release()
        cv2.destroyAllWindows()

        if embedding is not None:
            # Save to database
            print("\n[Database] Saving to database...")
            person_id = self.face_database.register_person(name, embedding, notes)
            print(f"[Success] ✓ Registered: '{name}' (ID={person_id})")

            # Save captured image
            image_dir = Path("data/registered_faces")
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"person_{person_id}_{name}.jpg"
            cv2.imwrite(str(image_path), captured_frame)
            print(f"[Saved] Image: {image_path}")
        else:
            print("\n[Failed] Registration not completed")

    def register_from_image(self, name: str, image_path: str, notes: str = ""):
        """
        Register face from image file

        Args:
            name: Person's name
            image_path: Path to image file
            notes: Optional notes
        """
        print(f"\n[Register] Name: '{name}'")
        print(f"[Image] Loading: {image_path}")

        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[Error] Failed to load image: {image_path}")
            return

        print(f"[Image] Size: {frame.shape[1]}x{frame.shape[0]}")

        # Extract face embedding
        print("[Processing] Extracting face embedding...")
        result = self.face_recognizer.extract_face_embedding(frame)

        if result is None:
            print("[Error] No face detected in image!")
            return

        embedding, face_bbox, confidence = result

        print(f"[Success] Face detected (confidence: {confidence:.3f})")
        print(f"[Embedding] Shape: {embedding.shape}, Norm: {np.linalg.norm(embedding):.3f}")

        # Draw bbox
        display_frame = frame.copy()
        x1, y1, x2, y2 = face_bbox.astype(int)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"{name}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show detected face
        cv2.imshow("Detected Face", display_frame)
        print("\n[Confirm] Is this face correct?")
        print("  Press ENTER to confirm and save")
        print("  Press ESC to cancel")

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == 13:  # ENTER
            # Save to database
            print("\n[Database] Saving to database...")
            person_id = self.face_database.register_person(name, embedding, notes)
            print(f"[Success] ✓ Registered: '{name}' (ID={person_id})")

            # Copy image to registered faces directory
            image_dir = Path("data/registered_faces")
            image_dir.mkdir(parents=True, exist_ok=True)
            dest_path = image_dir / f"person_{person_id}_{name}.jpg"
            cv2.imwrite(str(dest_path), display_frame)
            print(f"[Saved] Image: {dest_path}")
        else:
            print("\n[Cancel] Registration cancelled")

    def list_registered_persons(self):
        """List all registered persons"""
        persons = self.face_database.get_all_persons()

        print("\n" + "=" * 80)
        print("REGISTERED PERSONS")
        print("=" * 80)

        if len(persons) == 0:
            print("No persons registered yet.")
        else:
            registered = [p for p in persons if p['is_registered']]
            strangers = [p for p in persons if not p['is_registered']]

            if len(registered) > 0:
                print("\n【Known Persons】")
                print("-" * 80)
                for p in registered:
                    print(f"ID={p['person_id']:3d} | Name: {p['name']:20s} | "
                          f"Appearances: {p['appearance_count']:3d} | "
                          f"Last seen: {p['last_seen']}")

            if len(strangers) > 0:
                print("\n【Strangers (Auto-detected)】")
                print("-" * 80)
                for p in strangers:
                    print(f"ID={p['person_id']:3d} | Name: {p['name']:20s} | "
                          f"Appearances: {p['appearance_count']:3d} | "
                          f"Last seen: {p['last_seen']}")

        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Face Registration Tool for Life Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register from webcam
  python scripts/register_face.py --name "张三" --notes "家人"

  # Register from image file
  python scripts/register_face.py --name "李四" --image path/to/photo.jpg

  # List all registered persons
  python scripts/register_face.py --list
        """
    )

    parser.add_argument('--name', type=str, help='Person name to register')
    parser.add_argument('--image', type=str, help='Path to image file (if not provided, use webcam)')
    parser.add_argument('--notes', type=str, default='', help='Optional notes about this person')
    parser.add_argument('--config', type=str, default='config/config_gpu.yaml',
                       help='Path to config file')
    parser.add_argument('--list', action='store_true', help='List all registered persons')

    args = parser.parse_args()

    # Create tool
    tool = FaceRegistrationTool(config_path=args.config)

    if args.list:
        # List mode
        tool.list_registered_persons()
    elif args.name:
        # Registration mode
        if args.image:
            # Register from image
            tool.register_from_image(args.name, args.image, args.notes)
        else:
            # Register from webcam
            tool.register_from_webcam(args.name, args.notes)

        # Show updated list
        tool.list_registered_persons()
    else:
        parser.print_help()
        print("\n[Error] Please specify --name or --list")


if __name__ == "__main__":
    main()
