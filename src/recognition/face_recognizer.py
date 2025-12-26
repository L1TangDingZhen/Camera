"""
Face Recognizer using InsightFace (ArcFace)
High-precision GPU-accelerated face recognition
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
from pathlib import Path


class FaceRecognizer:
    """
    Face Recognizer using InsightFace

    Features:
    - GPU-accelerated face detection and recognition
    - ArcFace embeddings (512-dim)
    - High accuracy face matching
    """

    def __init__(self, config: dict):
        """
        Initialize face recognizer

        Args:
            config: Configuration dict with keys:
                - device: 'cuda:0' or 'cpu'
                - det_size: Detection input size (default: 640)
                - confidence: Detection confidence threshold (default: 0.5)
        """
        self.config = config
        self.device = config.get('device', 'cuda:0')
        self.det_size = config.get('det_size', 640)
        self.confidence_threshold = config.get('confidence', 0.5)

        # Initialize InsightFace
        try:
            from insightface.app import FaceAnalysis

            print(f"[FaceRecognizer] Initializing InsightFace...")
            print(f"[FaceRecognizer]   Device: {self.device}")
            print(f"[FaceRecognizer]   Detection size: {self.det_size}")

            # Create face analysis app
            self.app = FaceAnalysis(
                name='buffalo_l',  # High-accuracy model
                providers=['CUDAExecutionProvider'] if 'cuda' in self.device else ['CPUExecutionProvider']
            )

            # Prepare with detection size
            ctx_id = 0 if 'cuda' in self.device else -1
            self.app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))

            print(f"[FaceRecognizer] InsightFace initialized successfully ✓")

        except ImportError:
            raise ImportError(
                "InsightFace not installed!\n\n"
                "Install with:\n"
                "  pip install insightface onnxruntime-gpu\n\n"
                "Or CPU version:\n"
                "  pip install insightface onnxruntime\n"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize InsightFace: {e}")

    def extract_face_embedding(
        self,
        image: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Extract face embedding from image

        Args:
            image: Input image (H, W, 3) in BGR format
            bbox: Optional person bounding box [x1, y1, x2, y2, score]
                  If provided, will crop to this region first

        Returns:
            Tuple of (embedding, face_bbox, confidence) or None if no face detected
            - embedding: 512-dim face feature vector
            - face_bbox: [x1, y1, x2, y2] in original image coordinates
            - confidence: Detection confidence score
        """
        # Crop to person bbox if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            # Expand bbox slightly for better face detection
            h, w = image.shape[:2]
            expand = 0.1  # 10% expansion
            dx = int((x2 - x1) * expand)
            dy = int((y2 - y1) * expand)

            x1 = max(0, x1 - dx)
            y1 = max(0, y1 - dy)
            x2 = min(w, x2 + dx)
            y2 = min(h, y2 + dy)

            person_img = image[y1:y2, x1:x2]
            bbox_offset = np.array([x1, y1])
        else:
            person_img = image
            bbox_offset = np.array([0, 0])

        # Detect faces
        faces = self.app.get(person_img)

        if len(faces) == 0:
            return None

        # Use largest face (by bbox area)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Get embedding (512-dim ArcFace feature)
        embedding = face.normed_embedding

        # Get face bbox in original image coordinates
        face_bbox = face.bbox + np.array([bbox_offset[0], bbox_offset[1], bbox_offset[0], bbox_offset[1]])

        # Confidence score
        confidence = face.det_score

        return embedding, face_bbox, confidence

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two face embeddings

        Args:
            embedding1: First face embedding (512-dim)
            embedding2: Second face embedding (512-dim)

        Returns:
            Similarity score in [0, 1], higher is more similar
        """
        # Cosine similarity (embeddings are already normalized)
        similarity = np.dot(embedding1, embedding2)

        # Convert to [0, 1] range
        similarity = (similarity + 1) / 2

        return float(similarity)

    def match_face(
        self,
        embedding: np.ndarray,
        known_embeddings: List[np.ndarray],
        threshold: float = 0.5
    ) -> Tuple[int, float]:
        """
        Match face embedding against known face database

        Args:
            embedding: Query face embedding
            known_embeddings: List of known face embeddings
            threshold: Similarity threshold for positive match (default: 0.5)

        Returns:
            Tuple of (matched_index, similarity_score)
            - matched_index: Index in known_embeddings list, or -1 if no match
            - similarity_score: Best similarity score
        """
        if len(known_embeddings) == 0:
            return -1, 0.0

        # Compute similarities with all known faces
        similarities = [self.compute_similarity(embedding, known_emb) for known_emb in known_embeddings]

        # Find best match
        best_idx = int(np.argmax(similarities))
        best_similarity = similarities[best_idx]

        # Check if above threshold
        if best_similarity >= threshold:
            return best_idx, best_similarity
        else:
            return -1, best_similarity

    def cleanup(self):
        """Cleanup resources"""
        # InsightFace handles cleanup automatically
        pass
