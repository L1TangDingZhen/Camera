"""
Test TensorRT wrapper directly to isolate the issue
"""

import numpy as np
import cv2
from src.detectors.tensorrt_wrapper import TensorRTRTMPose

print("[1/4] Loading TensorRT engine...")
try:
    trt_model = TensorRTRTMPose(
        engine_path='models/rtmpose/rtmpose-s.engine',
        device='cuda:0'
    )
    print("  ✓ Engine loaded successfully")
except Exception as e:
    print(f"  ✗ Failed to load engine: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[2/4] Creating test image...")
# Create a dummy image (BGR format)
test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
test_bbox = np.array([100, 100, 300, 400, 0.9])  # x1, y1, x2, y2, score

print(f"  Image shape: {test_img.shape}")
print(f"  Image dtype: {test_img.dtype}")
print(f"  BBox: {test_bbox}")

print("\n[3/4] Running inference...")
try:
    keypoints = trt_model(test_img, test_bbox)
    print(f"  ✓ Inference succeeded!")
    print(f"  Keypoints shape: {keypoints.shape}")
    print(f"  Keypoints dtype: {keypoints.dtype}")
    print(f"  Keypoints range: x=[{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}], y=[{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")
    print(f"  Confidence range: [{keypoints[:, 2].min():.3f}, {keypoints[:, 2].max():.3f}]")

    # Check if keypoints look reasonable
    if keypoints.shape == (17, 3):
        print("  ✓ Keypoints have correct shape (17, 3)")
    else:
        print(f"  ✗ Unexpected keypoints shape: {keypoints.shape}")

except Exception as e:
    print(f"  ✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[4/4] Testing with real image...")
# Load a test image if available
try:
    real_img = cv2.imread('test_frame.jpg')
    if real_img is not None:
        print(f"  Loaded test_frame.jpg: {real_img.shape}")
        # Use full image as bbox
        h, w = real_img.shape[:2]
        real_bbox = np.array([0, 0, w, h, 1.0])

        keypoints = trt_model(real_img, real_bbox)
        print(f"  ✓ Real image inference succeeded!")
        print(f"  Keypoints confidence: {keypoints[:, 2]}")
    else:
        print("  (No test_frame.jpg found, skipping)")
except Exception as e:
    print(f"  Note: {e}")

print("\n" + "="*60)
print("TensorRT wrapper test completed!")
print("="*60)
