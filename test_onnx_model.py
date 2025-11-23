"""
Test ONNX model directly to verify it works
"""

import onnxruntime as ort
import numpy as np
import cv2


def test_onnx():
    print("[1/4] Loading ONNX model...")
    try:
        session = ort.InferenceSession(
            'models/rtmpose/rtmpose-s.onnx',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print(f"  ✓ Loaded successfully")
        print(f"  Provider: {session.get_providers()[0]}")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return

    # Print input/output info
    print("\n[2/4] Model info...")
    for inp in session.get_inputs():
        print(f"  Input: {inp.name}, shape: {inp.shape}, type: {inp.type}")
    for out in session.get_outputs():
        print(f"  Output: {out.name}, shape: {out.shape}, type: {out.type}")

    print("\n[3/4] Creating test input...")
    # Create a test input (simulating preprocessed image)
    test_input = np.random.randn(1, 3, 256, 192).astype(np.float32)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Input dtype: {test_input.dtype}")
    print(f"  Input range: [{test_input.min():.2f}, {test_input.max():.2f}]")

    print("\n[4/4] Running inference...")
    try:
        outputs = session.run(None, {'input': test_input})
        print(f"  ✓ Inference succeeded!")
        print(f"  Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"    Output {i}: shape={out.shape}, dtype={out.dtype}")
            print(f"              range=[{out.min():.4f}, {out.max():.4f}]")

        # Check if outputs look reasonable
        if len(outputs) >= 2:
            simcc_x = outputs[0]
            simcc_y = outputs[1]
            print(f"\n  SimCC X shape: {simcc_x.shape} (expected: (1, 17, 384))")
            print(f"  SimCC Y shape: {simcc_y.shape} (expected: (1, 17, 512))")

            if simcc_x.shape[1] == 17:
                print("  ✓ Output has 17 keypoints - looks correct!")
            else:
                print(f"  ✗ Unexpected keypoint count: {simcc_x.shape[1]}")

    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*60)
    print("ONNX model test completed!")
    print("="*60)


if __name__ == '__main__':
    test_onnx()
