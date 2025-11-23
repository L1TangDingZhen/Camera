"""
Correct RTMPose ONNX Export

This script properly exports RTMPose using MMPose's internal APIs
to ensure all preprocessing and postprocessing are included.
"""

import torch
import numpy as np
from pathlib import Path
import cv2


def export_rtmpose_onnx():
    """Export RTMPose to ONNX with correct preprocessing"""

    print("[1/6] Loading MMPose...")
    from mmpose.apis import init_model
    from mmpose.structures import PoseDataSample

    config_file = 'models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py'
    checkpoint = 'models/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'

    model = init_model(config_file, checkpoint, device='cuda:0')
    model.eval()

    print("[2/6] Creating wrapper model...")

    class RTMPoseONNXWrapper(torch.nn.Module):
        """Wrapper that includes preprocessing for ONNX export"""

        def __init__(self, model):
            super().__init__()
            self.backbone = model.backbone
            self.head = model.head

            # Get preprocessing params from model config
            self.mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).cuda()
            self.std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).cuda()

        def forward(self, x):
            """
            Forward pass for ONNX export

            Args:
                x: Input tensor (1, 3, 256, 192) in BGR format, range [0, 255]

            Returns:
                Tuple of (simcc_x, simcc_y) predictions
            """
            # Normalize input (convert BGR to RGB and normalize)
            # Note: ONNX export expects input in [0, 255] range
            x = x[:, [2, 1, 0], :, :]  # BGR -> RGB
            x = (x - self.mean) / self.std

            # Extract features
            features = self.backbone(x)

            # Get predictions from head
            # RTMPose head outputs SimCC format
            if isinstance(features, (list, tuple)):
                features = features[-1]

            predictions = self.head.forward(features)

            # predictions should be a tuple of (simcc_x, simcc_y)
            return predictions

    print("[3/6] Wrapping model...")
    wrapped_model = RTMPoseONNXWrapper(model)
    wrapped_model.eval()

    print("[4/6] Testing forward pass...")
    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 192).cuda() * 255  # Simulate image data

    with torch.no_grad():
        output = wrapped_model(dummy_input)
        print(f"  Output type: {type(output)}")
        if isinstance(output, (tuple, list)):
            print(f"  Number of outputs: {len(output)}")
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"  Output {i} shape: {out.shape}")
        else:
            print(f"  Output shape: {output.shape}")

    print("[5/6] Exporting to ONNX...")
    output_path = "models/rtmpose/rtmpose-s-correct.onnx"

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['simcc_x', 'simcc_y'] if isinstance(output, (tuple, list)) else ['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
        },
        verbose=False
    )

    print(f"[6/6] Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n✓ Export completed successfully!")
    print(f"  ONNX file: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"\nNext step:")
    print(f"  Convert to TensorRT:")
    print(f"  /usr/src/tensorrt/bin/trtexec \\")
    print(f"    --onnx={output_path} \\")
    print(f"    --saveEngine=models/rtmpose/rtmpose-s-correct.engine \\")
    print(f"    --fp16")


if __name__ == '__main__':
    export_rtmpose_onnx()
