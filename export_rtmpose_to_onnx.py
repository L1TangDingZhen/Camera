"""
Export RTMPose model to ONNX format

This script exports the RTMPose PyTorch model to ONNX format,
which can then be converted to TensorRT engine using trtexec.

Usage:
    python export_rtmpose_to_onnx.py --model rtmpose-s

Requirements:
    - mmpose
    - mmcv
    - torch
    - onnx (for verification)
"""

import argparse
import torch
import numpy as np
from pathlib import Path


def export_rtmpose_to_onnx(
    config_file: str,
    checkpoint: str,
    output_file: str,
    input_shape: tuple = (1, 3, 256, 192),
    opset_version: int = 11,
    simplify: bool = True,
):
    """Export RTMPose model to ONNX format"""

    print(f"[1/5] Loading MMPose dependencies...")
    from mmpose.apis import init_model

    print(f"[2/5] Loading RTMPose model...")
    print(f"  Config: {config_file}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Device: cuda:0")

    # Load model
    model = init_model(config_file, checkpoint, device='cuda:0')
    model.eval()

    print(f"[3/5] Preparing dummy input...")
    print(f"  Input shape: {input_shape}")

    # Create dummy input
    dummy_input = torch.randn(*input_shape).cuda()

    print(f"[4/5] Exporting to ONNX...")
    print(f"  Output: {output_file}")
    print(f"  Opset version: {opset_version}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"[5/5] Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)

    # Optionally simplify the ONNX model
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify
            print(f"  Simplifying ONNX model...")
            simplified_model, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(simplified_model, output_file)
                print(f"  ✓ Model simplified successfully")
            else:
                print(f"  ⚠ Simplification failed, using original model")
        except ImportError:
            print(f"  ⚠ onnx-simplifier not installed, skipping simplification")
            print(f"    Install with: pip install onnx-simplifier")

    # Get file size
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"\n✓ Export completed successfully!")
    print(f"  ONNX file: {output_file}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"\nNext steps:")
    print(f"  1. Convert to TensorRT: python convert_onnx_to_tensorrt.py")
    print(f"  2. Or use trtexec directly:")
    print(f"     /usr/src/tensorrt/bin/trtexec \\")
    print(f"       --onnx={output_file} \\")
    print(f"       --saveEngine={output_file.replace('.onnx', '.engine')} \\")
    print(f"       --fp16 \\")
    print(f"       --workspace=4096")


def main():
    parser = argparse.ArgumentParser(description='Export RTMPose to ONNX')
    parser.add_argument(
        '--model',
        type=str,
        default='rtmpose-s',
        choices=['rtmpose-s', 'rtmpose-m'],
        help='RTMPose model variant'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output ONNX file path (default: models/rtmpose/{model}.onnx)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        help='ONNX opset version (default: 11)'
    )
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Skip ONNX model simplification'
    )

    args = parser.parse_args()

    # Model paths
    model_configs = {
        'rtmpose-s': {
            'config': 'models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py',
            'checkpoint': 'models/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth',
        },
        'rtmpose-m': {
            'config': 'models/rtmpose/configs/rtmpose-m_8xb256-420e_coco-256x192.py',
            'checkpoint': 'models/rtmpose/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth',
        },
    }

    if args.model not in model_configs:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {list(model_configs.keys())}")
        return

    config = model_configs[args.model]
    config_file = config['config']
    checkpoint = config['checkpoint']

    # Output path
    if args.output is None:
        output_file = f"models/rtmpose/{args.model}.onnx"
    else:
        output_file = args.output

    # Check if files exist
    if not Path(config_file).exists():
        print(f"Error: Config file not found: {config_file}")
        return

    if not Path(checkpoint).exists():
        print(f"Error: Checkpoint file not found: {checkpoint}")
        return

    # Export
    export_rtmpose_to_onnx(
        config_file=config_file,
        checkpoint=checkpoint,
        output_file=output_file,
        opset_version=args.opset,
        simplify=not args.no_simplify,
    )


if __name__ == '__main__':
    main()
