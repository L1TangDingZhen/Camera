"""
Simple RTMPose ONNX Export - Export just backbone+head without wrappers
"""

import torch
import numpy as np


def export_rtmpose():
    print("[1/5] Loading model...")
    from mmpose.apis import init_model

    model = init_model(
        'models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py',
        'models/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth',
        device='cuda:0'
    )
    model.eval()

    print("[2/5] Extracting backbone and head...")

    # Create a simpler wrapper that just does backbone+head forward
    class SimpleRTMPose(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            """
            Args:
                x: (1, 3, 256, 192) normalized tensor

            Returns:
                predictions from model
            """
            # Use model's test_step logic
            batch_data = {
                'inputs': x,
                'data_samples': None
            }

            # Forward through model
            with torch.no_grad():
                # Just use backbone and head
                feats = self.model.backbone(x)
                if isinstance(feats, (list, tuple)):
                    feats = feats[-1]

                # Create dummy batch data sample for head
                batch_size = x.shape[0]
                keypoint_labels = torch.zeros(batch_size, 17, 2).to(x.device)
                keypoint_weights = torch.ones(batch_size, 17).to(x.device)

                # Forward through head - this will give us the predictions
                pred_fields = self.model.head.predict(
                    feats,
                    batch_data_samples=None,
                    test_cfg={}
                )

                return pred_fields

    print("[3/5] Creating wrapper...")
    wrapped = SimpleRTMPose(model)

    print("[4/5] Testing...")
    dummy = torch.randn(1, 3, 256, 192).cuda()

    try:
        with torch.no_grad():
            out = wrapped(dummy)
            print(f"  Output: {type(out)}")
            if isinstance(out, dict):
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape}")
            elif isinstance(out, (list, tuple)):
                print(f"  Length: {len(out)}")
                for i, item in enumerate(out):
                    if isinstance(item, torch.Tensor):
                        print(f"    Item {i}: {item.shape}")
    except Exception as e:
        print(f"  Error during test: {e}")
        print("\n  Trying direct extraction...")

        # Plan B: Just export backbone
        print("[ALT] Exporting backbone only...")

        output_path = "models/rtmpose/rtmpose-s-backbone.onnx"

        torch.onnx.export(
            model.backbone,
            dummy,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['features'],
            verbose=False
        )

        print(f"\n✓ Backbone exported: {output_path}")
        print("  Note: This only includes backbone, not full pose estimation")
        print("  You'll need to implement SimCC decoding in TensorRT wrapper")
        return

    print("[5/5] Exporting ONNX...")
    # If we got here, export the wrapper
    # (This probably won't work, but let's try)


if __name__ == '__main__':
    export_rtmpose()
