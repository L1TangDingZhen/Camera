"""
Inspect ONNX model structure to understand TensorRT compatibility issues
"""

import onnx
import numpy as np

onnx_path = 'models/rtmpose/rtmpose-s.onnx'

print("[1/3] Loading ONNX model...")
model = onnx.load(onnx_path)

print("\n[2/3] Model IR version and opset...")
print(f"  IR version: {model.ir_version}")
print(f"  Producer: {model.producer_name} {model.producer_version}")
print(f"  Opset imports:")
for opset in model.opset_import:
    print(f"    {opset.domain if opset.domain else 'default'}: {opset.version}")

print("\n[3/3] Input/Output information...")
print("\nInputs:")
for input_tensor in model.graph.input:
    shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"  Name: {input_tensor.name}")
    print(f"  Shape: {shape}")
    print(f"  Type: {input_tensor.type.tensor_type.elem_type}")

print("\nOutputs:")
for output_tensor in model.graph.output:
    shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in output_tensor.type.tensor_type.shape.dim]
    print(f"  Name: {output_tensor.name}")
    print(f"  Shape: {shape}")
    print(f"  Type: {output_tensor.type.tensor_type.elem_type}")

print("\n[4/4] Checking for unsupported ops...")
# Check if there are any custom ops or unsupported layers
all_ops = set()
for node in model.graph.node:
    all_ops.add(node.op_type)

print(f"\nUnique operations in model ({len(all_ops)}):")
for op in sorted(all_ops):
    print(f"  - {op}")

print("\n" + "="*60)
print("ONNX inspection complete!")
print("="*60)
