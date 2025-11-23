"""
诊断TensorRT性能问题
"""

import cv2
import numpy as np
import time
from src.detectors.tensorrt_wrapper import TensorRTRTMPose

print("="*60)
print("TensorRT性能诊断")
print("="*60)

# 创建测试图像
test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
test_bbox = np.array([100, 100, 400, 400, 0.9])

print("\n[1/3] 加载RTMPose TensorRT引擎...")
trt_model = TensorRTRTMPose(
    engine_path='models/rtmpose/rtmpose-s.engine',
    device='cuda:0'
)
print("  ✓ 加载完成")

print("\n[2/3] 预热推理（10次）...")
for i in range(10):
    _ = trt_model(test_img, test_bbox)
print("  ✓ 预热完成")

print("\n[3/3] 性能测试（100次）...")
times = []
for i in range(100):
    t0 = time.time()
    keypoints = trt_model(test_img, test_bbox)
    t1 = time.time()
    times.append((t1 - t0) * 1000)

times = np.array(times)

print("\n" + "="*60)
print("RTMPose TensorRT 性能统计")
print("="*60)
print(f"  平均耗时: {times.mean():.2f}ms")
print(f"  中位数:   {np.median(times):.2f}ms")
print(f"  最小值:   {times.min():.2f}ms")
print(f"  最大值:   {times.max():.2f}ms")
print(f"  标准差:   {times.std():.2f}ms")
print("="*60)

# 分解各个阶段的耗时
print("\n[4/4] 分解各阶段耗时...")

# 测试预处理
t0 = time.time()
for i in range(100):
    input_tensor = trt_model.preprocess(test_img, test_bbox)
t1 = time.time()
preprocess_time = (t1 - t0) / 100 * 1000

# 测试TensorRT推理
input_tensor = trt_model.preprocess(test_img, test_bbox)
t0 = time.time()
for i in range(100):
    outputs = trt_model.engine.infer(input_tensor)
t1 = time.time()
inference_time = (t1 - t0) / 100 * 1000

# 测试后处理
outputs = trt_model.engine.infer(input_tensor)
t0 = time.time()
for i in range(100):
    keypoints = trt_model.postprocess(outputs, test_bbox, test_img.shape[:2])
t1 = time.time()
postprocess_time = (t1 - t0) / 100 * 1000

print(f"\n  预处理:   {preprocess_time:.2f}ms")
print(f"  TensorRT推理: {inference_time:.2f}ms ⚡")
print(f"  后处理:   {postprocess_time:.2f}ms")
print(f"  总计:     {preprocess_time + inference_time + postprocess_time:.2f}ms")

print("\n" + "="*60)
if inference_time < 2.0:
    print("✓ TensorRT推理速度正常!")
else:
    print("✗ TensorRT推理速度异常慢!")
    print(f"  预期: <2ms, 实际: {inference_time:.2f}ms")
print("="*60)
