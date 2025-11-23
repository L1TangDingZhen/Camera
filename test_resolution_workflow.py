#!/usr/bin/env python3
"""测试完整的分辨率工作流程"""

import cv2
import yaml
import numpy as np

# 测试不同分辨率
test_configs = [
    (640, 480, "VGA"),
    (1280, 720, "HD 720p"),
    (1920, 1080, "FHD 1080p"),
]

print("=" * 70)
print("YOLOv8 分辨率工作流程测试")
print("=" * 70)

for width, height, name in test_configs:
    print(f"\n{'=' * 70}")
    print(f"测试分辨率: {width}x{height} ({name})")
    print('=' * 70)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  ✗ 无法读取帧")
        continue

    actual_w, actual_h = frame.shape[1], frame.shape[0]
    print(f"  摄像头实际输出: {actual_w}x{actual_h}")

    if actual_w != width or actual_h != height:
        print(f"  ⚠️ 摄像头未采用请求的分辨率")
        continue

    # 加载YOLOv8
    from ultralytics import YOLO

    model = YOLO('yolov8m.pt')
    model.to('cuda:0')

    # 预热
    _ = model(frame, verbose=False)

    # 测试推理
    import time
    times = []

    for i in range(10):
        ret, frame = cv2.VideoCapture(0).read()
        if not ret:
            break

        start = time.time()
        results = model(frame, conf=0.5, classes=[0], verbose=False, device='cuda:0')
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # 转为ms

    avg_time = np.mean(times)
    fps = 1000 / avg_time

    print(f"  YOLOv8 平均推理时间: {avg_time:.1f}ms")
    print(f"  YOLOv8 理论FPS: {fps:.1f}")

    # 计算内存占用估算
    frame_size_mb = (actual_w * actual_h * 3) / (1024 * 1024)
    print(f"  单帧内存占用: {frame_size_mb:.2f} MB")

print("\n" + "=" * 70)
print("总结:")
print("- YOLOv8会自动缩放输入图像到模型训练尺寸 (通常是640x640)")
print("- 更高分辨率 → 更高精度，但推理时间更长")
print("- 建议根据精度需求选择:")
print("    • 实时性优先: 640x480 (VGA)")
print("    • 平衡: 1280x720 (HD)")
print("    • 精度优先: 1920x1080 (FHD)")
print("=" * 70)
