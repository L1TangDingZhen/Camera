#!/usr/bin/env python3
"""测试所有摄像头设备，找出哪个是彩色的"""

import cv2
import numpy as np

def test_camera(device_id):
    """测试摄像头并返回信息"""
    cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    # 检查是否彩色
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # 检查是否真的有颜色（不是BGR转换的灰度图）
        b, g, r = cv2.split(frame)
        is_color = not (np.array_equal(b, g) and np.array_equal(g, r))
    else:
        is_color = False

    return {
        'device': device_id,
        'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
        'channels': frame.shape[2] if len(frame.shape) == 3 else 1,
        'is_color': is_color,
        'dtype': frame.dtype
    }

print("=" * 60)
print("摄像头设备检测")
print("=" * 60)

for i in range(4):
    info = test_camera(i)
    if info:
        print(f"\n/dev/video{i}:")
        print(f"  分辨率: {info['resolution']}")
        print(f"  通道数: {info['channels']}")
        print(f"  彩色: {'是' if info['is_color'] else '否'}")
        print(f"  数据类型: {info['dtype']}")
    else:
        print(f"\n/dev/video{i}: 不可用")

print("\n" + "=" * 60)
