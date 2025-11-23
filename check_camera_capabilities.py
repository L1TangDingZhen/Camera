#!/usr/bin/env python3
"""检查摄像头支持的所有分辨率"""

import cv2

def test_resolutions(device_id):
    """测试摄像头支持哪些分辨率"""
    common_resolutions = [
        (640, 480, "VGA"),
        (800, 600, "SVGA"),
        (1280, 720, "HD 720p"),
        (1920, 1080, "FHD 1080p"),
        (640, 360, "nHD"),
        (1280, 960, "SXGA-"),
        (2560, 1440, "QHD"),
        (3840, 2160, "4K UHD"),
    ]

    print(f"\n/dev/video{device_id} 支持的分辨率:")
    print("-" * 60)

    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    supported = []

    for width, height, name in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width == width and actual_height == height:
            # 尝试实际读取一帧验证
            ret, frame = cap.read()
            if ret and frame is not None and frame.shape[1] == width and frame.shape[0] == height:
                supported.append((width, height, name))
                print(f"  ✓ {width}x{height} ({name})")

    cap.release()

    if not supported:
        # 如果上面没找到，读取默认分辨率
        cap = cv2.VideoCapture(device_id)
        ret, frame = cap.read()
        if ret:
            print(f"  默认: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()

    return supported

print("=" * 60)
print("摄像头分辨率能力检测")
print("=" * 60)

# 检测video0
supported = test_resolutions(0)

print("\n" + "=" * 60)
print("补充说明:")
print("- VGA (640x480) 是最基础的分辨率")
print("- HD 720p (1280x720) 是高清")
print("- FHD 1080p (1920x1080) 是全高清")
print("- 4K UHD (3840x2160) 是超高清")
print("=" * 60)
