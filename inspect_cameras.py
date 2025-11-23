#!/usr/bin/env python3
"""详细检查摄像头设备属性"""

import cv2
import numpy as np

def get_camera_info(device_id):
    """获取摄像头详细信息"""
    cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        return None

    # 获取属性
    info = {
        'device': device_id,
        'backend': cap.getBackendName(),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'format': cap.get(cv2.CAP_PROP_FORMAT),
        'mode': cap.get(cv2.CAP_PROP_MODE),
    }

    # 转换FOURCC为可读字符
    fourcc = info['fourcc']
    fourcc_str = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    info['fourcc_str'] = fourcc_str

    # 读取一帧检查实际格式
    ret, frame = cap.read()
    if ret and frame is not None:
        info['actual_shape'] = frame.shape

        # 检查是否真的是彩色
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            b, g, r = cv2.split(frame)
            # 检查三个通道是否相同（如果相同说明是灰度图转的BGR）
            is_same_bg = np.array_equal(b, g)
            is_same_gr = np.array_equal(g, r)
            info['is_grayscale'] = is_same_bg and is_same_gr

            # 计算通道差异
            if not info['is_grayscale']:
                info['channel_variance'] = {
                    'b': float(np.var(b)),
                    'g': float(np.var(g)),
                    'r': float(np.var(r)),
                }
        else:
            info['is_grayscale'] = True

    cap.release()
    return info

print("=" * 70)
print("摄像头设备详细分析")
print("=" * 70)

for i in [0, 2]:
    info = get_camera_info(i)

    if info is None:
        print(f"\n/dev/video{i}: 不可用")
        continue

    print(f"\n/dev/video{i}:")
    print(f"  后端: {info['backend']}")
    print(f"  分辨率: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']}")
    print(f"  FOURCC: {info['fourcc_str']} (0x{info['fourcc']:08X})")
    print(f"  Format: {info['format']}")
    print(f"  Mode: {info['mode']}")
    print(f"  实际帧shape: {info.get('actual_shape', 'N/A')}")
    print(f"  类型: {'灰度图' if info.get('is_grayscale', True) else '彩色图'}")

    if not info.get('is_grayscale', True):
        print(f"  通道方差: B={info['channel_variance']['b']:.1f}, "
              f"G={info['channel_variance']['g']:.1f}, "
              f"R={info['channel_variance']['r']:.1f}")

print("\n" + "=" * 70)
print("\n解释:")
print("- 很多USB摄像头会在Linux下注册多个/dev/videoN设备")
print("- 不同设备可能提供不同的输出格式 (彩色/灰度/压缩等)")
print("- video0通常是主输出 (彩色)")
print("- video2可能是元数据流、IR流或低分辨率灰度流")
print("=" * 70)
