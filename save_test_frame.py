#!/usr/bin/env python3
"""保存一帧测试图像"""

import cv2
import yaml

with open('config/config_gpu.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

cap = cv2.VideoCapture(config['camera']['source'])

if not cap.isOpened():
    print("摄像头打开失败！")
    exit(1)

ret, frame = cap.read()
cap.release()

if ret:
    cv2.imwrite('/tmp/test_frame.jpg', frame)
    print(f"已保存测试帧: /tmp/test_frame.jpg")
    print(f"分辨率: {frame.shape[1]}x{frame.shape[0]}")
    print(f"通道数: {frame.shape[2]}")
else:
    print("无法读取帧")
