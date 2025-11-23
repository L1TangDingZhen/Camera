#!/usr/bin/env python3
"""测试640x480的FPS"""
import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"640x480配置: {actual_w}x{actual_h} @ {actual_fps} FPS")

times = []
for i in range(30):
    start = time.time()
    ret, frame = cap.read()
    elapsed = time.time() - start
    if ret:
        times.append(elapsed * 1000)

cap.release()

print(f"平均读帧时间: {np.mean(times):.2f}ms (理论FPS: {1000/np.mean(times):.1f})")
