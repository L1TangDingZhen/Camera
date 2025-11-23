#!/usr/bin/env python3
"""调试RTMPose错误"""

import cv2
import yaml
import numpy as np
import traceback

with open('config/config_gpu.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 初始化RTMPose
from src.detectors.pose_estimator_rtmpose import RTMPoseEstimator
pose_estimator = RTMPoseEstimator(config['models']['pose'])

# 初始化YOLOv8
from src.detectors.person_detector import PersonDetector
person_detector = PersonDetector(config['models']['person'])

# 读取一帧
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ret, frame = cap.read()
cap.release()

if not ret:
    print("无法读取帧")
    exit(1)

print(f"帧尺寸: {frame.shape}")

# 检测人体
detections = person_detector.detect(frame)

if detections is None or len(detections) == 0:
    print("没有检测到人，请确保有人在摄像头前")
    exit(0)

bbox = detections
print(f"检测到人: bbox={bbox[:4]}")

# 尝试姿态估计，捕获完整错误
try:
    keypoints = pose_estimator.estimate(frame, bbox)
    if keypoints is not None:
        print(f"成功！keypoints.shape={keypoints.shape}")
    else:
        print("返回None")
except Exception as e:
    print(f"\n完整错误堆栈：")
    traceback.print_exc()
