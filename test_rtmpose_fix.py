#!/usr/bin/env python3
"""测试RTMPose修复"""

import cv2
import numpy as np
import yaml
import sys

# 加载配置
with open('config/config_gpu.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 初始化摄像头
print(f"打开摄像头 /dev/video{config['camera']['source']}...")
cap = cv2.VideoCapture(config['camera']['source'])

if not cap.isOpened():
    print("摄像头打开失败！")
    sys.exit(1)

ret, frame = cap.read()
if not ret:
    print("无法读取帧！")
    cap.release()
    sys.exit(1)

print(f"摄像头: {frame.shape[1]}x{frame.shape[0]}, 通道={frame.shape[2]}")

# 初始化RTMPose
print("\n初始化RTMPose...")
from src.detectors.pose_estimator_rtmpose import RTMPoseEstimator

pose_estimator = RTMPoseEstimator(config['models']['pose'])

# 初始化YOLOv8
print("\n初始化YOLOv8...")
from src.detectors.person_detector import PersonDetector

person_detector = PersonDetector(config['models']['person'])

print("\n开始测试...")
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i}: 读取失败")
        continue

    # 检测人体
    detections = person_detector.detect(frame)

    if detections is None or len(detections) == 0:
        print(f"Frame {i}: 没有检测到人")
        continue

    # 取第一个人
    bbox = detections[0]
    print(f"Frame {i}: 检测到人, bbox={bbox[:4]}")

    # 姿态估计
    keypoints = pose_estimator.estimate(frame, bbox)

    if keypoints is None:
        print(f"Frame {i}: 姿态估计失败")
    else:
        print(f"Frame {i}: 姿态估计成功! shape={keypoints.shape}")

        # 检查关键点数量
        visible_count = np.sum(keypoints[:, 2] > 0.3)
        print(f"Frame {i}:   可见关键点: {visible_count}/17")

cap.release()
print("\n测试完成！")
