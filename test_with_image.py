#!/usr/bin/env python3
"""使用测试图像调试RTMPose"""

import cv2
import yaml
import traceback

with open('config/config_gpu.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 读取测试图像
frame = cv2.imread('/tmp/test_person.jpg')
print(f"测试图像尺寸: {frame.shape}")

# 初始化模块
from src.detectors.pose_estimator_rtmpose import RTMPoseEstimator
from src.detectors.person_detector import PersonDetector

pose_estimator = RTMPoseEstimator(config['models']['pose'])
person_detector = PersonDetector(config['models']['person'])

# 检测人体
detections = person_detector.detect(frame)

if detections is None or len(detections) == 0:
    print("没有检测到人")
    # 尝试全图
    print("尝试全图姿态估计...")
    bbox = None
else:
    bbox = detections
    print(f"检测到人: {bbox[:4]}")

# 姿态估计
try:
    keypoints = pose_estimator.estimate(frame, bbox)
    if keypoints is not None:
        print(f"✓ 成功！keypoints.shape={keypoints.shape}")
        print(f"  可见关键点数: {(keypoints[:, 2] > 0.3).sum()}/17")
    else:
        print("✗ 返回None")
except Exception as e:
    print(f"✗ 错误: {e}")
    print("\n完整错误堆栈：")
    traceback.print_exc()
