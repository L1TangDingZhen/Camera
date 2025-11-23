#!/usr/bin/env python3
"""获取RTMPose错误的完整堆栈信息"""

import cv2
import yaml
import traceback
import sys

with open('config/config_gpu.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 临时改回rtmpose
config['models']['pose']['backend'] = 'rtmpose'
config['models']['pose']['model'] = 'rtmpose-s'
config['models']['pose']['config_file'] = 'models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py'
config['models']['pose']['checkpoint'] = 'models/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'
config['models']['pose']['device'] = 'cuda:0'
config['models']['pose']['confidence'] = 0.3

frame = cv2.imread('/tmp/test_person.jpg')
print(f"测试图像: {frame.shape}")

from src.detectors.pose_estimator_rtmpose import RTMPoseEstimator
pose_estimator = RTMPoseEstimator(config['models']['pose'])

# 全图检测
print("\n尝试姿态估计...")
try:
    keypoints = pose_estimator.estimate(frame, None)
    if keypoints is not None:
        print(f"✓ 成功! shape={keypoints.shape}")
    else:
        print("✗ 返回None")
except Exception as e:
    print(f"\n{'='*70}")
    print("完整错误堆栈:")
    print('='*70)
    traceback.print_exc()
    print('='*70)
