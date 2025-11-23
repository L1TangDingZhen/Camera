"""
测试完整TensorRT流程（使用测试图像）
"""

import cv2
import numpy as np
import time
from src.detectors.person_detector import PersonDetector
from src.detectors.pose_estimator_rtmpose import RTMPoseEstimator

# 配置
config = {
    'person': {
        'model': 'models/yolov8n.engine',
        'device': 'cuda:0',
        'confidence': 0.5,
        'iou': 0.45
    },
    'pose': {
        'backend': 'rtmpose',
        'model': 'rtmpose-s',
        'config_file': 'models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py',
        'checkpoint': 'models/rtmpose/rtmpose-s.engine',
        'device': 'cuda:0',
        'confidence': 0.3
    }
}

print("="*60)
print("完整TensorRT流程测试")
print("="*60)

print("\n[1/5] 加载YOLO TensorRT引擎...")
detector = PersonDetector(config['person'])
print("  ✓ YOLO引擎加载成功")

print("\n[2/5] 加载RTMPose TensorRT引擎...")
pose_estimator = RTMPoseEstimator(config['pose'])
print("  ✓ RTMPose引擎加载成功")

print("\n[3/5] 创建测试图像（带人形模拟）...")
# 创建一个包含简单"人"形状的测试图像
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
# 画一个简单的人形（头+身体+手+腿）
cv2.circle(test_img, (320, 150), 30, (255, 255, 255), -1)  # 头
cv2.rectangle(test_img, (300, 180), (340, 280), (255, 255, 255), -1)  # 身体
cv2.rectangle(test_img, (260, 190), (300, 250), (255, 255, 255), -1)  # 左手
cv2.rectangle(test_img, (340, 190), (380, 250), (255, 255, 255), -1)  # 右手
cv2.rectangle(test_img, (300, 280), (320, 380), (255, 255, 255), -1)  # 左腿
cv2.rectangle(test_img, (320, 280), (340, 380), (255, 255, 255), -1)  # 右腿
print("  ✓ 测试图像创建完成")

print("\n[4/5] 运行检测流程...")
# YOLO检测
t0 = time.time()
bboxes = detector.detect(test_img)
t1 = time.time()
yolo_time = (t1 - t0) * 1000

if bboxes is None:
    bboxes = []

print(f"  YOLO检测: {yolo_time:.2f}ms, 检测到 {len(bboxes)} 个目标")

if len(bboxes) == 0:
    # 如果没检测到人，手动创建一个bbox
    print("  (未检测到人，使用手动bbox进行测试)")
    bbox = np.array([150, 100, 480, 400, 0.9])

    # RTMPose姿态估计
    t0 = time.time()
    keypoints = pose_estimator.estimate(test_img, bbox)
    t1 = time.time()
    rtmpose_time = (t1 - t0) * 1000

    print(f"  RTMPose姿态估计: {rtmpose_time:.2f}ms")
    print(f"  关键点数量: {len(keypoints)}")
    if isinstance(keypoints, np.ndarray):
        print(f"  平均置信度: {np.mean(keypoints[:, 2]):.3f}")  # 第3列是置信度
    else:
        print(f"  平均置信度: {np.mean([kp.confidence for kp in keypoints]):.3f}")

    print("\n[5/5] 性能总结:")
    print(f"  YOLO (TensorRT FP16):   {yolo_time:.2f}ms")
    print(f"  RTMPose (TensorRT FP16): {rtmpose_time:.2f}ms")
    print(f"  总耗时:                 {yolo_time + rtmpose_time:.2f}ms")
    print(f"  预计FPS (单人):         {1000/(yolo_time + rtmpose_time):.1f}")
else:
    # 使用检测到的第一个人
    bbox = bboxes[0]

    # RTMPose姿态估计
    t0 = time.time()
    keypoints = pose_estimator.estimate(test_img, bbox)
    t1 = time.time()
    rtmpose_time = (t1 - t0) * 1000

    print(f"  RTMPose姿态估计: {rtmpose_time:.2f}ms")
    print(f"  关键点数量: {len(keypoints)}")
    if isinstance(keypoints, np.ndarray):
        print(f"  平均置信度: {np.mean(keypoints[:, 2]):.3f}")  # 第3列是置信度
    else:
        print(f"  平均置信度: {np.mean([kp.confidence for kp in keypoints]):.3f}")

    print("\n[5/5] 性能总结:")
    print(f"  YOLO (TensorRT FP16):   {yolo_time:.2f}ms")
    print(f"  RTMPose (TensorRT FP16): {rtmpose_time:.2f}ms")
    print(f"  总耗时:                 {yolo_time + rtmpose_time:.2f}ms")
    print(f"  预计FPS (单人):         {1000/(yolo_time + rtmpose_time):.1f}")

print("\n" + "="*60)
print("✓ 完整TensorRT流程测试完成！")
print("="*60)
