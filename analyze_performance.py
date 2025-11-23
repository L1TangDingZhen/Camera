#!/usr/bin/env python3
"""详细性能分析"""

import cv2
import yaml
import time
import numpy as np

with open('config/config_gpu.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 测试摄像头读取速度
print("=" * 70)
print("1. 摄像头读取性能测试")
print("=" * 70)

cap = cv2.VideoCapture(config['camera']['source'])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['resolution'][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['resolution'][1])

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"摄像头配置: {actual_w}x{actual_h} @ {actual_fps} FPS")

camera_times = []
for i in range(30):
    start = time.time()
    ret, frame = cap.read()
    elapsed = time.time() - start
    if ret:
        camera_times.append(elapsed * 1000)

cap.release()

print(f"平均读帧时间: {np.mean(camera_times):.2f}ms (理论FPS: {1000/np.mean(camera_times):.1f})")
print(f"最小/最大: {np.min(camera_times):.2f}ms / {np.max(camera_times):.2f}ms")

# 测试YOLOv8性能
print(f"\n{'=' * 70}")
print("2. YOLOv8人体检测性能测试")
print("=" * 70)

from src.detectors.person_detector import PersonDetector

person_detector = PersonDetector(config['models']['person'])

cap = cv2.VideoCapture(config['camera']['source'])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['resolution'][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['resolution'][1])

yolo_times = []
for i in range(30):
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    detections = person_detector.detect(frame)
    elapsed = time.time() - start
    yolo_times.append(elapsed * 1000)

cap.release()

print(f"平均检测时间: {np.mean(yolo_times):.2f}ms (理论FPS: {1000/np.mean(yolo_times):.1f})")
print(f"最小/最大: {np.min(yolo_times):.2f}ms / {np.max(yolo_times):.2f}ms")

# 测试RTMPose性能
print(f"\n{'=' * 70}")
print("3. RTMPose姿态估计性能测试")
print("=" * 70)

from src.detectors.pose_estimator_rtmpose import RTMPoseEstimator

pose_estimator = RTMPoseEstimator(config['models']['pose'])

cap = cv2.VideoCapture(config['camera']['source'])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['resolution'][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['resolution'][1])

rtmpose_times = []
for i in range(30):
    ret, frame = cap.read()
    if not ret:
        break

    # 使用全图bbox
    h, w = frame.shape[:2]
    bbox = np.array([0, 0, w, h, 1.0], dtype=np.float32)

    start = time.time()
    keypoints = pose_estimator.estimate(frame, bbox)
    elapsed = time.time() - start
    rtmpose_times.append(elapsed * 1000)

cap.release()

print(f"平均姿态估计时间: {np.mean(rtmpose_times):.2f}ms (理论FPS: {1000/np.mean(rtmpose_times):.1f})")
print(f"最小/最大: {np.min(rtmpose_times):.2f}ms / {np.max(rtmpose_times):.2f}ms")

# 综合分析
print(f"\n{'=' * 70}")
print("4. 综合性能分析")
print("=" * 70)

total_time = np.mean(camera_times) + np.mean(yolo_times) + np.mean(rtmpose_times)
theoretical_fps = 1000 / total_time

print(f"摄像头读取:     {np.mean(camera_times):6.2f}ms  ({np.mean(camera_times)/total_time*100:5.1f}%)")
print(f"人体检测(YOLO): {np.mean(yolo_times):6.2f}ms  ({np.mean(yolo_times)/total_time*100:5.1f}%)")
print(f"姿态估计(RTM):  {np.mean(rtmpose_times):6.2f}ms  ({np.mean(rtmpose_times)/total_time*100:5.1f}%)")
print(f"{'-' * 70}")
print(f"总计:           {total_time:6.2f}ms")
print(f"理论FPS:        {theoretical_fps:6.1f}")

# 瓶颈分析
bottleneck = max([
    ("摄像头读取", np.mean(camera_times)),
    ("YOLOv8检测", np.mean(yolo_times)),
    ("RTMPose估计", np.mean(rtmpose_times))
], key=lambda x: x[1])

print(f"\n⚠️ 性能瓶颈: {bottleneck[0]} ({bottleneck[1]:.2f}ms)")

# 优化建议
print(f"\n{'=' * 70}")
print("5. 优化建议")
print("=" * 70)

if np.mean(yolo_times) > 50:
    print("🔧 YOLOv8优化:")
    print("   - 方案1: 换用YOLOv8n (当前用yolov8m)")
    print("   - 方案2: 降低输入分辨率到640x480")
    print("   - 方案3: 增加detection_interval (跳帧检测)")

if np.mean(rtmpose_times) > 30:
    print("🔧 RTMPose优化:")
    print("   - 方案1: 换用rtmpose-tiny (当前用rtmpose-s)")
    print("   - 方案2: 裁剪人体区域后再估计姿态")

if np.mean(camera_times) > 20:
    print("🔧 摄像头优化:")
    print("   - 方案1: 降低分辨率到640x480")
    print("   - 方案2: 检查USB带宽（是否与其他设备共享）")

print(f"\n预期提升:")
yolov8n_gain = (np.mean(yolo_times) - np.mean(yolo_times) * 0.4) if np.mean(yolo_times) > 50 else 0
rtmpose_tiny_gain = (np.mean(rtmpose_times) - np.mean(rtmpose_times) * 0.5) if np.mean(rtmpose_times) > 30 else 0

if yolov8n_gain > 0 or rtmpose_tiny_gain > 0:
    new_total = total_time - yolov8n_gain - rtmpose_tiny_gain
    new_fps = 1000 / new_total
    print(f"  使用YOLOv8n + RTMPose-tiny: {new_fps:.1f} FPS (提升 {new_fps - theoretical_fps:.1f} FPS)")

print("=" * 70)
