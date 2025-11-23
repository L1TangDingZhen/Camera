"""
对比TensorRT vs PyTorch的骨骼识别准确性
"""

import cv2
import numpy as np
import sys

# 读取一张真实图片进行测试
print("请拍摄一张测试图片...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("按空格键拍摄测试图片，按ESC退出...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头")
        sys.exit(1)

    cv2.imshow("Camera - Press SPACE to capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        print("用户取消")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)
    elif key == 32:  # SPACE
        test_frame = frame.copy()
        break

cap.release()
cv2.destroyAllWindows()

# 保存测试图片
cv2.imwrite('test_comparison.jpg', test_frame)
print(f"✓ 测试图片已保存: test_comparison.jpg ({test_frame.shape})")

# 加载检测器
from src.detectors.person_detector import PersonDetector
from src.detectors.pose_estimator_rtmpose import RTMPoseEstimator

detector_config = {
    'model': 'models/yolov8n.engine',
    'device': 'cuda:0',
    'confidence': 0.5,
    'iou': 0.45
}

print("\n[1/4] 加载人体检测器...")
detector = PersonDetector(detector_config)

print("\n[2/4] 检测人体...")
bboxes = detector.detect(test_frame)
if bboxes is None or len(bboxes) == 0:
    print("✗ 未检测到人体！")
    sys.exit(1)

bbox = bboxes[0]
print(f"  ✓ 检测到人体: bbox={bbox[:4]}, confidence={bbox[4]:.3f}")

# ============== 测试PyTorch模式 ==============
print("\n[3/4] 测试PyTorch RTMPose...")
pytorch_config = {
    'backend': 'rtmpose',
    'model': 'rtmpose-s',
    'config_file': 'models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py',
    'checkpoint': 'models/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth',
    'device': 'cuda:0',
    'confidence': 0.3
}

estimator_pytorch = RTMPoseEstimator(pytorch_config)
keypoints_pytorch = estimator_pytorch.estimate(test_frame, bbox)

print(f"  ✓ PyTorch关键点: {len(keypoints_pytorch)}个")
if hasattr(keypoints_pytorch[0], 'confidence'):
    pytorch_confidences = [kp.confidence for kp in keypoints_pytorch]
else:
    pytorch_confidences = keypoints_pytorch[:, 2]
print(f"  平均置信度: {np.mean(pytorch_confidences):.3f}")

# ============== 测试TensorRT模式 ==============
print("\n[4/4] 测试TensorRT RTMPose...")
tensorrt_config = {
    'backend': 'rtmpose',
    'model': 'rtmpose-s',
    'config_file': 'models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py',
    'checkpoint': 'models/rtmpose/rtmpose-s.engine',
    'device': 'cuda:0',
    'confidence': 0.3
}

estimator_tensorrt = RTMPoseEstimator(tensorrt_config)
keypoints_tensorrt = estimator_tensorrt.estimate(test_frame, bbox)

print(f"  ✓ TensorRT关键点: {len(keypoints_tensorrt)}个")
if hasattr(keypoints_tensorrt[0], 'confidence'):
    tensorrt_confidences = [kp.confidence for kp in keypoints_tensorrt]
else:
    tensorrt_confidences = keypoints_tensorrt[:, 2]
print(f"  平均置信度: {np.mean(tensorrt_confidences):.3f}")

# ============== 对比分析 ==============
print("\n" + "="*60)
print("准确性对比分析")
print("="*60)

# 转换为numpy数组便于对比
if hasattr(keypoints_pytorch[0], 'x'):
    kp_pytorch = np.array([[kp.x, kp.y, kp.confidence] for kp in keypoints_pytorch])
else:
    kp_pytorch = keypoints_pytorch

if hasattr(keypoints_tensorrt[0], 'x'):
    kp_tensorrt = np.array([[kp.x, kp.y, kp.confidence] for kp in keypoints_tensorrt])
else:
    kp_tensorrt = keypoints_tensorrt

# 计算位置差异
position_diff = np.linalg.norm(kp_pytorch[:, :2] - kp_tensorrt[:, :2], axis=1)
print(f"\n位置差异 (像素):")
print(f"  平均: {position_diff.mean():.2f}px")
print(f"  最大: {position_diff.max():.2f}px")
print(f"  最小: {position_diff.min():.2f}px")

# 计算置信度差异
confidence_diff = np.abs(kp_pytorch[:, 2] - kp_tensorrt[:, 2])
print(f"\n置信度差异:")
print(f"  平均: {confidence_diff.mean():.3f}")
print(f"  最大: {confidence_diff.max():.3f}")

# 可视化对比
vis_pytorch = test_frame.copy()
vis_tensorrt = test_frame.copy()

# 绘制PyTorch结果
for i, kp in enumerate(kp_pytorch):
    x, y, conf = int(kp[0]), int(kp[1]), kp[2]
    color = (0, 255, 0) if conf > 0.3 else (0, 255, 255)
    cv2.circle(vis_pytorch, (x, y), 3, color, -1)
    cv2.putText(vis_pytorch, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

# 绘制TensorRT结果
for i, kp in enumerate(kp_tensorrt):
    x, y, conf = int(kp[0]), int(kp[1]), kp[2]
    color = (0, 0, 255) if conf > 0.3 else (255, 0, 255)
    cv2.circle(vis_tensorrt, (x, y), 3, color, -1)
    cv2.putText(vis_tensorrt, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

# 并排显示
comparison = np.hstack([vis_pytorch, vis_tensorrt])
cv2.imwrite('comparison_pytorch_vs_tensorrt.jpg', comparison)
print(f"\n✓ 对比图像已保存: comparison_pytorch_vs_tensorrt.jpg")
print(f"  左侧: PyTorch (绿色)")
print(f"  右侧: TensorRT (红色)")

print("\n" + "="*60)
if position_diff.mean() < 10:
    print("✓ 位置精度良好 (平均差异<10px)")
else:
    print(f"✗ 位置精度较差 (平均差异={position_diff.mean():.2f}px)")
    print("  可能原因: 预处理/后处理bug")
print("="*60)
