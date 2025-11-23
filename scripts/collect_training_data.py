# #!/usr/bin/env python3
# """收集姿态训练数据

# 用法：
#     # 基础收集（手动标注）
#     python scripts/collect_training_data.py --duration 600 --output data/my_training.npz

#     # 自动标注（使用当前SVM预测）
#     python scripts/collect_training_data.py --auto-label --duration 300

#     # 实时显示
#     python scripts/collect_training_data.py --visualize
# """

# import argparse
# import cv2
# import numpy as np
# import time
# from pathlib import Path
# import sys
# from datetime import datetime

# sys.path.append(str(Path(__file__).parent.parent))

# from src.detectors.person_detector import PersonDetector
# from src.detectors.pose_estimator import MediaPipePoseEstimator
# from src.classifiers.pose_classifier import PoseClassifierSVM
# import yaml


# class DataCollector:
#     """数据收集器"""

#     def __init__(self, config_path='config/config_cpu.yaml', auto_label=False):
#         # 加载配置
#         with open(config_path, encoding='utf-8') as f:
#             self.config = yaml.safe_load(f)

#         # 初始化检测器
#         self.person_detector = PersonDetector(self.config)
#         self.pose_estimator = MediaPipePoseEstimator(self.config)

#         # 如果自动标注，加载SVM
#         self.auto_label = auto_label
#         self.classifier = None
#         if auto_label:
#             try:
#                 self.classifier = PoseClassifierSVM()
#                 if self.classifier.is_loaded:
#                     print("[INFO] 使用SVM自动标注")
#                 else:
#                     print("[WARN] SVM未加载，切换到手动标注")
#                     self.auto_label = False
#             except:
#                 print("[WARN] SVM加载失败，切换到手动标注")
#                 self.auto_label = False

#         # 数据缓冲
#         self.landmarks_buffer = []
#         self.labels_buffer = []
#         self.timestamps_buffer = []

#         # 手动标注状态
#         self.current_label = 'sitting'  # 默认
#         self.label_mapping = {'sitting': 0, 'standing': 1, 'lying': 2}

#     def collect(self, duration=60, visualize=True, save_path='data/collected_data.npz'):
#         """收集数据

#         Args:
#             duration: 收集时长（秒）
#             visualize: 是否显示窗口
#             save_path: 保存路径
#         """
#         cap = cv2.VideoCapture(0)

#         if not cap.isOpened():
#             print("[ERROR] 无法打开摄像头")
#             return

#         print("\n" + "="*60)
#         print("数据收集开始")
#         print("="*60)

#         if not self.auto_label:
#             print("\n手动标注按键:")
#             print("  's' - 坐姿 (sitting)")
#             print("  't' - 站姿 (standing)")
#             print("  'l' - 躺姿 (lying)")
#             print("  'q' - 退出")
#             print(f"\n当前标签: {self.current_label}")

#         print(f"\n目标时长: {duration}秒")
#         print(f"保存路径: {save_path}\n")

#         start_time = time.time()
#         frame_count = 0
#         collected_count = 0

#         try:
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # 经过时间
#                 elapsed = time.time() - start_time
#                 if elapsed >= duration:
#                     print("\n✓ 达到目标时长，停止收集")
#                     break

#                 frame_count += 1

#                 # 检测人体
#                 bbox = self.person_detector.detect(frame)

#                 if bbox is not None:
#                     # 姿态估计
#                     keypoints = self.pose_estimator.estimate(frame, bbox)

#                     if keypoints is not None:
#                         world_landmarks = self.pose_estimator.get_world_landmarks()

#                         if world_landmarks is None:
#                             continue

#                         # 获取标签
#                         if self.auto_label:
#                             # 自动标注
#                             pred = self.classifier.predict(world_landmarks)
#                             if pred:
#                                 label = self.label_mapping[pred]
#                             else:
#                                 continue  # 跳过无法预测的
#                         else:
#                             # 手动标注
#                             label = self.label_mapping[self.current_label]

#                         # 保存数据
#                         self.landmarks_buffer.append(world_landmarks.copy())
#                         self.labels_buffer.append(label)
#                         self.timestamps_buffer.append(time.time())
#                         collected_count += 1

#                         # 可视化
#                         if visualize:
#                             self._draw_visualization(
#                                 frame, keypoints, bbox,
#                                 label, collected_count, elapsed, duration
#                             )

#                 # 显示
#                 if visualize:
#                     cv2.imshow('Data Collection', frame)

#                     # 处理按键
#                     key = cv2.waitKey(1) & 0xFF
#                     if key == ord('q'):
#                         print("\n用户退出")
#                         break
#                     elif key == ord('s'):
#                         self.current_label = 'sitting'
#                         print(f"切换标签: {self.current_label}")
#                     elif key == ord('t'):
#                         self.current_label = 'standing'
#                         print(f"切换标签: {self.current_label}")
#                     elif key == ord('l'):
#                         self.current_label = 'lying'
#                         print(f"切换标签: {self.current_label}")

#                 # 每秒打印进度
#                 if frame_count % 30 == 0:
#                     print(f"进度: {elapsed:.1f}s / {duration}s | "
#                           f"已收集: {collected_count} 样本 | "
#                           f"当前标签: {self.current_label if not self.auto_label else 'auto'}")

#         finally:
#             cap.release()
#             cv2.destroyAllWindows()

#         # 保存数据
#         if collected_count > 0:
#             self._save_data(save_path)
#         else:
#             print("[WARN] 没有收集到任何数据")

#     def _draw_visualization(self, frame, keypoints, bbox, label, count, elapsed, duration):
#         """绘制可视化"""
#         # 绘制边界框
#         x1, y1, x2, y2 = bbox[:4].astype(int)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # 绘制关键点
#         if keypoints is not None:
#             for kp in keypoints:
#                 x, y, conf = kp
#                 if conf > 0.5:
#                     cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

#         # 信息面板
#         label_names = {0: 'sitting', 1: 'standing', 2: 'lying'}
#         colors = {0: (255, 200, 0), 1: (0, 255, 0), 2: (255, 0, 255)}

#         label_name = label_names[label]
#         color = colors[label]

#         # 绘制信息
#         cv2.putText(frame, f"Label: {label_name}", (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#         cv2.putText(frame, f"Count: {count}", (10, 70),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         # 进度条
#         progress = min(elapsed / duration, 1.0)
#         bar_width = int(progress * (frame.shape[1] - 20))
#         cv2.rectangle(frame, (10, frame.shape[0] - 30),
#                      (10 + bar_width, frame.shape[0] - 10),
#                      (0, 255, 0), -1)

#         # 时间
#         cv2.putText(frame, f"{elapsed:.1f}s / {duration}s",
#                    (10, frame.shape[0] - 40),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

#     def _save_data(self, save_path):
#         """保存数据"""
#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)

#         landmarks = np.array(self.landmarks_buffer)
#         labels = np.array(self.labels_buffer)
#         timestamps = np.array(self.timestamps_buffer)

#         # 统计
#         label_names = ['sitting', 'standing', 'lying']
#         label_counts = [np.sum(labels == i) for i in range(3)]

#         print("\n" + "="*60)
#         print("数据收集完成")
#         print("="*60)
#         print(f"\n总样本数: {len(landmarks)}")
#         print("类别分布:")
#         for name, count in zip(label_names, label_counts):
#             print(f"  {name}: {count} ({count/len(landmarks)*100:.1f}%)")

#         # 保存
#         np.savez(save_path,
#                 landmarks=landmarks,
#                 labels=labels,
#                 timestamps=timestamps)

#         print(f"\n✓ 数据已保存: {save_path}")
#         print(f"✓ 数据形状: landmarks {landmarks.shape}, labels {labels.shape}\n")


# def main():
#     parser = argparse.ArgumentParser(description='收集姿态训练数据')
#     parser.add_argument('--duration', type=int, default=60,
#                        help='收集时长（秒）')
#     parser.add_argument('--output', type=str,
#                        default=f'data/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz',
#                        help='输出文件路径')
#     parser.add_argument('--auto-label', action='store_true',
#                        help='使用SVM自动标注（而非手动）')
#     parser.add_argument('--visualize', action='store_true', default=True,
#                        help='显示可视化窗口')
#     parser.add_argument('--config', type=str, default='config/config_cpu.yaml',
#                        help='配置文件路径')

#     args = parser.parse_args()

#     # 创建收集器
#     collector = DataCollector(
#         config_path=args.config,
#         auto_label=args.auto_label
#     )

#     # 收集数据
#     collector.collect(
#         duration=args.duration,
#         visualize=args.visualize,
#         save_path=args.output
#     )


# if __name__ == '__main__':
#     main()
