#!/usr/bin/env python3
"""
数据录制工具 - 用于收集坐/站/躺姿态的训练数据

使用方法:
    python collect_data.py --countdown 5 --min-duration 30

按键控制:
    's' - 开始录制 Sitting（坐）
    't' - 开始录制 Standing（站）
    'l' - 开始录制 Lying（躺）
    'q' - 停止当前录制
    'ESC' - 退出程序
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import os
import json
from datetime import datetime
from typing import Optional, List, Tuple


class DataCollector:
    """姿态数据收集器"""

    # MediaPipe关键点到COCO格式的映射
    MEDIAPIPE_TO_COCO = {
        0: 0,   # nose
        11: 5,  # left_shoulder
        12: 6,  # right_shoulder
        13: 7,  # left_elbow
        14: 8,  # right_elbow
        15: 9,  # left_wrist
        16: 10, # right_wrist
        23: 11, # left_hip
        24: 12, # right_hip
        25: 13, # left_knee
        26: 14, # right_knee
        27: 15, # left_ankle
        28: 16, # right_ankle
    }

    def __init__(self, countdown: int = 5, min_duration: int = 30):
        """
        Args:
            countdown: 按键后倒计时秒数
            min_duration: 建议最小录制时长（秒）
        """
        self.countdown = countdown
        self.min_duration = min_duration

        # 初始化MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 录制状态
        self.is_recording = False
        self.current_pose_label = None
        self.record_start_time = None
        self.collected_samples = []

        # 数据存储
        self.output_dir = "training_data"
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_features(self, world_landmarks: np.ndarray) -> Optional[np.ndarray]:
        """从3D world landmarks提取特征向量

        Args:
            world_landmarks: (17, 4) [x, y, z, visibility]

        Returns:
            features: 特征向量（相对归一化的）
        """
        # 检查关键点可见性
        required_indices = [5, 6, 11, 12]  # 肩膀和臀部
        for idx in required_indices:
            if world_landmarks[idx][3] < 0.3:  # visibility < 0.3
                return None

        # 计算躯干长度（用于归一化）
        left_shoulder = world_landmarks[5][:3]
        right_shoulder = world_landmarks[6][:3]
        left_hip = world_landmarks[11][:3]
        right_hip = world_landmarks[12][:3]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center)

        if torso_length < 0.1:  # 躯干长度异常
            return None

        features = []

        # 1. 所有关键点的归一化3D坐标 (17 × 3 = 51维)
        for i in range(17):
            if world_landmarks[i][3] > 0:  # 可见
                features.extend(world_landmarks[i][:3] / torso_length)
            else:
                features.extend([0.0, 0.0, 0.0])  # 不可见用0填充

        # 2. 额外的几何特征

        # 2.1 躯干角度
        torso_vec = hip_center - shoulder_center
        vertical = np.array([0, -1, 0])
        torso_angle = np.degrees(np.arccos(
            np.clip(np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) + 1e-6), -1, 1)
        ))
        features.append(torso_angle / 90.0)  # 归一化到 [0, 2]

        # 2.2 髋膝Z轴差（如果膝盖可见）
        if world_landmarks[13][3] > 0.3 and world_landmarks[14][3] > 0.3:
            left_knee = world_landmarks[13][:3]
            right_knee = world_landmarks[14][:3]
            knee_center = (left_knee + right_knee) / 2
            hip_knee_z_diff = (hip_center[2] - knee_center[2]) / torso_length
            features.append(hip_knee_z_diff)
        else:
            features.append(0.0)

        # 2.3 髋膝距离
        if world_landmarks[13][3] > 0.3 and world_landmarks[14][3] > 0.3:
            left_knee = world_landmarks[13][:3]
            right_knee = world_landmarks[14][:3]
            knee_center = (left_knee + right_knee) / 2
            hip_knee_dist = np.linalg.norm(hip_center - knee_center) / torso_length
            features.append(hip_knee_dist)
        else:
            features.append(0.0)

        # 2.4 髋部高度（相对）
        hip_height = hip_center[1] / torso_length
        features.append(hip_height)

        # 2.5 肩膀宽度
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder) / torso_length
        features.append(shoulder_width)

        # 2.6 关键点可见性统计
        visibility_scores = [world_landmarks[i][3] for i in range(17)]
        features.append(np.mean(visibility_scores))
        features.append(np.min(visibility_scores))

        return np.array(features, dtype=np.float32)

    def draw_countdown(self, frame: np.ndarray, remaining: int) -> np.ndarray:
        """绘制倒计时"""
        h, w = frame.shape[:2]

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)

        # 倒计时数字
        text = f"{remaining}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 10
        thickness = 20
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)

        # 提示文字
        pose_name = {"sitting": "坐姿", "standing": "站姿", "lying": "躺姿"}.get(self.current_pose_label, "")
        hint = f"准备录制: {pose_name}"
        cv2.putText(frame, hint, (w//4 + 20, h//4 + 60), font, 1.5, (255, 255, 255), 2)

        return frame

    def draw_recording_status(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        """绘制录制状态"""
        h, w = frame.shape[:2]

        # 录制指示器（红点）
        cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)

        # 录制信息
        pose_name = {"sitting": "坐姿", "standing": "站姿", "lying": "躺姿"}.get(self.current_pose_label, "")
        text = f"正在录制: {pose_name}"
        cv2.putText(frame, text, (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        # 时长
        time_text = f"已录: {elapsed:.1f}s"
        if elapsed < self.min_duration:
            time_text += f" / 建议: {self.min_duration}s"
            color = (0, 165, 255)  # 橙色
        else:
            time_text += " (可按q停止)"
            color = (0, 255, 0)  # 绿色

        cv2.putText(frame, time_text, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 样本数
        sample_text = f"已采集: {len(self.collected_samples)} 帧"
        cv2.putText(frame, sample_text, (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    def draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """绘制操作说明"""
        h, w = frame.shape[:2]

        instructions = [
            "操作说明:",
            "  's' - 录制坐姿 (Sitting)",
            "  't' - 录制站姿 (Standing)",
            "  'l' - 录制躺姿 (Lying)",
            "  'q' - 停止录制",
            "  'ESC' - 退出程序",
            "",
            f"已保存数据: {self._count_saved_samples()} 条"
        ]

        y_offset = h - 250
        for i, line in enumerate(instructions):
            cv2.putText(frame, line, (20, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame

    def _count_saved_samples(self) -> int:
        """统计已保存的样本数"""
        count = 0
        for pose in ['sitting', 'standing', 'lying']:
            filepath = os.path.join(self.output_dir, f"{pose}_samples.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    count += len(data)
        return count

    def start_recording(self, pose_label: str):
        """开始录制某个姿态"""
        self.current_pose_label = pose_label
        self.record_start_time = None  # 倒计时结束后才设置
        self.collected_samples = []
        print(f"\n[INFO] 准备录制 {pose_label}，倒计时 {self.countdown} 秒...")

    def stop_recording(self):
        """停止录制并保存"""
        if not self.is_recording:
            return

        self.is_recording = False

        if len(self.collected_samples) == 0:
            print("[WARN] 没有采集到有效样本")
            return

        # 保存到文件
        filepath = os.path.join(self.output_dir, f"{self.current_pose_label}_samples.json")

        # 加载已有数据
        existing_data = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing_data = json.load(f)

        # 合并新数据
        existing_data.extend(self.collected_samples)

        # 保存
        with open(filepath, 'w') as f:
            json.dump(existing_data, f)

        print(f"[INFO] 已保存 {len(self.collected_samples)} 个样本到 {filepath}")
        print(f"[INFO] 该姿态总样本数: {len(existing_data)}")

        self.current_pose_label = None
        self.collected_samples = []

    def run(self):
        """运行数据收集"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("=" * 60)
        print("数据录制工具已启动")
        print("=" * 60)
        print("按键说明:")
        print("  's' - 录制坐姿")
        print("  't' - 录制站姿")
        print("  'l' - 录制躺姿")
        print("  'q' - 停止当前录制")
        print("  'ESC' - 退出程序")
        print("=" * 60)

        countdown_start_time = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # 镜像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe姿态估计
            results = self.pose.process(frame_rgb)

            # 绘制骨架
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # 状态机
            current_time = time.time()

            # 倒计时阶段
            if countdown_start_time is not None:
                elapsed = current_time - countdown_start_time
                remaining = self.countdown - int(elapsed)

                if remaining > 0:
                    frame = self.draw_countdown(frame, remaining)
                else:
                    # 倒计时结束，开始录制
                    countdown_start_time = None
                    self.is_recording = True
                    self.record_start_time = current_time
                    print(f"[INFO] 开始录制 {self.current_pose_label}!")

            # 录制阶段
            elif self.is_recording:
                elapsed = current_time - self.record_start_time
                frame = self.draw_recording_status(frame, elapsed)

                # 提取特征并保存
                if results.pose_world_landmarks:
                    # 转换为numpy数组
                    world_landmarks = np.zeros((17, 4), dtype=np.float32)
                    for mp_idx, coco_idx in self.MEDIAPIPE_TO_COCO.items():
                        wl = results.pose_world_landmarks.landmark[mp_idx]
                        world_landmarks[coco_idx] = [wl.x, wl.y, wl.z, wl.visibility]

                    # 提取特征
                    features = self.extract_features(world_landmarks)
                    if features is not None:
                        self.collected_samples.append({
                            'features': features.tolist(),
                            'label': self.current_pose_label,
                            'timestamp': current_time
                        })

            # 待机阶段
            else:
                frame = self.draw_instructions(frame)

            # 显示
            cv2.imshow('Data Collection Tool', frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                if self.is_recording:
                    self.stop_recording()
                break

            elif key == ord('s') and not self.is_recording and countdown_start_time is None:
                self.start_recording('sitting')
                countdown_start_time = current_time

            elif key == ord('t') and not self.is_recording and countdown_start_time is None:
                self.start_recording('standing')
                countdown_start_time = current_time

            elif key == ord('l') and not self.is_recording and countdown_start_time is None:
                self.start_recording('lying')
                countdown_start_time = current_time

            elif key == ord('q'):
                if countdown_start_time is not None:
                    # 取消倒计时
                    countdown_start_time = None
                    self.current_pose_label = None
                    print("[INFO] 已取消录制")
                elif self.is_recording:
                    # 停止录制
                    self.stop_recording()

        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

        print("\n[INFO] 数据收集已结束")
        print(f"[INFO] 数据保存在: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='姿态数据收集工具')
    parser.add_argument('--countdown', type=int, default=5,
                       help='按键后倒计时秒数 (默认: 5)')
    parser.add_argument('--min-duration', type=int, default=30,
                       help='建议最小录制时长/秒 (默认: 30)')

    args = parser.parse_args()

    collector = DataCollector(
        countdown=args.countdown,
        min_duration=args.min_duration
    )

    collector.run()


if __name__ == '__main__':
    main()
