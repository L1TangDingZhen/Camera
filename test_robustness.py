#!/usr/bin/env python3
"""
模型鲁棒性测试工具
测试SVM模型在不同环境下的准确率

使用方法:
    python test_robustness.py --label sitting --duration 30

会记录30秒内的预测结果，计算准确率
"""

import cv2
import numpy as np
import argparse
import time
import yaml
from pathlib import Path
from collections import Counter

from src.detectors import PersonDetector, PoseEstimatorFactory
from src.state import BehaviorStateMachine, ROIManager


class RobustnessTest:
    """鲁棒性测试工具"""

    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化组件
        print("[初始化] 加载人体检测器...")
        self.person_detector = PersonDetector(self.config['models']['person'])

        print("[初始化] 加载姿态估计器...")
        self.pose_estimator = PoseEstimatorFactory.create(self.config['models']['pose'])

        print("[初始化] 创建状态机...")
        roi_manager = ROIManager(self.config.get('roi', {}))
        self.state_machine = BehaviorStateMachine(self.config, roi_manager, database=None)

        # 初始化摄像头
        print("[初始化] 打开摄像头...")
        camera_config = self.config['camera']
        self.cap = cv2.VideoCapture(camera_config['source'])
        # Force MJPEG encoding for better bandwidth efficiency
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['resolution'][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['resolution'][1])
        self.cap.set(cv2.CAP_PROP_FPS, camera_config['fps'])

    def test_accuracy(self, ground_truth_label: str, duration: int = 30):
        """
        测试准确率

        Args:
            ground_truth_label: 真实标签 (sitting/standing/lying)
            duration: 测试时长（秒）
        """
        print(f"\n{'='*60}")
        print(f"  鲁棒性测试")
        print(f"  真实姿态: {ground_truth_label}")
        print(f"  测试时长: {duration} 秒")
        print(f"{'='*60}\n")

        print(f"请保持 {ground_truth_label} 姿势，测试将在3秒后开始...")
        time.sleep(3)

        predictions = []
        confidences = []
        keypoint_qualities = []

        start_time = time.time()
        frame_count = 0

        cv2.namedWindow('Robustness Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Robustness Test', 1280, 720)

        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()

            # 检测
            bbox = self.person_detector.detect(frame)
            keypoints = None
            world_landmarks = None

            if bbox is not None:
                keypoints = self.pose_estimator.estimate(frame, bbox)
                if hasattr(self.pose_estimator, 'get_world_landmarks'):
                    world_landmarks = self.pose_estimator.get_world_landmarks()

            # 更新状态机
            self.state_machine.update(bbox, keypoints, current_time, world_landmarks)
            current_state = self.state_machine.get_current_state()

            # 记录预测
            predictions.append(current_state.value)

            # 记录关键点质量
            if keypoints is not None:
                avg_confidence = np.mean(keypoints[:, 2])
                keypoint_qualities.append(avg_confidence)
            else:
                keypoint_qualities.append(0.0)

            # 记录SVM置信度（如果可用）
            if hasattr(self.state_machine, 'last_probabilities') and self.state_machine.last_probabilities:
                probs = self.state_machine.last_probabilities
                if current_state.value in probs:
                    confidences.append(probs[current_state.value])
                else:
                    confidences.append(0.0)

            # 可视化
            vis_frame = frame.copy()

            # 绘制进度
            elapsed = int(current_time - start_time)
            remaining = duration - elapsed
            cv2.putText(vis_frame, f"Time: {remaining}s", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # 绘制当前预测
            color = (0, 255, 0) if current_state.value == ground_truth_label else (0, 0, 255)
            cv2.putText(vis_frame, f"Ground Truth: {ground_truth_label}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis_frame, f"Prediction: {current_state.value}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

            # 绘制关键点质量
            if keypoints is not None:
                quality = np.mean(keypoints[:, 2])
                cv2.putText(vis_frame, f"Keypoint Quality: {quality:.2f}", (20, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Robustness Test', vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.cap.release()

        # 计算统计数据
        self._print_results(ground_truth_label, predictions, confidences, keypoint_qualities, frame_count)

    def _print_results(self, ground_truth: str, predictions: list,
                      confidences: list, qualities: list, frame_count: int):
        """打印测试结果"""
        print(f"\n{'='*60}")
        print(f"  测试结果")
        print(f"{'='*60}\n")

        # 计算准确率
        correct = sum(1 for p in predictions if p == ground_truth)
        accuracy = correct / len(predictions) * 100 if predictions else 0

        print(f"总帧数: {frame_count}")
        print(f"有效预测: {len(predictions)} 帧")
        print(f"正确预测: {correct} 帧")
        print(f"准确率: {accuracy:.2f}%\n")

        # 预测分布
        counter = Counter(predictions)
        print("预测分布:")
        for state, count in counter.most_common():
            percentage = count / len(predictions) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {state:10s}: {count:4d} 帧 ({percentage:5.1f}%) {bar}")

        # 关键点质量
        if qualities:
            avg_quality = np.mean(qualities)
            min_quality = np.min(qualities)
            max_quality = np.max(qualities)
            print(f"\n关键点检测质量:")
            print(f"  平均: {avg_quality:.3f}")
            print(f"  最小: {min_quality:.3f}")
            print(f"  最大: {max_quality:.3f}")

        # SVM置信度
        if confidences:
            avg_conf = np.mean(confidences)
            min_conf = np.min(confidences)
            max_conf = np.max(confidences)
            print(f"\nSVM预测置信度:")
            print(f"  平均: {avg_conf:.3f}")
            print(f"  最小: {min_conf:.3f}")
            print(f"  最大: {max_conf:.3f}")

        # 诊断建议
        print(f"\n{'='*60}")
        print(f"  诊断建议")
        print(f"{'='*60}\n")

        if accuracy >= 90:
            print("✅ 准确率优秀 (≥90%)，模型在当前环境下工作良好")
        elif accuracy >= 80:
            print("⚠️  准确率良好 (80-90%)，建议：")
            print("   - 检查光线是否充足")
            print("   - 确认摄像头位置能看到全身")
        elif accuracy >= 70:
            print("⚠️  准确率一般 (70-80%)，建议：")
            print("   - 增加当前环境的训练数据")
            print("   - 改善照明条件")
            print("   - 调整摄像头角度/距离")
        else:
            print("🔴 准确率较低 (<70%)，需要：")
            print("   - 在当前环境重新收集训练数据")
            print("   - 检查摄像头是否能清晰看到全身")
            print("   - 增加照明")

        if qualities and np.mean(qualities) < 0.5:
            print("\n🔴 关键点检测质量差 (<0.5):")
            print("   - 增加光线（开灯/开窗）")
            print("   - 调整摄像头位置（确保全身可见）")
            print("   - 检查摄像头是否对焦清晰")

        print()


def main():
    parser = argparse.ArgumentParser(description='模型鲁棒性测试工具')

    parser.add_argument('--config', type=str, default='config/config_cpu.yaml',
                       help='配置文件路径')
    parser.add_argument('--label', type=str, required=True,
                       choices=['sitting', 'standing', 'lying'],
                       help='真实姿态标签')
    parser.add_argument('--duration', type=int, default=30,
                       help='测试时长（秒）')

    args = parser.parse_args()

    # 检查配置文件
    if not Path(args.config).exists():
        print(f"错误: 配置文件不存在: {args.config}")
        return

    # 运行测试
    tester = RobustnessTest(args.config)
    tester.test_accuracy(args.label, args.duration)


if __name__ == '__main__':
    main()
