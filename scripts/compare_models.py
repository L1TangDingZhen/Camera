#!/usr/bin/env python3
"""
模型对比工具
对比不同模型组合的性能和准确率
"""

import argparse
import time
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tabulate import tabulate


class ModelComparator:
    """模型对比器"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 测试组合
        self.test_combinations = self._get_test_combinations()

    def _get_test_combinations(self) -> List[Tuple[str, str, str]]:
        """获取测试组合"""
        device = self.config['device']

        if device == 'cuda:0' or device.startswith('cuda'):
            # GPU环境：测试所有组合
            return [
                ('yolov8n.pt', 'mediapipe', device),
                ('yolov8s.pt', 'mediapipe', device),
                ('yolov8s.pt', 'rtmpose', device),
                ('yolov8s.pt', 'vitpose', device),
            ]
        else:
            # CPU环境：只测试轻量组合
            return [
                ('yolov8n.pt', 'mediapipe', 'cpu'),
                ('yolov8s.pt', 'mediapipe', 'cpu'),
            ]

    def run(self, duration: int = 60):
        """
        运行对比测试

        Args:
            duration: 每个组合测试时长（秒）
        """
        print("\n" + "="*70)
        print("  模型对比测试")
        print("="*70)
        print(f"\n测试时长: {duration}秒/组合")
        print(f"测试组合数: {len(self.test_combinations)}\n")

        results = []

        for idx, (det_model, pose_backend, device) in enumerate(self.test_combinations, 1):
            print(f"\n[{idx}/{len(self.test_combinations)}] 测试组合:")
            print(f"  检测模型: {det_model}")
            print(f"  姿态模型: {pose_backend}")
            print(f"  设备: {device}")

            try:
                result = self._test_combination(det_model, pose_backend, device, duration)
                results.append(result)

                print(f"  ✓ 完成")
                print(f"    FPS: {result['fps']:.1f}")
                print(f"    延迟: {result['latency']:.1f}ms")

            except Exception as e:
                print(f"  ✗ 失败: {e}")
                results.append({
                    'detector': det_model,
                    'pose': pose_backend,
                    'device': device,
                    'fps': 0,
                    'latency': 0,
                    'error': str(e)
                })

        # 打印对比表格
        self._print_comparison_table(results)

    def _test_combination(self, det_model: str, pose_backend: str,
                         device: str, duration: int) -> Dict:
        """测试单个组合"""
        from src.detectors import PersonDetector, PoseEstimatorFactory

        # 创建检测器
        det_config = {
            'model': f'models/{det_model}' if not det_model.startswith('models/') else det_model,
            'device': device,
            'confidence': 0.5,
            'iou': 0.45
        }

        pose_config = {
            'backend': pose_backend,
            'device': device,
            'confidence': 0.3
        }

        if pose_backend == 'mediapipe':
            pose_config['complexity'] = 1

        detector = PersonDetector(det_config)
        pose_estimator = PoseEstimatorFactory.create(pose_config)

        # 打开摄像头
        cap = cv2.VideoCapture(self.config['camera']['source'])

        # 测试
        start_time = time.time()
        frame_count = 0
        latencies = []

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()

            # 检测
            bbox = detector.detect(frame)

            # 姿态估计
            if bbox is not None:
                keypoints = pose_estimator.estimate(frame, bbox)

            frame_end = time.time()

            latencies.append((frame_end - frame_start) * 1000)  # ms
            frame_count += 1

        cap.release()

        # 计算指标
        total_time = time.time() - start_time
        fps = frame_count / total_time if total_time > 0 else 0
        avg_latency = np.mean(latencies) if latencies else 0

        return {
            'detector': det_model,
            'pose': pose_backend,
            'device': device,
            'fps': fps,
            'latency': avg_latency,
            'frames': frame_count,
        }

    def _print_comparison_table(self, results: List[Dict]):
        """打印对比表格"""
        print("\n" + "="*70)
        print("  对比结果")
        print("="*70 + "\n")

        # 准备表格数据
        headers = ['检测模型', '姿态模型', '设备', 'FPS', '延迟(ms)', '帧数']
        table_data = []

        for result in results:
            if 'error' in result:
                row = [
                    result['detector'],
                    result['pose'],
                    result['device'],
                    '失败',
                    '失败',
                    '-'
                ]
            else:
                row = [
                    result['detector'],
                    result['pose'],
                    result['device'],
                    f"{result['fps']:.1f}",
                    f"{result['latency']:.1f}",
                    result['frames']
                ]

            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt='grid'))

        # 推荐
        print("\n推荐配置:\n")

        # 找到最高FPS的组合
        valid_results = [r for r in results if 'error' not in r and r['fps'] > 0]

        if valid_results:
            best_fps = max(valid_results, key=lambda x: x['fps'])
            print(f"  最高性能: {best_fps['detector']} + {best_fps['pose']}")
            print(f"    FPS: {best_fps['fps']:.1f}, 延迟: {best_fps['latency']:.1f}ms")

            # 如果是PC环境，找CPU可用的组合
            cpu_results = [r for r in valid_results if r['device'] == 'cpu']
            if cpu_results:
                best_cpu = max(cpu_results, key=lambda x: x['fps'])
                print(f"\n  X390推荐: {best_cpu['detector']} + {best_cpu['pose']}")
                print(f"    FPS: {best_cpu['fps']:.1f}, 延迟: {best_cpu['latency']:.1f}ms")

        print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型对比工具')

    parser.add_argument('--config', type=str, default='config/config_pc.yaml',
                       help='配置文件路径')
    parser.add_argument('--device', type=str, choices=['pc', 'x390', 'jetson'],
                       help='设备类型')
    parser.add_argument('--duration', type=int, default=60,
                       help='每个组合测试时长（秒）')

    args = parser.parse_args()

    # 根据设备选择配置文件
    if args.device:
        config_path = f'config/config_{args.device}.yaml'
    else:
        config_path = args.config

    # 检查配置文件
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return

    # 运行对比
    comparator = ModelComparator(config_path)
    comparator.run(duration=args.duration)


if __name__ == '__main__':
    main()
