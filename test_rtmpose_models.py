#!/usr/bin/env python3
"""
RTMPose Pose Estimation Model Testing and Benchmarking

测试多个RTMPose模型的性能，支持：
- 单个或多个模型测试
- 自定义图片文件夹路径
- 详细的性能指标分析
- 结果保存到 rtmpose_test/ 文件夹

Usage:
    # 测试默认模型
    python test_rtmpose_models.py

    # 测试指定图片文件夹
    python test_rtmpose_models.py --image-dir test_images

    # 测试多个模型
    python test_rtmpose_models.py --models models/rtmpose/rtmpose-s-fp32.engine models/rtmpose/rtmpose-s-fp16.engine

    # 测试并保存详细报告
    python test_rtmpose_models.py --save-results --output-dir rtmpose_test
"""

import os
import sys
import json
import time
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2

try:
    from src.detectors import PoseEstimatorFactory
    from src.detectors.person_detector import PersonDetector
except ImportError:
    print("❌ src.detectors not available")
    sys.exit(1)


class RTMPoseTester:
    """RTMPose模型性能测试工具"""

    def __init__(self, image_dir="test_images", output_dir="rtmpose_test"):
        """
        初始化测试工具

        Args:
            image_dir: 图片文件夹路径
            output_dir: 结果输出文件夹
        """
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.image_dir.exists():
            print(f"❌ 图片文件夹不存在: {self.image_dir}")
            sys.exit(1)

        # 获取图片列表
        self.image_files = self._get_image_files()
        if not self.image_files:
            print(f"❌ 未找到图片: {self.image_dir}")
            sys.exit(1)

        print(f"✓ 找到 {len(self.image_files)} 张测试图片")

        # 初始化YOLO检测器（所有RTMPose模型共用）
        print(f"初始化YOLO人体检测器...")
        try:
            from ultralytics import YOLO
            self.detector = YOLO('models/yolov8n.pt')
        except Exception as e:
            print(f"⚠️  YOLO初始化失败: {e}")
            self.detector = None

    def _get_image_files(self):
        """获取所有图片文件"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        return sorted([
            f for f in self.image_dir.glob('*')
            if f.is_file() and f.suffix.lower() in extensions
        ])

    def test_model(self, model_config):
        """
        测试单个RTMPose模型

        Args:
            model_config: dict or str，模型配置或引擎路径

        Returns:
            dict: 性能测试结果
        """
        if isinstance(model_config, str):
            model_path = Path(model_config)
            model_name = model_path.name

            # 加载基础配置
            base_config_path = 'config/config_gpu.yaml'
            try:
                with open(base_config_path, 'r') as f:
                    base_config = yaml.safe_load(f)
                config = base_config['models']['pose'].copy()
            except:
                # Fallback if config不可用
                config = {
                    'backend': 'rtmpose',
                    'model': 'rtmpose-s',
                    'config_file': 'models/rtmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py'
                }

            # 始终使用rtmpose后端（无论是engine还是pth文件）
            config['backend'] = 'rtmpose'
            config['checkpoint'] = str(model_path)
        else:
            config = model_config
            model_name = config.get('checkpoint', 'unknown').split('/')[-1]

        print(f"\n{'='*80}")
        print(f"测试模型: {model_name}")
        print(f"{'='*80}")

        try:
            # 加载模型
            print(f"加载模型...")
            estimator = PoseEstimatorFactory.create(config)

            # 获取模型文件大小
            checkpoint = config.get('checkpoint', '')
            if checkpoint and Path(checkpoint).exists():
                model_size_mb = Path(checkpoint).stat().st_size / (1024 * 1024)
            else:
                model_size_mb = 0

            print(f"模型类型: {config.get('backend', 'unknown')}")
            if model_size_mb > 0:
                print(f"模型大小: {model_size_mb:.1f}MB")

            # 测试结果存储
            results = {
                'model_name': model_name,
                'model_config': config,
                'model_size_mb': model_size_mb,
                'timestamp': datetime.now().isoformat(),
                'test_images': [],
                'inference_times': [],
                'keypoint_counts': [],
                'confidence_scores': []
            }

            # 处理每张图片
            print(f"\n处理 {len(self.image_files)} 张图片...\n")

            for i, img_path in enumerate(self.image_files):
                img_name = img_path.name
                print(f"  [{i+1}/{len(self.image_files)}] {img_name}...", end=' ', flush=True)

                # 读取图片
                img = cv2.imread(str(img_path))
                if img is None:
                    print("❌ 读取失败")
                    continue

                # 检测人体
                if self.detector:
                    try:
                        pred = self.detector(img, verbose=False)
                        if pred and len(pred) > 0:
                            boxes = pred[0].boxes
                            persons = [box.xyxy.cpu().numpy()[0] for box in boxes
                                      if int(box.cls) == 0]
                            if not persons:
                                print("⚠️  未检测到人体")
                                results['test_images'].append({
                                    'filename': img_name,
                                    'inference_time_ms': 0,
                                    'std_time_ms': 0,
                                    'num_keypoints': 0,
                                    'confidence': 0
                                })
                                continue

                            bbox = persons[0]
                        else:
                            print("⚠️  未检测到人体")
                            results['test_images'].append({
                                'filename': img_name,
                                'inference_time_ms': 0,
                                'std_time_ms': 0,
                                'num_keypoints': 0,
                                'confidence': 0
                            })
                            continue
                    except Exception as e:
                        print(f"⚠️  检测失败: {e}")
                        continue
                else:
                    # 如果YOLO不可用，手动创建bbox
                    h, w = img.shape[:2]
                    bbox = [0, 0, w, h, 0.9]

                # 推理测试 (warmup + timing)
                inference_times = []

                # Warmup
                for _ in range(3):
                    _ = estimator.estimate(img, bbox)

                # 计时推理
                for _ in range(10):
                    t0 = time.time()
                    keypoints = estimator.estimate(img, bbox)
                    inference_times.append((time.time() - t0) * 1000)

                # 计算统计
                avg_time = np.mean(inference_times)
                std_time = np.std(inference_times)

                # 统计关键点
                if keypoints is not None and len(keypoints) > 0:
                    num_keypoints = len(keypoints)
                    # 计算平均置信度
                    confidences = [kpt[2] if len(kpt) > 2 else 0 for kpt in keypoints]
                    avg_confidence = np.mean(confidences)
                else:
                    num_keypoints = 0
                    avg_confidence = 0

                # 保存结果
                results['inference_times'].append(avg_time)
                results['keypoint_counts'].append(num_keypoints)
                if num_keypoints > 0:
                    results['confidence_scores'].append(avg_confidence)

                results['test_images'].append({
                    'filename': img_name,
                    'inference_time_ms': avg_time,
                    'std_time_ms': std_time,
                    'num_keypoints': num_keypoints,
                    'confidence': avg_confidence
                })

                print(f"✓ {avg_time:.1f}±{std_time:.1f}ms ({num_keypoints} kpts)")

            # 计算总体统计
            if results['inference_times']:
                results['summary'] = {
                    'total_images': len(self.image_files),
                    'successful_inferences': len(results['inference_times']),
                    'avg_inference_time_ms': float(np.mean(results['inference_times'])),
                    'std_inference_time_ms': float(np.std(results['inference_times'])),
                    'min_inference_time_ms': float(np.min(results['inference_times'])),
                    'max_inference_time_ms': float(np.max(results['inference_times'])),
                    'fps': float(1000.0 / np.mean(results['inference_times'])),
                    'total_keypoints': int(np.sum(results['keypoint_counts'])),
                    'avg_keypoints_per_image': float(np.mean(results['keypoint_counts'])),
                    'avg_confidence': float(np.mean(results['confidence_scores']))
                        if results['confidence_scores'] else 0.0
                }
            else:
                results['summary'] = {
                    'total_images': len(self.image_files),
                    'successful_inferences': 0,
                    'fps': 0,
                    'avg_confidence': 0
                }

            return results

        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_results(self, results):
        """打印性能报告"""
        if not results or 'summary' not in results:
            return

        summary = results['summary']

        print(f"\n{'='*80}")
        print(f"性能报告: {results['model_name']}")
        print(f"{'='*80}")

        print(f"\n📊 推理性能:")
        print(f"  平均推理时间: {summary['avg_inference_time_ms']:.2f}ms")
        print(f"  标准差: {summary['std_inference_time_ms']:.2f}ms")
        print(f"  范围: [{summary['min_inference_time_ms']:.2f}, {summary['max_inference_time_ms']:.2f}]ms")
        print(f"  FPS: {summary['fps']:.1f}")

        print(f"\n🦴 关键点统计:")
        print(f"  测试图片: {summary['total_images']}")
        print(f"  总关键点: {summary['total_keypoints']}")
        print(f"  平均每张: {summary['avg_keypoints_per_image']:.1f}")
        print(f"  平均置信度: {summary['avg_confidence']:.3f}")

        print(f"\n📈 逐图详情:")
        for item in results['test_images']:
            print(f"  {item['filename']:<20} "
                  f"{item['inference_time_ms']:>7.1f}ms "
                  f"{item['num_keypoints']:>2} kpts "
                  f"{item['confidence']:>6.3f}")

    def save_results(self, results):
        """保存结果到JSON和文本文件"""
        if not results:
            return

        model_name = results['model_name'].replace('.pt', '').replace('.engine', '')

        # 保存JSON
        json_file = self.output_dir / f"{model_name}_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ 结果保存: {json_file}")

        # 保存文本报告
        txt_file = self.output_dir / f"{model_name}_report.txt"
        with open(txt_file, 'w') as f:
            f.write(f"RTMPose Model Performance Report\n")
            f.write(f"Model: {results['model_name']}\n")
            f.write(f"Config: {results['model_config']}\n")
            if results['model_size_mb'] > 0:
                f.write(f"Size: {results['model_size_mb']:.1f}MB\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"\n")

            summary = results['summary']
            f.write(f"=== Performance Summary ===\n")
            f.write(f"Average Inference Time: {summary['avg_inference_time_ms']:.2f}ms\n")
            f.write(f"Std Dev: {summary['std_inference_time_ms']:.2f}ms\n")
            f.write(f"Range: [{summary['min_inference_time_ms']:.2f}, "
                   f"{summary['max_inference_time_ms']:.2f}]ms\n")
            f.write(f"FPS: {summary['fps']:.1f}\n")

            f.write(f"\n=== Keypoint Statistics ===\n")
            f.write(f"Total Images: {summary['total_images']}\n")
            f.write(f"Total Keypoints: {summary['total_keypoints']}\n")
            f.write(f"Avg Per Image: {summary['avg_keypoints_per_image']:.1f}\n")
            f.write(f"Avg Confidence: {summary['avg_confidence']:.3f}\n")

            f.write(f"\n=== Per-Image Details ===\n")
            for item in results['test_images']:
                f.write(f"{item['filename']:<25} "
                       f"{item['inference_time_ms']:>8.2f}ms "
                       f"{item['num_keypoints']:>3} kpts "
                       f"{item['confidence']:>7.3f}\n")

        print(f"✓ 报告保存: {txt_file}")


def main():
    parser = argparse.ArgumentParser(
        description='RTMPose Pose Estimation Model Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_rtmpose_models.py
  python test_rtmpose_models.py --image-dir test_images
  python test_rtmpose_models.py --models models/rtmpose/rtmpose-s-fp32.engine
  python test_rtmpose_models.py --models models/rtmpose/rtmpose-s-fp32.engine models/rtmpose/rtmpose-s-fp16.engine
  python test_rtmpose_models.py --models models/rtmpose/rtmpose-s-fp16.engine --save-results
        """
    )

    parser.add_argument('--image-dir', type=str, default='test_images',
                       help='图片文件夹路径')
    parser.add_argument('--models', nargs='+', default=['models/rtmpose/rtmpose-s-fp32.engine'],
                       help='要测试的RTMPose模型列表')
    parser.add_argument('--output-dir', type=str, default='rtmpose_test',
                       help='结果输出文件夹')
    parser.add_argument('--save-results', action='store_true',
                       help='保存详细结果到JSON/TXT')

    args = parser.parse_args()

    # 创建测试工具
    tester = RTMPoseTester(image_dir=args.image_dir, output_dir=args.output_dir)

    print(f"\n{'='*80}")
    print(f"RTMPose 模型性能测试")
    print(f"{'='*80}")
    print(f"图片文件夹: {args.image_dir}")
    print(f"待测模型: {len(args.models)}")
    for model in args.models:
        print(f"  - {model}")

    # 测试每个模型
    all_results = []
    for model_path in args.models:
        results = tester.test_model(model_path)
        if results:
            tester.print_results(results)
            if args.save_results:
                tester.save_results(results)
            all_results.append(results)

    # 对比报告
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"模型对比报告")
        print(f"{'='*80}\n")

        print(f"{'Model':<25} {'FPS':>8} {'Avg Time':>12} {'Size':>8} {'Avg Conf':>10}")
        print(f"{'-'*75}")
        for result in all_results:
            summary = result['summary']
            print(f"{result['model_name']:<25} "
                  f"{summary['fps']:>8.1f} "
                  f"{summary['avg_inference_time_ms']:>10.2f}ms "
                  f"{result['model_size_mb']:>7.1f}MB "
                  f"{summary['avg_confidence']:>10.3f}")

    print(f"\n{'='*80}")
    print(f"✅ 测试完成！")
    if args.save_results:
        print(f"结果已保存到: {args.output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
