#!/usr/bin/env python3
"""
YOLO Person Detection Model Testing and Benchmarking

测试多个YOLO模型的性能，支持：
- 单个或多个模型测试
- 自定义图片文件夹路径
- 详细的性能指标分析
- 结果保存到 yolo_test/ 文件夹

Usage:
    # 测试默认模型 (models/yolov8n.pt)
    python test_yolo_models.py

    # 测试指定图片文件夹
    python test_yolo_models.py --image-dir test_images

    # 测试多个模型
    python test_yolo_models.py --models models/yolov8n.pt models/yolov8s.pt

    # 测试并保存详细报告
    python test_yolo_models.py --save-results --output-dir yolo_test
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


class YOLOTester:
    """YOLO模型性能测试工具"""

    def __init__(self, image_dir="test_images", output_dir="yolo_test"):
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

    def _get_image_files(self):
        """获取所有图片文件"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        return sorted([
            f for f in self.image_dir.glob('*')
            if f.is_file() and f.suffix.lower() in extensions
        ])

    def test_model(self, model_path):
        """
        测试单个YOLO模型

        Args:
            model_path: 模型文件路径

        Returns:
            dict: 性能测试结果
        """
        model_path = Path(model_path)

        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return None

        print(f"\n{'='*80}")
        print(f"测试模型: {model_path.name}")
        print(f"{'='*80}")

        try:
            # 加载模型
            print(f"加载模型...")
            model = YOLO(str(model_path))

            # 获取模型信息
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"模型大小: {model_size_mb:.1f}MB")

            # 测试结果存储
            results = {
                'model_name': model_path.name,
                'model_path': str(model_path),
                'model_size_mb': model_size_mb,
                'timestamp': datetime.now().isoformat(),
                'test_images': [],
                'inference_times': [],
                'detection_counts': [],
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

                # 推理测试 (warmup + timing)
                inference_times = []

                # Warmup
                for _ in range(3):
                    model(img, verbose=False)

                # 计时推理
                for _ in range(10):
                    t0 = time.time()
                    pred = model(img, verbose=False)
                    inference_times.append((time.time() - t0) * 1000)

                # 计算统计
                avg_time = np.mean(inference_times)
                std_time = np.std(inference_times)

                # 获取检测结果
                if pred and len(pred) > 0:
                    boxes = pred[0].boxes
                    person_detections = []
                    confidences = []

                    for box in boxes:
                        if int(box.cls) == 0:  # COCO类别0是person
                            person_detections.append(box.xyxy.cpu().numpy()[0])
                            confidences.append(float(box.conf))

                    num_persons = len(person_detections)
                else:
                    num_persons = 0
                    confidences = []

                # 保存结果
                results['inference_times'].append(avg_time)
                results['detection_counts'].append(num_persons)
                if confidences:
                    results['confidence_scores'].extend(confidences)

                results['test_images'].append({
                    'filename': img_name,
                    'inference_time_ms': avg_time,
                    'std_time_ms': std_time,
                    'num_persons_detected': num_persons,
                    'confidences': confidences if confidences else None
                })

                print(f"✓ {avg_time:.1f}±{std_time:.1f}ms ({num_persons} person)")

            # 计算总体统计
            results['summary'] = {
                'total_images': len(self.image_files),
                'successful_inferences': len(results['inference_times']),
                'avg_inference_time_ms': float(np.mean(results['inference_times'])),
                'std_inference_time_ms': float(np.std(results['inference_times'])),
                'min_inference_time_ms': float(np.min(results['inference_times'])),
                'max_inference_time_ms': float(np.max(results['inference_times'])),
                'fps': float(1000.0 / np.mean(results['inference_times'])),
                'total_detections': int(np.sum(results['detection_counts'])),
                'avg_detections_per_image': float(np.mean(results['detection_counts'])),
                'avg_confidence': float(np.mean(results['confidence_scores']))
                    if results['confidence_scores'] else 0.0
            }

            return results

        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_results(self, results):
        """打印性能报告"""
        if not results:
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

        print(f"\n👥 检测统计:")
        print(f"  测试图片: {summary['total_images']}")
        print(f"  总检测人数: {summary['total_detections']}")
        print(f"  平均每张检测: {summary['avg_detections_per_image']:.2f} 人")
        print(f"  平均置信度: {summary['avg_confidence']:.3f}")

        print(f"\n📈 逐图详情:")
        for item in results['test_images']:
            print(f"  {item['filename']:<20} "
                  f"{item['inference_time_ms']:>7.1f}ms "
                  f"{item['num_persons_detected']:>2} person")

    def save_results(self, results):
        """保存结果到JSON和文本文件"""
        if not results:
            return

        model_name = results['model_name'].replace('.pt', '').replace('.engine', '')

        # 保存JSON
        json_file = self.output_dir / f"{model_name}_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ 结果保存: {json_file}")

        # 保存文本报告
        txt_file = self.output_dir / f"{model_name}_report.txt"
        with open(txt_file, 'w') as f:
            f.write(f"YOLO Model Performance Report\n")
            f.write(f"Model: {results['model_name']}\n")
            f.write(f"Path: {results['model_path']}\n")
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

            f.write(f"\n=== Detection Statistics ===\n")
            f.write(f"Total Images: {summary['total_images']}\n")
            f.write(f"Total Detections: {summary['total_detections']}\n")
            f.write(f"Avg Per Image: {summary['avg_detections_per_image']:.2f}\n")
            f.write(f"Avg Confidence: {summary['avg_confidence']:.3f}\n")

            f.write(f"\n=== Per-Image Details ===\n")
            for item in results['test_images']:
                f.write(f"{item['filename']:<25} "
                       f"{item['inference_time_ms']:>8.2f}ms "
                       f"{item['num_persons_detected']:>3} person\n")

        print(f"✓ 报告保存: {txt_file}")


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Person Detection Model Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_yolo_models.py
  python test_yolo_models.py --image-dir test_images
  python test_yolo_models.py --models models/yolov8n.pt models/yolov8s.pt
  python test_yolo_models.py --models models/yolov8n.pt --save-results
        """
    )

    parser.add_argument('--image-dir', type=str, default='test_images',
                       help='图片文件夹路径')
    parser.add_argument('--models', nargs='+', default=['models/yolov8n.pt'],
                       help='要测试的YOLO模型列表')
    parser.add_argument('--output-dir', type=str, default='yolo_test',
                       help='结果输出文件夹')
    parser.add_argument('--save-results', action='store_true',
                       help='保存详细结果到JSON/TXT')

    args = parser.parse_args()

    # 创建测试工具
    tester = YOLOTester(image_dir=args.image_dir, output_dir=args.output_dir)

    print(f"\n{'='*80}")
    print(f"YOLO 模型性能测试")
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

        print(f"{'Model':<20} {'FPS':>8} {'Avg Time':>12} {'Size':>8} {'Total Det':>10}")
        print(f"{'-'*70}")
        for result in all_results:
            summary = result['summary']
            print(f"{result['model_name']:<20} "
                  f"{summary['fps']:>8.1f} "
                  f"{summary['avg_inference_time_ms']:>10.2f}ms "
                  f"{result['model_size_mb']:>7.1f}MB "
                  f"{summary['total_detections']:>10}")

    print(f"\n{'='*80}")
    print(f"✅ 测试完成！")
    if args.save_results:
        print(f"结果已保存到: {args.output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
