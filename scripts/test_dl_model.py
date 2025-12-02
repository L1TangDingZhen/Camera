#!/usr/bin/env python3
"""测试深度学习模型的快速脚本

用法：
    # 实时测试LSTM模型
    python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --type lstm

    # 对比SVM vs LSTM
    python scripts/test_dl_model.py --compare

    # 在测试集上评估
    python scripts/test_dl_model.py --model models/pose_classifier_lstm.pth --test-data data/test.npz
"""

import argparse
import cv2
import numpy as np
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.detectors.person_detector import PersonDetector
from src.detectors.pose_estimator import MediaPipePoseEstimator
from src.classifiers.pose_classifier import PoseClassifierSVM
from src.classifiers.pose_classifier_dl import PoseClassifierDL
import yaml


class ModelTester:
    """模型测试器"""

    def __init__(self, config_path='config/config_cpu.yaml'):
        with open(config_path, encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.person_detector = PersonDetector(self.config['models']['person'])
        self.pose_estimator = MediaPipePoseEstimator(self.config)

    def test_realtime(self, model_path, model_type='lstm', compare_svm=False):
        """实时测试模型"""

        # 加载DL模型
        dl_model = PoseClassifierDL(
            model_path=model_path,
            model_type=model_type
        )

        if not dl_model.is_loaded:
            print(f"[ERROR] 模型加载失败: {model_path}")
            return

        # 如果需要对比，加载SVM
        svm_model = None
        if compare_svm:
            svm_model = PoseClassifierSVM()
            if not svm_model.is_loaded:
                print("[WARN] SVM未加载，无法对比")
                compare_svm = False

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] 无法打开摄像头")
            return

        print(f"\n{'='*60}")
        print(f"实时测试 - {model_type.upper()} 模型")
        print(f"{'='*60}\n")

        if compare_svm:
            print("对比模式: DL vs SVM")
        print("按 'q' 退出\n")

        frame_times = []
        dl_predictions = []
        svm_predictions = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()

                # 检测
                bbox = self.person_detector.detect(frame)

                if bbox is not None:
                    keypoints = self.pose_estimator.estimate(frame, bbox)

                    if keypoints is not None:
                        world_landmarks = self.pose_estimator.get_world_landmarks()

                        if world_landmarks is None:
                            continue
                        # DL预测
                        dl_probs = dl_model.predict_proba(world_landmarks)
                        dl_pred = max(dl_probs, key=dl_probs.get) if dl_probs else None

                        # SVM预测（如果需要）
                        svm_pred = None
                        svm_probs = None
                        if compare_svm and svm_model:
                            svm_probs = svm_model.predict_proba(world_landmarks)
                            svm_pred = max(svm_probs, key=svm_probs.get) if svm_probs else None

                        # 记录
                        if dl_pred:
                            dl_predictions.append(dl_pred)
                        if svm_pred:
                            svm_predictions.append(svm_pred)

                        # 可视化
                        self._draw_comparison(
                            frame, keypoints, bbox,
                            dl_pred, dl_probs,
                            svm_pred, svm_probs,
                            compare_svm
                        )

                # 计算FPS
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                fps = 1.0 / (np.mean(frame_times[-30:]) + 1e-6)

                # 显示FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Model Test', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        # 统计
        self._print_statistics(dl_predictions, svm_predictions, frame_times, compare_svm)

    def _draw_comparison(self, frame, keypoints, bbox, dl_pred, dl_probs,
                        svm_pred, svm_probs, compare):
        """绘制对比"""
        # 边界框
        x1, y1, x2, y2 = bbox[:4].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 关键点
        if keypoints is not None:
            for kp in keypoints:
                x, y, conf = kp
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        # DL预测
        y_offset = 30
        if dl_pred and dl_probs:
            cv2.putText(frame, f"DL: {dl_pred}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 30

            # 概率条
            for i, (label, prob) in enumerate(dl_probs.items()):
                if label in ['sitting', 'standing', 'lying']:
                    bar_length = int(prob * 200)
                    color = (0, 255, 0) if label == dl_pred else (100, 100, 100)
                    cv2.rectangle(frame, (10, y_offset + i*25),
                                (10 + bar_length, y_offset + i*25 + 15),
                                color, -1)
                    cv2.putText(frame, f"{label[:3]}: {prob:.2f}",
                               (220, y_offset + i*25 + 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # SVM预测（如果对比）
        if compare and svm_pred and svm_probs:
            y_offset = 200
            cv2.putText(frame, f"SVM: {svm_pred}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            y_offset += 30

            for i, (label, prob) in enumerate(svm_probs.items()):
                if label in ['sitting', 'standing', 'lying']:
                    bar_length = int(prob * 200)
                    color = (255, 200, 0) if label == svm_pred else (100, 100, 100)
                    cv2.rectangle(frame, (10, y_offset + i*25),
                                (10 + bar_length, y_offset + i*25 + 15),
                                color, -1)
                    cv2.putText(frame, f"{label[:3]}: {prob:.2f}",
                               (220, y_offset + i*25 + 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 一致性指示
            if dl_pred == svm_pred:
                cv2.putText(frame, "MATCH ✓", (10, 350),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "DIFFER ✗", (10, 350),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _print_statistics(self, dl_preds, svm_preds, frame_times, compare):
        """打印统计信息"""
        print(f"\n{'='*60}")
        print("测试统计")
        print(f"{'='*60}\n")

        # FPS
        avg_fps = 1.0 / (np.mean(frame_times) + 1e-6)
        print(f"平均FPS: {avg_fps:.1f}")
        print(f"平均帧时间: {np.mean(frame_times)*1000:.1f}ms")

        # DL预测分布
        if dl_preds:
            print(f"\nDL预测分布 (总共 {len(dl_preds)} 帧):")
            for label in ['sitting', 'standing', 'lying']:
                count = dl_preds.count(label)
                print(f"  {label}: {count} ({count/len(dl_preds)*100:.1f}%)")

        # 对比
        if compare and svm_preds and len(dl_preds) == len(svm_preds):
            agreement = sum(1 for d, s in zip(dl_preds, svm_preds) if d == s)
            print(f"\nDL vs SVM:")
            print(f"  一致率: {agreement}/{len(dl_preds)} ({agreement/len(dl_preds)*100:.1f}%)")

            # 不一致的case
            disagreements = [(d, s) for d, s in zip(dl_preds, svm_preds) if d != s]
            if disagreements:
                print(f"  不一致样例 (前10个):")
                for i, (d, s) in enumerate(disagreements[:10]):
                    print(f"    {i+1}. DL={d}, SVM={s}")

        print()

    def test_on_dataset(self, model_path, model_type, test_data_path):
        """在测试集上评估"""
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # 加载模型
        model = PoseClassifierDL(model_path=model_path, model_type=model_type)
        if not model.is_loaded:
            print(f"[ERROR] 模型加载失败")
            return

        # 加载数据
        data = np.load(test_data_path)
        landmarks = data['landmarks']
        labels = data['labels']

        print(f"\n{'='*60}")
        print(f"测试集评估 - {model_type.upper()}")
        print(f"{'='*60}\n")
        print(f"测试集大小: {len(landmarks)}")

        # 预测
        predictions = []
        for lm in landmarks:
            pred = model.predict(lm)
            if pred:
                label_map = {'sitting': 0, 'standing': 1, 'lying': 2}
                predictions.append(label_map[pred])
            else:
                predictions.append(-1)  # 无法预测

        predictions = np.array(predictions)

        # 过滤无效预测
        valid_mask = predictions >= 0
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]

        # 计算指标
        accuracy = accuracy_score(labels, predictions)
        print(f"\n准确率: {accuracy*100:.2f}%")

        # 分类报告
        print("\n分类报告:")
        print(classification_report(
            labels, predictions,
            target_names=['Sitting', 'Standing', 'Lying']
        ))

        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        print("混淆矩阵:")
        print(cm)
        print()


def main():
    parser = argparse.ArgumentParser(description='测试深度学习模型')
    parser.add_argument('--model', type=str,
                       default='models/pose_classifier_lstm.pth',
                       help='模型路径')
    parser.add_argument('--type', type=str, default='lstm',
                       choices=['mlp', 'lstm', 'transformer'],
                       help='模型类型')
    parser.add_argument('--compare', action='store_true',
                       help='对比DL和SVM')
    parser.add_argument('--test-data', type=str,
                       help='测试集路径（如果指定，在数据集上评估而非实时测试）')
    parser.add_argument('--config', type=str, default='config/config_cpu.yaml',
                       help='配置文件')

    args = parser.parse_args()

    tester = ModelTester(config_path=args.config)

    if args.test_data:
        # 数据集评估
        tester.test_on_dataset(args.model, args.type, args.test_data)
    else:
        # 实时测试
        tester.test_realtime(args.model, args.type, args.compare)


if __name__ == '__main__':
    main()
