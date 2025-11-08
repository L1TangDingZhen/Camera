#!/usr/bin/env python3
"""
SVM分类器训练脚本

使用方法:
    python train_svm.py --data-dir training_data --output models/pose_classifier_svm.pkl

训练完成后会输出:
    - 模型文件 (.pkl)
    - 准确率报告
    - 混淆矩阵
"""

import os
import json
import argparse
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple, List


class PoseClassifierTrainer:
    """姿态分类器训练器"""

    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.clf = None
        # 标签映射会在 load_data 中动态创建
        self.label_mapping = {}
        self.reverse_mapping = {}

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载训练数据

        Returns:
            X: 特征矩阵 (N, feature_dim)
            y: 标签向量 (N,)
        """
        all_features = []
        all_labels = []

        # 动态创建标签映射（只包含实际存在的类别）
        label_idx = 0
        for pose_label in ['sitting', 'standing', 'lying']:
            filepath = os.path.join(self.data_dir, f"{pose_label}_samples.json")

            if not os.path.exists(filepath):
                print(f"[WARN] 未找到 {pose_label} 的数据文件: {filepath}")
                continue

            # 为这个类别创建映射
            self.label_mapping[pose_label] = label_idx
            label_idx += 1

            with open(filepath, 'r') as f:
                samples = json.load(f)

            print(f"[INFO] 加载 {pose_label}: {len(samples)} 个样本")

            for sample in samples:
                all_features.append(sample['features'])
                all_labels.append(self.label_mapping[pose_label])

        # 创建反向映射
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}

        if len(all_features) == 0:
            raise ValueError("没有找到任何训练数据！请先运行 collect_data.py 收集数据。")

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)

        print(f"\n[INFO] 总样本数: {len(X)}")
        print(f"[INFO] 特征维度: {X.shape[1]}")

        # 检查类别分布
        unique, counts = np.unique(y, return_counts=True)
        print("\n[INFO] 类别分布:")
        for label_idx, count in zip(unique, counts):
            label_name = self.reverse_mapping[label_idx]
            print(f"  {label_name}: {count} ({count/len(y)*100:.1f}%)")

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
              grid_search: bool = True) -> dict:
        """训练SVM分类器

        Args:
            X: 特征矩阵
            y: 标签
            test_size: 测试集比例
            grid_search: 是否使用网格搜索寻找最优参数

        Returns:
            results: 训练结果字典
        """
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\n[INFO] 训练集: {len(X_train)} 样本")
        print(f"[INFO] 测试集: {len(X_test)} 样本")

        # 标准化特征
        print("\n[INFO] 标准化特征...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练SVM
        if grid_search:
            print("\n[INFO] 使用网格搜索寻找最优参数...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            }

            clf = GridSearchCV(
                SVC(probability=True, random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            clf.fit(X_train_scaled, y_train)

            print(f"\n[INFO] 最优参数: {clf.best_params_}")
            print(f"[INFO] 交叉验证准确率: {clf.best_score_:.4f}")

            self.clf = clf.best_estimator_
        else:
            print("\n[INFO] 训练SVM (默认参数)...")
            self.clf = SVC(kernel='rbf', probability=True, random_state=42)
            self.clf.fit(X_train_scaled, y_train)

        # 评估
        print("\n" + "=" * 60)
        print("训练结果")
        print("=" * 60)

        y_train_pred = self.clf.predict(X_train_scaled)
        y_test_pred = self.clf.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        print(f"\n训练集准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

        # 分类报告
        print("\n测试集分类报告:")
        print(classification_report(
            y_test, y_test_pred,
            target_names=[self.reverse_mapping[i] for i in sorted(self.reverse_mapping.keys())]
        ))

        # 混淆矩阵
        print("混淆矩阵:")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)

        # 概率校准检查
        print("\n[INFO] 概率输出测试:")
        test_probs = self.clf.predict_proba(X_test_scaled[:5])
        for i, probs in enumerate(test_probs):
            true_label = self.reverse_mapping[y_test[i]]
            pred_label = self.reverse_mapping[y_test_pred[i]]
            print(f"  样本{i+1}: 真实={true_label}, 预测={pred_label}")
            for j, prob in enumerate(probs):
                label_name = self.reverse_mapping[j]
                print(f"    {label_name}: {prob:.3f}")

        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'confusion_matrix': cm.tolist(),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
        }

        return results

    def save_model(self, output_path: str):
        """保存模型和scaler"""
        if self.clf is None:
            raise ValueError("模型未训练，无法保存")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        model_data = {
            'classifier': self.clf,
            'scaler': self.scaler,
            'label_mapping': self.label_mapping,
            'reverse_mapping': self.reverse_mapping
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n[INFO] 模型已保存到: {output_path}")

        # 保存元数据
        meta_path = output_path.replace('.pkl', '_meta.json')
        metadata = {
            'label_mapping': self.label_mapping,
            'feature_dim': int(self.scaler.n_features_in_),  # 转换为Python int
            'n_samples_trained': int(self.clf.n_support_.sum()),  # 转换为Python int
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[INFO] 元数据已保存到: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description='训练姿态分类器')
    parser.add_argument('--data-dir', type=str, default='training_data',
                       help='训练数据目录 (默认: training_data)')
    parser.add_argument('--output', type=str, default='models/pose_classifier_svm.pkl',
                       help='输出模型路径 (默认: models/pose_classifier_svm.pkl)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='测试集比例 (默认: 0.2)')
    parser.add_argument('--no-grid-search', action='store_true',
                       help='禁用网格搜索，使用默认参数')

    args = parser.parse_args()

    print("=" * 60)
    print("姿态分类器训练")
    print("=" * 60)

    trainer = PoseClassifierTrainer(data_dir=args.data_dir)

    # 加载数据
    X, y = trainer.load_data()

    # 训练
    results = trainer.train(
        X, y,
        test_size=args.test_size,
        grid_search=not args.no_grid_search
    )

    # 保存
    trainer.save_model(args.output)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"测试集准确率: {results['test_accuracy']*100:.2f}%")
    print(f"模型文件: {args.output}")
    print("\n使用方法:")
    print(f"  python main.py --config config/config_gpu.yaml")
    print("  (程序会自动加载训练好的模型)")
    print("=" * 60)


if __name__ == '__main__':
    main()
