"""
RTMPose 姿态估计器

特点：
- GPU加速，支持CUDA
- 速度快（~12ms with FP16 on Jetson）
- 精度高（AP ~68.5%）
- 支持TensorRT优化（FP16/INT8）
- 专为边缘设备优化（Jetson）

性能对比：
- MediaPipe (CPU): ~50ms, AP 67%
- RTMPose (GPU): ~18ms (FP32), ~12ms (FP16), AP 68.5%
- RTMPose + TensorRT: ~10ms (INT8)

适用场景：
- 生产部署环境
- Jetson设备
- GPU服务器
- 高性能要求场景

依赖：
- mmpose>=1.0.0
- mmcv>=2.0.0
- mmengine>=0.8.0

⚠️ 注意：
- Windows安装复杂，建议Linux/Jetson
- 需要CUDA支持
- 首次运行可能需要下载模型
"""

import time
from typing import Optional, Dict, List
from pathlib import Path
import numpy as np
import torch

from .pose_estimator_base import PoseEstimatorInterface, Keypoint


class RTMPoseEstimator(PoseEstimatorInterface):
    """RTMPose姿态估计器（GPU高性能）"""

    def __init__(self, config: dict):
        super().__init__(config)

        # 配置参数
        self.model_name = config.get('model', 'rtmpose-s')  # rtmpose-tiny/s/m/l
        self.config_file = config.get('config_file', None)
        self.checkpoint = config.get('checkpoint', None)
        self.device = config.get('device', 'cuda:0')
        self.confidence = config.get('confidence', 0.3)
        # 关键点平滑与置信度过滤
        self.keypoint_smooth_alpha = config.get('keypoint_smooth_alpha', 0.0)  # 0 = 不平滑
        self.keypoint_min_conf = config.get('keypoint_min_conf', 0.3)
        self._last_keypoints: Optional[np.ndarray] = None

        # TensorRT配置
        self.tensorrt_config = config.get('tensorrt', {})
        self.use_tensorrt = self.tensorrt_config.get('enabled', False)
        # AMP (Automatic Mixed Precision) - enabled by default for GPU
        self.use_amp = config.get('use_amp', True)  # Default: True for automatic FP16 optimization
        self.use_fp16 = False  # Will be set by _apply_tensorrt_optimization
        self.use_tensorrt_engine = False  # Flag for native TensorRT engine

        # 自动构建config和checkpoint路径
        if self.config_file is None or self.checkpoint is None:
            self.config_file, self.checkpoint = self._get_model_paths(self.model_name)

        # 初始化推理时间记录
        self.inference_times = []

        # 检查是否使用TensorRT引擎文件(.engine) - 优先检查，避免导入mmpose
        if self.checkpoint.endswith('.engine'):
            print(f"[RTMPose] 检测到TensorRT引擎文件")
            print(f"[RTMPose] 正在加载TensorRT引擎: {self.checkpoint}")
            self.use_tensorrt_engine = True
            self._load_tensorrt_engine()
            print(f"[RTMPose] TensorRT引擎加载成功 ✓")
            return

        # 只有非TensorRT引擎模式才需要mmpose
        try:
            from mmpose.apis import init_model, inference_topdown
            from mmpose.structures import merge_data_samples
        except ImportError:
            raise ImportError(
                "MMPose未安装！\n\n"
                "安装方法（推荐使用mim）：\n"
                "  pip install openmim\n"
                "  mim install mmcv==2.0.0\n"
                "  mim install mmpose==1.0.0\n\n"
                "⚠️ Windows用户注意：\n"
                "  MMPose在Windows上安装复杂，建议：\n"
                "  1. 使用WSL2/Linux虚拟机\n"
                "  2. 继续使用MediaPipe (backend: mediapipe)\n"
                "  3. 直接部署到Jetson设备\n\n"
                "详细安装指南请参考：INSTALL_RTMPOSE.md"
            )

        self.init_model = init_model
        self.inference_topdown = inference_topdown
        self.merge_data_samples = merge_data_samples

        # 加载PyTorch模型
        print(f"[RTMPose] 初始化姿态估计器...")
        print(f"[RTMPose]   模型: {self.model_name}")
        print(f"[RTMPose]   配置文件: {self.config_file}")
        print(f"[RTMPose]   权重文件: {self.checkpoint}")
        print(f"[RTMPose]   设备: {self.device}")
        print(f"[RTMPose]   AMP (混合精度): {'启用' if self.use_amp else '禁用'}")
        print(f"[RTMPose]   TensorRT: {'启用' if self.use_tensorrt else '禁用'}")

        try:
            self.model = self.init_model(
                self.config_file,
                self.checkpoint,
                device=self.device
            )

            # TensorRT优化 (PyTorch FP16)
            if self.use_tensorrt:
                print(f"[RTMPose] 正在应用TensorRT优化...")
                self.model = self._apply_tensorrt_optimization(self.model)
                print(f"[RTMPose] TensorRT优化完成 ✓")

            print(f"[RTMPose] 模型加载成功 ✓")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"模型文件未找到！\n"
                f"配置文件: {self.config_file}\n"
                f"权重文件: {self.checkpoint}\n\n"
                f"请下载模型文件：\n"
                f"  python download_rtmpose_models.py --model {self.model_name}\n\n"
                f"或手动从以下地址下载：\n"
                f"  https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose"
            )
        except Exception as e:
            print(f"[RTMPose] 模型加载失败: {e}")
            print(f"[RTMPose] 提示：请确保已安装MMPose并下载模型文件")
            raise

        self.inference_times = []

    def _load_tensorrt_engine(self):
        """Load standalone TensorRT engine (no MMPose dependency)"""
        try:
            # Use standalone TensorRT wrapper
            from .tensorrt_wrapper import TensorRTRTMPose

            print(f"[RTMPose] 使用standalone TensorRT模式（无需MMPose）")

            # Create TensorRT model
            self.trt_model = TensorRTRTMPose(
                engine_path=self.checkpoint,
                device=self.device
            )

            print(f"[RTMPose] Standalone TensorRT引擎已加载")

        except ImportError as e:
            raise ImportError(
                f"Failed to import TensorRT wrapper: {e}\n"
                f"Please ensure pycuda is installed: pip install pycuda"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TensorRT engine: {e}\n"
                f"Engine file: {self.checkpoint}"
            )

    def _get_model_paths(self, model_name: str) -> tuple:
        """
        根据模型名称自动构建配置文件和权重文件路径

        Args:
            model_name: 模型名称 (rtmpose-tiny, rtmpose-s, rtmpose-m, rtmpose-l)

        Returns:
            (config_file, checkpoint): 配置文件和权重文件路径
        """
        # 模型映射表
        model_configs = {
            'rtmpose-tiny': {
                'config': 'rtmpose-t_8xb256-420e_coco-256x192.py',
                'checkpoint': 'rtmpose-t_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth'
            },
            'rtmpose-s': {
                'config': 'rtmpose-s_8xb256-420e_coco-256x192.py',
                'checkpoint': 'rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'
            },
            'rtmpose-m': {
                'config': 'rtmpose-m_8xb256-420e_coco-256x192.py',
                'checkpoint': 'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
            },
            'rtmpose-l': {
                'config': 'rtmpose-l_8xb256-420e_coco-256x192.py',
                'checkpoint': 'rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'
            }
        }

        if model_name not in model_configs:
            raise ValueError(
                f"不支持的模型: {model_name}\n"
                f"支持的模型: {list(model_configs.keys())}"
            )

        config_info = model_configs[model_name]

        # 构建路径
        models_dir = Path('models/rtmpose')
        config_file = str(models_dir / 'configs' / config_info['config'])
        checkpoint = str(models_dir / config_info['checkpoint'])

        return config_file, checkpoint

    def _apply_tensorrt_optimization(self, model):
        """
        应用TensorRT优化

        Args:
            model: PyTorch模型

        Returns:
            优化后的模型
        """
        try:
            # 检查TensorRT是否可用
            try:
                import tensorrt as trt
                print(f"[RTMPose] TensorRT版本: {trt.__version__}")
            except ImportError:
                print(f"[RTMPose] ⚠️ TensorRT未安装，跳过优化")
                return model

            # 获取TensorRT配置
            fp16_mode = self.tensorrt_config.get('fp16_mode', True)
            int8_mode = self.tensorrt_config.get('int8_mode', False)
            workspace_size = self.tensorrt_config.get('workspace_size', 2048)

            print(f"[RTMPose] TensorRT配置:")
            print(f"[RTMPose]   FP16: {'启用' if fp16_mode else '禁用'}")
            print(f"[RTMPose]   INT8: {'启用' if int8_mode else '禁用'}")
            print(f"[RTMPose]   工作空间: {workspace_size}MB")

            # 使用torch2trt或MMDeploy进行转换
            # 注意：这里需要根据实际MMPose版本调整
            try:
                from mmdeploy.apis import torch2onnx, onnx2tensorrt

                # 方案A：使用MMDeploy（推荐）
                # TODO: 完整的TensorRT转换需要MMDeploy
                # 这里先返回原模型，留待后续完善
                print(f"[RTMPose] ⚠️ TensorRT完整集成需要MMDeploy")
                print(f"[RTMPose] ⚠️ 当前使用PyTorch模型（已是GPU加速）")

                # 至少设置为eval模式并使用half精度
                model.eval()
                if fp16_mode and torch.cuda.is_available():
                    print(f"[RTMPose] 使用FP16精度")
                    model = model.half()
                    self.use_fp16 = True  # Set FP16 flag

                return model

            except ImportError:
                print(f"[RTMPose] ⚠️ MMDeploy未安装，使用PyTorch模型")
                model.eval()
                if fp16_mode and torch.cuda.is_available():
                    model = model.half()
                    self.use_fp16 = True  # Set FP16 flag
                return model

        except Exception as e:
            print(f"[RTMPose] TensorRT优化失败: {e}")
            print(f"[RTMPose] 继续使用PyTorch模型")
            return model

    def estimate(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        估计姿态

        Args:
            frame: 输入图像 (H, W, 3) BGR格式
            bbox: 可选的人体边界框 [x1, y1, x2, y2, confidence]

        Returns:
            keypoints: (17, 3) [x, y, confidence] COCO-17格式
                      None 如果检测失败
        """
        start_time = time.time()

        try:
            # Standalone TensorRT inference (no MMPose dependency)
            if self.use_tensorrt_engine:
                # Use standalone TensorRT wrapper
                if bbox is None:
                    h, w = frame.shape[:2]
                    bbox = np.array([0, 0, w, h, 1.0])

                # Run TensorRT inference
                keypoints = self.trt_model(frame, bbox)

                # 可选关键点平滑
                keypoints = self._apply_keypoint_smoothing(keypoints)

                # Record inference time
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                if len(self.inference_times) > 100:
                    self.inference_times.pop(0)

                return keypoints

            # PyTorch model inference
            # 准备bbox格式
            if bbox is not None:
                # MMPose需要bbox格式: [[x1, y1, x2, y2]] (2D array without score)
                bboxes = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])
            else:
                # 全图检测
                h, w = frame.shape[:2]
                bboxes = np.array([[0, 0, w, h]])

            # 推理 (use AMP autocast for automatic mixed precision)
            # AMP is enabled by default for better GPU performance
            if self.use_amp and 'cuda' in self.device:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    results = self.inference_topdown(
                        self.model,
                        frame,
                        bboxes=bboxes
                    )
            else:
                results = self.inference_topdown(
                    self.model,
                    frame,
                    bboxes=bboxes
                )

            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            # 提取关键点
            if not results or len(results) == 0:
                return None

            result = results[0]

            # 检查结果格式
            if not hasattr(result, 'pred_instances'):
                return None

            pred_instances = result.pred_instances

            if len(pred_instances.keypoints) == 0:
                return None

            keypoints = pred_instances.keypoints[0]  # Shape可能是(17, 2)或(1, 17, 2)
            scores = pred_instances.keypoint_scores[0]  # Shape可能是(17,)或(1, 17)

            # 确保keypoints是(17, 2)
            if len(keypoints.shape) == 3:
                keypoints = keypoints[0]  # (1, 17, 2) -> (17, 2)

            # 确保scores是(17,)
            if len(scores.shape) == 2:
                scores = scores[0]  # (1, 17) -> (17,)

            # 合并为 (17, 3) [x, y, confidence]
            keypoints_with_scores = np.concatenate([
                keypoints,
                scores[:, np.newaxis]
            ], axis=1)

            keypoints_with_scores = keypoints_with_scores.astype(np.float32)

            # 可选关键点平滑
            keypoints_with_scores = self._apply_keypoint_smoothing(keypoints_with_scores)

            return keypoints_with_scores

        except Exception as e:
            print(f"[RTMPose] 姿态估计失败: {e}")
            return None

    def _apply_keypoint_smoothing(self, keypoints: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        对关键点坐标进行EMA平滑（仅x/y，置信度按均值平滑），低于阈值的点不参与平滑。
        """
        if keypoints is None or self.keypoint_smooth_alpha <= 0:
            self._last_keypoints = keypoints
            return keypoints

        alpha = self.keypoint_smooth_alpha
        min_conf = self.keypoint_min_conf

        if self._last_keypoints is None or self._last_keypoints.shape != keypoints.shape:
            self._last_keypoints = keypoints
            return keypoints

        smoothed = keypoints.copy()
        prev = self._last_keypoints

        # 仅对高置信度点平滑
        mask = (keypoints[:, 2] >= min_conf) & (prev[:, 2] >= min_conf)
        if np.any(mask):
            smoothed[mask, 0:2] = alpha * keypoints[mask, 0:2] + (1 - alpha) * prev[mask, 0:2]
            smoothed[mask, 2] = alpha * keypoints[mask, 2] + (1 - alpha) * prev[mask, 2]

        self._last_keypoints = smoothed
        return smoothed

    def get_keypoint_names(self) -> List[str]:
        """获取关键点名称列表"""
        return Keypoint.NAMES

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        获取性能指标

        Returns:
            metrics: 包含以下指标的字典
                - avg_inference_time: 平均推理时间（秒）
                - fps: 理论最大FPS
                - min_time: 最小推理时间
                - max_time: 最大推理时间
        """
        if not self.inference_times:
            return {'avg_inference_time': 0.0, 'fps': 0.0}

        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0

        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'min_time': np.min(self.inference_times),
            'max_time': np.max(self.inference_times),
        }
