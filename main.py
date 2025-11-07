#!/usr/bin/env python3
"""
Life Tracker - 行为监测系统主程序
支持三阶段部署: PC -> X390 -> Jetson
"""

import argparse
import time
import yaml
import cv2
import numpy as np
from pathlib import Path

from src.detectors import PersonDetector, PoseEstimatorFactory
from src.state import BehaviorStateMachine, ROIManager
from src.storage import EventLogger


class LifeTracker:
    """Life Tracker主类"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        print(f"\n{'='*60}")
        print(f"  Life Tracker - {self.config['name']}")
        print(f"  设备: {self.config['device']}")
        print(f"{'='*60}\n")

        # 初始化组件
        self._init_components()

    def _init_components(self):
        """初始化所有组件"""
        # 1. 创建检测器
        print("[初始化] 加载人体检测器...")
        self.person_detector = PersonDetector(self.config['models']['person'])

        print("[初始化] 加载姿态估计器...")
        self.pose_estimator = PoseEstimatorFactory.create(self.config['models']['pose'])

        # 2. 创建ROI管理器
        print("[初始化] 加载ROI管理器...")
        self.roi_manager = ROIManager(self.config.get('roi', {}))

        # 3. 创建状态机
        print("[初始化] 创建状态机...")
        self.state_machine = BehaviorStateMachine(self.config, self.roi_manager)

        # 4. 创建事件记录器
        print("[初始化] 创建事件记录器...")
        self.event_logger = EventLogger(self.config)

        # 5. 初始化摄像头
        print("[初始化] 打开摄像头...")
        camera_config = self.config['camera']
        self.cap = cv2.VideoCapture(camera_config['source'])

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera_config['source']}")

        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['resolution'][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['resolution'][1])
        self.cap.set(cv2.CAP_PROP_FPS, camera_config['fps'])

        # 运行参数
        self.show_visualization = True
        self.running = True

        print("\n[初始化] 所有组件加载完成!\n")

    def run(self):
        """主循环"""
        print("[运行] 开始监测...\n")

        frame_count = 0
        fps_calc_time = time.time()
        fps = 0

        # 性能分析
        enable_profiling = self.config.get('debug', {}).get('show_state_info', False)
        if enable_profiling:
            print("🔍 性能分析模式已启用\n")

        # 检测频率控制
        detection_interval = self.config.get('inference', {}).get('detection_interval', 1)
        detection_counter = 0  # 检测帧计数器
        cached_bbox = None  # 缓存的bbox
        if detection_interval > 1:
            print(f"⚡ 检测优化: 每{detection_interval}帧检测一次，中间帧复用结果\n")

        profiling_data = {
            'read_frame': [],
            'detection': [],
            'pose': [],
            'state_machine': [],
            'visualization': [],
            'waitkey': [],
            'total_frame': []
        }

        try:
            while self.running:
                frame_start = time.time()

                # 读取帧
                t0 = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    print("[错误] 无法读取摄像头画面")
                    break
                t1 = time.time()
                profiling_data['read_frame'].append((t1 - t0) * 1000)

                frame_count += 1
                current_time = time.time()

                # 1. 人体检测（每N帧检测一次，中间帧复用）
                detection_counter += 1
                t0 = time.time()

                if detection_counter % detection_interval == 0:
                    # 执行实际检测
                    bbox = self.person_detector.detect(frame)
                    cached_bbox = bbox  # 缓存结果
                    t1 = time.time()
                    profiling_data['detection'].append((t1 - t0) * 1000)
                else:
                    # 复用上一次的检测结果
                    bbox = cached_bbox
                    t1 = time.time()
                    # 不记录时间，因为没有实际检测

                # 2. 姿态估计
                t0 = time.time()
                keypoints = None
                world_landmarks = None
                if bbox is not None:
                    keypoints = self.pose_estimator.estimate(frame, bbox)
                    # 获取3D world landmarks（如果支持）
                    if hasattr(self.pose_estimator, 'get_world_landmarks'):
                        world_landmarks = self.pose_estimator.get_world_landmarks()
                t1 = time.time()
                profiling_data['pose'].append((t1 - t0) * 1000)

                # 保存关键点用于调试显示
                self._last_keypoints = keypoints

                # 3. 更新状态机（使用3D坐标）
                t0 = time.time()
                events = self.state_machine.update(bbox, keypoints, current_time, world_landmarks)
                t1 = time.time()
                profiling_data['state_machine'].append((t1 - t0) * 1000)

                # 4. 记录事件
                if events:
                    self.event_logger.log_events(events)

                # 5. 记录性能指标（每60秒）
                if frame_count % (self.config['camera']['fps'] * 60) == 0:
                    detector_metrics = self.person_detector.get_performance_metrics()
                    pose_metrics = self.pose_estimator.get_performance_metrics()
                    self.event_logger.log_performance(detector_metrics, pose_metrics)

                # 6. 可视化
                t0 = time.time()
                if self.show_visualization:
                    vis_frame = self._visualize(frame, bbox, keypoints, fps)
                    cv2.imshow('Life Tracker', vis_frame)
                t1 = time.time()
                profiling_data['visualization'].append((t1 - t0) * 1000)

                # 7. 处理按键
                t0 = time.time()
                if self.show_visualization:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[退出] 用户按下'q'键")
                        break
                    elif key == ord('r'):
                        # 切换ROI显示
                        self.show_roi = not getattr(self, 'show_roi', True)
                t1 = time.time()
                profiling_data['waitkey'].append((t1 - t0) * 1000)

                # 记录总帧时间
                frame_end = time.time()
                profiling_data['total_frame'].append((frame_end - frame_start) * 1000)

                # 8. 计算FPS
                if current_time - fps_calc_time >= 1.0:
                    fps = frame_count / (current_time - fps_calc_time)
                    frame_count = 0
                    fps_calc_time = current_time

                # 9. 每30帧输出性能分析（约3秒一次）
                if enable_profiling and len(profiling_data['total_frame']) >= 30:
                    self._print_profiling(profiling_data)
                    # 清空数据
                    for key in profiling_data:
                        profiling_data[key] = []

        except KeyboardInterrupt:
            print("\n[退出] 用户中断 (Ctrl+C)")

        finally:
            self.cleanup()

    def _visualize(self, frame: np.ndarray, bbox: np.ndarray,
                   keypoints: np.ndarray, fps: float) -> np.ndarray:
        """
        可视化

        Args:
            frame: 原始帧
            bbox: 边界框
            keypoints: 关键点
            fps: 帧率

        Returns:
            可视化后的帧
        """
        vis_frame = frame.copy()

        # 绘制ROI区域
        if getattr(self, 'show_roi', True):
            vis_frame = self.roi_manager.draw_zones(vis_frame)

        # 绘制bbox
        if bbox is not None:
            x1, y1, x2, y2, conf = bbox.astype(int)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Person: {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制关键点
        if keypoints is not None:
            self._draw_keypoints(vis_frame, keypoints)

        # 绘制状态信息
        self._draw_status(vis_frame, fps)

        return vis_frame

    def _draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray):
        """绘制关键点和骨架"""
        from src.detectors.base import Keypoint

        # 绘制关键点
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        # 绘制骨架
        connections = Keypoint.get_connections()
        for idx1, idx2 in connections:
            if keypoints[idx1, 2] > 0.3 and keypoints[idx2, 2] > 0.3:
                x1, y1 = keypoints[idx1, :2].astype(int)
                x2, y2 = keypoints[idx2, :2].astype(int)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    def _draw_status(self, frame: np.ndarray, fps: float):
        """绘制状态信息"""
        h, w = frame.shape[:2]

        # 调试模式：显示更详细的信息
        debug_mode = self.config.get('debug', {}).get('show_state_info', False)
        info_height = 150 if not debug_mode else 250

        # 背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 文本信息
        y_offset = 30
        line_height = 25

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height

        # 当前状态
        state = self.state_machine.get_current_state()
        state_color = {
            'sitting': (0, 255, 255),  # 黄色
            'lying': (255, 0, 255),    # 紫色
            'standing': (0, 255, 0),   # 绿色
            'sleeping': (255, 0, 0),   # 蓝色
            'absent': (128, 128, 128), # 灰色
        }.get(state.value, (255, 255, 255))

        cv2.putText(frame, f"State: {state.value}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        y_offset += line_height

        # 当前区域
        zone = self.state_machine.current_zone or "None"
        cv2.putText(frame, f"Zone: {zone}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += line_height

        # 状态持续时间
        duration = self.state_machine.get_state_duration(time.time())
        cv2.putText(frame, f"Duration: {duration:.1f}s", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += line_height

        # 调试信息
        if debug_mode and hasattr(self, '_last_keypoints') and self._last_keypoints is not None:
            from src.detectors.base import Keypoint, PoseUtils
            kp = self._last_keypoints

            # 计算关键指标
            try:
                # 身体角度
                body_angle = PoseUtils.get_body_orientation(kp)
                cv2.putText(frame, f"Body Angle: {body_angle:.1f}deg", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20

                # 膝盖角度
                if kp[Keypoint.LEFT_HIP, 2] > 0.3 and kp[Keypoint.LEFT_KNEE, 2] > 0.3 and kp[Keypoint.LEFT_ANKLE, 2] > 0.3:
                    knee_angle = PoseUtils.calculate_angle(
                        kp[Keypoint.LEFT_HIP, :2],
                        kp[Keypoint.LEFT_KNEE, :2],
                        kp[Keypoint.LEFT_ANKLE, :2]
                    )
                    cv2.putText(frame, f"Knee Angle: {knee_angle:.1f}deg", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset += 20

                # 身体高度
                body_height = PoseUtils.get_body_height(kp)
                cv2.putText(frame, f"Height: {body_height:.0f}px", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20

                # 诊断信息（显示判断依据）
                diagnosis = self.state_machine.get_diagnosis()
                if diagnosis:
                    y_offset += 10  # 空一行
                    cv2.putText(frame, "=== Diagnosis ===", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                    y_offset += 20

                    mode = diagnosis.get('mode', 'N/A')
                    cv2.putText(frame, f"Mode: {mode}", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 18

                    # 根据模式显示不同的诊断信息
                    if mode == '3d':
                        # 3D模式：显示真实3D特征
                        if 'torso_angle' in diagnosis:
                            torso_angle = diagnosis['torso_angle']
                            color = (0, 255, 255)  # 黄色
                            cv2.putText(frame, f"TorsoAngle: {torso_angle:.1f}deg (0=upright, 90=horizontal)",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                        if 'hip_knee_z_diff' in diagnosis:
                            z_diff = diagnosis['hip_knee_z_diff']
                            color = (0, 255, 255)
                            cv2.putText(frame, f"Hip-Knee Z: {z_diff:.1f}cm (>0=sitting, ~0=standing)",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                        if 'hip_knee_dist' in diagnosis:
                            dist = diagnosis['hip_knee_dist']
                            color = (0, 255, 255)
                            cv2.putText(frame, f"Hip-Knee Dist: {dist:.1f}cm (>35=extended)",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                        # 显示判断结果
                        if 'lying_check' in diagnosis:
                            lying = diagnosis['lying_check']
                            color = (0, 255, 0) if lying else (128, 128, 128)
                            cv2.putText(frame, f"Lying: {'YES' if lying else 'NO'}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                        if 'standing_check' in diagnosis:
                            standing = diagnosis['standing_check']
                            color = (0, 255, 0) if standing else (128, 128, 128)
                            cv2.putText(frame, f"Standing: {'YES' if standing else 'NO'}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                    elif mode == 'upper_body':
                        # 上半身模式：显示 body_angle 和 shoulder_hip_ratio
                        if 'body_angle' in diagnosis:
                            angle = diagnosis['body_angle']
                            angle_range = diagnosis.get('body_angle_range', (0, 0))
                            angle_ok = diagnosis.get('body_angle_ok', False)
                            color = (0, 255, 0) if angle_ok else (0, 0, 255)
                            status = "OK" if angle_ok else "FAIL"
                            cv2.putText(frame, f"Angle: {angle:.1f} [{angle_range[0]}-{angle_range[1]}] {status}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                        if 'shoulder_hip_ratio' in diagnosis:
                            ratio = diagnosis['shoulder_hip_ratio']
                            ratio_range = diagnosis.get('ratio_range', (0, 0))
                            ratio_ok = diagnosis.get('ratio_ok', False)
                            color = (0, 255, 0) if ratio_ok else (0, 0, 255)
                            status = "OK" if ratio_ok else "FAIL"
                            cv2.putText(frame, f"Ratio: {ratio:.2f} [{ratio_range[0]}-{ratio_range[1]}] {status}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                    elif mode == 'full_body':
                        # 全身模式：显示 knee_angle 和 hip_height_ratio
                        if 'knee_angle' in diagnosis:
                            knee_angle = diagnosis['knee_angle']
                            knee_threshold = diagnosis.get('knee_angle_threshold', 120)
                            knee_ok = diagnosis.get('knee_angle_ok', False)
                            color = (0, 255, 0) if knee_ok else (0, 0, 255)
                            status = "OK" if knee_ok else "FAIL"
                            cv2.putText(frame, f"Knee: {knee_angle:.1f} [<{knee_threshold}] {status}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

                        if 'hip_height_ratio' in diagnosis:
                            hip_ratio = diagnosis['hip_height_ratio']
                            hip_range = diagnosis.get('hip_ratio_range', (0, 0))
                            hip_ok = diagnosis.get('hip_ratio_ok', False)
                            color = (0, 255, 0) if hip_ok else (0, 0, 255)
                            status = "OK" if hip_ok else "FAIL"
                            cv2.putText(frame, f"HipRatio: {hip_ratio:.2f} [{hip_range[0]}-{hip_range[1]}] {status}",
                                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            y_offset += 18

            except:
                pass

        # 提示信息
        tips = "Press: 'q'=quit"
        if debug_mode:
            tips += " | Debug ON"
        cv2.putText(frame, tips, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _print_profiling(self, profiling_data):
        """打印性能分析报告"""
        print("\n" + "="*70)
        print("  性能分析报告 (30帧平均)")
        print("="*70)

        # 计算每个阶段的平均值
        total_avg = np.mean(profiling_data['total_frame'])

        stages = [
            ('读取帧', 'read_frame'),
            ('人体检测', 'detection'),
            ('姿态估计', 'pose'),
            ('状态机更新', 'state_machine'),
            ('可视化绘制', 'visualization'),
            ('waitKey', 'waitkey'),
        ]

        print(f"{'阶段':<12} {'平均耗时':>10} {'占比':>8} {'最小值':>10} {'最大值':>10}")
        print("-"*70)

        detection_count = len(profiling_data['detection'])
        total_frames = len(profiling_data['total_frame'])

        for name, key in stages:
            if profiling_data[key]:
                avg = np.mean(profiling_data[key])
                min_val = np.min(profiling_data[key])
                max_val = np.max(profiling_data[key])
                percentage = (avg / total_avg * 100) if total_avg > 0 else 0

                # 对检测阶段显示实际检测次数
                if key == 'detection' and detection_count < total_frames:
                    print(f"{name:<12} {avg:>8.2f}ms {percentage:>6.1f}% {min_val:>8.2f}ms {max_val:>8.2f}ms (仅{detection_count}次)")
                else:
                    print(f"{name:<12} {avg:>8.2f}ms {percentage:>6.1f}% {min_val:>8.2f}ms {max_val:>8.2f}ms")

        print("-"*70)
        print(f"{'总耗时':<12} {total_avg:>8.2f}ms {'100.0%':>7}")
        print(f"{'理论FPS':<12} {1000/total_avg:>8.1f}")

        # 显示检测优化信息
        detection_interval = self.config.get('inference', {}).get('detection_interval', 1)
        if detection_interval > 1:
            print(f"{'检测间隔':<12} 每{detection_interval}帧检测1次 (降低{(1-1/detection_interval)*100:.0f}%检测负载)")

        print("="*70 + "\n")

    def cleanup(self):
        """清理资源"""
        print("\n[清理] 释放资源...")

        if hasattr(self, 'cap'):
            self.cap.release()

        cv2.destroyAllWindows()

        if hasattr(self, 'event_logger'):
            self.event_logger.close()

        print("[清理] 完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Life Tracker - 行为监测系统')

    parser.add_argument('--config', type=str,
                       help='配置文件路径（直接指定）')
    parser.add_argument('--mode', type=str, choices=['cpu', 'gpu'],
                       help='运行模式: cpu（笔记本/X390）或 gpu（PC/Jetson），默认cpu')
    parser.add_argument('--no-vis', action='store_true',
                       help='不显示可视化窗口')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式：显示关键点、骨架和判断信息')

    args = parser.parse_args()

    # 选择配置文件
    if args.config:
        # 直接指定配置文件
        config_path = args.config
    elif args.mode:
        # 使用 --mode 参数
        config_path = f'config/config_{args.mode}.yaml'
    else:
        # 默认使用CPU模式
        config_path = 'config/config_cpu.yaml'

    # 检查配置文件
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        print(f"\n可用的配置文件：")
        print(f"  config/config_cpu.yaml  - CPU模式（笔记本/X390）")
        print(f"  config/config_gpu.yaml  - GPU模式（PC/Jetson）")
        return

    # 创建tracker
    tracker = LifeTracker(config_path)

    # 启用调试模式
    if args.debug:
        tracker.config['debug']['show_keypoints'] = True
        tracker.config['debug']['show_skeleton'] = True
        tracker.config['debug']['show_angles'] = True
        print("\n🔍 调试模式已启用")

    if args.no_vis:
        tracker.show_visualization = False

    # 运行
    tracker.run()


if __name__ == '__main__':
    main()
