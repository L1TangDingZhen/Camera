#!/usr/bin/env python3
"""
快速测试脚本 - 不需要摄像头
用生成的测试图像验证代码是否正常工作
"""

import sys
import time
import yaml
import numpy as np
import cv2

# 添加src到路径
sys.path.insert(0, '/home/user/Camera')

from src.detectors import PersonDetector, PoseEstimatorFactory
from src.state import BehaviorStateMachine, ROIManager
from src.storage import EventLogger


def create_test_frame(width=640, height=480):
    """创建测试图像（带人体轮廓）"""
    frame = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)

    # 绘制一个简单的人形（用于测试检测器）
    # 头部
    cv2.circle(frame, (320, 150), 30, (100, 150, 200), -1)
    # 身体
    cv2.rectangle(frame, (290, 180), (350, 320), (100, 150, 200), -1)
    # 左臂
    cv2.rectangle(frame, (250, 180), (290, 280), (100, 150, 200), -1)
    # 右臂
    cv2.rectangle(frame, (350, 180), (390, 280), (100, 150, 200), -1)
    # 左腿
    cv2.rectangle(frame, (290, 320), (315, 420), (100, 150, 200), -1)
    # 右腿
    cv2.rectangle(frame, (325, 320), (350, 420), (100, 150, 200), -1)

    return frame


def test_components():
    """测试各个组件"""
    print("\n" + "="*60)
    print("  Life Tracker 组件测试")
    print("="*60 + "\n")

    # 1. 测试配置加载
    print("[1/5] 测试配置加载...")
    try:
        with open('config/config_pc.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("  ✓ 配置文件加载成功")
    except Exception as e:
        print(f"  ✗ 配置加载失败: {e}")
        return False

    # 2. 测试人体检测器
    print("\n[2/5] 测试YOLOv8检测器...")
    try:
        detector_config = config['models']['person'].copy()
        detector_config['device'] = 'cpu'  # 强制使用CPU
        detector = PersonDetector(detector_config)

        # 测试推理
        test_frame = create_test_frame()
        bbox = detector.detect(test_frame)

        if bbox is not None:
            print(f"  ✓ 检测器正常，检测到人体: {bbox[:4]}")
        else:
            print(f"  ⚠ 检测器正常但未检测到人体（测试图像太简单）")

        metrics = detector.get_performance_metrics()
        print(f"  ✓ FPS: {metrics.get('fps', 0):.1f}")

    except Exception as e:
        print(f"  ✗ 检测器初始化失败: {e}")
        print(f"    提示: 首次运行会自动下载YOLOv8模型（~22MB）")
        return False

    # 3. 测试姿态估计器
    print("\n[3/5] 测试MediaPipe姿态估计...")
    try:
        pose_config = config['models']['pose'].copy()
        pose_config['backend'] = 'mediapipe'
        pose_config['device'] = 'cpu'

        pose_estimator = PoseEstimatorFactory.create(pose_config)

        # 测试推理
        if bbox is not None:
            keypoints = pose_estimator.estimate(test_frame, bbox)
            if keypoints is not None:
                print(f"  ✓ 姿态估计正常，关键点数: {len(keypoints)}")
            else:
                print(f"  ⚠ 姿态估计正常但未检测到关键点")
        else:
            print(f"  ⚠ 跳过（没有bbox）")

    except ImportError as e:
        print(f"  ✗ MediaPipe未安装: {e}")
        print(f"    运行: pip install mediapipe")
        return False
    except Exception as e:
        print(f"  ✗ 姿态估计器失败: {e}")
        return False

    # 4. 测试ROI管理器
    print("\n[4/5] 测试ROI管理器...")
    try:
        roi_manager = ROIManager(config.get('roi', {}))
        print(f"  ✓ ROI管理器初始化成功")
        print(f"    已加载区域数: {len(roi_manager.zones)}")
    except Exception as e:
        print(f"  ✗ ROI管理器失败: {e}")
        return False

    # 5. 测试状态机
    print("\n[5/5] 测试状态机...")
    try:
        state_machine = BehaviorStateMachine(config, roi_manager)

        # 模拟更新
        events = state_machine.update(bbox, keypoints if bbox else None, time.time())

        print(f"  ✓ 状态机初始化成功")
        print(f"    当前状态: {state_machine.get_current_state().value}")
        print(f"    触发事件数: {len(events)}")

    except Exception as e:
        print(f"  ✗ 状态机失败: {e}")
        return False

    # 6. 测试数据库
    print("\n[6/6] 测试数据库...")
    try:
        event_logger = EventLogger(config)

        # 测试记录事件
        if events:
            event_logger.log_events(events)

        print(f"  ✓ 数据库初始化成功")

        event_logger.close()

    except Exception as e:
        print(f"  ✗ 数据库失败: {e}")
        return False

    print("\n" + "="*60)
    print("  ✓ 所有组件测试通过！")
    print("="*60 + "\n")

    return True


def performance_test():
    """性能测试"""
    print("\n" + "="*60)
    print("  性能测试（10秒）")
    print("="*60 + "\n")

    with open('config/config_pc.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 使用CPU避免CUDA问题
    config['models']['person']['device'] = 'cpu'
    config['models']['pose']['backend'] = 'mediapipe'
    config['models']['pose']['device'] = 'cpu'

    detector = PersonDetector(config['models']['person'])
    pose_estimator = PoseEstimatorFactory.create(config['models']['pose'])

    frame = create_test_frame()

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 10:
        bbox = detector.detect(frame)
        if bbox is not None:
            keypoints = pose_estimator.estimate(frame, bbox)
        frame_count += 1

    elapsed = time.time() - start_time
    fps = frame_count / elapsed

    print(f"  处理帧数: {frame_count}")
    print(f"  总耗时: {elapsed:.2f}秒")
    print(f"  平均FPS: {fps:.1f}")
    print(f"  平均延迟: {1000/fps:.1f}ms\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Life Tracker 测试脚本')
    parser.add_argument('--perf', action='store_true', help='运行性能测试')
    args = parser.parse_args()

    # 组件测试
    success = test_components()

    # 性能测试
    if success and args.perf:
        performance_test()

    if success:
        print("💡 提示:")
        print("  - 所有组件工作正常")
        print("  - 要使用真实摄像头，需要连接USB摄像头")
        print("  - 运行: python main.py --device pc")
        print("  - 或使用视频文件: 修改config中的camera.source为视频路径\n")
    else:
        print("⚠ 部分组件测试失败，请检查依赖安装\n")
