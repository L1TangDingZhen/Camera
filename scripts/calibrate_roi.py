#!/usr/bin/env python3
"""
ROI标定工具
用于标定床、门、椅子等区域
"""

import argparse
import cv2
import yaml
import numpy as np
from pathlib import Path


class ROICalibrator:
    """ROI标定器"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 摄像头
        camera_config = self.config['camera']
        self.cap = cv2.VideoCapture(camera_config['source'])

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera_config['source']}")

        # Force MJPEG encoding for better bandwidth efficiency
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['resolution'][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['resolution'][1])

        # 标定状态
        self.current_frame = None
        self.zones = {}  # {zone_name: points}
        self.current_zone_name = None
        self.current_points = []

        # 可标定的区域
        self.available_zones = ['bed', 'door', 'chair', 'bathroom', 'custom']
        self.current_zone_idx = 0

        # 鼠标回调
        cv2.namedWindow('ROI Calibration')
        cv2.setMouseCallback('ROI Calibration', self._mouse_callback)

        print("\n" + "="*60)
        print("  ROI标定工具")
        print("="*60)
        print("\n操作说明:")
        print("  - 点击鼠标左键: 添加多边形顶点")
        print("  - 按 'c': 完成当前区域标定")
        print("  - 按 'n': 切换到下一个区域")
        print("  - 按 'u': 撤销最后一个点")
        print("  - 按 'r': 重置当前区域")
        print("  - 按 's': 保存所有区域配置")
        print("  - 按 'q': 退出")
        print("\n")

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            print(f"  添加点: ({x}, {y})")

    def run(self):
        """运行标定"""
        print(f"当前区域: {self.available_zones[self.current_zone_idx]}\n")

        while True:
            # 读取帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            self.current_frame = frame.copy()

            # 绘制
            vis_frame = self._draw(frame)

            # 显示
            cv2.imshow('ROI Calibration', vis_frame)

            # 处理按键
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n退出标定")
                break

            elif key == ord('c'):
                # 完成当前区域
                if len(self.current_points) >= 3:
                    zone_name = self.available_zones[self.current_zone_idx]
                    self.zones[zone_name] = self.current_points.copy()
                    print(f"✓ 完成区域标定: {zone_name} ({len(self.current_points)}个点)")
                    self.current_points = []

                    # 自动切换到下一个区域
                    self.current_zone_idx = (self.current_zone_idx + 1) % len(self.available_zones)
                    print(f"\n当前区域: {self.available_zones[self.current_zone_idx]}")
                else:
                    print("⚠ 至少需要3个点才能完成区域标定")

            elif key == ord('n'):
                # 切换区域
                self.current_zone_idx = (self.current_zone_idx + 1) % len(self.available_zones)
                print(f"\n当前区域: {self.available_zones[self.current_zone_idx]}")

            elif key == ord('u'):
                # 撤销最后一个点
                if self.current_points:
                    removed = self.current_points.pop()
                    print(f"  撤销点: {removed}")

            elif key == ord('r'):
                # 重置当前区域
                self.current_points = []
                print("  重置当前区域")

            elif key == ord('s'):
                # 保存配置
                self.save_config()

        self.cleanup()

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        """绘制标定界面"""
        vis_frame = frame.copy()

        # 1. 绘制已完成的区域
        for zone_name, points in self.zones.items():
            if len(points) >= 3:
                pts = np.array(points, dtype=np.int32)

                # 填充
                overlay = vis_frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)

                # 边界
                cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 2)

                # 名称
                center = pts.mean(axis=0).astype(int)
                cv2.putText(vis_frame, zone_name, tuple(center),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 2. 绘制当前正在标定的区域
        if len(self.current_points) > 0:
            # 绘制点
            for point in self.current_points:
                cv2.circle(vis_frame, point, 5, (0, 0, 255), -1)

            # 绘制线
            if len(self.current_points) >= 2:
                pts = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(vis_frame, [pts], False, (0, 0, 255), 2)

            # 绘制临时闭合线
            if len(self.current_points) >= 3:
                cv2.line(vis_frame, self.current_points[-1], self.current_points[0],
                        (0, 0, 255), 2, cv2.LINE_AA)

        # 3. 绘制状态信息
        self._draw_status(vis_frame)

        return vis_frame

    def _draw_status(self, frame: np.ndarray):
        """绘制状态信息"""
        h, w = frame.shape[:2]

        # 背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 文本
        y_offset = 35
        line_height = 30

        # 当前区域
        zone_name = self.available_zones[self.current_zone_idx]
        cv2.putText(frame, f"Current Zone: {zone_name}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += line_height

        # 点数
        cv2.putText(frame, f"Points: {len(self.current_points)}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += line_height

        # 已完成的区域
        cv2.putText(frame, f"Completed: {len(self.zones)}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 底部提示
        cv2.putText(frame, "c=complete, n=next, u=undo, r=reset, s=save, q=quit",
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def save_config(self):
        """保存配置"""
        if not self.zones:
            print("\n⚠ 没有标定任何区域，无需保存")
            return

        # 更新配置
        if 'roi' not in self.config:
            self.config['roi'] = {}
        if 'zones' not in self.config['roi']:
            self.config['roi']['zones'] = {}

        for zone_name, points in self.zones.items():
            self.config['roi']['zones'][zone_name] = {
                'enabled': True,
                'points': points
            }

        # 保存到文件
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, default_flow_style=False)

        print(f"\n✓ 配置已保存到: {self.config_path}")
        print(f"  已保存 {len(self.zones)} 个区域: {list(self.zones.keys())}")

        # 同时保存一份到roi_config.yaml
        roi_config_path = 'config/roi_config.yaml'
        roi_config = {'zones': {}}

        for zone_name, points in self.zones.items():
            roi_config['zones'][zone_name] = {
                'enabled': True,
                'points': points
            }

        with open(roi_config_path, 'w') as f:
            yaml.safe_dump(roi_config, f, allow_unicode=True, default_flow_style=False)

        print(f"  同时保存到: {roi_config_path}\n")

    def cleanup(self):
        """清理资源"""
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ROI标定工具')

    parser.add_argument('--config', type=str, default='config/config_pc.yaml',
                       help='配置文件路径')
    parser.add_argument('--device', type=str, choices=['pc', 'x390', 'jetson'],
                       help='设备类型（自动选择配置文件）')

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

    # 运行标定
    calibrator = ROICalibrator(config_path)
    calibrator.run()


if __name__ == '__main__':
    main()
