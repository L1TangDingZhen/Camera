"""
ROI（感兴趣区域）管理器
用于定义和检测人体是否在特定区域内
"""

from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np
import cv2


class ZoneType(Enum):
    """区域类型"""
    BED = "bed"
    DOOR = "door"
    CHAIR = "chair"
    BATHROOM = "bathroom"
    CUSTOM = "custom"


class Zone:
    """区域定义"""

    def __init__(self, name: str, zone_type: ZoneType, points: List[Tuple[int, int]]):
        """
        Args:
            name: 区域名称
            zone_type: 区域类型
            points: 多边形顶点坐标 [(x1, y1), (x2, y2), ...]
        """
        self.name = name
        self.zone_type = zone_type
        self.points = np.array(points, dtype=np.int32)

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        检测点是否在区域内

        Args:
            point: 点坐标 (x, y)

        Returns:
            是否在区域内
        """
        if len(self.points) < 3:
            return False

        result = cv2.pointPolygonTest(
            self.points,
            (float(point[0]), float(point[1])),
            False
        )
        return result >= 0

    def contains_bbox(self, bbox: np.ndarray, threshold: float = 0.5) -> bool:
        """
        检测边界框是否在区域内（基于中心点或面积重叠）

        Args:
            bbox: 边界框 [x1, y1, x2, y2, ...]
            threshold: 重叠阈值（0-1）

        Returns:
            是否在区域内
        """
        # 计算边界框中心点
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        return self.contains_point((center_x, center_y))

    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0),
             thickness: int = 2, alpha: float = 0.3) -> np.ndarray:
        """
        在画面上绘制区域

        Args:
            frame: 输入图像
            color: 颜色 (B, G, R)
            thickness: 线条粗细
            alpha: 填充透明度

        Returns:
            绘制后的图像
        """
        if len(self.points) < 3:
            return frame

        # 绘制填充多边形
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.points], color)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # 绘制边界
        cv2.polylines(frame, [self.points], True, color, thickness)

        # 绘制区域名称
        center = self.points.mean(axis=0).astype(int)
        cv2.putText(
            frame,
            self.name,
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        return frame


class ROIManager:
    """ROI管理器"""

    def __init__(self, config: dict):
        """
        Args:
            config: ROI配置
        """
        self.config = config
        self.zones: Dict[str, Zone] = {}

        # 加载区域配置
        self._load_zones()

    def _load_zones(self):
        """从配置加载区域"""
        zones_config = self.config.get('zones', {})

        for zone_name, zone_config in zones_config.items():
            if not zone_config.get('enabled', False):
                continue

            points = zone_config.get('points', [])
            if len(points) < 3:
                print(f"[ROIManager] 警告: 区域 {zone_name} 的点数少于3个，跳过")
                continue

            # 解析区域类型
            try:
                zone_type = ZoneType(zone_name)
            except ValueError:
                zone_type = ZoneType.CUSTOM

            zone = Zone(zone_name, zone_type, points)
            self.zones[zone_name] = zone

        print(f"[ROIManager] 加载了 {len(self.zones)} 个区域: {list(self.zones.keys())}")

    def add_zone(self, zone: Zone):
        """添加区域"""
        self.zones[zone.name] = zone

    def remove_zone(self, name: str):
        """删除区域"""
        if name in self.zones:
            del self.zones[name]

    def get_zone(self, name: str) -> Optional[Zone]:
        """获取区域"""
        return self.zones.get(name)

    def get_containing_zones(self, bbox: Optional[np.ndarray] = None,
                            point: Optional[Tuple[float, float]] = None) -> List[str]:
        """
        获取包含给定位置的所有区域

        Args:
            bbox: 边界框 [x1, y1, x2, y2, ...]
            point: 点坐标 (x, y)

        Returns:
            区域名称列表
        """
        containing_zones = []

        for name, zone in self.zones.items():
            if bbox is not None:
                if zone.contains_bbox(bbox):
                    containing_zones.append(name)
            elif point is not None:
                if zone.contains_point(point):
                    containing_zones.append(name)

        return containing_zones

    def is_in_zone(self, zone_name: str, bbox: Optional[np.ndarray] = None,
                   point: Optional[Tuple[float, float]] = None) -> bool:
        """
        检测是否在指定区域内

        Args:
            zone_name: 区域名称
            bbox: 边界框
            point: 点坐标

        Returns:
            是否在区域内
        """
        zone = self.get_zone(zone_name)
        if zone is None:
            return False

        if bbox is not None:
            return zone.contains_bbox(bbox)
        elif point is not None:
            return zone.contains_point(point)
        else:
            return False

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        在画面上绘制所有区域

        Args:
            frame: 输入图像

        Returns:
            绘制后的图像
        """
        for zone in self.zones.values():
            frame = zone.draw(frame)

        return frame

    def save_config(self, filepath: str):
        """保存ROI配置到文件"""
        import yaml

        zones_config = {}
        for name, zone in self.zones.items():
            zones_config[name] = {
                'enabled': True,
                'points': zone.points.tolist()
            }

        config = {'zones': zones_config}

        with open(filepath, 'w') as f:
            yaml.safe_dump(config, f)

        print(f"[ROIManager] ROI配置已保存到: {filepath}")

    @staticmethod
    def load_from_file(filepath: str) -> 'ROIManager':
        """从文件加载ROI配置"""
        import yaml

        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        return ROIManager(config)
