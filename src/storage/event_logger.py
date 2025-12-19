"""
事件记录器
负责将行为事件写入数据库和日志文件
"""

from typing import List
import time
from datetime import datetime

from .database import Database
from ..state.behavior_state import BehaviorEvent


class EventLogger:
    """事件记录器"""

    def __init__(self, config: dict):
        """
        Args:
            config: 配置字典
        """
        self.config = config

        # 数据库
        db_path = config.get('storage', {}).get('database', 'data/database.db')
        self.db = Database(db_path)

        # 日志配置
        self.log_file = config.get('logging', {}).get('file', 'logs/events.log')
        self.performance_metrics = config.get('logging', {}).get('performance_metrics', True)

        # 性能统计
        self.last_performance_log = time.time()
        self.performance_log_interval = 60  # 每60秒记录一次性能

        print(f"[EventLogger] 初始化完成")

    def log_events(self, events: List[BehaviorEvent]):
        """
        记录事件列表

        Args:
            events: 事件列表
        """
        for event in events:
            self._log_event(event)

    def _log_event(self, event: BehaviorEvent):
        """记录单个事件"""
        # 写入数据库
        self.db.insert_event(
            event_type=event.event_type.value,
            timestamp=event.timestamp,
            state=event.state.value,
            zone=event.zone,
            metadata=event.metadata,
            tracking_id=event.tracking_id
        )

        # 写入日志文件
        self._write_log_file(event)

        # 打印到控制台
        print(f"[Event] {event}")

    def _write_log_file(self, event: BehaviorEvent):
        """写入日志文件"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{event}\n")
        except Exception as e:
            print(f"[EventLogger] 写入日志文件失败: {e}")

    def log_performance(self, detector_metrics: dict, pose_metrics: dict):
        """
        记录性能指标

        Args:
            detector_metrics: 检测器性能指标
            pose_metrics: 姿态估计器性能指标
        """
        if not self.performance_metrics:
            return

        current_time = time.time()

        # 限制记录频率
        if current_time - self.last_performance_log < self.performance_log_interval:
            return

        self.last_performance_log = current_time

        # 记录检测器FPS
        if 'fps' in detector_metrics:
            self.db.insert_performance_metric(
                timestamp=current_time,
                metric_name='detector_fps',
                metric_value=detector_metrics['fps'],
                metadata=detector_metrics
            )

        # 记录姿态估计器FPS
        if 'fps' in pose_metrics:
            self.db.insert_performance_metric(
                timestamp=current_time,
                metric_name='pose_fps',
                metric_value=pose_metrics['fps'],
                metadata=pose_metrics
            )

    def log_state_change(self, timestamp: float, state: str, zone: str = None, duration: float = 0):
        """
        记录状态变化

        Args:
            timestamp: 时间戳
            state: 状态
            zone: 区域
            duration: 持续时间
        """
        self.db.insert_state_history(
            timestamp=timestamp,
            state=state,
            zone=zone,
            duration=duration
        )

    def get_recent_events(self, limit: int = 50) -> List[dict]:
        """获取最近的事件"""
        return self.db.get_events(limit=limit)

    def get_today_events(self) -> List[dict]:
        """获取今天的事件"""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = today_start.timestamp()

        return self.db.get_events(start_time=start_time)

    def close(self):
        """关闭记录器"""
        self.db.close()
