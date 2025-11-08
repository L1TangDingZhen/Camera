"""
活动会话跟踪器
记录和统计坐/站/躺等活动的时长
"""

import time
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from ..state.behavior_state import BehaviorState


@dataclass
class ActivitySession:
    """活动会话（一次连续的坐/站/躺等）"""
    state: str  # sitting, standing, lying, sleeping等
    start_time: float  # 开始时间戳
    end_time: Optional[float] = None  # 结束时间戳（None表示进行中）
    duration: Optional[float] = None  # 持续时长（秒）
    zone: Optional[str] = None  # 区域（bed, chair等）
    metadata: Optional[Dict] = None  # 额外信息

    def finish(self, end_time: float):
        """结束会话并计算时长"""
        self.end_time = end_time
        self.duration = end_time - self.start_time

    def is_active(self) -> bool:
        """会话是否仍在进行中"""
        return self.end_time is None

    def get_duration(self, current_time: Optional[float] = None) -> float:
        """获取持续时长（秒）"""
        if self.duration is not None:
            return self.duration
        elif current_time is not None:
            return current_time - self.start_time
        else:
            return time.time() - self.start_time


class SessionTracker:
    """活动会话跟踪器"""

    def __init__(self, database=None):
        """
        Args:
            database: Database实例（用于持久化）
        """
        self.db = database
        self.current_session: Optional[ActivitySession] = None
        self.session_history: List[ActivitySession] = []

        # 统计缓存
        self._today_cache: Optional[Dict] = None
        self._cache_date: Optional[str] = None

    def start_session(self, state: BehaviorState, timestamp: float, zone: Optional[str] = None):
        """开始新的活动会话

        Args:
            state: 行为状态
            timestamp: 开始时间戳
            zone: 区域
        """
        # 如果有正在进行的会话，先结束它
        if self.current_session is not None and self.current_session.is_active():
            self.end_session(timestamp)

        # 创建新会话
        self.current_session = ActivitySession(
            state=state.value,
            start_time=timestamp,
            zone=zone
        )

        print(f"[SessionTracker] 开始 {state.value} 会话 @ {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")

    def end_session(self, timestamp: float):
        """结束当前会话

        Args:
            timestamp: 结束时间戳
        """
        if self.current_session is None or not self.current_session.is_active():
            return

        # 结束会话
        self.current_session.finish(timestamp)
        duration_minutes = self.current_session.duration / 60

        print(f"[SessionTracker] 结束 {self.current_session.state} 会话, "
              f"时长: {duration_minutes:.1f} 分钟")

        # 保存到历史
        self.session_history.append(self.current_session)

        # 保存到数据库
        if self.db is not None:
            self.db.insert_state_history(
                timestamp=self.current_session.end_time,
                state=self.current_session.state,
                zone=self.current_session.zone,
                duration=self.current_session.duration
            )

        # 清除今日缓存
        self._today_cache = None

        # 清空当前会话
        self.current_session = None

    def update_session(self, state: BehaviorState, timestamp: float, zone: Optional[str] = None):
        """更新会话（状态变化时调用）

        Args:
            state: 当前状态
            timestamp: 时间戳
            zone: 当前区域
        """
        # 忽略UNKNOWN和ABSENT状态
        if state in [BehaviorState.UNKNOWN, BehaviorState.ABSENT]:
            if self.current_session is not None and self.current_session.is_active():
                self.end_session(timestamp)
            return

        # 检查状态是否变化
        if self.current_session is None or self.current_session.state != state.value:
            # 状态变化，开始新会话
            self.start_session(state, timestamp, zone)
        else:
            # 状态未变，更新zone（如果有变化）
            if zone is not None and self.current_session.zone != zone:
                self.current_session.zone = zone

    def get_current_duration(self) -> float:
        """获取当前会话的持续时长（秒）"""
        if self.current_session is None or not self.current_session.is_active():
            return 0.0
        return self.current_session.get_duration()

    def get_current_session_info(self) -> Optional[Dict]:
        """获取当前会话信息"""
        if self.current_session is None or not self.current_session.is_active():
            return None

        return {
            'state': self.current_session.state,
            'start_time': self.current_session.start_time,
            'duration': self.get_current_duration(),
            'zone': self.current_session.zone
        }

    # ==================== 统计查询 ====================

    def get_today_statistics(self) -> Dict:
        """获取今日统计"""
        today_str = datetime.now().strftime('%Y-%m-%d')

        # 检查缓存
        if self._cache_date == today_str and self._today_cache is not None:
            # 如果有正在进行的会话，更新当前状态的时长
            if self.current_session is not None and self.current_session.is_active():
                current_duration = self.get_current_duration()
                state_key = f"{self.current_session.state}_duration"
                self._today_cache[state_key] += current_duration
                self._today_cache['current_session'] = self.get_current_session_info()

            return self._today_cache.copy()

        # 计算今日统计
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_timestamp = today_start.timestamp()

        stats = {
            'date': today_str,
            'sitting_duration': 0.0,
            'standing_duration': 0.0,
            'lying_duration': 0.0,
            'sleeping_duration': 0.0,
            'total_sessions': 0,
            'sessions': []
        }

        # 从数据库查询今日会话
        if self.db is not None:
            history = self.db.get_state_history(start_time=start_timestamp, limit=10000)

            for record in history:
                state = record['state']
                duration = record.get('duration', 0)

                state_key = f"{state}_duration"
                if state_key in stats:
                    stats[state_key] += duration

                stats['total_sessions'] += 1
                stats['sessions'].append({
                    'state': state,
                    'timestamp': record['timestamp'],
                    'duration': duration,
                    'zone': record.get('zone')
                })

        # 添加当前进行中的会话
        if self.current_session is not None and self.current_session.is_active():
            current_duration = self.get_current_duration()
            state_key = f"{self.current_session.state}_duration"
            if state_key in stats:
                stats[state_key] += current_duration

            stats['current_session'] = self.get_current_session_info()

        # 缓存结果
        self._cache_date = today_str
        self._today_cache = stats.copy()

        return stats

    def get_sitting_statistics(self) -> Dict:
        """获取坐姿统计（今日）"""
        today_stats = self.get_today_statistics()

        sitting_duration = today_stats.get('sitting_duration', 0)
        sitting_sessions = [s for s in today_stats.get('sessions', []) if s['state'] == 'sitting']

        return {
            'total_duration': sitting_duration,
            'total_duration_minutes': sitting_duration / 60,
            'total_duration_hours': sitting_duration / 3600,
            'session_count': len(sitting_sessions),
            'average_session_duration': sitting_duration / len(sitting_sessions) if sitting_sessions else 0,
            'longest_session': max([s['duration'] for s in sitting_sessions], default=0),
            'current_sitting': (
                self.current_session.state == 'sitting' and self.current_session.is_active()
            ) if self.current_session else False,
            'current_sitting_duration': (
                self.get_current_duration() if (
                    self.current_session and
                    self.current_session.state == 'sitting' and
                    self.current_session.is_active()
                ) else 0
            )
        }

    def get_weekly_statistics(self) -> Dict:
        """获取本周统计"""
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())  # 本周一
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        stats = {
            'week_start': week_start.strftime('%Y-%m-%d'),
            'total_sitting': 0.0,
            'total_standing': 0.0,
            'total_lying': 0.0,
            'daily_breakdown': []
        }

        if self.db is None:
            return stats

        # 查询本周每一天的数据
        for day_offset in range(7):
            day = week_start + timedelta(days=day_offset)
            if day > today:
                break

            day_str = day.strftime('%Y-%m-%d')
            day_start = day.timestamp()
            day_end = (day + timedelta(days=1)).timestamp()

            history = self.db.get_state_history(start_time=day_start, end_time=day_end, limit=10000)

            day_stats = {
                'date': day_str,
                'sitting': 0.0,
                'standing': 0.0,
                'lying': 0.0
            }

            for record in history:
                state = record['state']
                duration = record.get('duration', 0)

                if state in day_stats:
                    day_stats[state] += duration

            stats['total_sitting'] += day_stats['sitting']
            stats['total_standing'] += day_stats['standing']
            stats['total_lying'] += day_stats['lying']
            stats['daily_breakdown'].append(day_stats)

        return stats

    def check_prolonged_sitting(self, threshold_minutes: int = 30) -> bool:
        """检查是否久坐（超过阈值）

        Args:
            threshold_minutes: 阈值（分钟）

        Returns:
            bool: 是否超过阈值
        """
        if self.current_session is None or self.current_session.state != 'sitting':
            return False

        if not self.current_session.is_active():
            return False

        current_duration_minutes = self.get_current_duration() / 60
        return current_duration_minutes >= threshold_minutes

    def format_duration(self, seconds: float) -> str:
        """格式化时长显示

        Args:
            seconds: 秒数

        Returns:
            str: 格式化字符串（如 "1h 23m" 或 "45m" 或 "30s"）
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            if minutes > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{hours}h"
