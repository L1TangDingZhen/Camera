"""
行为状态机
负责判断人的行为状态（坐/躺/睡眠等）和触发事件
"""

from enum import Enum
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
import time
import numpy as np

from ..detectors.base import Keypoint, PoseUtils
from .roi_manager import ROIManager


class BehaviorState(Enum):
    """行为状态"""
    UNKNOWN = "unknown"          # 未知
    ABSENT = "absent"            # 不在场
    STANDING = "standing"        # 站立
    SITTING = "sitting"          # 坐
    LYING = "lying"              # 躺
    SLEEPING = "sleeping"        # 睡眠


class EventType(Enum):
    """事件类型"""
    ENTER_ROOM = "enter_room"        # 进入房间
    LEAVE_ROOM = "leave_room"        # 离开房间
    ENTER_ZONE = "enter_zone"        # 进入区域
    LEAVE_ZONE = "leave_zone"        # 离开区域
    START_SITTING = "start_sitting"  # 开始坐
    START_LYING = "start_lying"      # 开始躺
    START_SLEEPING = "start_sleeping"  # 开始睡眠
    WAKE_UP = "wake_up"              # 醒来
    NIGHT_BATHROOM = "night_bathroom"  # 夜间如厕


@dataclass
class BehaviorEvent:
    """行为事件"""
    event_type: EventType
    timestamp: float
    state: BehaviorState
    zone: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __str__(self):
        dt = datetime.fromtimestamp(self.timestamp)
        zone_str = f" [{self.zone}]" if self.zone else ""
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] {self.event_type.value}{zone_str} - {self.state.value}"


class BehaviorStateMachine:
    """行为状态机"""

    def __init__(self, config: dict, roi_manager: Optional[ROIManager] = None):
        """
        Args:
            config: 配置字典
            roi_manager: ROI管理器
        """
        self.config = config
        self.roi_manager = roi_manager or ROIManager(config.get('roi', {}))

        # 阈值配置
        self.sitting_config = config.get('behavior', {}).get('sitting', {})
        self.lying_config = config.get('behavior', {}).get('lying', {})
        self.sleeping_config = config.get('behavior', {}).get('sleeping', {})
        self.events_config = config.get('behavior', {}).get('events', {})

        # 状态
        self.current_state = BehaviorState.ABSENT
        self.previous_state = BehaviorState.ABSENT
        self.state_start_time = time.time()
        self.current_zone: Optional[str] = None
        self.previous_zone: Optional[str] = None

        # 运动缓冲（用于判断睡眠）
        self.motion_buffer: List[float] = []
        self.max_buffer_size = 300  # 300帧约10秒（30fps）

        # 事件缓冲（防止抖动）
        self.event_timers: Dict[str, float] = {}

        # 诊断信息（用于调试）
        self.last_diagnosis: Dict = {}

        print(f"[BehaviorStateMachine] 初始化完成")

    def update(self, bbox: Optional[np.ndarray], keypoints: Optional[np.ndarray],
               timestamp: float) -> List[BehaviorEvent]:
        """
        更新状态机

        Args:
            bbox: 人体边界框 [x1, y1, x2, y2, confidence]
            keypoints: 关键点 (17, 3) [x, y, confidence]
            timestamp: 时间戳

        Returns:
            触发的事件列表
        """
        events = []

        # 1. 更新zone
        self._update_zone(bbox)

        # 2. 判断状态
        new_state = self._classify_behavior(bbox, keypoints)

        # 3. 检测状态变化
        if new_state != self.current_state:
            state_events = self._handle_state_change(new_state, timestamp)
            events.extend(state_events)

        # 4. 检测zone变化
        if self.current_zone != self.previous_zone:
            zone_events = self._handle_zone_change(timestamp)
            events.extend(zone_events)

        # 5. 更新运动缓冲
        if keypoints is not None:
            motion = self._calculate_motion(keypoints)
            self.motion_buffer.append(motion)
            if len(self.motion_buffer) > self.max_buffer_size:
                self.motion_buffer.pop(0)

        # 6. 检测睡眠
        if self.current_state == BehaviorState.LYING:
            sleep_event = self._detect_sleeping(timestamp)
            if sleep_event:
                events.append(sleep_event)

        # 7. 检测特殊事件（如夜起）
        special_events = self._detect_special_events(timestamp)
        events.extend(special_events)

        return events

    def _update_zone(self, bbox: Optional[np.ndarray]):
        """更新当前区域"""
        self.previous_zone = self.current_zone

        if bbox is None:
            self.current_zone = None
        else:
            zones = self.roi_manager.get_containing_zones(bbox=bbox)
            self.current_zone = zones[0] if zones else None

    def _classify_behavior(self, bbox: Optional[np.ndarray],
                          keypoints: Optional[np.ndarray]) -> BehaviorState:
        """
        分类行为状态

        Args:
            bbox: 人体边界框
            keypoints: 关键点

        Returns:
            行为状态
        """
        # 1. 没有检测到人
        if bbox is None or keypoints is None:
            self.last_diagnosis = {'mode': 'absent'}
            return BehaviorState.ABSENT

        # 2. 检查关键点质量
        if not self._check_keypoints_quality(keypoints):
            self.last_diagnosis = {'mode': 'unknown', 'reason': 'low_keypoint_quality'}
            return BehaviorState.UNKNOWN

        # 先计算所有诊断特征（无论最终判断结果如何）
        self._update_diagnosis(keypoints)

        # 3. 判断躺姿
        if self._is_lying(keypoints):
            return BehaviorState.LYING

        # 4. 判断坐姿
        if self._is_sitting(keypoints):
            return BehaviorState.SITTING

        # 5. 默认为站立
        return BehaviorState.STANDING

    def _update_diagnosis(self, keypoints: np.ndarray):
        """实时更新诊断信息（每帧都调用）"""
        from ..detectors.base import Keypoint, PoseUtils

        # 检查是否有下半身
        has_lower_body = (
            keypoints[Keypoint.LEFT_KNEE, 2] > 0.3 and
            keypoints[Keypoint.RIGHT_KNEE, 2] > 0.3 and
            keypoints[Keypoint.LEFT_ANKLE, 2] > 0.3 and
            keypoints[Keypoint.RIGHT_ANKLE, 2] > 0.3
        )

        # 计算基础特征
        body_angle = PoseUtils.get_body_orientation(keypoints)
        body_height = PoseUtils.get_body_height(keypoints)

        if has_lower_body:
            # 全身模式诊断
            self.last_diagnosis = {
                'mode': 'full_body',
                'body_angle': body_angle,
                'body_height_px': body_height,
            }
        else:
            # 上半身模式诊断
            left_shoulder = keypoints[Keypoint.LEFT_SHOULDER]
            right_shoulder = keypoints[Keypoint.RIGHT_SHOULDER]
            left_hip = keypoints[Keypoint.LEFT_HIP]
            right_hip = keypoints[Keypoint.RIGHT_HIP]

            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
            shoulder_hip_dist = abs(hip_y - shoulder_y)
            ratio = shoulder_hip_dist / (body_height + 1e-6)

            self.last_diagnosis = {
                'mode': 'upper_body',
                'body_angle': body_angle,
                'body_angle_range': (60, 110),
                'body_angle_ok': 60 < body_angle < 110,
                'shoulder_hip_ratio': ratio,
                'ratio_range': (0.3, 0.8),
                'ratio_ok': 0.3 < ratio < 0.8,
                'body_height_px': body_height,
            }

    def _check_keypoints_quality(self, keypoints: np.ndarray) -> bool:
        """检查关键点质量"""
        # 至少需要肩膀和臀部（支持只看到上半身的场景）
        required_points = [
            Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER,
            Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP,
        ]

        for idx in required_points:
            if keypoints[idx, 2] < 0.3:  # 置信度阈值
                return False

        return True

    def _is_lying(self, keypoints: np.ndarray) -> bool:
        """判断是否躺着"""
        # 计算身体角度
        body_angle = PoseUtils.get_body_orientation(keypoints)

        # 躺姿：身体接近水平
        angle_threshold = self.lying_config.get('body_angle_max', 30)
        if body_angle < angle_threshold:
            return True

        # 额外检查：肩膀和臀部高度差很小
        left_shoulder = keypoints[Keypoint.LEFT_SHOULDER]
        right_shoulder = keypoints[Keypoint.RIGHT_SHOULDER]
        left_hip = keypoints[Keypoint.LEFT_HIP]
        right_hip = keypoints[Keypoint.RIGHT_HIP]

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2

        body_height = PoseUtils.get_body_height(keypoints)
        height_diff_ratio = abs(shoulder_y - hip_y) / (body_height + 1e-6)

        ratio_threshold = self.lying_config.get('shoulder_hip_ratio', 0.15)
        if height_diff_ratio < ratio_threshold:
            return True

        return False

    def _is_sitting(self, keypoints: np.ndarray) -> bool:
        """判断是否坐着"""
        # 检查是否有下半身关键点（膝盖、脚踝）
        has_lower_body = (
            keypoints[Keypoint.LEFT_KNEE, 2] > 0.3 and
            keypoints[Keypoint.RIGHT_KNEE, 2] > 0.3 and
            keypoints[Keypoint.LEFT_ANKLE, 2] > 0.3 and
            keypoints[Keypoint.RIGHT_ANKLE, 2] > 0.3
        )

        if has_lower_body:
            # 完整身体检测：使用原有逻辑
            return self._is_sitting_full_body(keypoints)
        else:
            # 上半身检测：使用上半身特征判断（桌面摄像头场景）
            return self._is_sitting_upper_body(keypoints)

    def _is_sitting_full_body(self, keypoints: np.ndarray) -> bool:
        """完整身体的sitting判断（能看到腿）"""
        # 1. 臀部高度在一定范围内（相对于身高）
        left_hip = keypoints[Keypoint.LEFT_HIP]
        right_hip = keypoints[Keypoint.RIGHT_HIP]
        left_ankle = keypoints[Keypoint.LEFT_ANKLE]
        right_ankle = keypoints[Keypoint.RIGHT_ANKLE]

        hip_y = (left_hip[1] + right_hip[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2

        body_height = PoseUtils.get_body_height(keypoints)
        hip_height_ratio = (ankle_y - hip_y) / (body_height + 1e-6)

        min_ratio = self.sitting_config.get('hip_height_min', 0.3)
        max_ratio = self.sitting_config.get('hip_height_max', 0.6)

        if not (min_ratio < hip_height_ratio < max_ratio):
            return False

        # 2. 膝盖角度小于阈值
        left_knee_angle = PoseUtils.calculate_angle(
            keypoints[Keypoint.LEFT_HIP][:2],
            keypoints[Keypoint.LEFT_KNEE][:2],
            keypoints[Keypoint.LEFT_ANKLE][:2]
        )

        right_knee_angle = PoseUtils.calculate_angle(
            keypoints[Keypoint.RIGHT_HIP][:2],
            keypoints[Keypoint.RIGHT_KNEE][:2],
            keypoints[Keypoint.RIGHT_ANKLE][:2]
        )

        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        angle_threshold = self.sitting_config.get('knee_angle_max', 120)

        if avg_knee_angle < angle_threshold:
            return True

        return False

    def _is_sitting_upper_body(self, keypoints: np.ndarray) -> bool:
        """上半身的sitting判断（桌面摄像头，看不到腿）"""
        # 使用已经计算好的诊断信息
        if not self.last_diagnosis or self.last_diagnosis.get('mode') != 'upper_body':
            return False

        angle_ok = self.last_diagnosis.get('body_angle_ok', False)
        ratio_ok = self.last_diagnosis.get('ratio_ok', False)

        return angle_ok and ratio_ok

    def _calculate_motion(self, keypoints: np.ndarray) -> float:
        """计算运动量（用于判断睡眠）"""
        # 简单实现：计算所有关键点的平均位置变化
        if not hasattr(self, '_prev_keypoints'):
            self._prev_keypoints = keypoints
            return 0.0

        # 计算位移
        motion = np.mean(np.linalg.norm(
            keypoints[:, :2] - self._prev_keypoints[:, :2],
            axis=1
        ))

        self._prev_keypoints = keypoints
        return motion

    def _detect_sleeping(self, timestamp: float) -> Optional[BehaviorEvent]:
        """检测是否进入睡眠"""
        if self.current_state != BehaviorState.LYING:
            return None

        # 躺的时长
        lying_duration = timestamp - self.state_start_time

        # 最小躺的时长
        min_duration = self.sleeping_config.get('still_duration', 300)  # 5分钟

        if lying_duration < min_duration:
            return None

        # 检查运动量
        if len(self.motion_buffer) < 50:
            return None

        avg_motion = np.mean(self.motion_buffer[-50:])
        motion_threshold = self.sleeping_config.get('motion_threshold', 0.02)

        if avg_motion < motion_threshold:
            # 判定为睡眠
            self.current_state = BehaviorState.SLEEPING
            return BehaviorEvent(
                event_type=EventType.START_SLEEPING,
                timestamp=timestamp,
                state=BehaviorState.SLEEPING,
                zone=self.current_zone
            )

        return None

    def _handle_state_change(self, new_state: BehaviorState,
                            timestamp: float) -> List[BehaviorEvent]:
        """处理状态变化"""
        events = []

        # 防抖：状态必须持续一定时间才触发事件
        duration_threshold = self.events_config.get('enter_duration', 2)

        if timestamp - self.state_start_time < duration_threshold:
            return events

        # 记录状态变化
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = timestamp

        # 生成事件
        if new_state == BehaviorState.SITTING:
            events.append(BehaviorEvent(
                event_type=EventType.START_SITTING,
                timestamp=timestamp,
                state=new_state,
                zone=self.current_zone
            ))
        elif new_state == BehaviorState.LYING:
            events.append(BehaviorEvent(
                event_type=EventType.START_LYING,
                timestamp=timestamp,
                state=new_state,
                zone=self.current_zone
            ))
        elif self.previous_state == BehaviorState.SLEEPING:
            events.append(BehaviorEvent(
                event_type=EventType.WAKE_UP,
                timestamp=timestamp,
                state=new_state,
                zone=self.current_zone
            ))

        return events

    def _handle_zone_change(self, timestamp: float) -> List[BehaviorEvent]:
        """处理区域变化"""
        events = []

        # 离开旧区域
        if self.previous_zone:
            events.append(BehaviorEvent(
                event_type=EventType.LEAVE_ZONE,
                timestamp=timestamp,
                state=self.current_state,
                zone=self.previous_zone
            ))

        # 进入新区域
        if self.current_zone:
            events.append(BehaviorEvent(
                event_type=EventType.ENTER_ZONE,
                timestamp=timestamp,
                state=self.current_state,
                zone=self.current_zone
            ))

        return events

    def _detect_special_events(self, timestamp: float) -> List[BehaviorEvent]:
        """检测特殊事件（如夜起）"""
        events = []

        # 检测夜起：夜间从床上起来去浴室
        if self._is_night_time(timestamp):
            if (self.previous_zone == 'bed' and
                self.current_zone == 'bathroom' and
                self.previous_state == BehaviorState.LYING):

                events.append(BehaviorEvent(
                    event_type=EventType.NIGHT_BATHROOM,
                    timestamp=timestamp,
                    state=self.current_state,
                    zone='bathroom',
                    metadata={'from_zone': 'bed'}
                ))

        return events

    def _is_night_time(self, timestamp: float) -> bool:
        """判断是否夜间"""
        night_start_str = self.events_config.get('night_start', '22:00')
        night_end_str = self.events_config.get('night_end', '06:00')

        current_time = datetime.fromtimestamp(timestamp).time()
        night_start = datetime.strptime(night_start_str, '%H:%M').time()
        night_end = datetime.strptime(night_end_str, '%H:%M').time()

        if night_start < night_end:
            return night_start <= current_time <= night_end
        else:
            # 跨午夜
            return current_time >= night_start or current_time <= night_end

    def get_current_state(self) -> BehaviorState:
        """获取当前状态"""
        return self.current_state

    def get_state_duration(self, timestamp: float) -> float:
        """获取当前状态持续时间（秒）"""
        return timestamp - self.state_start_time

    def get_diagnosis(self) -> Dict:
        """获取最近的诊断信息（用于调试）"""
        return self.last_diagnosis
