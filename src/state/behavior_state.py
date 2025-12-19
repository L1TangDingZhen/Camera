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

# 尝试导入SVM分类器（可选）
try:
    from ..classifiers.pose_classifier import PoseClassifierSVM
    SVM_AVAILABLE = True
except ImportError:
    SVM_AVAILABLE = False
    print("[WARN] SVM分类器模块未找到，将使用基于规则的分类方法")

# 导入SessionTracker
try:
    from ..analytics.session_tracker import SessionTracker
    SESSION_TRACKER_AVAILABLE = True
except ImportError as e:
    SESSION_TRACKER_AVAILABLE = False
    print(f"[WARN] SessionTracker模块导入失败: {e}")
    print("[WARN] 时长统计功能将不可用")


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
    tracking_id: Optional[int] = None  # Multi-person tracking ID, None for single-person mode

    def __str__(self):
        dt = datetime.fromtimestamp(self.timestamp)
        zone_str = f" [{self.zone}]" if self.zone else ""
        person_str = f" [Person {self.tracking_id}]" if self.tracking_id is not None else ""
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}]{person_str} {self.event_type.value}{zone_str} - {self.state.value}"


class BehaviorStateMachine:
    """Behavior State Machine"""

    def __init__(self, config: dict, roi_manager: Optional[ROIManager] = None, database=None, person_id: Optional[int] = None):
        """
        Args:
            config: Configuration dictionary
            roi_manager: ROI manager
            database: Database instance (for SessionTracker)
            person_id: Person ID (used in multi-person mode, None for single-person mode)
        """
        self.config = config
        self.roi_manager = roi_manager or ROIManager(config.get('roi', {}))
        self.person_id = person_id  # Save person_id for multi-person mode

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

        # === 分类器选择（SVM / DL / Ensemble）===
        self.classifier = None
        self.last_probabilities: Optional[Dict[str, float]] = None
        self._init_classifier(config)

        # === 决策策略选择（Simple / RL）===
        self.rl_decision_agent = None
        self.use_rl_decision = False
        self._init_decision_strategy(config)

        # SessionTracker（时长统计）
        self.session_tracker = None
        if SESSION_TRACKER_AVAILABLE:
            self.session_tracker = SessionTracker(database=database)
            print(f"[BehaviorStateMachine] SessionTracker已启用")

        print(f"[BehaviorStateMachine] 初始化完成")

    def _init_classifier(self, config: dict):
        """初始化分类器（SVM / DL / Ensemble）"""
        classifier_config = config.get('behavior', {}).get('classifier', {})
        classifier_type = classifier_config.get('type', 'svm')

        if classifier_type == 'svm':
            # 使用SVM（原有）
            if SVM_AVAILABLE:
                model_path = classifier_config.get('model_path', 'models/pose_classifier_svm.pkl')
                self.classifier = PoseClassifierSVM(model_path)
                print(f"[BehaviorStateMachine] 使用SVM分类器")

        elif classifier_type == 'deep_learning':
            # 使用DL分类器
            try:
                from ..classifiers.pose_classifier_dl import PoseClassifierDL
                model_type = classifier_config.get('model_type', 'lstm')
                model_path = classifier_config.get('model_path', f'models/pose_classifier_{model_type}.pth')
                device = classifier_config.get('device', 'cuda')
                self.classifier = PoseClassifierDL(model_path, model_type, device)
                print(f"[BehaviorStateMachine] 使用DL分类器: {model_type}")
            except Exception as e:
                print(f"[ERROR] 加载DL分类器失败: {e}")
                print(f"[INFO] 降级到基于规则的分类")

        elif classifier_type == 'rl_ensemble':
            # 使用RL Ensemble分类器
            try:
                from ..classifiers.pose_classifier_ensemble import RLEnsembleClassifier

                # 加载多个基础分类器
                base_classifiers = []
                for model_cfg in classifier_config.get('ensemble_models', []):
                    if model_cfg['type'] == 'svm' and SVM_AVAILABLE:
                        clf = PoseClassifierSVM(model_cfg['path'])
                        base_classifiers.append(clf)
                    elif model_cfg['type'] == 'deep_learning':
                        from ..classifiers.pose_classifier_dl import PoseClassifierDL
                        clf = PoseClassifierDL(
                            model_path=model_cfg['path'],
                            model_type=model_cfg.get('model_type', 'lstm'),
                            device=classifier_config.get('device', 'cuda')
                        )
                        base_classifiers.append(clf)

                if len(base_classifiers) >= 2:
                    agent_path = classifier_config.get('agent_path')
                    self.classifier = RLEnsembleClassifier(base_classifiers, agent_path=agent_path)
                    print(f"[BehaviorStateMachine] 使用RL Ensemble分类器 ({len(base_classifiers)}个模型)")
                else:
                    print(f"[ERROR] Ensemble需要至少2个分类器，实际只有{len(base_classifiers)}个")
                    print(f"[INFO] 降级到基于规则的分类")
            except Exception as e:
                print(f"[ERROR] 加载Ensemble分类器失败: {e}")
                print(f"[INFO] 降级到基于规则的分类")

        else:
            print(f"[WARN] 未知的分类器类型: {classifier_type}")
            print(f"[INFO] 使用基于规则的分类")

    def _init_decision_strategy(self, config: dict):
        """初始化决策策略（Simple / RL）"""
        decision_config = config.get('behavior', {}).get('decision', {})
        decision_type = decision_config.get('type', 'simple')

        if decision_type == 'rl':
            # 使用RL决策
            try:
                from .rl_state_decision import RLDecisionAgent
                agent_path = decision_config.get('agent_path')
                self.rl_decision_agent = RLDecisionAgent(agent_path=agent_path)
                self.use_rl_decision = True
                print(f"[BehaviorStateMachine] 使用RL决策策略")
            except Exception as e:
                print(f"[ERROR] 加载RL决策失败: {e}")
                print(f"[INFO] 降级到简单防抖决策")
                self.use_rl_decision = False
        else:
            # 使用简单防抖（原有）
            self.use_rl_decision = False
            print(f"[BehaviorStateMachine] 使用简单防抖决策")

    def update(self, bbox: Optional[np.ndarray], keypoints: Optional[np.ndarray],
               timestamp: float, world_landmarks: Optional[np.ndarray] = None) -> List[BehaviorEvent]:
        """
        更新状态机

        Args:
            bbox: 人体边界框 [x1, y1, x2, y2, confidence]
            keypoints: 关键点 (17, 3) [x, y, confidence]
            timestamp: 时间戳
            world_landmarks: 3D世界坐标 (17, 4) [x, y, z, visibility]，单位：米

        Returns:
            触发的事件列表
        """
        events = []

        # 1. 更新zone
        self._update_zone(bbox)

        # 2. 判断状态（使用3D坐标优先）
        new_state = self._classify_behavior(bbox, keypoints, world_landmarks)

        # 3. 检测状态变化
        if new_state != self.current_state:
            state_events = self._handle_state_change(new_state, timestamp)
            events.extend(state_events)

        # 3.5 更新会话跟踪（时长统计）
        if self.session_tracker is not None:
            self.session_tracker.update_session(self.current_state, timestamp, self.current_zone)

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
                          keypoints: Optional[np.ndarray],
                          world_landmarks: Optional[np.ndarray] = None) -> BehaviorState:
        """
        分类行为状态（优先使用3D坐标）

        Args:
            bbox: 人体边界框
            keypoints: 2D关键点
            world_landmarks: 3D世界坐标

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

        # 3. 优先使用3D world landmarks判断
        if world_landmarks is not None and self._check_3d_quality(world_landmarks):
            # 使用3D判断（更准确，不受摄像头角度影响）
            self._update_diagnosis_3d(world_landmarks, keypoints)
            return self._classify_3d(world_landmarks)
        else:
            # 降级到2D判断（兼容性）
            self._update_diagnosis(keypoints)
            return self._classify_2d(keypoints)

    def _check_3d_quality(self, world_landmarks: np.ndarray) -> bool:
        """检查3D关键点质量"""
        from ..detectors.base import Keypoint

        # 至少需要肩膀和臀部
        required_points = [
            Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER,
            Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP,
        ]

        for idx in required_points:
            if world_landmarks[idx, 3] < 0.3:  # visibility阈值
                return False

        return True

    def _classify_3d(self, world_landmarks: np.ndarray) -> BehaviorState:
        """
        使用3D坐标判断姿态（核心逻辑）

        流程：
        1. 分类层：使用选定的分类器（SVM/DL/Ensemble）获取概率
        2. 决策层：
           - Simple：直接取最高概率
           - RL：RL agent决定何时输出
        3. 降级方案：基于规则的分类
        """
        # === 分类层：使用选定的分类器 ===
        if self.classifier is not None and hasattr(self.classifier, 'is_loaded') and self.classifier.is_loaded:
            probs = self.classifier.predict_proba(world_landmarks)
            if probs is not None:
                self.last_probabilities = probs  # 存储概率用于显示

                # === 决策层：Simple 或 RL ===
                if self.use_rl_decision and self.rl_decision_agent is not None:
                    # RL决策：可能输出状态，也可能等待
                    context = self._get_rl_context()
                    predicted_label, action, metadata = self.rl_decision_agent.decide(
                        probs, context, training=False
                    )

                    if predicted_label is None:
                        # RL决定等待/验证/拒绝，保持当前状态
                        return BehaviorState.UNKNOWN

                else:
                    # 简单决策：直接取最高概率
                    predicted_label = max(probs, key=probs.get)

                # 转换为BehaviorState枚举
                state_mapping = {
                    'sitting': BehaviorState.SITTING,
                    'standing': BehaviorState.STANDING,
                    'lying': BehaviorState.LYING
                }

                return state_mapping.get(predicted_label, BehaviorState.UNKNOWN)

        # === 降级方案：基于规则的分类 ===
        self.last_probabilities = None  # 清除概率

        # 判断躺姿（优先级最高）
        if self._is_lying_3d(world_landmarks):
            return BehaviorState.LYING

        # 判断站姿
        if self._is_standing_3d(world_landmarks):
            return BehaviorState.STANDING

        # 排除法：不是躺也不是站 → 就是坐
        # 这包容了所有坐姿变化：正坐、前倾、靠背、侧身等
        return BehaviorState.SITTING

    def _get_rl_context(self) -> Dict:
        """获取RL决策所需的上下文信息"""
        from datetime import datetime
        import time

        context = {
            'motion': 0.0,  # TODO: 从运动缓冲计算
            'hour_of_day': datetime.now().hour,
            'is_working_hours': 9 <= datetime.now().hour <= 18,
            'keypoint_visibility': 1.0,  # TODO: 从landmarks计算
            'time_since_last_change': time.time() - self.state_start_time,
            'current_duration': time.time() - self.state_start_time
        }
        return context

    def _classify_2d(self, keypoints: np.ndarray) -> BehaviorState:
        """使用2D坐标判断姿态（降级方案）"""
        # 判断躺姿
        if self._is_lying(keypoints):
            return BehaviorState.LYING

        # 判断坐姿
        if self._is_sitting(keypoints):
            return BehaviorState.SITTING

        # 默认为站立
        return BehaviorState.STANDING

    def _update_diagnosis_3d(self, world_landmarks: np.ndarray, keypoints: np.ndarray):
        """实时更新3D诊断信息"""
        from ..detectors.base import Keypoint

        # 提取关键关节的3D坐标
        left_shoulder = world_landmarks[Keypoint.LEFT_SHOULDER]
        right_shoulder = world_landmarks[Keypoint.RIGHT_SHOULDER]
        left_hip = world_landmarks[Keypoint.LEFT_HIP]
        right_hip = world_landmarks[Keypoint.RIGHT_HIP]

        shoulder_center = (left_shoulder[:3] + right_shoulder[:3]) / 2
        hip_center = (left_hip[:3] + right_hip[:3]) / 2

        # 躯干向量（从臀部指向肩膀）
        torso_vector = shoulder_center - hip_center
        torso_length = np.linalg.norm(torso_vector)

        # 躯干与重力方向（Y轴负方向）的夹角
        gravity_vector = np.array([0, -1, 0])  # Y轴向下是重力方向
        if torso_length > 0:
            torso_normalized = torso_vector / torso_length
            # 计算夹角（度）
            dot_product = np.dot(torso_normalized, gravity_vector)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            torso_angle = np.degrees(np.arccos(abs(dot_product)))
        else:
            torso_angle = 0

        # 检查是否有腿部关键点
        has_legs = (
            world_landmarks[Keypoint.LEFT_KNEE, 3] > 0.5 and
            world_landmarks[Keypoint.RIGHT_KNEE, 3] > 0.5
        )

        diagnosis = {
            'mode': '3d',
            'torso_angle': torso_angle,  # 躯干倾斜角度（0=垂直，90=水平）
            'torso_length': torso_length * 100,  # 转换为厘米显示
            'has_legs': has_legs,
        }

        # 如果有腿部，计算更多特征
        if has_legs:
            left_knee = world_landmarks[Keypoint.LEFT_KNEE]
            right_knee = world_landmarks[Keypoint.RIGHT_KNEE]
            knee_center = (left_knee[:3] + right_knee[:3]) / 2

            # 臀部和膝盖的Z轴（深度）差异
            hip_knee_z_diff = hip_center[2] - knee_center[2]  # 米

            # 臀部到膝盖的3D距离
            hip_knee_dist = np.linalg.norm(hip_center - knee_center)

            diagnosis.update({
                'hip_knee_z_diff': hip_knee_z_diff * 100,  # 厘米
                'hip_knee_dist': hip_knee_dist * 100,  # 厘米
            })

            # 判断依据
            diagnosis.update({
                'lying_check': torso_angle > 60,  # 躺：躯干接近水平
                'standing_check': hip_knee_z_diff < 0.05 and hip_knee_dist > 0.35,  # 站：同一平面且伸展
            })

        self.last_diagnosis = diagnosis

    def _is_lying_3d(self, world_landmarks: np.ndarray) -> bool:
        """判断是否躺着（3D）"""
        # 躯干角度 > 60度（接近水平）
        torso_angle = self.last_diagnosis.get('torso_angle', 0)
        return torso_angle > 60

    def _is_standing_3d(self, world_landmarks: np.ndarray) -> bool:
        """判断是否站立（3D）"""
        from ..detectors.base import Keypoint

        # 检查是否有腿部关键点
        has_legs = self.last_diagnosis.get('has_legs', False)

        if not has_legs:
            # 没有腿部信息，用躯干角度判断
            # 如果躯干非常垂直（<20度），可能是站立
            torso_angle = self.last_diagnosis.get('torso_angle', 0)
            # 但不能仅凭躯干角度判断，返回False让它进入sitting判断
            # （坐着时躯干也可以很垂直）
            return False

        left_hip = world_landmarks[Keypoint.LEFT_HIP]
        right_hip = world_landmarks[Keypoint.RIGHT_HIP]
        left_knee = world_landmarks[Keypoint.LEFT_KNEE]
        right_knee = world_landmarks[Keypoint.RIGHT_KNEE]

        hip_center = (left_hip[:3] + right_hip[:3]) / 2
        knee_center = (left_knee[:3] + right_knee[:3]) / 2

        # 条件1：臀部和膝盖在Z轴（深度）上接近（同一平面）
        hip_knee_z_diff = hip_center[2] - knee_center[2]  # 不用abs，保留正负
        # 站立时：膝盖在臀部前方（z_diff < 0）或接近（abs < 5cm）
        z_aligned = abs(hip_knee_z_diff) < 0.08  # 放宽到8厘米

        # 条件2：臀部到膝盖的3D距离 > 30厘米（身体伸展）
        hip_knee_dist = np.linalg.norm(hip_center - knee_center)
        body_extended = hip_knee_dist > 0.30  # 降低到30cm（用户数据33.9cm）

        # 条件3：躯干接近垂直（< 45度）
        torso_angle = self.last_diagnosis.get('torso_angle', 0)
        torso_upright = torso_angle < 45

        # 三个条件都满足 → 站立
        return z_aligned and body_extended and torso_upright

    def _update_diagnosis(self, keypoints: np.ndarray):
        """实时更新诊断信息（每帧都调用）"""
        from ..detectors.base import Keypoint, PoseUtils

        # 检查是否有下半身（更严格的判断）
        has_lower_body = (
            keypoints[Keypoint.LEFT_KNEE, 2] > 0.5 and  # 提高置信度阈值
            keypoints[Keypoint.RIGHT_KNEE, 2] > 0.5 and
            keypoints[Keypoint.LEFT_ANKLE, 2] > 0.5 and
            keypoints[Keypoint.RIGHT_ANKLE, 2] > 0.5 and
            # 检查膝盖是否在臀部下方（排除推测的关键点）
            keypoints[Keypoint.LEFT_KNEE, 1] > keypoints[Keypoint.LEFT_HIP, 1] and
            keypoints[Keypoint.RIGHT_KNEE, 1] > keypoints[Keypoint.RIGHT_HIP, 1]
        )

        # 计算基础特征
        body_angle = PoseUtils.get_body_orientation(keypoints)
        body_height = PoseUtils.get_body_height(keypoints)

        # 计算上半身特征（两种模式都需要）
        left_shoulder = keypoints[Keypoint.LEFT_SHOULDER]
        right_shoulder = keypoints[Keypoint.RIGHT_SHOULDER]
        left_hip = keypoints[Keypoint.LEFT_HIP]
        right_hip = keypoints[Keypoint.RIGHT_HIP]

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        shoulder_hip_dist = abs(hip_y - shoulder_y)
        ratio = shoulder_hip_dist / (body_height + 1e-6)

        if has_lower_body:
            # 全身模式诊断 - 计算膝盖角度和臀部高度比
            left_knee = keypoints[Keypoint.LEFT_KNEE]
            right_knee = keypoints[Keypoint.RIGHT_KNEE]
            left_ankle = keypoints[Keypoint.LEFT_ANKLE]
            right_ankle = keypoints[Keypoint.RIGHT_ANKLE]

            ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            hip_height_ratio = (ankle_y - hip_y) / (body_height + 1e-6)

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

            # 全身sitting条件
            hip_ratio_ok = 0.3 < hip_height_ratio < 0.6
            knee_angle_ok = avg_knee_angle < 120

            self.last_diagnosis = {
                'mode': 'full_body',
                'body_angle': body_angle,
                'body_height_px': body_height,
                'knee_angle': avg_knee_angle,
                'knee_angle_threshold': 120,
                'knee_angle_ok': knee_angle_ok,
                'hip_height_ratio': hip_height_ratio,
                'hip_ratio_range': (0.3, 0.6),
                'hip_ratio_ok': hip_ratio_ok,
                'shoulder_hip_ratio': ratio,
            }
        else:
            # 上半身模式诊断
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
