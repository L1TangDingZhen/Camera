"""
自适应平滑器（Adaptive Smoother）

完整的四层自适应平滑系统，解决姿态估计中的多种问题：
1. 置信度过滤：处理遮挡/低质量检测
2. 速度自适应死区：静止时稳定，移动时灵敏
3. 速度自适应EMA：根据速度调整平滑强度
4. 速度限制：防止异常跳变（检测丢失恢复）

核心特点：
- 静止时：强力稳定（大死区 + 强平滑）
- 慢速移动时：跟随灵敏（小死区 + 弱平滑）
- 快速移动时：完全响应（几乎禁用）

适用场景：
- VTuber 实时驱动
- 动作捕捉
- 姿态分析

性能：
- 轻量级（只有简单的数学运算）
- 自适应（自动根据速度调整参数）

使用示例：
    smoother = AdaptiveSmoother(
        conf_threshold=0.5,
        static_deadzone=4.0,
        moving_deadzone=1.5,
        speed_threshold=2.0,
        static_alpha=0.1,
        moving_alpha=0.4,
        max_velocity=50.0
    )

    smooth_keypoints = smoother.process(keypoints, dt=0.033)
"""

import numpy as np
from typing import Optional


class AdaptiveSmoother:
    """自适应平滑器 - 四层平滑系统"""

    def __init__(
        self,
        # 第①层：置信度过滤
        conf_threshold: float = 0.5,
        conf_enabled: bool = True,

        # 第②层：速度自适应死区
        static_deadzone: float = 4.0,
        moving_deadzone: float = 1.5,
        speed_threshold: float = 2.0,
        deadzone_enabled: bool = True,

        # 第③层：速度自适应 EMA
        static_alpha: float = 0.1,
        moving_alpha: float = 0.4,
        ema_enabled: bool = True,

        # 第④层：速度限制（防止异常跳变）
        max_velocity: float = 50.0,
        velocity_limit_enabled: bool = True,

        # 调试选项
        debug: bool = False
    ):
        """
        初始化自适应平滑器

        Args:
            # 置信度过滤参数
            conf_threshold: 置信度阈值，低于此值的点使用上一帧
                           推荐: 0.4～0.6
            conf_enabled: 是否启用置信度过滤

            # 死区参数
            static_deadzone: 静止时的死区阈值（像素）
                            推荐: 3.0～5.0
            moving_deadzone: 移动时的死区阈值（像素）
                            推荐: 1.0～2.0
            speed_threshold: 判断静止/移动的速度阈值（像素/帧）
                            推荐: 1.5～3.0
            deadzone_enabled: 是否启用死区

            # EMA 参数
            static_alpha: 静止时的 EMA 系数（0～1）
                         越小越平滑，推荐: 0.1～0.2
            moving_alpha: 移动时的 EMA 系数（0～1）
                         越大越跟手，推荐: 0.3～0.5
            ema_enabled: 是否启用 EMA

            # 速度限制参数
            max_velocity: 最大允许速度（像素/帧）
                         超过此速度认为是异常跳变，使用上一帧
                         推荐: 40～60
            velocity_limit_enabled: 是否启用速度限制

            # 调试
            debug: 是否打印调试信息
        """
        # 第①层：置信度过滤
        self.conf_threshold = conf_threshold
        self.conf_enabled = conf_enabled

        # 第②层：速度自适应死区
        self.static_deadzone = static_deadzone
        self.moving_deadzone = moving_deadzone
        self.speed_threshold = speed_threshold
        self.deadzone_enabled = deadzone_enabled

        # 第③层：速度自适应 EMA
        self.static_alpha = static_alpha
        self.moving_alpha = moving_alpha
        self.ema_enabled = ema_enabled

        # 第④层：速度限制
        self.max_velocity = max_velocity
        self.velocity_limit_enabled = velocity_limit_enabled

        # 调试
        self.debug = debug

        # 历史数据
        self.prev_keypoints: Optional[np.ndarray] = None
        self.smooth_keypoints: Optional[np.ndarray] = None
        self.prev_timestamp: Optional[float] = None

        # 统计信息
        self.frame_count = 0
        self.static_frame_count = 0
        self.moving_frame_count = 0

        print(f"[AdaptiveSmoother] 初始化完成")
        print(f"[AdaptiveSmoother]   第①层 置信度过滤: {'启用' if conf_enabled else '禁用'} (threshold={conf_threshold})")
        print(f"[AdaptiveSmoother]   第②层 自适应死区: {'启用' if deadzone_enabled else '禁用'} (static={static_deadzone}px, moving={moving_deadzone}px)")
        print(f"[AdaptiveSmoother]   第③层 自适应EMA: {'启用' if ema_enabled else '禁用'} (static_alpha={static_alpha}, moving_alpha={moving_alpha})")
        print(f"[AdaptiveSmoother]   第④层 速度限制: {'启用' if velocity_limit_enabled else '禁用'} (max={max_velocity}px/frame)")
        print(f"[AdaptiveSmoother]   速度阈值: {speed_threshold} px/frame")

    def process(
        self,
        keypoints: Optional[np.ndarray],
        dt: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        处理关键点，应用四层自适应平滑

        Args:
            keypoints: 输入关键点 (N, 3) [x, y, confidence]
                      通常 N=17 (COCO格式)
            dt: 时间间隔（秒），如果为None则使用帧间隔

        Returns:
            smoothed_keypoints: 平滑后的关键点 (N, 3)
                               None 如果输入为None
        """
        # 输入检查
        if keypoints is None:
            self._reset()
            return None

        self.frame_count += 1

        # 第一帧，直接保存并返回
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints.copy()
            self.smooth_keypoints = keypoints.copy()
            return keypoints

        # 检查形状是否匹配
        if self.prev_keypoints.shape != keypoints.shape:
            print(f"[AdaptiveSmoother] 警告：关键点数量变化 {self.prev_keypoints.shape} -> {keypoints.shape}")
            self.prev_keypoints = keypoints.copy()
            self.smooth_keypoints = keypoints.copy()
            return keypoints

        # 计算每个关键点的速度（像素/帧）
        speeds = self._calculate_speeds(keypoints, self.prev_keypoints)
        avg_speed = np.mean(speeds)

        # 判断是静止还是移动（用于统计和调试）
        is_static = avg_speed < self.speed_threshold

        if is_static:
            self.static_frame_count += 1
        else:
            self.moving_frame_count += 1

        if self.debug and self.frame_count % 30 == 0:
            print(f"[AdaptiveSmoother] Frame {self.frame_count}: avg_speed={avg_speed:.2f}, mode={'STATIC' if is_static else 'MOVING'}, max_speed={np.max(speeds):.2f}")

        # 初始化结果
        result = keypoints.copy()

        # ===== 第①层：置信度过滤 =====
        if self.conf_enabled:
            result = self._apply_confidence_filter(result, self.prev_keypoints)

        # ===== 第④层：速度限制（防止异常跳变）=====
        if self.velocity_limit_enabled:
            result = self._apply_velocity_limit(result, self.prev_keypoints, speeds)

        # ===== 第②层：速度自适应死区（逐点计算）=====
        if self.deadzone_enabled:
            # 每个点根据自己的速度选择死区阈值
            result = self._apply_adaptive_deadzone_per_point(
                result, self.prev_keypoints, speeds
            )

        # ===== 第③层：速度自适应 EMA（逐点计算）=====
        if self.ema_enabled:
            # 每个点根据自己的速度选择 alpha
            result = self._apply_adaptive_ema_per_point(
                result, self.smooth_keypoints, speeds
            )

        # 保存当前结果
        self.prev_keypoints = keypoints.copy()  # 保存原始关键点（用于下一帧速度计算）
        self.smooth_keypoints = result.copy()   # 保存平滑结果（用于 EMA）

        return result

    def _calculate_speeds(
        self,
        current: np.ndarray,
        prev: np.ndarray
    ) -> np.ndarray:
        """
        计算每个关键点的移动速度

        Args:
            current: 当前帧关键点 (N, 3)
            prev: 上一帧关键点 (N, 3)

        Returns:
            speeds: 每个关键点的速度 (N,) 单位：像素/帧
        """
        # 计算每个点的欧氏距离
        distances = np.linalg.norm(current[:, :2] - prev[:, :2], axis=1)
        return distances

    def _apply_confidence_filter(
        self,
        keypoints: np.ndarray,
        prev_keypoints: np.ndarray
    ) -> np.ndarray:
        """
        第①层：置信度过滤

        低置信度的点不可信，使用上一帧的坐标
        """
        result = keypoints.copy()

        for i in range(len(keypoints)):
            if keypoints[i, 2] < self.conf_threshold:
                # 低置信度，使用上一帧坐标
                result[i, :2] = prev_keypoints[i, :2]
                # 置信度也使用上一帧（可选，也可以保持当前帧的低置信度）
                result[i, 2] = prev_keypoints[i, 2]

        return result

    def _apply_adaptive_deadzone_per_point(
        self,
        keypoints: np.ndarray,
        prev_keypoints: np.ndarray,
        speeds: np.ndarray
    ) -> np.ndarray:
        """
        第②层：逐点自适应死区

        每个关键点根据自己的速度选择死区阈值：
        - 速度慢（< speed_threshold）→ 大死区（static_deadzone）
        - 速度快（>= speed_threshold）→ 小死区（moving_deadzone）

        这样可以避免"部分点跟上，部分点粘住"导致的骨架变形
        """
        result = keypoints.copy()

        for i in range(len(keypoints)):
            # 根据这个点的速度选择死区阈值
            if speeds[i] < self.speed_threshold:
                threshold = self.static_deadzone  # 慢速点用大死区
            else:
                threshold = self.moving_deadzone  # 快速点用小死区

            # 计算移动距离
            distance = np.linalg.norm(keypoints[i, :2] - prev_keypoints[i, :2])

            if distance < threshold:
                # 在死区内，使用上一帧坐标
                result[i, :2] = prev_keypoints[i, :2]

        return result

    def _apply_adaptive_ema_per_point(
        self,
        keypoints: np.ndarray,
        smooth_keypoints: np.ndarray,
        speeds: np.ndarray
    ) -> np.ndarray:
        """
        第③层：逐点自适应 EMA

        每个关键点根据自己的速度选择 alpha：
        - 速度慢（< speed_threshold）→ 小 alpha（强平滑）
        - 速度快（>= speed_threshold）→ 大 alpha（弱平滑）
        """
        result = keypoints.copy()

        for i in range(len(keypoints)):
            # 根据这个点的速度选择 alpha
            if speeds[i] < self.speed_threshold:
                alpha = self.static_alpha  # 慢速点用强平滑
            else:
                alpha = self.moving_alpha  # 快速点用弱平滑

            # EMA 平滑
            result[i, :2] = alpha * keypoints[i, :2] + (1 - alpha) * smooth_keypoints[i, :2]

        return result

    def _apply_velocity_limit(
        self,
        keypoints: np.ndarray,
        prev_keypoints: np.ndarray,
        speeds: np.ndarray
    ) -> np.ndarray:
        """
        第④层：速度限制（防止异常跳变）

        当某个点速度过大时（> max_velocity），认为是检测异常导致的跳变：
        - 可能是检测丢失后突然恢复
        - 可能是遮挡/误检
        直接使用上一帧坐标，避免骨架拉伸变形
        """
        result = keypoints.copy()

        for i in range(len(keypoints)):
            if speeds[i] > self.max_velocity:
                # 速度过大，认为是异常跳变，使用上一帧
                result[i, :2] = prev_keypoints[i, :2]
                # 也可以保持上一帧的置信度
                result[i, 2] = prev_keypoints[i, 2]

        return result

    def _reset(self):
        """重置平滑器状态"""
        self.prev_keypoints = None
        self.smooth_keypoints = None
        self.prev_timestamp = None

    def reset(self):
        """公开的重置方法（用于场景切换）"""
        self._reset()
        self.frame_count = 0
        self.static_frame_count = 0
        self.moving_frame_count = 0

    def set_conf_threshold(self, threshold: float):
        """动态调整置信度阈值"""
        self.conf_threshold = threshold
        print(f"[AdaptiveSmoother] 置信度阈值更新: {threshold}")

    def set_deadzone(self, static: float, moving: float):
        """动态调整死区阈值"""
        self.static_deadzone = static
        self.moving_deadzone = moving
        print(f"[AdaptiveSmoother] 死区阈值更新: static={static}px, moving={moving}px")

    def set_alpha(self, static: float, moving: float):
        """动态调整 EMA 系数"""
        self.static_alpha = static
        self.moving_alpha = moving
        print(f"[AdaptiveSmoother] EMA系数更新: static_alpha={static}, moving_alpha={moving}")

    def set_speed_threshold(self, threshold: float):
        """动态调整速度阈值"""
        self.speed_threshold = threshold
        print(f"[AdaptiveSmoother] 速度阈值更新: {threshold} px/frame")

    def get_stats(self) -> dict:
        """获取统计信息"""
        total = self.frame_count
        static_ratio = self.static_frame_count / total if total > 0 else 0
        moving_ratio = self.moving_frame_count / total if total > 0 else 0

        return {
            'frame_count': total,
            'static_frames': self.static_frame_count,
            'moving_frames': self.moving_frame_count,
            'static_ratio': static_ratio,
            'moving_ratio': moving_ratio,
            'conf_enabled': self.conf_enabled,
            'deadzone_enabled': self.deadzone_enabled,
            'ema_enabled': self.ema_enabled,
        }
