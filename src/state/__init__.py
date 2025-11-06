"""
状态管理模块
包含ROI管理和行为状态机
"""

from .roi_manager import ROIManager, Zone
from .behavior_state import BehaviorStateMachine, BehaviorState, BehaviorEvent

__all__ = [
    'ROIManager',
    'Zone',
    'BehaviorStateMachine',
    'BehaviorState',
    'BehaviorEvent',
]
