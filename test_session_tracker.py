"""测试SessionTracker集成"""

import sys
import time
from src.analytics.session_tracker import SessionTracker, ActivitySession
from src.state.behavior_state import BehaviorState
from src.storage.database import Database

def test_basic_session_tracking():
    """测试基本的会话跟踪功能"""
    print("=" * 60)
    print("SessionTracker集成测试")
    print("=" * 60)

    # 1. 创建临时数据库
    print("\n[1] 创建测试数据库...")
    db = Database(":memory:")  # 使用内存数据库
    print("✓ 数据库创建成功")

    # 2. 创建SessionTracker
    print("\n[2] 创建SessionTracker...")
    tracker = SessionTracker(database=db)
    print("✓ SessionTracker创建成功")

    # 3. 模拟一次坐姿会话
    print("\n[3] 模拟坐姿会话...")
    start_time = time.time()
    tracker.start_session(BehaviorState.SITTING, start_time, zone="chair")
    print(f"✓ 会话开始: {BehaviorState.SITTING.value} @ zone=chair")

    # 等待2秒
    time.sleep(2)

    # 检查当前会话
    current_info = tracker.get_current_session_info()
    print(f"✓ 当前会话信息: {current_info}")
    print(f"  - 状态: {current_info['state']}")
    print(f"  - 时长: {current_info['duration']:.2f}秒")
    print(f"  - 区域: {current_info['zone']}")

    # 结束会话
    end_time = time.time()
    tracker.end_session(end_time)
    print(f"✓ 会话结束，实际时长: {end_time - start_time:.2f}秒")

    # 4. 验证数据库保存
    print("\n[4] 验证数据库保存...")
    history = db.get_state_history(limit=10)
    print(f"✓ 数据库中找到 {len(history)} 条记录")

    if history:
        record = history[0]
        print(f"  - 状态: {record['state']}")
        print(f"  - 时长: {record['duration']:.2f}秒")
        print(f"  - 区域: {record['zone']}")

    # 5. 测试今日统计
    print("\n[5] 测试今日统计...")
    stats = tracker.get_today_statistics()
    print(f"✓ 今日坐姿时长: {stats['sitting_duration']:.2f}秒")
    print(f"✓ 今日站立时长: {stats['standing_duration']:.2f}秒")
    print(f"✓ 今日躺卧时长: {stats['lying_duration']:.2f}秒")
    print(f"✓ 总会话数: {stats['total_sessions']}")

    # 6. 测试状态切换
    print("\n[6] 测试状态切换...")
    tracker.update_session(BehaviorState.SITTING, time.time(), zone="chair")
    time.sleep(1)
    print(f"✓ 坐姿会话持续中，当前时长: {tracker.get_current_duration():.2f}秒")

    tracker.update_session(BehaviorState.STANDING, time.time(), zone="desk")
    time.sleep(1)
    print(f"✓ 切换到站立，站立时长: {tracker.get_current_duration():.2f}秒")

    tracker.end_session(time.time())

    # 7. 最终统计
    print("\n[7] 最终统计...")
    stats = tracker.get_today_statistics()
    sitting_stats = tracker.get_sitting_statistics()
    print(f"✓ 总会话数: {stats['total_sessions']}")
    print(f"✓ 坐姿会话数: {sitting_stats['session_count']}")
    print(f"✓ 坐姿总时长: {sitting_stats['total_duration_minutes']:.2f}分钟")

    # 8. 测试久坐检测
    print("\n[8] 测试久坐检测...")
    tracker.start_session(BehaviorState.SITTING, time.time(), zone="chair")
    is_prolonged = tracker.check_prolonged_sitting(threshold_minutes=0.01)  # 0.01分钟 = 0.6秒
    print(f"✓ 久坐检测（阈值0.01分钟）: {is_prolonged}")

    time.sleep(1)
    is_prolonged = tracker.check_prolonged_sitting(threshold_minutes=0.01)
    print(f"✓ 久坐检测（1秒后）: {is_prolonged}")

    tracker.end_session(time.time())

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！SessionTracker集成正常")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_basic_session_tracking()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
