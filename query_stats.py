#!/usr/bin/env python3
"""查询SessionTracker统计数据"""

import os
from src.storage.database import Database
from src.analytics.session_tracker import SessionTracker

def main():
    """查询并显示统计数据"""

    # 检查数据库是否存在
    db_path = 'data/database.db'
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        print("请先运行 main.py 以创建数据库")
        return

    # 连接数据库
    print("📊 SessionTracker 统计数据")
    print("=" * 60)

    db = Database(db_path)
    tracker = SessionTracker(database=db)

    # 1. 今日统计
    print("\n【今日统计】")
    stats = tracker.get_today_statistics()
    print(f"日期: {stats['date']}")
    print(f"总会话数: {stats['total_sessions']}")
    print()

    # 转换为小时和分钟
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    print(f"  🪑 坐姿: {format_time(stats['sitting_duration'])} ({stats['sitting_duration']/3600:.2f}h)")
    print(f"  🧍 站立: {format_time(stats['standing_duration'])} ({stats['standing_duration']/3600:.2f}h)")
    print(f"  🛏️  躺卧: {format_time(stats['lying_duration'])} ({stats['lying_duration']/3600:.2f}h)")
    print(f"  😴 睡眠: {format_time(stats['sleeping_duration'])} ({stats['sleeping_duration']/3600:.2f}h)")

    # 2. 当前会话
    if stats.get('current_session'):
        current = stats['current_session']
        print(f"\n【当前会话】")
        print(f"  状态: {current['state']}")
        print(f"  时长: {format_time(current['duration'])}")
        if current.get('zone'):
            print(f"  区域: {current['zone']}")

    # 3. 坐姿详细统计
    print(f"\n【坐姿详细统计】")
    sitting_stats = tracker.get_sitting_statistics()
    print(f"  总时长: {sitting_stats['total_duration_minutes']:.1f} 分钟 ({sitting_stats['total_duration_hours']:.2f}h)")
    print(f"  会话次数: {sitting_stats['session_count']}")

    if sitting_stats['session_count'] > 0:
        print(f"  平均每次: {sitting_stats['average_session_duration']/60:.1f} 分钟")
        print(f"  最长一次: {sitting_stats['longest_session']/60:.1f} 分钟")

    # 久坐检测
    if sitting_stats['current_sitting']:
        current_duration_min = sitting_stats['current_sitting_duration'] / 60
        print(f"  当前正在坐: {current_duration_min:.0f} 分钟")

        if tracker.check_prolonged_sitting(threshold_minutes=30):
            print(f"\n  ⚠️  久坐警告: 已持续坐姿超过30分钟!")

    # 4. 最近的会话记录
    print(f"\n【最近10次会话】")
    if stats['sessions']:
        print(f"  {'时间':<20} {'状态':<10} {'时长':<15} {'区域':<10}")
        print(f"  {'-'*60}")

        for session in stats['sessions'][:10]:
            from datetime import datetime
            time_str = datetime.fromtimestamp(session['end_time']).strftime('%Y-%m-%d %H:%M:%S')
            duration_str = format_time(session['duration'])
            zone_str = session.get('zone', '-')
            print(f"  {time_str:<20} {session['state']:<10} {duration_str:<15} {zone_str:<10}")
    else:
        print("  暂无会话记录")

    # 5. 本周统计
    print(f"\n【本周统计】")
    weekly = tracker.get_weekly_statistics()
    print(f"  周期: {weekly['week_start']} 到 {weekly['week_end']}")
    print(f"  总会话数: {weekly['total_sessions']}")
    print()
    print(f"  每日坐姿时长:")

    for day_stat in weekly['daily_breakdown']:
        sitting_hours = day_stat['sitting_duration'] / 3600
        if sitting_hours > 0:
            print(f"    {day_stat['date']}: {sitting_hours:.1f}h ({day_stat['sessions']} 次会话)")

    print("\n" + "=" * 60)
    print("💡 提示: 运行 'python main.py --config config/config_gpu.yaml' 开始记录")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
