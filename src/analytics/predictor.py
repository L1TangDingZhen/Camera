"""
Predictor - 久坐预测模型

基于历史数据预测用户的坐姿行为，提供智能提醒
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class SittingPredictor:
    """坐姿行为预测器"""

    def __init__(self, database=None):
        """
        Args:
            database: Database实例
        """
        self.db = database

    def predict_next_sitting_duration(self, current_hour: Optional[int] = None) -> Dict:
        """预测下次坐姿的持续时长

        基于历史数据分析：
        1. 整体平均坐姿时长
        2. 当前时段的平均坐姿时长
        3. 最近一周的趋势

        Args:
            current_hour: 当前小时（0-23），如果为None则使用当前时间

        Returns:
            Dict: 预测结果
                - predicted_duration_minutes: 预测时长（分钟）
                - confidence: 置信度 (0-1)
                - based_on: 预测依据
                - recommendation: 建议
        """
        if self.db is None:
            return self._empty_prediction("数据库未连接")

        if current_hour is None:
            current_hour = datetime.now().hour

        # 获取最近30天的坐姿数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)

        history = self.db.get_state_history(
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            limit=10000
        )

        # 筛选坐姿记录
        sitting_records = [r for r in history if r['state'] == 'sitting' and r.get('duration')]

        if len(sitting_records) < 3:
            return self._empty_prediction("历史数据不足（至少需要3次坐姿记录）")

        # 1. 计算整体平均值
        durations = [r['duration'] / 60 for r in sitting_records]  # 转换为分钟
        overall_avg = statistics.mean(durations)
        overall_median = statistics.median(durations)

        # 2. 计算当前时段的平均值（±2小时）
        hour_range_records = []
        for record in sitting_records:
            record_time = datetime.fromtimestamp(record['timestamp'])
            record_hour = record_time.hour

            # 检查是否在当前时段（±2小时）
            hour_diff = abs(record_hour - current_hour)
            if hour_diff <= 2 or hour_diff >= 22:  # 考虑跨天情况
                hour_range_records.append(record['duration'] / 60)

        if len(hour_range_records) >= 2:
            hour_avg = statistics.mean(hour_range_records)
            confidence = min(0.9, 0.5 + len(hour_range_records) * 0.05)  # 基于样本数的置信度
            prediction_base = "当前时段历史数据"
            predicted_duration = hour_avg
        else:
            # 使用整体平均值
            confidence = min(0.7, 0.3 + len(sitting_records) * 0.02)
            prediction_base = "整体历史数据"
            predicted_duration = overall_avg

        # 3. 调整预测值（避免极端值）
        # 使用中位数来平滑异常值
        if abs(predicted_duration - overall_median) > overall_median * 0.5:
            predicted_duration = (predicted_duration + overall_median) / 2
            confidence *= 0.8

        # 4. 生成建议
        recommendation = self._generate_recommendation(predicted_duration, overall_avg)

        return {
            'predicted_duration_minutes': round(predicted_duration, 1),
            'confidence': round(confidence, 2),
            'based_on': prediction_base,
            'recommendation': recommendation,
            'statistics': {
                'overall_average': round(overall_avg, 1),
                'overall_median': round(overall_median, 1),
                'hour_average': round(hour_avg, 1) if hour_range_records else None,
                'sample_count': len(sitting_records),
                'hour_sample_count': len(hour_range_records)
            }
        }

    def predict_optimal_reminder_time(self) -> Dict:
        """预测最佳提醒时间

        分析用户的坐姿模式，找出最容易久坐的时段

        Returns:
            Dict: 预测结果
                - high_risk_hours: 高风险时段列表
                - recommended_reminder_interval: 建议提醒间隔（分钟）
                - pattern_description: 模式描述
        """
        if self.db is None:
            return {'error': '数据库未连接'}

        # 获取最近14天的数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=14)

        history = self.db.get_state_history(
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            limit=10000
        )

        # 按小时统计坐姿时长
        hour_sitting_time = defaultdict(list)

        for record in history:
            if record['state'] == 'sitting' and record.get('duration'):
                record_time = datetime.fromtimestamp(record['timestamp'])
                hour = record_time.hour
                duration_minutes = record['duration'] / 60
                hour_sitting_time[hour].append(duration_minutes)

        if not hour_sitting_time:
            return {
                'high_risk_hours': [],
                'recommended_reminder_interval': 30,
                'pattern_description': '暂无足够数据分析模式'
            }

        # 计算每小时的平均坐姿时长
        hour_avg_sitting = {}
        for hour, durations in hour_sitting_time.items():
            hour_avg_sitting[hour] = statistics.mean(durations)

        # 找出高风险时段（坐姿时长超过中位数的时段）
        median_duration = statistics.median(hour_avg_sitting.values())
        high_risk_hours = [
            hour for hour, avg_duration in hour_avg_sitting.items()
            if avg_duration > median_duration * 1.2
        ]

        high_risk_hours.sort()

        # 计算建议提醒间隔
        all_durations = [d for durations in hour_sitting_time.values() for d in durations]
        avg_sitting_duration = statistics.mean(all_durations)

        if avg_sitting_duration > 45:
            recommended_interval = 25  # 频繁提醒
            pattern = "您的坐姿时长较长，建议频繁提醒"
        elif avg_sitting_duration > 30:
            recommended_interval = 30  # 标准提醒
            pattern = "您的坐姿时长适中，建议标准提醒"
        else:
            recommended_interval = 40  # 宽松提醒
            pattern = "您的坐姿时长较短，可以宽松提醒"

        # 格式化高风险时段描述
        if high_risk_hours:
            risk_periods = self._format_hour_ranges(high_risk_hours)
            pattern += f"。高风险时段: {risk_periods}"

        return {
            'high_risk_hours': high_risk_hours,
            'recommended_reminder_interval': recommended_interval,
            'pattern_description': pattern,
            'hour_statistics': {
                str(hour): round(avg, 1)
                for hour, avg in sorted(hour_avg_sitting.items())
            }
        }

    def detect_anomaly(self) -> Dict:
        """检测今日坐姿行为是否异常

        对比今日坐姿时长与历史平均值，判断是否异常

        Returns:
            Dict: 异常检测结果
                - is_anomaly: 是否异常
                - today_sitting_hours: 今日坐姿时长（小时）
                - average_sitting_hours: 历史平均（小时）
                - deviation_percentage: 偏差百分比
                - severity: 严重程度 (low/medium/high)
        """
        if self.db is None:
            return {'error': '数据库未连接'}

        # 获取今日数据
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_start = today.timestamp()
        today_end = (today + timedelta(days=1)).timestamp()

        today_history = self.db.get_state_history(
            start_time=today_start,
            end_time=today_end,
            limit=1000
        )

        today_sitting = sum(
            r.get('duration', 0) for r in today_history if r['state'] == 'sitting'
        ) / 3600  # 转换为小时

        # 获取过去30天的平均值（排除今天）
        month_ago = today - timedelta(days=30)
        month_start = month_ago.timestamp()

        history = self.db.get_state_history(
            start_time=month_start,
            end_time=today_start,
            limit=10000
        )

        # 按天分组计算每天的坐姿时长
        daily_sitting = defaultdict(float)
        for record in history:
            if record['state'] == 'sitting' and record.get('duration'):
                record_date = datetime.fromtimestamp(record['timestamp']).date()
                daily_sitting[record_date] += record['duration'] / 3600

        if not daily_sitting:
            return {
                'is_anomaly': False,
                'message': '历史数据不足，无法判断'
            }

        # 计算平均值和标准差
        daily_values = list(daily_sitting.values())
        avg_sitting = statistics.mean(daily_values)
        std_sitting = statistics.stdev(daily_values) if len(daily_values) > 1 else 0

        # 计算偏差
        deviation = today_sitting - avg_sitting
        deviation_pct = (deviation / avg_sitting * 100) if avg_sitting > 0 else 0

        # 判断异常
        is_anomaly = abs(deviation) > std_sitting * 1.5 if std_sitting > 0 else abs(deviation_pct) > 50

        # 判断严重程度
        if abs(deviation_pct) < 20:
            severity = 'low'
        elif abs(deviation_pct) < 50:
            severity = 'medium'
        else:
            severity = 'high'

        return {
            'is_anomaly': is_anomaly,
            'today_sitting_hours': round(today_sitting, 2),
            'average_sitting_hours': round(avg_sitting, 2),
            'deviation_percentage': round(deviation_pct, 1),
            'severity': severity,
            'message': self._generate_anomaly_message(deviation_pct, is_anomaly)
        }

    def _empty_prediction(self, reason: str) -> Dict:
        """返回空预测结果"""
        return {
            'predicted_duration_minutes': 0,
            'confidence': 0,
            'based_on': reason,
            'recommendation': '请积累更多数据以获得更准确的预测'
        }

    def _generate_recommendation(self, predicted_minutes: float, avg_minutes: float) -> str:
        """生成建议"""
        if predicted_minutes > 45:
            return "预计您将久坐，建议设置25分钟提醒"
        elif predicted_minutes > 30:
            return "预计坐姿时长适中，建议设置30分钟提醒"
        else:
            return "预计坐姿时长较短，保持良好习惯"

    def _format_hour_ranges(self, hours: List[int]) -> str:
        """格式化小时范围"""
        if not hours:
            return "无"

        # 合并连续的小时段
        ranges = []
        start = hours[0]
        end = hours[0]

        for i in range(1, len(hours)):
            if hours[i] == end + 1:
                end = hours[i]
            else:
                if start == end:
                    ranges.append(f"{start}时")
                else:
                    ranges.append(f"{start}-{end}时")
                start = hours[i]
                end = hours[i]

        # 添加最后一个范围
        if start == end:
            ranges.append(f"{start}时")
        else:
            ranges.append(f"{start}-{end}时")

        return "、".join(ranges)

    def _generate_anomaly_message(self, deviation_pct: float, is_anomaly: bool) -> str:
        """生成异常消息"""
        if not is_anomaly:
            return "今日坐姿时长正常"

        if deviation_pct > 0:
            return f"⚠️ 今日坐姿时长显著高于平均水平（+{abs(deviation_pct):.0f}%），请注意多活动"
        else:
            return f"✅ 今日坐姿时长低于平均水平（{abs(deviation_pct):.0f}%），保持良好习惯！"


class SmartReminder:
    """智能提醒系统

    基于预测模型提供个性化提醒
    """

    def __init__(self, predictor: SittingPredictor):
        self.predictor = predictor
        self.last_reminder_time = None

    def should_remind(self, current_sitting_duration_minutes: float) -> Tuple[bool, str]:
        """判断是否应该提醒

        Args:
            current_sitting_duration_minutes: 当前坐姿持续时长（分钟）

        Returns:
            Tuple[bool, str]: (是否提醒, 提醒消息)
        """
        # 获取最佳提醒间隔
        optimal_reminder = self.predictor.predict_optimal_reminder_time()
        reminder_interval = optimal_reminder.get('recommended_reminder_interval', 30)

        # 判断是否到达提醒时间
        if current_sitting_duration_minutes >= reminder_interval:
            message = self._generate_reminder_message(current_sitting_duration_minutes)
            self.last_reminder_time = datetime.now()
            return True, message

        return False, ""

    def _generate_reminder_message(self, duration_minutes: float) -> str:
        """生成提醒消息"""
        messages = [
            f"您已坐姿{duration_minutes:.0f}分钟，该起身活动了！",
            f"久坐{duration_minutes:.0f}分钟啦！站起来伸展一下吧～",
            f"坐了{duration_minutes:.0f}分钟了，来做个简单的拉伸吧！",
            f"已经坐了{duration_minutes:.0f}分钟，喝杯水、走动走动～"
        ]

        # 根据时长选择不同的消息
        import random
        return random.choice(messages)

    def get_daily_summary(self) -> str:
        """生成每日总结"""
        anomaly = self.predictor.detect_anomaly()

        if anomaly.get('error'):
            return "暂无足够数据生成总结"

        summary = f"📊 今日坐姿时长: {anomaly['today_sitting_hours']}小时\n"
        summary += f"📈 历史平均: {anomaly['average_sitting_hours']}小时\n"
        summary += f"💬 {anomaly['message']}"

        return summary
