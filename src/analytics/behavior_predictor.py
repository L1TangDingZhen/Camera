"""
Behavior Pattern Predictor - 智能行为模式预测

基于历史数据学习用户的日常作息规律，预测当前时间应该处于什么状态
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class BehaviorPatternAnalyzer:
    """行为模式分析器

    学习用户的日常作息规律，识别行为模式
    """

    def __init__(self, database=None):
        """
        Args:
            database: Database实例
        """
        self.db = database
        self.patterns = {}  # 缓存的行为模式

    def analyze_hourly_patterns(self, days: int = 14) -> Dict:
        """分析每小时的行为模式

        统计过去N天，每个小时通常处于什么状态

        Args:
            days: 分析天数

        Returns:
            Dict: {
                0: {'sitting': 0.95, 'standing': 0.05, 'lying': 0.0, 'most_common': 'sitting'},
                1: {...},
                ...
                23: {...}
            }
        """
        if self.db is None:
            return {}

        # 获取历史数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        history = self.db.get_state_history(
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            limit=100000
        )

        # 按小时统计每种状态的时长
        hour_state_duration = defaultdict(lambda: defaultdict(float))

        for record in history:
            if not record.get('duration'):
                continue

            record_time = datetime.fromtimestamp(record['timestamp'])
            hour = record_time.hour
            state = record['state']
            duration = record['duration']

            hour_state_duration[hour][state] += duration

        # 计算每小时各状态的占比
        hourly_patterns = {}

        for hour in range(24):
            if hour not in hour_state_duration:
                hourly_patterns[hour] = {
                    'sitting': 0.0,
                    'standing': 0.0,
                    'lying': 0.0,
                    'sleeping': 0.0,
                    'most_common': 'unknown',
                    'confidence': 0.0,
                    'total_duration': 0.0
                }
                continue

            state_durations = hour_state_duration[hour]
            total_duration = sum(state_durations.values())

            if total_duration == 0:
                hourly_patterns[hour] = {
                    'sitting': 0.0,
                    'standing': 0.0,
                    'lying': 0.0,
                    'sleeping': 0.0,
                    'most_common': 'unknown',
                    'confidence': 0.0,
                    'total_duration': 0.0
                }
                continue

            # 计算占比
            percentages = {
                'sitting': state_durations.get('sitting', 0) / total_duration,
                'standing': state_durations.get('standing', 0) / total_duration,
                'lying': state_durations.get('lying', 0) / total_duration,
                'sleeping': state_durations.get('sleeping', 0) / total_duration
            }

            # 找出最常见的状态
            most_common = max(percentages, key=percentages.get)
            confidence = percentages[most_common]

            hourly_patterns[hour] = {
                'sitting': percentages['sitting'],
                'standing': percentages['standing'],
                'lying': percentages['lying'],
                'sleeping': percentages['sleeping'],
                'most_common': most_common,
                'confidence': confidence,
                'total_duration': total_duration
            }

        self.patterns = hourly_patterns
        return hourly_patterns

    def analyze_weekday_patterns(self, weeks: int = 4) -> Dict:
        """分析工作日vs周末的行为模式差异

        Args:
            weeks: 分析周数

        Returns:
            Dict: {
                'weekday': {...},  # 工作日模式
                'weekend': {...}   # 周末模式
            }
        """
        if self.db is None:
            return {}

        end_time = datetime.now()
        start_time = end_time - timedelta(weeks=weeks)

        history = self.db.get_state_history(
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            limit=100000
        )

        # 分别统计工作日和周末
        weekday_state_duration = defaultdict(float)
        weekend_state_duration = defaultdict(float)

        for record in history:
            if not record.get('duration'):
                continue

            record_time = datetime.fromtimestamp(record['timestamp'])
            is_weekend = record_time.weekday() >= 5  # 5=Saturday, 6=Sunday
            state = record['state']
            duration = record['duration']

            if is_weekend:
                weekend_state_duration[state] += duration
            else:
                weekday_state_duration[state] += duration

        # 计算占比
        def calc_percentages(state_durations):
            total = sum(state_durations.values())
            if total == 0:
                return {
                    'sitting': 0.0,
                    'standing': 0.0,
                    'lying': 0.0,
                    'sleeping': 0.0
                }
            return {
                state: duration / total
                for state, duration in state_durations.items()
            }

        return {
            'weekday': calc_percentages(weekday_state_duration),
            'weekend': calc_percentages(weekend_state_duration)
        }

    def get_typical_routine(self) -> List[Dict]:
        """获取典型的日常作息

        Returns:
            List[Dict]: [
                {'time_range': '0-6时', 'typical_state': 'sleeping', 'confidence': 0.95},
                {'time_range': '7-9时', 'typical_state': 'sitting', 'confidence': 0.75},
                ...
            ]
        """
        if not self.patterns:
            self.analyze_hourly_patterns()

        routine = []
        current_state = None
        start_hour = 0

        for hour in range(24):
            pattern = self.patterns.get(hour, {})
            typical_state = pattern.get('most_common', 'unknown')
            confidence = pattern.get('confidence', 0.0)

            if typical_state != current_state:
                # 状态变化，记录上一个时段
                if current_state is not None:
                    routine.append({
                        'time_range': f'{start_hour}-{hour}时',
                        'typical_state': current_state,
                        'confidence': confidence
                    })
                current_state = typical_state
                start_hour = hour

        # 记录最后一个时段
        if current_state is not None:
            routine.append({
                'time_range': f'{start_hour}-24时',
                'typical_state': current_state,
                'confidence': self.patterns.get(23, {}).get('confidence', 0.0)
            })

        return routine


class StatePredictor:
    """状态预测器

    基于行为模式，预测当前时间应该处于什么状态
    """

    def __init__(self, pattern_analyzer: BehaviorPatternAnalyzer):
        """
        Args:
            pattern_analyzer: 行为模式分析器
        """
        self.analyzer = pattern_analyzer

    def predict_current_state(self, current_time: Optional[datetime] = None) -> Dict:
        """预测当前时间应该处于什么状态

        Args:
            current_time: 当前时间，默认使用系统时间

        Returns:
            Dict: {
                'predicted_state': 'sitting',
                'confidence': 0.85,
                'probabilities': {'sitting': 0.85, 'standing': 0.10, 'lying': 0.05},
                'explanation': '根据您过去14天的数据，此时您通常在坐姿工作'
            }
        """
        if current_time is None:
            current_time = datetime.now()

        hour = current_time.hour
        is_weekend = current_time.weekday() >= 5

        # 获取行为模式
        if not self.analyzer.patterns:
            self.analyzer.analyze_hourly_patterns()

        pattern = self.analyzer.patterns.get(hour, {})

        if not pattern or pattern.get('total_duration', 0) == 0:
            return {
                'predicted_state': 'unknown',
                'confidence': 0.0,
                'probabilities': {},
                'explanation': '历史数据不足，无法预测'
            }

        predicted_state = pattern['most_common']
        confidence = pattern['confidence']

        probabilities = {
            'sitting': pattern.get('sitting', 0.0),
            'standing': pattern.get('standing', 0.0),
            'lying': pattern.get('lying', 0.0),
            'sleeping': pattern.get('sleeping', 0.0)
        }

        # 生成解释
        explanation = self._generate_explanation(
            predicted_state, hour, is_weekend, confidence
        )

        return {
            'predicted_state': predicted_state,
            'confidence': confidence,
            'probabilities': probabilities,
            'explanation': explanation,
            'time': current_time.strftime('%H:%M')
        }

    def _generate_explanation(
        self, state: str, hour: int, is_weekend: bool, confidence: float
    ) -> str:
        """生成预测解释"""

        state_names = {
            'sitting': '坐姿',
            'standing': '站立',
            'lying': '躺卧',
            'sleeping': '睡眠'
        }

        state_name = state_names.get(state, state)
        day_type = '周末' if is_weekend else '工作日'
        confidence_pct = int(confidence * 100)

        if hour >= 0 and hour < 6:
            time_desc = '凌晨'
        elif hour >= 6 and hour < 9:
            time_desc = '早晨'
        elif hour >= 9 and hour < 12:
            time_desc = '上午'
        elif hour >= 12 and hour < 14:
            time_desc = '中午'
        elif hour >= 14 and hour < 18:
            time_desc = '下午'
        elif hour >= 18 and hour < 22:
            time_desc = '晚上'
        else:
            time_desc = '深夜'

        return f'根据您过去的{day_type}{time_desc}数据，此时您有{confidence_pct}%的时间处于{state_name}状态'

    def compare_with_actual(
        self, actual_state: str, current_time: Optional[datetime] = None
    ) -> Dict:
        """对比预测状态与实际状态

        Args:
            actual_state: 实际状态
            current_time: 当前时间

        Returns:
            Dict: {
                'predicted': 'standing',
                'actual': 'sitting',
                'match': False,
                'suggestion': '根据您的习惯，现在是站立活动的时间，建议起身走动'
            }
        """
        prediction = self.predict_current_state(current_time)
        predicted_state = prediction['predicted_state']

        match = (predicted_state == actual_state)

        suggestion = self._generate_suggestion(
            predicted_state, actual_state, prediction
        )

        return {
            'predicted': predicted_state,
            'actual': actual_state,
            'match': match,
            'confidence': prediction['confidence'],
            'suggestion': suggestion,
            'explanation': prediction['explanation']
        }

    def _generate_suggestion(
        self, predicted: str, actual: str, prediction: Dict
    ) -> str:
        """生成智能建议"""

        if predicted == 'unknown':
            return ''

        if predicted == actual:
            return '✅ 您的当前状态符合日常习惯，保持良好！'

        # 状态不匹配，给出建议
        state_names = {
            'sitting': '坐姿',
            'standing': '站立',
            'lying': '躺卧',
            'sleeping': '睡眠'
        }

        predicted_name = state_names.get(predicted, predicted)
        actual_name = state_names.get(actual, actual)
        confidence_pct = int(prediction['confidence'] * 100)

        suggestions = {
            ('standing', 'sitting'): f'💡 根据您的习惯，现在通常是{predicted_name}活动的时间（{confidence_pct}%概率），建议起身走动一下',
            ('sitting', 'standing'): f'💡 根据您的习惯，现在通常是{predicted_name}工作的时间（{confidence_pct}%概率）',
            ('lying', 'sitting'): f'💡 根据您的习惯，现在通常是{predicted_name}休息的时间（{confidence_pct}%概率），建议适当休息',
            ('sleeping', 'sitting'): f'💡 根据您的习惯，现在通常是{predicted_name}时间（{confidence_pct}%概率），注意休息',
            ('standing', 'lying'): f'💡 根据您的习惯，现在通常是{predicted_name}的时间（{confidence_pct}%概率）',
            ('sitting', 'lying'): f'💡 根据您的习惯，现在通常是{predicted_name}的时间（{confidence_pct}%概率）'
        }

        return suggestions.get(
            (predicted, actual),
            f'💡 根据您的习惯，现在通常是{predicted_name}（{confidence_pct}%概率），当前是{actual_name}'
        )


class SmartBehaviorSuggestion:
    """智能行为建议系统

    整合预测、检测和建议，提供全方位的智能提醒
    """

    def __init__(self, database=None):
        """
        Args:
            database: Database实例
        """
        self.analyzer = BehaviorPatternAnalyzer(database)
        self.predictor = StatePredictor(self.analyzer)

    def get_smart_suggestion(self, current_state: str) -> Dict:
        """获取智能建议

        Args:
            current_state: 当前实际状态

        Returns:
            Dict: {
                'has_suggestion': True,
                'type': 'behavior_mismatch',  # 或 'prolonged_sitting'
                'priority': 'high',  # low/medium/high
                'message': '...',
                'details': {...}
            }
        """
        # 先分析行为模式
        self.analyzer.analyze_hourly_patterns()

        # 预测当前应该做什么
        comparison = self.predictor.compare_with_actual(current_state)

        # 判断是否需要建议
        if comparison['match']:
            # 状态匹配，检查是否久坐
            if current_state == 'sitting':
                return {
                    'has_suggestion': False,
                    'type': 'normal',
                    'priority': 'low',
                    'message': '您的当前状态符合日常习惯',
                    'details': comparison
                }
            else:
                return {
                    'has_suggestion': False,
                    'type': 'normal',
                    'priority': 'low',
                    'message': comparison['suggestion'],
                    'details': comparison
                }
        else:
            # 状态不匹配
            priority = 'high' if comparison['confidence'] > 0.7 else 'medium'

            return {
                'has_suggestion': True,
                'type': 'behavior_mismatch',
                'priority': priority,
                'message': comparison['suggestion'],
                'details': comparison
            }

    def get_daily_routine_summary(self) -> Dict:
        """获取日常作息总结

        Returns:
            Dict: {
                'routine': [...],
                'summary': '您的典型作息：0-6时睡眠，7-12时坐姿工作，...'
            }
        """
        self.analyzer.analyze_hourly_patterns()
        routine = self.analyzer.get_typical_routine()

        # 生成总结文本
        summary_parts = []
        for item in routine:
            state_names = {
                'sitting': '坐姿工作',
                'standing': '站立活动',
                'lying': '躺卧休息',
                'sleeping': '睡眠',
                'unknown': '活动未知'
            }
            state_desc = state_names.get(item['typical_state'], item['typical_state'])
            summary_parts.append(f"{item['time_range']}{state_desc}")

        summary = '您的典型作息：' + '，'.join(summary_parts)

        return {
            'routine': routine,
            'summary': summary,
            'analysis_period': '过去14天'
        }
