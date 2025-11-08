#!/usr/bin/env python3
"""
Web Dashboard - 久坐提醒系统数据可视化

提供Web界面显示活动统计数据、图表和预测
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from src.storage.database import Database
from src.analytics.session_tracker import SessionTracker
from src.analytics.predictor import SittingPredictor, SmartReminder
from datetime import datetime, timedelta
import os

app = Flask(__name__)
CORS(app)

# 全局数据库和追踪器实例
db = None
tracker = None
predictor = None
smart_reminder = None

def init_db():
    """初始化数据库连接"""
    global db, tracker, predictor, smart_reminder
    db_path = 'data/database.db'

    if not os.path.exists(db_path):
        print(f"[WARN] 数据库文件不存在: {db_path}")
        print("请先运行 main.py 以创建数据库")

    db = Database(db_path)
    tracker = SessionTracker(database=db)
    predictor = SittingPredictor(database=db)
    smart_reminder = SmartReminder(predictor)
    print(f"[Web Dashboard] 数据库已连接: {db_path}")
    print(f"[Web Dashboard] 预测模块已启用")

@app.route('/')
def index():
    """主仪表盘页面"""
    return render_template('dashboard.html')

@app.route('/api/stats/today')
def get_today_stats():
    """API: 获取今日统计"""
    try:
        stats = tracker.get_today_statistics()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats/sitting')
def get_sitting_stats():
    """API: 获取坐姿统计"""
    try:
        stats = tracker.get_sitting_statistics()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats/weekly')
def get_weekly_stats():
    """API: 获取本周统计"""
    try:
        stats = tracker.get_weekly_statistics()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats/current')
def get_current_session():
    """API: 获取当前会话信息"""
    try:
        if tracker.current_session and tracker.current_session.is_active():
            session_info = tracker.get_current_session_info()
            return jsonify({
                'success': True,
                'data': session_info,
                'has_session': True
            })
        else:
            return jsonify({
                'success': True,
                'data': None,
                'has_session': False
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats/history')
def get_history():
    """API: 获取历史记录"""
    try:
        days = int(request.args.get('days', 7))

        # 计算时间范围
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # 查询数据库
        history = db.get_state_history(
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            limit=10000
        )

        return jsonify({
            'success': True,
            'data': history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/alert/prolonged_sitting')
def check_prolonged_sitting():
    """API: 检查久坐警告"""
    try:
        threshold = int(request.args.get('threshold', 30))
        is_prolonged = tracker.check_prolonged_sitting(threshold_minutes=threshold)

        result = {
            'alert': is_prolonged,
            'threshold_minutes': threshold
        }

        if is_prolonged:
            sitting_stats = tracker.get_sitting_statistics()
            result['current_duration_minutes'] = sitting_stats.get('current_sitting_duration', 0) / 60

        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/prediction/next_sitting')
def predict_next_sitting():
    """API: 预测下次坐姿时长"""
    try:
        prediction = predictor.predict_next_sitting_duration()
        return jsonify({
            'success': True,
            'data': prediction
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/prediction/optimal_reminder')
def predict_optimal_reminder():
    """API: 预测最佳提醒时间"""
    try:
        prediction = predictor.predict_optimal_reminder_time()
        return jsonify({
            'success': True,
            'data': prediction
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/prediction/anomaly')
def detect_anomaly():
    """API: 检测异常"""
    try:
        anomaly = predictor.detect_anomaly()
        return jsonify({
            'success': True,
            'data': anomaly
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def main():
    """启动Web服务器"""
    import argparse

    parser = argparse.ArgumentParser(description='Web Dashboard for Life Tracker')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # 初始化数据库
    init_db()

    print("=" * 60)
    print("🌐 久坐提醒系统 - Web Dashboard")
    print("=" * 60)
    print(f"访问地址: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)

    # 启动Flask应用
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

if __name__ == '__main__':
    main()
