"""
SQLite数据库管理
存储事件、状态历史和性能指标
"""

import sqlite3
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path


class Database:
    """SQLite数据库管理器"""

    def __init__(self, db_path: str = "data/database.db"):
        """
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path

        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 连接数据库
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # 返回字典格式

        # 初始化表
        self._init_tables()

        print(f"[Database] 数据库已连接: {db_path}")

    def _init_tables(self):
        """初始化数据库表"""
        cursor = self.conn.cursor()

        # 事件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                state TEXT NOT NULL,
                zone TEXT,
                metadata TEXT,
                tracking_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration: Add tracking_id column to existing table
        self._migrate_add_tracking_id()

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp
            ON events(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON events(event_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_tracking_id
            ON events(tracking_id)
        """)

        # 状态历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                state TEXT NOT NULL,
                zone TEXT,
                duration REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 性能指标表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 每日统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                total_events INTEGER,
                sleep_duration REAL,
                sitting_duration REAL,
                lying_duration REAL,
                standing_duration REAL,
                night_bathroom_count INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()
        print("[Database] 数据表初始化完成")

    def _migrate_add_tracking_id(self):
        """Migration: Add tracking_id column to events table (if not exists)"""
        cursor = self.conn.cursor()

        # Check if column already exists
        cursor.execute("PRAGMA table_info(events)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'tracking_id' not in columns:
            print("[Database] Migration: Adding tracking_id column to events table...")
            cursor.execute("ALTER TABLE events ADD COLUMN tracking_id INTEGER")
            self.conn.commit()
            print("[Database] Migration complete")

    # ==================== 事件操作 ====================

    def insert_event(self, event_type: str, timestamp: float, state: str,
                    zone: Optional[str] = None, metadata: Optional[Dict] = None,
                    tracking_id: Optional[int] = None):
        """Insert event"""
        cursor = self.conn.cursor()

        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute("""
            INSERT INTO events (event_type, timestamp, state, zone, metadata, tracking_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (event_type, timestamp, state, zone, metadata_json, tracking_id))

        self.conn.commit()

    def get_events(self, start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  event_type: Optional[str] = None,
                  limit: int = 100) -> List[Dict]:
        """查询事件"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        events = []
        for row in cursor.fetchall():
            event = dict(row)
            if event['metadata']:
                event['metadata'] = json.loads(event['metadata'])
            events.append(event)

        return events

    def get_event_count(self, start_time: Optional[float] = None,
                       end_time: Optional[float] = None) -> int:
        """获取事件数量"""
        cursor = self.conn.cursor()

        query = "SELECT COUNT(*) FROM events WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        cursor.execute(query, params)
        return cursor.fetchone()[0]

    # ==================== 状态历史操作 ====================

    def insert_state_history(self, timestamp: float, state: str,
                            zone: Optional[str] = None, duration: float = 0):
        """插入状态历史"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO state_history (timestamp, state, zone, duration)
            VALUES (?, ?, ?, ?)
        """, (timestamp, state, zone, duration))

        self.conn.commit()

    def get_state_history(self, start_time: Optional[float] = None,
                         end_time: Optional[float] = None,
                         limit: int = 1000) -> List[Dict]:
        """查询状态历史"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM state_history WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        return [dict(row) for row in cursor.fetchall()]

    # ==================== 性能指标操作 ====================

    def insert_performance_metric(self, timestamp: float, metric_name: str,
                                  metric_value: float, metadata: Optional[Dict] = None):
        """插入性能指标"""
        cursor = self.conn.cursor()

        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute("""
            INSERT INTO performance_metrics (timestamp, metric_name, metric_value, metadata)
            VALUES (?, ?, ?, ?)
        """, (timestamp, metric_name, metric_value, metadata_json))

        self.conn.commit()

    def get_performance_metrics(self, metric_name: str,
                               start_time: Optional[float] = None,
                               end_time: Optional[float] = None,
                               limit: int = 1000) -> List[Dict]:
        """查询性能指标"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM performance_metrics WHERE metric_name = ?"
        params = [metric_name]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        metrics = []
        for row in cursor.fetchall():
            metric = dict(row)
            if metric['metadata']:
                metric['metadata'] = json.loads(metric['metadata'])
            metrics.append(metric)

        return metrics

    # ==================== 每日统计操作 ====================

    def upsert_daily_stats(self, date: str, stats: Dict):
        """插入或更新每日统计"""
        cursor = self.conn.cursor()

        metadata_json = json.dumps(stats.get('metadata', {}))

        cursor.execute("""
            INSERT OR REPLACE INTO daily_stats
            (date, total_events, sleep_duration, sitting_duration,
             lying_duration, standing_duration, night_bathroom_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date,
            stats.get('total_events', 0),
            stats.get('sleep_duration', 0),
            stats.get('sitting_duration', 0),
            stats.get('lying_duration', 0),
            stats.get('standing_duration', 0),
            stats.get('night_bathroom_count', 0),
            metadata_json
        ))

        self.conn.commit()

    def get_daily_stats(self, date: str) -> Optional[Dict]:
        """获取某天的统计"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM daily_stats WHERE date = ?
        """, (date,))

        row = cursor.fetchone()
        if row:
            stats = dict(row)
            if stats['metadata']:
                stats['metadata'] = json.loads(stats['metadata'])
            return stats

        return None

    def get_daily_stats_range(self, start_date: str, end_date: str) -> List[Dict]:
        """获取日期范围内的统计"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM daily_stats
            WHERE date >= ? AND date <= ?
            ORDER BY date ASC
        """, (start_date, end_date))

        stats_list = []
        for row in cursor.fetchall():
            stats = dict(row)
            if stats['metadata']:
                stats['metadata'] = json.loads(stats['metadata'])
            stats_list.append(stats)

        return stats_list

    # ==================== 工具方法 ====================

    def backup(self, backup_path: str):
        """备份数据库"""
        import shutil

        shutil.copy2(self.db_path, backup_path)
        print(f"[Database] 数据库已备份到: {backup_path}")

    def cleanup_old_data(self, retention_days: int = 90):
        """清理旧数据"""
        cursor = self.conn.cursor()

        cutoff_time = (datetime.now() - timedelta(days=retention_days)).timestamp()

        # 清理旧事件
        cursor.execute("""
            DELETE FROM events WHERE timestamp < ?
        """, (cutoff_time,))

        # 清理旧状态历史
        cursor.execute("""
            DELETE FROM state_history WHERE timestamp < ?
        """, (cutoff_time,))

        # 清理旧性能指标
        cursor.execute("""
            DELETE FROM performance_metrics WHERE timestamp < ?
        """, (cutoff_time,))

        self.conn.commit()

        print(f"[Database] 已清理 {retention_days} 天前的数据")

    def close(self):
        """关闭数据库连接"""
        self.conn.close()
        print("[Database] 数据库连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
