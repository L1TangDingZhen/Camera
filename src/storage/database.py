"""
SQLite Database Manager
Stores events, state history, and performance metrics
"""

import sqlite3
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path


class Database:
    """SQLite Database Manager"""

    def __init__(self, db_path: str = "data/database.db"):
        """
        Args:
            db_path: Database file path
        """
        self.db_path = db_path

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return dictionary format

        # Initialize tables
        self._init_tables()

        print(f"[Database] Connected to database: {db_path}")

    def _init_tables(self):
        """Initialize database tables"""
        cursor = self.conn.cursor()

        # Events table
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

        # Migration: Add camera_id column to existing table
        self._migrate_add_camera_id()

        # Create indexes
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

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_camera_id
            ON events(camera_id)
        """)

        # State history table
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

        # Performance metrics table
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

        # Daily statistics table
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
        print("[Database] Database tables initialized")

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

    def _migrate_add_camera_id(self):
        """Migration: Add camera_id column to events table (if not exists)"""
        cursor = self.conn.cursor()

        # Check if column already exists
        cursor.execute("PRAGMA table_info(events)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'camera_id' not in columns:
            print("[Database] Migration: Adding camera_id column to events table...")
            cursor.execute("ALTER TABLE events ADD COLUMN camera_id INTEGER DEFAULT 0")
            self.conn.commit()
            print("[Database] Migration complete")

    # ==================== Event operations ====================

    def insert_event(self, event_type: str, timestamp: float, state: str,
                    zone: Optional[str] = None, metadata: Optional[Dict] = None,
                    tracking_id: Optional[int] = None, camera_id: Optional[int] = None):
        """Insert event"""
        cursor = self.conn.cursor()

        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute("""
            INSERT INTO events (event_type, timestamp, state, zone, metadata, tracking_id, camera_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (event_type, timestamp, state, zone, metadata_json, tracking_id, camera_id))

        self.conn.commit()

    def get_events(self, start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  event_type: Optional[str] = None,
                  limit: int = 100) -> List[Dict]:
        """Query events"""
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
        """Get event count"""
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

    # ==================== State history operations ====================

    def insert_state_history(self, timestamp: float, state: str,
                            zone: Optional[str] = None, duration: float = 0):
        """Insert state history"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO state_history (timestamp, state, zone, duration)
            VALUES (?, ?, ?, ?)
        """, (timestamp, state, zone, duration))

        self.conn.commit()

    def get_state_history(self, start_time: Optional[float] = None,
                         end_time: Optional[float] = None,
                         limit: int = 1000) -> List[Dict]:
        """Query state history"""
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

    # ==================== Performance metrics operations ====================

    def insert_performance_metric(self, timestamp: float, metric_name: str,
                                  metric_value: float, metadata: Optional[Dict] = None):
        """Insert performance metrics"""
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
        """Query performance metrics"""
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

    # ==================== Daily statistics operations ====================

    def upsert_daily_stats(self, date: str, stats: Dict):
        """Insert or update daily statistics"""
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
        """Get statistics for a specific day"""
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
        """Get statistics within date range"""
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

    # ==================== Utility methods ====================

    def backup(self, backup_path: str):
        """Backup database"""
        import shutil

        shutil.copy2(self.db_path, backup_path)
        print(f"[Database] Database backed up to: {backup_path}")

    def cleanup_old_data(self, retention_days: int = 90):
        """Clean old data"""
        cursor = self.conn.cursor()

        cutoff_time = (datetime.now() - timedelta(days=retention_days)).timestamp()

        # Clean old events
        cursor.execute("""
            DELETE FROM events WHERE timestamp < ?
        """, (cutoff_time,))

        # Clean old state history
        cursor.execute("""
            DELETE FROM state_history WHERE timestamp < ?
        """, (cutoff_time,))

        # Clean old performance metrics
        cursor.execute("""
            DELETE FROM performance_metrics WHERE timestamp < ?
        """, (cutoff_time,))

        self.conn.commit()

        print(f"[Database] Cleaned data older than {retention_days} days")

    def close(self):
        """Close database connection"""
        self.conn.close()
        print("[Database] Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
