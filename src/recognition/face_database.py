"""
Face Database Management
Stores and manages known face embeddings with SQLite backend
"""

import numpy as np
import sqlite3
import pickle
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from datetime import datetime


class FaceDatabase:
    """
    Face Database Manager

    Features:
    - Store face embeddings in SQLite
    - Support manual registration (known persons)
    - Auto-create temporary IDs for strangers
    - Track first_seen and last_seen timestamps
    """

    def __init__(self, db_path: str = "data/faces.db"):
        """
        Initialize face database

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Create database directory if not exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

        print(f"[FaceDatabase] Database initialized: {db_path}")

    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()

        # Persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                is_registered INTEGER DEFAULT 0,
                embedding BLOB NOT NULL,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                appearance_count INTEGER DEFAULT 1,
                notes TEXT
            )
        ''')

        # Face history table (for tracking multiple appearances)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence FLOAT,
                camera_id INTEGER,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        ''')

        self.conn.commit()

        print("[FaceDatabase] Tables created/verified")

    def register_person(
        self,
        name: str,
        embedding: np.ndarray,
        notes: str = ""
    ) -> int:
        """
        Manually register a known person

        Args:
            name: Person's name
            embedding: Face embedding (512-dim)
            notes: Optional notes

        Returns:
            person_id of registered person
        """
        cursor = self.conn.cursor()

        # Serialize embedding
        embedding_blob = pickle.dumps(embedding)

        cursor.execute('''
            INSERT INTO persons (name, is_registered, embedding, notes)
            VALUES (?, 1, ?, ?)
        ''', (name, embedding_blob, notes))

        self.conn.commit()
        person_id = cursor.lastrowid

        print(f"[FaceDatabase] Registered person: '{name}' (ID={person_id})")

        return person_id

    def add_stranger(self, embedding: np.ndarray) -> int:
        """
        Add a stranger (auto-create temporary ID)

        Args:
            embedding: Face embedding

        Returns:
            person_id of created stranger
        """
        cursor = self.conn.cursor()

        # Count existing strangers to generate name
        cursor.execute("SELECT COUNT(*) FROM persons WHERE is_registered = 0")
        stranger_count = cursor.fetchone()[0]

        name = f"访客{stranger_count + 1}"

        # Serialize embedding
        embedding_blob = pickle.dumps(embedding)

        cursor.execute('''
            INSERT INTO persons (name, is_registered, embedding)
            VALUES (?, 0, ?)
        ''', (name, embedding_blob))

        self.conn.commit()
        person_id = cursor.lastrowid

        print(f"[FaceDatabase] New stranger detected: '{name}' (ID={person_id})")

        return person_id

    def update_last_seen(self, person_id: int, confidence: float, camera_id: int = 0):
        """
        Update last_seen timestamp and add to history

        Args:
            person_id: Person ID
            confidence: Recognition confidence
            camera_id: Camera ID (default: 0)
        """
        cursor = self.conn.cursor()

        # Update last_seen and increment appearance_count
        cursor.execute('''
            UPDATE persons
            SET last_seen = CURRENT_TIMESTAMP,
                appearance_count = appearance_count + 1
            WHERE person_id = ?
        ''', (person_id,))

        # Add to history
        cursor.execute('''
            INSERT INTO face_history (person_id, confidence, camera_id)
            VALUES (?, ?, ?)
        ''', (person_id, confidence, camera_id))

        self.conn.commit()

    def get_all_persons(self) -> List[Dict]:
        """
        Get all persons from database

        Returns:
            List of person dicts with keys:
            - person_id, name, is_registered, embedding, first_seen, last_seen, appearance_count
        """
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT person_id, name, is_registered, embedding, first_seen, last_seen, appearance_count
            FROM persons
            ORDER BY is_registered DESC, last_seen DESC
        ''')

        persons = []
        for row in cursor.fetchall():
            person_id, name, is_registered, embedding_blob, first_seen, last_seen, appearance_count = row

            # Deserialize embedding
            embedding = pickle.loads(embedding_blob)

            persons.append({
                'person_id': person_id,
                'name': name,
                'is_registered': bool(is_registered),
                'embedding': embedding,
                'first_seen': first_seen,
                'last_seen': last_seen,
                'appearance_count': appearance_count
            })

        return persons

    def get_person(self, person_id: int) -> Optional[Dict]:
        """
        Get person by ID

        Args:
            person_id: Person ID

        Returns:
            Person dict or None if not found
        """
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT person_id, name, is_registered, embedding, first_seen, last_seen, appearance_count
            FROM persons
            WHERE person_id = ?
        ''', (person_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        person_id, name, is_registered, embedding_blob, first_seen, last_seen, appearance_count = row

        embedding = pickle.loads(embedding_blob)

        return {
            'person_id': person_id,
            'name': name,
            'is_registered': bool(is_registered),
            'embedding': embedding,
            'first_seen': first_seen,
            'last_seen': last_seen,
            'appearance_count': appearance_count
        }

    def rename_person(self, person_id: int, new_name: str):
        """
        Rename a person

        Args:
            person_id: Person ID
            new_name: New name
        """
        cursor = self.conn.cursor()

        cursor.execute('''
            UPDATE persons
            SET name = ?
            WHERE person_id = ?
        ''', (new_name, person_id))

        self.conn.commit()

        print(f"[FaceDatabase] Renamed person ID={person_id} to '{new_name}'")

    def delete_person(self, person_id: int):
        """
        Delete a person from database

        Args:
            person_id: Person ID
        """
        cursor = self.conn.cursor()

        # Delete history
        cursor.execute('DELETE FROM face_history WHERE person_id = ?', (person_id,))

        # Delete person
        cursor.execute('DELETE FROM persons WHERE person_id = ?', (person_id,))

        self.conn.commit()

        print(f"[FaceDatabase] Deleted person ID={person_id}")

    def get_statistics(self) -> Dict:
        """
        Get database statistics

        Returns:
            Dict with keys: total_persons, registered_persons, strangers
        """
        cursor = self.conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM persons')
        total = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM persons WHERE is_registered = 1')
        registered = cursor.fetchone()[0]

        strangers = total - registered

        return {
            'total_persons': total,
            'registered_persons': registered,
            'strangers': strangers
        }

    def close(self):
        """Close database connection"""
        self.conn.close()
        print("[FaceDatabase] Database closed")
