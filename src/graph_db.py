import sqlite3
import os
from typing import List, Dict, Tuple
from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

class GraphDatabase:
    def __init__(self):
        self.db_path = settings.paths.database
        self._init_db()

    def _init_db(self):
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                name TEXT,
                category TEXT,
                description TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT,
                target TEXT,
                relation TEXT,
                probability REAL,
                FOREIGN KEY(source) REFERENCES nodes(id),
                FOREIGN KEY(target) REFERENCES nodes(id),
                PRIMARY KEY (source, target, relation)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                threat_id TEXT,
                is_correct BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def add_node(self, id: str, name: str, category: str, description: str = ""):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("INSERT OR IGNORE INTO nodes (id, name, category, description) VALUES (?, ?, ?, ?)",
                         (id, name, category, description))
            conn.commit()
        finally:
            conn.close()

    def add_edge(self, source: str, target: str, relation: str, probability: float):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("INSERT OR REPLACE INTO edges (source, target, relation, probability) VALUES (?, ?, ?, ?)",
                         (source, target, relation, probability))
            conn.commit()
        finally:
            conn.close()

    def get_successors(self, node_id: str) -> List[Tuple[str, Dict]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT target, relation, probability FROM edges WHERE source=?", (node_id,))
        rows = cursor.fetchall()
        conn.close()
        return [(r[0], {"relation": r[1], "probability": r[2]}) for r in rows]

    def save_feedback(self, threat_id: str, is_correct: bool):
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO feedback (threat_id, is_correct) VALUES (?, ?)", (threat_id, is_correct))
        conn.commit()
        conn.close()
        logger.info(f"Feedback saved for {threat_id}: Correct={is_correct}")

    def get_feedback_weight(self, threat_id: str) -> float:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT is_correct FROM feedback WHERE threat_id=?", (threat_id,))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows: return 1.0
        correct_count = sum(1 for r in rows if r[0])
        return correct_count / len(rows)
