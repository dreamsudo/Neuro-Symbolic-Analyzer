import os
import sys
import sqlite3
from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

class SystemEnvironment:
    @staticmethod
    def initialize():
        logger.info(f"Initializing {settings.app_name} Environment...")
        dirs = [
            settings.paths.raw_data,
            settings.paths.knowledge_base,
            settings.paths.models,
            settings.paths.logs,
            settings.paths.reports,
            os.path.dirname(settings.paths.database)
        ]
        for d in dirs:
            if not os.path.exists(d):
                try:
                    os.makedirs(d)
                    logger.info(f"Created directory: {d}")
                except OSError as e:
                    logger.error(f"Failed to create directory {d}: {e}")
                    sys.exit(1)
        
        SystemEnvironment._init_db()
        logger.info("Environment initialized successfully.")

    @staticmethod
    def _init_db():
        try:
            conn = sqlite3.connect(settings.paths.database)
            cursor = conn.cursor()
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
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
