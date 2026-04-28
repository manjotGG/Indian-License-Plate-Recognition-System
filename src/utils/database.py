"""
Database module for storing ALPR results
Supports SQLite for plate history and analytics
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ALPRDatabase:
    """SQLite database for ALPR results storage"""
    
    def __init__(self, db_path: str = "alpr_results.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Plates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT NOT NULL,
                    confidence REAL,
                    state TEXT,
                    district TEXT,
                    series TEXT,
                    number TEXT,
                    bbox TEXT,  -- JSON string
                    image_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    validated BOOLEAN DEFAULT 0
                )
            ''')
            
            # Processing sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT,
                    total_plates INTEGER,
                    processing_time REAL,
                    status TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_plate_result(self, plate_data: Dict, image_path: str):
        """Save a single plate detection result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO plates 
                    (plate_text, confidence, state, district, series, number, bbox, image_path, validated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    plate_data.get('plate_text', ''),
                    plate_data.get('confidence', 0.0),
                    plate_data.get('state', ''),
                    plate_data.get('district', ''),
                    plate_data.get('series', ''),
                    plate_data.get('number', ''),
                    str(plate_data.get('bbox', [])),
                    image_path,
                    plate_data.get('validated', False)
                ))
                
                conn.commit()
                logger.info(f"Saved plate result: {plate_data.get('plate_text', '')}")
                
        except Exception as e:
            logger.error(f"Failed to save plate result: {e}")
    
    def save_processing_session(self, result: Dict):
        """Save processing session information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO processing_sessions 
                    (image_path, total_plates, processing_time, status)
                    VALUES (?, ?, ?, ?)
                ''', (
                    result.get('image_path', ''),
                    len(result.get('plates', [])),
                    result.get('processing_time', 0.0),
                    result.get('status', 'unknown')
                ))
                
                # Save individual plates
                for plate in result.get('plates', []):
                    self.save_plate_result(plate, result.get('image_path', ''))
                
                conn.commit()
                logger.info(f"Saved processing session for {result.get('image_path', '')}")
                
        except Exception as e:
            logger.error(f"Failed to save processing session: {e}")
    
    def get_plate_history(self, plate_text: str = None, limit: int = 100) -> List[Dict]:
        """Get plate detection history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if plate_text:
                    cursor.execute('''
                        SELECT * FROM plates 
                        WHERE plate_text = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (plate_text, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM plates 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (limit,))
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get plate history: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total plates
                cursor.execute('SELECT COUNT(*) FROM plates')
                total_plates = cursor.fetchone()[0]
                
                # Valid plates
                cursor.execute('SELECT COUNT(*) FROM plates WHERE validated = 1')
                valid_plates = cursor.fetchone()[0]
                
                # Unique plates
                cursor.execute('SELECT COUNT(DISTINCT plate_text) FROM plates')
                unique_plates = cursor.fetchone()[0]
                
                # Processing sessions
                cursor.execute('SELECT COUNT(*) FROM processing_sessions')
                total_sessions = cursor.fetchone()[0]
                
                return {
                    'total_plates': total_plates,
                    'valid_plates': valid_plates,
                    'unique_plates': unique_plates,
                    'total_sessions': total_sessions,
                    'validation_rate': valid_plates / total_plates if total_plates > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    db = ALPRDatabase()
    
    # Example plate data
    plate_data = {
        'plate_text': 'MH 04 DV 4321',
        'confidence': 0.95,
        'state': 'MH',
        'district': '04',
        'series': 'DV',
        'number': '4321',
        'bbox': [100, 200, 300, 250],
        'validated': True
    }
    
    db.save_plate_result(plate_data, "sample.jpg")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"Database stats: {stats}")