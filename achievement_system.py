"""
Achievement System for Garbage Classification
Tracks user classifications and awards badges/achievements
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os


class AchievementSystem:
    """
    Manages user achievements, badges, and classification statistics.
    Uses SQLite database for data persistence.
    """
    
    def __init__(self, db_path='achievements.db'):
        """
        Initialize the achievement system.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id TEXT PRIMARY KEY,
                total_classifications INTEGER DEFAULT 0,
                recyclable_count INTEGER DEFAULT 0,
                hazardous_count INTEGER DEFAULT 0,
                kitchen_count INTEGER DEFAULT 0,
                other_count INTEGER DEFAULT 0,
                last_classification_date TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # Classification history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                class_name TEXT,
                category TEXT,
                confidence REAL,
                timestamp TEXT,
                FOREIGN KEY (user_id) REFERENCES user_stats(user_id)
            )
        ''')
        
        # Achievements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_achievements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                achievement_id TEXT,
                unlocked_at TEXT,
                FOREIGN KEY (user_id) REFERENCES user_stats(user_id),
                UNIQUE(user_id, achievement_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_or_create_user(self, user_id: str = 'default') -> Dict:
        """
        Get user statistics or create a new user if not exists.
        
        Args:
            user_id: Unique user identifier (default: 'default' for single-user mode)
            
        Returns:
            Dictionary containing user statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT * FROM user_stats WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        if row is None:
            # Create new user
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO user_stats 
                (user_id, total_classifications, recyclable_count, hazardous_count, 
                 kitchen_count, other_count, last_classification_date, created_at, updated_at)
                VALUES (?, 0, 0, 0, 0, 0, ?, ?, ?)
            ''', (user_id, None, now, now))
            conn.commit()
            
            # Return default stats
            stats = {
                'user_id': user_id,
                'total_classifications': 0,
                'recyclable_count': 0,
                'hazardous_count': 0,
                'kitchen_count': 0,
                'other_count': 0,
                'last_classification_date': None,
                'created_at': now,
                'updated_at': now
            }
        else:
            # Return existing stats
            stats = {
                'user_id': row[0],
                'total_classifications': row[1],
                'recyclable_count': row[2],
                'hazardous_count': row[3],
                'kitchen_count': row[4],
                'other_count': row[5],
                'last_classification_date': row[6],
                'created_at': row[7],
                'updated_at': row[8]
            }
        
        conn.close()
        return stats
    
    def record_classification(self, user_id: str, class_name: str, category: str, 
                             confidence: float, user_id_param: str = 'default'):
        """
        Record a classification and update user statistics.
        
        Args:
            user_id: User identifier (deprecated, kept for compatibility)
            class_name: Predicted class name (e.g., 'plastic', 'battery')
            category: Broad category (e.g., 'Recyclable', 'Hazardous')
            confidence: Confidence score (0.0 to 1.0)
            user_id_param: Actual user identifier to use
        """
        # Use user_id_param instead of user_id
        actual_user_id = user_id_param if user_id_param != 'default' else 'default'
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current stats
        stats = self.get_or_create_user(actual_user_id)
        
        # Update statistics
        new_total = stats['total_classifications'] + 1
        category_field = f"{category.lower()}_count"
        
        # Update category count
        if category == 'Recyclable':
            new_category_count = stats['recyclable_count'] + 1
            cursor.execute('''
                UPDATE user_stats 
                SET total_classifications = ?,
                    recyclable_count = ?,
                    last_classification_date = ?,
                    updated_at = ?
                WHERE user_id = ?
            ''', (new_total, new_category_count, datetime.now().isoformat(), 
                  datetime.now().isoformat(), actual_user_id))
        elif category == 'Hazardous':
            new_category_count = stats['hazardous_count'] + 1
            cursor.execute('''
                UPDATE user_stats 
                SET total_classifications = ?,
                    hazardous_count = ?,
                    last_classification_date = ?,
                    updated_at = ?
                WHERE user_id = ?
            ''', (new_total, new_category_count, datetime.now().isoformat(), 
                  datetime.now().isoformat(), actual_user_id))
        elif category == 'Kitchen':
            new_category_count = stats['kitchen_count'] + 1
            cursor.execute('''
                UPDATE user_stats 
                SET total_classifications = ?,
                    kitchen_count = ?,
                    last_classification_date = ?,
                    updated_at = ?
                WHERE user_id = ?
            ''', (new_total, new_category_count, datetime.now().isoformat(), 
                  datetime.now().isoformat(), actual_user_id))
        elif category == 'Other':
            new_category_count = stats['other_count'] + 1
            cursor.execute('''
                UPDATE user_stats 
                SET total_classifications = ?,
                    other_count = ?,
                    last_classification_date = ?,
                    updated_at = ?
                WHERE user_id = ?
            ''', (new_total, new_category_count, datetime.now().isoformat(), 
                  datetime.now().isoformat(), actual_user_id))
        else:
            # Unknown category, just update total
            cursor.execute('''
                UPDATE user_stats 
                SET total_classifications = ?,
                    last_classification_date = ?,
                    updated_at = ?
                WHERE user_id = ?
            ''', (new_total, datetime.now().isoformat(), 
                  datetime.now().isoformat(), actual_user_id))
        
        # Record in history
        cursor.execute('''
            INSERT INTO classification_history 
            (user_id, class_name, category, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (actual_user_id, class_name, category, confidence, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_user_achievements(self, user_id: str = 'default') -> List[str]:
        """
        Get list of unlocked achievements for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of achievement IDs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT achievement_id FROM user_achievements 
            WHERE user_id = ?
        ''', (user_id,))
        
        achievements = [row[0] for row in cursor.fetchall()]
        conn.close()
        return achievements
    
    def unlock_achievement(self, user_id: str, achievement_id: str) -> bool:
        """
        Unlock an achievement for a user.
        
        Args:
            user_id: User identifier
            achievement_id: Achievement ID
            
        Returns:
            True if achievement was newly unlocked, False if already unlocked
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if already unlocked
        cursor.execute('''
            SELECT id FROM user_achievements 
            WHERE user_id = ? AND achievement_id = ?
        ''', (user_id, achievement_id))
        
        if cursor.fetchone() is not None:
            conn.close()
            return False  # Already unlocked
        
        # Unlock achievement
        cursor.execute('''
            INSERT INTO user_achievements (user_id, achievement_id, unlocked_at)
            VALUES (?, ?, ?)
        ''', (user_id, achievement_id, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        return True  # Newly unlocked
    
    def check_and_unlock_achievements(self, user_id: str = 'default', 
                                     achievement_config: Optional[Dict] = None) -> List[str]:
        """
        Check user statistics and unlock new achievements.
        
        Args:
            user_id: User identifier
            achievement_config: Achievement configuration dictionary
            
        Returns:
            List of newly unlocked achievement IDs
        """
        if achievement_config is None:
            return []
        
        stats = self.get_or_create_user(user_id)
        unlocked = []
        
        # Check each achievement
        for achievement_id, config in achievement_config.items():
            # Skip if already unlocked
            if achievement_id in self.get_user_achievements(user_id):
                continue
            
            # Check conditions
            condition_met = False
            
            if config['type'] == 'total_classifications':
                condition_met = stats['total_classifications'] >= config['threshold']
            elif config['type'] == 'category_count':
                category_field = f"{config['category'].lower()}_count"
                condition_met = stats.get(category_field, 0) >= config['threshold']
            elif config['type'] == 'hazardous_count':
                condition_met = stats['hazardous_count'] >= config['threshold']
            elif config['type'] == 'all_categories':
                # Check if user has classified at least one item from each category
                condition_met = (stats['recyclable_count'] >= config['threshold'] and
                               stats['hazardous_count'] >= config['threshold'] and
                               stats['kitchen_count'] >= config['threshold'] and
                               stats['other_count'] >= config['threshold'])
            
            if condition_met:
                if self.unlock_achievement(user_id, achievement_id):
                    unlocked.append(achievement_id)
        
        return unlocked
    
    def get_recent_classifications(self, user_id: str = 'default', limit: int = 10) -> List[Dict]:
        """
        Get recent classification history.
        
        Args:
            user_id: User identifier
            limit: Maximum number of records to return
            
        Returns:
            List of classification dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT class_name, category, confidence, timestamp
            FROM classification_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'class_name': row[0],
                'category': row[1],
                'confidence': row[2],
                'timestamp': row[3]
            })
        
        conn.close()
        return results

