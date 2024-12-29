import sqlite3
from typing import Dict, Any, Optional
import json
import threading

class Database:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Database, cls).__new__(cls)
                cls._instance.init_db()
            return cls._instance
    
    def __init__(self):
        self.db_path = 'bot/user_data.db'
        
    def init_db(self):
        """Инициализация базы данных и создание необходимых таблиц"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Таблица для пользовательских настроек
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    settings TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица для контекста разговора
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_context (
                    user_id INTEGER,
                    context TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id)
                )
            ''')
            
            # Таблица для хранения выбранной модели
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_models (
                    user_id INTEGER PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_user_settings(self, user_id: int, settings: Dict[str, Any]) -> None:
        """Сохранение пользовательских настроек"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            settings_json = json.dumps(settings)
            cursor.execute('''
                INSERT INTO user_settings (user_id, settings)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET 
                settings = excluded.settings,
                updated_at = CURRENT_TIMESTAMP
            ''', (user_id, settings_json))
            conn.commit()
    
    def get_user_settings(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Получение пользовательских настроек"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT settings FROM user_settings WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
            return None
    
    def save_conversation_context(self, user_id: int, context: Dict[str, Any]) -> None:
        """Сохранение контекста разговора"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            context_json = json.dumps(context)
            cursor.execute('''
                INSERT INTO conversation_context (user_id, context)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET 
                context = excluded.context,
                updated_at = CURRENT_TIMESTAMP
            ''', (user_id, context_json))
            conn.commit()
    
    def get_conversation_context(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Получение контекста разговора"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT context FROM conversation_context WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
            return None
    
    def delete_user_data(self, user_id: int) -> None:
        """Удаление всех данных пользователя"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM user_settings WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM conversation_context WHERE user_id = ?', (user_id,))
            conn.commit()
    
    def save_user_model(self, user_id: int, model_name: str) -> None:
        """Сохранение выбранной модели пользователя"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_models (user_id, model_name)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET 
                model_name = excluded.model_name,
                updated_at = CURRENT_TIMESTAMP
            ''', (user_id, model_name))
            conn.commit()
    
    def get_user_model(self, user_id: int) -> Optional[str]:
        """Получение выбранной модели пользователя"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT model_name FROM user_models WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            if result:
                return result[0]
            return None 