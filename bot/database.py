import sqlite3
from typing import Dict, Any, Optional
import json
import threading
import os
import logging

class Database:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                instance = super(Database, cls).__new__(cls)
                # Используем путь к текущему файлу для создания базы данных
                current_dir = os.path.dirname(os.path.abspath(__file__))
                instance.db_path = os.path.join(current_dir, 'user_data.db')
                cls._instance = instance
                instance.init_db()
            return cls._instance
    
    def __init__(self):
        """Этот метод может быть вызван несколько раз, поэтому здесь не должно быть инициализации"""
        pass
        
    def init_db(self):
        """Инициализация базы данных и создание необходимых таблиц"""
        try:
            logging.info(f'Initializing database at {self.db_path}')
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
                logging.info('Database initialized successfully')
        except Exception as e:
            logging.error(f'Error initializing database: {e}', exc_info=True)
            raise
    
    def save_user_settings(self, user_id: int, settings: Dict[str, Any]) -> None:
        """Сохранение пользовательских настроек"""
        try:
            logging.info(f'Saving settings for user_id={user_id}')
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                settings_json = json.dumps(settings, ensure_ascii=False)
                cursor.execute('''
                    INSERT INTO user_settings (user_id, settings)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET 
                    settings = excluded.settings,
                    updated_at = CURRENT_TIMESTAMP
                ''', (user_id, settings_json))
                conn.commit()
                logging.info(f'Settings saved successfully for user_id={user_id}')
        except Exception as e:
            logging.error(f'Error saving user settings: {e}', exc_info=True)
            raise
    
    def get_user_settings(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Получение пользовательских настроек"""
        try:
            logging.info(f'Getting settings for user_id={user_id}')
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT settings FROM user_settings WHERE user_id = ?', (user_id,))
                result = cursor.fetchone()
                if result:
                    logging.info(f'Settings found for user_id={user_id}')
                    return json.loads(result[0])
                logging.info(f'No settings found for user_id={user_id}')
                return None
        except Exception as e:
            logging.error(f'Error getting user settings: {e}', exc_info=True)
            raise
    
    def save_conversation_context(self, user_id: int, context: Dict[str, Any]) -> None:
        """Сохранение контекста разговора"""
        try:
            logging.info(f'Saving conversation context for user_id={user_id}')
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                context_json = json.dumps(context, ensure_ascii=False)
                logging.debug(f'Context to save: {context_json[:200]}...')  # Логируем только начало для безопасности
                cursor.execute('''
                    INSERT INTO conversation_context (user_id, context)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET 
                    context = excluded.context,
                    updated_at = CURRENT_TIMESTAMP
                ''', (user_id, context_json))
                conn.commit()
                logging.info(f'Conversation context saved successfully for user_id={user_id}')
        except sqlite3.Error as e:
            logging.error(f'SQLite error saving conversation context: {e}', exc_info=True)
            raise
        except Exception as e:
            logging.error(f'Error saving conversation context: {e}', exc_info=True)
            raise
    
    def get_conversation_context(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Получение контекста разговора"""
        try:
            logging.info(f'Getting conversation context for user_id={user_id}')
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT context FROM conversation_context WHERE user_id = ?', (user_id,))
                result = cursor.fetchone()
                if result:
                    logging.info(f'Conversation context found for user_id={user_id}')
                    context = json.loads(result[0])
                    logging.debug(f'Loaded context: {str(context)[:200]}...')  # Логируем только начало для безопасности
                    return context
                logging.info(f'No conversation context found for user_id={user_id}')
                return None
        except sqlite3.Error as e:
            logging.error(f'SQLite error getting conversation context: {e}', exc_info=True)
            raise
        except json.JSONDecodeError as e:
            logging.error(f'JSON decode error in conversation context: {e}', exc_info=True)
            return None
        except Exception as e:
            logging.error(f'Error getting conversation context: {e}', exc_info=True)
            raise
    
    def delete_user_data(self, user_id: int) -> None:
        """Удаление всех данных пользователя"""
        try:
            logging.info(f'Deleting all data for user_id={user_id}')
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM user_settings WHERE user_id = ?', (user_id,))
                cursor.execute('DELETE FROM conversation_context WHERE user_id = ?', (user_id,))
                cursor.execute('DELETE FROM user_models WHERE user_id = ?', (user_id,))
                conn.commit()
                logging.info(f'All data deleted successfully for user_id={user_id}')
        except Exception as e:
            logging.error(f'Error deleting user data: {e}', exc_info=True)
            raise
    
    def save_user_model(self, user_id: int, model_name: str) -> None:
        """Сохранение выбранной модели пользователя"""
        try:
            logging.info(f'Saving model {model_name} for user_id={user_id}')
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
                logging.info(f'Model saved successfully for user_id={user_id}')
        except Exception as e:
            logging.error(f'Error saving user model: {e}', exc_info=True)
            raise
    
    def get_user_model(self, user_id: int) -> Optional[str]:
        """Получение выбранной модели пользователя"""
        try:
            logging.info(f'Getting model for user_id={user_id}')
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT model_name FROM user_models WHERE user_id = ?', (user_id,))
                result = cursor.fetchone()
                if result:
                    logging.info(f'Model found for user_id={user_id}: {result[0]}')
                    return result[0]
                logging.info(f'No model found for user_id={user_id}')
                return None
        except Exception as e:
            logging.error(f'Error getting user model: {e}', exc_info=True)
            raise 