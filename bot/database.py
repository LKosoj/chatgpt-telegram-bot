import sqlite3
from typing import Dict, Any, Optional, List
import json
import threading
import os
import logging
import hashlib

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
                        parse_mode TEXT NOT NULL,
                        temperature FLOAT NOT NULL,
                        max_tokens_percent INTEGER DEFAULT 100,
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

                # Таблица для хранения информации об изображениях
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        chat_id INTEGER NOT NULL,
                        file_id TEXT NOT NULL,
                        file_id_hash TEXT NOT NULL,
                        file_path TEXT,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES user_settings(user_id)
                    )
                ''')

                # Добавляем индекс для быстрого поиска по хешу
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_file_id_hash ON images(file_id_hash)
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
    
    def save_conversation_context(self, user_id: int, context: Dict[str, Any], parse_mode: str, temperature: float, max_tokens_percent: int = 100) -> None:
        """Сохранение контекста разговора"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                context_json = json.dumps(context, ensure_ascii=False)
                logging.debug(f'Context to save: {context_json[:200]}...')  # Логируем только начало для безопасности
                cursor.execute('''
                    INSERT INTO conversation_context (user_id, context, parse_mode, temperature, max_tokens_percent)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET 
                    context = excluded.context,
                    parse_mode = excluded.parse_mode,
                    temperature = excluded.temperature,
                    max_tokens_percent = excluded.max_tokens_percent,
                    updated_at = CURRENT_TIMESTAMP
                ''', (user_id, context_json, parse_mode, temperature, max_tokens_percent))
                conn.commit()
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
                cursor.execute('SELECT context, parse_mode, temperature, max_tokens_percent FROM conversation_context WHERE user_id = ?', (user_id,))
                result = cursor.fetchone()
                if result:
                    context = json.loads(result[0]) if result[0] is not None else {}
                    parse_mode = result[1] if result[1] is not None else 'HTML'
                    temperature = round(result[2], 2) if result[2] is not None else 0.8
                    max_tokens_percent = result[3] if result[3] is not None else 100
                    logging.debug(f'Loaded context: {str(context)[:200]}...')  # Логируем только начало для безопасности
                    return context, parse_mode, temperature, max_tokens_percent
                logging.info(f'No conversation context found for user_id={user_id}')
                return None, 'HTML', 0.8, 80
        except sqlite3.Error as e:
            logging.error(f'SQLite error getting conversation context: {e}', exc_info=True)
            return None, 'HTML', 0.8, 80
        except json.JSONDecodeError as e:
            logging.error(f'JSON decode error in conversation context: {e}', exc_info=True)
            return None, 'HTML', 0.8, 80
        except Exception as e:
            logging.error(f'Error getting conversation context: {e}', exc_info=True)
            return None, 'HTML', 0.8, 80
    
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
        except Exception as e:
            logging.error(f'Error saving user model: {e}', exc_info=True)
            raise
    
    def get_user_model(self, user_id: int) -> Optional[str]:
        """Получение выбранной модели пользователя"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT model_name FROM user_models WHERE user_id = ?', (user_id,))
                result = cursor.fetchone()
                if result:
                    return result[0]
                return None
        except Exception as e:
            logging.error(f'Error getting user model: {e}', exc_info=True)
            raise

    def save_image(self, user_id: int, chat_id: int, file_id: str, file_path: Optional[str] = None) -> int:
        """Сохранение информации об изображении"""
        try:
            # Генерируем хеш для file_id
            hash_object = hashlib.md5(file_id.encode())
            file_id_hash = hash_object.hexdigest()[:8]

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO images (user_id, chat_id, file_id, file_id_hash, file_path)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, chat_id, file_id, file_id_hash, file_path))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logging.error(f'Error saving image: {e}', exc_info=True)
            raise

    def get_user_images(self, user_id: int, chat_id: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение списка изображений пользователя"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if chat_id is not None:
                    cursor.execute('''
                        SELECT id, file_id, file_id_hash, file_path, status, created_at, updated_at
                        FROM images 
                        WHERE user_id = ? AND chat_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    ''', (user_id, chat_id, limit))
                else:
                    cursor.execute('''
                        SELECT id, file_id, file_id_hash, file_path, status, created_at, updated_at
                        FROM images 
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    ''', (user_id, limit))
                
                rows = cursor.fetchall()
                return [
                    {
                        'id': row[0],
                        'file_id': row[1],
                        'file_id_hash': row[2],
                        'file_path': row[3],
                        'status': row[4],
                        'created_at': row[5],
                        'updated_at': row[6]
                    }
                    for row in rows
                ]
        except Exception as e:
            logging.error(f'Error getting user images: {e}', exc_info=True)
            raise

    def update_image_status(self, image_id: int, status: str) -> None:
        """Обновление статуса изображения"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE images 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status, image_id))
                conn.commit()
        except Exception as e:
            logging.error(f'Error updating image status: {e}', exc_info=True)
            raise

    def cleanup_old_images(self, days: int = 7) -> None:
        """Очистка устаревших данных об изображениях"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM images 
                    WHERE created_at < datetime('now', '-' || ? || ' days')
                ''', (days,))
                conn.commit()
        except Exception as e:
            logging.error(f'Error cleaning up old images: {e}', exc_info=True)
            raise 