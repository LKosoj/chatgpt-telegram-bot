import sqlite3
from typing import Dict, Any, Optional, List, ContextManager, Generator
from contextlib import contextmanager
import json
import threading
import os
import logging
import hashlib
import uuid
import random
from functools import lru_cache
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


class Database:
    _instance = None
    _lock = threading.Lock()
    _connection_lock = threading.Lock()
    _local = threading.local()
    
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
        
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Контекстный менеджер для потокобезопасного доступа к соединению с базой данных.
        Каждый поток получает свое собственное соединение.
        """
        if not hasattr(self._local, 'connection'):
            with self._connection_lock:
                self._local.connection = sqlite3.connect(self.db_path)
                self._local.connection.row_factory = sqlite3.Row
        
        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise
        else:
            self._local.connection.commit()

    def __del__(self):
        """Закрываем соединения при удалении объекта"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection
        
    def init_db(self):
        """Инициализация базы данных и создание необходимых таблиц"""
        try:
            logger.info(f'Initializing database at {self.db_path}')
            with self.get_connection() as conn:
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
                
                # Таблица для контекста разговора с поддержкой сессий
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_context (
                        user_id INTEGER,
                        context TEXT NOT NULL,
                        model TEXT NOT NULL,
                        parse_mode TEXT NOT NULL,
                        temperature FLOAT NOT NULL,
                        max_tokens_percent INTEGER DEFAULT 100,
                        session_id TEXT,
                        session_name TEXT DEFAULT NULL,
                        is_active INTEGER DEFAULT 0,
                        message_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, session_id)
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

                # Проверяем необходимость миграции
                #cursor.execute("PRAGMA table_info(conversation_context)")
                #columns = [column[1] for column in cursor.fetchall()]
                
                # Если нет новых колонок, выполняем миграцию
                #if 'session_id' not in columns:
                #    logger.warning('Performing database migration for conversation_context')
                #    self.migrate_conversation_context()

                # Индекс для быстрого поиска сессий (создаем после миграции)
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_conversation_context_session 
                    ON conversation_context(user_id, session_id, is_active)
                ''')

                conn.commit()
                logger.info('Database initialized successfully')
        except Exception as e:
            logger.error(f'Error initializing database: {e}', exc_info=True)
            raise
    
    def save_user_settings(self, user_id: int, settings: Dict[str, Any]) -> None:
        """Сохранение пользовательских настроек"""
        try:
            logger.info(f'Saving settings for user_id={user_id}')
            with self.get_connection() as conn:
                cursor = conn.cursor()
                settings_json = json.dumps(settings, ensure_ascii=False)
                cursor.execute('''
                    INSERT INTO user_settings (user_id, settings)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET 
                    settings = excluded.settings,
                    updated_at = CURRENT_TIMESTAMP
                ''', (user_id, settings_json))
        except Exception as e:
            logger.error(f'Error saving user settings: {e}', exc_info=True)
            raise
    
    def get_user_settings(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Получение пользовательских настроек"""
        try:
            logger.info(f'Getting settings for user_id={user_id}')
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT settings FROM user_settings WHERE user_id = ?', (user_id,))
                result = cursor.fetchone()
                if result:
                    logger.info(f'Settings found for user_id={user_id}')
                    return json.loads(result[0])
                logger.info(f'No settings found for user_id={user_id}')
                return None
        except Exception as e:
            logger.error(f'Error getting user settings: {e}', exc_info=True)
            raise
    
    def get_active_session_id(self, user_id: int) -> Optional[str]:
        """Получение ID активной сессии"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT session_id FROM conversation_context WHERE user_id = ? AND is_active = 1
            ''', (user_id,))
            result = cursor.fetchone()
            return result[0] if result else None

    def save_conversation_context(self, user_id: int, context: Dict[str, Any], parse_mode: str, temperature: float, max_tokens_percent: int = 100, session_id: str = None, openai_helper = None) -> None:
        """Сохранение контекста разговора с поддержкой сессий"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                context_json = json.dumps(context, ensure_ascii=False)
                
                # Проверяем наличие активной сессии
                cursor.execute('''
                    SELECT session_id FROM conversation_context 
                    WHERE user_id = ? AND is_active = 1
                ''', (user_id,))
                result = cursor.fetchone()
                
                # Если нет активной сессии, создаем новую
                if not result and not session_id:
                    logger.info(f"Создаем новую сессию для пользователя {user_id}")
                    session_id = self.create_session(user_id, openai_helper=openai_helper)
                    if not session_id:
                        raise ValueError(f"Не удалось создать сессию для пользователя {user_id}")
                elif not session_id and result:
                    session_id = result[0]
                
                # Если сессия всё ещё не определена, вызываем исключение
                if not session_id:
                    raise ValueError(f"Не удалось определить сессию для пользователя {user_id}")
                
                # Считаем количество пользовательских сообщений в сессии
                message_count = len([msg for msg in context.get('messages', []) if msg.get('role') == 'user'])

                # Обновляем существующую сессию
                cursor.execute('''
                    UPDATE conversation_context 
                    SET context = ?, 
                        parse_mode = ?, 
                        temperature = ?, 
                        max_tokens_percent = ?,
                        message_count = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND session_id = ?
                ''', (context_json, parse_mode, temperature, max_tokens_percent, message_count, user_id, session_id))
                
                if cursor.rowcount == 0:
                    logger.info(f"cursor.rowcount: {cursor.rowcount}")


                # Если ни одна строка не обновлена, создаем новую запись
                if cursor.rowcount == 0:
                    logger.info(f"Создаем новую запись для сессии {session_id}")
                    cursor.execute('''
                        INSERT INTO conversation_context 
                        (user_id, session_id, context, parse_mode, temperature, max_tokens_percent, is_active, message_count)
                        VALUES (?, ?, ?, ?, ?, ?, 1, 1)
                    ''', (user_id, session_id, context_json, parse_mode, temperature, max_tokens_percent))

                # Если имя сессии = ... меняем его
                cursor.execute('''
                    SELECT session_name FROM conversation_context WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))
                result = cursor.fetchone()

                if result and result[0] == "...":
                    # Ищем первое сообщение пользователя
                    user_message = next(
                        (msg['content'] for msg in context.get('messages', []) 
                         if msg.get('role') == 'user'), 
                        None
                    )
                    # Если сообщение найдено, генерируем название
                    if user_message and openai_helper:
                        if len(user_message) > 20:
                            session_name, _ = openai_helper.ask_sync(
                                f"Создай короткое название (до 20 символов) для чата на основе сообщения: {user_message}\nВыведи только ОДНО название, без кавычек и никаких дополнительных символов.",
                                user_id,
                                "Ты специалист по созданию коротких и точных названий для чатов."
                            )
                            logger.info(f"!!!!!!!!Название сессии: {session_name}")
                        else:
                            session_name = user_message[:20]
                        session_name = session_name.strip()[:20]
                    else:
                        # Если сообщение не найдено, используем стандартное название
                        session_name = "..."
                    
                    cursor.execute('''
                        UPDATE conversation_context SET session_name = ? WHERE user_id = ? AND session_id = ?
                    ''', (session_name, user_id, session_id))
                                    
        except Exception as e:
            logger.error(f'Ошибка сохранения контекста сессии: {e}', exc_info=True)
            raise
    
    def get_conversation_context(self, user_id: int, session_id: str = None, openai_helper = None) -> Optional[Dict[str, Any]]:
        """Получение контекста разговора с поддержкой сессий"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Проверяем наличие активной сессии
                cursor.execute('''
                    SELECT session_id FROM conversation_context 
                    WHERE user_id = ? AND is_active = 1
                ''', (user_id,))
                result = cursor.fetchone()
                
                # Если нет активной сессии и не указан session_id, создаем новую
                if not result and not session_id:
                    logger.info(f"Создаем новую сессию для пользователя {user_id}")
                    session_id = self.create_session(user_id, openai_helper=openai_helper)
                    if not session_id:
                        logger.warning(f"Не удалось создать сессию для пользователя {user_id}")
                        return None, 'HTML', 0.8, 80, None
                elif not session_id and result:
                    session_id = result[0]
                
                # Если сессия всё ещё не определена, используем значения по умолчанию
                if not session_id:
                    logger.warning(f"Не удалось определить сессию для пользователя {user_id}")
                    return None, 'HTML', 0.8, 80, None
                
                cursor.execute('''
                    SELECT context, parse_mode, temperature, max_tokens_percent 
                    FROM conversation_context 
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))
                
                result = cursor.fetchone()
                if result:
                    context = json.loads(result[0]) if result[0] is not None else {'messages': []}
                    parse_mode = result[1] if result[1] is not None else 'HTML'
                    temperature = round(result[2], 2) if result[2] is not None else 0.8
                    max_tokens_percent = result[3] if result[3] is not None else 100
                    
                    return context, parse_mode, temperature, max_tokens_percent, session_id
                
                logger.info(f"Контекст не найден для сессии {session_id}, возвращаем значения по умолчанию")
                return None, 'HTML', 0.8, 80, None
                
        except Exception as e:
            logger.error(f'Ошибка получения контекста сессии: {e}', exc_info=True)
            return None, 'HTML', 0.8, 80, None
    
    def delete_user_data(self, user_id: int) -> None:
        """Удаление всех данных пользователя"""
        try:
            logger.info(f'Deleting all data for user_id={user_id}')
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM user_settings WHERE user_id = ?', (user_id,))
                cursor.execute('DELETE FROM conversation_context WHERE user_id = ?', (user_id,))
                logger.info(f'All data deleted successfully for user_id={user_id}')
        except Exception as e:
            logger.error(f'Error deleting user data: {e}', exc_info=True)
            raise
    
    def save_user_model(self, user_id: int, model_name: str) -> None:
        """Сохранение выбранной модели пользователя"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Обновляем модель в активной сессии
                cursor.execute('''
                    UPDATE conversation_context SET model = ? WHERE user_id = ? AND is_active = 1
                ''', (model_name, user_id))
        except Exception as e:
            logger.error(f'Error saving user model: {e}', exc_info=True)
            raise
    
    def save_image(self, user_id: int, chat_id: int, file_id: str, file_path: Optional[str] = None) -> int:
        """Сохранение информации об изображении"""
        try:
            # Генерируем хеш для file_id
            hash_object = hashlib.md5(file_id.encode())
            file_id_hash = hash_object.hexdigest()[:8]

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO images (user_id, chat_id, file_id, file_id_hash, file_path)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, chat_id, file_id, file_id_hash, file_path))
                return cursor.lastrowid
        except Exception as e:
            logger.error(f'Error saving image: {e}', exc_info=True)
            raise

    def get_user_images(self, user_id: int, chat_id: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение списка изображений пользователя"""
        try:
            with self.get_connection() as conn:
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
            logger.error(f'Error getting user images: {e}', exc_info=True)
            raise

    def update_image_status(self, image_id: int, status: str) -> None:
        """Обновление статуса изображения"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE images 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status, image_id))
        except Exception as e:
            logger.error(f'Error updating image status: {e}', exc_info=True)
            raise

    def cleanup_old_images(self, days: int = 7) -> None:
        """Очистка устаревших данных об изображениях"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM images 
                    WHERE created_at < datetime('now', '-' || ? || ' days')
                ''', (days,))
        except Exception as e:
            logger.error(f'Error cleaning up old images: {e}', exc_info=True)
            raise

    def count_user_sessions(self, user_id: int) -> int:
        """
        Подсчет количества сессий пользователя
        
        :param user_id: Идентификатор пользователя
        :return: Количество активных сессий
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM conversation_context 
                    WHERE user_id = ?
                """, (user_id,))
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Ошибка при подсчете сессий пользователя: {e}")
            return 0

    def delete_oldest_session(self, user_id: int) -> bool:
        """
        Удаление самых старых сессий пользователя до достижения лимита
        
        :param user_id: Идентификатор пользователя
        :return: Успешность удаления
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Максимальное количество сессий
                max_sessions=os.getenv('MAX_SESSIONS', 5)
                
                # SQL-запрос для удаления старых сессий
                cursor.execute("""
                    DELETE FROM conversation_context 
                    WHERE user_id = ? AND session_id IN (
                        SELECT session_id 
                        FROM conversation_context 
                        WHERE user_id = ? 
                        ORDER BY created_at ASC 
                        LIMIT MAX(0, (
                            (SELECT COUNT(*) FROM conversation_context WHERE user_id = ?) - (? - 1)
                        ))
                    )
                """, (user_id, user_id, user_id, max_sessions))
                
                return True
        
        except sqlite3.Error as e:
            logger.error(f"Ошибка при удалении старых сессий: {e}")
            return False

    def get_mode_from_context(self, context: Dict[str, Any]) -> Optional[Dict]:
        """
        Получает системное сообщение из контекста сессии
        
        :param context: Контекст сессии
        :return: Системное сообщение или None
        """
        try:
            if isinstance(context, dict) and 'messages' in context:
                system_messages = [
                    msg for msg in context.get('messages', []) 
                    if msg.get('role') == 'system'
                ]
                                
                if system_messages:
                    return system_messages[0]
                
                return None
            
            logger.warning(f"Некорректный формат контекста: {type(context)}")
            return None
        
        except Exception as e:
            logger.error(f'Ошибка получения системного сообщения из контекста: {e}', exc_info=True)
            return None

    def create_session(
        self, 
        user_id: int, 
        session_name: str = None, 
        max_sessions: int = 5,
        first_message: str = None,
        openai_helper = None
    ) -> Optional[str]:
        """
        Создание новой сессии с сохранением режима из активной сессии
        
        :param user_id: Идентификатор пользователя
        :param session_name: Название сессии (опционально)
        :param max_sessions: Максимальное количество активных сессий
        :param first_message: Первое сообщение для генерации названия
        :param openai_helper: Экземпляр OpenAIHelper для генерации названия
        :return: Идентификатор новой сессии или None
        """
        try:
            # Удаляем старые сессии
            self.delete_oldest_session(user_id)
            
            # Получаем данные из активной сессии
            sessions = self.list_user_sessions(user_id, 1)
            
            active_session = next((s for s in sessions if s['is_active']), None)

            # Значения по умолчанию
            parse_mode = 'HTML'
            temperature = 0.8
            max_tokens_percent = 100
            system_message = None

            # Если нет активной сессии, создаем новую принудительно
            if not active_session:
                # Генерируем новый session_id
                session_id = str(uuid.uuid4())
                
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Создаем начальный контекст
                    context = {
                        'messages': [],
                    }
                    
                    # Создаем новую сессию
                    cursor.execute("""
                        INSERT INTO conversation_context 
                        (user_id, context, parse_mode, temperature, max_tokens_percent, 
                         session_id, session_name, created_at, is_active, message_count, model) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?)
                    """, (
                        user_id,
                        json.dumps(context),
                        parse_mode,
                        temperature,
                        max_tokens_percent,
                        session_id, 
                        "...", 
                        datetime.now(),
                        openai_helper.config['model'] if openai_helper else 'deepseek/deepseek-chat-0324-alt-structured'
                    ))
                    
                    logger.info(f"Создана первая сессия {session_id} для пользователя {user_id}")
                    return session_id
            
            # Если есть активная сессия, используем стандартную логику            
            if active_session:
                # Получаем настройки из активной сессии
                parse_mode = active_session.get('parse_mode', parse_mode)
                temperature = active_session.get('temperature', temperature)
                max_tokens_percent = active_session.get('max_tokens_percent', max_tokens_percent)
                
                # Получаем системное сообщение
                system_message = self.get_mode_from_context(active_session['context'])
                                
                # Генерируем новый session_id
                session_id = str(uuid.uuid4())
                
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Деактивируем все предыдущие сессии
                    cursor.execute("""
                        UPDATE conversation_context 
                        SET is_active = 0 
                        WHERE user_id = ?
                    """, (user_id,))

                    # Определяем название сессии
                    final_session_name = "..."
                    
                    # Создаем начальный контекст
                    context = {
                        'messages': [system_message] if system_message else [],
                    }
                    context_json = json.dumps(context, ensure_ascii=False)
                    # Создаем новую сессию
                    cursor.execute("""
                        INSERT INTO conversation_context 
                        (user_id, context, parse_mode, temperature, max_tokens_percent, 
                         session_id, session_name, created_at, is_active, message_count, model) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?)
                    """, (
                        user_id,
                        context_json,
                        parse_mode,
                        temperature,
                        max_tokens_percent,
                        session_id, 
                        final_session_name, 
                        datetime.now(),
                        openai_helper.config['model'] if openai_helper else 'deepseek/deepseek-chat-0324-alt-structured'
                    ))
                    
                    return session_id
                
            return None
                
        except Exception as e:
            logger.error(f"Ошибка при создании сессии: {e}", exc_info=True)
            return None

    def list_user_sessions(self, user_id: int, is_active: int = 0) -> List[Dict[str, Any]]:
        """Получение списка сессий пользователя"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        session_id, 
                        session_name, 
                        is_active, 
                        message_count, 
                        updated_at,
                        context,
                        parse_mode,
                        temperature,
                        max_tokens_percent,
                        model
                    FROM conversation_context 
                    WHERE user_id = ? AND session_id IS NOT NULL AND is_active = case when ? = 1 then 1 else is_active end
                    ORDER BY updated_at DESC
                ''', (user_id, is_active))
                sessions = cursor.fetchall()
                return [
                    {
                        'session_id': session[0],
                        'session_name': session[1],
                        'is_active': bool(session[2]),
                        'message_count': session[3],
                        'updated_at': session[4],
                        'context': json.loads(session[5]) if session[5] else {},
                        'parse_mode': session[6],
                        'temperature': session[7],
                        'max_tokens_percent': session[8],
                        'model': session[9]
                    } for session in sessions
                ]
        except Exception as e:
            logger.error(f'Ошибка получения списка сессий: {e}', exc_info=True)
            return []

    def switch_active_session(self, user_id: int, session_id: str):
        """Переключение активной сессии"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Деактивируем все сессии пользователя
                cursor.execute('''
                    UPDATE conversation_context 
                    SET is_active = 0 
                    WHERE user_id = ?
                ''', (user_id,))
                
                # Активируем выбранную сессию
                cursor.execute('''
                    UPDATE conversation_context 
                    SET is_active = 1 
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))
        except Exception as e:
            logger.error(f'Ошибка переключения сессии: {e}', exc_info=True)
            raise

    def delete_session(self, user_id: int, session_id: str, openai_helper=None):
        """Удаление сессии"""
        try:
            # Проверяем количество сессий пользователя
            session_count = self.count_user_sessions(user_id)
                
            # Если это последняя сессия, создаем новую перед удалением
            if session_count == 1:
                # Создаем новую сессию
                new_session_id = self.create_session(user_id, openai_helper=openai_helper)
                
                if not new_session_id:
                    logger.error(f"Не удалось создать новую сессию для пользователя {user_id}")

            # Получаем данные из активной сессии
            sessions = self.list_user_sessions(user_id, 1)
            active_session = next((s for s in sessions if s['is_active']), None)

            # Если удаляется активная сессия, создаем новую
            if active_session.get('session_id') == session_id:
                new_session_id = self.create_session(user_id, openai_helper=openai_helper)
                if not new_session_id:
                    logger.error(f"Не удалось создать новую сессию для пользователя {user_id}")
                else:
                    self.switch_active_session(user_id, new_session_id)
            
            # Удаляем сессию
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Удаляем сессию
                cursor.execute('''
                    DELETE FROM conversation_context 
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))
            
        except Exception as e:
            logger.error(f'Ошибка удаления сессии: {e}', exc_info=True)
            raise

    def migrate_conversation_context(self):
        """
        Миграция существующих данных в новую структуру сессий
        Преобразует существующие записи в первую сессию для каждого пользователя
        """
        try:
            logger.info('Начало миграции conversation_context')
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Проверяем существующие колонки
                cursor.execute("PRAGMA table_info(conversation_context)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'session_id' not in columns:
                    # Создаем временную таблицу
                    cursor.execute('ALTER TABLE conversation_context RENAME TO conversation_context_old')
                    
                    # Создаем новую таблицу с поддержкой сессий
                    cursor.execute('''
                        CREATE TABLE conversation_context (
                            user_id INTEGER,
                            context TEXT NOT NULL,
                            parse_mode TEXT NOT NULL,
                            temperature FLOAT NOT NULL,
                            max_tokens_percent INTEGER DEFAULT 100,
                            session_id TEXT,
                            session_name TEXT DEFAULT NULL,
                            is_active INTEGER DEFAULT 0,
                            message_count INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (user_id, session_id)
                        )
                    ''')
                    
                    # Создаем индекс для поиска сессий
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_conversation_context_session 
                        ON conversation_context(user_id, session_id, is_active)
                    ''')
                    
                    # Миграция данных с генерацией сессий
                    cursor.execute('''
                        INSERT INTO conversation_context 
                        (user_id, context, parse_mode, temperature, max_tokens_percent, 
                         session_id, session_name, is_active, message_count, created_at, updated_at)
                        SELECT 
                            user_id, 
                            COALESCE(context, '[]'),
                            COALESCE(parse_mode, 'HTML'),
                            COALESCE(temperature, 0.8),
                            COALESCE(max_tokens_percent, 100),
                            hex(randomblob(16)) as session_id,
                            'Первоначальная сессия' as session_name,
                            1 as is_active,
                            0 as message_count,
                            COALESCE(created_at, CURRENT_TIMESTAMP),
                            COALESCE(updated_at, CURRENT_TIMESTAMP)
                        FROM conversation_context_old
                    ''')
                    
                    # Удаляем старую таблицу
                    cursor.execute('DROP TABLE conversation_context_old')
                    
                    logger.info('Миграция conversation_context завершена успешно')
                else:
                    logger.info('Миграция не требуется, таблица уже в новом формате')
        except Exception as e:
            logger.error(f'Ошибка миграции базы данных: {e}', exc_info=True)
            raise 

    def get_session_details(self, user_id: int, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает детали конкретной сессии.
        
        :param user_id: ID пользователя
        :param session_id: ID сессии
        :return: Словарь с деталями сессии или None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        session_id, 
                        session_name, 
                        is_active, 
                        message_count, 
                        created_at, 
                        context
                    FROM conversation_context 
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))
                result = cursor.fetchone()
                
                if result:
                    session_details = {
                        'session_id': result[0],
                        'session_name': result[1],
                        'is_active': bool(result[2]),
                        'message_count': result[3],
                        'created_at': result[4],
                        'context': json.loads(result[5]) if result[5] else {}
                    }
                    return session_details
                return None
        except Exception as e:
            logger.error(f'Error getting session details: {e}', exc_info=True)
            return None

    def export_sessions_to_yaml(self, user_id: int) -> str:
        """
        Экспорт всех сессий пользователя в YAML-файл
        
        :param user_id: Идентификатор пользователя
        :return: Путь к сгенерированному YAML-файлу
        """
        try:
            # Получаем список всех сессий пользователя
            sessions = self.list_user_sessions(user_id)
            
            # Подготавливаем данные для экспорта
            export_data = {
                'user_id': user_id,
                'total_sessions': len(sessions),
                'sessions': []
            }
            
            for session in sessions:
                session_export = {
                    'session_id': session['session_id'],
                    'session_name': session['session_name'],
                    'is_active': session['is_active'],
                    'message_count': session['message_count'],
                    'updated_at': str(session['updated_at']),
                    'parse_mode': session['parse_mode'],
                    'temperature': session['temperature'],
                    'max_tokens_percent': session['max_tokens_percent'],
                    'context': session['context']
                }
                export_data['sessions'].append(session_export)
            
            # Создаем директорию для экспорта, если она не существует
            export_dir = os.path.join(os.getcwd(), 'exports')
            os.makedirs(export_dir, exist_ok=True)
            
            # Генерируем имя файла с отметкой времени
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'sessions_export_{user_id}_{timestamp}.yaml'
            filepath = os.path.join(export_dir, filename)
            
            # Сохраняем в YAML
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.safe_dump(export_data, f, allow_unicode=True, default_flow_style=False)
            
            logger.info(f"Экспортировано сессий: {len(sessions)} в файл {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f'Ошибка экспорта сессий в YAML: {e}', exc_info=True)
            return None 