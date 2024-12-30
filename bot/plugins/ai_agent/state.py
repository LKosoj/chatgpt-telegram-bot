"""
Модуль для управления состоянием и восстановления системы
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import aiosqlite
import pickle
from dataclasses import asdict, is_dataclass
from .models import AgentRole, TaskStatus


class StateEncoder(json.JSONEncoder):
    """Кастомный JSON энкодер для сериализации сложных объектов"""
    
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (AgentRole, TaskStatus)):
            return obj.value
        return super().default(obj)


class RollbackMechanism:
    """Механизм отката состояния системы"""
    
    def __init__(self, storage_path: str, max_states: int = 10):
        """
        Инициализация механизма отката
        
        Args:
            storage_path: путь для хранения состояний
            max_states: максимальное количество сохраняемых состояний
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_states = max_states
        self.current_state_id = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Настройка логгера"""
        logger = logging.getLogger('RollbackMechanism')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
        
    async def save_state(self, state: Dict[str, Any]) -> str:
        """
        Сохранение состояния системы
        
        Args:
            state: текущее состояние системы
            
        Returns:
            str: ID сохраненного состояния
        """
        try:
            # Генерируем ID состояния
            state_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Добавляем метаданные
            state_with_meta = {
                "id": state_id,
                "timestamp": datetime.now().isoformat(),
                "state": state
            }
            
            # Сохраняем состояние в файл
            state_file = self.storage_path / f"state_{state_id}.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_with_meta, f, cls=StateEncoder, ensure_ascii=False, indent=2)
            
            self.current_state_id = state_id
            self.logger.info(f"State {state_id} saved successfully")
            
            # Очищаем старые состояния
            await self._cleanup_old_states()
            
            return state_id
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            raise
            
    async def load_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """
        Загрузка состояния системы
        
        Args:
            state_id: ID состояния для загрузки
            
        Returns:
            Optional[Dict[str, Any]]: загруженное состояние или None
        """
        try:
            state_file = self.storage_path / f"state_{state_id}.json"
            if not state_file.exists():
                self.logger.warning(f"State {state_id} not found")
                return None
                
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            self.logger.info(f"State {state_id} loaded successfully")
            return state_data["state"]
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            return None
            
    async def rollback(self, state_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Откат к предыдущему состоянию
        
        Args:
            state_id: ID состояния для отката (если не указан, используется предыдущее)
            
        Returns:
            Optional[Dict[str, Any]]: состояние после отката или None
        """
        try:
            if state_id is None:
                # Получаем список всех состояний
                states = sorted([
                    f.stem.replace('state_', '') 
                    for f in self.storage_path.glob('state_*.json')
                ])
                
                if not states:
                    self.logger.warning("No states available for rollback")
                    return None
                    
                # Находим предыдущее состояние
                current_index = states.index(self.current_state_id)
                if current_index == 0:
                    self.logger.warning("Already at oldest state")
                    return None
                    
                state_id = states[current_index - 1]
            
            # Загружаем состояние
            state = await self.load_state(state_id)
            if state:
                self.current_state_id = state_id
                self.logger.info(f"Rolled back to state {state_id}")
            
            return state
        except Exception as e:
            self.logger.error(f"Error during rollback: {str(e)}")
            return None
            
    async def _cleanup_old_states(self):
        """Очистка старых состояний"""
        try:
            states = sorted([
                f for f in self.storage_path.glob('state_*.json')
            ], key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Удаляем лишние состояния
            for state_file in states[self.max_states:]:
                state_file.unlink()
                self.logger.info(f"Removed old state file: {state_file.name}")
        except Exception as e:
            self.logger.error(f"Error cleaning up old states: {str(e)}")


class RecoveryManager:
    """Менеджер восстановления системы"""
    
    def __init__(self, db_path: str, rollback: RollbackMechanism):
        """
        Инициализация менеджера восстановления
        
        Args:
            db_path: путь к файлу базы данных
            rollback: механизм отката состояний
        """
        self.db_path = db_path
        self.rollback = rollback
        self.logger = self.rollback.logger
        
    async def initialize(self):
        """Инициализация базы данных для восстановления"""
        async with aiosqlite.connect(self.db_path) as db:
            # Создаем таблицу для хранения состояний восстановления
            await db.execute('''
                CREATE TABLE IF NOT EXISTS recovery_points (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    state_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Создаем таблицу для логов восстановления
            await db.execute('''
                CREATE TABLE IF NOT EXISTS recovery_logs (
                    id TEXT PRIMARY KEY,
                    recovery_point_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    FOREIGN KEY (recovery_point_id) REFERENCES recovery_points(id)
                )
            ''')
            
            await db.commit()
            
    async def create_recovery_point(
        self,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Создание точки восстановления
        
        Args:
            metadata: дополнительные метаданные
            
        Returns:
            str: ID точки восстановления
        """
        try:
            # Сохраняем текущее состояние
            state_id = await self.rollback.save_state({
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata
            })
            
            # Создаем запись в базе данных
            async with aiosqlite.connect(self.db_path) as db:
                recovery_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                await db.execute('''
                    INSERT INTO recovery_points (id, state_id, status, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (
                    recovery_id,
                    state_id,
                    "created",
                    json.dumps(metadata) if metadata else None
                ))
                await db.commit()
            
            self.logger.info(f"Created recovery point {recovery_id}")
            return recovery_id
        except Exception as e:
            self.logger.error(f"Error creating recovery point: {str(e)}")
            raise
            
    async def recover(self, recovery_id: str) -> bool:
        """
        Восстановление системы из точки восстановления
        
        Args:
            recovery_id: ID точки восстановления
            
        Returns:
            bool: успешность восстановления
        """
        try:
            # Получаем информацию о точке восстановления
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute('''
                    SELECT * FROM recovery_points WHERE id = ?
                ''', (recovery_id,))
                recovery_point = await cursor.fetchone()
                
                if not recovery_point:
                    self.logger.warning(f"Recovery point {recovery_id} not found")
                    return False
                
                # Создаем запись в логах
                log_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                await db.execute('''
                    INSERT INTO recovery_logs (id, recovery_point_id, action, status)
                    VALUES (?, ?, ?, ?)
                ''', (log_id, recovery_id, "start_recovery", "in_progress"))
                
                # Выполняем восстановление
                state = await self.rollback.load_state(recovery_point["state_id"])
                if not state:
                    await db.execute('''
                        UPDATE recovery_logs 
                        SET status = ?, details = ?
                        WHERE id = ?
                    ''', ("failed", "State not found", log_id))
                    await db.commit()
                    return False
                
                # Обновляем статус
                await db.execute('''
                    UPDATE recovery_logs 
                    SET status = ?, details = ?
                    WHERE id = ?
                ''', ("completed", "Recovery successful", log_id))
                await db.commit()
                
                self.logger.info(f"Successfully recovered from point {recovery_id}")
                return True
        except Exception as e:
            self.logger.error(f"Error during recovery: {str(e)}")
            if 'db' in locals() and 'log_id' in locals():
                await db.execute('''
                    UPDATE recovery_logs 
                    SET status = ?, details = ?
                    WHERE id = ?
                ''', ("failed", str(e), log_id))
                await db.commit()
            return False
            
    async def get_recovery_points(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Получение списка точек восстановления
        
        Args:
            limit: максимальное количество точек
            
        Returns:
            List[Dict[str, Any]]: список точек восстановления
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM recovery_points
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            points = await cursor.fetchall()
            return [dict(point) for point in points]
            
    async def get_recovery_logs(
        self,
        recovery_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Получение логов восстановления
        
        Args:
            recovery_id: ID точки восстановления (опционально)
            limit: максимальное количество записей
            
        Returns:
            List[Dict[str, Any]]: список логов
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            if recovery_id:
                cursor = await db.execute('''
                    SELECT * FROM recovery_logs
                    WHERE recovery_point_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (recovery_id, limit))
            else:
                cursor = await db.execute('''
                    SELECT * FROM recovery_logs
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
            
            logs = await cursor.fetchall()
            return [dict(log) for log in logs]
            
    async def cleanup_old_points(self, days: int = 30):
        """
        Очистка старых точек восстановления
        
        Args:
            days: количество дней, после которых точка считается устаревшей
        """
        try:
            threshold = datetime.now() - timedelta(days=days)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Получаем устаревшие точки
                cursor = await db.execute('''
                    SELECT id, state_id FROM recovery_points
                    WHERE timestamp < ?
                ''', (threshold,))
                old_points = await cursor.fetchall()
                
                for point in old_points:
                    # Удаляем файл состояния
                    state_file = self.rollback.storage_path / f"state_{point[1]}.json"
                    if state_file.exists():
                        state_file.unlink()
                    
                    # Удаляем записи из базы
                    await db.execute('''
                        DELETE FROM recovery_logs
                        WHERE recovery_point_id = ?
                    ''', (point[0],))
                    
                    await db.execute('''
                        DELETE FROM recovery_points
                        WHERE id = ?
                    ''', (point[0],))
                
                await db.commit()
                
            self.logger.info(f"Cleaned up {len(old_points)} old recovery points")
        except Exception as e:
            self.logger.error(f"Error cleaning up old points: {str(e)}")
            
    async def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Валидация состояния системы
        
        Args:
            state: состояние для проверки
            
        Returns:
            bool: результат валидации
        """
        try:
            required_fields = ["timestamp", "metadata"]
            
            # Проверяем наличие обязательных полей
            if not all(field in state for field in required_fields):
                self.logger.warning("Missing required fields in state")
                return False
            
            # Проверяем формат timestamp
            try:
                datetime.fromisoformat(state["timestamp"])
            except ValueError:
                self.logger.warning("Invalid timestamp format")
                return False
            
            # Проверяем метаданные
            if state["metadata"] is not None and not isinstance(state["metadata"], dict):
                self.logger.warning("Invalid metadata format")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating state: {str(e)}")
            return False 