"""
Модуль для управления кэшированием
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import aiofiles
import hashlib
from .agent_logger import AgentLogger
from .exceptions import CacheError
from .utils import Cache


@dataclass
class CacheStats:
    """Статистика кэша"""
    hits: int = 0
    misses: int = 0
    size_bytes: int = 0
    items_count: int = 0
    last_cleanup: Optional[datetime] = None
    
    @property
    def hit_rate(self) -> float:
        """Коэффициент попаданий"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheItem:
    """Элемент кэша"""
    key: str
    value: Any
    priority: int
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    size_bytes: int = 0


class CacheManager:
    """Менеджер кэширования"""
    
    def __init__(
        self,
        cache_dir: str = "cache",
        max_size: int = 1024 * 1024 * 100,  # 100MB
        cleanup_interval: int = 3600,  # 1 час
        max_items: int = 10000,
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация менеджера кэширования
        
        Args:
            cache_dir: директория для кэша
            max_size: максимальный размер кэша в байтах
            cleanup_interval: интервал очистки в секундах
            max_items: максимальное количество элементов
            logger: логгер (опционально)
        """
        self.logger = logger or AgentLogger("logs")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.max_items = max_items
        
        # Базовый кэш
        self.cache = Cache(cache_dir, max_size, logger)
        
        # Метаданные кэша
        self.items: Dict[str, CacheItem] = {}
        self.priority_items: Dict[int, Set[str]] = defaultdict(set)
        
        # Статистика
        self.stats = CacheStats()
        
        # Задачи очистки
        self.cleanup_tasks: Set[asyncio.Task] = set()
        
    async def start(self):
        """Запуск менеджера кэширования"""
        # Загружаем метаданные
        await self._load_metadata()
        
        # Запускаем периодическую очистку
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self.cleanup_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self.cleanup_tasks.discard)
        
        self.logger.info("Cache manager started")
        
    async def stop(self):
        """Остановка менеджера кэширования"""
        # Отменяем задачи очистки
        for task in self.cleanup_tasks:
            task.cancel()
            
        # Ждем завершения задач
        if self.cleanup_tasks:
            await asyncio.gather(*self.cleanup_tasks, return_exceptions=True)
            
        # Сохраняем метаданные
        await self._save_metadata()
        
        self.logger.info("Cache manager stopped")
        
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Any]:
        """
        Получение значения из кэша
        
        Args:
            key: ключ
            default: значение по умолчанию
            
        Returns:
            Optional[Any]: значение
        """
        try:
            # Проверяем метаданные
            if key not in self.items:
                self.stats.misses += 1
                return default
                
            item = self.items[key]
            
            # Проверяем срок действия
            if item.expires_at and item.expires_at < datetime.now():
                await self.delete(key)
                self.stats.misses += 1
                return default
                
            # Получаем значение
            value = await self.cache.get(key)
            if value is None:
                self.stats.misses += 1
                return default
                
            # Обновляем статистику
            self.stats.hits += 1
            item.last_accessed = datetime.now()
            item.access_count += 1
            
            return value
            
        except Exception as e:
            self.logger.error(
                "Error getting from cache",
                error=str(e),
                extra={"key": key}
            )
            return default
            
    async def set(
        self,
        key: str,
        value: Any,
        expires_in: Optional[int] = None,
        priority: int = 0
    ):
        """
        Сохранение значения в кэш
        
        Args:
            key: ключ
            value: значение
            expires_in: время жизни в секундах
            priority: приоритет (0-9, где 9 - наивысший)
        """
        try:
            # Проверяем размер кэша
            if len(self.items) >= self.max_items:
                await self._cleanup_by_priority()
                
            # Сохраняем значение
            await self.cache.set(key, value, expires_in)
            
            # Создаем метаданные
            size = len(json.dumps(value).encode())
            expires_at = (
                datetime.now() + timedelta(seconds=expires_in)
                if expires_in else None
            )
            
            item = CacheItem(
                key=key,
                value=value,
                priority=priority,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=size
            )
            
            # Обновляем метаданные
            if key in self.items:
                old_priority = self.items[key].priority
                self.priority_items[old_priority].remove(key)
                
            self.items[key] = item
            self.priority_items[priority].add(key)
            
            # Обновляем статистику
            self.stats.size_bytes += size
            self.stats.items_count = len(self.items)
            
        except Exception as e:
            self.logger.error(
                "Error setting to cache",
                error=str(e),
                extra={"key": key}
            )
            
    async def delete(self, key: str):
        """
        Удаление значения из кэша
        
        Args:
            key: ключ
        """
        try:
            # Удаляем значение
            await self.cache.delete(key)
            
            # Обновляем метаданные
            if key in self.items:
                item = self.items[key]
                self.priority_items[item.priority].remove(key)
                self.stats.size_bytes -= item.size_bytes
                del self.items[key]
                
            self.stats.items_count = len(self.items)
            
        except Exception as e:
            self.logger.error(
                "Error deleting from cache",
                error=str(e),
                extra={"key": key}
            )
            
    async def clear(self):
        """Очистка кэша"""
        try:
            # Очищаем значения
            await self.cache.clear()
            
            # Очищаем метаданные
            self.items.clear()
            self.priority_items.clear()
            
            # Сбрасываем статистику
            self.stats = CacheStats()
            
            self.logger.info("Cache cleared")
            
        except Exception as e:
            self.logger.error(
                "Error clearing cache",
                error=str(e)
            )
            
    async def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики кэша
        
        Returns:
            Dict[str, Any]: статистика кэша
        """
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "size_bytes": self.stats.size_bytes,
            "size_mb": round(self.stats.size_bytes / (1024 * 1024), 2),
            "items_count": self.stats.items_count,
            "last_cleanup": (
                self.stats.last_cleanup.isoformat()
                if self.stats.last_cleanup else None
            ),
            "priority_distribution": {
                priority: len(items)
                for priority, items in self.priority_items.items()
            }
        }
        
    async def _load_metadata(self):
        """Загрузка метаданных"""
        try:
            metadata_file = self.cache_dir / "metadata.json"
            if not metadata_file.exists():
                return
                
            async with aiofiles.open(metadata_file, "r") as f:
                data = json.loads(await f.read())
                
            # Восстанавливаем элементы
            for item_data in data["items"]:
                item = CacheItem(
                    key=item_data["key"],
                    value=item_data["value"],
                    priority=item_data["priority"],
                    created_at=datetime.fromisoformat(
                        item_data["created_at"]
                    ),
                    expires_at=(
                        datetime.fromisoformat(item_data["expires_at"])
                        if item_data.get("expires_at") else None
                    ),
                    last_accessed=(
                        datetime.fromisoformat(item_data["last_accessed"])
                        if item_data.get("last_accessed") else None
                    ),
                    access_count=item_data["access_count"],
                    size_bytes=item_data["size_bytes"]
                )
                
                self.items[item.key] = item
                self.priority_items[item.priority].add(item.key)
                
            # Восстанавливаем статистику
            stats_data = data["stats"]
            self.stats = CacheStats(
                hits=stats_data["hits"],
                misses=stats_data["misses"],
                size_bytes=stats_data["size_bytes"],
                items_count=stats_data["items_count"],
                last_cleanup=(
                    datetime.fromisoformat(stats_data["last_cleanup"])
                    if stats_data.get("last_cleanup") else None
                )
            )
            
        except Exception as e:
            self.logger.error(
                "Error loading cache metadata",
                error=str(e)
            )
            
    async def _save_metadata(self):
        """Сохранение метаданных"""
        try:
            metadata = {
                "items": [
                    {
                        "key": item.key,
                        "value": item.value,
                        "priority": item.priority,
                        "created_at": item.created_at.isoformat(),
                        "expires_at": (
                            item.expires_at.isoformat()
                            if item.expires_at else None
                        ),
                        "last_accessed": (
                            item.last_accessed.isoformat()
                            if item.last_accessed else None
                        ),
                        "access_count": item.access_count,
                        "size_bytes": item.size_bytes
                    }
                    for item in self.items.values()
                ],
                "stats": {
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "size_bytes": self.stats.size_bytes,
                    "items_count": self.stats.items_count,
                    "last_cleanup": (
                        self.stats.last_cleanup.isoformat()
                        if self.stats.last_cleanup else None
                    )
                }
            }
            
            metadata_file = self.cache_dir / "metadata.json"
            async with aiofiles.open(metadata_file, "w") as f:
                await f.write(
                    json.dumps(metadata, ensure_ascii=False, indent=2)
                )
                
        except Exception as e:
            self.logger.error(
                "Error saving cache metadata",
                error=str(e)
            )
            
    async def _cleanup_by_priority(self):
        """Очистка по приоритету"""
        try:
            # Сортируем элементы по приоритету (от низшего к высшему)
            items = sorted(
                self.items.values(),
                key=lambda x: (x.priority, x.last_accessed or x.created_at)
            )
            
            # Удаляем элементы с низким приоритетом
            removed = 0
            target = len(items) - int(self.max_items * 0.9)  # Оставляем 90%
            
            for item in items:
                if removed >= target:
                    break
                    
                await self.delete(item.key)
                removed += 1
                
            self.logger.info(f"Cleaned up {removed} low priority items")
            
        except Exception as e:
            self.logger.error(
                "Error cleaning up by priority",
                error=str(e)
            )
            
    async def _periodic_cleanup(self):
        """Периодическая очистка"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Очищаем просроченные элементы
                now = datetime.now()
                expired = [
                    key for key, item in self.items.items()
                    if item.expires_at and item.expires_at < now
                ]
                
                for key in expired:
                    await self.delete(key)
                    
                # Очищаем по размеру если нужно
                if self.stats.size_bytes > self.max_size:
                    await self._cleanup_by_priority()
                    
                self.stats.last_cleanup = now
                await self._save_metadata()
                
                self.logger.info(
                    f"Periodic cleanup completed, removed {len(expired)} items"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in periodic cleanup",
                    error=str(e)
                )
                await asyncio.sleep(60)  # Ждем минуту перед повторной попыткой 