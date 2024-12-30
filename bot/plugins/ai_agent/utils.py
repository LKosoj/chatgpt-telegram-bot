"""
Модуль с утилитами
"""

import os
import json
import asyncio
import hashlib
import aiohttp
import aiofiles
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from .logging import AgentLogger


class Cache:
    """Кэш для данных"""
    
    def __init__(
        self,
        cache_dir: str,
        max_size: int = 1024 * 1024 * 100,  # 100MB
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация кэша
        
        Args:
            cache_dir: директория для кэша
            max_size: максимальный размер кэша в байтах
            logger: логгер (опционально)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.logger = logger or AgentLogger("logs")
        
    def _get_cache_path(self, key: str) -> Path:
        """
        Получение пути к файлу кэша
        
        Args:
            key: ключ
            
        Returns:
            Path: путь к файлу
        """
        # Создаем хэш ключа
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
        
    async def get(self, key: str) -> Optional[Any]:
        """
        Получение данных из кэша
        
        Args:
            key: ключ
            
        Returns:
            Optional[Any]: данные
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        try:
            async with aiofiles.open(cache_path, "r") as f:
                data = json.loads(await f.read())
                
            # Проверяем срок действия
            if data.get("expires_at"):
                expires_at = datetime.fromisoformat(data["expires_at"])
                if expires_at < datetime.now():
                    await self.delete(key)
                    return None
                    
            return data["value"]
            
        except Exception as e:
            self.logger.error(
                "Error reading from cache",
                error=e,
                extra={"key": key}
            )
            return None
            
    async def set(
        self,
        key: str,
        value: Any,
        expires_in: Optional[int] = None
    ):
        """
        Сохранение данных в кэш
        
        Args:
            key: ключ
            value: данные
            expires_in: время жизни в секундах
        """
        try:
            # Проверяем размер кэша
            await self._cleanup_if_needed()
            
            cache_path = self._get_cache_path(key)
            
            data = {
                "key": key,
                "value": value,
                "created_at": datetime.now().isoformat()
            }
            
            if expires_in:
                data["expires_at"] = (
                    datetime.now().timestamp() + expires_in
                ).isoformat()
                
            async with aiofiles.open(cache_path, "w") as f:
                await f.write(json.dumps(data, ensure_ascii=False))
                
        except Exception as e:
            self.logger.error(
                "Error writing to cache",
                error=e,
                extra={"key": key}
            )
            
    async def delete(self, key: str):
        """
        Удаление данных из кэша
        
        Args:
            key: ключ
        """
        try:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                os.remove(cache_path)
                
        except Exception as e:
            self.logger.error(
                "Error deleting from cache",
                error=e,
                extra={"key": key}
            )
            
    async def clear(self):
        """Очистка кэша"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                os.remove(cache_file)
                
        except Exception as e:
            self.logger.error(
                "Error clearing cache",
                error=e
            )
            
    async def _cleanup_if_needed(self):
        """Очистка кэша при превышении размера"""
        try:
            total_size = sum(
                f.stat().st_size
                for f in self.cache_dir.glob("*.json")
            )
            
            if total_size > self.max_size:
                # Удаляем старые файлы
                cache_files = sorted(
                    self.cache_dir.glob("*.json"),
                    key=lambda x: x.stat().st_mtime
                )
                
                for f in cache_files:
                    os.remove(f)
                    total_size -= f.stat().st_size
                    if total_size <= self.max_size:
                        break
                        
        except Exception as e:
            self.logger.error(
                "Error cleaning up cache",
                error=e
            )


class RateLimiter:
    """Ограничитель запросов"""
    
    def __init__(
        self,
        calls: int,
        period: int,
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация ограничителя
        
        Args:
            calls: количество запросов
            period: период в секундах
            logger: логгер (опционально)
        """
        self.calls = calls
        self.period = period
        self.logger = logger or AgentLogger("logs")
        self.timestamps: List[float] = []
        
    async def acquire(self):
        """Получение разрешения на запрос"""
        now = datetime.now().timestamp()
        
        # Удаляем старые метки
        self.timestamps = [
            ts for ts in self.timestamps
            if now - ts <= self.period
        ]
        
        if len(self.timestamps) >= self.calls:
            # Вычисляем время ожидания
            wait_time = self.period - (now - self.timestamps[0])
            if wait_time > 0:
                self.logger.info(
                    f"Rate limit exceeded, waiting {wait_time:.2f}s"
                )
                await asyncio.sleep(wait_time)
                
        self.timestamps.append(now)


class APIClient:
    """HTTP клиент с поддержкой кэширования и rate limiting"""
    
    def __init__(
        self,
        cache_dir: str,
        rate_limit_calls: int = 60,
        rate_limit_period: int = 60,
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация клиента
        
        Args:
            cache_dir: директория для кэша
            rate_limit_calls: количество запросов
            rate_limit_period: период в секундах
            logger: логгер (опционально)
        """
        self.cache = Cache(cache_dir, logger=logger)
        self.rate_limiter = RateLimiter(
            rate_limit_calls,
            rate_limit_period,
            logger=logger
        )
        self.logger = logger or AgentLogger("logs")
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Получение сессии
        
        Returns:
            aiohttp.ClientSession: сессия
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
        
    async def close(self):
        """Закрытие клиента"""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        GET запрос
        
        Args:
            url: URL
            params: параметры запроса
            headers: заголовки
            cache_ttl: время жизни кэша
            
        Returns:
            Dict[str, Any]: ответ
        """
        # Формируем ключ кэша
        cache_key = f"get:{url}:{json.dumps(params or {})}"
        
        # Проверяем кэш
        if cache_ttl:
            cached = await self.cache.get(cache_key)
            if cached:
                return cached
                
        # Получаем разрешение
        await self.rate_limiter.acquire()
        
        try:
            session = await self._get_session()
            async with session.get(
                url,
                params=params,
                headers=headers
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Сохраняем в кэш
                if cache_ttl:
                    await self.cache.set(
                        cache_key,
                        data,
                        cache_ttl
                    )
                    
                return data
                
        except Exception as e:
            self.logger.error(
                "Error making GET request",
                error=e,
                extra={"url": url}
            )
            raise
            
    async def post(
        self,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        POST запрос
        
        Args:
            url: URL
            json_data: данные
            headers: заголовки
            
        Returns:
            Dict[str, Any]: ответ
        """
        # Получаем разрешение
        await self.rate_limiter.acquire()
        
        try:
            session = await self._get_session()
            async with session.post(
                url,
                json=json_data,
                headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()
                
        except Exception as e:
            self.logger.error(
                "Error making POST request",
                error=e,
                extra={"url": url}
            )
            raise


class FileManager:
    """Менеджер файлов"""
    
    def __init__(
        self,
        base_dir: str,
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация менеджера
        
        Args:
            base_dir: базовая директория
            logger: логгер (опционально)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or AgentLogger("logs")
        
    async def save_json(
        self,
        data: Any,
        filename: str,
        subdir: Optional[str] = None
    ):
        """
        Сохранение JSON файла
        
        Args:
            data: данные
            filename: имя файла
            subdir: поддиректория
        """
        try:
            # Создаем директорию
            save_dir = self.base_dir
            if subdir:
                save_dir = save_dir / subdir
                save_dir.mkdir(parents=True, exist_ok=True)
                
            # Добавляем расширение
            if not filename.endswith(".json"):
                filename += ".json"
                
            file_path = save_dir / filename
            
            async with aiofiles.open(file_path, "w") as f:
                await f.write(
                    json.dumps(data, ensure_ascii=False, indent=2)
                )
                
            self.logger.info(
                f"Saved JSON file: {file_path}"
            )
            
        except Exception as e:
            self.logger.error(
                "Error saving JSON file",
                error=e,
                extra={"filename": filename}
            )
            raise
            
    async def load_json(
        self,
        filename: str,
        subdir: Optional[str] = None
    ) -> Optional[Any]:
        """
        Загрузка JSON файла
        
        Args:
            filename: имя файла
            subdir: поддиректория
            
        Returns:
            Optional[Any]: данные
        """
        try:
            # Формируем путь
            file_path = self.base_dir
            if subdir:
                file_path = file_path / subdir
                
            # Добавляем расширение
            if not filename.endswith(".json"):
                filename += ".json"
                
            file_path = file_path / filename
            
            if not file_path.exists():
                return None
                
            async with aiofiles.open(file_path, "r") as f:
                return json.loads(await f.read())
                
        except Exception as e:
            self.logger.error(
                "Error loading JSON file",
                error=e,
                extra={"filename": filename}
            )
            return None
            
    async def list_files(
        self,
        pattern: str = "*",
        subdir: Optional[str] = None
    ) -> List[str]:
        """
        Список файлов
        
        Args:
            pattern: шаблон имени
            subdir: поддиректория
            
        Returns:
            List[str]: список файлов
        """
        try:
            # Формируем путь
            search_dir = self.base_dir
            if subdir:
                search_dir = search_dir / subdir
                
            return [
                str(f.relative_to(self.base_dir))
                for f in search_dir.glob(pattern)
                if f.is_file()
            ]
            
        except Exception as e:
            self.logger.error(
                "Error listing files",
                error=e,
                extra={"pattern": pattern}
            )
            return []
            
    async def delete_file(
        self,
        filename: str,
        subdir: Optional[str] = None
    ) -> bool:
        """
        Удаление файла
        
        Args:
            filename: имя файла
            subdir: поддиректория
            
        Returns:
            bool: успешность удаления
        """
        try:
            # Формируем путь
            file_path = self.base_dir
            if subdir:
                file_path = file_path / subdir
                
            file_path = file_path / filename
            
            if file_path.exists():
                os.remove(file_path)
                self.logger.info(
                    f"Deleted file: {file_path}"
                )
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(
                "Error deleting file",
                error=e,
                extra={"filename": filename}
            )
            return False 