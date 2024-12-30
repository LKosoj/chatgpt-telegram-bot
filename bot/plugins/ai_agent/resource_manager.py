"""
Модуль для управления ресурсами системы
"""

import asyncio
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import aiohttp
from .agent_logger import AgentLogger
from .constants import ResourceType
from .monitoring import ProcessMonitor


@dataclass
class ResourceUsage:
    """Использование ресурсов"""
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    threads_count: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ResourceManager:
    """Менеджер ресурсов"""
    
    def __init__(
        self,
        logger: Optional[AgentLogger] = None,
        max_parallel_requests: int = 10,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        cleanup_interval: int = 3600
    ):
        """
        Инициализация менеджера ресурсов
        
        Args:
            logger: логгер (опционально)
            max_parallel_requests: максимальное количество параллельных запросов
            cpu_threshold: порог CPU в процентах
            memory_threshold: порог памяти в процентах
            cleanup_interval: интервал очистки в секундах
        """
        self.logger = logger or AgentLogger("logs")
        self.max_parallel_requests = max_parallel_requests
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.cleanup_interval = cleanup_interval
        
        # Семафор для ограничения параллельных запросов
        self.request_semaphore = asyncio.Semaphore(max_parallel_requests)
        
        # Монитор процессов
        self.process_monitor = ProcessMonitor(
            logger=self.logger,
            cpu_threshold=cpu_threshold,
            memory_threshold=memory_threshold
        )
        
        # Активные ресурсы
        self.active_resources: Dict[ResourceType, Set[str]] = defaultdict(set)
        
        # История использования ресурсов
        self.resource_history: List[ResourceUsage] = []
        
        # Задачи очистки
        self.cleanup_tasks: Set[asyncio.Task] = set()
        
    async def start(self):
        """Запуск менеджера ресурсов"""
        # Запускаем периодическую очистку
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self.cleanup_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self.cleanup_tasks.discard)
        
        self.logger.info("Resource manager started")
        
    async def stop(self):
        """Остановка менеджера ресурсов"""
        # Отменяем задачи очистки
        for task in self.cleanup_tasks:
            task.cancel()
            
        # Ждем завершения задач
        if self.cleanup_tasks:
            await asyncio.gather(*self.cleanup_tasks, return_exceptions=True)
            
        self.logger.info("Resource manager stopped")
        
    async def acquire_resource(
        self,
        resource_type: ResourceType,
        resource_id: str
    ) -> bool:
        """
        Получение ресурса
        
        Args:
            resource_type: тип ресурса
            resource_id: идентификатор ресурса
            
        Returns:
            bool: успешность получения
        """
        try:
            # Проверяем доступность ресурсов
            if not await self._check_resource_availability():
                self.logger.warning(
                    "Resource limits exceeded",
                    extra={
                        "resource_type": resource_type,
                        "resource_id": resource_id
                    }
                )
                return False
                
            # Добавляем ресурс в активные
            self.active_resources[resource_type].add(resource_id)
            
            # Обновляем метрики
            await self._update_resource_metrics()
            
            self.logger.info(
                f"Resource acquired: {resource_type.value}/{resource_id}"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "Error acquiring resource",
                error=str(e),
                extra={
                    "resource_type": resource_type,
                    "resource_id": resource_id
                }
            )
            return False
            
    async def release_resource(
        self,
        resource_type: ResourceType,
        resource_id: str
    ):
        """
        Освобождение ресурса
        
        Args:
            resource_type: тип ресурса
            resource_id: идентификатор ресурса
        """
        try:
            # Удаляем ресурс из активных
            if resource_id in self.active_resources[resource_type]:
                self.active_resources[resource_type].remove(resource_id)
                
            # Обновляем метрики
            await self._update_resource_metrics()
            
            self.logger.info(
                f"Resource released: {resource_type.value}/{resource_id}"
            )
            
        except Exception as e:
            self.logger.error(
                "Error releasing resource",
                error=str(e),
                extra={
                    "resource_type": resource_type,
                    "resource_id": resource_id
                }
            )
            
    async def get_resource_usage(self) -> ResourceUsage:
        """
        Получение использования ресурсов
        
        Returns:
            ResourceUsage: использование ресурсов
        """
        process = psutil.Process()
        
        return ResourceUsage(
            cpu_percent=process.cpu_percent(),
            memory_percent=process.memory_percent(),
            memory_rss=process.memory_info().rss,
            threads_count=process.num_threads()
        )
        
    async def get_active_resources(
        self,
        resource_type: Optional[ResourceType] = None
    ) -> Dict[ResourceType, Set[str]]:
        """
        Получение активных ресурсов
        
        Args:
            resource_type: тип ресурса (опционально)
            
        Returns:
            Dict[ResourceType, Set[str]]: активные ресурсы
        """
        if resource_type:
            return {resource_type: self.active_resources[resource_type]}
        return dict(self.active_resources)
        
    async def cleanup_resources(
        self,
        resource_type: Optional[ResourceType] = None
    ):
        """
        Очистка неиспользуемых ресурсов
        
        Args:
            resource_type: тип ресурса (опционально)
        """
        try:
            # Получаем список ресурсов для очистки
            resources = (
                [resource_type] if resource_type
                else list(ResourceType)
            )
            
            cleaned_count = 0
            for res_type in resources:
                # Проверяем каждый активный ресурс
                for res_id in list(self.active_resources[res_type]):
                    if await self._should_cleanup_resource(res_type, res_id):
                        await self.release_resource(res_type, res_id)
                        cleaned_count += 1
                        
            if cleaned_count > 0:
                self.logger.info(
                    f"Cleaned up {cleaned_count} resources"
                )
                
        except Exception as e:
            self.logger.error(
                "Error cleaning up resources",
                error=str(e)
            )
            
    async def _check_resource_availability(self) -> bool:
        """
        Проверка доступности ресурсов
        
        Returns:
            bool: доступность ресурсов
        """
        # Получаем текущие метрики
        usage = await self.get_resource_usage()
        
        # Проверяем CPU
        if usage.cpu_percent > self.cpu_threshold:
            return False
            
        # Проверяем память
        if usage.memory_percent > self.memory_threshold:
            return False
            
        # Проверяем количество параллельных запросов
        if not self.request_semaphore.locked():
            return False
            
        return True
        
    async def _update_resource_metrics(self):
        """Обновление метрик ресурсов"""
        usage = await self.get_resource_usage()
        self.resource_history.append(usage)
        
        # Оставляем только последние 1000 записей
        if len(self.resource_history) > 1000:
            self.resource_history = self.resource_history[-1000:]
            
    async def _should_cleanup_resource(
        self,
        resource_type: ResourceType,
        resource_id: str
    ) -> bool:
        """
        Проверка необходимости очистки ресурса
        
        Args:
            resource_type: тип ресурса
            resource_id: идентификатор ресурса
            
        Returns:
            bool: необходимость очистки
        """
        # TODO: Реализовать конкретные правила очистки
        # для каждого типа ресурса
        return False
        
    async def _periodic_cleanup(self):
        """Периодическая очистка ресурсов"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_resources()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in periodic cleanup",
                    error=str(e)
                )
                await asyncio.sleep(60)  # Ждем минуту перед повторной попыткой 