"""
Модуль для инструментов отладки
"""

import os
import sys
import time
import cProfile
import pstats
import tracemalloc
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from functools import wraps
from .logging import AgentLogger


@dataclass
class ProfileResult:
    """Результат профилирования"""
    function_name: str
    total_time: float
    calls_count: int
    time_per_call: float
    cumulative_time: float
    callers: Dict[str, int]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemorySnapshot:
    """Снимок памяти"""
    total_size: int
    peak_size: int
    allocated_blocks: int
    top_allocations: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)


def profile_function(func: Callable) -> Callable:
    """
    Декоратор для профилирования функции
    
    Args:
        func: функция для профилирования
        
    Returns:
        Callable: обернутая функция
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            return profiler.runcall(func, *args, **kwargs)
        finally:
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats()
            
    return wrapper


def async_profile_function(func: Callable) -> Callable:
    """
    Декоратор для профилирования асинхронной функции
    
    Args:
        func: функция для профилирования
        
    Returns:
        Callable: обернутая функция
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            return await profiler.runcall(func, *args, **kwargs)
        finally:
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats()
            
    return wrapper


class DebugTools:
    """Инструменты отладки"""
    
    def __init__(
        self,
        debug_dir: str = "debug",
        logger: Optional[AgentLogger] = None,
        enable_memory_tracking: bool = True,
        enable_profiling: bool = True
    ):
        """
        Инициализация инструментов
        
        Args:
            debug_dir: директория для отладочных данных
            logger: логгер (опционально)
            enable_memory_tracking: включить отслеживание памяти
            enable_profiling: включить профилирование
        """
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or AgentLogger("debug_logs")
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_profiling = enable_profiling
        
        # Результаты профилирования
        self.profile_results: Dict[str, List[ProfileResult]] = {}
        
        # Снимки памяти
        self.memory_snapshots: List[MemorySnapshot] = []
        
        # Инициализация отслеживания памяти
        if enable_memory_tracking:
            tracemalloc.start()
            
    def start_profiling(self, name: str):
        """
        Начало профилирования
        
        Args:
            name: название профиля
        """
        if not self.enable_profiling:
            return
            
        self.profile_results[name] = []
        profiler = cProfile.Profile()
        profiler.enable()
        return profiler
        
    def stop_profiling(self, profiler: cProfile.Profile, name: str):
        """
        Остановка профилирования
        
        Args:
            profiler: профайлер
            name: название профиля
        """
        if not self.enable_profiling:
            return
            
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Сохраняем результаты
        for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
            result = ProfileResult(
                function_name=func_name,
                total_time=tt,
                calls_count=cc,
                time_per_call=tt/cc if cc > 0 else 0,
                cumulative_time=ct,
                callers={str(caller): count for caller, count in callers.items()}
            )
            self.profile_results[name].append(result)
            
        # Сохраняем в файл
        stats_file = self.debug_dir / f"{name}_profile.stats"
        stats.dump_stats(str(stats_file))
        
    def take_memory_snapshot(self) -> MemorySnapshot:
        """
        Создание снимка памяти
        
        Returns:
            MemorySnapshot: снимок памяти
        """
        if not self.enable_memory_tracking:
            return None
            
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        result = MemorySnapshot(
            total_size=sum(stat.size for stat in top_stats),
            peak_size=tracemalloc.get_traced_memory()[1],
            allocated_blocks=len(top_stats),
            top_allocations=[
                {
                    "file": str(stat.traceback[0].filename),
                    "line": stat.traceback[0].lineno,
                    "size": stat.size,
                    "count": stat.count
                }
                for stat in top_stats[:10]
            ]
        )
        
        self.memory_snapshots.append(result)
        return result
        
    def compare_memory_snapshots(
        self,
        snapshot1: MemorySnapshot,
        snapshot2: MemorySnapshot
    ) -> Dict[str, Any]:
        """
        Сравнение снимков памяти
        
        Args:
            snapshot1: первый снимок
            snapshot2: второй снимок
            
        Returns:
            Dict[str, Any]: результат сравнения
        """
        return {
            "total_size_diff": snapshot2.total_size - snapshot1.total_size,
            "peak_size_diff": snapshot2.peak_size - snapshot1.peak_size,
            "blocks_diff": snapshot2.allocated_blocks - snapshot1.allocated_blocks,
            "time_between": (snapshot2.timestamp - snapshot1.timestamp).total_seconds()
        }
        
    def get_profile_results(
        self,
        name: Optional[str] = None
    ) -> Union[List[ProfileResult], Dict[str, List[ProfileResult]]]:
        """
        Получение результатов профилирования
        
        Args:
            name: название профиля (опционально)
            
        Returns:
            Union[List[ProfileResult], Dict[str, List[ProfileResult]]]: результаты
        """
        if name:
            return self.profile_results.get(name, [])
        return self.profile_results
        
    def get_memory_snapshots(
        self,
        limit: Optional[int] = None
    ) -> List[MemorySnapshot]:
        """
        Получение снимков памяти
        
        Args:
            limit: ограничение количества (опционально)
            
        Returns:
            List[MemorySnapshot]: снимки памяти
        """
        if limit:
            return self.memory_snapshots[-limit:]
        return self.memory_snapshots
        
    def save_debug_info(self, filename: str):
        """
        Сохранение отладочной информации
        
        Args:
            filename: имя файла
        """
        try:
            data = {
                "profile_results": {
                    name: [
                        {
                            "function_name": r.function_name,
                            "total_time": r.total_time,
                            "calls_count": r.calls_count,
                            "time_per_call": r.time_per_call,
                            "cumulative_time": r.cumulative_time,
                            "callers": r.callers,
                            "timestamp": r.timestamp.isoformat()
                        }
                        for r in results
                    ]
                    for name, results in self.profile_results.items()
                },
                "memory_snapshots": [
                    {
                        "total_size": s.total_size,
                        "peak_size": s.peak_size,
                        "allocated_blocks": s.allocated_blocks,
                        "top_allocations": s.top_allocations,
                        "timestamp": s.timestamp.isoformat()
                    }
                    for s in self.memory_snapshots
                ]
            }
            
            debug_file = self.debug_dir / filename
            with open(debug_file, "w") as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved debug info to {debug_file}")
            
        except Exception as e:
            self.logger.error(
                "Error saving debug info",
                error=str(e),
                extra={"filename": filename}
            )
            
    def cleanup(self):
        """Очистка ресурсов"""
        if self.enable_memory_tracking:
            tracemalloc.stop()
            
        self.profile_results.clear()
        self.memory_snapshots.clear() 