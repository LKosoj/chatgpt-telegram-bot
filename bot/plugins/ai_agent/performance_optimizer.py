"""
Модуль для оптимизации производительности AI агента
"""

import asyncio
import time
import gc
import psutil
import dataclasses
from typing import Any, Dict, List, Callable, Coroutine, Optional
from functools import wraps

from .monitoring import ProcessMonitor, MetricsCollector
from .logging import AgentLogger


@dataclasses.dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    execution_time: float = 0.0
    memory_usage: int = 0
    cpu_usage: float = 0.0
    query_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclasses.dataclass
class OptimizationResult:
    """Результат оптимизации"""
    original_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    improvements: Dict[str, float]
    recommendations: List[str]


class PerformanceOptimizer:
    """Класс для оптимизации производительности"""

    def __init__(self, process_monitor: ProcessMonitor, metrics_collector: MetricsCollector, logger: AgentLogger):
        """Инициализация оптимизатора производительности"""
        self.process_monitor = process_monitor
        self.metrics_collector = metrics_collector
        self.logger = logger
        
        # Пороговые значения для метрик
        self.thresholds = {
            "execution_time": 1.0,  # секунды
            "memory_usage": 100 * 1024 * 1024,  # 100 MB
            "cpu_usage": 80.0,  # проценты
            "cache_miss_rate": 0.3  # процент промахов кэша
        }

    async def optimize_query(self, query_func: Callable[..., Coroutine[Any, Any, Any]]) -> OptimizationResult:
        """Оптимизация запроса"""
        # Собираем исходные метрики
        original_metrics = await self._collect_metrics(query_func)
        
        # Анализируем узкие места
        bottlenecks = self._analyze_bottlenecks(original_metrics)
        
        if not bottlenecks:
            return OptimizationResult(
                original_metrics=original_metrics,
                optimized_metrics=original_metrics,
                improvements={},
                recommendations=["Оптимизация не требуется - все метрики в пределах нормы"]
            )
        
        # Оптимизируем функцию
        optimized_func = self._optimize_function(query_func, bottlenecks)
        
        # Собираем метрики после оптимизации
        optimized_metrics = await self._collect_metrics(optimized_func)
        
        # Вычисляем улучшения
        improvements = self._calculate_improvements(original_metrics, optimized_metrics)
        
        # Генерируем рекомендации
        recommendations = self._generate_recommendations(bottlenecks, improvements)
        
        return OptimizationResult(
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            improvements=improvements,
            recommendations=recommendations
        )

    async def profile_bottlenecks(self, func: Callable[..., Coroutine[Any, Any, Any]]) -> Dict[str, Any]:
        """Профилирование узких мест"""
        start_time = time.time()
        result = await func()
        total_time = time.time() - start_time
        
        # Собираем информацию о вызовах
        calls_info = {
            "function_name": func.__name__,
            "args_count": func.__code__.co_argcount,
            "locals_count": func.__code__.co_nlocals,
            "stack_size": func.__code__.co_stacksize
        }
        
        return {
            "result": result,
            "calls_info": calls_info,
            "total_time": total_time
        }

    async def optimize_memory(self) -> Dict[str, Any]:
        """Оптимизация использования памяти"""
        # Запускаем сборщик мусора
        gc.collect()
        
        # Получаем информацию о памяти
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Находим утечки памяти
        objects = gc.get_objects()
        top_leaks = []
        
        for obj in objects[:100]:  # Анализируем первые 100 объектов
            size = sys.getsizeof(obj)
            if size > 1024 * 1024:  # Объекты больше 1MB
                top_leaks.append({
                    "type": type(obj).__name__,
                    "size": size,
                    "id": id(obj)
                })
        
        return {
            "total_allocated": memory_info.rss,
            "top_leaks": sorted(top_leaks, key=lambda x: x["size"], reverse=True),
            "objects_collected": gc.get_count()
        }

    async def optimize_cpu(self) -> Dict[str, Any]:
        """Оптимизация использования CPU"""
        # Получаем информацию о CPU
        cpu_times = psutil.cpu_times()
        
        # Получаем топ процессов по использованию CPU
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Сортируем по использованию CPU
        top_processes = sorted(processes, key=lambda x: x["cpu_percent"], reverse=True)[:10]
        
        return {
            "cpu_times": dataclasses.asdict(cpu_times),
            "top_processes": top_processes,
            "total_cpu": psutil.cpu_percent()
        }

    async def collect_performance_metrics(self) -> Dict[str, Any]:
        """Сбор метрик производительности"""
        # Системные метрики
        system_metrics = await self.metrics_collector.collect_system_metrics()
        
        # Метрики процесса
        process_metrics = await self.process_monitor.get_process_metrics()
        
        # Метрики памяти
        memory = psutil.virtual_memory()
        
        return {
            "system": system_metrics,
            "processes": process_metrics,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "timestamp": time.time()
        }

    async def _collect_metrics(self, func: Callable[..., Coroutine[Any, Any, Any]]) -> PerformanceMetrics:
        """Сбор метрик для функции"""
        # Замеряем время выполнения
        start_time = time.time()
        await func()
        execution_time = time.time() - start_time
        
        # Получаем метрики процесса
        process = psutil.Process()
        memory_usage = process.memory_info().rss
        cpu_usage = process.cpu_percent()
        
        # Получаем метрики кэша
        cache_metrics = await self.metrics_collector.get_cache_metrics()
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            query_count=cache_metrics["query_count"],
            cache_hits=cache_metrics["hits"],
            cache_misses=cache_metrics["misses"]
        )

    def _analyze_bottlenecks(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """Анализ узких мест"""
        bottlenecks = []
        
        # Проверяем время выполнения
        if metrics.execution_time > self.thresholds["execution_time"]:
            bottlenecks.append({
                "type": "execution_time",
                "value": metrics.execution_time,
                "threshold": self.thresholds["execution_time"],
                "severity": "high" if metrics.execution_time > self.thresholds["execution_time"] * 2 else "medium"
            })
        
        # Проверяем использование памяти
        if metrics.memory_usage > self.thresholds["memory_usage"]:
            bottlenecks.append({
                "type": "memory_usage",
                "value": metrics.memory_usage,
                "threshold": self.thresholds["memory_usage"],
                "severity": "high" if metrics.memory_usage > self.thresholds["memory_usage"] * 2 else "medium"
            })
        
        # Проверяем использование CPU
        if metrics.cpu_usage > self.thresholds["cpu_usage"]:
            bottlenecks.append({
                "type": "cpu_usage",
                "value": metrics.cpu_usage,
                "threshold": self.thresholds["cpu_usage"],
                "severity": "high" if metrics.cpu_usage > self.thresholds["cpu_usage"] * 1.5 else "medium"
            })
        
        # Проверяем эффективность кэша
        if metrics.query_count > 0:
            cache_miss_rate = metrics.cache_misses / metrics.query_count
            if cache_miss_rate > self.thresholds["cache_miss_rate"]:
                bottlenecks.append({
                    "type": "cache_efficiency",
                    "value": cache_miss_rate,
                    "threshold": self.thresholds["cache_miss_rate"],
                    "severity": "medium"
                })
        
        return bottlenecks

    def _optimize_function(self, func: Callable[..., Coroutine[Any, Any, Any]], bottlenecks: List[Dict[str, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
        """Оптимизация функции на основе найденных узких мест"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Применяем оптимизации на основе узких мест
            for bottleneck in bottlenecks:
                if bottleneck["type"] == "execution_time":
                    # Добавляем кэширование результатов
                    cache_key = f"{func.__name__}:{args}:{kwargs}"
                    cached_result = await self.metrics_collector.get_cache_value(cache_key)
                    if cached_result is not None:
                        return cached_result
                
                elif bottleneck["type"] == "memory_usage":
                    # Запускаем сборщик мусора перед выполнением
                    gc.collect()
                
                elif bottleneck["type"] == "cpu_usage":
                    # Добавляем небольшую задержку для снижения нагрузки
                    await asyncio.sleep(0.1)
            
            # Выполняем функцию
            result = await func(*args, **kwargs)
            
            # Кэшируем результат если есть проблемы со временем выполнения
            if any(b["type"] == "execution_time" for b in bottlenecks):
                cache_key = f"{func.__name__}:{args}:{kwargs}"
                await self.metrics_collector.set_cache_value(cache_key, result)
            
            return result
        
        return wrapper

    def _calculate_improvements(self, original: PerformanceMetrics, optimized: PerformanceMetrics) -> Dict[str, float]:
        """Вычисление улучшений производительности"""
        improvements = {}
        
        # Улучшение времени выполнения
        if original.execution_time > 0:
            improvements["execution_time"] = (
                (original.execution_time - optimized.execution_time) / original.execution_time * 100
            )
        
        # Улучшение использования памяти
        if original.memory_usage > 0:
            improvements["memory_usage"] = (
                (original.memory_usage - optimized.memory_usage) / original.memory_usage * 100
            )
        
        # Улучшение использования CPU
        if original.cpu_usage > 0:
            improvements["cpu_usage"] = (
                (original.cpu_usage - optimized.cpu_usage) / original.cpu_usage * 100
            )
        
        # Улучшение эффективности кэша
        if original.query_count > 0 and optimized.query_count > 0:
            original_hit_rate = original.cache_hits / original.query_count
            optimized_hit_rate = optimized.cache_hits / optimized.query_count
            improvements["cache_efficiency"] = (optimized_hit_rate - original_hit_rate) * 100
        
        return improvements

    def _generate_recommendations(self, bottlenecks: List[Dict[str, Any]], improvements: Dict[str, float]) -> List[str]:
        """Генерация рекомендаций по оптимизации"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "execution_time":
                if improvements.get("execution_time", 0) > 0:
                    recommendations.append(
                        f"Время выполнения улучшено на {improvements['execution_time']:.1f}% "
                        f"благодаря кэшированию результатов"
                    )
                else:
                    recommendations.append(
                        "Рекомендуется оптимизировать алгоритм или добавить кэширование "
                        "для улучшения времени выполнения"
                    )
            
            elif bottleneck["type"] == "memory_usage":
                if improvements.get("memory_usage", 0) > 0:
                    recommendations.append(
                        f"Использование памяти улучшено на {improvements['memory_usage']:.1f}% "
                        f"благодаря оптимизации управления памятью"
                    )
                else:
                    recommendations.append(
                        "Рекомендуется оптимизировать использование памяти: "
                        "освобождать неиспользуемые ресурсы и избегать утечек памяти"
                    )
            
            elif bottleneck["type"] == "cpu_usage":
                if improvements.get("cpu_usage", 0) > 0:
                    recommendations.append(
                        f"Использование CPU улучшено на {improvements['cpu_usage']:.1f}% "
                        f"благодаря оптимизации вычислений"
                    )
                else:
                    recommendations.append(
                        "Рекомендуется оптимизировать вычисления или распределить "
                        "нагрузку для снижения использования CPU"
                    )
            
            elif bottleneck["type"] == "cache_efficiency":
                if improvements.get("cache_efficiency", 0) > 0:
                    recommendations.append(
                        f"Эффективность кэша улучшена на {improvements['cache_efficiency']:.1f}% "
                        f"благодаря оптимизации стратегии кэширования"
                    )
                else:
                    recommendations.append(
                        "Рекомендуется пересмотреть стратегию кэширования для "
                        "улучшения эффективности кэша"
                    )
        
        # Добавляем рекомендации по метрикам, которые ухудшились
        for metric, improvement in improvements.items():
            if improvement < 0:
                recommendations.append(
                    f"Внимание: метрика {metric} ухудшилась на {abs(improvement):.1f}%. "
                    f"Требуется дополнительный анализ"
                )
        
        return recommendations 