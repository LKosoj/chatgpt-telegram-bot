"""
Тесты для системы оптимизации производительности
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from ..performance_optimizer import (
    PerformanceOptimizer,
    PerformanceMetrics,
    OptimizationResult
)
from ..monitoring import ProcessMonitor, MetricsCollector
from ..logging import AgentLogger


@pytest.fixture
def logger():
    """Фикстура для логгера"""
    return AgentLogger("test_logs")


@pytest.fixture
def process_monitor():
    """Фикстура для монитора процессов"""
    monitor = Mock(spec=ProcessMonitor)
    monitor.get_process_metrics = AsyncMock(return_value={
        "cpu_percent": 50,
        "memory_percent": 60,
        "threads": 10
    })
    return monitor


@pytest.fixture
def metrics_collector():
    """Фикстура для сборщика метрик"""
    collector = Mock(spec=MetricsCollector)
    collector.collect_system_metrics = AsyncMock(return_value={
        "cpu_usage": 50,
        "memory_usage": 60,
        "disk_usage": 70
    })
    collector.get_cache_metrics = AsyncMock(return_value={
        "hits": 80,
        "misses": 20,
        "query_count": 100
    })
    return collector


@pytest.fixture
def optimizer(process_monitor, metrics_collector, logger):
    """Фикстура для оптимизатора"""
    return PerformanceOptimizer(process_monitor, metrics_collector, logger)


async def test_slow_function():
    """Тестовая медленная функция"""
    await asyncio.sleep(0.1)
    return 42


async def test_memory_intensive_function():
    """Тестовая функция с интенсивным использованием памяти"""
    data = [i for i in range(1000000)]
    await asyncio.sleep(0.1)
    return len(data)


@pytest.mark.asyncio
async def test_optimize_query(optimizer):
    """Тест оптимизации запроса"""
    result = await optimizer.optimize_query(test_slow_function)
    
    assert isinstance(result, OptimizationResult)
    assert isinstance(result.original_metrics, PerformanceMetrics)
    assert isinstance(result.optimized_metrics, PerformanceMetrics)
    assert isinstance(result.improvements, dict)
    assert isinstance(result.recommendations, list)


@pytest.mark.asyncio
async def test_profile_bottlenecks(optimizer):
    """Тест профилирования узких мест"""
    result = await optimizer.profile_bottlenecks(test_slow_function)
    
    assert isinstance(result, dict)
    assert "result" in result
    assert "calls_info" in result
    assert "total_time" in result
    assert result["result"] == 42


@pytest.mark.asyncio
async def test_optimize_memory(optimizer):
    """Тест оптимизации памяти"""
    result = await optimizer.optimize_memory()
    
    assert isinstance(result, dict)
    assert "total_allocated" in result
    assert "top_leaks" in result
    assert "objects_collected" in result
    assert isinstance(result["top_leaks"], list)


@pytest.mark.asyncio
async def test_optimize_cpu(optimizer):
    """Тест оптимизации CPU"""
    result = await optimizer.optimize_cpu()
    
    assert isinstance(result, dict)
    assert "cpu_times" in result
    assert "top_processes" in result
    assert "total_cpu" in result
    assert isinstance(result["top_processes"], list)


@pytest.mark.asyncio
async def test_collect_performance_metrics(optimizer):
    """Тест сбора метрик производительности"""
    result = await optimizer.collect_performance_metrics()
    
    assert isinstance(result, dict)
    assert "system" in result
    assert "processes" in result
    assert "memory" in result
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_collect_metrics(optimizer):
    """Тест сбора метрик для функции"""
    metrics = await optimizer._collect_metrics(test_slow_function)
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.execution_time > 0
    assert metrics.memory_usage >= 0
    assert metrics.cpu_usage >= 0
    assert metrics.query_count >= 0
    assert metrics.cache_hits >= 0
    assert metrics.cache_misses >= 0


def test_analyze_bottlenecks(optimizer):
    """Тест анализа узких мест"""
    metrics = PerformanceMetrics(
        execution_time=2.0,
        memory_usage=200 * 1024 * 1024,
        cpu_usage=90,
        query_count=100,
        cache_hits=60,
        cache_misses=40
    )
    
    bottlenecks = optimizer._analyze_bottlenecks(metrics)
    
    assert isinstance(bottlenecks, list)
    assert len(bottlenecks) > 0
    
    for bottleneck in bottlenecks:
        assert "type" in bottleneck
        assert "value" in bottleneck
        assert "threshold" in bottleneck
        assert "severity" in bottleneck


def test_optimize_function(optimizer):
    """Тест оптимизации функции"""
    bottlenecks = [
        {
            "type": "execution_time",
            "value": 2.0,
            "threshold": 1.0,
            "severity": "high"
        },
        {
            "type": "memory_usage",
            "value": 200 * 1024 * 1024,
            "threshold": 100 * 1024 * 1024,
            "severity": "medium"
        }
    ]
    
    async def test_func():
        return 42
        
    optimized = optimizer._optimize_function(test_func, bottlenecks)
    
    assert asyncio.iscoroutinefunction(optimized)
    assert optimized.__name__ == "wrapper"


def test_calculate_improvements(optimizer):
    """Тест вычисления улучшений"""
    original = PerformanceMetrics(
        execution_time=2.0,
        memory_usage=200 * 1024 * 1024,
        cpu_usage=90,
        query_count=100,
        cache_hits=60,
        cache_misses=40
    )
    
    optimized = PerformanceMetrics(
        execution_time=1.0,
        memory_usage=100 * 1024 * 1024,
        cpu_usage=45,
        query_count=100,
        cache_hits=80,
        cache_misses=20
    )
    
    improvements = optimizer._calculate_improvements(original, optimized)
    
    assert isinstance(improvements, dict)
    assert improvements["execution_time"] == 50.0  # 50% улучшение
    assert improvements["memory_usage"] == 50.0  # 50% улучшение
    assert improvements["cpu_usage"] == 50.0  # 50% улучшение
    assert improvements["cache_efficiency"] > 0


def test_generate_recommendations(optimizer):
    """Тест генерации рекомендаций"""
    bottlenecks = [
        {
            "type": "execution_time",
            "value": 2.0,
            "threshold": 1.0,
            "severity": "high"
        },
        {
            "type": "memory_usage",
            "value": 200 * 1024 * 1024,
            "threshold": 100 * 1024 * 1024,
            "severity": "medium"
        }
    ]
    
    improvements = {
        "execution_time": 50.0,
        "memory_usage": -10.0,
        "cpu_usage": 30.0,
        "cache_efficiency": 20.0
    }
    
    recommendations = optimizer._generate_recommendations(bottlenecks, improvements)
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert any("кэширование" in r.lower() for r in recommendations)
    assert any("память" in r.lower() for r in recommendations)
    assert any("ухудшилась" in r.lower() for r in recommendations) 