"""
Тесты для инструментов отладки
"""

import pytest
import json
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from ..debug_tools import (
    DebugTools,
    ProfileResult,
    MemorySnapshot,
    profile_function,
    async_profile_function
)
from ..logging import AgentLogger


@pytest.fixture
def logger():
    """Фикстура для логгера"""
    return AgentLogger("test_logs")


@pytest.fixture
def debug_dir(tmp_path):
    """Фикстура для директории отладки"""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    return debug_dir


@pytest.fixture
def debug_tools(debug_dir, logger):
    """Фикстура для инструментов отладки"""
    return DebugTools(str(debug_dir), logger)


@pytest.fixture
def profile_result():
    """Фикстура для результата профилирования"""
    return ProfileResult(
        function_name="test_func",
        total_time=1.5,
        calls_count=10,
        time_per_call=0.15,
        cumulative_time=2.0,
        callers={"caller1": 5, "caller2": 3}
    )


@pytest.fixture
def memory_snapshot():
    """Фикстура для снимка памяти"""
    return MemorySnapshot(
        total_size=1024 * 1024,
        peak_size=2048 * 1024,
        allocated_blocks=100,
        top_allocations=[
            {
                "file": "test.py",
                "line": 10,
                "size": 1024,
                "count": 1
            }
        ]
    )


def test_debug_tools_initialization(debug_dir, logger):
    """Тест инициализации инструментов"""
    tools = DebugTools(str(debug_dir), logger)
    
    assert tools.debug_dir == debug_dir
    assert tools.logger == logger
    assert isinstance(tools.profile_results, dict)
    assert isinstance(tools.memory_snapshots, list)


def test_profile_function_decorator():
    """Тест декоратора профилирования"""
    @profile_function
    def test_func():
        time.sleep(0.1)
        return 42
        
    with patch('pstats.Stats') as mock_stats:
        result = test_func()
        
        assert result == 42
        assert mock_stats.called


@pytest.mark.asyncio
async def test_async_profile_function_decorator():
    """Тест декоратора профилирования для асинхронных функций"""
    @async_profile_function
    async def test_func():
        await asyncio.sleep(0.1)
        return 42
        
    with patch('pstats.Stats') as mock_stats:
        result = await test_func()
        
        assert result == 42
        assert mock_stats.called


def test_profiling(debug_tools):
    """Тест профилирования"""
    # Начинаем профилирование
    profiler = debug_tools.start_profiling("test_profile")
    
    # Выполняем тестовую функцию
    time.sleep(0.1)
    
    # Останавливаем профилирование
    debug_tools.stop_profiling(profiler, "test_profile")
    
    # Проверяем результаты
    results = debug_tools.get_profile_results("test_profile")
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Проверяем файл статистики
    stats_file = debug_tools.debug_dir / "test_profile_profile.stats"
    assert stats_file.exists()


def test_memory_tracking(debug_tools):
    """Тест отслеживания памяти"""
    # Создаем снимок памяти
    snapshot = debug_tools.take_memory_snapshot()
    
    assert isinstance(snapshot, MemorySnapshot)
    assert snapshot.total_size > 0
    assert snapshot.peak_size > 0
    assert snapshot.allocated_blocks > 0
    assert isinstance(snapshot.top_allocations, list)


def test_compare_memory_snapshots(debug_tools, memory_snapshot):
    """Тест сравнения снимков памяти"""
    snapshot1 = memory_snapshot
    snapshot2 = MemorySnapshot(
        total_size=2048 * 1024,
        peak_size=4096 * 1024,
        allocated_blocks=200,
        top_allocations=[],
        timestamp=snapshot1.timestamp + timedelta(seconds=10)
    )
    
    comparison = debug_tools.compare_memory_snapshots(snapshot1, snapshot2)
    
    assert comparison["total_size_diff"] == 1024 * 1024
    assert comparison["peak_size_diff"] == 2048 * 1024
    assert comparison["blocks_diff"] == 100
    assert comparison["time_between"] == 10.0


def test_get_profile_results(debug_tools, profile_result):
    """Тест получения результатов профилирования"""
    # Добавляем тестовые результаты
    debug_tools.profile_results["test_profile"] = [profile_result]
    
    # Получаем результаты конкретного профиля
    results = debug_tools.get_profile_results("test_profile")
    assert len(results) == 1
    assert results[0] == profile_result
    
    # Получаем все результаты
    all_results = debug_tools.get_profile_results()
    assert isinstance(all_results, dict)
    assert "test_profile" in all_results


def test_get_memory_snapshots(debug_tools, memory_snapshot):
    """Тест получения снимков памяти"""
    # Добавляем тестовые снимки
    debug_tools.memory_snapshots = [memory_snapshot] * 3
    
    # Получаем все снимки
    snapshots = debug_tools.get_memory_snapshots()
    assert len(snapshots) == 3
    
    # Получаем ограниченное количество снимков
    limited = debug_tools.get_memory_snapshots(limit=2)
    assert len(limited) == 2


def test_save_debug_info(debug_tools, profile_result, memory_snapshot):
    """Тест сохранения отладочной информации"""
    # Добавляем тестовые данные
    debug_tools.profile_results["test_profile"] = [profile_result]
    debug_tools.memory_snapshots = [memory_snapshot]
    
    # Сохраняем информацию
    debug_tools.save_debug_info("debug_info.json")
    
    # Проверяем файл
    debug_file = debug_tools.debug_dir / "debug_info.json"
    assert debug_file.exists()
    
    # Проверяем содержимое
    with open(debug_file) as f:
        data = json.load(f)
        
    assert "profile_results" in data
    assert "memory_snapshots" in data
    assert len(data["profile_results"]["test_profile"]) == 1
    assert len(data["memory_snapshots"]) == 1


def test_cleanup(debug_tools, profile_result, memory_snapshot):
    """Тест очистки ресурсов"""
    # Добавляем тестовые данные
    debug_tools.profile_results["test_profile"] = [profile_result]
    debug_tools.memory_snapshots = [memory_snapshot]
    
    # Очищаем ресурсы
    debug_tools.cleanup()
    
    assert len(debug_tools.profile_results) == 0
    assert len(debug_tools.memory_snapshots) == 0 