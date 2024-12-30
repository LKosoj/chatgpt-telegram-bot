"""
Тесты для мониторинга
"""

import pytest
import os
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from ..monitoring import ProcessMonitor, MetricsCollector
from ..models import AgentRole, TaskStatus
from ..logging import AgentLogger


@pytest.fixture
def logger():
    """Фикстура для логгера"""
    return AgentLogger("test_logs")


@pytest.fixture
async def process_monitor(logger):
    """Фикстура для монитора процессов"""
    monitor = ProcessMonitor(
        logger=logger,
        cpu_threshold=80,
        memory_threshold=80,
        thread_threshold=100
    )
    yield monitor


@pytest.fixture
async def metrics_collector(logger, tmp_path):
    """Фикстура для сборщика метрик"""
    db_path = str(tmp_path / "metrics.db")
    collector = MetricsCollector(db_path, logger)
    await collector.initialize()
    yield collector


@pytest.mark.asyncio
async def test_process_monitor_initialization(logger):
    """Тест инициализации монитора процессов"""
    monitor = ProcessMonitor(
        logger=logger,
        cpu_threshold=80,
        memory_threshold=80,
        thread_threshold=100
    )
    
    assert monitor.cpu_threshold == 80
    assert monitor.memory_threshold == 80
    assert monitor.thread_threshold == 100


@pytest.mark.asyncio
async def test_cpu_monitoring(process_monitor):
    """Тест мониторинга CPU"""
    # Мокаем psutil
    with patch("psutil.cpu_percent") as mock_cpu:
        # Нормальная нагрузка
        mock_cpu.return_value = 50
        metrics = await process_monitor.get_cpu_metrics()
        assert metrics["usage"] == 50
        assert not metrics["alert"]
        
        # Высокая нагрузка
        mock_cpu.return_value = 90
        metrics = await process_monitor.get_cpu_metrics()
        assert metrics["usage"] == 90
        assert metrics["alert"]


@pytest.mark.asyncio
async def test_memory_monitoring(process_monitor):
    """Тест мониторинга памяти"""
    # Мокаем psutil
    with patch("psutil.virtual_memory") as mock_memory:
        # Нормальное использование
        mock_memory.return_value = Mock(
            percent=50,
            total=16000000000,
            available=8000000000
        )
        metrics = await process_monitor.get_memory_metrics()
        assert metrics["usage"] == 50
        assert not metrics["alert"]
        
        # Высокое использование
        mock_memory.return_value = Mock(
            percent=90,
            total=16000000000,
            available=1600000000
        )
        metrics = await process_monitor.get_memory_metrics()
        assert metrics["usage"] == 90
        assert metrics["alert"]


@pytest.mark.asyncio
async def test_thread_monitoring(process_monitor):
    """Тест мониторинга потоков"""
    # Мокаем psutil
    with patch("psutil.Process") as mock_process:
        mock_process.return_value.num_threads.return_value = 50
        
        metrics = await process_monitor.get_thread_metrics()
        assert metrics["count"] == 50
        assert not metrics["alert"]
        
        # Много потоков
        mock_process.return_value.num_threads.return_value = 150
        metrics = await process_monitor.get_thread_metrics()
        assert metrics["count"] == 150
        assert metrics["alert"]


@pytest.mark.asyncio
async def test_system_metrics(process_monitor):
    """Тест системных метрик"""
    with patch("psutil.cpu_percent") as mock_cpu, \
         patch("psutil.virtual_memory") as mock_memory, \
         patch("psutil.Process") as mock_process:
        
        mock_cpu.return_value = 50
        mock_memory.return_value = Mock(
            percent=60,
            total=16000000000,
            available=6400000000
        )
        mock_process.return_value.num_threads.return_value = 70
        
        metrics = await process_monitor.get_system_metrics()
        
        assert "cpu" in metrics
        assert "memory" in metrics
        assert "threads" in metrics
        assert not any(m["alert"] for m in metrics.values())


@pytest.mark.asyncio
async def test_metrics_collector_initialization(tmp_path):
    """Тест инициализации сборщика метрик"""
    db_path = str(tmp_path / "metrics.db")
    collector = MetricsCollector(db_path)
    await collector.initialize()
    
    assert os.path.exists(db_path)


@pytest.mark.asyncio
async def test_collect_agent_metrics(metrics_collector):
    """Тест сбора метрик агентов"""
    # Метрики для исследователя
    researcher_metrics = {
        "tasks_completed": 5,
        "avg_research_time": 2.5,
        "success_rate": 0.8
    }
    await metrics_collector.collect_agent_metrics(
        AgentRole.RESEARCHER,
        researcher_metrics
    )
    
    # Метрики для планировщика
    planner_metrics = {
        "tasks_planned": 10,
        "avg_planning_time": 1.5,
        "plan_success_rate": 0.9
    }
    await metrics_collector.collect_agent_metrics(
        AgentRole.PLANNER,
        planner_metrics
    )
    
    # Получаем метрики
    metrics = await metrics_collector.get_agent_metrics(
        AgentRole.RESEARCHER
    )
    assert metrics is not None
    assert metrics["tasks_completed"] == 5
    
    metrics = await metrics_collector.get_agent_metrics(
        AgentRole.PLANNER
    )
    assert metrics is not None
    assert metrics["tasks_planned"] == 10


@pytest.mark.asyncio
async def test_collect_task_metrics(metrics_collector):
    """Тест сбора метрик задач"""
    task_metrics = {
        "task_id": "test_task",
        "duration": 5.0,
        "status": TaskStatus.COMPLETED,
        "agent": AgentRole.EXECUTOR
    }
    
    await metrics_collector.collect_task_metrics(task_metrics)
    
    # Получаем метрики
    metrics = await metrics_collector.get_task_metrics("test_task")
    assert metrics is not None
    assert metrics["duration"] == 5.0
    assert metrics["status"] == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_get_metrics_by_timerange(metrics_collector):
    """Тест получения метрик по временному диапазону"""
    # Добавляем метрики
    for i in range(3):
        await metrics_collector.collect_agent_metrics(
            AgentRole.RESEARCHER,
            {
                "tasks_completed": i,
                "timestamp": datetime.now() - timedelta(days=i)
            }
        )
    
    # Получаем метрики за последний день
    start_time = datetime.now() - timedelta(days=1)
    metrics = await metrics_collector.get_metrics_by_timerange(
        start_time=start_time
    )
    
    assert len(metrics) > 0
    assert all(m["timestamp"] >= start_time for m in metrics)


@pytest.mark.asyncio
async def test_aggregate_metrics(metrics_collector):
    """Тест агрегации метрик"""
    # Добавляем метрики
    for i in range(5):
        await metrics_collector.collect_task_metrics({
            "task_id": f"task_{i}",
            "duration": i + 1,
            "status": TaskStatus.COMPLETED
        })
    
    # Получаем агрегированные метрики
    aggregated = await metrics_collector.aggregate_metrics(
        metric_type="duration",
        aggregation="avg"
    )
    
    assert isinstance(aggregated, float)
    assert aggregated > 0


@pytest.mark.asyncio
async def test_performance_alerts(metrics_collector):
    """Тест оповещений о производительности"""
    # Добавляем метрики с низкой производительностью
    await metrics_collector.collect_agent_metrics(
        AgentRole.EXECUTOR,
        {
            "success_rate": 0.3,
            "response_time": 10.0
        }
    )
    
    # Проверяем оповещения
    alerts = await metrics_collector.check_performance_alerts()
    assert len(alerts) > 0
    assert any(
        alert["type"] == "low_success_rate"
        for alert in alerts
    )


@pytest.mark.asyncio
async def test_metrics_cleanup(metrics_collector):
    """Тест очистки старых метрик"""
    # Добавляем старые метрики
    old_time = datetime.now() - timedelta(days=31)
    await metrics_collector.collect_agent_metrics(
        AgentRole.RESEARCHER,
        {
            "tasks_completed": 1,
            "timestamp": old_time
        }
    )
    
    # Очищаем старые метрики
    deleted = await metrics_collector.cleanup_old_metrics(
        days_to_keep=30
    )
    assert deleted > 0
    
    # Проверяем, что старые метрики удалены
    metrics = await metrics_collector.get_metrics_by_timerange(
        start_time=old_time
    )
    assert len(metrics) == 0 