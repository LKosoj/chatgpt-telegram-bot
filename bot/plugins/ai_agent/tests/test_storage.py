"""
Тесты для хранилища
"""

import pytest
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from ..storage import ResearchStorage, TaskStorage
from ..models import (
    ResearchResult,
    Task,
    ActionPlan,
    TaskStatus,
    TaskPriority,
    AgentRole
)
from ..logging import AgentLogger


@pytest.fixture
def logger():
    """Фикстура для логгера"""
    return AgentLogger("test_logs")


@pytest.fixture
async def research_storage(logger, tmp_path):
    """Фикстура для хранилища исследований"""
    db_path = str(tmp_path / "test.db")
    storage = ResearchStorage(db_path, logger)
    await storage.initialize()
    yield storage


@pytest.fixture
async def task_storage(logger, tmp_path):
    """Фикстура для хранилища задач"""
    db_path = str(tmp_path / "test.db")
    storage = TaskStorage(db_path, logger)
    await storage.initialize()
    yield storage


@pytest.mark.asyncio
async def test_research_storage_initialization(tmp_path):
    """Тест инициализации хранилища исследований"""
    db_path = str(tmp_path / "test.db")
    storage = ResearchStorage(db_path)
    await storage.initialize()
    
    assert os.path.exists(db_path)


@pytest.mark.asyncio
async def test_task_storage_initialization(tmp_path):
    """Тест инициализации хранилища задач"""
    db_path = str(tmp_path / "test.db")
    storage = TaskStorage(db_path)
    await storage.initialize()
    
    assert os.path.exists(db_path)


@pytest.mark.asyncio
async def test_save_research_result(research_storage):
    """Тест сохранения результата исследования"""
    result = ResearchResult(
        query="Test query",
        data={"result": "test"},
        source="test",
        relevance=0.9
    )
    
    result_id = await research_storage.save_result(result)
    assert result_id is not None
    
    # Проверяем сохранение с метаданными
    result_id = await research_storage.save_result(
        result,
        metadata={"test": "metadata"}
    )
    assert result_id is not None


@pytest.mark.asyncio
async def test_get_research_result(research_storage):
    """Тест получения результата исследования"""
    # Сохраняем результат
    result = ResearchResult(
        query="Test query",
        data={"result": "test"},
        source="test",
        relevance=0.9
    )
    result_id = await research_storage.save_result(result)
    
    # Получаем результат
    saved_result = await research_storage.get_result(result_id)
    assert saved_result is not None
    assert saved_result.query == result.query
    assert saved_result.data == result.data
    
    # Проверяем несуществующий результат
    saved_result = await research_storage.get_result(999)
    assert saved_result is None


@pytest.mark.asyncio
async def test_search_research_results(research_storage):
    """Тест поиска результатов исследований"""
    # Сохраняем несколько результатов
    results = [
        ResearchResult(
            query=f"Test query {i}",
            data={"result": f"test {i}"},
            source="test",
            relevance=0.9 - i * 0.1
        )
        for i in range(3)
    ]
    
    for result in results:
        await research_storage.save_result(result)
        
    # Поиск по запросу
    found = await research_storage.search_results("Test query")
    assert len(found) > 0
    
    # Поиск с минимальной релевантностью
    found = await research_storage.search_results(
        "Test query",
        min_relevance=0.8
    )
    assert len(found) > 0
    assert all(r.relevance >= 0.8 for r in found)
    
    # Поиск с лимитом
    found = await research_storage.search_results(
        "Test query",
        limit=2
    )
    assert len(found) <= 2


@pytest.mark.asyncio
async def test_research_relations(research_storage):
    """Тест связей между результатами исследований"""
    # Сохраняем результаты
    result1 = ResearchResult(
        query="Test query 1",
        data={"result": "test 1"}
    )
    result2 = ResearchResult(
        query="Test query 2",
        data={"result": "test 2"}
    )
    
    id1 = await research_storage.save_result(result1)
    id2 = await research_storage.save_result(result2)
    
    # Создаем связь
    await research_storage.add_relation(
        id1,
        id2,
        "similar",
        weight=0.8
    )
    
    # Получаем связанные результаты
    related = await research_storage.get_related_results(id1)
    assert len(related) > 0
    
    # Получаем связанные результаты по типу
    related = await research_storage.get_related_results(
        id1,
        relation_type="similar"
    )
    assert len(related) > 0


@pytest.mark.asyncio
async def test_create_task(task_storage):
    """Тест создания задачи"""
    task = Task(
        task_id="test_task",
        description="Test task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        assigned_to=AgentRole.RESEARCHER
    )
    
    task_id = await task_storage.create_task(task)
    assert task_id == task.task_id
    
    # Создание с метаданными
    task_id = await task_storage.create_task(
        task,
        metadata={"test": "metadata"}
    )
    assert task_id == task.task_id


@pytest.mark.asyncio
async def test_get_task(task_storage):
    """Тест получения задачи"""
    # Создаем задачу
    task = Task(
        task_id="test_task",
        description="Test task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        assigned_to=AgentRole.RESEARCHER
    )
    await task_storage.create_task(task)
    
    # Получаем задачу
    saved_task = await task_storage.get_task(task.task_id)
    assert saved_task is not None
    assert saved_task.task_id == task.task_id
    assert saved_task.description == task.description
    
    # Получаем несуществующую задачу
    saved_task = await task_storage.get_task("nonexistent")
    assert saved_task is None


@pytest.mark.asyncio
async def test_update_task_status(task_storage):
    """Тест обновления статуса задачи"""
    # Создаем задачу
    task = Task(
        task_id="test_task",
        description="Test task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM
    )
    await task_storage.create_task(task)
    
    # Обновляем статус
    await task_storage.update_task_status(
        task.task_id,
        TaskStatus.IN_PROGRESS
    )
    
    # Проверяем обновление
    saved_task = await task_storage.get_task(task.task_id)
    assert saved_task.status == TaskStatus.IN_PROGRESS
    
    # Обновляем с метаданными
    await task_storage.update_task_status(
        task.task_id,
        TaskStatus.COMPLETED,
        metadata={"completion_time": "2023-01-01"}
    )


@pytest.mark.asyncio
async def test_get_tasks_by_status(task_storage):
    """Тест получения задач по статусу"""
    # Создаем задачи
    tasks = [
        Task(
            task_id=f"task_{i}",
            description=f"Task {i}",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM
        )
        for i in range(3)
    ]
    
    for task in tasks:
        await task_storage.create_task(task)
        
    # Получаем задачи
    pending_tasks = await task_storage.get_tasks_by_status(
        TaskStatus.PENDING
    )
    assert len(pending_tasks) > 0
    
    # Получаем с лимитом
    pending_tasks = await task_storage.get_tasks_by_status(
        TaskStatus.PENDING,
        limit=2
    )
    assert len(pending_tasks) <= 2


@pytest.mark.asyncio
async def test_action_plans(task_storage):
    """Тест планов действий"""
    # Создаем задачу
    task = Task(
        task_id="test_task",
        description="Test task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM
    )
    await task_storage.create_task(task)
    
    # Создаем план
    plan = ActionPlan(
        plan_id="test_plan",
        task_id=task.task_id,
        steps=[]
    )
    
    # Сохраняем план
    await task_storage.save_action_plan(task.task_id, plan)
    
    # Получаем план
    saved_plan = await task_storage.get_action_plan(task.task_id)
    assert saved_plan is not None
    assert saved_plan.plan_id == plan.plan_id
    assert saved_plan.task_id == plan.task_id


@pytest.mark.asyncio
async def test_subtasks(task_storage):
    """Тест подзадач"""
    # Создаем основную задачу
    parent_task = Task(
        task_id="parent_task",
        description="Parent task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM
    )
    await task_storage.create_task(parent_task)
    
    # Создаем подзадачи
    subtasks = [
        Task(
            task_id=f"subtask_{i}",
            description=f"Subtask {i}",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            parent_task_id=parent_task.task_id
        )
        for i in range(3)
    ]
    
    for task in subtasks:
        await task_storage.create_task(task)
        
    # Получаем подзадачи
    saved_subtasks = await task_storage.get_subtasks(
        parent_task.task_id
    )
    assert len(saved_subtasks) == len(subtasks) 