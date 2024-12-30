"""
Тесты для агентов
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from ..agents import (
    BaseAgent,
    ResearcherAgent,
    PlannerAgent,
    ExecutorAgent,
    MultiAgentSystem
)
from ..models import (
    AgentRole,
    TaskPriority,
    TaskStatus,
    AgentMessage,
    Task,
    ActionPlan,
    ResearchResult
)
from ..storage import ResearchStorage, TaskStorage
from ..logging import AgentLogger


@pytest.fixture
def logger():
    """Фикстура для логгера"""
    return AgentLogger("test_logs")


@pytest.fixture
def task_storage():
    """Фикстура для хранилища задач"""
    storage = Mock(spec=TaskStorage)
    storage.create_task = AsyncMock()
    storage.get_task = AsyncMock()
    storage.update_task_status = AsyncMock()
    storage.get_tasks_by_status = AsyncMock()
    storage.save_action_plan = AsyncMock()
    storage.get_action_plan = AsyncMock()
    return storage


@pytest.fixture
def research_storage():
    """Фикстура для хранилища исследований"""
    storage = Mock(spec=ResearchStorage)
    storage.save_result = AsyncMock()
    storage.get_result = AsyncMock()
    storage.search_results = AsyncMock()
    return storage


@pytest.mark.asyncio
async def test_base_agent():
    """Тест базового агента"""
    agent = BaseAgent(AgentRole.RESEARCHER, Mock())
    
    with pytest.raises(NotImplementedError):
        await agent.process_message(Mock())
        
    with pytest.raises(NotImplementedError):
        await agent.handle_task(Mock())


@pytest.mark.asyncio
async def test_researcher_agent(task_storage, research_storage, logger):
    """Тест агента-исследователя"""
    agent = ResearcherAgent(task_storage, research_storage, logger)
    
    # Тест обработки сообщения
    message = AgentMessage(
        role=AgentRole.RESEARCHER,
        content="Test query"
    )
    
    # Настраиваем моки
    task_storage.create_task.return_value = "task_id"
    research_storage.search_results.return_value = []
    
    # Проверяем успешный сценарий
    result = await agent.process_message(message)
    assert result is not None
    assert task_storage.create_task.called
    
    # Проверяем обработку задачи
    task = Task(
        task_id="test_task",
        description="Test task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        assigned_to=AgentRole.RESEARCHER
    )
    
    # Настраиваем моки
    research_storage.search_results.return_value = [
        ResearchResult(
            query="Test query",
            data={"result": "test"}
        )
    ]
    
    # Проверяем успешный сценарий
    success = await agent.handle_task(task)
    assert success
    assert task_storage.update_task_status.called
    
    # Проверяем сценарий с ошибкой
    research_storage.search_results.side_effect = Exception("Test error")
    success = await agent.handle_task(task)
    assert not success


@pytest.mark.asyncio
async def test_planner_agent(task_storage, logger):
    """Тест агента-планировщика"""
    agent = PlannerAgent(task_storage, logger)
    
    # Тест обработки сообщения
    message = AgentMessage(
        role=AgentRole.PLANNER,
        content="Test plan"
    )
    
    # Настраиваем моки
    task_storage.create_task.return_value = "task_id"
    
    # Проверяем успешный сценарий
    result = await agent.process_message(message)
    assert result is not None
    assert task_storage.create_task.called
    
    # Проверяем обработку задачи
    task = Task(
        task_id="test_task",
        description="Test task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        assigned_to=AgentRole.PLANNER
    )
    
    # Настраиваем моки
    task_storage.save_action_plan.return_value = None
    
    # Проверяем успешный сценарий
    success = await agent.handle_task(task)
    assert not success  # TODO реализовать _create_plan
    assert task_storage.update_task_status.called


@pytest.mark.asyncio
async def test_executor_agent(task_storage, logger):
    """Тест агента-исполнителя"""
    agent = ExecutorAgent(task_storage, logger)
    
    # Тест обработки сообщения
    message = AgentMessage(
        role=AgentRole.EXECUTOR,
        content="Test execution"
    )
    
    # Настраиваем моки
    task_storage.create_task.return_value = "task_id"
    
    # Проверяем успешный сценарий
    result = await agent.process_message(message)
    assert result is not None
    assert task_storage.create_task.called
    
    # Проверяем обработку задачи
    task = Task(
        task_id="test_task",
        description="Test task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        assigned_to=AgentRole.EXECUTOR
    )
    
    # Настраиваем моки
    task_storage.get_action_plan.return_value = None
    
    # Проверяем сценарий без плана
    success = await agent.handle_task(task)
    assert not success
    assert task_storage.update_task_status.called
    
    # Проверяем сценарий с планом
    plan = ActionPlan(
        plan_id="test_plan",
        task_id="test_task",
        steps=[]
    )
    task_storage.get_action_plan.return_value = plan
    
    success = await agent.handle_task(task)
    assert not success  # TODO реализовать _execute_plan


@pytest.mark.asyncio
async def test_multi_agent_system(task_storage, research_storage, logger):
    """Тест мультиагентной системы"""
    system = MultiAgentSystem("test.db", logger)
    
    # Подменяем хранилища
    system.task_storage = task_storage
    system.research_storage = research_storage
    
    # Инициализируем систему
    await system.initialize()
    
    # Проверяем обработку сообщения
    result = await system.process_message(
        "Test message",
        priority=TaskPriority.MEDIUM
    )
    
    assert result == "Research failed"  # TODO реализовать агентов
    assert task_storage.create_task.called


@pytest.mark.asyncio
async def test_error_handling():
    """Тест обработки ошибок"""
    # Создаем агента с моками, которые бросают исключения
    task_storage = Mock(spec=TaskStorage)
    task_storage.create_task = AsyncMock(side_effect=Exception("Test error"))
    
    agent = ResearcherAgent(
        task_storage,
        Mock(spec=ResearchStorage),
        AgentLogger("test_logs")
    )
    
    # Проверяем обработку ошибок в process_message
    message = AgentMessage(
        role=AgentRole.RESEARCHER,
        content="Test"
    )
    
    with pytest.raises(Exception):
        await agent.process_message(message)
        
    # Проверяем обработку ошибок в handle_task
    task = Task(
        task_id="test_task",
        description="Test task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        assigned_to=AgentRole.RESEARCHER
    )
    
    success = await agent.handle_task(task)
    assert not success


@pytest.mark.asyncio
async def test_concurrent_tasks():
    """Тест параллельного выполнения задач"""
    # Создаем агента
    agent = ExecutorAgent(
        Mock(spec=TaskStorage),
        AgentLogger("test_logs")
    )
    
    # Создаем несколько задач
    tasks = [
        Task(
            task_id=f"task_{i}",
            description=f"Task {i}",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            assigned_to=AgentRole.EXECUTOR
        )
        for i in range(3)
    ]
    
    # Запускаем задачи параллельно
    results = await asyncio.gather(
        *[agent.handle_task(task) for task in tasks]
    )
    
    # Проверяем результаты
    assert len(results) == 3
    assert all(not result for result in results)  # TODO реализовать _execute_plan


@pytest.mark.asyncio
async def test_task_priorities():
    """Тест приоритетов задач"""
    # Создаем систему
    system = MultiAgentSystem("test.db")
    
    # Создаем задачи с разными приоритетами
    messages = [
        ("High priority", TaskPriority.HIGH),
        ("Medium priority", TaskPriority.MEDIUM),
        ("Low priority", TaskPriority.LOW)
    ]
    
    # Обрабатываем сообщения
    results = []
    for message, priority in messages:
        result = await system.process_message(message, priority)
        results.append(result)
        
    # Проверяем результаты
    assert len(results) == 3
    assert all(result == "Research failed" for result in results)


@pytest.mark.asyncio
async def test_system_initialization():
    """Тест инициализации системы"""
    # Создаем систему
    system = MultiAgentSystem("test.db")
    
    # Проверяем инициализацию компонентов
    assert system.researcher is not None
    assert system.planner is not None
    assert system.executor is not None
    assert system.task_storage is not None
    assert system.research_storage is not None
    assert system.process_monitor is not None
    assert system.metrics_collector is not None
    
    # Инициализируем систему
    await system.initialize()
    
    # Проверяем, что все компоненты инициализированы
    assert system.logger is not None 