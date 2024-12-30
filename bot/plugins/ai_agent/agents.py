"""
Модуль с агентами системы
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from .models import (
    AgentRole,
    TaskPriority,
    TaskStatus,
    AgentMessage,
    PlanStep,
    ActionPlan,
    ResearchResult,
    Task
)
from .storage import ResearchStorage, TaskStorage
from .logging import AgentLogger
from .monitoring import ProcessMonitor, MetricsCollector


class BaseAgent:
    """Базовый класс агента"""
    
    def __init__(
        self,
        role: AgentRole,
        task_storage: TaskStorage,
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация агента
        
        Args:
            role: роль агента
            task_storage: хранилище задач
            logger: логгер (опционально)
        """
        self.role = role
        self.task_storage = task_storage
        self.logger = logger or AgentLogger("logs", role)
        self.current_task: Optional[Task] = None
        
    async def process_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """
        Обработка входящего сообщения
        
        Args:
            message: входящее сообщение
            
        Returns:
            Optional[AgentMessage]: ответное сообщение
        """
        raise NotImplementedError
        
    async def handle_task(self, task: Task) -> bool:
        """
        Обработка задачи
        
        Args:
            task: задача
            
        Returns:
            bool: успешность выполнения
        """
        raise NotImplementedError
        
    async def _update_task_status(
        self,
        task: Task,
        status: TaskStatus,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Обновление статуса задачи
        
        Args:
            task: задача
            status: новый статус
            metadata: метаданные
        """
        await self.task_storage.update_task_status(
            task.task_id,
            status,
            metadata
        )
        self.logger.info(
            f"Updated task {task.task_id} status to {status.value}",
            metadata
        )


class ResearcherAgent(BaseAgent):
    """Агент-исследователь"""
    
    def __init__(
        self,
        task_storage: TaskStorage,
        research_storage: ResearchStorage,
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация агента
        
        Args:
            task_storage: хранилище задач
            research_storage: хранилище исследований
            logger: логгер (опционально)
        """
        super().__init__(AgentRole.RESEARCHER, task_storage, logger)
        self.research_storage = research_storage
        
    async def process_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """
        Обработка входящего сообщения
        
        Args:
            message: входящее сообщение
            
        Returns:
            Optional[AgentMessage]: ответное сообщение
        """
        # Создаем задачу исследования
        task = Task(
            task_id=str(uuid.uuid4()),
            description=message.content,
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            assigned_to=self.role
        )
        await self.task_storage.create_task(task)
        
        # Обрабатываем задачу
        success = await self.handle_task(task)
        
        # Формируем ответ
        if success:
            return AgentMessage(
                role=self.role,
                content="Research completed successfully"
            )
        else:
            return AgentMessage(
                role=self.role,
                content="Research failed"
            )
            
    async def handle_task(self, task: Task) -> bool:
        """
        Обработка задачи исследования
        
        Args:
            task: задача
            
        Returns:
            bool: успешность выполнения
        """
        try:
            self.current_task = task
            await self._update_task_status(
                task,
                TaskStatus.IN_PROGRESS
            )
            
            # Поиск существующих исследований
            existing_results = await self.research_storage.search_results(
                task.description
            )
            
            if existing_results:
                # Используем существующие результаты
                await self._update_task_status(
                    task,
                    TaskStatus.COMPLETED,
                    {"reused_results": len(existing_results)}
                )
                return True
                
            # Проводим новое исследование
            result = await self._conduct_research(task)
            
            if result:
                # Сохраняем результаты
                await self.research_storage.save_result(result)
                await self._update_task_status(
                    task,
                    TaskStatus.COMPLETED,
                    {"new_research": True}
                )
                return True
            else:
                await self._update_task_status(
                    task,
                    TaskStatus.FAILED,
                    {"reason": "Research failed"}
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "Error handling research task",
                error=e,
                extra={"task_id": task.task_id}
            )
            await self._update_task_status(
                task,
                TaskStatus.FAILED,
                {"error": str(e)}
            )
            return False
            
        finally:
            self.current_task = None
            
    async def _conduct_research(self, task: Task) -> Optional[ResearchResult]:
        """
        Проведение исследования
        
        Args:
            task: задача
            
        Returns:
            Optional[ResearchResult]: результат исследования
        """
        # TODO: Реализовать проведение исследования
        # Здесь должна быть логика работы с внешними API,
        # анализ данных и т.д.
        pass


class PlannerAgent(BaseAgent):
    """Агент-планировщик"""
    
    def __init__(
        self,
        task_storage: TaskStorage,
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация агента
        
        Args:
            task_storage: хранилище задач
            logger: логгер (опционально)
        """
        super().__init__(AgentRole.PLANNER, task_storage, logger)
        
    async def process_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """
        Обработка входящего сообщения
        
        Args:
            message: входящее сообщение
            
        Returns:
            Optional[AgentMessage]: ответное сообщение
        """
        # Создаем задачу планирования
        task = Task(
            task_id=str(uuid.uuid4()),
            description=message.content,
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            assigned_to=self.role
        )
        await self.task_storage.create_task(task)
        
        # Обрабатываем задачу
        success = await self.handle_task(task)
        
        # Формируем ответ
        if success:
            return AgentMessage(
                role=self.role,
                content="Plan created successfully"
            )
        else:
            return AgentMessage(
                role=self.role,
                content="Plan creation failed"
            )
            
    async def handle_task(self, task: Task) -> bool:
        """
        Обработка задачи планирования
        
        Args:
            task: задача
            
        Returns:
            bool: успешность выполнения
        """
        try:
            self.current_task = task
            await self._update_task_status(
                task,
                TaskStatus.IN_PROGRESS
            )
            
            # Создаем план действий
            plan = await self._create_plan(task)
            
            if plan:
                # Сохраняем план
                await self.task_storage.save_action_plan(
                    task.task_id,
                    plan
                )
                await self._update_task_status(
                    task,
                    TaskStatus.COMPLETED,
                    {"plan_id": plan.plan_id}
                )
                return True
            else:
                await self._update_task_status(
                    task,
                    TaskStatus.FAILED,
                    {"reason": "Plan creation failed"}
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "Error handling planning task",
                error=e,
                extra={"task_id": task.task_id}
            )
            await self._update_task_status(
                task,
                TaskStatus.FAILED,
                {"error": str(e)}
            )
            return False
            
        finally:
            self.current_task = None
            
    async def _create_plan(self, task: Task) -> Optional[ActionPlan]:
        """
        Создание плана действий
        
        Args:
            task: задача
            
        Returns:
            Optional[ActionPlan]: план действий
        """
        # TODO: Реализовать создание плана
        # Здесь должна быть логика анализа задачи,
        # декомпозиции на шаги и т.д.
        pass


class ExecutorAgent(BaseAgent):
    """Агент-исполнитель"""
    
    def __init__(
        self,
        task_storage: TaskStorage,
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация агента
        
        Args:
            task_storage: хранилище задач
            logger: логгер (опционально)
        """
        super().__init__(AgentRole.EXECUTOR, task_storage, logger)
        self.active_tasks: Set[str] = set()
        
    async def process_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """
        Обработка входящего сообщения
        
        Args:
            message: входящее сообщение
            
        Returns:
            Optional[AgentMessage]: ответное сообщение
        """
        # Создаем задачу выполнения
        task = Task(
            task_id=str(uuid.uuid4()),
            description=message.content,
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            assigned_to=self.role
        )
        await self.task_storage.create_task(task)
        
        # Обрабатываем задачу
        success = await self.handle_task(task)
        
        # Формируем ответ
        if success:
            return AgentMessage(
                role=self.role,
                content="Task executed successfully"
            )
        else:
            return AgentMessage(
                role=self.role,
                content="Task execution failed"
            )
            
    async def handle_task(self, task: Task) -> bool:
        """
        Обработка задачи выполнения
        
        Args:
            task: задача
            
        Returns:
            bool: успешность выполнения
        """
        if task.task_id in self.active_tasks:
            self.logger.warning(
                f"Task {task.task_id} is already being processed"
            )
            return False
            
        try:
            self.active_tasks.add(task.task_id)
            self.current_task = task
            await self._update_task_status(
                task,
                TaskStatus.IN_PROGRESS
            )
            
            # Получаем план действий
            plan = await self.task_storage.get_action_plan(task.task_id)
            
            if not plan:
                await self._update_task_status(
                    task,
                    TaskStatus.FAILED,
                    {"reason": "No action plan found"}
                )
                return False
                
            # Выполняем план
            success = await self._execute_plan(plan)
            
            if success:
                await self._update_task_status(
                    task,
                    TaskStatus.COMPLETED
                )
                return True
            else:
                await self._update_task_status(
                    task,
                    TaskStatus.FAILED,
                    {"reason": "Plan execution failed"}
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "Error handling execution task",
                error=e,
                extra={"task_id": task.task_id}
            )
            await self._update_task_status(
                task,
                TaskStatus.FAILED,
                {"error": str(e)}
            )
            return False
            
        finally:
            self.active_tasks.remove(task.task_id)
            self.current_task = None
            
    async def _execute_plan(self, plan: ActionPlan) -> bool:
        """
        Выполнение плана действий
        
        Args:
            plan: план действий
            
        Returns:
            bool: успешность выполнения
        """
        # TODO: Реализовать выполнение плана
        # Здесь должна быть логика выполнения шагов плана,
        # обработка ошибок и т.д.
        pass


class MultiAgentSystem:
    """Мультиагентная система"""
    
    def __init__(
        self,
        db_path: str,
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация системы
        
        Args:
            db_path: путь к файлу базы данных
            logger: логгер (опционально)
        """
        self.logger = logger or AgentLogger("logs")
        
        # Создаем хранилища
        self.task_storage = TaskStorage(db_path, self.logger)
        self.research_storage = ResearchStorage(db_path, self.logger)
        
        # Создаем агентов
        self.researcher = ResearcherAgent(
            self.task_storage,
            self.research_storage,
            self.logger
        )
        self.planner = PlannerAgent(
            self.task_storage,
            self.logger
        )
        self.executor = ExecutorAgent(
            self.task_storage,
            self.logger
        )
        
        # Создаем мониторинг
        self.process_monitor = ProcessMonitor(db_path)
        self.metrics_collector = MetricsCollector(db_path)
        
    async def initialize(self):
        """Инициализация системы"""
        # Инициализируем хранилища
        await self.task_storage.initialize()
        await self.research_storage.initialize()
        
        # Инициализируем мониторинг
        await self.process_monitor.initialize()
        await self.metrics_collector.initialize()
        
        # Запускаем мониторинг
        asyncio.create_task(self.process_monitor.start_monitoring())
        
        self.logger.info("Multi-agent system initialized")
        
    async def process_message(
        self,
        message: str,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> str:
        """
        Обработка входящего сообщения
        
        Args:
            message: сообщение
            priority: приоритет
            
        Returns:
            str: результат обработки
        """
        try:
            # Создаем основную задачу
            main_task = Task(
                task_id=str(uuid.uuid4()),
                description=message,
                status=TaskStatus.PENDING,
                priority=priority
            )
            await self.task_storage.create_task(main_task)
            
            # Исследуем задачу
            research_msg = AgentMessage(
                role=AgentRole.RESEARCHER,
                content=message
            )
            research_result = await self.researcher.process_message(
                research_msg
            )
            
            if not research_result:
                return "Research failed"
                
            # Создаем план
            plan_msg = AgentMessage(
                role=AgentRole.PLANNER,
                content=message
            )
            plan_result = await self.planner.process_message(plan_msg)
            
            if not plan_result:
                return "Planning failed"
                
            # Выполняем план
            exec_msg = AgentMessage(
                role=AgentRole.EXECUTOR,
                content=message
            )
            exec_result = await self.executor.process_message(exec_msg)
            
            if not exec_result:
                return "Execution failed"
                
            return "Task completed successfully"
            
        except Exception as e:
            self.logger.error(
                "Error processing message",
                error=e,
                extra={"message": message}
            )
            return f"Error: {str(e)}" 