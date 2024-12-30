"""
Модуль для хранения данных системы
"""

import json
import aiosqlite
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from .models import (
    ResearchResult,
    Task,
    ActionPlan,
    AgentRole,
    TaskStatus
)
from .logging import AgentLogger


class ResearchStorage:
    """Хранилище результатов исследований"""
    
    def __init__(
        self,
        db_path: str,
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация хранилища
        
        Args:
            db_path: путь к файлу базы данных
            logger: логгер (опционально)
        """
        self.db_path = db_path
        self.logger = logger or AgentLogger("logs")
        
    async def initialize(self):
        """Инициализация базы данных"""
        async with aiosqlite.connect(self.db_path) as db:
            # Создаем таблицу для результатов исследований
            await db.execute('''
                CREATE TABLE IF NOT EXISTS research_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    result_data TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    relevance REAL,
                    metadata TEXT
                )
            ''')
            
            # Создаем таблицу для связей между исследованиями
            await db.execute('''
                CREATE TABLE IF NOT EXISTS research_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL,
                    FOREIGN KEY (source_id) REFERENCES research_results (id),
                    FOREIGN KEY (target_id) REFERENCES research_results (id)
                )
            ''')
            
            # Создаем индексы
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_research_query
                ON research_results (query)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_research_timestamp
                ON research_results (timestamp)
            ''')
            
            await db.commit()
            
    async def save_result(
        self,
        result: ResearchResult,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Сохранение результата исследования
        
        Args:
            result: результат исследования
            metadata: метаданные
            
        Returns:
            int: идентификатор сохраненного результата
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                INSERT INTO research_results (
                    query, result_data, source,
                    relevance, metadata
                )
                VALUES (?, ?, ?, ?, ?)
            ''', (
                result.query,
                json.dumps(result.data, ensure_ascii=False),
                result.source,
                result.relevance,
                json.dumps(metadata, ensure_ascii=False) if metadata else None
            ))
            await db.commit()
            
            result_id = cursor.lastrowid
            self.logger.info(
                f"Saved research result {result_id}",
                {"query": result.query}
            )
            return result_id
            
    async def get_result(self, result_id: int) -> Optional[ResearchResult]:
        """
        Получение результата исследования
        
        Args:
            result_id: идентификатор результата
            
        Returns:
            Optional[ResearchResult]: результат исследования
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM research_results WHERE id = ?
            ''', (result_id,))
            
            row = await cursor.fetchone()
            if not row:
                return None
                
            return ResearchResult(
                query=row['query'],
                data=json.loads(row['result_data']),
                source=row['source'],
                relevance=row['relevance']
            )
            
    async def search_results(
        self,
        query: str,
        limit: int = 10,
        min_relevance: float = 0.5
    ) -> List[ResearchResult]:
        """
        Поиск результатов исследований
        
        Args:
            query: поисковый запрос
            limit: максимальное количество результатов
            min_relevance: минимальная релевантность
            
        Returns:
            List[ResearchResult]: список результатов
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM research_results
                WHERE relevance >= ?
                AND (
                    query LIKE ?
                    OR result_data LIKE ?
                )
                ORDER BY relevance DESC, timestamp DESC
                LIMIT ?
            ''', (
                min_relevance,
                f"%{query}%",
                f"%{query}%",
                limit
            ))
            
            rows = await cursor.fetchall()
            return [
                ResearchResult(
                    query=row['query'],
                    data=json.loads(row['result_data']),
                    source=row['source'],
                    relevance=row['relevance']
                )
                for row in rows
            ]
            
    async def add_relation(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        weight: float = 1.0
    ):
        """
        Добавление связи между результатами
        
        Args:
            source_id: идентификатор исходного результата
            target_id: идентификатор целевого результата
            relation_type: тип связи
            weight: вес связи
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO research_relations (
                    source_id, target_id,
                    relation_type, weight
                )
                VALUES (?, ?, ?, ?)
            ''', (source_id, target_id, relation_type, weight))
            await db.commit()
            
    async def get_related_results(
        self,
        result_id: int,
        relation_type: Optional[str] = None
    ) -> List[ResearchResult]:
        """
        Получение связанных результатов
        
        Args:
            result_id: идентификатор результата
            relation_type: тип связи
            
        Returns:
            List[ResearchResult]: список связанных результатов
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            query = '''
                SELECT r.* FROM research_results r
                JOIN research_relations rel
                ON r.id = rel.target_id
                WHERE rel.source_id = ?
            '''
            params = [result_id]
            
            if relation_type:
                query += " AND rel.relation_type = ?"
                params.append(relation_type)
                
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            
            return [
                ResearchResult(
                    query=row['query'],
                    data=json.loads(row['result_data']),
                    source=row['source'],
                    relevance=row['relevance']
                )
                for row in rows
            ]


class TaskStorage:
    """Хранилище задач"""
    
    def __init__(
        self,
        db_path: str,
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация хранилища
        
        Args:
            db_path: путь к файлу базы данных
            logger: логгер (опционально)
        """
        self.db_path = db_path
        self.logger = logger or AgentLogger("logs")
        
    async def initialize(self):
        """Инициализация базы данных"""
        async with aiosqlite.connect(self.db_path) as db:
            # Создаем таблицу для задач
            await db.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    assigned_to TEXT,
                    parent_task_id TEXT,
                    metadata TEXT,
                    FOREIGN KEY (parent_task_id) REFERENCES tasks (task_id)
                )
            ''')
            
            # Создаем таблицу для планов действий
            await db.execute('''
                CREATE TABLE IF NOT EXISTS action_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    plan_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
                )
            ''')
            
            # Создаем индексы
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_task_status
                ON tasks (status)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_task_priority
                ON tasks (priority)
            ''')
            
            await db.commit()
            
    async def create_task(
        self,
        task: Task,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Создание задачи
        
        Args:
            task: задача
            metadata: метаданные
            
        Returns:
            str: идентификатор задачи
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO tasks (
                    task_id, description, status,
                    priority, assigned_to, parent_task_id,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.task_id,
                task.description,
                task.status.value,
                task.priority.value,
                task.assigned_to.value if task.assigned_to else None,
                task.parent_task_id,
                json.dumps(metadata, ensure_ascii=False) if metadata else None
            ))
            await db.commit()
            
            self.logger.info(
                f"Created task {task.task_id}",
                {"description": task.description}
            )
            return task.task_id
            
    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Получение задачи
        
        Args:
            task_id: идентификатор задачи
            
        Returns:
            Optional[Task]: задача
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM tasks WHERE task_id = ?
            ''', (task_id,))
            
            row = await cursor.fetchone()
            if not row:
                return None
                
            return Task(
                task_id=row['task_id'],
                description=row['description'],
                status=TaskStatus(row['status']),
                priority=int(row['priority']),
                assigned_to=AgentRole(row['assigned_to'])
                if row['assigned_to'] else None,
                parent_task_id=row['parent_task_id']
            )
            
    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Обновление статуса задачи
        
        Args:
            task_id: идентификатор задачи
            status: новый статус
            metadata: метаданные
        """
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now().isoformat()
            
            if status == TaskStatus.IN_PROGRESS:
                await db.execute('''
                    UPDATE tasks
                    SET status = ?, started_at = ?, metadata = ?
                    WHERE task_id = ?
                ''', (
                    status.value,
                    now,
                    json.dumps(metadata, ensure_ascii=False)
                    if metadata else None,
                    task_id
                ))
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                await db.execute('''
                    UPDATE tasks
                    SET status = ?, completed_at = ?, metadata = ?
                    WHERE task_id = ?
                ''', (
                    status.value,
                    now,
                    json.dumps(metadata, ensure_ascii=False)
                    if metadata else None,
                    task_id
                ))
            else:
                await db.execute('''
                    UPDATE tasks
                    SET status = ?, metadata = ?
                    WHERE task_id = ?
                ''', (
                    status.value,
                    json.dumps(metadata, ensure_ascii=False)
                    if metadata else None,
                    task_id
                ))
                
            await db.commit()
            
            self.logger.info(
                f"Updated task {task_id} status to {status.value}",
                metadata
            )
            
    async def get_tasks_by_status(
        self,
        status: TaskStatus,
        limit: int = 10
    ) -> List[Task]:
        """
        Получение задач по статусу
        
        Args:
            status: статус
            limit: максимальное количество задач
            
        Returns:
            List[Task]: список задач
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM tasks
                WHERE status = ?
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
            ''', (status.value, limit))
            
            rows = await cursor.fetchall()
            return [
                Task(
                    task_id=row['task_id'],
                    description=row['description'],
                    status=TaskStatus(row['status']),
                    priority=int(row['priority']),
                    assigned_to=AgentRole(row['assigned_to'])
                    if row['assigned_to'] else None,
                    parent_task_id=row['parent_task_id']
                )
                for row in rows
            ]
            
    async def save_action_plan(
        self,
        task_id: str,
        plan: ActionPlan
    ):
        """
        Сохранение плана действий
        
        Args:
            task_id: идентификатор задачи
            plan: план действий
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO action_plans (
                    task_id, plan_data, status
                )
                VALUES (?, ?, ?)
            ''', (
                task_id,
                json.dumps(plan.to_dict(), ensure_ascii=False),
                plan.status.value
            ))
            await db.commit()
            
            self.logger.info(
                f"Saved action plan for task {task_id}",
                {"steps_count": len(plan.steps)}
            )
            
    async def get_action_plan(
        self,
        task_id: str
    ) -> Optional[ActionPlan]:
        """
        Получение плана действий
        
        Args:
            task_id: идентификатор задачи
            
        Returns:
            Optional[ActionPlan]: план действий
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM action_plans
                WHERE task_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (task_id,))
            
            row = await cursor.fetchone()
            if not row:
                return None
                
            plan_data = json.loads(row['plan_data'])
            return ActionPlan.from_dict(plan_data)
            
    async def get_subtasks(
        self,
        parent_task_id: str
    ) -> List[Task]:
        """
        Получение подзадач
        
        Args:
            parent_task_id: идентификатор родительской задачи
            
        Returns:
            List[Task]: список подзадач
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM tasks
                WHERE parent_task_id = ?
                ORDER BY priority DESC, created_at ASC
            ''', (parent_task_id,))
            
            rows = await cursor.fetchall()
            return [
                Task(
                    task_id=row['task_id'],
                    description=row['description'],
                    status=TaskStatus(row['status']),
                    priority=int(row['priority']),
                    assigned_to=AgentRole(row['assigned_to'])
                    if row['assigned_to'] else None,
                    parent_task_id=row['parent_task_id']
                )
                for row in rows
            ] 