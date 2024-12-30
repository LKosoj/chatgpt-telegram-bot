"""
Модуль с моделями данных системы
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union


class AgentRole(str, Enum):
    """Роли агентов"""
    RESEARCHER = "researcher"
    PLANNER = "planner"
    EXECUTOR = "executor"


class TaskPriority(int, Enum):
    """Приоритеты задач"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(str, Enum):
    """Статусы задач"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentMessage:
    """Сообщение агента"""
    role: AgentRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            **asdict(self),
            'role': self.role.value,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Создание из словаря"""
        return cls(
            role=AgentRole(data['role']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata')
        )


@dataclass
class PlanStep:
    """Шаг плана"""
    step_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            **asdict(self),
            'status': self.status.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanStep':
        """Создание из словаря"""
        return cls(
            step_id=data['step_id'],
            description=data['description'],
            status=TaskStatus(data['status']),
            dependencies=data.get('dependencies', []),
            estimated_duration=data.get('estimated_duration'),
            actual_duration=data.get('actual_duration'),
            result=data.get('result')
        )


@dataclass
class ActionPlan:
    """План действий"""
    plan_id: str
    task_id: str
    steps: List[PlanStep]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            **asdict(self),
            'status': self.status.value,
            'steps': [step.to_dict() for step in self.steps],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionPlan':
        """Создание из словаря"""
        return cls(
            plan_id=data['plan_id'],
            task_id=data['task_id'],
            steps=[PlanStep.from_dict(s) for s in data['steps']],
            status=TaskStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
            if data.get('updated_at') else None,
            metadata=data.get('metadata')
        )


@dataclass
class ResearchResult:
    """Результат исследования"""
    query: str
    data: Dict[str, Any]
    source: Optional[str] = None
    relevance: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchResult':
        """Создание из словаря"""
        return cls(
            query=data['query'],
            data=data['data'],
            source=data.get('source'),
            relevance=data.get('relevance', 1.0),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


@dataclass
class Task:
    """Задача"""
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_to: Optional[AgentRole] = None
    parent_task_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            **asdict(self),
            'status': self.status.value,
            'priority': self.priority.value,
            'assigned_to': self.assigned_to.value if self.assigned_to else None,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat()
            if self.started_at else None,
            'completed_at': self.completed_at.isoformat()
            if self.completed_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Создание из словаря"""
        return cls(
            task_id=data['task_id'],
            description=data['description'],
            status=TaskStatus(data['status']),
            priority=TaskPriority(data['priority']),
            assigned_to=AgentRole(data['assigned_to'])
            if data.get('assigned_to') else None,
            parent_task_id=data.get('parent_task_id'),
            created_at=datetime.fromisoformat(data['created_at']),
            started_at=datetime.fromisoformat(data['started_at'])
            if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at'])
            if data.get('completed_at') else None,
            metadata=data.get('metadata')
        ) 