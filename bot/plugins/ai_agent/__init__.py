"""
AI Agent System - мультиагентная система для обработки и анализа информации
"""

from .agents import MultiAgentSystem, AIAgentPlugin
from .nlp import NLPAnalyzer
from .storage import ResearchStorage
from .executor import ActionExecutor
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

__version__ = '1.0.0'
__all__ = [
    'MultiAgentSystem',
    'AIAgentPlugin',
    'NLPAnalyzer',
    'ResearchStorage',
    'ActionExecutor',
    'AgentRole',
    'TaskPriority',
    'TaskStatus',
    'AgentMessage',
    'PlanStep',
    'ActionPlan',
    'ResearchResult',
    'Task'
] 