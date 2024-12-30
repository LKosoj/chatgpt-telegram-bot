"""
Тесты для системы обучения
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime
from ..training_system import (
    TrainingSystem,
    TrainingModule,
    TrainingProgress
)
from ..logging import AgentLogger


@pytest.fixture
def logger():
    """Фикстура для логгера"""
    return AgentLogger("test_logs")


@pytest.fixture
def training_dir(tmp_path):
    """Фикстура для директории обучения"""
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    return training_dir


@pytest.fixture
def training_system(training_dir, logger):
    """Фикстура для системы обучения"""
    return TrainingSystem(str(training_dir), logger)


@pytest.fixture
def test_module():
    """Фикстура для тестового модуля"""
    return TrainingModule(
        title="Test Module",
        description="Test description",
        content="Test content",
        difficulty="beginner",
        prerequisites=["Python basics"],
        exercises=[],
        examples=[]
    )


def test_training_system_initialization(training_dir, logger):
    """Тест инициализации системы обучения"""
    system = TrainingSystem(str(training_dir), logger)
    
    assert system.training_dir == training_dir
    assert system.logger == logger
    assert isinstance(system.modules, dict)
    assert isinstance(system.progress, dict)


def test_create_training_module(training_system):
    """Тест создания обучающего модуля"""
    module = training_system.create_training_module(
        title="Test Module",
        description="Test description",
        content="Test content",
        difficulty="beginner",
        prerequisites=["Python basics"],
        exercises=[],
        examples=[]
    )
    
    assert isinstance(module, TrainingModule)
    assert module.title == "Test Module"
    assert module.description == "Test description"
    assert module.difficulty == "beginner"
    assert "Test Module" in training_system.modules


def test_add_exercise(training_system, test_module):
    """Тест добавления упражнения"""
    training_system.add_exercise(
        module=test_module,
        title="Test Exercise",
        description="Exercise description",
        task="Exercise task",
        solution="Exercise solution",
        hints=["Hint 1", "Hint 2"]
    )
    
    assert len(test_module.exercises) == 1
    exercise = test_module.exercises[0]
    assert exercise["title"] == "Test Exercise"
    assert exercise["description"] == "Exercise description"
    assert exercise["solution"] == "Exercise solution"
    assert len(exercise["hints"]) == 2


def test_add_example(training_system, test_module):
    """Тест добавления примера"""
    training_system.add_example(
        module=test_module,
        title="Test Example",
        code="print('test')",
        description="Example description",
        output="test"
    )
    
    assert len(test_module.examples) == 1
    example = test_module.examples[0]
    assert example["title"] == "Test Example"
    assert example["code"] == "print('test')"
    assert example["description"] == "Example description"
    assert example["output"] == "test"


def test_process_feedback(training_system, test_module):
    """Тест обработки обратной связи"""
    training_system.process_feedback(
        module=test_module,
        rating=4,
        comment="Great module!",
        suggestions=["Add more examples"]
    )
    
    assert len(test_module.feedback) == 1
    feedback = test_module.feedback[0]
    assert feedback["rating"] == 4
    assert feedback["comment"] == "Great module!"
    assert len(feedback["suggestions"]) == 1
    assert isinstance(feedback["timestamp"], datetime)


def test_track_progress(training_system):
    """Тест отслеживания прогресса"""
    training_system.track_progress(
        module_id="test_module",
        exercise_id="exercise1",
        score=0.8,
        feedback={"comment": "Good job!"}
    )
    
    assert "test_module" in training_system.progress
    progress = training_system.progress["test_module"]
    assert isinstance(progress, TrainingProgress)
    assert "exercise1" in progress.completed_exercises
    assert progress.quiz_scores["exercise1"] == 0.8
    assert len(progress.feedback_given) == 1


def test_update_training_materials(training_system, test_module):
    """Тест обновления обучающих материалов"""
    old_timestamp = test_module.last_updated
    
    updates = {
        "description": "Updated description",
        "difficulty": "intermediate"
    }
    
    training_system.update_training_materials(test_module, updates)
    
    assert test_module.description == "Updated description"
    assert test_module.difficulty == "intermediate"
    assert test_module.last_updated > old_timestamp


def test_save_and_load_module(training_system, test_module):
    """Тест сохранения и загрузки модуля"""
    # Добавляем тестовые данные
    training_system.add_exercise(
        module=test_module,
        title="Test Exercise",
        description="Exercise description",
        task="Exercise task",
        solution="Exercise solution",
        hints=["Hint 1"]
    )
    
    training_system.add_example(
        module=test_module,
        title="Test Example",
        code="print('test')",
        description="Example description"
    )
    
    # Сохраняем модуль
    training_system.save_module(test_module)
    
    # Проверяем файл
    module_file = training_system.training_dir / "test_module.json"
    assert module_file.exists()
    
    # Загружаем модуль
    loaded_module = training_system.load_module("test_module.json")
    
    assert loaded_module.title == test_module.title
    assert loaded_module.description == test_module.description
    assert len(loaded_module.exercises) == len(test_module.exercises)
    assert len(loaded_module.examples) == len(test_module.examples)


def test_export_to_markdown(training_system, test_module):
    """Тест экспорта в Markdown"""
    # Добавляем тестовые данные
    training_system.add_exercise(
        module=test_module,
        title="Test Exercise",
        description="Exercise description",
        task="Exercise task",
        solution="Exercise solution",
        hints=["Hint 1"]
    )
    
    training_system.add_example(
        module=test_module,
        title="Test Example",
        code="print('test')",
        description="Example description",
        output="test"
    )
    
    # Экспортируем в Markdown
    training_system.export_to_markdown(test_module, "test_module.md")
    
    # Проверяем файл
    doc_file = training_system.training_dir / "test_module.md"
    assert doc_file.exists()
    
    # Проверяем содержимое
    content = doc_file.read_text()
    assert "# Test Module" in content
    assert "## Информация" in content
    assert "## Примеры" in content
    assert "## Упражнения" in content
    assert "```python" in content 