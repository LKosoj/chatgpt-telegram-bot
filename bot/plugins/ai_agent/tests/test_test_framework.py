"""
Тесты для фреймворка тестирования
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from ..test_framework import (
    TestFramework,
    TestSuite,
    TestResult
)
from ..logging import AgentLogger


@pytest.fixture
def logger():
    """Фикстура для логгера"""
    return AgentLogger("test_logs")


@pytest.fixture
def tests_dir(tmp_path):
    """Фикстура для директории тестов"""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    return tests_dir


@pytest.fixture
def test_suite():
    """Фикстура для тестового набора"""
    return TestSuite(
        name="test_suite",
        tests=[
            "test_module1.py",
            "test_module2.py"
        ],
        parallel=True
    )


@pytest.fixture
def test_results():
    """Фикстура для результатов тестов"""
    return [
        TestResult(
            name="test_module1.py",
            status="passed",
            duration=1.5
        ),
        TestResult(
            name="test_module2.py",
            status="failed",
            duration=0.5,
            error="Test error",
            traceback="Test traceback"
        )
    ]


@pytest.mark.asyncio
async def test_framework_initialization(tests_dir, logger):
    """Тест инициализации фреймворка"""
    framework = TestFramework(str(tests_dir), logger)
    
    assert framework.tests_dir == tests_dir
    assert framework.logger == logger
    assert isinstance(framework.suites, dict)
    assert isinstance(framework.results, dict)
    assert framework.executor is not None


@pytest.mark.asyncio
async def test_add_suite(tests_dir, logger, test_suite):
    """Тест добавления набора тестов"""
    framework = TestFramework(str(tests_dir), logger)
    
    # Добавляем набор
    framework.add_suite(test_suite)
    
    assert test_suite.name in framework.suites
    assert test_suite.name in framework.results
    assert framework.suites[test_suite.name] == test_suite
    assert framework.results[test_suite.name] == []


@pytest.mark.asyncio
async def test_run_suite(tests_dir, logger, test_suite):
    """Тест запуска набора тестов"""
    framework = TestFramework(str(tests_dir), logger)
    framework.add_suite(test_suite)
    
    # Мокаем _run_test
    async def mock_run_test(test_name, suite):
        return TestResult(
            name=test_name,
            status="passed",
            duration=1.0
        )
        
    with patch.object(
        framework,
        "_run_test",
        side_effect=mock_run_test
    ):
        # Запускаем тесты
        results = await framework.run_suite(test_suite.name)
        
        assert len(results) == len(test_suite.tests)
        for result in results:
            assert isinstance(result, TestResult)
            assert result.status == "passed"
            assert result.duration == 1.0


@pytest.mark.asyncio
async def test_run_test(tests_dir, logger, test_suite):
    """Тест запуска отдельного теста"""
    framework = TestFramework(str(tests_dir), logger)
    
    # Мокаем pytest
    with patch("pytest.main", return_value=0):
        # Запускаем тест
        result = await framework._run_test(
            "test_module.py",
            test_suite
        )
        
        assert isinstance(result, TestResult)
        assert result.status == "passed"
        assert result.duration > 0


@pytest.mark.asyncio
async def test_run_test_error(tests_dir, logger, test_suite):
    """Тест ошибки запуска теста"""
    framework = TestFramework(str(tests_dir), logger)
    
    # Мокаем pytest с ошибкой
    with patch("pytest.main", side_effect=Exception("Test error")):
        # Запускаем тест
        result = await framework._run_test(
            "test_module.py",
            test_suite
        )
        
        assert isinstance(result, TestResult)
        assert result.status == "failed"
        assert result.error == "Test error"
        assert result.traceback is not None


def test_get_results(tests_dir, logger, test_suite, test_results):
    """Тест получения результатов"""
    framework = TestFramework(str(tests_dir), logger)
    framework.add_suite(test_suite)
    framework.results[test_suite.name] = test_results
    
    # Получаем результаты конкретного набора
    suite_results = framework.get_results(test_suite.name)
    assert suite_results == test_results
    
    # Получаем все результаты
    all_results = framework.get_results()
    assert isinstance(all_results, dict)
    assert test_suite.name in all_results
    assert all_results[test_suite.name] == test_results


def test_get_stats(tests_dir, logger, test_suite, test_results):
    """Тест получения статистики"""
    framework = TestFramework(str(tests_dir), logger)
    framework.add_suite(test_suite)
    framework.results[test_suite.name] = test_results
    
    # Получаем статистику конкретного набора
    stats = framework.get_stats(test_suite.name)
    assert stats["total"] == 2
    assert stats["passed"] == 1
    assert stats["failed"] == 1
    assert stats["success_rate"] == 0.5
    assert stats["total_duration"] == 2.0
    
    # Получаем общую статистику
    all_stats = framework.get_stats()
    assert all_stats == stats


def test_save_load_results(
    tests_dir,
    logger,
    test_suite,
    test_results,
    tmp_path
):
    """Тест сохранения и загрузки результатов"""
    framework = TestFramework(str(tests_dir), logger)
    framework.add_suite(test_suite)
    framework.results[test_suite.name] = test_results
    
    # Сохраняем результаты
    results_file = tmp_path / "results.json"
    framework.save_results(str(results_file))
    assert results_file.exists()
    
    # Загружаем результаты в новый фреймворк
    new_framework = TestFramework(str(tests_dir), logger)
    new_framework.load_results(str(results_file))
    
    # Проверяем загруженные результаты
    loaded_results = new_framework.results[test_suite.name]
    assert len(loaded_results) == len(test_results)
    
    for orig, loaded in zip(test_results, loaded_results):
        assert loaded.name == orig.name
        assert loaded.status == orig.status
        assert loaded.duration == orig.duration
        assert loaded.error == orig.error
        assert loaded.traceback == orig.traceback


def test_cleanup(tests_dir, logger, test_suite, test_results):
    """Тест очистки ресурсов"""
    framework = TestFramework(str(tests_dir), logger)
    framework.add_suite(test_suite)
    framework.results[test_suite.name] = test_results
    
    # Очищаем ресурсы
    framework.cleanup()
    
    assert len(framework.suites) == 0
    assert len(framework.results) == 0
    assert framework.executor._shutdown 