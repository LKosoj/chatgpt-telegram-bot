"""
Модуль для системы тестирования
"""

import os
import sys
import json
import asyncio
import pytest
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from .logging import AgentLogger


@dataclass
class TestResult:
    """Результат теста"""
    name: str
    status: str
    duration: float
    error: Optional[str] = None
    traceback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestSuite:
    """Набор тестов"""
    name: str
    tests: List[str]
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    parallel: bool = False
    timeout: int = 300


class TestFramework:
    """Фреймворк для тестирования"""
    
    def __init__(
        self,
        tests_dir: str = "tests",
        logger: Optional[AgentLogger] = None,
        max_workers: int = 4
    ):
        """
        Инициализация фреймворка
        
        Args:
            tests_dir: директория с тестами
            logger: логгер (опционально)
            max_workers: максимальное количество параллельных тестов
        """
        self.tests_dir = Path(tests_dir)
        self.logger = logger or AgentLogger("test_logs")
        self.max_workers = max_workers
        
        # Наборы тестов
        self.suites: Dict[str, TestSuite] = {}
        
        # Результаты тестов
        self.results: Dict[str, List[TestResult]] = {}
        
        # Пул потоков для параллельного выполнения
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def add_suite(self, suite: TestSuite):
        """
        Добавление набора тестов
        
        Args:
            suite: набор тестов
        """
        self.suites[suite.name] = suite
        self.results[suite.name] = []
        
    async def run_suite(self, name: str) -> List[TestResult]:
        """
        Запуск набора тестов
        
        Args:
            name: название набора
            
        Returns:
            List[TestResult]: результаты тестов
        """
        suite = self.suites.get(name)
        if not suite:
            raise ValueError(f"Test suite not found: {name}")
            
        try:
            # Выполняем setup
            if suite.setup:
                await suite.setup()
                
            # Запускаем тесты
            if suite.parallel:
                # Параллельное выполнение
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(
                        self.executor,
                        self._run_test,
                        test,
                        suite
                    )
                    for test in suite.tests
                ]
                results = await asyncio.gather(*tasks)
            else:
                # Последовательное выполнение
                results = []
                for test in suite.tests:
                    result = await self._run_test(test, suite)
                    results.append(result)
                    
            # Сохраняем результаты
            self.results[name].extend(results)
            
            return results
            
        except Exception as e:
            self.logger.error(
                "Error running test suite",
                error=str(e),
                extra={"suite": name}
            )
            raise
            
        finally:
            # Выполняем teardown
            if suite.teardown:
                await suite.teardown()
                
    async def _run_test(
        self,
        test_name: str,
        suite: TestSuite
    ) -> TestResult:
        """
        Запуск отдельного теста
        
        Args:
            test_name: название теста
            suite: набор тестов
            
        Returns:
            TestResult: результат теста
        """
        start_time = time.time()
        
        try:
            # Импортируем и выполняем тест
            module_name = test_name.replace("/", ".").replace(".py", "")
            test_module = __import__(module_name, fromlist=["*"])
            
            # Запускаем pytest
            pytest.main(["-v", test_name])
            
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                status="passed",
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                status="failed",
                duration=duration,
                error=str(e),
                traceback=str(sys.exc_info()[2])
            )
            
    def get_results(
        self,
        suite_name: Optional[str] = None
    ) -> Union[List[TestResult], Dict[str, List[TestResult]]]:
        """
        Получение результатов тестов
        
        Args:
            suite_name: название набора (опционально)
            
        Returns:
            Union[List[TestResult], Dict[str, List[TestResult]]]: результаты
        """
        if suite_name:
            return self.results.get(suite_name, [])
        return self.results
        
    def get_stats(
        self,
        suite_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Получение статистики тестов
        
        Args:
            suite_name: название набора (опционально)
            
        Returns:
            Dict[str, Any]: статистика
        """
        results = (
            self.results.get(suite_name, [])
            if suite_name
            else [r for rs in self.results.values() for r in rs]
        )
        
        total = len(results)
        passed = len([r for r in results if r.status == "passed"])
        failed = len([r for r in results if r.status == "failed"])
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
            "total_duration": sum(r.duration for r in results)
        }
        
    def save_results(self, filename: str):
        """
        Сохранение результатов в файл
        
        Args:
            filename: имя файла
        """
        try:
            # Преобразуем результаты в JSON
            data = {
                suite_name: [
                    {
                        "name": r.name,
                        "status": r.status,
                        "duration": r.duration,
                        "error": r.error,
                        "traceback": r.traceback,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in results
                ]
                for suite_name, results in self.results.items()
            }
            
            # Сохраняем в файл
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved test results to {filename}")
            
        except Exception as e:
            self.logger.error(
                "Error saving test results",
                error=str(e),
                extra={"filename": filename}
            )
            
    def load_results(self, filename: str):
        """
        Загрузка результатов из файла
        
        Args:
            filename: имя файла
        """
        try:
            # Загружаем из файла
            with open(filename) as f:
                data = json.load(f)
                
            # Преобразуем в объекты TestResult
            self.results = {
                suite_name: [
                    TestResult(
                        name=r["name"],
                        status=r["status"],
                        duration=r["duration"],
                        error=r.get("error"),
                        traceback=r.get("traceback"),
                        timestamp=datetime.fromisoformat(r["timestamp"])
                    )
                    for r in results
                ]
                for suite_name, results in data.items()
            }
            
            self.logger.info(f"Loaded test results from {filename}")
            
        except Exception as e:
            self.logger.error(
                "Error loading test results",
                error=str(e),
                extra={"filename": filename}
            )
            
    def cleanup(self):
        """Очистка ресурсов"""
        self.executor.shutdown(wait=True)
        self.results.clear()
        self.suites.clear() 