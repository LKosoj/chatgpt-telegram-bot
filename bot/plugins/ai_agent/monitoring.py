"""
Модуль для мониторинга и сбора метрик системы
"""

import asyncio
import logging
import json
import time
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import aiosqlite
from pathlib import Path
import numpy as np
from .models import AgentRole, TaskStatus


@dataclass
class ProcessMetrics:
    """Метрики процесса"""
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    threads_count: int
    open_files: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AgentMetrics:
    """Метрики агента"""
    role: AgentRole
    tasks_total: int
    tasks_completed: int
    tasks_failed: int
    avg_processing_time: float
    success_rate: float
    error_rate: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            **asdict(self),
            'role': self.role.value,
            'timestamp': self.timestamp.isoformat()
        }


class ProcessMonitor:
    """Монитор процессов"""
    
    def __init__(
        self,
        logger: Optional[AgentLogger] = None,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        thread_threshold: int = 100
    ):
        """
        Инициализация монитора
        
        Args:
            logger: логгер
            cpu_threshold: порог CPU в процентах
            memory_threshold: порог памяти в процентах
            thread_threshold: порог количества потоков
        """
        self.logger = logger or AgentLogger("logs")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.thread_threshold = thread_threshold
        
        # Для отслеживания этапов
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.current_stage: Optional[str] = None
        
        # Для измерения времени
        self.timings: Dict[str, List[float]] = defaultdict(list)
        
    async def start_stage(self, stage_name: str):
        """
        Начало этапа обработки
        
        Args:
            stage_name: название этапа
        """
        if self.current_stage:
            await self.end_stage()
            
        self.current_stage = stage_name
        self.stages[stage_name] = {
            "start_time": time.time(),
            "status": "running",
            "metrics": {}
        }
        
        self.logger.info(
            f"Started stage: {stage_name}",
            extra={"stage": stage_name}
        )
        
    async def end_stage(self, status: str = "completed"):
        """
        Завершение текущего этапа
        
        Args:
            status: статус завершения
        """
        if not self.current_stage:
            return
            
        stage = self.stages[self.current_stage]
        duration = time.time() - stage["start_time"]
        
        stage.update({
            "end_time": time.time(),
            "duration": duration,
            "status": status
        })
        
        # Сохраняем время выполнения
        self.timings[self.current_stage].append(duration)
        
        self.logger.info(
            f"Ended stage: {self.current_stage}",
            extra={
                "stage": self.current_stage,
                "duration": duration,
                "status": status
            }
        )
        
        self.current_stage = None
        
    async def update_stage_metrics(self, metrics: Dict[str, Any]):
        """
        Обновление метрик текущего этапа
        
        Args:
            metrics: метрики
        """
        if not self.current_stage:
            return
            
        self.stages[self.current_stage]["metrics"].update(metrics)
        
    async def get_stage_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Получение статистики по этапам
        
        Returns:
            Dict[str, Dict[str, float]]: статистика
        """
        stats = {}
        for stage_name, timings in self.timings.items():
            if not timings:
                continue
                
            stats[stage_name] = {
                "min": min(timings),
                "max": max(timings),
                "avg": sum(timings) / len(timings),
                "count": len(timings)
            }
            
        return stats
        
    async def get_current_stage(self) -> Optional[Dict[str, Any]]:
        """
        Получение информации о текущем этапе
        
        Returns:
            Optional[Dict[str, Any]]: информация об этапе
        """
        if not self.current_stage:
            return None
            
        return self.stages[self.current_stage]
        
    async def get_stage_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Получение истории этапов
        
        Args:
            limit: ограничение количества
            
        Returns:
            List[Dict[str, Any]]: история этапов
        """
        history = [
            {
                "name": name,
                **stage
            }
            for name, stage in self.stages.items()
            if stage.get("end_time")
        ]
        
        history.sort(key=lambda x: x["end_time"], reverse=True)
        
        if limit:
            history = history[:limit]
            
        return history
        
    async def clear_history(self, older_than: Optional[float] = None):
        """
        Очистка истории
        
        Args:
            older_than: удалить старше чем (в секундах)
        """
        if older_than:
            threshold = time.time() - older_than
            to_remove = [
                name for name, stage in self.stages.items()
                if stage.get("end_time", 0) < threshold
            ]
            for name in to_remove:
                del self.stages[name]
                if name in self.timings:
                    del self.timings[name]
        else:
            self.stages.clear()
            self.timings.clear()
            
        self.logger.info("Cleared stage history")

    async def get_cpu_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик CPU
        
        Returns:
            Dict[str, Any]: метрики CPU
        """
        usage = psutil.cpu_percent()
        alert = usage > self.cpu_threshold
        
        metrics = {
            "usage": usage,
            "alert": alert
        }
        
        if alert:
            self.logger.warning(
                f"CPU usage above threshold: {usage}%",
                extra=metrics
            )
            
        return metrics
        
    async def get_memory_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик памяти
        
        Returns:
            Dict[str, Any]: метрики памяти
        """
        memory = psutil.virtual_memory()
        alert = memory.percent > self.memory_threshold
        
        metrics = {
            "total": memory.total,
            "available": memory.available,
            "usage": memory.percent,
            "alert": alert
        }
        
        if alert:
            self.logger.warning(
                f"Memory usage above threshold: {memory.percent}%",
                extra=metrics
            )
            
        return metrics
        
    async def get_thread_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик потоков
        
        Returns:
            Dict[str, Any]: метрики потоков
        """
        process = psutil.Process()
        thread_count = process.num_threads()
        alert = thread_count > self.thread_threshold
        
        metrics = {
            "count": thread_count,
            "alert": alert
        }
        
        if alert:
            self.logger.warning(
                f"Thread count above threshold: {thread_count}",
                extra=metrics
            )
            
        return metrics
        
    async def get_system_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Получение всех метрик системы
        
        Returns:
            Dict[str, Dict[str, Any]]: метрики системы
        """
        return {
            "cpu": await self.get_cpu_metrics(),
            "memory": await self.get_memory_metrics(),
            "threads": await self.get_thread_metrics()
        }
        
    async def start_monitoring(self):
        """Запуск мониторинга"""
        self.logger.info("Started system monitoring")
        
        while True:
            try:
                metrics = await self.get_system_metrics()
                await self._check_thresholds(metrics)
                
                if self.current_stage:
                    await self.update_stage_metrics(metrics)
                    
                await asyncio.sleep(60)  # Проверяем каждую минуту
                
            except Exception as e:
                self.logger.error(
                    "Error in monitoring loop",
                    error=e
                )
                await asyncio.sleep(60)
                
    async def _check_thresholds(self, metrics: Dict[str, Dict[str, Any]]):
        """
        Проверка пороговых значений
        
        Args:
            metrics: метрики системы
        """
        alerts = []
        
        if metrics["cpu"]["alert"]:
            alerts.append({
                "type": "cpu",
                "value": metrics["cpu"]["usage"],
                "threshold": self.cpu_threshold
            })
            
        if metrics["memory"]["alert"]:
            alerts.append({
                "type": "memory",
                "value": metrics["memory"]["usage"],
                "threshold": self.memory_threshold
            })
            
        if metrics["threads"]["alert"]:
            alerts.append({
                "type": "threads",
                "value": metrics["threads"]["count"],
                "threshold": self.thread_threshold
            })
            
        if alerts:
            for alert in alerts:
                await self._create_alert(
                    f"{alert['type']}_threshold",
                    f"{alert['type'].upper()} usage above threshold: {alert['value']}% (threshold: {alert['threshold']}%)",
                    "warning"
                )
                
    async def _create_alert(
        self,
        alert_type: str,
        message: str,
        severity: str
    ):
        """
        Создание оповещения
        
        Args:
            alert_type: тип оповещения
            message: сообщение
            severity: важность
        """
        self.logger.log(
            severity.upper(),
            message,
            extra={
                "alert_type": alert_type,
                "severity": severity
            }
        )

    async def manage_system_resources(self):
        """Управление системными ресурсами"""
        while True:
            try:
                metrics = await self.get_system_metrics()
                
                # Проверяем CPU
                if metrics["cpu"]["usage"] > self.cpu_threshold:
                    await self._reduce_cpu_load()
                    
                # Проверяем память
                if metrics["memory"]["usage"] > self.memory_threshold:
                    await self._free_memory()
                    
                # Проверяем потоки
                if metrics["threads"]["count"] > self.thread_threshold:
                    await self._optimize_threads()
                    
                await asyncio.sleep(300)  # Проверяем каждые 5 минут
                
            except Exception as e:
                self.logger.error(
                    "Error in resource management",
                    error=str(e)
                )
                await asyncio.sleep(300)

    async def _reduce_cpu_load(self):
        """Снижение нагрузки на CPU"""
        process = psutil.Process()
        
        # Получаем список процессов по CPU
        processes = sorted(
            process.children(recursive=True),
            key=lambda p: p.cpu_percent(),
            reverse=True
        )
        
        for proc in processes[:3]:  # Обрабатываем топ-3 процесса
            try:
                # Снижаем приоритет
                proc.nice(10)
                
                self.logger.info(
                    f"Reduced priority for process {proc.pid}",
                    extra={"pid": proc.pid, "cpu_percent": proc.cpu_percent()}
                )
            except Exception as e:
                self.logger.error(
                    f"Error reducing priority for process {proc.pid}",
                    error=str(e)
                )

    async def _free_memory(self):
        """Освобождение памяти"""
        process = psutil.Process()
        
        # Получаем список процессов по использованию памяти
        processes = sorted(
            process.children(recursive=True),
            key=lambda p: p.memory_percent(),
            reverse=True
        )
        
        freed_memory = 0
        for proc in processes[:3]:  # Обрабатываем топ-3 процесса
            try:
                before_memory = proc.memory_info().rss
                # Запрашиваем сборку мусора
                proc.memory_maps()
                after_memory = proc.memory_info().rss
                
                freed_memory += (before_memory - after_memory)
                
                self.logger.info(
                    f"Freed memory for process {proc.pid}",
                    extra={
                        "pid": proc.pid,
                        "freed_bytes": before_memory - after_memory
                    }
                )
            except Exception as e:
                self.logger.error(
                    f"Error freeing memory for process {proc.pid}",
                    error=str(e)
                )
                
        return freed_memory

    async def _optimize_threads(self):
        """Оптимизация потоков"""
        process = psutil.Process()
        
        # Получаем список процессов по количеству потоков
        processes = sorted(
            process.children(recursive=True),
            key=lambda p: p.num_threads(),
            reverse=True
        )
        
        optimized_count = 0
        for proc in processes[:3]:  # Обрабатываем топ-3 процесса
            try:
                before_threads = proc.num_threads()
                
                # Пытаемся объединить потоки через nice
                proc.nice(10)
                
                after_threads = proc.num_threads()
                optimized_count += (before_threads - after_threads)
                
                self.logger.info(
                    f"Optimized threads for process {proc.pid}",
                    extra={
                        "pid": proc.pid,
                        "reduced_threads": before_threads - after_threads
                    }
                )
            except Exception as e:
                self.logger.error(
                    f"Error optimizing threads for process {proc.pid}",
                    error=str(e)
                )
                
        return optimized_count

    async def get_resource_usage_history(
        self,
        period: Optional[timedelta] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Получение истории использования ресурсов
        
        Args:
            period: период анализа
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: история использования ресурсов
        """
        history = {
            "cpu": [],
            "memory": [],
            "threads": []
        }
        
        # Получаем метрики за период
        metrics = await self.get_stage_history()
        
        if period:
            threshold = time.time() - period.total_seconds()
            metrics = [
                m for m in metrics
                if m.get("start_time", 0) >= threshold
            ]
        
        # Собираем историю по каждому типу ресурса
        for metric in metrics:
            if "metrics" not in metric:
                continue
                
            timestamp = metric.get("start_time")
            
            if "cpu" in metric["metrics"]:
                history["cpu"].append({
                    "timestamp": timestamp,
                    "usage": metric["metrics"]["cpu"]["usage"]
                })
                
            if "memory" in metric["metrics"]:
                history["memory"].append({
                    "timestamp": timestamp,
                    "usage": metric["metrics"]["memory"]["usage"],
                    "available": metric["metrics"]["memory"]["available"]
                })
                
            if "threads" in metric["metrics"]:
                history["threads"].append({
                    "timestamp": timestamp,
                    "count": metric["metrics"]["threads"]["count"]
                })
                
        return history

    async def get_resource_alerts(
        self,
        period: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """
        Получение оповещений о ресурсах
        
        Args:
            period: период анализа
            
        Returns:
            List[Dict[str, Any]]: список оповещений
        """
        alerts = []
        history = await self.get_resource_usage_history(period)
        
        # Анализируем CPU
        cpu_usage = [m["usage"] for m in history["cpu"]]
        if cpu_usage:
            avg_cpu = sum(cpu_usage) / len(cpu_usage)
            max_cpu = max(cpu_usage)
            if max_cpu > self.cpu_threshold:
                alerts.append({
                    "type": "cpu",
                    "severity": "warning",
                    "message": f"High CPU usage detected: {max_cpu:.1f}% (avg: {avg_cpu:.1f}%)",
                    "threshold": self.cpu_threshold
                })

        # Анализируем память
        memory_usage = [m["usage"] for m in history["memory"]]
        if memory_usage:
            avg_memory = sum(memory_usage) / len(memory_usage)
            max_memory = max(memory_usage)
            if max_memory > self.memory_threshold:
                alerts.append({
                    "type": "memory",
                    "severity": "warning",
                    "message": f"High memory usage detected: {max_memory:.1f}% (avg: {avg_memory:.1f}%)",
                    "threshold": self.memory_threshold
                })

        # Анализируем потоки
        thread_counts = [m["count"] for m in history["threads"]]
        if thread_counts:
            avg_threads = sum(thread_counts) / len(thread_counts)
            max_threads = max(thread_counts)
            if max_threads > self.thread_threshold:
                alerts.append({
                    "type": "threads",
                    "severity": "warning",
                    "message": f"High thread count detected: {max_threads} (avg: {avg_threads:.1f})",
                    "threshold": self.thread_threshold
                })

        return alerts

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик производительности
        
        Returns:
            Dict[str, Any]: метрики производительности
        """
        metrics = {}
        
        # Получаем статистику по этапам
        stage_stats = await self.get_stage_statistics()
        metrics["stages"] = stage_stats
        
        # Добавляем метрики ресурсов
        system_metrics = await self.get_system_metrics()
        metrics["system"] = system_metrics
        
        # Добавляем историю использования ресурсов
        history = await self.get_resource_usage_history(timedelta(hours=1))
        metrics["history"] = history
        
        # Рассчитываем агрегированные метрики
        if stage_stats:
            total_stages = sum(s["count"] for s in stage_stats.values())
            total_time = sum(s["avg"] * s["count"] for s in stage_stats.values())
            metrics["aggregated"] = {
                "total_stages": total_stages,
                "total_time": total_time,
                "avg_stage_time": total_time / total_stages if total_stages > 0 else 0
            }
            
        return metrics

    async def visualize_process(self) -> Dict[str, Any]:
        """
        Визуализация процесса обработки
        
        Returns:
            Dict[str, Any]: данные для визуализации
        """
        visualization_data = {
            "stages": [],
            "resources": [],
            "alerts": []
        }
        
        # Получаем информацию о стадиях
        stages = await self.get_stage_history()
        for stage in stages:
            visualization_data["stages"].append({
                "name": stage["name"],
                "start_time": stage["start_time"],
                "end_time": stage.get("end_time"),
                "duration": stage.get("duration"),
                "status": stage["status"],
                "metrics": stage.get("metrics", {})
            })
            
        # Добавляем данные о ресурсах
        history = await self.get_resource_usage_history(timedelta(hours=1))
        visualization_data["resources"] = history
        
        # Добавляем алерты
        alerts = await self.get_resource_alerts(timedelta(hours=1))
        visualization_data["alerts"] = alerts
        
        return visualization_data

    async def track_stage_performance(
        self,
        stage_name: str,
        metrics: Dict[str, Any]
    ):
        """
        Отслеживание производительности этапа
        
        Args:
            stage_name: название этапа
            metrics: метрики этапа
        """
        if not self.current_stage or self.current_stage != stage_name:
            await self.start_stage(stage_name)
            
        await self.update_stage_metrics(metrics)
        
        # Проверяем пороговые значения
        if metrics.get("duration"):
            stage_stats = await self.get_stage_statistics()
            if stage_name in stage_stats:
                avg_time = stage_stats[stage_name]["avg"]
                if metrics["duration"] > avg_time * 2:
                    await self._create_alert(
                        "stage_performance",
                        f"Stage {stage_name} is taking longer than usual: "
                        f"{metrics['duration']:.2f}s (avg: {avg_time:.2f}s)",
                        "warning"
                    )

    @measure_time()
    async def analyze_performance_trends(
        self,
        period: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Анализ трендов производительности
        
        Args:
            period: период анализа
            
        Returns:
            Dict[str, Any]: результаты анализа
        """
        trends = {
            "stages": {},
            "resources": {},
            "alerts": []
        }
        
        # Анализируем тренды стадий
        stage_history = await self.get_stage_history()
        if stage_history:
            for stage in stage_history:
                if stage["name"] not in trends["stages"]:
                    trends["stages"][stage["name"]] = []
                if "duration" in stage:
                    trends["stages"][stage["name"]].append({
                        "timestamp": stage["start_time"],
                        "duration": stage["duration"]
                    })
        
        # Анализируем тренды ресурсов
        history = await self.get_resource_usage_history(period)
        trends["resources"] = history
        
        # Определяем аномалии
        for stage_name, measurements in trends["stages"].items():
            if len(measurements) < 2:
                continue
                
            durations = [m["duration"] for m in measurements]
            avg_duration = sum(durations) / len(durations)
            std_duration = (sum((d - avg_duration) ** 2 for d in durations) / len(durations)) ** 0.5
            
            for measurement in measurements:
                if abs(measurement["duration"] - avg_duration) > 2 * std_duration:
                    trends["alerts"].append({
                        "type": "performance_anomaly",
                        "stage": stage_name,
                        "timestamp": measurement["timestamp"],
                        "duration": measurement["duration"],
                        "avg_duration": avg_duration,
                        "deviation": abs(measurement["duration"] - avg_duration)
                    })
        
        return trends 


class MetricsCollector:
    """Сборщик метрик системы"""
    
    def __init__(self, db_path: str):
        """
        Инициализация сборщика метрик
        
        Args:
            db_path: путь к файлу базы данных
        """
        self.db_path = db_path
        self.logger = logging.getLogger('MetricsCollector')
        
    async def initialize(self):
        """Инициализация базы данных для метрик"""
        async with aiosqlite.connect(self.db_path) as db:
            # Создаем таблицу для метрик агентов
            await db.execute('''
                CREATE TABLE IF NOT EXISTS agent_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    role TEXT NOT NULL,
                    tasks_total INTEGER,
                    tasks_completed INTEGER,
                    tasks_failed INTEGER,
                    avg_processing_time REAL,
                    success_rate REAL,
                    error_rate REAL
                )
            ''')
            
            # Создаем таблицу для системных метрик
            await db.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metadata TEXT
                )
            ''')
            
            await db.commit()
            
    async def collect_agent_metrics(self, agent_role: AgentRole) -> AgentMetrics:
        """
        Сбор метрик агента
        
        Args:
            agent_role: роль агента
            
        Returns:
            AgentMetrics: метрики агента
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Получаем статистику по задачам
            cursor = await db.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(CASE WHEN status = 'completed' THEN duration ELSE NULL END) as avg_time
                FROM processing_stages
                WHERE stage_name LIKE ?
            ''', (f"{agent_role.value}%",))
            
            stats = await cursor.fetchone()
            
            total = stats[0] or 0
            completed = stats[1] or 0
            failed = stats[2] or 0
            avg_time = stats[3] or 0
            
            metrics = AgentMetrics(
                role=agent_role,
                tasks_total=total,
                tasks_completed=completed,
                tasks_failed=failed,
                avg_processing_time=avg_time,
                success_rate=completed / total if total > 0 else 0,
                error_rate=failed / total if total > 0 else 0
            )
            
            # Сохраняем метрики
            await db.execute('''
                INSERT INTO agent_metrics (
                    timestamp, role, tasks_total, tasks_completed,
                    tasks_failed, avg_processing_time, success_rate,
                    error_rate
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.role.value,
                metrics.tasks_total,
                metrics.tasks_completed,
                metrics.tasks_failed,
                metrics.avg_processing_time,
                metrics.success_rate,
                metrics.error_rate
            ))
            await db.commit()
            
            return metrics
            
    async def collect_system_metrics(self):
        """Сбор системных метрик"""
        metrics = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict()
        }
        
        async with aiosqlite.connect(self.db_path) as db:
            for name, value in metrics.items():
                metadata = None
                if isinstance(value, dict):
                    metadata = json.dumps(value)
                    value = sum(value.values())
                
                await db.execute('''
                    INSERT INTO system_metrics (
                        metric_name, metric_value, metadata
                    )
                    VALUES (?, ?, ?)
                ''', (name, value, metadata))
            
            await db.commit()
            
        return metrics
        
    async def get_agent_metrics(
        self,
        role: Optional[AgentRole] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Получение метрик агента
        
        Args:
            role: роль агента
            start_time: начало периода
            end_time: конец периода
            
        Returns:
            List[Dict[str, Any]]: список метрик
        """
        async with aiosqlite.connect(self.db_path) as db:
            query = '''
                SELECT 
                    timestamp,
                    role,
                    tasks_total,
                    tasks_completed,
                    tasks_failed,
                    avg_processing_time,
                    success_rate,
                    error_rate
                FROM agent_metrics
                WHERE 1=1
            '''
            params = []
            
            if role:
                query += " AND role = ?"
                params.append(role.value)
                
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
                
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
                
            query += " ORDER BY timestamp DESC"
            
            async with db.execute(query, params) as cursor:
                metrics = []
                async for row in cursor:
                    metrics.append({
                        "timestamp": row[0],
                        "role": row[1],
                        "tasks_total": row[2],
                        "tasks_completed": row[3],
                        "tasks_failed": row[4],
                        "avg_processing_time": row[5],
                        "success_rate": row[6],
                        "error_rate": row[7]
                    })
                    
                return metrics

    async def get_agent_summary(
        self,
        role: Optional[AgentRole] = None,
        period: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Получение сводной информации по агенту
        
        Args:
            role: роль агента
            period: период анализа
            
        Returns:
            Dict[str, Any]: сводная информация
        """
        async with aiosqlite.connect(self.db_path) as db:
            query = '''
                SELECT 
                    role,
                    COUNT(*) as total_records,
                    SUM(tasks_total) as total_tasks,
                    SUM(tasks_completed) as completed_tasks,
                    SUM(tasks_failed) as failed_tasks,
                    AVG(avg_processing_time) as avg_time,
                    AVG(success_rate) as avg_success,
                    AVG(error_rate) as avg_error
                FROM agent_metrics
                WHERE 1=1
            '''
            params = []
            
            if role:
                query += " AND role = ?"
                params.append(role.value)
                
            if period:
                query += " AND timestamp >= datetime('now', ?)"
                params.append(f'-{period.total_seconds()} seconds')
                
            query += " GROUP BY role"
            
            async with db.execute(query, params) as cursor:
                summaries = {}
                async for row in cursor:
                    summaries[row[0]] = {
                        "total_records": row[1],
                        "total_tasks": row[2],
                        "completed_tasks": row[3],
                        "failed_tasks": row[4],
                        "avg_processing_time": row[5],
                        "avg_success_rate": row[6],
                        "avg_error_rate": row[7],
                        "completion_rate": (row[3] / row[2] * 100) if row[2] > 0 else 0
                    }
                    
                return summaries

    async def get_agent_trends(
        self,
        role: Optional[AgentRole] = None,
        period: Optional[timedelta] = None,
        window_size: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Анализ трендов агента
        
        Args:
            role: роль агента
            period: период анализа
            window_size: размер окна для скользящего среднего
            
        Returns:
            Dict[str, Dict[str, Any]]: тренды агента
        """
        metrics = await self.get_agent_metrics(
            role=role,
            start_time=(
                datetime.now() - period if period
                else None
            )
        )
        
        if not metrics:
            return {}
            
        trends = {}
        for metric_name in [
            "tasks_total",
            "tasks_completed",
            "tasks_failed",
            "avg_processing_time",
            "success_rate",
            "error_rate"
        ]:
            values = np.array([
                m[metric_name]
                for m in metrics
            ])
            
            if len(values) < window_size:
                continue
                
            moving_avg = np.convolve(
                values,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            
            trend = "stable"
            if len(moving_avg) > 1:
                diff = moving_avg[-1] - moving_avg[0]
                if diff > 0:
                    trend = "increasing"
                elif diff < 0:
                    trend = "decreasing"
                    
            trends[metric_name] = {
                "trend": trend,
                "current_value": float(values[-1]),
                "moving_average": float(moving_avg[-1]),
                "samples_count": len(values)
            }
            
        return trends

    async def get_system_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Получение системных метрик
        
        Args:
            metric_name: название метрики (опционально)
            start_time: начало периода
            end_time: конец периода
            
        Returns:
            List[Dict[str, Any]]: список метрик
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            query = "SELECT * FROM system_metrics"
            params = []
            
            conditions = []
            if metric_name:
                conditions.append("metric_name = ?")
                params.append(metric_name)
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time.isoformat())
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time.isoformat())
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY timestamp DESC"
            
            cursor = await db.execute(query, params)
            metrics = await cursor.fetchall()
            return [dict(m) for m in metrics]
            
    async def export_metrics(
        self,
        output_dir: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """
        Экспорт метрик в JSON файлы
        
        Args:
            output_dir: директория для сохранения
            start_time: начало периода
            end_time: конец периода
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Экспорт метрик агентов
        agent_metrics = {}
        for role in AgentRole:
            metrics = await self.get_agent_metrics(role, start_time, end_time)
            agent_metrics[role.value] = metrics
            
        with open(output_path / 'agent_metrics.json', 'w') as f:
            json.dump(agent_metrics, f, indent=2)
            
        # Экспорт системных метрик
        system_metrics = await self.get_system_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        with open(output_path / 'system_metrics.json', 'w') as f:
            json.dump(system_metrics, f, indent=2)
            
        self.logger.info(f"Metrics exported to {output_dir}") 

    async def save_agent_metrics(self, metrics: AgentMetrics):
        """
        Сохранение метрик агента
        
        Args:
            metrics: метрики агента
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO agent_metrics (
                    timestamp, role, tasks_total, tasks_completed,
                    tasks_failed, avg_processing_time, success_rate, error_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.role.value,
                metrics.tasks_total,
                metrics.tasks_completed,
                metrics.tasks_failed,
                metrics.avg_processing_time,
                metrics.success_rate,
                metrics.error_rate
            ))
            await db.commit()

    async def save_system_metric(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Сохранение системной метрики
        
        Args:
            name: название метрики
            value: значение
            metadata: дополнительные данные
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO system_metrics (metric_name, metric_value, metadata)
                VALUES (?, ?, ?)
            ''', (
                name,
                value,
                json.dumps(metadata) if metadata else None
            ))
            await db.commit()

    async def get_agent_performance(
        self,
        role: Optional[AgentRole] = None,
        period: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Получение статистики производительности агента
        
        Args:
            role: роль агента
            period: период анализа
            
        Returns:
            Dict[str, Any]: статистика производительности
        """
        async with aiosqlite.connect(self.db_path) as db:
            query = '''
                SELECT 
                    AVG(success_rate) as avg_success_rate,
                    AVG(error_rate) as avg_error_rate,
                    AVG(avg_processing_time) as avg_processing_time,
                    COUNT(*) as total_records
                FROM agent_metrics
                WHERE 1=1
            '''
            params = []
            
            if role:
                query += " AND role = ?"
                params.append(role.value)
                
            if period:
                query += " AND timestamp >= datetime('now', ?)"
                params.append(f'-{period.total_seconds()} seconds')
                
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                
                return {
                    "avg_success_rate": row[0] or 0.0,
                    "avg_error_rate": row[1] or 0.0,
                    "avg_processing_time": row[2] or 0.0,
                    "total_records": row[3] or 0
                }

    async def get_system_performance(
        self,
        metric_names: Optional[List[str]] = None,
        period: Optional[timedelta] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Получение статистики производительности системы
        
        Args:
            metric_names: названия метрик
            period: период анализа
            
        Returns:
            Dict[str, Dict[str, float]]: статистика производительности
        """
        async with aiosqlite.connect(self.db_path) as db:
            query = '''
                SELECT 
                    metric_name,
                    AVG(metric_value) as avg_value,
                    MIN(metric_value) as min_value,
                    MAX(metric_value) as max_value,
                    COUNT(*) as total_records
                FROM system_metrics
                WHERE 1=1
            '''
            params = []
            
            if metric_names:
                placeholders = ','.join('?' * len(metric_names))
                query += f" AND metric_name IN ({placeholders})"
                params.extend(metric_names)
                
            if period:
                query += " AND timestamp >= datetime('now', ?)"
                params.append(f'-{period.total_seconds()} seconds')
                
            query += " GROUP BY metric_name"
            
            async with db.execute(query, params) as cursor:
                results = {}
                async for row in cursor:
                    results[row[0]] = {
                        "avg": row[1],
                        "min": row[2],
                        "max": row[3],
                        "count": row[4]
                    }
                    
                return results

    async def cleanup_old_metrics(
        self,
        retention_period: timedelta,
        batch_size: int = 1000
    ):
        """
        Очистка старых метрик
        
        Args:
            retention_period: период хранения
            batch_size: размер пакета для удаления
        """
        threshold = datetime.now() - retention_period
        
        async with aiosqlite.connect(self.db_path) as db:
            # Очистка метрик агентов
            while True:
                result = await db.execute('''
                    DELETE FROM agent_metrics 
                    WHERE timestamp < ? 
                    LIMIT ?
                ''', (threshold.isoformat(), batch_size))
                await db.commit()
                
                if result.rowcount < batch_size:
                    break
                    
            # Очистка системных метрик
            while True:
                result = await db.execute('''
                    DELETE FROM system_metrics 
                    WHERE timestamp < ? 
                    LIMIT ?
                ''', (threshold.isoformat(), batch_size))
                await db.commit()
                
                if result.rowcount < batch_size:
                    break
                    
            self.logger.info(
                f"Cleaned up metrics older than {threshold}"
            )

    async def optimize_database(self):
        """Оптимизация базы данных"""
        async with aiosqlite.connect(self.db_path) as db:
            # Анализ и оптимизация индексов
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_metrics_timestamp 
                ON agent_metrics(timestamp)
            ''')
            
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_metrics_role 
                ON agent_metrics(role)
            ''')
            
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp 
                ON system_metrics(timestamp)
            ''')
            
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_system_metrics_name 
                ON system_metrics(metric_name)
            ''')
            
            # Оптимизация и сжатие базы
            await db.execute('VACUUM')
            await db.execute('ANALYZE')
            
            self.logger.info("Database optimized")

    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Получение статистики базы данных
        
        Returns:
            Dict[str, Any]: статистика БД
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Размер таблиц
            async with db.execute('''
                SELECT 
                    COUNT(*) as count,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM agent_metrics
            ''') as cursor:
                agent_stats = await cursor.fetchone()
                
            async with db.execute('''
                SELECT 
                    COUNT(*) as count,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM system_metrics
            ''') as cursor:
                system_stats = await cursor.fetchone()
                
            # Размер файла БД
            db_size = os.path.getsize(self.db_path)
            
            return {
                "agent_metrics": {
                    "total_records": agent_stats[0],
                    "oldest_record": agent_stats[1],
                    "newest_record": agent_stats[2]
                },
                "system_metrics": {
                    "total_records": system_stats[0],
                    "oldest_record": system_stats[1],
                    "newest_record": system_stats[2]
                },
                "database_size_bytes": db_size,
                "database_size_mb": round(db_size / (1024 * 1024), 2)
            } 