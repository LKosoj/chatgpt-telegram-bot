"""
Модуль для логирования действий системы
"""

import logging
import json
import sys
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Generator
from logging.handlers import RotatingFileHandler
import aiofiles
from .models import AgentRole


class StructuredFormatter(logging.Formatter):
    """Форматтер для структурированных логов"""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Форматирование записи лога
        
        Args:
            record: запись лога
            
        Returns:
            str: отформатированная запись
        """
        # Базовые поля
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        # Добавляем extra данные
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra
            
        # Добавляем информацию об исключении
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
            
        return json.dumps(log_data, ensure_ascii=False)


class AgentLogger:
    """Логгер для агентов системы"""
    
    def __init__(
        self,
        log_dir: str,
        agent_role: Optional[AgentRole] = None,
        max_bytes: int = 10_485_760,  # 10MB
        backup_count: int = 5,
        structured: bool = True
    ):
        """
        Инициализация логгера
        
        Args:
            log_dir: директория для логов
            agent_role: роль агента (опционально)
            max_bytes: максимальный размер файла лога
            backup_count: количество файлов бэкапа
            structured: использовать структурированные логи
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent_role = agent_role
        self.structured = structured
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """
        Настройка логгера
        
        Returns:
            logging.Logger: настроенный логгер
        """
        logger_name = (
            f"Agent_{self.agent_role.value}"
            if self.agent_role
            else "System"
        )
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Файловый обработчик
            log_file = self.log_dir / f"{logger_name.lower()}.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            
            # Выбираем форматтер
            if self.structured:
                formatter = StructuredFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Консольный обработчик
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
        return logger
        
    def _format_message(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Форматирование сообщения для лога
        
        Args:
            message: сообщение
            extra: дополнительные данные
            
        Returns:
            str: отформатированное сообщение
        """
        if not extra:
            return message
            
        try:
            extra_str = json.dumps(extra, ensure_ascii=False)
            return f"{message} | Extra: {extra_str}"
        except Exception as e:
            self.logger.warning(f"Error formatting extra data: {str(e)}")
            return message
            
    def info(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Логирование информационного сообщения
        
        Args:
            message: сообщение
            extra: дополнительные данные
        """
        self.logger.info(self._format_message(message, extra))
        
    def warning(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Логирование предупреждения
        
        Args:
            message: сообщение
            extra: дополнительные данные
        """
        self.logger.warning(self._format_message(message, extra))
        
    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Логирование ошибки
        
        Args:
            message: сообщение
            error: объект исключения
            extra: дополнительные данные
        """
        if error:
            message = f"{message} | Error: {str(error)}"
        self.logger.error(self._format_message(message, extra))
        
    def debug(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Логирование отладочного сообщения
        
        Args:
            message: сообщение
            extra: дополнительные данные
        """
        self.logger.debug(self._format_message(message, extra))
        
    def critical(
        self,
        message: str,
        error: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Логирование критической ошибки
        
        Args:
            message: сообщение
            error: объект исключения
            extra: дополнительные данные
        """
        if error:
            message = f"{message} | Error: {str(error)}"
        self.logger.critical(self._format_message(message, extra))
        
    def log_task_start(
        self,
        task_id: str,
        task_type: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Логирование начала задачи
        
        Args:
            task_id: идентификатор задачи
            task_type: тип задачи
            extra: дополнительные данные
        """
        message = f"Started task {task_id} of type {task_type}"
        self.info(message, extra)
        
    def log_task_end(
        self,
        task_id: str,
        status: str,
        duration: float,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Логирование завершения задачи
        
        Args:
            task_id: идентификатор задачи
            status: статус завершения
            duration: длительность выполнения
            extra: дополнительные данные
        """
        message = (
            f"Completed task {task_id} with status {status} "
            f"(duration: {duration:.2f}s)"
        )
        self.info(message, extra)
        
    def log_agent_action(
        self,
        action: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Логирование действия агента
        
        Args:
            action: действие
            status: статус
            details: детали действия
        """
        message = f"Agent action: {action} - Status: {status}"
        self.info(message, details)
        
    def log_system_event(
        self,
        event_type: str,
        description: str,
        severity: str = "info",
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Логирование системного события
        
        Args:
            event_type: тип события
            description: описание
            severity: важность
            extra: дополнительные данные
        """
        message = f"System event: {event_type} - {description}"
        
        if severity == "error":
            self.error(message, extra=extra)
        elif severity == "warning":
            self.warning(message, extra=extra)
        elif severity == "critical":
            self.critical(message, extra=extra)
        else:
            self.info(message, extra=extra)
            
    def log_metric(
        self,
        metric_name: str,
        value: Union[int, float, str],
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Логирование метрики
        
        Args:
            metric_name: название метрики
            value: значение
            extra: дополнительные данные
        """
        message = f"Metric: {metric_name} = {value}"
        self.info(message, extra)
        
    def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any]
    ):
        """
        Логирование ошибки с контекстом
        
        Args:
            error: объект исключения
            context: контекст ошибки
        """
        self.error(
            "Error occurred with context",
            error=error,
            extra=context
        )
        
    async def search_logs(
        self,
        query: str,
        log_file: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Поиск по логам
        
        Args:
            query: поисковый запрос
            log_file: имя файла лога (опционально)
            start_time: начало периода
            end_time: конец периода
            level: уровень логирования
            limit: максимальное количество результатов
            
        Returns:
            List[Dict[str, Any]]: найденные записи
        """
        results = []
        
        # Определяем файлы для поиска
        if log_file:
            log_files = [self.log_dir / log_file]
        else:
            log_files = list(self.log_dir.glob("*.log"))
            
        # Компилируем регулярное выражение
        pattern = re.compile(query, re.IGNORECASE)
        
        # Ищем в каждом файле
        for file_path in log_files:
            if not file_path.exists():
                continue
                
            async with aiofiles.open(file_path, "r") as f:
                async for line in f:
                    # Парсим запись
                    try:
                        if self.structured:
                            record = json.loads(line)
                        else:
                            # Парсим обычный формат
                            parts = line.split(" - ", 3)
                            if len(parts) != 4:
                                continue
                                
                            record = {
                                "timestamp": parts[0],
                                "logger": parts[1],
                                "level": parts[2],
                                "message": parts[3].strip()
                            }
                            
                        # Проверяем временной период
                        if start_time or end_time:
                            timestamp = datetime.fromisoformat(
                                record["timestamp"].split(".")[0]
                            )
                            if start_time and timestamp < start_time:
                                continue
                            if end_time and timestamp > end_time:
                                continue
                                
                        # Проверяем уровень
                        if level and record["level"] != level:
                            continue
                            
                        # Проверяем совпадение
                        if pattern.search(record["message"]):
                            results.append(record)
                            if len(results) >= limit:
                                return results
                                
                    except Exception as e:
                        self.error(
                            "Error parsing log record",
                            error=e,
                            extra={"line": line}
                        )
                        
        return results
        
    async def export_logs(
        self,
        output_dir: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[str] = None,
        format: str = "json"
    ):
        """
        Экспорт логов
        
        Args:
            output_dir: директория для экспорта
            start_time: начало периода
            end_time: конец периода
            level: уровень логирования
            format: формат экспорта (json или csv)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Собираем логи из всех файлов
        logs = []
        for log_file in self.log_dir.glob("*.log"):
            async with aiofiles.open(log_file, "r") as f:
                async for line in f:
                    try:
                        if self.structured:
                            record = json.loads(line)
                        else:
                            # Парсим обычный формат
                            parts = line.split(" - ", 3)
                            if len(parts) != 4:
                                continue
                                
                            record = {
                                "timestamp": parts[0],
                                "logger": parts[1],
                                "level": parts[2],
                                "message": parts[3].strip()
                            }
                            
                        # Проверяем временной период
                        if start_time or end_time:
                            timestamp = datetime.fromisoformat(
                                record["timestamp"].split(".")[0]
                            )
                            if start_time and timestamp < start_time:
                                continue
                            if end_time and timestamp > end_time:
                                continue
                                
                        # Проверяем уровень
                        if level and record["level"] != level:
                            continue
                            
                        logs.append(record)
                        
                    except Exception as e:
                        self.error(
                            "Error parsing log record",
                            error=e,
                            extra={"line": line}
                        )
                        
        # Экспортируем в выбранном формате
        if format == "json":
            output_file = output_path / "logs.json"
            async with aiofiles.open(output_file, "w") as f:
                await f.write(json.dumps(logs, indent=2))
        else:
            output_file = output_path / "logs.csv"
            async with aiofiles.open(output_file, "w") as f:
                # Записываем заголовок
                headers = ["timestamp", "level", "logger", "message"]
                await f.write(",".join(headers) + "\n")
                
                # Записываем данные
                for record in logs:
                    values = [
                        record["timestamp"],
                        record["level"],
                        record["logger"],
                        record["message"].replace(",", ";")
                    ]
                    await f.write(",".join(values) + "\n")
                    
        self.info(
            f"Logs exported to {output_file}",
            extra={
                "format": format,
                "records_count": len(logs)
            }
        )
        
    async def rotate_logs(self):
        """Принудительная ротация логов"""
        for handler in self.logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                handler.doRollover()
                
        self.info("Log files rotated") 