"""
Модуль с константами системы
"""

from enum import Enum, auto
from typing import Dict, Any


# Общие константы
VERSION = "1.0.0"
DEFAULT_ENCODING = "utf-8"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1


# Пути и директории
class Paths:
    """Пути к файлам и директориям"""
    CONFIG_DIR = "config"
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    CACHE_DIR = "cache"
    MODELS_DIR = "models"
    CONFIG_FILE = "ai_agent.json"
    DATABASE_FILE = "ai_agent.db"


# Размеры и лимиты
class Limits:
    """Размеры и лимиты"""
    MAX_MESSAGE_LENGTH = 4096
    MAX_TASKS = 100
    MAX_CACHE_SIZE = 1024 * 1024 * 100  # 100MB
    MAX_LOG_SIZE = 1024 * 1024 * 10  # 10MB
    MAX_BATCH_SIZE = 32
    MAX_SEQUENCE_LENGTH = 512
    MAX_CONCURRENT_TASKS = 10


# Временные интервалы
class Intervals:
    """Временные интервалы"""
    CACHE_TTL = 3600  # 1 час
    TASK_TIMEOUT = 300  # 5 минут
    CLEANUP_INTERVAL = 86400  # 1 день
    METRICS_INTERVAL = 60  # 1 минута
    RATE_LIMIT_PERIOD = 60  # 1 минута


# Приоритеты
class Priority(int, Enum):
    """Приоритеты"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


# Статусы
class Status(str, Enum):
    """Статусы"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    ERROR = "error"


# Типы ресурсов
class ResourceType(str, Enum):
    """Типы ресурсов"""
    TASK = "task"
    RESEARCH = "research"
    PLAN = "plan"
    ACTION = "action"
    MODEL = "model"
    CACHE = "cache"
    LOG = "log"
    CONFIG = "config"


# Уровни логирования
class LogLevel(str, Enum):
    """Уровни логирования"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# HTTP методы
class HttpMethod(str, Enum):
    """HTTP методы"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


# Типы контента
class ContentType(str, Enum):
    """Типы контента"""
    JSON = "application/json"
    TEXT = "text/plain"
    HTML = "text/html"
    XML = "application/xml"
    FORM = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"


# HTTP заголовки
class HttpHeaders:
    """HTTP заголовки"""
    CONTENT_TYPE = "Content-Type"
    AUTHORIZATION = "Authorization"
    USER_AGENT = "User-Agent"
    ACCEPT = "Accept"
    ACCEPT_ENCODING = "Accept-Encoding"
    CONNECTION = "Connection"
    CACHE_CONTROL = "Cache-Control"


# HTTP коды ответов
class HttpStatus:
    """HTTP коды ответов"""
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    TIMEOUT = 408
    TOO_MANY_REQUESTS = 429
    SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503


# Типы событий
class EventType(str, Enum):
    """Типы событий"""
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    RESEARCH_STARTED = "research.started"
    RESEARCH_COMPLETED = "research.completed"
    PLAN_CREATED = "plan.created"
    PLAN_UPDATED = "plan.updated"
    ACTION_EXECUTED = "action.executed"
    ERROR_OCCURRED = "error.occurred"
    METRIC_RECORDED = "metric.recorded"
    CACHE_UPDATED = "cache.updated"
    CONFIG_CHANGED = "config.changed"


# Типы метрик
class MetricType(str, Enum):
    """Типы метрик"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


# Настройки по умолчанию
DEFAULT_CONFIG: Dict[str, Any] = {
    "version": VERSION,
    "encoding": DEFAULT_ENCODING,
    "timeout": DEFAULT_TIMEOUT,
    "max_retries": MAX_RETRIES,
    "retry_delay": RETRY_DELAY,
    "paths": {
        "config_dir": Paths.CONFIG_DIR,
        "data_dir": Paths.DATA_DIR,
        "logs_dir": Paths.LOGS_DIR,
        "cache_dir": Paths.CACHE_DIR,
        "models_dir": Paths.MODELS_DIR,
        "config_file": Paths.CONFIG_FILE,
        "database_file": Paths.DATABASE_FILE
    },
    "limits": {
        "max_message_length": Limits.MAX_MESSAGE_LENGTH,
        "max_tasks": Limits.MAX_TASKS,
        "max_cache_size": Limits.MAX_CACHE_SIZE,
        "max_log_size": Limits.MAX_LOG_SIZE,
        "max_batch_size": Limits.MAX_BATCH_SIZE,
        "max_sequence_length": Limits.MAX_SEQUENCE_LENGTH,
        "max_concurrent_tasks": Limits.MAX_CONCURRENT_TASKS
    },
    "intervals": {
        "cache_ttl": Intervals.CACHE_TTL,
        "task_timeout": Intervals.TASK_TIMEOUT,
        "cleanup_interval": Intervals.CLEANUP_INTERVAL,
        "metrics_interval": Intervals.METRICS_INTERVAL,
        "rate_limit_period": Intervals.RATE_LIMIT_PERIOD
    },
    "logging": {
        "level": LogLevel.INFO,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S"
    },
    "http": {
        "timeout": DEFAULT_TIMEOUT,
        "max_retries": MAX_RETRIES,
        "retry_delay": RETRY_DELAY,
        "headers": {
            HttpHeaders.CONTENT_TYPE: ContentType.JSON,
            HttpHeaders.USER_AGENT: f"AIAgent/{VERSION}",
            HttpHeaders.ACCEPT: ContentType.JSON,
            HttpHeaders.ACCEPT_ENCODING: "gzip, deflate",
            HttpHeaders.CONNECTION: "keep-alive",
            HttpHeaders.CACHE_CONTROL: "no-cache"
        }
    }
} 