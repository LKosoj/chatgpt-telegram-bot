"""
Модуль с исключениями системы
"""

from typing import Optional, Dict, Any


class AIAgentError(Exception):
    """Базовое исключение системы"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация исключения
        
        Args:
            message: сообщение об ошибке
            details: детали ошибки
        """
        super().__init__(message)
        self.details = details or {}


class ConfigError(AIAgentError):
    """Ошибка конфигурации"""
    pass


class StorageError(AIAgentError):
    """Ошибка хранилища"""
    pass


class NLPError(AIAgentError):
    """Ошибка NLP"""
    pass


class AgentError(AIAgentError):
    """Ошибка агента"""
    pass


class TaskError(AIAgentError):
    """Ошибка задачи"""
    pass


class ValidationError(AIAgentError):
    """Ошибка валидации"""
    
    def __init__(
        self,
        message: str,
        field: str,
        value: Any,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация исключения
        
        Args:
            message: сообщение об ошибке
            field: поле с ошибкой
            value: некорректное значение
            details: детали ошибки
        """
        super().__init__(message, details)
        self.field = field
        self.value = value


class ResourceNotFoundError(AIAgentError):
    """Ошибка отсутствия ресурса"""
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация исключения
        
        Args:
            message: сообщение об ошибке
            resource_type: тип ресурса
            resource_id: идентификатор ресурса
            details: детали ошибки
        """
        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class APIError(AIAgentError):
    """Ошибка API"""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация исключения
        
        Args:
            message: сообщение об ошибке
            status_code: код ответа
            response: ответ сервера
            details: детали ошибки
        """
        super().__init__(message, details)
        self.status_code = status_code
        self.response = response


class RateLimitError(APIError):
    """Ошибка превышения лимита запросов"""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация исключения
        
        Args:
            message: сообщение об ошибке
            retry_after: время до следующей попытки
            details: детали ошибки
        """
        super().__init__(message, details=details)
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """Ошибка аутентификации"""
    pass


class AuthorizationError(APIError):
    """Ошибка авторизации"""
    pass


class TimeoutError(AIAgentError):
    """Ошибка таймаута"""
    
    def __init__(
        self,
        message: str,
        timeout: float,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация исключения
        
        Args:
            message: сообщение об ошибке
            timeout: таймаут в секундах
            details: детали ошибки
        """
        super().__init__(message, details)
        self.timeout = timeout


class RetryError(AIAgentError):
    """Ошибка повторных попыток"""
    
    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация исключения
        
        Args:
            message: сообщение об ошибке
            attempts: количество попыток
            last_error: последняя ошибка
            details: детали ошибки
        """
        super().__init__(message, details)
        self.attempts = attempts
        self.last_error = last_error


class CacheError(AIAgentError):
    """Ошибка кэша"""
    pass


class FileError(AIAgentError):
    """Ошибка файловой системы"""
    
    def __init__(
        self,
        message: str,
        path: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация исключения
        
        Args:
            message: сообщение об ошибке
            path: путь к файлу
            details: детали ошибки
        """
        super().__init__(message, details)
        self.path = path


class DatabaseError(AIAgentError):
    """Ошибка базы данных"""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация исключения
        
        Args:
            message: сообщение об ошибке
            query: SQL запрос
            params: параметры запроса
            details: детали ошибки
        """
        super().__init__(message, details)
        self.query = query
        self.params = params


class ModelError(AIAgentError):
    """Ошибка модели"""
    
    def __init__(
        self,
        message: str,
        model_name: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация исключения
        
        Args:
            message: сообщение об ошибке
            model_name: название модели
            details: детали ошибки
        """
        super().__init__(message, details)
        self.model_name = model_name 