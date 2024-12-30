"""
Модуль с декораторами
"""

import asyncio
import functools
import time
from typing import Any, Callable, Optional, Type, TypeVar, Dict
from .exceptions import (
    TimeoutError,
    RetryError,
    ValidationError,
    AIAgentError
)
from .logging import AgentLogger


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
    logger: Optional[AgentLogger] = None
) -> Callable[[F], F]:
    """
    Декоратор для повторных попыток
    
    Args:
        max_attempts: максимальное количество попыток
        delay: задержка между попытками
        exceptions: перехватываемые исключения
        logger: логгер
        
    Returns:
        Callable: декорированная функция
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logger or AgentLogger("logs")
            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        _logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {delay}s",
                            extra={
                                "error": str(e),
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts
                            }
                        )
                        await asyncio.sleep(delay)
                    else:
                        _logger.error(
                            f"All {max_attempts} attempts failed",
                            error=e
                        )
                        
            raise RetryError(
                f"Failed after {max_attempts} attempts",
                attempts=max_attempts,
                last_error=last_error
            )
            
        return wrapper  # type: ignore
        
    return decorator


def timeout(
    seconds: float,
    logger: Optional[AgentLogger] = None
) -> Callable[[F], F]:
    """
    Декоратор для таймаута
    
    Args:
        seconds: таймаут в секундах
        logger: логгер
        
    Returns:
        Callable: декорированная функция
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logger or AgentLogger("logs")
            
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                _logger.error(
                    f"Operation timed out after {seconds}s"
                )
                raise TimeoutError(
                    f"Operation timed out after {seconds}s",
                    timeout=seconds
                )
                
        return wrapper  # type: ignore
        
    return decorator


def validate_args(**validators: Callable[[Any], bool]) -> Callable[[F], F]:
    """
    Декоратор для валидации аргументов
    
    Args:
        **validators: функции валидации
        
    Returns:
        Callable: декорированная функция
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Получаем имена аргументов
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            
            # Объединяем позиционные и именованные аргументы
            all_args = dict(zip(arg_names, args))
            all_args.update(kwargs)
            
            # Проверяем каждый аргумент
            for arg_name, validator in validators.items():
                if arg_name in all_args:
                    value = all_args[arg_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Invalid value for argument '{arg_name}'",
                            field=arg_name,
                            value=value
                        )
                        
            return await func(*args, **kwargs)
            
        return wrapper  # type: ignore
        
    return decorator


def log_calls(
    logger: Optional[AgentLogger] = None,
    level: str = "INFO"
) -> Callable[[F], F]:
    """
    Декоратор для логирования вызовов
    
    Args:
        logger: логгер
        level: уровень логирования
        
    Returns:
        Callable: декорированная функция
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logger or AgentLogger("logs")
            
            # Логируем начало вызова
            _logger.log(
                level,
                f"Calling {func.__name__}",
                extra={
                    "args": args,
                    "kwargs": kwargs
                }
            )
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                
                # Логируем успешное завершение
                _logger.log(
                    level,
                    f"Finished {func.__name__}",
                    extra={
                        "duration": time.time() - start_time,
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                # Логируем ошибку
                _logger.error(
                    f"Error in {func.__name__}",
                    error=e,
                    extra={
                        "duration": time.time() - start_time,
                        "success": False
                    }
                )
                raise
                
        return wrapper  # type: ignore
        
    return decorator


def cache_result(
    ttl: Optional[int] = None,
    key_builder: Optional[Callable[..., str]] = None
) -> Callable[[F], F]:
    """
    Декоратор для кэширования результатов
    
    Args:
        ttl: время жизни кэша в секундах
        key_builder: функция построения ключа
        
    Returns:
        Callable: декорированная функция
    """
    cache: Dict[str, tuple[Any, float]] = {}
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Формируем ключ кэша
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
            # Проверяем кэш
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if not ttl or time.time() - timestamp < ttl:
                    return result
                    
            # Вычисляем результат
            result = await func(*args, **kwargs)
            
            # Сохраняем в кэш
            cache[cache_key] = (result, time.time())
            
            return result
            
        return wrapper  # type: ignore
        
    return decorator


def handle_errors(
    *error_types: Type[Exception],
    logger: Optional[AgentLogger] = None,
    default: Any = None
) -> Callable[[F], F]:
    """
    Декоратор для обработки ошибок
    
    Args:
        *error_types: типы ошибок
        logger: логгер
        default: значение по умолчанию
        
    Returns:
        Callable: декорированная функция
    """
    if not error_types:
        error_types = (Exception,)
        
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logger or AgentLogger("logs")
            
            try:
                return await func(*args, **kwargs)
            except error_types as e:
                _logger.error(
                    f"Error in {func.__name__}",
                    error=e
                )
                return default
                
        return wrapper  # type: ignore
        
    return decorator


def async_task(func: F) -> F:
    """
    Декоратор для асинхронных задач
    
    Args:
        func: функция
        
    Returns:
        Callable: декорированная функция
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.create_task(func(*args, **kwargs))
        
    return wrapper  # type: ignore


def measure_time(
    logger: Optional[AgentLogger] = None
) -> Callable[[F], F]:
    """
    Декоратор для измерения времени выполнения
    
    Args:
        logger: логгер
        
    Returns:
        Callable: декорированная функция
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logger or AgentLogger("logs")
            
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            _logger.info(
                f"{func.__name__} took {duration:.2f}s",
                extra={"duration": duration}
            )
            
            return result
            
        return wrapper  # type: ignore
        
    return decorator 