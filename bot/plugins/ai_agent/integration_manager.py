"""
Модуль для управления внешними интеграциями AI агента
"""

import asyncio
import json
import os
import dataclasses
from typing import Dict, List, Any, Optional, Callable, Coroutine
from datetime import datetime
from aiohttp import ClientSession, ClientError
from cryptography.fernet import Fernet

from .logging import AgentLogger


@dataclasses.dataclass
class IntegrationConfig:
    """Конфигурация интеграции"""
    name: str
    api_url: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1
    rate_limit: Optional[int] = None
    enabled: bool = True


@dataclasses.dataclass
class IntegrationMetrics:
    """Метрики интеграции"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_errors: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None


class Integration:
    """Базовый класс для интеграций"""
    
    def __init__(self, config: IntegrationConfig, logger: AgentLogger):
        """Инициализация интеграции"""
        self.config = config
        self.logger = logger
        self.metrics = IntegrationMetrics()
        self.session: Optional[ClientSession] = None
        self._request_times: List[float] = []
        self._last_request_time: Optional[float] = None
        
    async def initialize(self) -> None:
        """Инициализация сессии"""
        if not self.session:
            self.session = ClientSession()
            
    async def shutdown(self) -> None:
        """Завершение работы интеграции"""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Выполнение запроса к API"""
        if not self.config.enabled:
            raise ValueError(f"Интеграция {self.config.name} отключена")
            
        if not self.session:
            await self.initialize()
            
        # Проверяем rate limit
        if self.config.rate_limit:
            await self._check_rate_limit()
            
        # Добавляем авторизацию
        headers = kwargs.pop("headers", {})
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
            
        url = f"{self.config.api_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        start_time = datetime.now()
        attempt = 0
        
        while attempt < self.config.max_retries:
            try:
                async with self.session.request(
                    method,
                    url,
                    headers=headers,
                    timeout=self.config.timeout,
                    **kwargs
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Обновляем метрики
                    self._update_metrics(start_time, success=True)
                    
                    return data
                    
            except ClientError as e:
                attempt += 1
                self._update_metrics(start_time, success=False, error=str(e))
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * attempt)
                else:
                    raise
                    
    async def _check_rate_limit(self) -> None:
        """Проверка rate limit"""
        if self._last_request_time:
            elapsed = datetime.now().timestamp() - self._last_request_time
            if elapsed < (1 / self.config.rate_limit):
                await asyncio.sleep((1 / self.config.rate_limit) - elapsed)
                
    def _update_metrics(
        self,
        start_time: datetime,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Обновление метрик"""
        now = datetime.now()
        duration = (now - start_time).total_seconds()
        
        self.metrics.total_requests += 1
        self._request_times.append(duration)
        self._last_request_time = now.timestamp()
        self.metrics.last_request_time = now
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            self.metrics.total_errors += 1
            self.metrics.last_error_time = now
            self.metrics.last_error_message = error
            
        # Обновляем среднее время ответа
        self.metrics.average_response_time = sum(self._request_times) / len(self._request_times)
        
        # Ограничиваем историю времени ответа
        if len(self._request_times) > 1000:
            self._request_times = self._request_times[-1000:]


class IntegrationManager:
    """Менеджер интеграций"""
    
    def __init__(self, logger: AgentLogger, config_path: str = "config/integrations"):
        """Инициализация менеджера интеграций"""
        self.logger = logger
        self.config_path = config_path
        self.integrations: Dict[str, Integration] = {}
        self.encryption_key: Optional[bytes] = None
        
        # Создаем директорию для конфигураций если не существует
        os.makedirs(config_path, exist_ok=True)
        
    def initialize_encryption(self, key: Optional[bytes] = None) -> None:
        """Инициализация шифрования"""
        if key:
            self.encryption_key = key
        else:
            self.encryption_key = Fernet.generate_key()
            
    async def load_integrations(self) -> None:
        """Загрузка интеграций из конфигурационных файлов"""
        for filename in os.listdir(self.config_path):
            if filename.endswith(".json"):
                path = os.path.join(self.config_path, filename)
                with open(path) as f:
                    config_data = json.load(f)
                    
                # Расшифровываем секретные данные
                if self.encryption_key and "api_key" in config_data:
                    fernet = Fernet(self.encryption_key)
                    config_data["api_key"] = fernet.decrypt(
                        config_data["api_key"].encode()
                    ).decode()
                    
                config = IntegrationConfig(**config_data)
                self.integrations[config.name] = Integration(config, self.logger)
                
    async def initialize_all(self) -> None:
        """Инициализация всех интеграций"""
        for integration in self.integrations.values():
            await integration.initialize()
            
    async def shutdown_all(self) -> None:
        """Завершение работы всех интеграций"""
        for integration in self.integrations.values():
            await integration.shutdown()
            
    def add_integration(self, config: IntegrationConfig) -> None:
        """Добавление новой интеграции"""
        if config.name in self.integrations:
            raise ValueError(f"Интеграция {config.name} уже существует")
            
        self.integrations[config.name] = Integration(config, self.logger)
        
        # Сохраняем конфигурацию
        self._save_config(config)
        
    def remove_integration(self, name: str) -> None:
        """Удаление интеграции"""
        if name not in self.integrations:
            raise ValueError(f"Интеграция {name} не найдена")
            
        del self.integrations[name]
        
        # Удаляем конфигурационный файл
        config_file = os.path.join(self.config_path, f"{name}.json")
        if os.path.exists(config_file):
            os.remove(config_file)
            
    def enable_integration(self, name: str) -> None:
        """Включение интеграции"""
        if name not in self.integrations:
            raise ValueError(f"Интеграция {name} не найдена")
            
        self.integrations[name].config.enabled = True
        self._save_config(self.integrations[name].config)
        
    def disable_integration(self, name: str) -> None:
        """Отключение интеграции"""
        if name not in self.integrations:
            raise ValueError(f"Интеграция {name} не найдена")
            
        self.integrations[name].config.enabled = False
        self._save_config(self.integrations[name].config)
        
    def get_integration(self, name: str) -> Integration:
        """Получение интеграции по имени"""
        if name not in self.integrations:
            raise ValueError(f"Интеграция {name} не найдена")
            
        return self.integrations[name]
        
    def get_all_metrics(self) -> Dict[str, IntegrationMetrics]:
        """Получение метрик всех интеграций"""
        return {
            name: integration.metrics
            for name, integration in self.integrations.items()
        }
        
    def _save_config(self, config: IntegrationConfig) -> None:
        """Сохранение конфигурации интеграции"""
        config_data = dataclasses.asdict(config)
        
        # Шифруем секретные данные
        if self.encryption_key and config.api_key:
            fernet = Fernet(self.encryption_key)
            config_data["api_key"] = fernet.encrypt(
                config.api_key.encode()
            ).decode()
            
        path = os.path.join(self.config_path, f"{config.name}.json")
        with open(path, "w") as f:
            json.dump(config_data, f, indent=4)
            
    async def execute_with_retry(
        self,
        integration_name: str,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args,
        **kwargs
    ) -> Any:
        """Выполнение функции с повторными попытками"""
        integration = self.get_integration(integration_name)
        
        attempt = 0
        while attempt < integration.config.max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                self.logger.error(
                    f"Ошибка при выполнении {func.__name__} для {integration_name}: {str(e)}"
                )
                
                if attempt < integration.config.max_retries:
                    await asyncio.sleep(integration.config.retry_delay * attempt)
                else:
                    raise 