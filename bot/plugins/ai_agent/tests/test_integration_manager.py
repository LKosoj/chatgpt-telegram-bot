"""
Тесты для менеджера интеграций
"""

import os
import json
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from aiohttp import ClientError

from ..integration_manager import (
    IntegrationManager,
    Integration,
    IntegrationConfig,
    IntegrationMetrics
)
from ..logging import AgentLogger


@pytest.fixture
def logger():
    """Фикстура для логгера"""
    return AgentLogger("test_logs")


@pytest.fixture
def config_path(tmp_path):
    """Фикстура для пути к конфигурациям"""
    return str(tmp_path / "integrations")


@pytest.fixture
def manager(logger, config_path):
    """Фикстура для менеджера интеграций"""
    return IntegrationManager(logger, config_path)


@pytest.fixture
def test_config():
    """Фикстура для тестовой конфигурации"""
    return IntegrationConfig(
        name="test_integration",
        api_url="http://test.com/api",
        api_key="test_key",
        timeout=10,
        max_retries=2,
        retry_delay=1,
        rate_limit=10
    )


@pytest.mark.asyncio
async def test_add_integration(manager, test_config):
    """Тест добавления интеграции"""
    manager.add_integration(test_config)
    
    assert test_config.name in manager.integrations
    assert isinstance(manager.integrations[test_config.name], Integration)
    assert os.path.exists(os.path.join(manager.config_path, f"{test_config.name}.json"))


@pytest.mark.asyncio
async def test_remove_integration(manager, test_config):
    """Тест удаления интеграции"""
    manager.add_integration(test_config)
    manager.remove_integration(test_config.name)
    
    assert test_config.name not in manager.integrations
    assert not os.path.exists(os.path.join(manager.config_path, f"{test_config.name}.json"))


@pytest.mark.asyncio
async def test_enable_disable_integration(manager, test_config):
    """Тест включения и отключения интеграции"""
    manager.add_integration(test_config)
    
    manager.disable_integration(test_config.name)
    assert not manager.integrations[test_config.name].config.enabled
    
    manager.enable_integration(test_config.name)
    assert manager.integrations[test_config.name].config.enabled


@pytest.mark.asyncio
async def test_get_integration(manager, test_config):
    """Тест получения интеграции"""
    manager.add_integration(test_config)
    integration = manager.get_integration(test_config.name)
    
    assert isinstance(integration, Integration)
    assert integration.config == test_config


@pytest.mark.asyncio
async def test_get_all_metrics(manager, test_config):
    """Тест получения метрик"""
    manager.add_integration(test_config)
    metrics = manager.get_all_metrics()
    
    assert test_config.name in metrics
    assert isinstance(metrics[test_config.name], IntegrationMetrics)


@pytest.mark.asyncio
async def test_load_integrations(manager, test_config):
    """Тест загрузки интеграций"""
    # Сохраняем тестовую конфигурацию
    manager.add_integration(test_config)
    
    # Создаем новый менеджер
    new_manager = IntegrationManager(logger, manager.config_path)
    await new_manager.load_integrations()
    
    assert test_config.name in new_manager.integrations
    loaded_config = new_manager.integrations[test_config.name].config
    assert loaded_config.name == test_config.name
    assert loaded_config.api_url == test_config.api_url


@pytest.mark.asyncio
async def test_encryption(manager, test_config):
    """Тест шифрования данных"""
    manager.initialize_encryption()
    manager.add_integration(test_config)
    
    # Проверяем что API ключ зашифрован в файле
    config_file = os.path.join(manager.config_path, f"{test_config.name}.json")
    with open(config_file) as f:
        saved_config = json.load(f)
        assert saved_config["api_key"] != test_config.api_key
        
    # Проверяем что после загрузки ключ расшифрован
    new_manager = IntegrationManager(logger, manager.config_path)
    new_manager.encryption_key = manager.encryption_key
    await new_manager.load_integrations()
    
    loaded_integration = new_manager.get_integration(test_config.name)
    assert loaded_integration.config.api_key == test_config.api_key


@pytest.mark.asyncio
async def test_integration_request(manager, test_config):
    """Тест выполнения запроса через интеграцию"""
    manager.add_integration(test_config)
    integration = manager.get_integration(test_config.name)
    
    # Мокаем aiohttp.ClientSession
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"data": "test"})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()
    
    with patch("aiohttp.ClientSession.request", return_value=mock_response):
        result = await integration.make_request("GET", "test")
        
    assert result == {"data": "test"}
    assert integration.metrics.total_requests == 1
    assert integration.metrics.successful_requests == 1


@pytest.mark.asyncio
async def test_integration_request_error(manager, test_config):
    """Тест обработки ошибок запроса"""
    manager.add_integration(test_config)
    integration = manager.get_integration(test_config.name)
    
    # Мокаем aiohttp.ClientSession для генерации ошибки
    mock_response = Mock()
    mock_response.__aenter__ = AsyncMock(side_effect=ClientError())
    mock_response.__aexit__ = AsyncMock()
    
    with patch("aiohttp.ClientSession.request", return_value=mock_response):
        with pytest.raises(ClientError):
            await integration.make_request("GET", "test")
            
    assert integration.metrics.total_requests == 1
    assert integration.metrics.failed_requests == 1
    assert integration.metrics.total_errors == 1


@pytest.mark.asyncio
async def test_rate_limit(manager, test_config):
    """Тест ограничения частоты запросов"""
    test_config.rate_limit = 2  # 2 запроса в секунду
    manager.add_integration(test_config)
    integration = manager.get_integration(test_config.name)
    
    # Мокаем успешный ответ
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"data": "test"})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()
    
    with patch("aiohttp.ClientSession.request", return_value=mock_response):
        start_time = asyncio.get_event_loop().time()
        
        # Делаем 3 запроса
        for _ in range(3):
            await integration.make_request("GET", "test")
            
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Проверяем что прошло минимум 1 секунда (1/rate_limit * 2)
        assert elapsed >= 1.0


@pytest.mark.asyncio
async def test_execute_with_retry(manager, test_config):
    """Тест выполнения функции с повторными попытками"""
    manager.add_integration(test_config)
    
    # Создаем тестовую функцию, которая падает первые 2 раза
    attempts = 0
    
    async def test_func():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise Exception("Test error")
        return "success"
    
    result = await manager.execute_with_retry(test_config.name, test_func)
    
    assert result == "success"
    assert attempts == 2  # Функция должна быть вызвана 2 раза


@pytest.mark.asyncio
async def test_initialize_shutdown(manager, test_config):
    """Тест инициализации и завершения работы"""
    manager.add_integration(test_config)
    
    await manager.initialize_all()
    for integration in manager.integrations.values():
        assert integration.session is not None
        
    await manager.shutdown_all()
    for integration in manager.integrations.values():
        assert integration.session is None 