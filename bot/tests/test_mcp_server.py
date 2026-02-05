import os
import json
import pytest

pytest.importorskip("mcp")
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

# Импортируем класс плагина
from bot.plugins.mcp_server import MCPServerPlugin


@pytest.fixture
def mcp_plugin():
    """Фикстура для создания экземпляра плагина"""
    with patch('bot.plugins.mcp_server.MCPServerPlugin._get_config_path') as mock_config_path:
        # Создаем временный файл для конфигурации
        temp_config = tempfile.NamedTemporaryFile(delete=False)
        temp_config.close()
        
        # Мокаем путь к файлу конфигурации
        mock_config_path.return_value = Path(temp_config.name)
        
        # Создаем экземпляр плагина
        plugin = MCPServerPlugin()
        
        # Возвращаем экземпляр и имя временного файла
        yield plugin
        
        # Удаляем временный файл
        if os.path.exists(temp_config.name):
            os.unlink(temp_config.name)


@pytest.fixture
def mock_env_vars():
    """Фикстура для мока переменных окружения"""
    with patch.dict(os.environ, {
        'ADMIN_USER_IDS': '123,456',
        'MCP_SERVERS_ALLOWED_USERS': '123,456,789',
        'MCP_REQUEST_TIMEOUT': '10',
        'DEFAULT_MCP_SERVERS': 'test:http://test.com'
    }):
        yield


@pytest.mark.asyncio
async def test_get_spec(mcp_plugin, mock_env_vars):
    """Тест получения спецификаций функций"""
    specs = mcp_plugin.get_spec()
    
    # Проверяем наличие базовых функций управления
    assert any(spec['name'] == 'register_mcp_server' for spec in specs)
    assert any(spec['name'] == 'list_mcp_servers' for spec in specs)
    assert any(spec['name'] == 'remove_mcp_server' for spec in specs)
    
    # Проверяем общее количество спецификаций (3 базовые + возможные из серверов)
    assert len(specs) >= 3


@pytest.mark.asyncio
async def test_register_server(mcp_plugin, mock_env_vars):
    """Тест регистрации сервера"""
    # Мокаем _fetch_server_tools
    mcp_plugin._fetch_server_tools = AsyncMock(return_value=[
        {"name": "test_function", "description": "Test function", "parameters": {}}
    ])
    
    # Вызываем функцию регистрации
    result = await mcp_plugin.register_server(
        server_name="test_server", 
        base_url="http://example.com", 
        user_id=123
    )
    
    # Проверяем результат
    assert result['success'] is True
    assert "test_server" in mcp_plugin.servers
    assert mcp_plugin.servers["test_server"]["base_url"] == "http://example.com"
    assert len(mcp_plugin.servers["test_server"]["tools"]) == 1


@pytest.mark.asyncio
async def test_register_server_unauthorized(mcp_plugin, mock_env_vars):
    """Тест регистрации сервера неавторизованным пользователем"""
    result = await mcp_plugin.register_server(
        server_name="test_server", 
        base_url="http://example.com", 
        user_id=999  # ID отсутствует в ADMIN_USER_IDS
    )
    
    # Проверяем отказ в доступе
    assert 'error' in result
    assert 'доступно только администраторам' in result['error']


@pytest.mark.asyncio
async def test_list_servers(mcp_plugin, mock_env_vars):
    """Тест получения списка серверов"""
    # Добавляем тестовый сервер
    mcp_plugin.servers = {
        "test_server": {
            "base_url": "http://example.com",
            "api_key": None,
            "description": "Test server",
            "tools": [{"name": "test_function"}]
        }
    }
    
    # Получаем список серверов
    result = await mcp_plugin.list_servers()
    
    # Проверяем результат
    assert 'servers' in result
    assert len(result['servers']) == 1
    assert result['servers'][0]['name'] == "test_server"
    assert result['servers'][0]['tools_count'] == 1


@pytest.mark.asyncio
async def test_remove_server(mcp_plugin, mock_env_vars):
    """Тест удаления сервера"""
    # Добавляем тестовый сервер
    mcp_plugin.servers = {
        "test_server": {
            "base_url": "http://example.com",
            "tools": []
        }
    }
    
    # Удаляем сервер
    result = await mcp_plugin.remove_server(
        server_name="test_server", 
        user_id=123  # Администратор
    )
    
    # Проверяем результат
    assert result['success'] is True
    assert "test_server" not in mcp_plugin.servers


@pytest.mark.asyncio
async def test_remove_server_unauthorized(mcp_plugin, mock_env_vars):
    """Тест удаления сервера неавторизованным пользователем"""
    # Добавляем тестовый сервер
    mcp_plugin.servers = {
        "test_server": {
            "base_url": "http://example.com",
            "tools": []
        }
    }
    
    # Пытаемся удалить сервер неавторизованным пользователем
    result = await mcp_plugin.remove_server(
        server_name="test_server", 
        user_id=999  # Не администратор
    )
    
    # Проверяем отказ в доступе
    assert 'error' in result
    assert 'доступно только администраторам' in result['error']
    assert "test_server" in mcp_plugin.servers


@pytest.mark.asyncio
async def test_call_mcp_function(mcp_plugin, mock_env_vars):
    """Тест вызова функции на MCP сервере"""
    # Добавляем тестовый сервер
    mcp_plugin.servers = {
        "test_server": {
            "base_url": "http://example.com",
            "api_key": "test_key",
            "tools": [{"name": "test_function"}]
        }
    }
    
    # Мокаем httpx клиент
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"result": "success"})
    
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    
    # Вызываем функцию с моком httpx клиента
    with patch('httpx.AsyncClient', return_value=mock_client):
        result = await mcp_plugin.call_mcp_function(
            server_name="test_server",
            function_name="test_function",
            param1="value1"
        )
    
    # Проверяем результат
    assert result == {"result": "success"}
    
    # Проверяем, что был выполнен запрос с правильными параметрами
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args[1]
    
    # Проверяем URL
    assert "http://example.com/execute" in call_args['url']
    
    # Проверяем заголовки
    assert call_args['headers']['Authorization'] == "Bearer test_key"
    
    # Проверяем тело запроса
    assert call_args['json']['name'] == "test_function"
    assert call_args['json']['arguments']['param1'] == "value1"


@pytest.mark.asyncio
async def test_execute_filter_internal_params(mcp_plugin, mock_env_vars):
    """Тест фильтрации внутренних параметров при вызове execute"""
    # Добавляем тестовый сервер
    mcp_plugin.servers = {
        "test_server": {
            "base_url": "http://example.com",
            "tools": [{"name": "test_function"}]
        }
    }
    
    # Мокаем call_mcp_function
    mcp_plugin.call_mcp_function = AsyncMock(return_value={"result": "success"})
    
    # Вызываем execute с внутренним параметром user_id
    await mcp_plugin.execute(
        function_name="test_server_test_function",
        helper=None,
        param1="value1",
        user_id=123
    )
    
    # Проверяем, что user_id был удален из параметров
    call_args = mcp_plugin.call_mcp_function.call_args[1]
    assert "user_id" not in call_args
    assert "param1" in call_args
    assert call_args["param1"] == "value1"


@pytest.mark.asyncio
async def test_handle_mcp_servers_command(mcp_plugin, mock_env_vars):
    """Тест обработчика команды /mcp_servers"""
    # Добавляем тестовый сервер
    mcp_plugin.servers = {
        "test_server": {
            "base_url": "http://example.com",
            "description": "Test server",
            "tools": [{"name": "test_function"}]
        }
    }
    
    # Мокаем объект update
    update = MagicMock()
    update.effective_user.id = 123  # Админ
    
    # Мокаем list_servers
    mcp_plugin.list_servers = AsyncMock(return_value={
        "servers": [
            {
                "name": "test_server",
                "base_url": "http://example.com",
                "description": "Test server",
                "tools_count": 1,
                "tools": ["test_function"]
            }
        ]
    })
    
    # Вызываем обработчик команды
    result = await mcp_plugin.handle_mcp_servers_command(update, None)
    
    # Проверяем результат
    assert isinstance(result, dict)
    assert "text" in result
    assert "parse_mode" in result
    assert result["parse_mode"] == "Markdown"
    
    # Проверка текста
    text = result["text"]
    assert "Зарегистрированные MCP серверы" in text
    assert "**test_server**" in text  # Проверяем форматирование жирным
    assert "`http://example.com`" in text  # Проверяем форматирование кода
    assert "Test server" in text
    
    # Для администратора должны быть инструкции по управлению
    assert "**Управление серверами**" in text
    assert "Для добавления: `" in text
    assert "Для удаления: `" in text
    
    # Тест отказа в доступе неавторизованному пользователю
    update.effective_user.id = 999  # Не в списке разрешенных пользователей
    mcp_plugin.is_user_allowed = MagicMock(return_value=False)
    
    result = await mcp_plugin.handle_mcp_servers_command(update, None)
    assert isinstance(result, dict)
    assert "text" in result
    assert "parse_mode" in result
    assert "У вас нет доступа" in result["text"]


@pytest.mark.asyncio
async def test_user_access_control(mcp_plugin, mock_env_vars):
    """Тест контроля доступа пользователей"""
    # Проверяем администраторов
    assert mcp_plugin.is_admin(123) is True
    assert mcp_plugin.is_admin(456) is True
    assert mcp_plugin.is_admin(999) is False
    
    # Проверяем разрешенных пользователей
    assert mcp_plugin.is_user_allowed(123) is True  # Админ всегда разрешен
    assert mcp_plugin.is_user_allowed(789) is True  # В списке разрешенных
    assert mcp_plugin.is_user_allowed(999) is False  # Не в списке
    
    # Проверяем параметр * для MCP_SERVERS_ALLOWED_USERS
    with patch.dict(os.environ, {'MCP_SERVERS_ALLOWED_USERS': '*'}):
        mcp_plugin.allowed_users = mcp_plugin._get_allowed_users()
        assert mcp_plugin.is_user_allowed(999) is True  # Любой пользователь


if __name__ == "__main__":
    pytest.main() 
