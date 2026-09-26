import os
import json
import pytest

pytest.importorskip("mcp")
import httpx
from mcp import types
from unittest.mock import patch, AsyncMock, MagicMock

# Импортируем класс плагина
from bot.plugins.mcp_server import MCPServerPlugin
from bot.i18n import localized_text
from bot import net_safety


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


@pytest.fixture
def mcp_plugin(tmp_path, mock_env_vars):
    """Фикстура для создания экземпляра плагина"""
    plugin = MCPServerPlugin()
    plugin.initialize(openai=MagicMock(config={"bot_language": "ru"}), storage_root=str(tmp_path))
    yield plugin


def test_constructor_has_no_config_path_side_effects(tmp_path, monkeypatch):
    config_path = tmp_path / "mcp_servers.json"
    monkeypatch.setenv("MCP_SERVERS_CONFIG_PATH", str(config_path))

    with patch.object(MCPServerPlugin, "load_servers_config") as load:
        plugin = MCPServerPlugin()

    load.assert_not_called()
    assert plugin.config_path is None
    assert not config_path.exists()


def test_initialize_loads_storage_config(tmp_path, mock_env_vars):
    config = tmp_path / "mcp_servers.json"
    config.write_text(json.dumps({"stored": {"base_url": "http://stored", "tools": []}}), encoding="utf-8")

    plugin = MCPServerPlugin()
    plugin.initialize(openai=MagicMock(config={"bot_language": "ru"}), storage_root=str(tmp_path))

    assert plugin.config_path == config
    assert plugin.servers["stored"]["base_url"] == "http://stored"


def test_corrupt_config_preserves_existing_servers(tmp_path, mock_env_vars):
    config = tmp_path / "mcp_servers.json"
    config.write_text("{not-json", encoding="utf-8")
    plugin = MCPServerPlugin()
    plugin.servers = {"live": {"base_url": "http://live", "tools": []}}
    plugin.initialize(openai=MagicMock(config={"bot_language": "ru"}), storage_root=str(tmp_path))

    assert plugin.servers == {"live": {"base_url": "http://live", "tools": []}}


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
    assert localized_text('mcp_register_admin_only', 'ru') in result['error']


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
async def test_list_servers_supports_stdio_config(mcp_plugin, mock_env_vars):
    mcp_plugin.servers = {
        "stdio_server": {
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
            "description": "Stdio server",
            "tools": [{"name": "stdio_tool"}],
        }
    }

    result = await mcp_plugin.list_servers()

    assert result["servers"][0]["transport"] == "stdio"
    assert result["servers"][0]["base_url"] == ""
    assert result["servers"][0]["command"] == "python"
    assert result["servers"][0]["args"] == ["server.py"]


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
    assert localized_text('mcp_remove_admin_only', 'ru') in result['error']
    assert "test_server" in mcp_plugin.servers


@pytest.mark.asyncio
async def test_register_server_invalidates_function_index(mcp_plugin, mock_env_vars):
    """register_server должен инвалидировать индекс функций после успешной регистрации."""
    mcp_plugin._fetch_server_tools = AsyncMock(return_value=[
        {"name": "test_function", "description": "Test function", "parameters": {}}
    ])
    called = []
    mcp_plugin._invalidate_function_index = lambda: called.append(True)

    result = await mcp_plugin.register_server(
        server_name="test_server",
        base_url="http://example.com",
        user_id=123,
    )

    assert result['success'] is True
    assert called


@pytest.mark.asyncio
async def test_remove_server_invalidates_function_index(mcp_plugin, mock_env_vars):
    """remove_server должен инвалидировать индекс функций после удаления."""
    mcp_plugin.servers = {
        "test_server": {
            "base_url": "http://example.com",
            "tools": []
        }
    }
    called = []
    mcp_plugin._invalidate_function_index = lambda: called.append(True)

    result = await mcp_plugin.remove_server(
        server_name="test_server",
        user_id=123
    )

    assert result['success'] is True
    assert called


@pytest.mark.asyncio
async def test_refresh_server_tools_invalidates_function_index(mcp_plugin, mock_env_vars):
    """_refresh_server_tools (фоновое обновление списка тулов) должен инвалидировать индекс."""
    mcp_plugin.servers = {
        "test_server": {
            "transport": "http",
            "base_url": "http://example.com",
            "tools": []
        }
    }
    mcp_plugin._fetch_server_tools = AsyncMock(return_value=[
        {"name": "test_function", "description": "Test function", "parameters": {}}
    ])
    called = []
    mcp_plugin._invalidate_function_index = lambda: called.append(True)

    await mcp_plugin._refresh_server_tools("test_server")

    assert called


@pytest.mark.asyncio
async def test_call_mcp_function(mcp_plugin, mock_env_vars, monkeypatch):
    """Тест вызова функции на MCP сервере"""
    # Добавляем тестовый сервер
    mcp_plugin.servers = {
        "test_server": {
            "base_url": "http://example.com",
            "api_key": "test_key",
            "tools": [{"name": "test_function"}]
        }
    }

    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == "http://example.com/execute"
        assert request.headers["authorization"] == "Bearer test_key"
        body = json.loads(request.content)
        assert body["name"] == "test_function"
        assert body["arguments"]["param1"] == "value1"
        return httpx.Response(200, json={"result": "success"})

    original_safe_request = net_safety.safe_request
    monkeypatch.setattr(net_safety.socket, "getaddrinfo", lambda *a, **kw: [
        (net_safety.socket.AF_INET, net_safety.socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))
    ])
    monkeypatch.setattr(
        net_safety,
        "safe_request",
        lambda *a, **kw: original_safe_request(*a, **kw, transport=httpx.MockTransport(handler)),
    )

    # Вызываем функцию, HTTP уходит через httpx.MockTransport (см. handler выше)
    result = await mcp_plugin.call_mcp_function(
        server_name="test_server",
        function_name="test_function",
        param1="value1"
    )

    # Проверяем результат
    assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_call_mcp_function_rejects_private_base_url_by_default(mcp_plugin, mock_env_vars, monkeypatch):
    """SSRF-защита: без MCP_ALLOW_PRIVATE_HOSTS вызов на приватный base_url не уходит в сеть."""
    mcp_plugin.servers = {
        "test_server": {
            "base_url": "http://127.0.0.1:9999",
            "api_key": "test_key",
            "tools": [{"name": "test_function"}]
        }
    }

    def handler(request):
        raise AssertionError("HTTP request must not be made for a private base_url")

    original_safe_request = net_safety.safe_request
    monkeypatch.setattr(
        net_safety,
        "safe_request",
        lambda *a, **kw: original_safe_request(*a, **kw, transport=httpx.MockTransport(handler)),
    )

    result = await mcp_plugin.call_mcp_function(
        server_name="test_server",
        function_name="test_function",
        param1="value1"
    )

    assert "error" in result


@pytest.mark.asyncio
async def test_call_mcp_function_allows_private_base_url_with_flag(mcp_plugin, mock_env_vars, monkeypatch):
    """С allow_private_hosts=True вызов на приватный base_url доходит до сервера."""
    mcp_plugin.allow_private_hosts = True
    mcp_plugin.servers = {
        "test_server": {
            "base_url": "http://127.0.0.1:9999",
            "api_key": "test_key",
            "tools": [{"name": "test_function"}]
        }
    }

    def handler(request):
        assert request.method == "POST"
        return httpx.Response(200, json={"result": "success"})

    original_safe_request = net_safety.safe_request
    monkeypatch.setattr(
        net_safety,
        "safe_request",
        lambda *a, **kw: original_safe_request(*a, **kw, transport=httpx.MockTransport(handler)),
    )

    result = await mcp_plugin.call_mcp_function(
        server_name="test_server",
        function_name="test_function",
        param1="value1"
    )

    assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_fetch_server_tools_rejects_private_base_url_by_default(mcp_plugin, mock_env_vars, monkeypatch):
    """SSRF-защита: _fetch_server_tools на приватный base_url без флага не уходит в сеть."""

    def handler(request):
        raise AssertionError("HTTP request must not be made for a private base_url")

    original_safe_get = net_safety.safe_get
    monkeypatch.setattr(
        net_safety,
        "safe_get",
        lambda *a, **kw: original_safe_get(*a, **kw, transport=httpx.MockTransport(handler)),
    )

    result = await mcp_plugin._fetch_server_tools("http://127.0.0.1:9999")

    assert result == []


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
        user_id=123,
        chat_id=456,
        request_context=object(),
    )
    
    # Проверяем, что user_id был удален из параметров
    call_args = mcp_plugin.call_mcp_function.call_args[1]
    assert "user_id" not in call_args
    assert "chat_id" not in call_args
    assert "request_context" not in call_args
    assert "param1" in call_args
    assert call_args["param1"] == "value1"


@pytest.mark.asyncio
async def test_execute_list_servers_respects_allowed_users(mcp_plugin, mock_env_vars):
    mcp_plugin.is_user_allowed = MagicMock(return_value=False)

    result = await mcp_plugin.execute(
        function_name="list_mcp_servers",
        helper=None,
        user_id=999,
    )

    assert "error" in result
    mcp_plugin.is_user_allowed.assert_called_once_with(999)


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
    assert localized_text('mcp_list_title', 'ru') in text
    assert "**test_server**" in text  # Проверяем форматирование жирным
    assert "`http://example.com`" in text  # Проверяем форматирование кода
    assert "Test server" in text
    
    # Для администратора должны быть инструкции по управлению
    assert localized_text('mcp_admin_section_title', 'ru') in text
    assert localized_text('mcp_admin_add_http', 'ru').split(':')[0] in text
    assert localized_text('mcp_admin_remove', 'ru').split(':')[0] in text
    
    # Тест отказа в доступе неавторизованному пользователю
    update.effective_user.id = 999  # Не в списке разрешенных пользователей
    mcp_plugin.is_user_allowed = MagicMock(return_value=False)
    
    result = await mcp_plugin.handle_mcp_servers_command(update, None)
    assert isinstance(result, dict)
    assert "text" in result
    assert "parse_mode" in result
    assert localized_text('mcp_access_denied', 'ru') in result["text"]


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


@pytest.mark.asyncio
async def test_fetch_stdio_tools_maps_nested_schema(mcp_plugin, mock_env_vars):
    """Регрессия на §3.8 архитектурного обзора: ListToolsResult.tools, tool.input_schema,
    вложенные object/enum/required не должны теряться."""
    tool = types.Tool(
        name="get_weather",
        description="Get weather",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"]},
                "coords": {
                    "type": "object",
                    "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}},
                    "required": ["lat", "lon"],
                },
            },
            "required": ["city"],
        },
    )
    session = AsyncMock()
    session.list_tools = AsyncMock(return_value=types.ListToolsResult(tools=[tool]))

    result = await mcp_plugin._fetch_stdio_tools(session)

    assert result == [{
        "name": "get_weather",
        "description": "Get weather",
        "parameters": tool.input_schema,
    }]


@pytest.mark.asyncio
async def test_fetch_stdio_tools_defaults_empty_schema(mcp_plugin, mock_env_vars):
    tool = types.Tool(name="ping", description=None, input_schema={})
    session = AsyncMock()
    session.list_tools = AsyncMock(return_value=types.ListToolsResult(tools=[tool]))

    result = await mcp_plugin._fetch_stdio_tools(session)

    assert result == [{
        "name": "ping",
        "description": "Инструмент ping",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }]


@pytest.mark.asyncio
async def test_get_or_create_session_uses_send_ping(mcp_plugin, mock_env_vars):
    """Регрессия: ClientSession.ping() не существует ни в 1.26, ни в 2.x — только send_ping()."""
    fake_session = MagicMock(spec=["send_ping"])  # spec без "ping" — .ping() бросит AttributeError
    fake_session.send_ping = AsyncMock(return_value=None)
    mcp_plugin.sessions["srv"] = fake_session

    session = await mcp_plugin._get_or_create_session("srv")

    assert session is fake_session
    fake_session.send_ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_mcp_function_stdio_returns_structured_content(mcp_plugin, mock_env_vars):
    mcp_plugin.servers = {"srv": {"transport": "stdio", "command": "python", "tools": []}}
    call_result = types.CallToolResult(
        content=[types.TextContent(type="text", text="42")],
        structured_content={"answer": 42},
        is_error=False,
    )
    fake_session = AsyncMock()
    fake_session.call_tool = AsyncMock(return_value=call_result)
    mcp_plugin._get_or_create_session = AsyncMock(return_value=fake_session)

    result = await mcp_plugin.call_mcp_function(server_name="srv", function_name="answer")

    assert result == {"answer": 42}
    json.dumps(result)  # не требует default=str — регрессия на несериализуемый pydantic-объект


@pytest.mark.asyncio
async def test_call_mcp_function_stdio_error_result(mcp_plugin, mock_env_vars):
    mcp_plugin.servers = {"srv": {"transport": "stdio", "command": "python", "tools": []}}
    call_result = types.CallToolResult(
        content=[types.TextContent(type="text", text="boom")],
        is_error=True,
    )
    fake_session = AsyncMock()
    fake_session.call_tool = AsyncMock(return_value=call_result)
    mcp_plugin._get_or_create_session = AsyncMock(return_value=fake_session)

    result = await mcp_plugin.call_mcp_function(server_name="srv", function_name="boom")

    assert result == {"error": "boom"}


if __name__ == "__main__":
    pytest.main()


def test_mcp_call_result_reports_non_text_content():
    from bot.plugins.mcp_server import _mcp_call_result_to_dict

    result = types.CallToolResult(
        content=[types.ImageContent(type="image", data="aGVsbG8=", mimeType="image/png")],
        isError=False,
    )
    out = _mcp_call_result_to_dict(result)
    assert out["omitted_content"] == ["image"]
    assert "non-text" in out["result"]

    mixed = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="caption"),
            types.ImageContent(type="image", data="aGVsbG8=", mimeType="image/png"),
        ],
        isError=False,
    )
    out = _mcp_call_result_to_dict(mixed)
    assert out["result"] == "caption"
    assert out["omitted_content"] == ["image"]
