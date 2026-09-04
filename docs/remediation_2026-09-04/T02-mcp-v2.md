# T02. Перевод `bot/plugins/mcp_server.py` на mcp 2.x и починка stdio-обнаружения инструментов

Статус: план (код не менялся). Все находки ниже проверены запуском реального кода против
`mcp==2.1.1`, установленного в отдельный venv `/tmp/probe-mcp` (не влияет на проект), а не
взяты из памяти или из текста аудита. Там, где текст аудита (`docs/architecture_code_review_2026-09-04.md`,
`docs/audit_remediation_plan_2026-09-04.md`) расходится с фактическим файлом — ниже явно
указано, что было перепроверено и почему.

## 0. Как проверялось (для повторяемости)

```bash
~/.local/bin/uv venv /tmp/probe-mcp
~/.local/bin/uv pip install --python /tmp/probe-mcp/bin/python "mcp==2.1.1" httpx pytest pytest-asyncio
cd /srv/git_projects/chatgpt-telegram-bot
/tmp/probe-mcp/bin/python -m pytest bot/tests/test_mcp_server.py -q   # 15 passed — см. §5
```
Сигнатуры и поля ниже получены через `inspect.signature(...)`, `Model.model_fields`, реальное
создание `mcp.types.Tool/CallToolResult/ListToolsResult` и попытку `json.dumps(...)` — не по
документации на память. Мапping-функция из §3 проверена отдельным scratch-скриптом (не
коммитился) на вложенных объектах/enum/required — результат воспроизведён в §3.

Важно: `bot/tests/test_mcp_server.py` **проходит все 15 тестов и под mcp 1.26 (текущая
система), и под mcp 2.1.1** — потому что ни один существующий тест не вызывает
`_fetch_stdio_tools`, `_get_or_create_session`, `_connect_to_server_stdio` или stdio-ветку
`call_mcp_function`. Зелёный CI сегодня ничего не говорит о работоспособности stdio-транспорта;
это и объясняет, как баг из §3.8 архитектурного обзора остался незамеченным.

## 1. Цель

1. Заменить в `bot/plugins/mcp_server.py` вызовы API `mcp`, изменившиеся между 1.x и 2.x
   (переименованные методы, camelCase→snake_case поля моделей), чтобы код не падал на
   `AttributeError` при переходе на `mcp>=2.1,<3` (версия из T01).
2. Починить `_fetch_stdio_tools` — сейчас у stdio-серверов ecли инструменты в принципе не
   получить, а если бы получить, то поля собирались бы из несуществующих атрибутов
   (`tool.parameters`, `tool.required_parameters`) вместо `tool.inputSchema`/`tool.input_schema`.
3. Дать stdio-ветке тот же корректный маппинг «схема инструмента → `parameters` для OpenAI
   function calling», который HTTP-ветка получает бесплатно (см. §2.3 — там это не баг, а
   другой протокол).
4. Не трогать политику: `register_mcp_server` остаётся вызываемым моделью как есть; ссылка на
   §7.3 (показ полных описаний тулов и команды процесса пользователю) — только пометка "куда
   добавить позже", без реализации.

## 2. Таблица «старый API → новый API» (проверенные имена)

### 2.1. Общее (`mcp` 1.26 → `mcp` 2.1.1)

| Место в коде | Старый API (mcp 1.26, установлен сейчас) | Новый API (mcp 2.1.1) | Проверено как |
|---|---|---|---|
| Список инструментов | `session.list_tools()` → `types.ListToolsResult` с полем `.tools: list[Tool]` | Без изменений по форме — тоже `.tools`; **баг не в этом**, а в том, что код в `_fetch_stdio_tools` вообще не читает `.tools` (см. §3) | `ListToolsResult.model_fields` в обеих версиях |
| Поле схемы тула | `Tool.inputSchema` (атрибут `tool.inputSchema`, alias отсутствует — это и есть имя поля) | `Tool.input_schema` (атрибут `tool.input_schema`; JSON-алиас `inputSchema` сохранён для wire-формата, но **питоновский атрибут — только `input_schema`**, `tool.inputSchema` → `AttributeError`) | `Tool.model_fields` (`alias=` показывает JSON-имя, атрибут — имя поля) + реальное создание объекта и обращение к обоим именам |
| Параметры/required тула | **Не существовали никогда** — код читает `tool.parameters` / `tool.required_parameters`, которых нет ни в 1.x, ни в 2.x `Tool` | То же — их по-прежнему нет | `Tool.model_fields.keys()` в обеих версиях не содержит ни `parameters`, ни `required_parameters` |
| Проверка живости сессии | `session.ping()` — **такого метода нет уже в mcp 1.26** (баг существует и сегодня, до всякой миграции) | `session.send_ping() -> types.EmptyResult` | `'ping' in dir(ClientSession)` → `False` в обеих версиях; `'send_ping' in dir(...)` → `True` в обеих |
| Таймаут на сессию/вызов | `read_timeout_seconds: datetime.timedelta \| None` (конструктор `ClientSession`, `call_tool(...)`) | `read_timeout_seconds: float \| None` (те же места) | `inspect.signature(ClientSession.__init__ / call_tool)` в обеих версиях |
| Результат вызова тула | `session.call_tool(...) -> types.CallToolResult` с полями `content`, `isError`, `structuredContent` (это и есть имена атрибутов в 1.x, alias отсутствует) | `types.CallToolResult` с полями `content`, `is_error` (алиас `isError`), `structured_content` (алиас `structuredContent`) | `CallToolResult.model_fields` в обеих версиях; создание объекта + `json.dumps` |
| Импорты | `from mcp.client.stdio import stdio_client, StdioServerParameters`; `from mcp import ClientSession` | **Без изменений** — оба re-export'а сохранены в 2.1.1 | Прямой `from ... import ...` под `mcp==2.1.1` — успешно |
| `StdioServerParameters` поля | `command`, `args`, `env` | Без изменений; добавлены необязательные `cwd`, `encoding`, `encoding_error_handler` (умолчания есть, код не обязан их передавать) | `StdioServerParameters.model_fields` |
| Транзитивный HTTP-стек | `httpx` + `httpx-sse` (используются *внутри* пакета `mcp`, не в этом файле — см. §2.3) | `httpx2` + `httpcore2` (новые PyPI-пакеты, отдельные от `httpx`/`httpcore`; ставятся как зависимости `mcp`, устанавливаются транзитивно) | `uv pip install mcp==2.1.1` реально подтянул `httpx2==2.12.0`, `httpcore2==2.12.0` (не `httpx`/`httpcore`) |
| `streamablehttp_client` | существовал в 1.x под этим именем в `mcp.client.streamable_http` | переименован в `streamable_http_client` (тот же модуль); **этот файл его не использует** и трогать не нужно — см. §2.3 | `ImportError: cannot import name 'streamablehttp_client' ... Did you mean: 'streamable_http_client'` при попытке импорта старого имени |

### 2.2. Что НЕ меняется (проверено, чтобы не чинить несуществующее)

- Вызов `stdio_client(server_params)` — сигнатура `(server: StdioServerParameters, errlog=...)`, совместима as-is.
- `ClientSession(read_stream, write_stream)` — первые два позиционных параметра не изменились.
- `await session.initialize()` — без изменений.
- `await session.list_tools()` — вызывается без аргументов в коде, сигнатура (в обеих версиях)
  принимает необязательные kwargs; совместимо.
- `await session.call_tool(function_name, arguments=kwargs)` — совместимо позиционно/по
  ключу в обеих версиях.

### 2.3. Важное уточнение по HTTP-ветке (расхождение с текстом аудита)

`docs/audit_remediation_plan_2026-09-04.md:94-97` пишет "тот же маппинг для HTTP-ветки
(`_fetch_server_tools`, ~:275-283)". Это устаревшая ссылка на строки (см. заметку про дрейф
`file:line` — в текущем файле `_fetch_server_tools` находится на **:783-809**, а :275-291 —
это цикл в `get_spec()`, который просто копирует уже готовые словари `tool.copy()`, ничего не
парсит из `mcp`).

Более того, по существу: `_fetch_server_tools`/`call_mcp_function` (HTTP-ветка) **не
используют пакет `mcp` вообще** — это собственный REST-протокол бота (`GET {base_url}/tools`,
`POST {base_url}/execute`), задокументированный в `bot/README_MCP.md:129-152`: удалённый
сервер обязан сам отдавать список тулов **уже в формате OpenAI function calling**
(`{"name", "description", "parameters": {"type": "object", "properties": {...}, "required": [...]}}`).
`_fetch_server_tools` просто возвращает `response.json()` как есть — маппинг тут не нужен,
потому что переводить нечего.

**Вывод: правки HTTP-ветки в рамках T02 не требуются.** Если позже понадобится подключаться к
*настоящим* MCP-серверам по HTTP (Streamable HTTP/SSE из спеки MCP, JSON-RPC-рукопожатие,
`session.initialize()` и т.д.) — это отдельная фича (новый `transport: "streamable_http"`,
использующий `mcp.client.streamable_http.streamable_http_client` + тот же owner-task
паттерн, что и у stdio, см. §7), а не починка текущего HTTP-транспорта.

## 3. Правки по `file:line` (обязательные для T02)

Текущие номера строк — из `bot/plugins/mcp_server.py` в HEAD на момент планирования (886
строк). Проверяйте их заново перед правкой, если файл успеет измениться (T01 меняет только
`requirements.txt`/зависимости, этот файл не трогает).

### 3.1. Новая функция маппинга схемы (добавить рядом с импортами, после `logger = ...`, т.е. около :17-19)

```python
def _mcp_input_schema_to_openai_parameters(input_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """MCP inputSchema — уже полноценная JSON Schema (type/properties/required/enum/
    вложенные object и array), совпадающая по форме с полем "parameters" в OpenAI
    function calling. Поэтому берём её как есть, а не пересобираем по одному полю:
    старый код терял enum, вложенные объекты и всё, кроме верхнеуровневых type+description.
    """
    if isinstance(input_schema, dict) and input_schema:
        return input_schema
    return {"type": "object", "properties": {}, "required": []}
```

Проверено (scratch-скрипт, не в репозитории) на схеме с вложенным `object`, `enum` и
`required` — все поля доходят до результата без потерь (см. §0).

### 3.2. `_fetch_stdio_tools` (сейчас :355-395) — заменить целиком

Было (обращается к `mcp_tools` как к самому списку, а не `.tools`; читает несуществующие
`tool.parameters`/`tool.required_parameters`):

```python
            mcp_tools = await session.list_tools()
            openai_tools = []
            for tool in mcp_tools:
                openai_tool = {
                    "name": tool.name,
                    ...
                if tool.parameters:
                    for param_name, param_schema in tool.parameters.items():
                        ...
                        if param_name in (tool.required_parameters or []):
```

Стало:

```python
    async def _fetch_stdio_tools(self, session: ClientSession) -> List[Dict]:
        """
        Получает список инструментов от MCP сервера через stdio транспорт

        :param session: Активная сессия клиента MCP
        :return: Список инструментов в формате спецификаций OpenAI
        """
        try:
            # list_tools() -> types.ListToolsResult; сами тулы лежат в .tools
            mcp_tools = await session.list_tools()

            openai_tools = []
            for tool in mcp_tools.tools:
                openai_tools.append({
                    "name": tool.name,
                    "description": tool.description or f"Инструмент {tool.name}",
                    # tool.input_schema — snake_case в mcp 2.x (было tool.inputSchema в 1.x)
                    "parameters": _mcp_input_schema_to_openai_parameters(tool.input_schema),
                })

            return openai_tools

        except Exception as e:
            logger.error(f"Ошибка при получении инструментов через stdio: {str(e)}")
            return []
```

### 3.3. `_get_or_create_session` (сейчас :397-410) — `.ping()` → `.send_ping()`

Было (:405):
```python
                    await self.sessions[server_name].ping()
```
Стало:
```python
                    # ClientSession.ping() не существует ни в mcp 1.26, ни в 2.1.1 —
                    # только send_ping(). До этой правки health-check всегда падал в
                    # except и код на каждый вызов молча пересоздавал stdio-процесс.
                    await self.sessions[server_name].send_ping()
```
Это не следствие перехода на 2.x — баг воспроизводится и на установленной сейчас `mcp==1.26`
(проверено: `'ping' in dir(ClientSession)` → `False` в обеих версиях). Чинить в любом случае в
этом же PR, раз файл и так трогаем ради миграции.

### 3.4. `call_mcp_function`, stdio-ветка (сейчас :680-693) — конвертировать `CallToolResult` в dict

Было (:689-690):
```python
                result = await session.call_tool(function_name, arguments=kwargs)
                return result
```
Стало:
```python
                result = await session.call_tool(function_name, arguments=kwargs)
                return _mcp_call_result_to_dict(result)
```
Плюс функция рядом с `_mcp_input_schema_to_openai_parameters` (§3.1):
```python
def _mcp_call_result_to_dict(result: Any) -> Dict[str, Any]:
    """call_mcp_function() отдаёт результат прямо в PluginManager.call_function()
    (bot/plugin_manager.py:487: json.dumps(result, default=str, ensure_ascii=False)).
    Без этой конвертации CallToolResult — pydantic-объект, json.dumps падает на нём и
    default=str превращает ответ в нечитаемый repr вместо текста/данных для модели.
    """
    is_error = getattr(result, "is_error", False)
    content = getattr(result, "content", None) or []
    texts = [block.text for block in content if getattr(block, "type", None) == "text"]
    if is_error:
        return {"error": "; ".join(texts) or "MCP tool call failed"}
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured if isinstance(structured, dict) else {"result": structured}
    return {"result": "\n".join(texts) if texts else None}
```

**Важно про рамки задачи.** Формально T02 в аудите назван "MCP 2.x и **stdio-обнаружение**
инструментов" — этот пункт (3.4) про *вызов* тула, не про обнаружение. Но:
- баг проявляется в той же stdio-ветке, того же файла, того же перехода на mcp 2.x
  (`CallToolResult` меняет `isError`→`is_error`, `structuredContent`→`structured_content` —
  ровно то же переименование полей, что и у `Tool`/`inputSchema`);
- без этой правки миграция "работает" только наполовину: тулы обнаруживаются (после 3.2), но
  вызов любого stdio-тула возвращает пользователю нечитаемый repr вместо результата.

Если ревьюер сочтёт это отдельной задачей — можно вынести в отдельный PR, но рекомендация:
сделать вместе, так как переиспользует тот же контекст (поля `CallToolResult`) и тот же файл.

### 3.5. `_connect_to_server_stdio` (сейчас :459-516) — таймаут на сессию (рекомендуется, не обязательно)

Сейчас HTTP-ветка читает `MCP_REQUEST_TIMEOUT` (:711, :795), а stdio — нет: у
`ClientSession(read_stream, write_stream)` (:492-493) таймаут не выставлен вообще, то есть
`list_tools()`/`call_tool()` могут зависнуть на неотвечающем дочернем процессе навсегда.
Раз мы всё равно меняем сигнатуру таймаута с `timedelta` на `float` (см. таблицу в §2.1), имеет
смысл сразу передать его в конструктор:

Было:
```python
                    session = await stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    )
```
Стало:
```python
                    timeout = float(os.getenv("MCP_REQUEST_TIMEOUT", "30"))
                    session = await stack.enter_async_context(
                        ClientSession(read_stream, write_stream, read_timeout_seconds=timeout)
                    )
```
Это не ломает совместимость (параметр уже существовал в 1.x как `timedelta`, но никогда не
передавался — значит, сейчас таймаута нет вовсе) и напрямую связано с миграцией на float-based
таймауты. Если хочется минимизировать диф — можно отложить как отдельный тикет, помечаю как
рекомендацию, не жёсткое требование T02.

### 3.6. Что НЕ трогаем (явно, чтобы не было соблазна "заодно улучшить")

- `_fetch_server_tools` (:783-809) и HTTP-ветка `call_mcp_function` (:694-738) — см. §2.3, не
  используют `mcp`, чинить нечего.
- `get_spec()` (:173-291) — копирует уже нормализованные словари, менять не нужно.
- `register_server`, `list_servers`, `remove_server`, `handle_mcp_servers_command`,
  `get_commands` — вне зоны mcp-API, не трогаем.
- Модель-вызываемый `register_mcp_server` остаётся как есть (задача политики — вне T02).

## 4. Где позже добавить показ описаний пользователю (§7.3, не реализуется в T02)

`docs/architecture_code_review_2026-09-04.md`, §7.3 (OWASP MCP Top 10, MCP03 Tool Poisoning):
показывать пользователю полные описания тулов и команду локального сервера, пиновать описания
по хэшу, требовать подтверждение при смене схемы, не давать модели самой регистрировать
серверы. Точки, куда это может лечь позже (без реализации сейчас):

- `register_server()` (:518-607) — после успешного `_fetch_stdio_tools`/`_fetch_server_tools`
  здесь есть полный список `tools_data` (имя/описание/схема каждого тула) и, для stdio, точная
  команда запуска (`kwargs["command"]`, `kwargs.get("args", [])`) — естественное место
  посчитать хэш описаний и потребовать явное подтверждение админа перед сохранением.
- `handle_mcp_servers_command()` (:740-781) — сейчас показывает только `tools_count`, не сами
  описания (:773). Здесь можно расширить вывод полными описаниями тулов и командой процесса
  для stdio-серверов.
- `_refresh_server_tools()` (:319-353) — точка, где схема тула может измениться на лету при
  фоновом обновлении; здесь нужно будет сравнивать новый хэш со старым и не применять
  молча, а сигналить администратору (сейчас — просто перезаписывает `server_config["tools"]`
  и сохраняет).
- Модель-вызываемый `register_mcp_server` (спека :182-233) — политика "не давать модели самой
  регистрировать серверы" потребует убрать эту функцию из `get_spec()` и оставить только
  команду `/mcp_servers` + прямой вызов админом; выходит за рамки T02 по прямому указанию
  задачи, отдельный тикет.

## 5. Изменения тестов (`bot/tests/test_mcp_server.py`)

Сейчас файл (383 строки, 15 тестов) вообще не покрывает stdio-путь: нет тестов на
`_fetch_stdio_tools`, `_get_or_create_session`, `_connect_to_server_stdio`, stdio-ветку
`call_mcp_function`. Все 15 текущих тестов проходят и под mcp 1.26, и под mcp 2.1.1 без единой
правки — их трогать не нужно. Добавить новые тесты, использующие реальные объекты
`mcp.types.Tool/ListToolsResult/CallToolResult/TextContent` (не заглушки-моки с произвольными
атрибутами — так тест ловит реальные переименования полей, а не собственные допущения):

```python
from mcp import types

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
```

Пятый тест (`test_get_or_create_session_uses_send_ping`) — единственный место, где `spec=[...]`
в `MagicMock` важен: без него `MagicMock` создаёт `.ping` автоматически и тест не ловит
регрессию к старому API.

`pytest.importorskip("mcp")` (:5) оставить как есть — после T01 `mcp` есть в
`requirements.txt`/`requirements-dev.txt`, но `importorskip` не мешает и защищает от случайного
локального запуска без зависимости.

## 6. Команды проверки

До T01 (нет `~/.venvs/ctb`) — прогонять новые/изменённые тесты в `/tmp/probe-mcp` (та же mcp
2.1.1, которую эта задача целевая):
```bash
~/.local/bin/uv pip install --python /tmp/probe-mcp/bin/python httpx pytest pytest-asyncio
cd /srv/git_projects/chatgpt-telegram-bot
/tmp/probe-mcp/bin/python -m pytest bot/tests/test_mcp_server.py -q
```
После T01 (венв `~/.venvs/ctb` с `mcp>=2.1,<3` из обновлённого `requirements.txt`):
```bash
~/.venvs/ctb/bin/python -m pytest bot/tests -q -p no:cacheprovider
~/.venvs/ctb/bin/python -m pytest bot/tests/test_mcp_server.py -v
```
Полный прогон (`tests/` + `bot/tests/`) — обязателен перед мержем по правилам AGENTS.md, даже
если правка локальна к одному плагину, т.к. `PluginManager` грузит все плагины при старте и
`bot/tests/test_mcp_server.py` не единственное место, ссылающееся на реестр плагинов:
```bash
~/.venvs/ctb/bin/python -m pytest -q
```

## 7. Риски

- **Конфликт зависимостей `httpx` vs `httpx2`.** Ложная тревога — это разные PyPI-пакеты
  (`httpx2`/`httpcore2` — новые форки, публикуемые отдельно от `httpx`/`httpcore`). `mcp==2.1.1`
  тянет `httpx2`/`httpcore2` транзитивно; `import httpx` в этом файле (:5) и во всём остальном
  проекте продолжает резолвиться в обычный `httpx`, версия которого регулируется T01 отдельно.
  Проверено установкой `mcp==2.1.1` в чистый venv — `httpx`/`httpcore` не подтягиваются и не
  конфликтуют.
- **`spec=` в `MagicMock` для регрессии на `.ping()`.** Если тест на send_ping написан без
  `spec=[...]`, `MagicMock().ping()` тихо вернёт ещё один `MagicMock`, тест позеленеет и ничего
  не проверит. Обязательно `spec=["send_ping"]` (см. §5).
- **`_mcp_call_result_to_dict` (§3.4) не входит в буквальный список T02.** Решение оставлено на
  усмотрение ревьюера — можно вынести в отдельный PR/коммит того же T02, но не пропускать
  молча: без него stdio-вызов тула технически "работает", но возвращает мусор.
- **Таймаут на stdio-сессию (§3.5) — новое поведение, не только починка API.** До этой правки
  зависший дочерний процесс мог держать `call_tool()`/`list_tools()` вечно; после — будет
  падать по таймауту `MCP_REQUEST_TIMEOUT` (по умолчанию 30 c). Если у кого-то есть
  легитимно долгие stdio-тулы (>30 c), им придётся поднять `MCP_REQUEST_TIMEOUT`. Пометить в
  changelog/README при внедрении.
- **Мультипроцессный побочный эффект стороннего пакета:** переход на `mcp>=2.1` — это T01, не
  T02; если T01 ещё не выполнен, `requirements.txt` продолжит резолвить `mcp>=1.0.0` в 1.26, а
  правки из §3.2-3.4 (снейк-кейс поля) **сломают код под 1.26**, потому что там поле называется
  `tool.inputSchema` (camelCase), а не `tool.input_schema`. **Порядок обязателен: T01 → T02**,
  как и написано в аудите ("Волна 2 ... после T01, на новом venv"). Если по каким-то причинам
  T02 нужно накатить раньше T01, единственный безопасный вариант — сделать
  `_mcp_input_schema_to_openai_parameters` версийно-независимым:
  `getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)` (не рекомендуется
  как постоянное решение — держать поддержку двух версий SDK ради временного разрыва между
  задачами того же аудита избыточно; проще выполнять по порядку).

## 8. Критерии готовности

- [ ] `_fetch_stdio_tools` использует `mcp_tools.tools` и `tool.input_schema` (не
  `tool.parameters`/`tool.required_parameters`), маппинг схемы — через
  `_mcp_input_schema_to_openai_parameters` (проверяет вложенные object/enum/required).
- [ ] `_get_or_create_session` вызывает `send_ping()`, не `ping()`.
- [ ] `call_mcp_function` (stdio-ветка) возвращает JSON-сериализуемый `dict`, не
  `CallToolResult` (решение по объёму — см. риск в §7, но если сделано, то именно так).
- [ ] `_fetch_server_tools`/HTTP-ветка `call_mcp_function` не изменены (см. §2.3/§3.6).
- [ ] `register_mcp_server` как модель-вызываемый тул не изменён (политика вне T02).
- [ ] Новые тесты из §5 добавлены и используют реальные `mcp.types.*`, а не самодельные
  дублирующие структуры; `MagicMock(spec=[...])` в тесте на `send_ping` без `ping` в spec.
- [ ] `requirements.txt` уже даёт `mcp>=2.1,<3` (T01 выполнен раньше — см. риск в §7).
- [ ] `~/.venvs/ctb/bin/python -m pytest -q` — весь набор тестов зелёный (не только
  `test_mcp_server.py` — плагин грузится PluginManager при старте, есть кросс-ссылки в
  `tests/test_no_hardcoded_plugin_refs.py` и `tests/test_plugin_manager.py`).
- [ ] `bot/README_MCP.md` не требует правок (документированный HTTP-протокол не менялся;
  если решите сделать §3.4 — стоит добавить фразу, что stdio-тулы теперь возвращают
  `structured_content`/текст, а не сырой объект — не обязательно для готовности T02, но полезно).

## Постскриптум после ревью (2026-09-04)

Реализовано по плану (mcp 2.x, `read_timeout_seconds`, нормализация схем через
`_mcp_input_schema_to_openai_parameters`). Ревью (Sonnet, persona reviewer): ошибок нет, одно
предупреждение — `_mcp_call_result_to_dict` (`bot/plugins/mcp_server.py`) разбирал только текстовые
блоки `content`; если MCP-инструмент вернул только картинку/аудио/ресурс, модель получала
`{"result": None}` без признака, что данные вообще были. Исправлено: нетекстовые блоки теперь помечаются в результате как пропущенные (см. `_mcp_call_result_to_dict`).
Замечания без действия: схемы с `$ref`/`$defs`/`anyOf`/пустые `{}` проходят `Draft7Validator.check_schema`;
тип `read_timeout_seconds` (`float | None`) подтверждён по `inspect.signature` в `mcp==2.1.1`.
