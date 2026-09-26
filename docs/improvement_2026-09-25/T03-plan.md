# T03. SSRF-защита исходящих HTTP-запросов — план реализации

Статус: план для разработчика. Ссылки на строки сверены с HEAD 08bc457 на момент
написания плана (2026-09-25); разработчик должен перечитать точку вызова перед правкой,
а не доверять номеру строки вслепую (номера могли сдвинуться из-за параллельных задач
волны 1 — T01/T02/T04 не трогают файлы этой задачи, поэтому дрейф маловероятен, но
проверка дешёвая).

## 0. Итоговое решение по «закреплению IP» (пиннингу) для async httpx

Закрепление IP — это когда мы резолвим hostname в IP заранее, проверяем, что IP не
приватный (не 127.0.0.1 / 169.254.x.x / 10.x и т.п.), а затем **физически подключаемся
именно к этому IP**, а не даём библиотеке резолвить домен второй раз перед соединением.
Без этого есть окно между проверкой и реальным подключением (DNS rebinding): атакующий
владеет доменом с очень коротким TTL и подсовывает публичный IP на проверку, а приватный
— на реальное подключение.

**Синхронный путь (`urllib`, унаследован от `skills.py`) уже умеет закреплять IP** —
кастомные `http.client.HTTPConnection`/`HTTPSConnection` с переопределённым `connect()`,
подключение к заранее проверенному IP, а для HTTPS — `ssl_context.wrap_socket(sock,
server_hostname=host)`, т.е. SNI и проверка сертификата остаются по настоящему имени
хоста. Переносим этот код в `net_safety.py` без изменений логики.

**Для асинхронного пути (httpx) устойчивое закрепление IP исследовано отдельно.**
Технически оно возможно: `httpcore.AsyncConnectionPool` принимает публичный именованный
параметр `network_backend` (проверено: `inspect.signature(httpcore.AsyncConnectionPool.__init__)`
в `~/.venvs/ctb`, httpcore 1.0.9); TLS-рукопожатие (`stream.start_tls(ssl_context,
server_hostname=...)`) происходит на уровне httpcore ПОСЛЕ `connect_tcp()`, используя
исходное имя хоста, а не IP — то есть подмена целевого IP в `connect_tcp()` кастомного
`AsyncNetworkBackend` не ломает SNI/проверку сертификата, аналогично urllib-варианту.

Решение: **не делать это в T03**, ограничиться проверкой URL до запроса + повторной
проверкой каждого шага редиректа (это прямо разрешено мастер-планом как fallback).
Причины:
- `httpx.AsyncHTTPTransport.__init__` не принимает `network_backend` — пришлось бы либо
  писать собственный `httpx.AsyncBaseTransport` (копируя ~30 строк `handle_async_request`
  из `httpx/_transports/default.py`), либо лезть в приватный атрибут `_pool` уже
  созданного транспорта. Оба варианта — код, завязанный на внутренности httpcore,
  без какого-либо покрытия тестами в проекте сегодня.
- Атака требует контроля атакующим авторитативного DNS с очень коротким TTL, точно
  подгаданным на окно между нашей проверкой и реальным connect (доли секунды) — узкий,
  дорогой в эксплуатации сценарий по сравнению с закрываемой сейчас дырой (полное
  отсутствие проверки вообще).
- Мастер-план явно разрешает этот fallback («если сложно... допускается»).

Это ограничение фиксируется в docstring `net_safety.py` (см. §1) — если требования
изменятся, есть готовое направление (custom `AsyncNetworkBackend`), но реализация вне
рамок T03.

## 1. `bot/net_safety.py` (новый модуль)

Модуль не читает переменные окружения сам — все лимиты/флаги передают вызывающие плагины
явно (каждый плагин продолжает сам читать свой `os.getenv`, как сейчас). Это держит
модуль чистым и тестируемым без monkeypatch окружения.

```python
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB — тот же порядок, что
                                       # openai_helper.session_log_max_bytes
```

### Исключения
- `UnsafeURLError(Exception)` — URL/редирект отклонён (схема, userinfo, приватный IP,
  DNS не резолвится, превышен лимит редиректов).
- `ResponseTooLargeError(Exception)` — тело ответа превысило `max_bytes`.
- `SafeHTTPStatusError(Exception)` — статус ответа ≥ 400 (аналог `httpx.HTTPStatusError`).
  Поля `.status_code: int`, `.text: str`, `.response` (self, чтобы код вида
  `exc.response.status_code` / `exc.response.text`, уже используемый в
  `mcp_server.py:770-771`, продолжал работать без переписывания).

### Резолв и проверка (синхронные и async-варианты)

```python
def resolve_public_ip(host: str) -> str | None:
    """Первый IP из getaddrinfo(host) с is_global==True, иначе None.
    Перенесено из SkillsPlugin._resolve_safe_ip (bot/plugins/skills.py:2068) без
    изменений: если в ответе DNS есть и приватные, и публичные адреса — берём первый
    публичный, не отказываем целиком (это НЕ то же самое, что validate_public_url)."""

async def resolve_public_ip_async(host: str) -> str | None:
    """asyncio.to_thread(resolve_public_ip, host) — getaddrinfo блокирующий."""

def validate_public_url(url: str, *, allow_private: bool = False) -> str | None:
    """None если URL безопасен, иначе причина отказа. Проверяет: схема http/https;
    отсутствие userinfo (user:pass@host — новая проверка, в текущем
    SkillsPlugin._validate_external_url её нет, добавляем по ТЗ мастер-плана);
    hostname присутствует; ВСЕ IP из getaddrinfo(host) — is_global (в отличие от
    resolve_public_ip: если среди ответов есть хоть один не-глобальный IP, весь URL
    отклоняется — это осознанно более строгая проверка для этапа валидации/редиректов).
    allow_private=True пропускает только проверку приватности IP (схему и userinfo
    проверяет всегда) — используется MCP-плагином для локальных серверов администратора."""

async def validate_public_url_async(url: str, *, allow_private: bool = False) -> str | None:
    """То же на resolve_public_ip_async — для вызовов из async-кода (safe_request),
    чтобы не блокировать event loop блокирующим getaddrinfo."""
```

Общую часть (разбор схемы/userinfo/hostname, цикл проверки `is_global` по списку
`getaddrinfo`) вынести в приватные хелперы `_check_url_shape()` и `_all_global(infos)`,
чтобы sync/async версии не расходились логикой по одной из них забытой правкой.

### Синхронный путь: `safe_urlopen`

```python
def safe_urlopen(
    url_or_request: str | urllib.request.Request,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
):
    """Перенос SkillsPlugin._safe_open (bot/plugins/skills.py:2084-2147) как есть:
    те же кастомные _PinnedHTTPConnection/_PinnedHTTPSConnection/_PinnedHTTPHandler/
    _PinnedHTTPSHandler/_Validating(HTTPRedirectHandler), тот же opener. Добавлено:
    1) вызов validate_public_url(url) ДО открытия (сейчас в skills.py эту проверку
       делает вызывающий код в _download_url_to_path — переносим внутрь, чтобы
       safe_urlopen был безопасен сам по себе, не полагаясь на дисциплину вызывающего);
       при отказе — raise UnsafeURLError(reason), а не молчаливый пропуск;
    2) opener.max_redirections = max_redirects;
    3) возвращаемый response оборачивается в _ByteCappedResponse(raw, max_bytes) —
       см. ниже. Except-ветки редиректа (_Validating.redirect_request) и «нет
       публичного IP» (_build_conn) НЕ переводятся на UnsafeURLError — оставляем
       urllib.error.HTTPError / urllib.error.URLError как в оригинале: это часть
       контракта urllib-фреймворка редиректов (HTTPRedirectHandler ожидает HTTPError
       из redirect_request), трогать рискованно без надобности.
    """
```

`_ByteCappedResponse` — internal, не экспортируется:
- Оборачивает сырой `http.client.HTTPResponse`-подобный объект.
- `read(amt=None)`: если `amt is None`, читает из `_raw` фиксированными кусками
  (`_CHUNK = 65536`) в цикле, накапливая суммарный счётчик; как только суммарный счётчик
  превышает `max_bytes` — `raise ResponseTooLargeError`, не дожидаясь EOF (иначе не
  потоковое чтение, а просто пост-фактум проверка после того, как уже всё скачано).
  Если `amt` задан (так делает `shutil.copyfileobj`, по умолчанию блоками 64 КБ) — читает
  `_raw.read(amt)` напрямую, обновляет счётчик, при превышении — тот же raise. Оба пути
  кода в skills.py используют либо `shutil.copyfileobj(response, target)`
  (`_download_url_to_path`, `bot/plugins/skills.py:2258`), либо `response.read()` без
  аргументов (`_github_default_branch`, `bot/plugins/skills.py:2061`) — оба должны
  оставаться рабочими.
- `__enter__`/`__exit__`/`close()` — делегируют в `_raw` (оба текущих вызова используют
  `with self._safe_open(...) as response:`).
- Остальные атрибуты (`.status`, `.headers`, …) — не проксируются явно, так как текущий
  код их не использует; если понадобятся — добавить `__getattr__` делегирование.

### Асинхронный путь: `safe_request` / `safe_get`

```python
async def safe_request(
    method: str,
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    headers: dict | None = None,
    json: Any = None,
    allow_private: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SafeResponse:
    """httpx.AsyncClient(transport=transport, follow_redirects=False, timeout=timeout).
    Ручной цикл редиректов (до max_redirects шагов): перед КАЖДЫМ запросом (включая
    первый и каждую цель редиректа) — validate_public_url_async(current_url,
    allow_private=allow_private); при отказе — UnsafeURLError. Запрос выполняется через
    client.stream(method, url, headers=headers, json=json) — потоковое чтение
    (aiter_bytes) с накоплением в bytearray и обрывом (ResponseTooLargeError) как
    только пройден max_bytes, НЕ через response.content (который вычитывает всё тело
    небуферизованно до какой-либо проверки размера). На 3xx с Location — резолвим
    относительный URL через httpx.URL(current_url).join(location), уменьшаем счётчик
    редиректов, повторяем. Превышение max_redirects — UnsafeURLError. Метод и JSON-тело
    при редиректе НЕ меняются (упрощение: не реализуем понижение POST→GET для 301/302,
    как это делают браузеры/requests — в реальных вызовах этого проекта редиректы на
    POST не ожидаются; задокументировать как известное упрощение).
    transport=... — только для тестов (httpx.MockTransport), в проде не передаётся.
    """

async def safe_get(url: str, **kwargs) -> SafeResponse:
    return await safe_request("GET", url, **kwargs)
```

`SafeResponse` — `@dataclass(frozen=True)` с полями `status_code: int`,
`headers: httpx.Headers`, `content: bytes`; свойство `text` (`content.decode("utf-8",
errors="replace")`), метод `json()` (`json.loads(self.text)`), метод `raise_for_status()`
(`raise SafeHTTPStatusError(status_code, text)` при `status_code >= 400`).

## 2. Миграция по файлам

### 2.1 `bot/plugins/skills.py` (владение: блок `:2067-2261` и точки вызова)

Импорт: добавить `from .. import net_safety` (или именованные импорты — на усмотрение
разработчика, лишь бы не расползлось по файлу). Импорты `http.client`, `socket`,
`ipaddress`, `urllib.error` — **удалить**: используются только внутри переносимого блока
(проверено — `python3` grep по файлу, других вхождений нет); `urllib.request` и
`urllib.parse` — оставить, используются вне блока (напр. `bot/plugins/skills.py:2058`
строит `urllib.request.Request(...)`, `_github_blob_info`/`_make_github_tree_url` и
другие используют `urllib.parse`).

Методы `_resolve_safe_ip` (`:2068`), `_safe_open` (`:2084`), `_validate_external_url`
(`:2149`) становятся тонкими обёртками (сигнатуры не меняются — их напрямую
monkeypatch'ат тесты `tests/test_skills_plugin.py`? нет, тесты патчат
`_download_url_to_path` целиком, но сигнатуры этих трёх методов используются внутри
самого skills.py: `_github_default_branch` вызывает `self._validate_external_url(api_url)`
и `self._safe_open(request, timeout=self.install_timeout)` на `:2056`/`:2060`):

```python
def _resolve_safe_ip(self, host):
    return net_safety.resolve_public_ip(host)

def _validate_external_url(self, url):
    return net_safety.validate_public_url(url)

def _safe_open(self, url, timeout):
    return net_safety.safe_urlopen(url, timeout=timeout, max_bytes=self.install_max_bytes)
```

Новое поле `self.install_max_bytes`, инициализируется в `initialize()`
(`bot/plugins/skills.py:137-141`, рядом с `self.install_timeout`):
```python
self.install_max_bytes = self._env_int(
    "SKILLS_INSTALL_MAX_BYTES", default=50_000_000, minimum=1_000_000,
)
```
(переиспользует существующий `_env_int` хелпер, `bot/plugins/skills.py:3659`).

`_download_url_to_path` (`:2248-2261`) — тело не меняется: явный предварительный вызов
`self._validate_external_url(url)` перед `self._safe_open(...)` остаётся (дублирует
проверку, которая теперь есть и внутри `safe_urlopen`, но это дёшево и сохраняет точный
текст текущей ошибки `"Refused to download skill source URL: {validation_error}"` без
риска его случайно изменить). `shutil.copyfileobj(response, target)` теперь пойдёт через
`_ByteCappedResponse.read()` и упадёт `ResponseTooLargeError` при превышении лимита —
ловится существующим общим `except Exception as exc:` (`:2259`), сообщение станет
`"Failed to download skill source URL: {exc}"` с текстом из `ResponseTooLargeError`.

Вызовы `_download_url_to_path`: `:1919` (install из произвольного URL) и `:2042`
(`_download_github_repo_source`, архив с `codeload.github.com`) — не трогать, сигнатура
не изменилась.

`_github_default_branch` (`:2054-2066`) — не трогать, обёртки совместимы.

### 2.2 `bot/plugins/codeinterpreter.py` (владение: только `download_file`)

`download_file` (`:811-856`), вызывается из `run_code` (`:726`, `if data_path.startswith('http')`).

Заменить тело блока `:840-849`:
```python
async with httpx.AsyncClient() as client:
    response = await client.get(url)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)
    logging.info(...)
    return save_path
```
на
```python
response = await net_safety.safe_get(url, max_bytes=self.max_download_bytes, timeout=30.0)
response.raise_for_status()
with open(save_path, 'wb') as f:
    f.write(response.content)
logging.info(...)
return save_path
```
`except httpx.HTTPError as e:` (`:851`) расширить до
`except (httpx.HTTPError, net_safety.SafeHTTPStatusError) as e:` — чтобы статус-ошибки
логировались через прежнюю ветку "Ошибка HTTP при скачивании файла", а не через общий
`except Exception`. `net_safety.UnsafeURLError`/`ResponseTooLargeError` можно оставить
падать в общий `except Exception as e:` (`:854`) — тот уже возвращает `None` и логирует,
поведение для вызывающего кода (`run_code`) не отличается от любой другой сегодняшней
ошибки скачивания.

`self.max_download_bytes` — добавить в `__init__` (`:87`):
`self.max_download_bytes = int(os.getenv("CODEINTERPRETER_MAX_DOWNLOAD_BYTES", 50_000_000))`.

Импорт: `from .. import net_safety`. `httpx` остаётся импортированным (нужен для
`except httpx.HTTPError`).

Известный смежный баг (НЕ чинить в T03, не в периметре задачи): если `download_file`
вернёт `None` (в т.ч. теперь из-за отказа SSRF-проверки), `run_code` (`:728`,
`os.path.splitext(data_path)`) упадёт на `None`, т.к. этот путь уже сегодня не
обрабатывает `None` от `download_file` ни для одной причины ошибки. Указать в отчёте
разработчика ревьюеру как отдельно найденный, но не устраняемый в этой задаче дефект.

### 2.3 `bot/plugins/text_summarizer.py` (владение: весь файл)

`_extract_text_from_url` (`:34-63`) — точка вызова `:39-40`:
```python
async with httpx.AsyncClient() as client:
    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
```
→
```python
response = await net_safety.safe_get(url, max_bytes=MAX_DOWNLOAD_BYTES, timeout=15.0)
response.raise_for_status()
soup = BeautifulSoup(response.content, 'html.parser')
```
(`doc = readability.Document(response.content)` ниже по функции не трогать —
`.content` совместимо, тип `bytes`.) Весь блок уже обёрнут в `except Exception as e:
... return ""` (`:61-63`) — новые типы исключений ловятся автоматически, менять
except не нужно.

`_generate_summary_url` (`:65-93`) — **не трогать**: HTTP-назначение — фиксированный
`https://300.ya.ru/api/sharing-url` (не пользовательский URL), это не SSRF-вектор.

Добавить `import os` (сейчас отсутствует) и модульную константу:
```python
MAX_DOWNLOAD_BYTES = int(os.environ.get("TEXT_SUMMARIZER_MAX_DOWNLOAD_BYTES", 10_000_000))
```
Импорт: `from .. import net_safety`.

Тестов на этот файл сейчас нет (`tests/test_text_summarizer_plugin.py` не существует) —
создать новый, см. §3.

### 2.4 `bot/plugins/mcp_server.py` (владение: HTTP-вызовы и регистрация URL)

`__init__` (`:66-80`) — добавить два поля рядом с `self.admin_ids`/`self.allowed_users`:
```python
self.allow_private_hosts = os.getenv("MCP_ALLOW_PRIVATE_HOSTS", "false").strip().lower() in {
    "1", "true", "yes", "on",
}  # тот же паттерн, что PluginManager.strict_validation, bot/plugin_manager.py:66
self.max_response_bytes = int(os.getenv("MCP_MAX_RESPONSE_BYTES", "10000000"))
```

`_fetch_server_tools` (`:821-847`) — заменить блок `:836-843`:
```python
async with httpx.AsyncClient(timeout=timeout) as client:
    response = await client.get(urljoin(base_url, "/tools"), headers=headers)
    response.raise_for_status()
    return response.json()
```
→
```python
response = await net_safety.safe_get(
    urljoin(base_url, "/tools"),
    headers=headers,
    timeout=timeout,
    max_bytes=self.max_response_bytes,
    allow_private=self.allow_private_hosts,
)
response.raise_for_status()
return response.json()
```
Существующий общий `except Exception as e: ... return []` (`:845-847`) уже покрывает
`UnsafeURLError`/`ResponseTooLargeError`/`SafeHTTPStatusError` — менять не нужно. Это
покрывает и «проверка при регистрации» (`register_server` HTTP-ветка `:608-626` не
делает собственного HTTP-вызова, только вызывает `_fetch_server_tools` на `:614`), и
«перед каждым HTTP-вызовом» для получения списка тулов.

`call_mcp_function` (`:701-776`) — HTTP-ветка `:732-776`, заменить блок `:753-758`:
```python
async with httpx.AsyncClient(timeout=timeout) as client:
    response = await client.post(urljoin(base_url, "/execute"), headers=headers, json=request_data)
    response.raise_for_status()
    result = response.json()
    return result
```
→
```python
response = await net_safety.safe_request(
    "POST",
    urljoin(base_url, "/execute"),
    headers=headers,
    json=request_data,
    timeout=timeout,
    max_bytes=self.max_response_bytes,
    allow_private=self.allow_private_hosts,
)
response.raise_for_status()
result = response.json()
return result
```
`except httpx.HTTPStatusError as e:` (`:765`) → `except net_safety.SafeHTTPStatusError as e:`
— тело блока (`:766-773`, читает `e.response.status_code`/`e.response.text`) не меняется,
т.к. `SafeHTTPStatusError.response` — self-ссылка с теми же полями. `except Exception as
e:` (`:774-776`) остаётся, ловит `UnsafeURLError`/`ResponseTooLargeError`.

Импорт: добавить `from .. import net_safety`; **удалить `import httpx`** (`:5`) — после
миграции обоих вызовов в файле не остаётся других обращений к `httpx.*` (проверено
grep — только импорт и два мигрируемых блока).

`bot/README_MCP.md` — не входит в список «Владение файлами» T03 буквально, но шаг 4
задачи прямо требует задокументировать там флаг `MCP_ALLOW_PRIVATE_HOSTS` (это,
по всей видимости, недосмотр мастер-плана — шаг требует правку файла, не перечисленного
в списке владения; правим, так как без этого документация вводит в заблуждение).
Правки:
- В блок «Переменные окружения» (`bot/README_MCP.md:29-43`) добавить строку:
  `MCP_ALLOW_PRIVATE_HOSTS=false` с комментарием — разрешает регистрацию/вызов серверов
  с `base_url`, который резолвится в приватный/loopback/link-local IP (нужно для
  локальных серверов вроде примера `DEFAULT_MCP_SERVERS=weather:http://localhost:8080,...`
  чуть выше в этом же блоке — иначе такие серверы после T03 перестанут отвечать на
  первый реальный HTTP-вызов).
- В разделе «Безопасность» (`bot/README_MCP.md:226-237`) — короткий абзац: HTTP-запросы
  к MCP-серверам по умолчанию отклоняют цели с приватными/loopback/link-local адресами
  (защита от SSRF); `MCP_ALLOW_PRIVATE_HOSTS=true` снимает это ограничение для доверенных
  локальных серверов администратора.

### 2.5 `bot/plugins/haiper_image_to_video.py` (владение: только скачивание `video_url`)

Внутри `_process_animate_command`, блок `:730-739`:
```python
temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
async with aiohttp.ClientSession() as session:
    async with session.get(video_url) as response:
        if response.status != 200:
            raise ValueError(self.t("haiper_video_download_failed", status=response.status))
        temp_file.write(await response.read())
temp_file.close()
```
→
```python
temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
response = await net_safety.safe_get(video_url, max_bytes=self.max_video_bytes, timeout=60.0)
if response.status_code != 200:
    raise ValueError(self.t("haiper_video_download_failed", status=response.status_code))
temp_file.write(response.content)
temp_file.close()
```
Своего `try/except` вокруг этого не нужно: и раньше, и теперь весь блок выполняется
внутри внешнего `try` метода `_process_animate_command` (`except Exception as e:` на
`:774`, форматирует `self.t("haiper_process_error", error=str(e), ...)`), а `finally`
(`:790-796`) уже подчищает `temp_file` по имени файла независимо от того, где именно
внутри `try` произошёл сбой — `UnsafeURLError`/`ResponseTooLargeError` от `safe_get`
обрабатываются этим существующим путём так же, как раньше обрабатывалась любая ошибка
`aiohttp.ClientError`.

`self.max_video_bytes` — добавить в `__init__` плагина (`:208`):
```python
self.max_video_bytes = int(os.getenv("HAIPER_MAX_VIDEO_BYTES", 49 * 1024 * 1024))
```
(49 МиБ — то же значение, что `agent_tools.DELIVERY_MAX_ARTIFACT_BYTES`,
`bot/plugins/agent_tools.py:234`, ограничение размера файла на отправку в Telegram; видео
всё равно не уйдёт пользователю, если больше).

Импорт: `from .. import net_safety`. `import aiohttp` (`:5`) — оставить, используется в
`_process_video_task` (`:456`) для опроса статуса задачи, не только для скачивания видео.

### 2.6 `bot/plugins/webshot.py` (владение: весь файл)

Отличается от остальных: пользовательский `url` никогда не становится хостом реального
соединения бота — он подставляется как ЧАСТЬ ПУТИ в фиксированный URL
`https://image.thum.io/get/maxAge/12/width/720/{kwargs["url"]}` (`:36`), скачивание
всегда идёт на `image.thum.io` (публичный, доверенный сторонний хост). Поэтому
`validate_public_url`/`net_safety` здесь не нужны — они защитили бы соединение к хосту,
который и так не атакуемый; SSRF-защиту цели фактически обеспечивает сам thum.io на
своей стороне. По мастер-плану для этого файла требуется только «лимит размера ответа» —
делаем именно это, без смены `requests`/`asyncio.to_thread` на `net_safety` (иначе
пришлось бы переписывать оба существующих зелёных теста в
`tests/test_webshot_plugin.py`, которые точно проверяют число вызовов `to_thread`/`get`
и их `timeout`).

Добавить константу (`os` уже импортирован, `:3`):
```python
MAX_WEBSHOT_BYTES = int(os.environ.get("WEBSHOT_MAX_IMAGE_BYTES", 8_000_000))
```
В `execute` (`:34-58`), после `if response.status_code == 200:` (`:44`), перед созданием
директории/записью файла:
```python
if response.status_code == 200:
    if len(response.content) > MAX_WEBSHOT_BYTES:
        return {'result': 'Unable to screenshot website'}
    if not os.path.exists("uploads/webshot"):
        ...
```
Известное упрощение (зафиксировать комментарием в коде): `requests.get(...)` без
`stream=True` уже буферизует весь ответ в память ДО этой проверки — лимит защищает от
пересылки/сохранения слишком большого файла, но не от кратковременного расхода памяти на
сам приём ответа. Полноценный потоковый вариант (`stream=True` + `iter_content`) возможен,
но требует переделывать оба существующих теста (`FakeRequests`/фейковый `response` без
`.iter_content`) — вне периметра «лимит размера ответа» из мастер-плана; не делать, если
не попросят отдельно.

### 2.7 `bot/plugins/github_analysis.py` (владение: весь файл)

`analyze_github_code` (`:63-120`), HTTP-блок `:70-91`:
```python
headers = ""
...
async with aiohttp.ClientSession() as session:
    async with session.get(url, headers=headers) as response:
        ...
        if response.status != 200:
            return {'error': f'Failed to fetch repository contents: {response.status}'}
        contents = await response.read()
        try:
            data = contents.decode('utf-8')
            contents = json.loads(data)
        except UnicodeDecodeError as e:
            ...
```
→
```python
try:
    response = await net_safety.safe_get(
        url, max_bytes=self.max_response_bytes, timeout=15.0,
    )
except (net_safety.UnsafeURLError, net_safety.ResponseTooLargeError) as exc:
    logging.error(f"Refused GitHub API request: {exc}")
    return {'error': f'Refused GitHub API request: {exc}'}
if response.status_code != 200:
    logging.info(f"Failed to fetch repository contents: {response.status_code}")
    return {'error': f'Failed to fetch repository contents: {response.status_code}'}
contents = response.content
try:
    data = contents.decode('utf-8')
    contents = json.loads(data)
except UnicodeDecodeError as e:
    logging.error(f"Decoding error: {e}")
    contents = ""
```
`headers = ""` (`:65`) — не осмысленный код (пустая строка, не словарь), убрать вместе с
заменой вызова: `safe_get`/`safe_request` принимает `headers: dict | None`, строка `""`
туда не годится по типу. Это вынужденная минимальная правка ради совместимости сигнатуры,
не самостоятельный рефакторинг — поведение (запрос без авторизации) не меняется, GitHub
API и раньше де-факто получал вызов без осмысленных заголовков.

`self.max_response_bytes` — добавить в `__init__` (`:17-20`, тот же стиль
`os.environ.get`, что уже используют `max_tokens`/`temperature` в этом файле):
```python
self.max_response_bytes = int(os.environ.get('GITHUB_ANALYSIS_MAX_RESPONSE_BYTES', 5_000_000))
```

Импорт: `from .. import net_safety`; **удалить `import aiohttp`** (`:4`) — единственное
использование было в мигрируемом блоке (проверено grep).

Хост здесь всегда `api.github.com` (собран в f-строке `:64`, не из пользовательского
ввода) — `owner`/`repo`/`path` только в PATH, не в hostname, так что `validate_public_url`
внутри `safe_get` всегда будет проходить без вопросов; `allow_private` не нужен.

Тестов на этот файл сейчас нет (`tests/test_github_analysis_plugin.py` не существует) —
создать новый, см. §3.

## 3. Тесты

### 3.1 `tests/test_net_safety.py` (новый)

Резолв/валидация (мок `socket.getaddrinfo` через `monkeypatch.setattr(net_safety.socket,
"getaddrinfo", fake)`):
- `test_resolve_public_ip_skips_private_and_returns_first_global` — ответ
  `[127.0.0.1, 8.8.8.8]` → `"8.8.8.8"`.
- `test_resolve_public_ip_returns_none_when_all_private` — только `127.0.0.1`/`10.0.0.1`
  → `None`.
- `test_resolve_public_ip_returns_none_on_gaierror`.
- `test_resolve_public_ip_async_uses_to_thread` — тот же сценарий через await, плюс
  проверка, что `asyncio.to_thread` реально используется (не блокирует напрямую) —
  например, через `monkeypatch` на `asyncio.to_thread` со счётчиком вызовов.
- `test_validate_public_url_rejects_non_http_scheme` — `"file:///etc/passwd"`,
  `"ftp://x/"`.
- `test_validate_public_url_rejects_missing_hostname` — `"http:///path"`.
- `test_validate_public_url_rejects_userinfo` — `"http://user:pass@example.com/"` (новая
  проверка, которой не было в `_validate_external_url`).
- `test_validate_public_url_rejects_loopback_v4` — getaddrinfo → `127.0.0.1`.
- `test_validate_public_url_rejects_loopback_v6` — getaddrinfo → `::1`.
- `test_validate_public_url_rejects_link_local_metadata` — `169.254.169.254` (облачный
  metadata-эндпоинт — главный практический сценарий SSRF).
- `test_validate_public_url_rejects_cgnat` — `100.64.0.1`.
- `test_validate_public_url_rejects_private_v4` — `10.0.0.1`.
- `test_validate_public_url_accepts_public_host` — getaddrinfo → `8.8.8.8` → `None`.
- `test_validate_public_url_allow_private_bypasses_ip_check_but_not_scheme` —
  `allow_private=True` + loopback → `None`; `allow_private=True` + `ftp://` → всё равно
  отказ (схема и userinfo проверяются всегда).
- `test_validate_public_url_async_matches_sync_semantics` — тот же набор случаев через
  await, с моком `socket.getaddrinfo` (не `resolve_public_ip_async` напрямую — иначе тест
  не проверяет реальный путь).

Синхронный путь:
- `test_safe_urlopen_rejects_unsafe_initial_url` — getaddrinfo → loopback; дополнительно
  `monkeypatch.setattr(net_safety.socket, "create_connection", spy)` со `spy`, кидающим
  `AssertionError`, чтобы доказать: до реального соединения дело не доходит,
  `UnsafeURLError` летит раньше.
- `test_safe_urlopen_enforces_max_bytes_on_read_with_amt` — фейковый `_raw`-объект
  (простой класс с `.read(amt)`, отдающий данные по кусочкам) обёрнутый в
  `_ByteCappedResponse`; вызовы `.read(8192)` в цикле (имитация `shutil.copyfileobj`) →
  `ResponseTooLargeError` при превышении, до исчерпания источника.
- `test_safe_urlopen_enforces_max_bytes_on_read_without_amt` — то же для `.read()` без
  аргументов (путь `_github_default_branch`), проверить, что `_raw.read` вызывается
  кусками ограниченного размера, а не один раз без лимита (иначе это не потоковое
  чтение, а пост-проверка).
- `test_safe_urlopen_is_context_manager` — `with` работает, `_raw.close()` вызывается.

(Полноценная симуляция редиректа через настоящий `urllib`-opener без сети — избыточно
сложно и не даёт прироста покрытия сверх уже протестированного `validate_public_url`,
который `_Validating.redirect_request` вызывает как есть; не пишем отдельный тест на это
для sync-пути — только для async, где логика редиректов написана заново, а не
унаследована от urllib.)

Асинхронный путь (`httpx.MockTransport`, по образцу `tests/test_hindsight_client.py`):
- `test_safe_get_returns_content_and_status` — happy path, `getaddrinfo` → публичный IP.
- `test_safe_get_raise_for_status_matches_error_shape` — 500 → `SafeHTTPStatusError`,
  `.response.status_code == 500`, `.response.text` содержит тело.
- `test_safe_get_rejects_unsafe_initial_url_without_network_call` — getaddrinfo →
  loopback; transport-handler кидает `AssertionError`, если вызван — доказывает, что
  запрос не уходит вообще.
- `test_safe_get_enforces_max_bytes` — handler отдаёт тело больше `max_bytes` →
  `ResponseTooLargeError`.
- `test_safe_get_rejects_redirect_to_private_ip` — handler: первый хост → 302 на
  `http://internal.example/...`; `getaddrinfo` для `internal.example` → `127.0.0.1`;
  handler кидает `AssertionError`, если получает запрос на `internal.example` →
  доказывает, что второй прыжок не выполняется.
- `test_safe_get_follows_redirect_to_public_target` — 302 на публичный хост → итоговый
  ответ с этого хоста, `status_code`/`content` от финального прыжка.
- `test_safe_get_too_many_redirects_raises` — handler всегда отвечает 302 на себя же
  (публичный IP); `max_redirects=2` → `UnsafeURLError` после превышения, а не бесконечный
  цикл.
- `test_safe_request_post_sends_json_body` — handler проверяет `request.method == "POST"`
  и `json.loads(request.content) == {...}`.
- `test_safe_request_allow_private_permits_loopback` — `allow_private=True` + loopback +
  handler отвечает 200 → успех (сценарий MCP с `MCP_ALLOW_PRIVATE_HOSTS=true`).

### 3.2 `tests/test_skills_plugin.py` (существующие тесты — не трогать, только добавить)

Существующие тесты (`:1221`, `:1531`, `:1560`, `:1590`, `:1726`) монкейпатчат
`plugin._download_url_to_path` напрямую как атрибут экземпляра — сигнатура метода не
меняется, значит все они остаются зелёными без правок.

Добавить:
- `test_download_url_to_path_delegates_to_net_safety` — НЕ патчить
  `_download_url_to_path`, вместо этого монкейпатчить `skills.net_safety.safe_urlopen` на
  фейк, вызвать `plugin._download_url_to_path(url, tmp_path)` напрямую, убедиться, что
  дошло до `net_safety.safe_urlopen` с ожидаемым `url`/`timeout`/`max_bytes` — закрывает
  риск, что обёртка внутри skills.py случайно не вызывает настоящий модуль.
- `test_validate_external_url_wrapper_matches_net_safety` — то же для
  `_validate_external_url`/`_resolve_safe_ip` (можно один параметризованный тест на оба).
- `test_install_skill_rejects_url_resolving_to_private_ip` — сквозной тест: НЕ патчить
  `_download_url_to_path`, замокать `socket.getaddrinfo` на приватный IP, вызвать
  `execute("install_skill", package="http://169.254.169.254/x.zip", ...)`, убедиться в
  `result["success"] is False` и вменяемом тексте ошибки — единственный тест, который
  реально проверяет, что вся цепочка (skills → net_safety) работает end-to-end, а не
  только то, что моки вызваны.

### 3.3 `tests/test_codeinterpreter_plugin.py`

- `test_download_file_uses_safe_get_and_writes_content` — монкейпатч
  `codeinterpreter.net_safety.safe_get` на фейковую корутину, возвращающую заданные байты
  и `status_code=200`; вызвать `plugin.download_file(url)`; проверить, что файл записан и
  `safe_get` вызван с этим `url`.
- `test_download_file_returns_none_on_unsafe_url` — фейк `safe_get`, кидающий
  `net_safety.UnsafeURLError`; `download_file` возвращает `None`, файл не создан.
- `test_download_file_returns_none_on_response_too_large` — то же для
  `ResponseTooLargeError`.

### 3.4 `tests/test_text_summarizer_plugin.py` (новый)

- `test_extract_text_from_url_uses_safe_get` — `httpx.MockTransport` (модуль патчить не
  обязательно, если `safe_get` принимает `transport=` — но `_extract_text_from_url` сам
  не прокидывает `transport` наружу, поэтому здесь проще замокать
  `text_summarizer.net_safety.safe_get` напрямую фейковой корутиной с HTML-телом,
  проверить, что `BeautifulSoup` находит `summary-scroll`).
- `test_extract_text_from_url_returns_empty_string_on_unsafe_url` — фейк `safe_get`
  кидает `UnsafeURLError` → результат `""` (существующий генерик `except Exception`).

### 3.5 `bot/tests/test_mcp_server.py`

`test_call_mcp_function` (`:267-315`) **требует переписывания** — сейчас мокает
`httpx.AsyncClient` целиком и проверяет `mock_client.post.assert_called_once()`; после
миграции на `net_safety.safe_request` (который использует `client.stream(...)`, а не
`client.post(...)`) этот мок ничего не перехватит, и `patch('httpx.AsyncClient', ...)`
всё ещё технически сработал бы (т.к. `net_safety.py` тоже делает `httpx.AsyncClient(...)`
по модульному имени), но конкретные ожидания на `.post` — нет. Переписать на
`httpx.MockTransport`, по образцу `tests/test_hindsight_client.py`:
```python
def handler(request):
    assert request.method == "POST"
    assert str(request.url) == "http://example.com/execute"
    assert request.headers["authorization"] == "Bearer test_key"
    body = json.loads(request.content)
    assert body["name"] == "test_function"
    assert body["arguments"]["param1"] == "value1"
    return httpx.Response(200, json={"result": "success"})

monkeypatch.setattr(
    "bot.plugins.mcp_server.net_safety.safe_request",
    lambda *a, **kw: net_safety.safe_request(*a, **kw, transport=httpx.MockTransport(handler)),
)
```
(или проще — если `call_mcp_function` не прокидывает `transport` наружу, монкейпатчить
`socket.getaddrinfo` для `example.com` на публичный IP и патчить сам `httpx.AsyncClient`
конструктор внутри `net_safety` модуля через `monkeypatch.setattr(net_safety.httpx,
"AsyncClient", ...)`, возвращая клиент с реальным `httpx.MockTransport` —
на усмотрение разработчика, любой вариант допустим, лишь бы не требовал реальной сети).
Остальные тесты этого файла (`test_register_server*`, `test_refresh_server_tools*`)
монкейпатчат `_fetch_server_tools`/`_invalidate_function_index` напрямую — не
затрагиваются, остаются зелёными без правок.

Добавить:
- `test_call_mcp_function_rejects_private_base_url_by_default` — `server["base_url"] =
  "http://127.0.0.1:9999"`, `getaddrinfo` не мокать (loopback и так не `is_global`) →
  результат содержит `error`, реальный HTTP-запрос не выполняется (transport-handler
  кидает `AssertionError`, если вызван).
- `test_call_mcp_function_allows_private_base_url_with_flag` — тот же URL,
  `plugin.allow_private_hosts = True` (или через `monkeypatch.setenv` + пересоздание
  плагина) → запрос доходит до transport-handler.
- `test_fetch_server_tools_rejects_private_base_url_by_default` — аналогично для
  `_fetch_server_tools`/`register_server`.

### 3.6 `tests/test_webshot_plugin.py`

Существующие два теста (`:45-85`) не трогать. Добавить:
- `test_execute_rejects_oversized_response` — `FakeRequests` возвращает
  `SimpleNamespace(status_code=200, content=b"x" * (MAX_WEBSHOT_BYTES + 1))`
  (`monkeypatch.setattr(webshot, "MAX_WEBSHOT_BYTES", 10)` для маленького порога в
  тесте, не тянуть реальные 8 МБ) → результат `{'result': 'Unable to screenshot
  website'}`, файл в `uploads/webshot` не создан.

### 3.7 `tests/test_haiper_image_to_video_async_db.py` (или новый файл рядом)

- `test_process_animate_command_downloads_video_via_safe_get` — монкейпатч
  `haiper_image_to_video.net_safety.safe_get`, проверить вызов с `video_url` и
  `max_bytes=plugin.max_video_bytes`, проверить, что байты попадают в отправляемый файл.
- `test_process_animate_command_reports_unsafe_video_url` — фейк `safe_get` кидает
  `UnsafeURLError` → сообщение через `haiper_process_error` (существующий путь), временный
  файл удалён (`finally`-блок).

Оба теста потребуют минимального харнеса для `_process_animate_command` (message/task
фикстуры) — свериться со стилем существующих фикстур в этом файле (`FakeMessage`,
`FakeQuery`, `:33-78`) и/или с тем, как тестируются другие ветки этого метода, если такие
тесты уже есть в других файлах пакета (поиск не проводился глубже — разработчику
проверить `grep -l _process_animate_command tests/ bot/tests/` перед написанием, чтобы не
дублировать существующий харнес).

### 3.8 `tests/test_github_analysis_plugin.py` (новый)

- `test_analyze_github_code_uses_safe_get` — монкейпатч `github_analysis.net_safety.safe_get`
  фейком, возвращающим JSON-тело со списком файлов; проверить, что `owner`/`repo`/`path`
  корректно собираются в URL, переданный в `safe_get`.
- `test_analyze_github_code_reports_unsafe_url` — фейк кидает `UnsafeURLError` → `{'error':
  ...}` с этим текстом, `analyze_code_with_chatgpt` не вызывается.

## 4. Приёмка

```bash
~/.venvs/ctb/bin/python -m pytest tests/test_net_safety.py tests/test_skills_plugin.py \
  tests/test_codeinterpreter_plugin.py tests/test_text_summarizer_plugin.py \
  tests/test_webshot_plugin.py tests/test_haiper_image_to_video_async_db.py \
  tests/test_github_analysis_plugin.py bot/tests/test_mcp_server.py \
  -q --no-header -p no:cacheprovider

~/.venvs/ctb/bin/python -m pytest tests -q --no-header -p no:cacheprovider   # полный прогон

~/.venvs/ctb/bin/python -m ruff check bot/net_safety.py bot/plugins/skills.py \
  bot/plugins/codeinterpreter.py bot/plugins/text_summarizer.py bot/plugins/mcp_server.py \
  bot/plugins/haiper_image_to_video.py bot/plugins/webshot.py bot/plugins/github_analysis.py \
  tests/test_net_safety.py

python3 -m mypy bot/net_safety.py bot/plugins/skills.py bot/plugins/codeinterpreter.py \
  bot/plugins/text_summarizer.py bot/plugins/mcp_server.py \
  bot/plugins/haiper_image_to_video.py bot/plugins/webshot.py bot/plugins/github_analysis.py \
  --python-executable ~/.venvs/ctb/bin/python --ignore-missing-imports
# сравнить с /tmp/impl/mypy_before.txt по этим же файлам — новых ошибок появляться не должно
# (T01 ещё не поставил baseline-скрипт в волне 1, сверка вручную по списку файлов)
```

Готово, когда: все команды выше проходят; `git status --short` показывает изменения
только в файлах раздела 2 плюс новые тестовые файлы; ни один `tools:`/`get_spec()` не
менялся.

## 5. Риски

- **Локальные MCP-серверы.** `MCP_ALLOW_PRIVATE_HOSTS=false` по умолчанию — это смена
  поведения: сегодня `DEFAULT_MCP_SERVERS=weather:http://localhost:8080,...` (пример из
  `bot/README_MCP.md:39`) работает, после T03 такие серверы перестанут отвечать на первый
  же реальный вызов, пока админ не выставит флаг. Явно задокументировано в README (§2.4).
- **Синхронный DNS в асинхронном коде.** `resolve_public_ip`/`validate_public_url`
  (не-async варианты) делают блокирующий `socket.getaddrinfo`. Используются напрямую
  только из `safe_urlopen` (уже синхронный по своей природе, install-путь, вызывается
  редко и из административного действия) — контролируемо. Все места, где проверка идёт
  из `async def` (весь async-путь mcp_server/text_summarizer/codeinterpreter/
  github_analysis/haiper), обязаны идти через `*_async`-варианты (`safe_get`/
  `safe_request` делают это сами внутри) — ревьюеру стоит явно проверить, что ни один
  новый вызов не тянет синхронный `validate_public_url`/`resolve_public_ip` напрямую из
  async-функции в затронутых плагинах.
- **DNS rebinding в async-пути остаётся неполностью закрытым** — см. §0, осознанное и
  задокументированное ограничение, не баг.
- **`webshot.py` не получает host-валидацию** — осознанное решение (см. §2.6), но при
  ревью может выглядеть как непоследовательность с остальными файлами; в коде и здесь
  зафиксировано объяснение.
- **`bot/tests/test_mcp_server.py::test_call_mcp_function`** — единственный тест во всей
  задаче, который ломается гарантированно (не просто теоретически) заменой реализации;
  явно выделен в §3.5 с готовым направлением переписывания, чтобы разработчик не тратил
  время на диагностику, почему он красный.
- **Порядок волны 1.** T01/T02/T04 идут параллельно в том же дереве, но не пересекаются
  по файлам с T03 (сверено со списками владения в мастер-плане) — конфликтов слияния не
  ожидается.
