# T03. SSRF-защита — ревью

## Раунд 1

Проверено: `git diff HEAD` по всем файлам владения T03, новые/изменённые тесты, план
`docs/improvement_2026-09-25/T03-plan.md` и мастер-план. Прогнаны: тесты области T03
(169 passed), полный `pytest tests` (1707 passed), `ruff check` по всем изменённым файлам
(чисто), `mypy` по 8 файлам T03 с построчным сравнением с `/tmp/impl/mypy_before.keep`
(нормализовано без номеров строк — 0 новых ошибок, 1 старая ошибка в skills.py пропала
как побочный эффект делегирования типа в net_safety).

Итог: 0 ERROR, 2 WARNING, 1 NIT.

### WARNING 1 — `is_global` пропускает NAT64-адреса с приватным embedded IPv4

`bot/net_safety.py:94` (`resolve_public_ip`) и `bot/net_safety.py:135` (`_all_global`,
используется `validate_public_url`/`validate_public_url_async`) полагаются на
`ipaddress.IPv6Address.is_global`. Для адресов в известном NAT64-префиксе `64:ff9b::/96`
(RFC 6052 — механизм трансляции IPv6→IPv4, распространён в IPv6-only облачных
сетях/подсетях с DNS64) этот метод стандартной библиотеки возвращает `True`, даже если
встроенный IPv4-адрес — приватный/loopback/link-local. Проверено напрямую:

```python
>>> import ipaddress
>>> ipaddress.ip_address("64:ff9b::169.254.169.254").is_global
True   # а должно быть False — это облачный metadata-эндпоинт
>>> ipaddress.ip_address("64:ff9b::127.0.0.1").is_global
True   # loopback
```

Сценарий отказа: домен атакующего резолвится (AAAA-запись) в `64:ff9b::a9fe:a9fe`
(= `64:ff9b::169.254.169.254`). И `validate_public_url` (используется в `safe_urlopen` и
как первичная/редиректная проверка в `safe_request`), и `resolve_public_ip` (пин IP в
`safe_urlopen`/`_build_conn`) пропускают такой адрес как «глобальный». На сетях с
NAT64/DNS64-шлюзом (нередкая конфигурация IPv6-only контейнерных/VPC-окружений, mobile
carrier grade NAT) соединение реально транслируется на приватный/metadata-адрес — рабочий
SSRF-байпас на облачный metadata-эндпоинт. На сети без NAT64-шлюза соединение просто не
устанавливается (timeout), т.е. эксплуатируемость зависит от сетевого окружения
развёртывания — поэтому WARNING, а не ERROR.

Это не упомянуто явно в плане/фокусе задачи (там перечислены IPv4-mapped IPv6, decimal/
octal-хосты — оба на деле корректно закрыты, т.к. код всегда доверяет резолву через
`getaddrinfo`, а не буквальному парсингу хоста), но это тот же класс проблемы
(embedded-адрес обманывает проверку `is_global`) и в прямом периметре задачи «SSRF».

Предлагаемое исправление (не самое дешёвое, но точечное): в `_all_global`/
`resolve_public_ip` дополнительно проверять принадлежность `ipaddress.ip_network("64:ff9b::/96")`
и для таких адресов доставать нижние 32 бита как `IPv4Address`, проверяя `is_global` уже
у него — общий хелпер, который используют обе функции, чтобы не разойтись логикой.

### WARNING 2 — необработанный `UnicodeError` из `getaddrinfo` (не регрессия T03, но расширенный периметр)

`resolve_public_ip` (`bot/net_safety.py:84-87`), `validate_public_url`
(`bot/net_safety.py:153-156`), `validate_public_url_async` (`bot/net_safety.py:169-172`) —
везде `socket.getaddrinfo(host, None)` обёрнут только в `except socket.gaierror`. Для
хоста с меткой (label) длиннее 63 октетов `getaddrinfo` поднимает не `gaierror`, а
`UnicodeError` (IDNA-кодирование хоста падает раньше системного резолва) — он никем не
перехватывается и вылетает из `validate_public_url*`/`resolve_public_ip` наружу как
необработанное исключение вместо ожидаемой по контракту строки-причины отказа/`None`.
Проверено:

```python
>>> from bot import net_safety
>>> net_safety.validate_public_url("http://" + "a"*300 + ".com/x")
UnicodeError: label empty or too long
```

Это дословно перенесённая (без изменений логики, как и требует план §2.1) старая логика
`SkillsPlugin._validate_external_url`/`_resolve_safe_ip` — баг существовал и до T03,
регрессией не является. Но раньше этот путь использовал только `skills.py` install-флоу;
после T03 та же функция (и та же дыра в обработке исключений) обслуживает ещё 5 плагинов
через `safe_get`/`safe_request`. У всех новых потребителей вызов обёрнут широким
`except Exception` в точке использования (codeinterpreter, text_summarizer, mcp_server,
haiper, github_analysis — везде проверено по диффу), поэтому там `UnicodeError`
благополучно ловится наравне с любой другой ошибкой скачивания и не всплывает.

Конкретно пробита эта дыра в `bot/plugins/skills.py:2146-2158`
(`_download_url_to_path`, не тронут T03): `validation_error = self._validate_external_url(url)`
на `:2147` стоит ДО начала `try` на `:2153` — значит `UnicodeError` от слишком длинного
хоста в `package` (`install_skill`, ввод обычного пользователя Telegram, не только
админа) вылетает из `_download_url_to_path` необработанным, дальше вверх по стеку через
`execute()` (там вообще нет try/except) и гасится только на уровне
`asyncio.gather(..., return_exceptions=True)` в `bot/openai_tool_handler.py:218` — то есть
конкретный tool-call падает с сырым исключением вместо аккуратного
`{"success": False, "error": "Refused to download skill source URL: ..."}`, которое
возвращают все остальные ветки отказа в этой функции. Само по себе SSRF не открывает
(соединение не устанавливается ни при каком раскладе), только ломает graceful degradation
для одного конкретного класса невалидных URL. Voспроизведено:

```python
>>> plugin._download_url_to_path("http://" + "a"*300 + ".com/skill.zip", tmp_path)
UnicodeError: label empty or too long   # вместо (None, "Failed to download ...")
```

Дешёвое исправление в одном месте — расширить все три `except socket.gaierror` в
`net_safety.py` до `except (socket.gaierror, UnicodeError)` — закрыло бы это сразу для
всех 6 потребителей, включая `skills.py`, без правки самого `skills.py`.

### NIT — не добавлен комментарий про буферизацию в `webshot.py`

План (§2.6) явно просит зафиксировать комментарием в коде известное упрощение:
`requests.get(...)` без `stream=True` уже буферизует весь ответ в память до проверки
`len(response.content) > MAX_WEBSHOT_BYTES` (`bot/plugins/webshot.py:47`), т.е. лимит не
защищает от кратковременного расхода памяти на приём самого ответа. В диффе такого
комментария нет. Поведенчески не влияет (сознательно допущенное планом упрощение), только
расходится с буквой плана.

### Что проверено и не вызвало вопросов

- IPv4-mapped IPv6 (`::ffff:127.0.0.1` и т.п.) — корректно отклоняются, `is_global`
  правильно разворачивает mapped-адрес (проверено эмпирически).
- Decimal/octal/hex host-нотация (`2130706433`, `0x7f000001`, `017700000001`) — не
  байпас: код никогда не парсит хост литералом, только через `getaddrinfo`, который на
  этой системе (glibc) канонизирует такие хосты в обычный dotted-decimal перед тем, как
  `ipaddress.ip_address` их увидит.
- DNS, возвращающий смешанный публичный+приватный набор — `validate_public_url`/`_async`
  через `_all_global` отклоняет URL целиком, если хоть один адрес не глобальный (строже,
  чем `resolve_public_ip`, как и задумано планом).
- Редирект на не-http(s) схему — отклоняется на каждой итерации (`_check_url_shape`) и в
  sync-, и в async-пути.
- Относительные/protocol-relative редиректы — `httpx.URL(...).join(location)` (async) и
  встроенный `urllib.parse.urljoin` внутри `HTTPRedirectHandler.http_error_302` (sync,
  до вызова кастомного `redirect_request`) корректно разворачивают их в абсолютный URL
  перед повторной валидацией.
- Блокирующий DNS в async-коде — грепом по всем 7 файлам плагинов подтверждено: только
  `skills.py`'s синхронные обёртки (`_resolve_safe_ip`/`_safe_open`/`_validate_external_url`)
  используют sync-варианты `net_safety`, а сам install-путь целиком уходит в
  `asyncio.to_thread` (`bot/plugins/skills.py:1523`). Ни один из 5 остальных плагинов не
  вызывает sync-варианты напрямую из `async def` — все идут через `safe_get`/`safe_request`.
- Байт-кап при потоковом чтении — sync (`_ByteCappedResponse`, тест на чтение кусками и с
  `amt=None`) и async (`aiter_bytes` + инкрементальная проверка `len(content) > max_bytes`
  внутри `async with client.stream(...)`, до `response.content`) оба обрывают до полного
  вычитывания тела, не пост-фактум.
- Очистка соединений — оба пути закрывают ресурсы через контекст-менеджеры
  (`_ByteCappedResponse.__exit__`/`close`, `async with httpx.AsyncClient(...)` +
  `async with client.stream(...)`) независимо от того, штатно завершился запрос или упал
  `ResponseTooLargeError`/`UnsafeURLError`.
- userinfo (`user:pass@host`) — новая проверка, отклоняет весь URL при наличии `@` в
  authority-части независимо от того, что стоит по разные стороны от `@` (безопаснее, чем
  пытаться угадать, какая часть настоящий хост).
- Тексты ошибок не содержат чувствительных внутренних деталей (в частности,
  `validate_public_url` больше не включает в сообщение конкретный отклонённый IP, в
  отличие от старого `_validate_external_url` — небольшое улучшение, не регрессия).
- `mcp_server.py`, `codeinterpreter.py`, `github_analysis.py`, `text_summarizer.py`,
  `haiper_image_to_video.py`, `bot/README_MCP.md` — миграция на `net_safety` соответствует
  плану дословно; лимиты байт/флаг `MCP_ALLOW_PRIVATE_HOSTS` подключены везде, где
  требовал план; `get_spec()`/`tools:` не менялись.
- Тесты (`tests/test_net_safety.py` и новые/изменённые тесты по плагинам) реально
  проверяют защитное поведение (через `AssertionError`-ловушки в HTTP-хендлерах,
  подтверждающие, что запрос вообще не ушёл в сеть), а не только что моки были вызваны.

## Раунд 2

Проверено: все три находки раунда 1 — построчно прочитан текущий `bot/net_safety.py`
(новый файл целиком) и связанные тесты; `git diff HEAD` по всем файлам владения T03
перечитан заново на предмет новых/пропущенных проблем сверх раунда 1. Прогнаны: тесты
области T03 (185 passed, было 169 — прирост от новых NAT64/UnicodeError/ValueError-тестов),
полный `pytest tests` (1727 passed), `ruff check` по всем изменённым файлам T03 (чисто),
`mypy` по 8 файлам области T03 с построчным сравнением с `/tmp/impl/mypy_before.keep`
(нормализовано по `(файл, код ошибки)` без номеров строк — 0 новых/выросших пар ошибок;
1 пара `bot/plugins/skills.py`/`[return-value]` пропала, как и в раунде 1).

Итог: 0 ERROR, 0 WARNING, 0 NIT. Все три находки раунда 1 исправлены и закрыты тестами.

### WARNING 1 (раунд 1) — исправлено

`bot/net_safety.py:85-116` добавляет `_nat64_embedded_ipv4()` (обе NAT64-подсети —
well-known `64:ff9b::/96` и local-use `64:ff9b:1::/48`, RFC 6052 §2.2) и `_is_global_ip()`
— общий хелпер, которым теперь пользуются и `resolve_public_ip` (`:139`), и `_all_global`
(`:180`), так что sync- и async-проверки (`validate_public_url`/`_async`, оба идут через
`_all_global`) не могут разойтись логикой. Разработчик расширил защиту тем же приёмом
и на 6to4 (`2002::/16`, через встроенный `ip.sixtofour`) и Teredo (`2001::/32`, через
`ip.teredo`, требует глобальности и сервера, и клиента) — оба являются тем же классом
проблемы (embedded IPv4 обманывает `is_global`), что и упомянуто в тексте находки
(«тот же класс проблемы... в прямом периметре задачи»), поэтому это не выход за рамки
находки, а её полное закрытие для всех стандартных механизмов встраивания.

Проверено вручную побитово (см. запуск ниже) — `_nat64_embedded_ipv4` для well-known
префикса берёт `packed[12:16]` (биты 96-127, PL=96 по таблице RFC 6052 §2.2), для
local-use — `packed[6:8] + packed[9:11]` (биты 48-63 + 72-87 вокруг зарезервированного
байта "u" на битах 64-71, PL=48) — совпадает с эталонной таблицей RFC. Тесты
`tests/test_net_safety.py:170-227` (NAT64 well-known/local-use, 6to4, Teredo — по паре
приватный/публичный embedded-адрес на каждый механизм) используют предвычисленные
адреса и присланный auto-getaddrinfo-мок; пересчитал независимо оба NAT64-адреса
(`64:ff9b:1:a9fe:a9:fe00::` → embedded `169.254.169.254`, `64:ff9b:1:808:8:800::` →
embedded `8.8.8.8`) и Teredo-адреса (`...:5601:5601` → client `169.254.169.254` через
XOR c `0xffffffff`, `...:f7f7:fbfb` → client `8.8.4.4`) — значения верны.

```python
>>> import ipaddress
>>> ipaddress.ip_address("64:ff9b::169.254.169.254").packed.hex()
'0064ff9b0000000000000000a9fea9fe'   # packed[12:16] = a9fea9fe = 169.254.169.254
```

### WARNING 2 (раунд 1) — исправлено

Все три места (`resolve_public_ip:126-132`, `validate_public_url:198-203`,
`validate_public_url_async:216-221`) расширили `except socket.gaierror` до
`except (socket.gaierror, UnicodeError, ValueError)` с комментарием, ссылающимся на
находку раунда 1. Дословно воспроизведён баг-сценарий из находки:
`tests/test_skills_plugin.py::test_download_url_to_path_handles_unicode_error_like_other_validation_errors`
вызывает `plugin._download_url_to_path("http://" + "a"*300 + ".example/skill.zip", tmp_path)`
БЕЗ монкейпатча `getaddrinfo` (настоящая IDNA-ошибка от слишком длинной метки хоста) и
проверяет `error.startswith("Refused to download skill source URL:")` — именно та
graceful-деградация, которой не было до фикса (находка показывала сырой `UnicodeError`
из этой же точки). Прогнано — тест проходит:

```
tests/test_skills_plugin.py::test_download_url_to_path_handles_unicode_error_like_other_validation_errors PASSED
```

`ValueError` в тот же except добавлен сверх буквы находки (которая просила только
`UnicodeError`) — защитное расширение того же самого except-блока на соседний класс
ошибок `getaddrinfo`, с отдельными тестами
(`test_resolve_public_ip_returns_none_on_value_error`,
`test_validate_public_url_returns_reason_on_value_error`) через явный фейк, а не
воспроизведённый сценарий (на данной платформе `socket.getaddrinfo` не наблюдался
кидающим `ValueError` ни на одном опробованном плохом хосте). Не вредит — тот же
принцип «трактуем как ошибку резолва», ни одного поведенческого риска не создаёт.

### NIT (раунд 1) — исправлено

`bot/plugins/webshot.py:47-50` теперь содержит требуемый комментарий (буферизация всего
ответа в память до проверки размера, `requests.get` без `stream=True`) — ровно то, что
просил план §2.6.

### Полный передиагностический просмотр диффа (сверх находок раунда 1)

Построчно сверены с планом все файлы владения: `bot/plugins/codeinterpreter.py`,
`bot/plugins/text_summarizer.py`, `bot/plugins/mcp_server.py`,
`bot/plugins/haiper_image_to_video.py`, `bot/plugins/github_analysis.py`,
`bot/README_MCP.md`, `bot/tests/test_mcp_server.py` — дифф этих файлов не менялся между
раундом 1 и раундом 2 (только `bot/net_safety.py`, `tests/test_net_safety.py`,
`bot/plugins/webshot.py`, `tests/test_skills_plugin.py` получили правки). Новые
тестовые файлы (`tests/test_github_analysis_plugin.py`, `tests/test_text_summarizer_plugin.py`)
и изменённые (`tests/test_codeinterpreter_plugin.py`, `tests/test_haiper_image_to_video_async_db.py`,
`tests/test_webshot_plugin.py`) построчно совпадают с §3 плана. Проверено границами
владения: `git status --short` вне владения T03 — только файлы T01/T02/T04 (параллельные
задачи той же волны), ни одного стороннего файла T03 не тронул. Импорты `httpx`
(`mcp_server.py`), `aiohttp` (`github_analysis.py`) удалены полностью — грепом
подтверждено отсутствие остаточных обращений; `httpx`/`aiohttp` там, где план требовал
оставить (`codeinterpreter.py` — `except httpx.HTTPError`, `haiper_image_to_video.py` —
`_process_video_task`), присутствуют и используются. Новых проблем не найдено.
