# Повторный аудит архитектуры и кода — chatgpt-telegram-bot

Дата: 2026-09-04. HEAD: `af382fb`. Предыдущий аудит: `docs/architecture_code_review_2026-07-02.md`,
план исправлений: `docs/audit_remediation_plan_2026-07-02.md`.

Метод: skill `dev-experts` (персоны `architect` и `reviewer`) и skill `bug-hunters`
(роль `python-hunter` в режиме code-first, каждая находка адверсарно оспорена как `logic-hunter`,
в отчёт попали только находки с уверенностью MEDIUM и выше). Семь параллельных субагентов:
архитектура ядра, LLM-пайплайн, персистентность и рантайм, Telegram-слой, плагины и
безопасность, инфраструктура/тесты/зависимости, веб-исследование зависимостей и практик.
Ключевые находки перепроверены координатором чтением кода. В репозитории ничего не менялось,
кроме этого файла.

Полный прогон тестов (`python3 -m pytest -q`, системный Python 3.12, PTB 22.6, openai 2.20):
**1449 passed, 1 skipped, 11 warnings**, плюс одно `Task was destroyed but it is pending` для
`SessionLogger._writer_loop` (`bot/session_logger.py:243`) при закрытии event loop.

Легенда: **CRITICAL** — приводит к зависанию, потере данных или дыре в безопасности;
**HIGH** — реальный сбой при обычном использовании; **MEDIUM** — ошибка при определённых условиях;
**LOW** — качество/поддерживаемость. Уверенность: CERTAIN (воспроизведено), HIGH, MEDIUM.

Термины: **event loop** — единственный поток, в котором крутятся все async-обработчики бота; если
он заблокирован, бот не отвечает никому. **deadlock** (взаимная блокировка) — две стороны ждут
друг друга навсегда. **RLock** — замок, который может держать только один поток.
**tool-call** — вызов инструмента (плагина) моделью. **re-entry** — повторный вызов модели после
выполнения инструментов. **ContextVar** — переменная, которая «едет» вместе с текущей
async-задачей и наследуется вложенными вызовами.

---

## 1. Резюме

1. Точечные баги прошлого аудита (P0) в основном закрыты, а структурный долг не тронут: из 10
   архитектурных проблем полностью не закрыта ни одна, два самых больших файла выросли
   (`telegram_bot.py` 5491 → 6340 строк, `openai_helper.py` 3822 → 4326).
2. Найдены **три новых сценария полного зависания бота** (§3.1–3.3): deadlock между
   транзакцией `DbHandle` и синхронным чтением настроек; deadlock вложенного
   `get_chat_response` из плагинов; синхронный `exec()` пользовательского кода в процессе.
3. `deep_analysis` (codeinterpreter) — это **выполнение кода модели через `exec()` в процессе
   бота** с доступом к `os.environ` и, значит, к `OPENAI_API_KEY`/`TELEGRAM_BOT_TOKEN`. Описание
   инструмента обещает «sandboxed environment», которого нет.
4. CI **не запускает основной набор тестов** (только MCP) и работает на Python 3.10, хотя код
   требует 3.11+.
5. `requirements.txt` без верхних границ у `openai` и `mcp`: чистая установка сегодня ставит
   `openai 3.x` (перешёл на `httpx2`, ломает `http_client=httpx.AsyncClient()`) и `mcp 2.x`
   (переименованы поля и транспорты). Пины `Pillow~=10.3`, `lxml-html-clean==0.1.1`,
   `spotipy~=2.23` держат открытыми известные CVE. 33 из 60 зависимостей не импортируются.
6. Новый слой `ai_provider`/`chat_run` — полумигрированный: два живых non-stream пути за флагом,
   провайдер не инжектится, стриминг обходит событийную модель, `PluginToolAdapter` мёртв.

---

## 2. Статус проблем прошлого аудита

| # | Проблема (2026-07-02) | Статус | Доказательство |
|---|---|---|---|
| 1 | God-модули | **Открыта, хуже** | `ChatGPTTelegramBot` 183 метода; `_process_message_locked` `bot/telegram_bot.py:4096` — 524 строки; `handle_function_call` `bot/openai_tool_handler.py:1086` — 620 строк, 19 параметров, рекурсия `:1690-1702`. |
| 2 | Конфиг — рассыпанные dict'ы | **Частично** | Числа унифицированы (`_parse_numeric_env` `bot/__main__.py:53-76`). 14 ключей продублированы между `openai_config` и `telegram_config` (`:206`↔`:302`, `:208`↔`:318`, …); `plugin_manager.config.update(openai_config)` `:359`; булевы парсятся двумя стилями (17 × `.lower()=='true'` против 5 × `parse_bool_env`). Второй источник дефолтов — 34 `setdefault` в `bot/openai_helper.py:309-342` (`temperature` 0.7 против 1.0 в `__main__`). |
| 3 | PluginManager ↔ OpenAIHelper | **Открыта** | `set_openai` `bot/__main__.py:369`, `bot/plugin_manager.py:80-97`, повтор `bot/telegram_bot.py:182-183`; `Plugin.execute(..., helper, ...)` `bot/plugins/plugin.py:124`. |
| 4 | БД читает env | **Открыта** | `bot/database.py:26-30, 33-39, 42-60, 79, 1164`. |
| 5 | Singleton Database | **Открыта (безопаснее)** | `__new__` `bot/database.py:64-90`; `_instance` после `init_db` (P0-09 закрыт); `_reset_singleton` `:96-104`. |
| 6 | Sync I/O на loop | **Частично** | БД в боте через `_db_call` с guard `bot/telegram_bot.py:898-902`. Остались: `UsageTracker` (§3.9), `disabled_plugins_for_user` (§3.1), `get_current_model` `bot/openai_helper.py:4147/4154`. |
| 7 | Дублирование пайплайнов | **Частично, +1 копия** | Мутаторы теперь во всех текстовых путях (`bot/openai_helper.py:1544, 2129, 2185`). Но legacy-тело `:934-1091` и `ChatRun.run_non_stream` `bot/chat_run.py:47-257` — два почти идентичных текста за флагом `chat_run_variant_b_enabled` `:922`. Стриминг в боте — три копии (`bot/telegram_bot.py:3331-3341`, `:4496-4506`, `:4962-4970`). |
| 8 | Мёртвый провайдер-слой | **Открыта** | Кортежи пусты `bot/model_constants.py:19-30`; 15 потребителей в helper, `bot/plugin_manager.py:20,357`, `bot/telegram_bot.py:45,4224,4907`. |
| 9 | Ретраи с побочными эффектами | **По сути закрыта** | tenacity удалён; ретрай только вокруг `create` `bot/openai_helper.py:486-501`. Остаток: SDK `max_retries: 3` `:271` поверх своих 3 × 20 с. |
| 10 | Граница ядро ↔ плагины | **Частично** | `openai_tool_handler.py` под линтер-тестом (`tests/test_no_hardcoded_plugin_refs.py:31`). Остались reach-through `bot/openai_tool_handler.py:1455, 1606`; бот мутирует `openai.conversations` `bot/telegram_bot.py:2450-2467, 6155-6174`, приватные методы через `getattr` `:2445, 2470, 3701, 4067`. |

---

## 3. Критические и высокие находки (новые)

### 3.1. CRITICAL — deadlock: транзакция `DbHandle` + синхронное чтение настроек на loop

- **Где:** `bot/plugins/db_handle.py:95-107` (`__aenter__` открывает `Database.transaction()` на
  DB-воркере), `bot/database.py:126-128` (`get_connection` берёт `_op_lock` и держит до выхода из
  `with`, то есть до `__aexit__` транзакции). Синхронный вызов БД из потока loop:
  `bot/plugin_manager.py:182, 192` (`get_user_settings` в `disabled_plugins_for_user`), достижимо
  на каждом сообщении через `bot/openai_helper.py:1302` и при каждом диспатче хуков через
  `_active_plugin_instances`; также `bot/openai_helper.py:2037` → `:4147/4154`.
- **Кто открывает транзакции:** `bot/plugins/agent_tools.py:1340, 1906, 1926` (в том числе из
  `on_session_reset`, который диспатчится при старте каждого запроса `bot/telegram_bot.py:2795`).
- **Уверенность:** HIGH (механизм CERTAIN, воспроизведён скриптом). **Статус:** NEW; прошлый
  аудит писал «прямого deadlock нет» — он появился вместе с `DbHandle.transaction()`.
- **Механизм:** задача A вошла в `async with db_handle.transaction()`: воркер держит `_op_lock` и
  ждёт следующей команды от loop. Задача B на loop зовёт sync `get_user_settings()` →
  `_op_lock.acquire()` блокирует поток loop. Loop больше не выполняет ничего, включая `__aexit__`
  задачи A. Таймаутов нет; бот стоит до рестарта. Тот же риск на shutdown:
  `bot/telegram_bot.py:5932` → `db.shutdown()` берёт `_op_lock` на loop (`bot/database.py:214`).
- **Смежное:** `Database._run_in_db_thread` берёт `_db_handle_transaction_lock`
  (`bot/database.py:244-250`), который `__aenter__` держит всё тело транзакции. Любой
  `*_async`/`db_handle.fetch_*` внутри `async with db_handle.transaction()` ждёт сам себя. Сейчас
  нарушителей нет, docstring `bot/plugins/db_handle.py:248-257` не предупреждает.
- **Направление фикса:** убрать sync-вызовы БД из потока loop (сделать
  `disabled_plugins_for_user` async или гарантировать `user_settings_scope` на всех путях), и/или
  не держать `_op_lock` через `await` — открывать транзакцию как единую sync-операцию на воркере.

### 3.2. HIGH — deadlock вложенного `get_chat_response` из плагинов

- **Где:** `bot/openai_helper.py:874` (`state_key = conversation_state_key or
  self._chat_state_key(chat_id)`), `:3170` (`_CHAT_STATE_KEY.get() or chat_id`), `:884-895`
  (обход замка только если `_chat_lock_bypass_enabled(chat_id)`), `:3221-3223` (сравнение
  `str(chat_id)`). Вызывающие: `bot/plugins/ask_your_pdf.py:364` (`chat_id=hash(file_path)`),
  `bot/plugins/language_learning.py:187`, `bot/plugins/conversation_analytics.py:233, 253, 271, 289`.
- **Уверенность:** CERTAIN (репродукция с реальным `OpenAIHelper`). **Статус:** NEW.
- **Механизм:** плагин внутри tool-call зовёт `get_chat_response` с синтетическим `chat_id`.
  Замок на чат уже удерживается внешним ходом; вложенный вызов берёт `state_key` из ContextVar
  (внешний), а обход не срабатывает, потому что bypass выставлен на внешний `chat_id`, а сравнение
  идёт с `hash(...)`. `async with lock` ждёт сам себя; таймаута вокруг `gather` tool-calls нет.
  Второй дефект: даже при обходе замка вложенный запрос уходит в историю **внешнего** чата (текст
  PDF лёг бы в разговор пользователя и в БД). `ask()` (`:795`) имеет guard `_in_active_turn`,
  `get_chat_response` — нет. Тесты не ловят: `tests/test_ask_your_pdf.py:53` подменяет
  `get_chat_response` на `AsyncMock`.
- **Смежное (MEDIUM):** `bot/plugins/show_me_diagrams.py:161, 186` зовёт `get_chat_response` с
  тем же `chat_id` — замок обходится, но вложенный ход вставляет `user`/`assistant` между
  `assistant(tool_calls)` и ещё не записанным `tool`-ответом; на следующем запросе
  `_repair_tool_call_history` (`bot/openai_helper.py:3093-3140`) считает настоящий `tool`-ответ
  сиротой и удаляет его.
- **Направление фикса:** плагины должны использовать `helper.ask()`/`ModelUtilities.one_shot`
  (не пишут в историю, не берут замок), а `get_chat_response` — отвергать вложенный вызов внутри
  активного хода тем же guard, что у `ask()`.

### 3.3. CRITICAL (безопасность) — `deep_analysis`: голый `exec()` в процессе бота

- **Где:** `bot/plugins/codeinterpreter.py:424` (`exec(code, exec_globals, exec_globals)`),
  `:420` (фильтр `"rm -r" in code or "os.system" in code`), `:126` (описание «sandboxed
  Jupyter-style environment»), `:435-439` (авто-`pip install` любого отсутствующего модуля),
  `:47-62` (таймаут через SIGALRM — работает только в главном потоке).
- **Уверенность:** CERTAIN. **Статус:** NEW (в прошлом аудите плагин не разбирался).
- **Механизм:** `os` и `sys` лежат в globals; `print(os.environ['OPENAI_API_KEY'])` возвращает
  секрет в ответ. Фильтр обходится `shutil.rmtree`, `subprocess.run`, `__import__('os').popen`.
  Модель получает недоверенный ввод из web-поиска, PDF, MCP и памяти — это prompt-injection → RCE
  и утечка секретов. `exec` синхронный внутри async `execute` → тяжёлый код блокирует loop для
  всех; `sys.stdout` подменяется глобально на время выполнения. Авто-`pip install` по имени из
  текста ошибки — вектор typosquatting. Рядом `download_file` (`:811`) тянет произвольный URL.
  Плагин грузится при пустом `PLUGINS` и доступен во всех режимах с `tools: [All]`.
- **Направление фикса:** минимум — отдельный процесс с чистым окружением (без секретов), без
  сети, с лимитами CPU/памяти/времени; целевой уровень — bubblewrap/Docker+gVisor (см. §7.3).
  До этого — честное описание инструмента и выключение по умолчанию.

### 3.4. CRITICAL (процесс) — CI не запускает тесты и работает на неподдерживаемом Python

- `.github/workflows/ci.yml:26-31`: `python -m pytest bot/tests/` — только MCP-тесты (1 файл),
  затем тот же файл ещё раз с `-v`. 1364 теста из `tests/` в CI не выполняются. Файл не менялся с
  2025-05-24.
- `ci.yml:20`: Python 3.10; `bot/utils.py:559` использует `asyncio.timeout` (3.11+), Dockerfile —
  3.12, README:194 обещает «3.9+» (упадёт на `bot/plugins/plugin.py:11` `str | None`).
- `python-package-conda.yml`: ставит `environment.yml` без `pytest-asyncio` → async-тесты не
  выполняются; единственный линтер проекта — `flake8 --select=E9,F63,F7,F82`.
- Нет ruff/mypy/pyright/pre-commit конфигов (при этом `pyflakes`/`ruff check bot/` дают 0
  замечаний — код чист, но это ничем не закреплено).

### 3.5. CRITICAL (сборка) — зависимости без верхних границ и с открытыми CVE

Источник: PyPI JSON и GitHub Advisories на 2026-09-04 (детали и ссылки в §6).

- `openai>=2.14.0` без верхней границы → ставится **3.8.0**. v3 заменил `httpx` на `httpx2`;
  `http_client=httpx.AsyncClient()` в `bot/openai_helper.py:262` перестаёт приниматься.
  Локально стоит 2.20, поэтому тесты зелёные; Docker-сборка с нуля сломается.
- `mcp>=1.0.0` → ставится **2.1.1**: `streamablehttp_client` удалён, поля camelCase → snake_case,
  таймауты `float` вместо `timedelta`. Плюс 5 CVE в 1.x до 1.28.1.
- `Pillow~=10.3.0` — 12 незакрытых CVE (PSD/FITS/PDF/font-парсеры, fix 12.1.1–12.3.0). Бот
  принимает картинки от пользователей.
- `lxml-html-clean==0.1.1` — CVE-2024-52595, CVE-2026-28348/28350, XSS в `xlink:href` (fix 0.4.5).
- `spotipy~=2.23.0` — пин блокирует фикс CVE-2025-27154 (fix 2.25.1).
- `httpx==0.27.0` — mcp 1.29 требует `>=0.27.1`.
- `python-telegram-bot==21.1.1` (2024-04) при том, что разработка и тесты идут на 22.6.
- Мёртвые/переименованные: `pytube` (не работает с 2023; `bot/plugins/youtube_audio_extractor.py`),
  `duckduckgo_search` → `ddgs` (`bot/plugins/ddg_translate.py:3`).
- 33 из 60 зависимостей не импортируются в `bot/` (AST-проверка): `asyncpg`, `telethon`,
  `opencv-python`, `scikit-learn`, `fastapi`, `uvicorn`, `moviepy`, `pygame`, `nltk`, `tenacity`,
  `youtube-transcript-api`, `google-api-python-client`, … Из-за них в Dockerfile живут `g++
  libc6-dev`. `pytest`, `pytest-asyncio`, `ipython` — в runtime-зависимостях. `pygments`
  импортируется (`bot/plugins/github_analysis.py:7`) но не объявлен — приезжает через `ipython`.
- Нет lock-файла и `pyproject.toml`; половина строк без пинов.

### 3.6. HIGH — потеря финального чанка стрима при ошибке последнего `edit`

- **Где:** `bot/telegram_bot.py:4492-4508` (chat), `:3327-3343` (vision), `:4958-4975` (inline).
- **Уверенность:** CERTAIN (репро-тест). **Статус:** NEW.
- **Механизм:** при `RetryAfter`/`TimedOut`/`Exception` на последней итерации выполняется
  `continue`, а финальный чанк отдаётся генератором ровно один раз. Повтора нет: пользователь
  видит промежуточный текст, `i += 1; total_tokens = int(tokens)` пропускается →
  `_record_chat_usage` пишет 0. `edit_message_with_retry` (`bot/utils.py:585-645`) ретраит только
  `BadRequest`. `backoff += 5` ни на что не влияет после выхода из цикла.

### 3.7. HIGH — обработка медиа-группы и inline-запросов без conversation lock

- `_process_vision_media_group` `bot/telegram_bot.py:2925-3084`: одиночный `vision` (`:3417`) и
  `process_message` (`:4038`) берут `_get_conversation_lock`, альбом — нет. Параллельный текст
  читает/пишет `conversation_context` одновременно → потеря сообщений истории.
- `handle_callback_inline_query` `:4848-5025`: `chat_id=user_id` совпадает с ключом личной
  переписки (`bot/conversation_key.py`), лок не берётся (MEDIUM из-за редкости сценария).

### 3.8. HIGH — stdio-MCP: обнаружение инструментов всегда возвращает пустой список

- **Где:** `bot/plugins/mcp_server.py:364-386`. `session.list_tools()` возвращает pydantic-модель
  `ListToolsResult`; `for tool in mcp_tools` итерирует пары `(field, value)` → `tool.name` бросает
  `AttributeError` → `except` → `return []`. Даже при правильной итерации код читает
  `tool.parameters`/`tool.required_parameters`, которых у `Tool` нет (есть `inputSchema`).
- **Уверенность:** CERTAIN (проверено на установленном SDK). Каждый stdio-сервер регистрируется
  с нулём инструментов; HTTP-ветка не затронута.

### 3.9. HIGH — `UsageTracker`: SQLite + `fsync` синхронно на loop на каждом сообщении

- `bot/telegram_bot.py:5051` (`is_within_budget`, каждое сообщение) → `bot/utils.py:783-800` →
  `UsageTracker.__init__` (`bot/usage_tracker.py:412-425`: DDL + импорт legacy JSON) и
  `get_current_cost()`. Запись: `_record_chat_usage` `bot/telegram_bot.py:799-811` (вызовы
  `:2813, 4310, 4604, 4929, 5013`) → `bot/usage_tracker.py:557-609`: `sqlite3.connect` + INSERT + 2
  UPDATE + temp-файл + `os.fsync` + `os.replace`, всё под классовым `_file_lock` (`:344`), который
  делится с `prune_store` в `to_thread`. Ничего не обёрнуто в `to_thread`.
- Также `bot/telegram_bot.py:1020-1025` — синхронный PIL на loop; `bot/plugins/webshot.py:37`
  `requests.get` **без таймаута** и `bot/plugins/movie_info.py:117,143,169,205` без таймаута —
  внутри async `execute`.

---

## 4. Средние находки

### 4.1. LLM-пайплайн (`bot/openai_helper.py`, `bot/openai_tool_handler.py`)

- **`tools: []` на re-entry → 400.** `bot/openai_tool_handler.py:1046-1047` и `:1645-1647` сужают
  список до пустого; `bot/openai_helper.py:467-468` отправляет его как есть (`if tools is not
  None`). Путь: `final_delivery_required` взводится при **любом** артефакте
  (`:1495`; артефакт = любой абсолютный путь в `value/file_path/path/...`, `bot/tool_result.py:8,
  55`), а `agent_tools` подключён лишь в 3 из 25 режимов → `_filter_tools_to_names(...,
  {deliver_to_user})` даёт `[]`. NEW.
- **Лимит `functions_max_consecutive_calls` не жёсткий при `final_delivery_required`.**
  `_reentry_tool_choice` `:905-910` возвращает `"auto"` до проверки лимита; если `deliver_to_user`
  раз за разом отвечает `success: False` (`bot/plugins/agent_tools.py:2785-2810`), цикл ограничен
  только «усталостью» модели. Контракт AGENTS.md обещает детерминированную остановку. KNOWN,
  частично.
- **Провайдер превращает отсутствующие `prompt/completion_tokens` в 0.**
  `bot/ai_providers/openai_compatible.py` `_usage()`/`_int_or_zero` (~:250-268) →
  `resolve_chat_cost` даёт `(0.0, 'model_split')` вместо `model_blended` по `total_tokens`.
  Нарушает правило AGENTS.md «report unknown rather than a split that silently omits». NEW.
- **Первый stream-chunk без tool_call-дельт → tool-call теряется** (`:1212-1256`, ветка `else`
  отдаёт поток как текст). Актуально для gateway-моделей, шлющих `role`-only/текстовые дельты
  первыми. KNOWN-STILL-OPEN.
- **Vision-путь минует мутаторы и ремонт истории** (`bot/openai_helper.py:2514`; legacy
  `role: "function"` при vision-модели вне `get_model_choices()` `:3438-3484`). KNOWN.
- **Stream-путь отдаёт ошибки как текст** (`:1206, 1229, 1293` `yield f"Error...: {e}"`), нет
  ретрая пустого ответа, который есть у `ChatRun`. KNOWN.
- **`_save_conversation_context` создаёт LLM-задачу имени сессии на каждое сохранение**
  (`:713-724`); в первом ходе 2–3 задачи без дедупликации по `session_id` → 2–3 платных вызова.
  NEW, MEDIUM.
- **INFO-логирование полных payload'ов** (`bot/openai_helper.py:511-515`,
  `bot/openai_tool_handler.py:1492-1496, 1650-1654`) — персональные данные в логах. KNOWN.
- **`_tool_result_summary` не знает `ok: false`** (`:1768-1772` против `bot/tool_result.py:112,
  124`); компакция трогает только `role=="tool"` (`:1855-1880`). KNOWN.
- Двойные ретраи (SDK 3 × свои 3 × 20 с) под замком чата — до минуты блокировки чата. KNOWN.

### 4.2. Персистентность

- **`get_conversation_context` глотает ошибки чтения** (`bot/database.py:939-982`, аннотация
  `Optional[Dict]`, реально 5-кортеж; `except Exception` → `(None, ...)`); потребители
  `bot/openai_helper.py:1178-1189, 1457-1466, 2460-2469` трактуют `None` как «контекста нет» →
  `reset_chat_history` → новая сессия, при `MAX_SESSIONS` — удаление самой старой. Временный
  `database is locked` неотличим от «данных нет». KNOWN-STILL-OPEN.
- **`allowed_user_ids` без `strip()` в списании бюджета** (`bot/utils.py:823`; `is_allowed`
  `:680` стрипает) → при `"123, 456"` расход разрешённого пользователя пишется ещё и в `guests`.
  KNOWN.
- **`list_user_sessions` тянет и парсит полные истории всех сессий** (`bot/database.py:1310-1349`)
  на каждом запросе для выбора модели (`bot/openai_helper.py:4147-4179`,
  `bot/openai_tool_handler.py:281`). KNOWN.
- `SessionLogger._writer_loop` не гасится до закрытия loop (виден в прогоне тестов).
- `bot/plugins/chief.py:194` — `aiohttp.ClientSession` без `close_async` → «Unclosed client
  session» на shutdown.
- Миграции `_apply_schema_migrations` идут внутри `init_db` (`:353`) без `transaction()`,
  DDL-шаги автокоммитятся по одному; безопасность держится на эвристике восстановления.

### 4.3. Telegram-слой

- **Сырые строки ошибок и запроса с `parse_mode=MARKDOWN`** (`bot/telegram_bot.py:2605-2606,
  2659-2660, 2727-2729, 2852-2853, 3198-3200, 4985`): `_` или `*` в тексте исключения или inline-
  запросе → `BadRequest`, пользователь не получает сообщение об ошибке. NEW.
- **Описания изображений без разбиения по 4096** (`:1071-1082`, `:3040-3052`). NEW.
- **Имя файла и mime-type в промпт без обработки** (`:528, 534-548`; для пересланного текста
  защита есть, для имени файла до 255 символов — нет). NEW, MEDIUM.
- **Image-edit через контекст не учитывает usage и бюджет** (`:4152-4160` → `:1027-1036`). NEW.
- KNOWN-STILL-OPEN: `update.message is None` для отредактированных команд (`:1130, 1155, 1323,
  2514, 5585/5589`); `vision` без `return` после `media_type_fail` (`:3217-3225`);
  `_image_edit_source_file_id` всегда `None` для «последней картинки» (`:994-1008`);
  `list_user_sessions(user_id)` в группах при ключе беседы по chat id (`:1164, 2833, 4564`);
  удаление сессии без проверки `_protected_session_ids` (`:6162-6175`); `plugin_menu_entries`
  перезаписывается глобально (`:5578`); `query.message.delete()` без guard (`:1381, 5606,
  6110`); `/stats` частично экранирован (`:1178, 1184, 1290`).
- `bot/html_utils.py:1689-1726` — `subprocess.run` для PlantUML без `timeout` (в `to_thread`,
  loop не блокирует, поток может зависнуть).

### 4.4. Плагины и безопасность

- **Обход политики терминала обёртками** (`bot/command_policy.py:105, 108, 128, 434`): `( rm -rf
  / )`, `{ rm -rf /; }`, `for d in /; do rm -rf $d; done`, `timeout 10 rm -rf /`, `nohup rm -rf
  / &`, `xargs rm -rf`, `( git push -f )`, `curl … | sudo sh` → `allow`, тогда как голая форма →
  `require_approval`. Это не обфускация, а обычные конструкции. CERTAIN (исполнено).
- **`skills`: `env = os.environ.copy()`** (`bot/plugins/skills.py:3703, 2515`) — скрипты скилов и
  `npx skills` получают `OPENAI_API_KEY`/`TELEGRAM_BOT_TOKEN`. KNOWN (отдельно от принятой
  default-open политики).
- **MCP: `register_mcp_server` — model-callable tool** (`bot/plugins/mcp_server.py:183`), описания
  удалённых инструментов попадают в промпт без ревью (`:275-283`) — ровно вектор tool poisoning
  из OWASP MCP Top 10 (см. §7.3).
- Конфиг `plugin_manager.config` = `plugin_config ∪ openai_config` (`bot/__main__.py:359`) —
  любой плагин видит `api_key`, `yandex_api_token`, `assemblyai_api_key` без запроса.
- `PROXY`/`OPENAI_PROXY`/`TELEGRAM_PROXY` описаны в README:269,294 и читаются в конфиг
  (`bot/__main__.py:210, 319`), но **нигде не применяются**: `bot/openai_helper.py:262` создаёт
  `httpx.AsyncClient()` без прокси (строка с прокси закомментирована `:261`), builder в
  `bot/telegram_bot.py:6288-6301` не вызывает `.proxy(...)`. Проверено координатором.

### 4.5. Инфраструктура и гигиена репозитория

- `docker-compose.yml`: `TELEGRAM_LOCAL_MODE` по умолчанию `true` (`bot/__main__.py:293`) и
  `base_url=http://localhost:8081/bot`, а сервиса local Bot API в compose нет → `docker compose
  up` по README падает по сети.
- `.dockerignore` не исключает `bot/skills/` (28 MB локального состояния), `ai_docs_site/`
  (9.5 MB), `tests/`, `.ai-docs/`, `docs/`, `evals/`.
- `bot/plugins/plantuml.jar` — 20.7 MB в git (98 % pack-файла). `ai_docs_site/` — собранный сайт
  закоммичен, а исходники `.ai-docs/` в `.gitignore` (сайт нельзя пересобрать из клона).
  `bot/plugins/pdf_cache/cache_metadata.json` — runtime-файл в git, который код перезаписывает
  (`bot/plugins/ask_your_pdf.py:34-45, 121-123`). `IMG_3980.jpg` — без ссылок.
- `.env` на диске с правами `-rwxrwx---` (группа может писать токены); `chmod 600`.
- Healthcheck проверяет только PID 1 и права на каталоги; зависший loop его проходит. Нет
  `mem_limit`/`logging.max-size` при плагине, исполняющем произвольный код.
- Устаревшие мажоры actions (`checkout@v3`, `setup-python@v4/v3`, `docker/*@v2`);
  `publish.yaml:44` пушит в чужой неймспейс `n3d1117/...`.

### 4.6. Документация

- README описывает env-переменные, которых код не читает: `MAX_SUBAGENTS`,
  `MAX_SUBAGENT_TOOL_ROUNDS`, `MIN_SUBAGENT_TOOL_ROUNDS`, `DELIVERY_DEDUP_WINDOW_SECONDS`,
  `DELIVERY_MAX_ARTIFACT_BYTES`, `TASKS_TTL_SECONDS`, `SUBAGENT_BLOCKED_FUNCTIONS`,
  `CHAT_RUN_VARIANT_B_ENABLED` (README:289; в коде только `setdefault`). `.env.example` содержит
  мёртвые `STABLE_DIFFUSION_TOKEN`, `ENABLE_LOCAL_FILE_SERVER`, `LOCAL_FILE_SERVER_HOST/PORT`,
  `ENABLE_MCP_SERVERS`.
- Читаются, но не описаны в README: `SUMMARY_MODEL`, `TERMINAL_APPROVAL_MODE`,
  `HINDSIGHT_AUTONOMOUS_CAPTURE_ENABLED`, `MCP_SERVERS_CONFIG_PATH`, `TERMINAL_OUTPUT_BYTE_LIMIT`,
  `PLUGINS`, `STREAM`, `TEMPERATURE`; в `.env.example` нет ещё ~14 ключей.
- AGENTS.md: из 76 ссылок `file:line` 5 сдвинуты: `bot/openai_tool_handler.py:181` и
  `:1358-1360` → `:1396-1402`; `:416-420` → `:449-457`; `:868` → `:905`, `:930-931` → `:968`;
  `bot/__main__.py:349-362` → `:357-370`; `bot/plugin_manager.py:428` (валидация ниже), `:993` →
  `:996`, `:865/:892` → `:868/:895`.
- README:194 «Python 3.9+» — неверно, минимум 3.11.

---

## 5. Архитектура: новые наблюдения и предложения

### 5.1. Слой `ai_provider` / `chat_run` — полумигрированный

- Провайдер создаётся на каждый вызов внутри helper (`bot/openai_helper.py:614-625`), замыкаясь
  на `self._timed_create`; `AIProvider` Protocol нигде не является типом поля/параметра, подменить
  провайдер снаружи нельзя. Единственный второй провайдер — `FakeAIProvider`
  (`bot/ai_providers/fake.py`), только для тестов, но живёт в production-пакете.
- Стриминг обходит события: `_AIProviderStreamProxy` (`:100-169`) отдаёт сырые SDK-chunk'и,
  `AITextDelta`/`AIToolCallReceived` в стрим-пути никто не читает.
- `AIProviderResponse` имитирует форму SDK (`choices[i].message`, `bot/ai_provider.py:33-46`),
  чтобы потребители не менялись — абстракция «похожа на то, что абстрагирует».
- `ChatRun.run_non_stream` — копия legacy-тела, ходит в helper через name-mangled приватные
  имена (`helper._OpenAIHelper__common_get_chat_response` `bot/chat_run.py:64`,
  `__handle_function_call` `:80, 110, 158`, `__add_to_history` `:207, 218`) и напрямую пишет
  `helper._gate_fired`/`_chat_request_*` (`:62, 73, 86, 93`). Тесты пиннят оба пути.
- `bot/plugin_tool_adapter.py` — мёртв в рантайме (только `tests/test_plugin_tool_adapter.py`).
- `_guard_tool_call` (`bot/plugin_manager.py:509-530`) инстанцирует все плагины на каждый
  tool-call ради `guard_tool_call`, которого нет ни в одном плагине (fail-open по умолчанию).
- Резолв «имя функции → плагин» без индекса: `get_plugin_name_by_function_name` (`:599-624`) и
  `get_spec_by_function_name` (`:587-597`) — полные проходы с `get_spec()` каждого плагина
  (403 строки у `agent_tools`, 377 у `skills`) на каждый вызов инструмента, при том что индексы
  `_model_tool_name_to_canonical` уже есть (`:72-73`).
- `RequestContext` (`bot/request_context.py`) и `chat_response_utils.py` — единственные новые
  абстракции, которые работают как задумано; это образец для дальнейшего выноса.
- Пер-запросное состояние всё так же в разделяемых dict'ах на инстансе (`_chat_request_models`,
  `_chat_request_usage_split`, `_chat_request_extra_tokens`, `_gate_fired`
  `bot/openai_helper.py:284-295`), `.pop(state_key)` по 41 месту, `_without_chat_lock` (`:3226`).
- `PluginManager.bot = getattr(openai, "bot")` (`bot/plugin_manager.py:83`) читается, когда бот
  ещё `None` (`bot/telegram_bot.py:175`); реальный `application.bot` появляется в `post_init`
  (`:6306`), а `set_openai` после этого не вызывается. Нужно проверить, какие плагины на это
  опираются.

### 5.2. Размеры

| Модуль | Строк | Функций | Главный класс |
|---|---|---|---|
| `bot/telegram_bot.py` | 6340 | 213 | `ChatGPTTelegramBot` 183 метода |
| `bot/openai_helper.py` | 4326 | 129 | `OpenAIHelper` 116 |
| `bot/plugins/agent_tools.py` | 4062 | 143 | `AgentToolsPlugin` 134 |
| `bot/plugins/skills.py` | 3849 | 136 | `SkillsPlugin` 129 |
| `bot/plugins/hindsight_memory.py` | 3394 | 133 | `HindsightMemoryPlugin` 117 |
| `bot/html_utils.py` | 1860 | 21 | `HTMLVisualizer` 8 методов на 1781 строку |
| `bot/openai_tool_handler.py` | 1706 | 61 | `handle_function_call` 620 строк |
| `bot/database.py` | 1673 | 85 | `Database` 82 |

Всего в `bot/` 49 085 строк; пять файлов дают 45 %.

### 5.3. Предложения (ранжированы по «рычаг / риск»)

**П1. Закрыть двойной non-stream путь.** Рекомендация: удалить legacy-тело
`bot/openai_helper.py:934-1091` и флаг `chat_run_variant_b_enabled` (`bot/__main__.py:214`,
`bot/openai_helper.py:342, 688-690, 922`), перевести `ChatRun` на публичные точки входа helper'а.
Альтернативы: legacy делегирует в `ChatRun` (флаг остаётся); параметризовать тесты по флагу
(фиксирует дублирование). Без этого любое изменение пайплайна делается дважды.

**П2. Удалить мёртвый провайдер-слой `model_constants`.** Кортежи пусты, 15+ ветвлений
недостижимы, транспорт зависит от «семейств моделей» (`bot/telegram_bot.py:45`). Замена —
capability-конфиг по образцу `MODEL_CONTEXT_WINDOWS` (`bot/__main__.py:120-156`). Не наполнять
кортежи из env «на всякий случай»: ветки два месяца не исполнялись.

**П3. Конфиг: один источник.** Сначала `shared = {...}` для 14 дублей и единая политика булевых
(мягкая, как у чисел); убрать дублирующие `setdefault` в helper; `Database.configure(...)` вместо
чтения env внизу; плагинам отдавать только их prefix-сегмент. Полноценные dataclass'ы/pydantic —
позже, если появится проверка типов в CI.

**П4. Публичный API состояния сессии в helper** (`load_session`, `replace_system_message`,
`evict`) вместо мутаций `openai.conversations` из бота (`bot/telegram_bot.py:2436-2481,
6149-6174`) и `getattr(self.openai, '_...')`. `ConversationState`-объект — после П1.

**П5. Индекс «имя функции → (плагин, spec)»** в `load_plugins` рядом с
`_model_tool_name_to_canonical`; удалить `_guard_tool_call` или объявить `guard_tool_call` в
базовом `Plugin` и решить fail-open/closed осознанно.

**П6. Увести `UsageTracker` с loop:** `asyncio.to_thread` вокруг `record_*` (трекер уже
потокобезопасен, `bot/usage_tracker.py:149`), либо очередь + writer (тогда сначала починить
lifecycle у `SessionLogger._writer_loop`).

**П7. Один стриминговый рендерер** вместо трёх копий в боте (`:3229-3400`, `:4096-4619`,
`:4848-5025`): `stream_to_telegram(update, chunks, on_direct_result, ...)`, который владеет
backoff/чанкованием и **повторяет финальный чанк** (закрывает §3.6). Первый реальный шов для
разрезания `ChatGPTTelegramBot`.

**П8. Удалить или подключить `plugin_tool_adapter.py`**; `FakeAIProvider` перенести в `tests/`.

Порядок: П1 → П2 → П8 (чистка, низкий риск) → П6 → П3 → П5 → П4 → П7. Не начинать с «разбить
`openai_helper.py` на модули»: пока живут два пути и мёртвые семейства моделей, разбиение
перетащит дубли в новые файлы.

---

## 6. Зависимости (веб-исследование, 2026-09-04)

| Пин | Последняя | Риск |
|---|---|---|
| `python-telegram-bot==21.1.1` | 22.8 (2026-06) | Bot API 7.2 vs 9.3; 22.0 удалил deprecations v20; 22.8 требует `httpx>=0.27,<0.29`. Используемые `concurrent_updates/local_mode/base_url` не менялись. |
| `openai>=2.14.0` | 3.8.0 (2026-09-03) | v3 → `httpx2`, `http_client=httpx.AsyncClient()` ломается. Фикс: `openai>=2.14,<3` до миграции. |
| `httpx==0.27.0` | 0.28.1; стюардство у `httpx2` | mcp 1.29 требует `>=0.27.1`. Рекомендуемо `httpx>=0.27.1,<0.29`. |
| `tiktoken==0.7.0` | 0.14.0 | Не знает `gpt-5`/o200k-модели → fallback `cl100k_base`, неверный подсчёт (`bot/openai_helper.py:3947`). |
| `Pillow~=10.3.0` | 12.3.0 | 12 CVE (2026-25990, -40192, -42308/10/11, -54058/59/60, -55379/80, -59199/205). |
| `lxml-html-clean==0.1.1` | 0.4.5 | CVE-2024-52595, CVE-2026-28348/28350, XSS xlink:href. |
| `mcp>=1.0.0` | 2.1.1 (1.x: 1.29.1) | v2 breaking; CVE в 1.x: 2025-53366, -53365, -66416, 2026-52869, -59950 (fix 1.28.1). Фикс: `mcp>=1.28.1,<2`. |
| `spotipy~=2.23.0` | 2.26.0 | CVE-2025-27154 (fix 2.25.1). |
| `pytube~=15.0.0` | мёртв с 2023 | `pytubefix`/`yt-dlp`. |
| `duckduckgo_search` | переименован в `ddgs` 9.16 | `from ddgs import DDGS`. |
| `telethon==1.33.1` | 1.44.0 | Не импортируется в `bot/`. |
| `asyncpg==0.29.0` | 0.31.0 | Не импортируется; не собирается на 3.13. |
| `moviepy>=1.0.3` | 2.2.1 | 2.x breaking; не импортируется. |
| `tenacity==8.3.0` | 9.1.4 | Не импортируется. |

Источники: PyPI JSON, GitHub Advisories, changelog PTB, `openai-python` release v3.0.0 и
`httpx2.md`, MCP SDK migration guide, changelog `lxml_html_clean`.

---

## 7. Практики (веб-исследование)

### 7.1. Telegram

- `concurrent_updates(True)` = до 256 параллельных апдейтов; повторные `edit_message_text` одного
  сообщения чаще ~1 р/с под `AIORateLimiter` замораживают бота для всех (PTB issue #3765).
  Троттлить правки по времени, финальный текст слать один раз и **с повтором при `RetryAfter`**.
- Лимит 4096 символов: резать по границам сущностей, отправлять последовательно.

### 7.2. OpenAI

- Chat Completions не deprecated, но «Responses is recommended for all new projects»; миграция
  плановая, не срочная. `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo` выключаются 2026-10-23.
- Каждый `tool_call_id` обязан получить `tool`-ответ сразу после assistant-сообщения (иначе 400);
  `tools` не может быть пустым массивом; `strict: true` требует `additionalProperties: false`.
- `stream_options.include_usage` поддержан не всеми gateway — текущая opt-in политика верна.

### 7.3. Безопасность агентов

- Sandbox: shared-kernel Docker недостаточен для враждебного кода; консенсус — gVisor или
  Firecracker; локальный минимум — bubblewrap с сетевым allowlist (Claude Code, Codex CLI).
  Текстовый фильтр команд — UX-гейт, не граница безопасности (это признаёт и сам
  `bot/command_policy.py:4-7`).
- MCP (OWASP MCP Top 10, MCP03 Tool Poisoning; спека 2025-11-25): показывать пользователю полные
  описания инструментов и команду локального сервера, пиновать описания по хэшу, требовать
  подтверждение при смене схемы, не давать модели самой регистрировать серверы.

### 7.4. Инструменты качества

- `uv` + `uv.lock` (или `pip-compile` → `requirements.lock`), `ruff` (check + format), `mypy` или
  `pyright`, `pip-audit` в CI, `pre-commit`.

---

## 8. Тесты

- 1364 теста в `tests/`, детерминированные, без сети; `evals/` отгорожен тремя замками —
  правильно.
- Общих фикстур нет: `tests/conftest.py` 21 строка; **213 классов `Fake*/Dummy*/Stub*` в 53
  файлах** (`FakeMessage` — 15 копий, `FakeDB` — 12, `FakeHelper` — 11). Нужен `tests/fakes.py`.
- Грубое покрытие (функция упоминается хоть в одном тесте): `telegram_bot.py` 33 %,
  `openai_tool_handler.py` 22 %, `openai_helper.py` 67 %, `plugin_manager.py` 66 %,
  `database.py` 67 %. Без тестов: `_handle_session_callback_locked` (232 строки),
  `_run_vision_model_request`, `_process_vision_media_group`, `_get_chat_response_stream_locked`,
  `_retry_missing_delivery_tool`, миграции `_migrate_conversation_context_to_sessions` и
  `_recover_from_failed_migration`.
- 6 тестов без `assert`; `tests/test_skills_plugin.py:2404-2445` — реальный `fork()` +
  `sleep(30)` в ребёнке, ~2.5 с и зависимость от ядра (кандидат на флак).
- Тесты, которые бы поймали §3.2 и §3.6, отсутствуют: `tests/test_ask_your_pdf.py:53` мокает
  `get_chat_response`; кейса «ошибка `edit` на финальном чанке» нет.

---

## 9. Рекомендуемый порядок работ

**Волна 0 — остановить кровотечение (дни).**
1. `requirements.txt`: `openai>=2.14,<3`, `mcp>=1.28.1,<2`, `httpx>=0.27.1,<0.29`,
   `Pillow>=12.3`, `lxml-html-clean>=0.4.5`, `lxml>=6.1.1`, `spotipy>=2.25.1`; удалить 33
   неиспользуемых; `Pygments` явно; dev-зависимости в `requirements-dev.txt`; lock-файл.
2. `ci.yml`: `python -m pytest -q`, Python 3.12, `ruff check`, `pip-audit`; удалить
   `python-package-conda.yml`.
3. `deep_analysis`: выключить по умолчанию (или убрать из `tools: [All]`-режимов) и поправить
   описание, пока нет изоляции.
4. `chmod 600 .env`; `.dockerignore`; `TELEGRAM_LOCAL_MODE` в compose.

**Волна 1 — зависания и потери данных (неделя).**
5. §3.1: `disabled_plugins_for_user` → async (или обязательный `user_settings_scope`),
   sync-вызовы БД из loop под запрет тем же guard, что `_db_call`; не держать `_op_lock` через
   `await` в `DbHandle.transaction`.
6. §3.2: guard вложенного `get_chat_response` внутри активного хода; плагины
   `ask_your_pdf`/`language_learning`/`conversation_analytics`/`show_me_diagrams` → `ask()` или
   `ModelUtilities.one_shot`.
7. §3.6: повтор финального чанка при `RetryAfter`/`TimedOut` (в рамках П7 или точечно в трёх
   местах).
8. §3.7: conversation lock для медиа-групп и inline.
9. §3.8: `mcp_tools.tools` и `inputSchema` в `_fetch_stdio_tools`.
10. §3.9 / П6: `UsageTracker` в `to_thread`; таймауты `requests` в `webshot`/`movie_info`.

**Волна 2 — корректность пайплайна.**
11. `tools: []` на re-entry → не отправлять `tools`; жёсткий лимит при
    `final_delivery_required`; `final_delivery_required` только когда `deliver_to_user` реально в
    allow-list.
12. Провайдер: `None` вместо 0 для отсутствующих токенов.
13. `get_conversation_context`: различать «нет данных» и «ошибка чтения».
14. Прокси: применить `proxy` в `httpx.AsyncClient(proxy=...)` и `builder.proxy(...)`, либо
    убрать из README.
15. Terminal policy: нормализовать обёртки `( … )`, `{ … }`, `timeout`, `nohup`, `xargs`,
    `for/if` перед матчингом; в документации явно назвать фильтр UX-гейтом.

**Волна 3 — архитектура (П1 → П2 → П8 → П3 → П5 → П4 → П7)** и sandbox для
`deep_analysis`/`terminal` (bubblewrap как минимум).

**Волна 4 — документация:** README env-таблица, `.env.example`, AGENTS.md line-refs (одним
проходом), «Python 3.11+».
