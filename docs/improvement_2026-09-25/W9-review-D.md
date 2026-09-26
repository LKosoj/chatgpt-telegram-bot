# W9 — ревью группы D (Telegram-хендлеры / утилиты / вспомогательные плагины)

Ревьюер проверил `git diff HEAD` (HEAD `08bc457`) для закреплённых файлов: `bot/telegram_bot.py`
(>6000 строк, читался диапазонами), `bot/telegram_stream.py`, `bot/utils.py` (всё, кроме
delivery/access-частей — `is_allowed`, `_charge_user_and_guest*`, artifact-логика в
`handle_direct_result`, — это зона группы A), `bot/env_utils.py`, `bot/html_utils.py`, 27 файлов
`bot/plugins/*.py` (ask_your_pdf, auto_tts, chief, conversation_analytics, crypto,
ddg_image_search, ddg_translate, ddg_web_search, google_web_search, iplocation, jina_web_search,
language_learning, movie_info, pravo_gov_ru_api, prompt_perfect, reaction, show_me_diagrams,
spotify, task_management, text_document_qa, vkusvill, weather, web_research, website_content,
wolfram_alpha, youtube_audio_extractor, youtube_transcript), `scripts/mypy_baseline.py`,
`mypy_baseline.json`, `pyproject.toml`, `.github/workflows/ci.yml`, `.gitignore`, `.env.example`,
`README.md`, `README.ru.md`, `AGENTS.md`, плюс закреплённые тесты. Не переподнимались: удаление
`bot/skills/**` без бэкапа, правки `.env.example`/`.gitignore` (T07), текст skills_agent
"Function … returned" (T08), compat-view свойства с count guard.

## Раунд 1

### Summary

- ERROR: 0
- WARNING: 2
- NIT: 0

### Что проверено и почему нет замечаний

- **`get_spec()` — 27 плагинов, побайтовое сравнение с HEAD.** Скрипт (`git archive HEAD` во
  временный каталог, чистый импорт `bot.plugins.<module>` из HEAD и из рабочего дерева с
  чисткой `sys.modules`, для плагинов, требующих credentials при `__init__` —
  chief/movie_info/spotify/wolfram_alpha — заданы фиктивные env-переменные) показал:
  `json.dumps(get_spec(), sort_keys=True)` идентичен HEAD для всех 27 плагинов. Изменения в этих
  файлах — только типизация (`Dict`→`dict`, `Optional`, `-> List[Dict]`), новый атрибут
  `returns_untrusted_content` (принят в T08) и несколько добавленных веток
  `return {"error": f"Unknown function: {function_name}"}` (mypy-driven, ветка недостижима до
  диспетчеризации выше по коду — проверено построчно в ask_your_pdf.py, chief.py, spotify.py,
  weather.py). `chief.py`'s `_parse_menu_preferences` получил новую ветку
  `else: raise ValueError(...)` — санкционировано T02-планом.
- **`bot/telegram_bot.py`** (диф 952 строк по `--stat`) — это T13a (`require_message`,
  `require_query`, `require_accessible_message`, `_warn_vision_document_without_mime_type`,
  `Protocol`-класс `_AuthWrappedCallback`, точечные `assert`/`cast` для mypy-сужения Optional) +
  T12b/T12d (дедуп-хелперы: `_build_busy_status`, `_dispatch_session_reset`,
  `_dispatch_user_message`, `_send_rich_markdown_if_fits`, `_send_markdown_or_plain`,
  `_build_mode_group_keyboard`, `_resolve_plugin_command_or_reply`). Каждый новый `assert X is
  not None` прослежен до гарантирующего условия по коду выше (например: `assert message is not
  None` в `_queue_vision_media_group` гарантирован веткой `if not message: return None` в
  `_image_attachment_from_message`; `assert inline_message_id is not None` в
  `handle_callback_inline_query` гарантирован тем, что паттерн `^gpt:` в
  `CallbackQueryHandler` навешан исключительно на кнопки `InlineQueryResultArticle`,
  отправляемые через `answer_inline_query`). Ни один не сужает старое поведение — в местах, где
  раньше было бы необработанное `AttributeError`/`TypeError` на том же граничном случае, теперь
  `AssertionError`, не новый краш-путь. Найден реальный fix (в рамках T13a's None-check scope, не
  посторонняя правка): `elif update.message.document and
  update.message.document.mime_type.startswith('image/')` (падало на `mime_type=None`) заменено
  на `elif message.document and message.document.mime_type and
  message.document.mime_type.startswith('image/')` + новая ветка логирования
  `_warn_vision_document_without_mime_type`. `Update.message`/`callback_query` — атрибуты
  `__slots__` (не property) в `telegram/_update.py`, поэтому mypy корректно сужает Optional
  через локальную переменную по всему файлу — подтверждено чистым прогоном mypy (см. ниже).
- **`bot/telegram_stream.py`** — диф 17 строк: новый `retry_after_seconds(exc: RetryAfter) ->
  float` (обрабатывает int/`timedelta` в зависимости от `PTB_TIMEDELTA`), применён вместо
  `asyncio.sleep(exc.retry_after)`. Соответствует T02-плану.
- **`bot/html_utils.py`** — диф 2 строки: `_generate_plantuml` `-> str` → `-> None`; тело метода
  делает только голый `return`, единственный вызывающий код результат отбрасывает — корректно.
- **`bot/utils.py`** (вне delivery/access-зоны) — типизация (`BusyStatusMessage.message`,
  `markdown_stack`, `User | None`), `assert update.effective_chat is not None` в
  `wrap_with_indicator` (внутри `try/except Exception`, безопасно), `resize_image_if_needed`'s
  `resized_img: Image.Image = img` (не переприсваивает параметр, эквивалентно), новая
  `parse_model_choices(raw, default_model) -> list[str]` (T12a) — покрыта
  `tests/test_utils_parse_model_choices.py` (7 тестов, все проверяют заявленное поведение).
- **`scripts/mypy_baseline.py` / `mypy_baseline.json`** — baseline пуст (0 ошибок), проверено
  напрямую (`mypy bot` — 0 ошибок) и через `check`-команду (прошла).
- **CI/конфиги** (`.github/workflows/ci.yml`, `pyproject.toml`, `.gitignore`, `.env.example`) —
  соответствуют T01/T07 один в один.
- **Тесты из списка владения** — прочитаны все построчно; ни одна не тестирует не то, что
  заявляет в докстринге/имени. В частности `tests/test_t12d_agent_tools_ask_callback.py`
  (докстринг: "handle_ask_callback markup cleanup is shared via `_clear_ask_user_markup`")
  реально проверяет и single-select, и multi-select-confirm ветки через
  `FakeQuery.edit_message_reply_markup`/`answer` — соответствует заявленному.

Прогоны: `mypy bot` — "Success: no issues found in 85 source files"; `ruff check bot tests
scripts` — "All checks passed!"; `MYPY_PYTHON=... python3 scripts/mypy_baseline.py check` —
passed; полный `pytest` — 1956 passed (3 benign warnings, ~71s); целевой набор тестов из списка
владения — 96 passed.

### WARNING — `bot/env_utils.py` не содержит `parse_kv_list`, требуемый T12-plan (A1)

- **Файл:** `bot/env_utils.py` (весь файл, 28 строк — только `env_bool`)
- **Почему:** `docs/improvement_2026-09-25/T12-plan.md:24-53` (раздел "A1") требует в этом новом
  файле две функции — `env_bool` (перенесена корректно, используется в `bot/__main__.py` в 17
  местах) и `parse_kv_list(raw, *, value_parser, warn_prefix) -> dict[str, T]` — с явным
  указанием заменить дублирующиеся циклы разбора `key=value,key2=value2` в
  `bot/pricing.py:37-52` (`load_model_token_prices`) и `bot/__main__.py:122-`
  (`parse_model_context_windows_env`) на вызовы общего хелпера. `parse_kv_list` в дереве не
  найден (`git grep -n parse_kv_list` — 0 совпадений вне `docs/`). Обе целевые функции по-прежнему
  содержат собственный копипаст-цикл `for item in raw.split(','): ... item.partition('=') ...`
  без изменений (проверено чтением `bot/pricing.py:37-52` и `bot/__main__.py:122-152` —
  идентичны по форме плану). `tests/test_env_utils.py` тоже содержит только тесты `env_bool`
  (3 теста), тестов `parse_kv_list` нет, что подтверждает: функция не просто не подключена, а
  вообще не написана — часть T12a не выполнена.
- **Сценарий проявления:** Не баг рантайма (обе функции работают как раньше), но
  задокументированное в плане T12 сокращение дублирования не произошло — риск в том, что при
  следующей правке формата `MODEL_TOKEN_PRICES`/`MODEL_CONTEXT_WINDOWS` придётся синхронно
  редактировать два независимых цикла разбора вместо одного места, и лог-сообщения/поведение на
  невалидных записях будут и дальше по отдельности проверяться в двух тестах.
- **Предлагаемый fix:** Добавить `parse_kv_list` в `bot/env_utils.py` по сигнатуре из
  T12-plan.md A1 и переключить на неё `bot/pricing.py:load_model_token_prices` и
  `bot/__main__.py:parse_model_context_windows_env` без изменения их внешних сигнатур/текстов
  логов (как явно требует план), плюс тесты в `tests/test_env_utils.py`.

### WARNING — `AGENTS.md:32` ссылается на неверный диапазон строк в `bot/__main__.py`

- **Файл:** `AGENTS.md:32`
- **Почему:** Текст: "...a second process against the same lock file logs ERROR and exits
  non-zero (`bot/__main__.py:206-213`)." Реальный блок acquire/except для instance lock —
  `bot/__main__.py:190-197`:
  ```
  lock_path = os.environ.get('INSTANCE_LOCK_PATH') or default_lock_path(
      os.environ.get('DB_PATH')
  )
  try:
      acquire_instance_lock(lock_path)
  except InstanceLockError as exc:
      logging.error("Instance lock unavailable error=%s", log_exception_shape(exc))
      exit(1)
  ```
  Строки 206-213 — это уже блок `# Setup configurations` (`model_choices =
  parse_model_list_env(...)` и далее), не имеющий отношения к instance lock. (То же расхождение
  независимо зафиксировано группой A в `W9-review-A.md` с тем же предлагаемым исправлением —
  дублирую здесь, так как `AGENTS.md` входит и в мою зону владения по w9.txt.)
- **Сценарий проявления:** Разработчик открывает `bot/__main__.py:206-213`, ожидая увидеть логику
  instance-lock, и вместо этого видит парсинг конфигурации моделей — теряет время или неверно
  цитирует код при объяснении поведения.
- **Предлагаемый fix:** Заменить `206-213` на `190-197`.

## Раунд 2

### Summary

- ERROR: 0
- WARNING: 1
- NIT: 0

Раунд-1 WARNING 1 (`parse_kv_list` не реализован) принят как есть (deviation note в конце
T12-plan.md) — не переподнимаю.

### Проверено в раунде 2

- **Раунд-1 WARNING 2 (неверный диапазон строк в AGENTS.md) — исправлено.** `AGENTS.md:32`
  теперь ссылается на `bot/__main__.py:190-197`; построчно сверено с файлом — это ровно блок
  `lock_path = os.environ.get('INSTANCE_LOCK_PATH') or default_lock_path(...)` … `except
  InstanceLockError as exc: ... exit(1)`. Соседняя ссылка `bot/__main__.py:184-188` (required
  env vars) тоже сверена и совпадает.
- **15 случайных file:line ссылок в AGENTS.md** (скрипт выбрал случайные 15 из 71 уникальной
  ссылки на `.py:строка[-строка]` во всём файле, не только в моей зоне) — каждая открыта и
  сверена построчно с текущим деревом: `openai_helper.py:3727` (`_summarize_and_trim`),
  `plugins/skills.py:219` (`on_before_chat_request`), `plugin_manager.py:388`
  (`_format_specs_for_model`), `plugins/plugin.py:12,20` (`plugin_id`,
  `get_function_prefix`), `openai_helper.py:1184-1185` (`current_mode['tools']`),
  `plugins/agent_tools.py:2356` (`_validate_plan_tasks`), `plugins/skills.py:388`
  (`get_spec`), `plugin_manager.py:532,562` (`json.loads(arguments)`,
  `validate_function_args`), `plugins/agent_tools.py:41` (`describe_plan_lifecycle`),
  `openai_tool_handler.py:1712-1713` (delivery-tool narrowing), `pricing.py:25`
  (`DEFAULT_MODEL_TOKEN_PRICES`), `plugins/agent_tools.py:347` (`on_before_chat_request`),
  `telegram_bot.py:6326-6339` (`ApplicationBuilder`/local-mode блок), `plugins/hindsight_memory.py:1191-1209`
  (`hindsight_finalize_jobs` DDL), `plugin_manager.py:813-823` (`_normalize_specs`). Все 15
  совпали текстуально и по номеру строки — новых расхождений не найдено.
- **`bot/telegram_bot.py` `require_message`/`require_query`/`require_accessible_message`
  (`:130-169`) — `sys._getframe(1)` и PII в логах.** `sys._getframe(1)` берёт кадр
  непосредственного вызывающего — эти три функции всегда вызываются изнутри другого метода/теста
  (11+6+9 call sites, все внутри `async def`-хендлеров или тестовых функций), поэтому кадр 1
  всегда существует; `ValueError: call stack is not deep enough` возможен только если саму
  `require_*` вызвать с верхнего уровня модуля, чего в дереве нет. В лог идут только
  `f_code.co_name` (имя функции-вызывающего, не данные пользователя) и `update.update_id`
  (внутренний числовой ID апдейта Telegram, не PII) — текста сообщения, имени пользователя,
  chat_id туда не попадает. Подтверждено и тестами: `tests/test_t13a_require_helpers.py:109-136`
  явно проверяет, что в `caplog.text` присутствует имя вызывающей тест-функции (доказательство,
  что `_getframe(1)` резолвится в реального вызывающего) и что PII туда не попадает по
  построению. Реальных вызовов не осталось непокрытыми: `require_message(` — 11,
  `require_query(` — 6, `require_accessible_message(` — 9, все внутри методов класса. Замечаний
  нет.
- **README/README.ru/.env.example — новые env-переменные этого changeset.** Полный скан
  `git diff HEAD` по `os.environ`/`getenv`/`env_bool`/`_positive_int_env` во всех `.py`-файлах
  дерева (не только в моей зоне владения, т.к. `.env.example`/`README*` — моя зона документации)
  с последующей сверкой каждого кандидата против `git show HEAD:<file>` (не существовал ли он уже
  до изменений):
  - `ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER` — в `.env.example`, `README.md`, `README.ru.md`
    (таблица) и `bot/__main__.py:190` (обвязка `telegram_config`) — везде согласовано,
    default `true` совпадает во всех четырёх местах.
  - `INSTANCE_LOCK_PATH` — в `.env.example:101`, `README.md:246`, `README.ru.md:254` — текст
    описания (дефолтный путь рядом с `DB_PATH`, поведение при конфликте) идентичен по смыслу в
    обеих версиях README.
  - `MCP_ALLOW_PRIVATE_HOSTS` — не в `.env.example`/`README.md`/`README.ru.md`, но это
    осознанно: `T03-plan.md` прямо требует задокументировать её в `bot/README_MCP.md` (не в
    списке владения T03, но правка обоснована в самом плане), и она там есть
    (`bot/README_MCP.md:47,246`). `bot/README_MCP.md` вне зоны владения группы D (не входит в
    список файлов w9.txt round 1/2) — не мой файл, но конкретно про этот флаг план не требовал
    трогать `.env.example`/`README.md`, так что расхождения нет.
  - `AGENT_CRON_JOB_LEASE_SECONDS` (`bot/plugins/agent_cron.py:37`) и `REMINDER_LEASE_SECONDS`
    (`bot/plugins/reminders.py:23`) — **не переменные окружения**: это обычные
    Python-константы (`AGENT_CRON_JOB_LEASE_SECONDS = 1800`, `REMINDER_LEASE_SECONDS = 120`),
    нигде не читаются через `os.environ`/`getenv`. Задание ориентировало проверить их как
    кандидатов — по факту документировать в `.env.example` нечего.
  - `CODEINTERPRETER_MAX_DOWNLOAD_BYTES`, `GITHUB_ANALYSIS_MAX_RESPONSE_BYTES`,
    `HAIPER_MAX_VIDEO_BYTES`, `MCP_MAX_RESPONSE_BYTES`, `TEXT_SUMMARIZER_MAX_DOWNLOAD_BYTES`,
    `WEBSHOT_MAX_IMAGE_BYTES` (все из T03-plan.md, все реально новые — отсутствуют в
    `git show HEAD` для соответствующих файлов) не задокументированы в `.env.example`/README, но
    T03-plan.md не требует этого явно (только `MCP_ALLOW_PRIVATE_HOSTS` помечена как требующая
    документации из-за смены дефолтного поведения). Это соответствует существующей практике
    репозитория: скрипт-сверка показала 27 переменных окружения в `bot/*.py`/`bot/plugins/*.py`,
    отсутствующих в `.env.example` уже на `HEAD` (`GOOGLE_API_KEY`, `SPOTIFY_CLIENT_ID`,
    `MONTHLY_GUEST_BUDGET` и т.д.) — новые safety-limit-переменные с разумными дефолтами
    продолжают тот же (несовершенный, но давний) паттерн, а не новую регрессию. Не поднимаю как
    находку.
  - `LLM_RATE_LIMIT_RETRY_ATTEMPTS`/`LLM_RATE_LIMIT_RETRY_WAIT_SECONDS` — были обычными
    Python-константами в `bot/openai_helper.py` на HEAD (не env-переменные), удалены из текущего
    дерева (`git grep` в рабочем дереве — 0 совпадений); ни в `.env.example`, ни в `README.md`,
    ни в `README.ru.md`, ни в `AGENTS.md` они никогда не упоминались — устаревшей документации
    нет.
  - Найдена настоящая находка — см. WARNING ниже (`MAX_CHAT_STATES`).
- **`.github/workflows/ci.yml`** — новый шаг `Type check with mypy` (`pip install mypy` +
  `python scripts/mypy_baseline.py check`) корректен: `actions/setup-python` ставит Python 3.12
  (совпадает с `pyproject.toml`'s `[tool.mypy] python_version = "3.12"`), `mypy_baseline.py`
  без `MYPY_PYTHON` берёт `sys.executable` — тот же интерпретатор, в который шагом выше поставлен
  `mypy` и все зависимости из `requirements-dev.txt`. `pyproject.toml`'s
  `ignore_missing_imports = true` уже покрывает то, что при локальном запуске приходится
  передавать флагом `--ignore-missing-imports` — в CI это не нужно. `mypy` не запинен версией,
  но `ruff`/`pip-audit` в `requirements-dev.txt` тоже без пина — существующий паттерн, не новая
  проблема.
- **Прогоны:** `~/.venvs/ctb/bin/python -m pytest tests bot/tests -q` — 1956 passed, 3 warnings,
  63с. `ruff check bot tests bot/tests scripts` — All checks passed. `mypy bot
  --python-executable ~/.venvs/ctb/bin/python --ignore-missing-imports` — "Success: no issues
  found in 85 source files". `MYPY_PYTHON=~/.venvs/ctb/bin/python ~/.venvs/ctb/bin/python
  scripts/mypy_baseline.py check` — "mypy baseline check passed." `mypy_baseline.json` — `{}`,
  согласовано с чистым прогоном.

### WARNING — `MAX_CHAT_STATES` не задокументирован в `.env.example`/README, хотя план явно просил

- **Файл:** `.env.example` (нет строки `MAX_CHAT_STATES`); также отсутствует в `README.md` и
  `README.ru.md`.
- **Почему:** `bot/conversation_state.py:40` (новый файл, не существовал на HEAD) читает
  `DEFAULT_MAX_CHAT_STATES = _positive_int_env("MAX_CHAT_STATES", 1000)` — это настоящая, новая
  переменная окружения, ограничивающая LRU-вытеснение `ChatStateRegistry`.
  `docs/improvement_2026-09-25/T11-plan.md:685-688` прямо фиксирует это как известный пробел:
  «`.env.example` is not in T11's file ownership, so this plan does not add a line there —
  flagging as a follow-up the implementer or a later task should do (one line, `#
  MAX_CHAT_STATES=1000`, next to `MAX_CONVERSATION_AGE_MINUTES=18000` at `.env.example:63`)».
  `.env.example` — в зоне владения группы D, но правка так и не внесена (`git diff HEAD --
  .env.example` не содержит `MAX_CHAT_STATES`, весь диф файла — только `ALLOW_GROUP_MEMBERS_...`
  и `INSTANCE_LOCK_PATH`). Соседняя переменная той же категории (лимиты состояния разговора),
  `MAX_CONVERSATION_AGE_MINUTES`, при этом есть и в `.env.example:63`, и в README-таблицах обеих
  версий (`README.md:296`, `README.ru.md:304`) — так что `MAX_CHAT_STATES` явно выпадает из уже
  существующего для этой же группы переменных паттерна документирования, а не просто попадает
  под общий (терпимый) пробел, описанный выше для safety-limit-переменных T03.
- **Сценарий проявления:** Администратор, ограничивающий память процесса через `.env`, видит
  `MAX_CONVERSATION_AGE_MINUTES` в примере и в README, но не узнаёт про существование
  `MAX_CHAT_STATES` (второй независимый лимит — LRU по количеству чат-состояний, а не по
  возрасту) и не может им управлять без чтения исходников `bot/conversation_state.py`.
- **Предлагаемый fix:** В `.env.example` рядом со строкой 63 (`MAX_CONVERSATION_AGE_MINUTES=18000`)
  добавить `# MAX_CHAT_STATES=1000` с комментарием (LRU-лимит на число одновременно хранимых
  чат-состояний); добавить строку в таблицу `README.md`/`README.ru.md` рядом с
  `MAX_CONVERSATION_AGE_MINUTES` по аналогии с уже существующими строками этой таблицы.

## Раунд 3

### Summary

- ERROR: 0
- WARNING: 0
- NIT: 0

### Раунд-2 WARNING (`MAX_CHAT_STATES` не задокументирован) — исправлено, проверено консистентно с кодом

- **`.env.example:64-65`** — добавлено ровно там, где просил follow-up в
  `T11-plan.md:685-688`, сразу под `MAX_CONVERSATION_AGE_MINUTES=18000`:
  ```
  # Max number of per-chat conversation states kept in memory (LRU-evicted beyond this cap).
  # MAX_CHAT_STATES=1000
  ```
  Закомментировано (показывает дефолт), как и соседние опциональные переменные в этом же блоке.
- **`README.md:297`**: `| \`MAX_CHAT_STATES\` | \`1000\` | int | Max per-chat conversation
  states kept in memory (LRU-evicted beyond this cap). |` — вставлена в таблицу сразу после
  строки `MAX_CONVERSATION_AGE_MINUTES`, до `TEMPERATURE`.
  **`README.ru.md:305`**: тот же смысл по-русски (`Максимум состояний чатов в памяти
  (LRU-вытеснение сверх лимита).`), в той же позиции таблицы.
- **Сверка с `bot/conversation_state.py`**:
  - **Default** — `DEFAULT_MAX_CHAT_STATES = _positive_int_env("MAX_CHAT_STATES", 1000)`
    (`bot/conversation_state.py:40`) → `1000` во всех трёх местах совпадает.
  - **Смысл** — `ChatStateRegistry` (`bot/conversation_state.py:63-` докстринг: "One record per
    chat/session key... LRU" + `sweep()`: `over_cap = len(self._states) > self._max_states`)
    соответствует тексту "kept in memory (LRU-evicted beyond this cap)" во всех трёх файлах.
  - **Клэмпинг к ≥1** — `_positive_int_env` (`bot/conversation_state.py:32-37`): `max(1,
    int(os.getenv(name, str(default))))`, значение `0`/отрицательное клэмпится к `1` молча (без
    warning), а нечисловая строка откатывается на дефолт через `except ValueError`. Ни в
    `.env.example`, ни в README это поведение не описано. Проверил, является ли это пробелом:
    функция дословно совпадает (с комментарием "mirrors bot/openai_tool_handler.py:58-62") с уже
    существующим в дереве `_positive_int_env` из `bot/openai_tool_handler.py:60-63`
    (`TOOL_CALL_PARALLELISM`, `TOOL_CALL_GLOBAL_PARALLELISM`) и с одноимённой функцией в
    `bot/telegram_bot.py:69` (`PLUGIN_MENU_PAGE_SIZE`) — у этих трёх переменных в README.md/
    README.ru.md (`TOOL_CALL_PARALLELISM`/`TOOL_CALL_GLOBAL_PARALLELISM`: `README.md:312-313`,
    `README.ru.md:320-321`; `PLUGIN_MENU_PAGE_SIZE`: `README.md:381`, `README.ru.md:389`) клэмпинг
    к ≥1 тоже нигде не упомянут — это существующий паттерн документирования для этого класса
    переменных в репозитории (в отличие от `TERMINAL_OUTPUT_BYTE_LIMIT`, где клэмпинг к минимуму
    явно описан и в `.env.example:187-188`, и в `README.md:502`, — там минимум не 1, а 1024, и
    именно поэтому он документируется). Не поднимаю как находку: `MAX_CHAT_STATES` следует тому же
    established-паттерну, что и остальные `_positive_int_env`-переменные, а не выпадает из него.
    Поведение к тому же покрыто тестами (`tests/test_conversation_state.py:29-51` —
    `test_positive_int_env_falls_back_to_default_on_invalid_value`,
    `test_positive_int_env_clamps_zero_to_one`, `test_positive_int_env_clamps_negative_to_one`,
    `test_positive_int_env_parses_a_valid_value`, `test_positive_int_env_uses_default_when_unset`).
- **Диф трёх файлов с раунда 2** (`git diff HEAD -- .env.example README.md README.ru.md`,
  сверено построчно с раунд-1/раунд-2 текстом ревью): единственное изменение с раунда 2 — эти 4
  строки (`MAX_CHAT_STATES` в `.env.example` + по одной строке в каждой таблице). Опечаток,
  рассинхрона порядка колонок или формата таблицы не найдено; расположение строки в обеих
  README-таблицах идентично порядку в `.env.example` (сразу за `MAX_CONVERSATION_AGE_MINUTES`).
  Остальной диф файлов (`ALLOW_GROUP_MEMBERS_VIA_AUTHORIZED_USER`, `INSTANCE_LOCK_PATH`,
  переформулировка `GUEST_BUDGET`) уже проверен в раунде 2 (первые два пункта) или относится к
  зоне владения группы A (`GUEST_BUDGET`/`is_allowed`/`_charge_user_and_guest*` — за пределами
  зоны D по шапке этого файла), не переоткрываю.

### Полная верификация

- `~/.venvs/ctb/bin/python -m pytest tests bot/tests -q --no-header -p no:cacheprovider` — 1959
  passed, 3 warnings, 65.86s.
- `~/.venvs/ctb/bin/python -m ruff check bot tests scripts` — All checks passed!
- `python3 -m mypy bot --python-executable ~/.venvs/ctb/bin/python` — "Success: no issues found
  in 85 source files" (только информационные `annotation-unchecked` note-строки для untyped
  функций, как и в раундах 1-2).
- `MYPY_PYTHON=~/.venvs/ctb/bin/python python3 scripts/mypy_baseline.py check` — "mypy baseline
  check passed."
