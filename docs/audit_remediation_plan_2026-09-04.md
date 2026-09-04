# План доработки по аудиту 2026-09-04

Источник: `docs/architecture_code_review_2026-09-04.md`. Скорректирован по указаниям владельца:

- Зависимости **поднимаем** до актуальных версий и приводим код к ним (openai 3.x, mcp 2.x,
  python-telegram-bot 22.8), а не ограничиваем сверху.
- `deep_analysis` (codeinterpreter) **не выключаем** и не трогаем.
- Прокси **должен работать**, если задан в настройках.
- `plugin_tool_adapter.py` и `_guard_tool_call` **не удаляем**.
- Из гигиены репозитория делаем только: убрать `bot/plugins/pdf_cache/cache_metadata.json` и
  `IMG_3980.jpg` из индекса. `plantuml.jar`, `ai_docs_site/`, `environment.yml` не трогаем.

Ограничения: коммиты и ветки не создаются; внешние сервисы при проверке не вызываются;
`.venv` проекта недоступен по правам, для проверки создаётся `~/.venvs/ctb` через `uv`.

Процесс на каждую задачу: планировщик (Sonnet, только чтение) → разработчик (Sonnet) →
код-ревью (Sonnet) → исправление замечаний. Задачи внутри волны, не пересекающиеся по файлам,
выполняются параллельно. Планы задач: `docs/remediation_2026-09-04/T##-*.md`.

Целевые версии (PyPI, 2026-09-04): openai 3.8.0, httpx2 2.12.0, mcp 2.1.1,
python-telegram-bot 22.8 (httpx>=0.27,<0.29), Pillow 12.3.0, lxml 6.1.3, lxml-html-clean 0.4.5,
spotipy 2.26.0, tiktoken 0.14.0, ddgs 9.16.0, pytubefix 10.11.0, telegramify-markdown 1.2.0,
readability-lxml 0.9, Pygments 2.21.0.

---

## Волна 1 — зависимости, инфраструктура, изолированные исправления (параллельно)

### T01. Зависимости: поднять версии и привести код
- `requirements.txt`: `openai>=3.8,<4`, `httpx2>=2.7,<3` (транзитивно от openai), `mcp>=2.1,<3`,
  `python-telegram-bot>=22.8,<23`, `httpx>=0.27,<0.29` (нужен PTB), `Pillow>=12.3`, `lxml>=6.1.3`,
  `lxml-html-clean>=0.4.5`, `readability-lxml>=0.9`, `spotipy>=2.26`, `tiktoken>=0.14`,
  `ddgs` вместо `duckduckgo_search`, `pytubefix` вместо `pytube`, `telegramify-markdown>=1.2`,
  явно `Pygments`, `PyPDF2` (если `ask_your_pdf` его использует). Удалить 33 неиспользуемых
  (проверить AST-скриптом: `asyncpg`, `telethon`, `opencv-python`, `scikit-learn`, `fastapi`,
  `uvicorn`, `moviepy`, `pygame`, `nltk`, `tenacity`, `youtube-transcript-api`,
  `google-api-python-client`, `chardet`, `pyswisseph`, `assemblyai`, `pytz`, `SpeechRecognition`,
  `jinja2`, `Markdown`, `deep-translator`, `countryinfo`, `forex_python`, `qrcode`,
  `typing-extensions`, `ratelimit`, `ffmpeg-python`, `requests-toolbelt`, `trafilatura`,
  дубликат `beautifulsoup4`). Для `codeinterpreter` оставить `numpy pandas matplotlib plotly sympy`.
- `requirements-dev.txt`: `-r requirements.txt` + `pytest`, `pytest-asyncio`, `ruff`, `pip-audit`.
- Код под openai 3.x: `bot/openai_helper.py:255-275` — `http_client` строить из `httpx2.AsyncClient`
  (`proxy=config['proxy']` если задан); убрать `openai.api_base = ...` (`:265`, мёртвая строка
  из v0 API). Проверить остальные места использования SDK-типов (`CompletionUsage`,
  `ChatCompletionChunk`, `.construct`) в `bot/ai_providers/openai_compatible.py`,
  `bot/openai_tool_handler.py`, тестах.
- Код под PTB 22.8: `bot/telegram_bot.py:6288-6303` — `builder.proxy(config['proxy'])` и
  `get_updates_proxy` если задан; проверить `Defaults`, `proxy_url`, таймауты `start_polling`.
- `bot/plugins/ddg_translate.py:3` → `from ddgs import DDGS`; `bot/plugins/youtube_audio_extractor.py`
  → `pytubefix`.
- Dockerfile: убрать `g++ libc6-dev` если сборка проходит без них.
- Проверка: `uv venv ~/.venvs/ctb && uv pip install -r requirements-dev.txt`, полный
  `~/.venvs/ctb/bin/python -m pytest -q` зелёный; `pip-audit` без findings по нашим пинам.
- Файлы: `requirements.txt`, `requirements-dev.txt`, `Dockerfile`, `bot/openai_helper.py`
  (только конструктор), `bot/telegram_bot.py` (только `run()`), `bot/plugins/ddg_translate.py`,
  `bot/plugins/youtube_audio_extractor.py`, `tests/test_telegram_builder_config.py`.

### T03. CI и инфраструктура
- `.github/workflows/ci.yml`: Python 3.12, `pip install -r requirements-dev.txt`,
  `ruff check bot tests`, `python -m pytest -q`, `pip-audit -r requirements.txt`; актуальные
  мажоры actions. Удалить `.github/workflows/python-package-conda.yml`.
- `pyproject.toml` (новый, только `[tool.ruff]` с текущими правилами E/F, target 3.12) и починить
  2 × F401 в тестах.
- `.dockerignore`: `bot/skills/`, `ai_docs_site/`, `tests/`, `.ai-docs/`, `.attachments/`,
  `docs/`, `evals/`, `examples/`, `*.md`, `.ruff_cache`.
- `docker-compose.yml`: `TELEGRAM_LOCAL_MODE: ${TELEGRAM_LOCAL_MODE:-false}`; README:235 сверить.
- `chmod 600 .env` (локальная система).
- README.md/README.ru.md: «Python 3.11+».

### T04. Гигиена репозитория
- `git rm --cached bot/plugins/pdf_cache/cache_metadata.json`, `bot/plugins/pdf_cache/` в
  `.gitignore`; `bot/plugins/ask_your_pdf.py:34-45` — не создавать каталоги и файл в `__init__`,
  только в `initialize()`.
- `git rm IMG_3980.jpg`.

### T05. Политика терминала: обёртки команд
- `bot/command_policy.py`: перед матчингом нормализовать `( … )`, `{ …; }`, `timeout N`, `nohup`,
  `xargs`, `for … do … done`, `if … then … fi`, `env`, `command`, `builtin`; при `curl … | sh`
  — правило «pipe в shell». Тесты в `tests/test_command_policy*.py` на все перечисленные формы.

### T06. Провайдер: `None` вместо 0 для отсутствующих токенов
- `bot/ai_providers/openai_compatible.py` `_usage`/`_int_or_zero`: отсутствующее поле →
  `None`; потребители (`bot/chat_run.py:242`, `bot/utils.py:854-862`, `bot/pricing.py`,
  `bot/telegram_bot.py:805-811`) переносят `None` в `aggregate_usage_split`/`resolve_chat_cost`
  (ветка `model_blended`). Тест: `CompletionUsage.construct(total_tokens=5)` → `model_blended`.

### T07. Блокирующие HTTP в плагинах
- `bot/plugins/webshot.py:37,40`, `bot/plugins/movie_info.py:117,143,169,205`: `timeout=` и
  `asyncio.to_thread` (или `aiohttp`); `webshot` — `os.remove` под `contextlib.suppress`.
- `bot/plugins/chief.py:194`: `close_async()` закрывает `aiohttp.ClientSession`.

## Волна 2 — зависания и корректность ядра (после T01, на новом venv)

### T02. MCP 2.x и stdio-обнаружение инструментов
- `bot/plugins/mcp_server.py`: API mcp 2.x (транспорты, `httpx2`, snake_case поля, `float`
  таймауты); `_fetch_stdio_tools:364-386` — `mcp_tools.tools`, `tool.inputSchema` →
  OpenAI-схема; тот же маппинг для HTTP-ветки. `bot/tests/test_mcp_server.py` обновить.

### T08. Deadlock: `DbHandle.transaction` + sync-чтение настроек на loop
- `bot/plugin_manager.py:176-196` `disabled_plugins_for_user`: убрать sync `get_user_settings` из
  потока loop — либо async-вариант с предварительной загрузкой в `user_settings_scope` на всех
  путях (`bot/telegram_bot.py:883`, `:1288`, `:1583`, `bot/plugins/agent_cron.py:290`,
  `bot/openai_helper.py:1302`), либо кэш, который при отсутствии scope не ходит в БД синхронно.
- `bot/openai_helper.py:2037` → async `get_current_model`; `bot/telegram_bot.py:258,268` —
  удалить мёртвый sync `_get_user_language`.
- `bot/plugins/db_handle.py`: docstring `transaction()` — предупреждение о вызовах
  `*_async`/`fetch_*` внутри тела; guard, который бросает понятную ошибку вместо зависания.
- Тест: транзакция плагина + параллельный запрос настроек завершаются (с `wait_for`).

### T09. Вложенный `get_chat_response` из плагинов
- `bot/openai_helper.py:874-895`: guard `_in_active_turn` как у `ask()` (`:795`) — при вложенном
  вызове бросать `RuntimeError` с понятным текстом (или делегировать в `ask()`).
- Плагины `ask_your_pdf.py:364`, `language_learning.py:187`, `conversation_analytics.py:233,253,
  271,289`, `show_me_diagrams.py:161,186` → `helper.ask(...)`/`ModelUtilities.one_shot`.
- Тесты: `tests/test_ask_your_pdf.py` без `AsyncMock` на `get_chat_response`; тест на guard.

### T10. Пустые `tools` и жёсткий лимит re-entry
- `bot/openai_helper.py:467-468`: `if tools:`; `bot/openai_tool_handler.py:1046-1047, 1645-1647`:
  пустой список → `tools=None`, `tool_choice="none"`.
- `:1495` — `final_delivery_required` только если `deliver_to_user` входит в allow-list режима;
  `_reentry_tool_choice:905-910` — после `max_consecutive_calls + N` (N=2) принудительно `"none"`.
- Тесты в `tests/test_openai_helper_tool_calls.py`.

### T11. `get_conversation_context`: ошибка ≠ «нет данных»
- `bot/database.py:939-982`: исключение пробрасывать (или возвращать явный маркер), аннотация
  по факту (кортеж); дефолт `max_tokens_percent` единый.
- Потребители `bot/openai_helper.py:1178-1189, 1457-1466, 2460-2469`: при ошибке чтения не
  создавать новую сессию — вернуть ошибку пользователю. Тест в `tests/test_database.py`.

## Волна 3 — Telegram-слой (последовательно, один файл)

### T12. Единый стриминговый рендерер и повтор финального чанка
- Новый `bot/telegram_stream.py`: `stream_to_telegram(update, context, chunks, *, on_direct_result,
  ...)` с backoff, чанкованием, `RetryAfter`-повтором и **гарантированной доставкой финального
  чанка**. Три места (`bot/telegram_bot.py:3229-3400`, `:4096-4619`, `:4848-5025`) переводятся на
  него. Тесты: `tests/test_telegram_streaming.py` + репро «ошибка edit на финале».

### T13. Замки и мелкие ошибки Telegram-слоя
- `_process_vision_media_group:2925-3084` и `handle_callback_inline_query:4848-5025` —
  `_get_conversation_lock`.
- `escape_markdown` для `str(e)`/`query` в `:2605, 2659, 2727, 2852, 3198, 4985`; `split_into_chunks`
  в `:1071-1082`, `:3040-3052`; `vision` `return` после `media_type_fail` `:3217-3225`; имя
  файла/mime в промпт как данные (`:528, 534-548`, по образцу `_forwarded_text_prompt`);
  `_edit_image_from_context` учитывает usage.

### T14. Учёт расходов и PIL вне loop
- `_record_chat_usage` и `record_*` (`bot/telegram_bot.py:799-811, 1064, 1082, 2598, 2652,
  2768, 3044`) → `asyncio.to_thread`; `is_within_budget:5051` → `to_thread`;
  `_telegram_image_as_png:1020-1025` → `to_thread`.

## Волна 4 — архитектура (П1 → П2 → П3 → П4, П5 параллельно с П1)

### T15. Один non-stream путь
- Удалить legacy-тело `bot/openai_helper.py:934-1091` и флаг `chat_run_variant_b_enabled`
  (`bot/__main__.py:214`, `bot/openai_helper.py:342, 688-690, 922`); `ChatRun` перевести на
  публичные методы helper (`_common_get_chat_response`, `_handle_function_call`,
  `_add_to_history` вместо name-mangled). Тесты с `= False` удалить/переписать.

### T18. Индекс «имя функции → (плагин, spec)»
- `bot/plugin_manager.py`: строить в `load_plugins`/`reload_plugins` рядом с
  `_model_tool_name_to_canonical`; `get_spec_by_function_name`, `get_plugin_name_by_function_name`,
  `is_function_allowed` читают индекс; MCP-плагин с динамическими спеками инвалидирует индекс.

### T16. Удалить мёртвый провайдер-слой
- `bot/model_constants.py:19-30` и все ветки `in (O_MODELS + …)` в `bot/openai_helper.py`,
  `bot/plugin_manager.py:20,357`, `bot/telegram_bot.py:45,4224,4907`. Решение о стриме — из
  конфига, не из семейства модели.

### T17. Один источник конфига
- `bot/__main__.py`: `shared` для 14 дублей, единая мягкая политика булевых; удалить
  дублирующие `setdefault` в `bot/openai_helper.py:309-342` (оставить нужные тестам с теми же
  значениями); `Database.configure(db_path, max_sessions, journal_mode, default_model)` вместо
  чтения env в `bot/database.py:26-60, 79, 1164`; плагинам — только prefix-сегмент.

### T19. Публичный API сессии в helper
- `OpenAIHelper.load_session`, `replace_system_message`, `evict`; бот (`bot/telegram_bot.py:
  2436-2481, 3701, 4067, 6149-6174`) перестаёт трогать `conversations`/приватные методы.
  Линтер-тест на `getattr(self.openai, '_`.

## Волна 5 — тесты и документация

### T20. Общие тестовые заглушки
- `tests/fakes.py` (`FakeMessage`, `FakeDB`, `FakeHelper`, `FakeEncoding`, `FakePluginManager`)
  + фикстуры в `tests/conftest.py`; мигрировать самые частые дубли; `assert` в
  `test_concurrent_access_smoke`.

### T21. Документация
- README/README.ru: убрать 8 несуществующих env, добавить недокументированные;
  `.env.example`: убрать 5 мёртвых ключей, добавить отсутствующие; AGENTS.md: все `file:line`
  одним проходом по итоговому коду; `.cli-proxy/.codebase_map/nodes/*.md` `Last reviewed`.

## Волна 6 — финальный цикл ревью
Субагент-ревьюер по полному `git diff` → исправление → повтор до «нет ошибок и предупреждений».
