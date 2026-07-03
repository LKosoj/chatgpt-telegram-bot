# План доработки по аудиту chatgpt-telegram-bot

Дата: 2026-07-02.

Источник: `docs/architecture_code_review_2026-07-02.md`.

Использованные роли: `dev-experts` (`architect`, `python-dev`, `tester`, `reviewer`),
`97-dev`, `brainstorming`. Предварительные планы подготовлены subagents по четырём
дорожкам: core, persistence, plugins, delivery/infra.

## Ограничения и допущения

- `bot/plugins/skills.py` default-open поведение не меняется: дефолтные разрешения
  установок/запуска скриптов, модель `confirmed` и текущая доступность исполнения остаются
  как есть по прямому требованию пользователя. Риски из аудита фиксируются как
  осознанно принятые, а не как забытая работа.
- Все остальные пункты аудита берутся в работу в порядке ниже. Если пункт нельзя закрыть
  одной безопасной правкой, он разбивается на инкременты с отдельными тестами.
- Внешние сервисы не вызываются при проверке, если конкретная задача этого не требует.
- Коммиты и ветки не создаются.
- Основной интерпретатор для проверок: `.venv/bin/python`.

## Критерии готовности

1. Все P0, кроме явно исключённого поведения `skills.py`, закрыты кодом и regression-тестами.
2. P1 закрыты архитектурными инкрементами без больших переписываний ради переписывания.
3. P2 закрыты cleanup/test/doc/infra правками либо сведены к удалённым/неактуальным пунктам
   после проверки кода.
4. Обновлены затронутые `.cli-proxy/.codebase_map/nodes/*.md` `Last reviewed`.
5. Полный `.venv/bin/python -m pytest -q` зелёный.
6. Финальный reviewer-loop subagent не возвращает ошибок и предупреждений.

## Порядок исполнения

### Волна 0 — baseline и safety rails

- Зафиксировать текущий `git status --short`.
- Запустить targeted baseline для известных падающих областей:
  `.venv/bin/python -m pytest bot/tests/test_mcp_server.py tests/test_llm_gateway_client.py tests/test_agent_tools_verify.py::test_clear_plan_clears_pending_verify -q`.
- Не блокировать P0 на baseline, но отделять pre-existing failures от новых регрессий.

### Волна 1 — P0: реальные поломки и безопасность

1. `P0-01` Voice flow NameError.
   - Файлы: `bot/telegram_bot.py`, focused Telegram/usage test.
   - Правка: `_convert_audio()` возвращает `duration_seconds`; `record_transcription_seconds`
     получает это значение вместо несуществующего `audio_track`.
   - Проверка: successful transcribe test с fake `AudioSegment` и `openai.transcribe`.

2. `P0-02` Inline callback auth bypass.
   - Файлы: `bot/telegram_bot.py`, `tests/test_callback_authorization.py`.
   - Правка: `handle_callback_inline_query` делает early `check_allowed_and_within_budget`.
   - Проверка: unauthorized `gpt:<id>` не вызывает OpenAI, не мутирует cache/usage.

3. `P0-03` Tool reentry hard limit.
   - Файлы: `bot/openai_tool_handler.py`, `tests/test_openai_helper_tool_calls.py`.
   - Правка: жёсткий `functions_max_consecutive_calls` применяется всегда; delivery escape
     разрешён только для реального delivery-контракта, не для любого успешного tool.
   - Проверка: generic successful tool на лимите не получает новый tool loop; текущий
     delivery-after-artifact сценарий остаётся рабочим; delivery reentry/repair после
     лимита видят только `agent_tools.deliver_to_user`, plain-text tool-intent repair после
     лимита не получает tools; reviewer subagent вернул `NO FINDINGS / NO WARNINGS`.

4. `P0-04` RateLimit retry duplicates user messages.
   - Файлы: `bot/openai_helper.py`, `tests/test_openai_helper_tool_calls.py`.
   - Правка: убрать retry с методов, которые мутируют историю; ретраить только LLM create
     или оставить один SDK-level retry.
   - Проверка: fake RateLimit не создаёт два user-message в history/DB; reviewer subagent
     вернул `NO FINDINGS / NO WARNINGS`.

5. `P0-05` Hindsight finalize poison-pill.
   - Файлы: `bot/plugins/hindsight_memory.py`,
     `tests/test_hindsight_finalize_worker_integration.py`.
   - Правка: JSON parse per row; bad job помечается через failed/retry policy, валидные jobs
     из того же batch продолжают обработку.
   - Проверка: corrupt `messages` не блокирует валидную finalize job.

6. `P0-06` UsageTracker JSON corruption.
   - Файлы: `bot/usage_tracker.py`, `tests/test_usage_record_helpers.py`.
   - Правка: единый atomic write через temp file + `os.replace`, `parents=True`,
     in-process lock; corrupt JSON переименовывается и не роняет пользователя.
   - Текущее состояние: legacy JSON остаётся только источником импорта/snapshot; новые usage
     events пишутся в SQLite (`usage.sqlite3`).
   - Проверка: corrupt JSON test, atomic write test, concurrent snapshot overwrite test,
     existing usage helper tests; reviewer subagent вернул `NO FINDINGS / NO WARNINGS`.

7. `P0-07` HTML XSS/injection.
   - Файлы: `bot/html_utils.py`, `tests/test_html_utils_responsive.py` или security test.
   - Правка: не де-экранировать `<pre><code>`; Mermaid/URL вставки экранировать через
     `html.escape(..., quote=True)`; проверить Mermaid security-level; encoded Mermaid
     arrows/line-breaks нормализуются без декодирования `&lt;script` в тег.
   - Проверка: fenced code `<script>`, Mermaid `</div><script>`, URL с кавычками,
     `_create_mermaid_container` encoded script/arrows/line-breaks; reviewer subagent
     вернул `NO FINDINGS / NO WARNINGS`.

8. `P0-08` Blocking HTML delivery on event loop.
   - Файлы: `bot/utils.py`, `bot/html_utils.py`, delivery tests.
   - Правка: `advanced_visualization` и чтение файла через `asyncio.to_thread`; guard на
     `None`/missing output path.
   - Проверка: `send_long_response_as_file` вызывает `to_thread` и отправляет документ.

9. `P0-09` Database lifecycle flake/cross-instance close.
   - Файлы: `bot/database.py`, `tests/test_database.py`,
     `tests/test_agent_tools_verify.py`.
   - Правка: `_local` становится instance attr; singleton присваивается после `init_db`;
     `__del__` не закрывает thread-local другого живого instance.
   - Проверка: DB lifecycle tests + flaky agent-tools verify test.

10. `P0-10` requests CVE.
    - Файл: `requirements.txt`.
    - Правка: `requests>=2.32.4,<3`.
    - Проверка: dependency file inspection + targeted tests; full install/build отдельным
      infra gate.

11. `P0-SKILLS` Accepted risk.
    - Файл: `bot/plugins/skills.py`.
    - Статус: не менять default-open поведение по требованию пользователя.
    - Нельзя менять в этой волне: `SKILLS_SCRIPT_ADMIN_USER_IDS="*"`,
      `SKILLS_ALLOW_INSTALLS=True`, `SKILLS_INSTALL_ADMIN_USER_IDS="*"`, модель
      `confirmed` и env-поведение subprocess, если это меняет совместимость исполнения.

### Волна 2 — P1: высокий рычаг, без больших прыжков

1. `P1-A` Завершить async DB facade. **STATUS: DONE (2026-07-02).**
   - Подтверждено, что основные wrappers уже есть: settings, session/context, image,
     export, prune.
   - Убраны подтверждённые async-path обходы:
     `openai_helper.reset_chat_history` больше не зовёт `Database.get_mode_from_context`
     из async-пути; system-message парсится локально и устойчив к `messages=None`.
     `_dispatch_before_create_session_prune` закреплён тестом на async DB facade.
     `agent_tools.run_subagents` предпочитает `get_current_model_async`.
     `haiper_image_to_video` переводит все 5 async image-read paths на
     `get_user_images_async` при наличии wrapper.
   - Tests: `tests/test_reset_chat_history_async.py`,
     `tests/test_openai_helper_db_offload.py`, `tests/test_agent_tools_plugin.py`,
     `tests/test_haiper_image_to_video_async_db.py` — `101 passed`.
   - Reviewer loop: core reset/pre-prune — clean; `agent_tools` — clean; `haiper` —
     clean (`NO FINDINGS / NO WARNINGS`).
   - Static check documented exceptions:
     legacy fallback внутри `haiper._get_user_images` при отсутствии async API;
     sync public `OpenAIHelper.get_current_model`;
     sync `_get_user_language` для sync callers;
     shutdown lifecycle `self.db.shutdown`;
     plugin-owned direct DB internals в `agent_tools` вне async core facade.

2. `P1-B` DB robustness и retention. **STATUS: DONE (2026-07-02).**
   - Реализация уже была закрыта в текущем коде: rollback при commit error,
     `SQLITE_JOURNAL_MODE` whitelist, migration registry, индексы
     `images`/`tool_call_events`, prune methods.
   - Усилены тесты без production changes: registry invariant
     `TARGET_SCHEMA_VERSION`, commit-failure rollback + cross-thread lock release,
     retention index columns, cutoff-safe prune tests, `cleanup_old_images` rowcount.
   - Tests: `tests/test_database.py` — `44 passed`; P1-A/P1-B bundle — `145 passed`.
   - Reviewer loop: первый review нашёл flaky cutoff margins и same-thread lock check;
     после исправления re-review вернул `NO FINDINGS / NO WARNINGS`.

3. `P1-C` Tool/plugin boundary hardening. **STATUS: DONE (2026-07-02).**
   - `PluginManager.call_function` теперь при наличии `RequestContext` удаляет модельные
     `chat_id`, `user_id`, `message_id`, `request_context` и передаёт в `plugin.execute`
     только доверенные framework-значения из контекста.
   - Legacy-путь без `RequestContext` сохранён: `openai_tool_handler` уже очищает
     framework-аргументы модели и инжектит доверенные `chat_id`/`user_id` перед вызовом
     менеджера.
   - Закреплено тестами: strict schema + framework args, legacy overwrite в handler,
     `CancelledError` через реальный `PluginManager`, `config=None` default-load,
     module-local plugin class filter, missing-spec error telemetry, sanitized-name
     collision round-trip.
   - Tests: `tests/test_plugin_manager.py`, `tests/test_plugin_arg_validation.py`,
     `tests/test_plugin_chat_id_contract.py`, `tests/test_openai_helper_tool_calls.py`,
     `tests/test_tool_result.py`, `bot/tests/test_mcp_server.py` под `-W error` —
     `142 passed, 1 skipped`.
   - Reviewer loop: `NO FINDINGS / NO WARNINGS`.

4. `P1-D` Central authorization. **STATUS: DONE (2026-07-02).**
   - Known bypasses уже закрыты текущим кодом и закреплены тестами: `/help` не раскрывает
     plugin help unauthorized user; inline `gpt:` callback не трогает cache/OpenAI без
     allow/budget.
   - Добавлен TTL cache для group membership checks в `utils.is_user_in_group`, включая
     покрытие положительного cache-hit/expiry и неожиданный `BadRequest`, который не
     кэшируется и не проглатывается.
   - Небюджетные command/callback paths переведены на `_ensure_allowed`; budgeted
     `prompt`/media/inline paths остались на `check_allowed_and_within_budget`, `/restart`
     остался admin-only.
   - Введён registration-level wrapper для non-budget command/callback handlers; plugin
     disabled checks сохранены внутри plugin handlers.
   - Исправлен lifecycle leak в `run()`: PTB получает `close_loop=created_loop`, cleanup не
     запускается через `run_until_complete` на running loop, created loop закрывается только
     если он действительно создан ботом.
   - Tests: `tests/test_telegram_builder_config.py`, `tests/test_telegram_streaming.py`,
     `tests/test_callback_authorization.py`, `tests/test_plugin_handlers_registration.py`,
     `tests/test_usage_budget.py` под `-W error` — `90 passed`; P1-C/P1-D combined gate —
     `231 passed, 1 skipped`.
   - Reviewer loop: первый review нашёл lifecycle bug и недостающую BadRequest coverage;
     после исправлений re-review вернул `NO FINDINGS / NO WARNINGS`.

5. `P1-E` Delivery consolidation. **STATUS: DONE (2026-07-03).**
   - `split_into_chunks` корректно закрывает/переоткрывает fenced code blocks на границах
     чанков и не превращает inline ```` ```ticks``` ```` в persistent fence state.
   - `agent_delivery` использует общий direct-result payload parser, основной
     renderer/entities для markdown-текста и общий cleanup helper для успешной отправки
     одиночных `format=path` artifacts.
   - Final artifacts сохраняются через `preserve_after_delivery=True`; одиночные path
     artifacts удаляются после успешной отправки, включая нормализованные `~/...` пути.
   - Post-delivery cleanup directives (`cleanup_skill`, `cleanup_skills`) закреплены
     тестами для sync/async cleanup, мусорных директив и successful direct-result delivery.
   - Tests: `tests/test_agent_delivery.py`, `tests/test_plugin_direct_results.py`,
     `tests/test_telegram_streaming.py`, `tests/test_telegram_markdown_entities.py`,
     `tests/test_utils_send_long_response_file.py` под `-W error` — `71 passed`.
   - Reviewer loop: первый review нашёл inline triple-backtick edge case, второй —
     cleanup `~/...` path leak; оба исправлены regression-тестами. Re-review для fenced
     blocks, path cleanup и directive coverage вернул `NO FINDINGS / NO WARNINGS`.

6. `P1-F` LLM/provider/config correctness. **STATUS: DONE (2026-07-03).**
   - Provider-family requests теперь не отправляют одновременно `max_tokens` и
     `max_completion_tokens`: закрыты основной chat path и empty-response retry after tools.
   - `MODEL_CONTEXT_WINDOWS`/`model_context_windows` закреплены тестами, включая invalid и
     non-positive env entries с warning fallback.
   - `chat_modes_registry` закреплён тестами на initial/reload YAML parse fallback,
     `content=None`, defensive `all_modes()` copy и non-mapping mode entries.
   - `llm_gateway_client` закреплён тестами на `httpx.HTTPError` wrapping для
     `post_json`/`get_json`/`post_multipart`, capped error body, private temp image dir и
     восстановление cached temp dir после удаления.
   - Тестовая стабилизация event-loop ownership сохранила контракт `close_loop=True` для
     loop, созданного ботом, и `close_loop=False` для уже существующего current loop.
   - Tests: `tests/test_openai_helper_tool_calls.py`, `tests/test_telegram_builder_config.py`,
     `tests/test_plugin_handlers_registration.py`, `tests/test_chat_modes_registry.py`,
     `tests/test_llm_gateway_client.py` под `-W error` — `147 passed`.
   - Reviewer loop: первый review нашёл duplicate token-param в empty-response retry, второй —
     stale cached temp dir в `llm_gateway_client`; оба исправлены regression-тестами.
     Lane A/B/C re-review вернул `NO FINDINGS / NO WARNINGS`.

7. `P1-G` Plugin large modules. **STATUS: DONE (2026-07-03).**
   - Production-код по audit bullets уже был закрыт текущим деревом: MCP constructor не
     грузит/пишет config до `initialize()`, corrupt config сохраняет live `servers`;
     `DbHandle.transaction()` использует реальный `Database.transaction()`;
     AgentTools не создаёт repo-local runtime files по умолчанию, clear удаляет plan
     contract/checkpoint/runtime state, runtime dict mutations применяются на event loop,
     plan-scope locks bounded; Hindsight идёт через `DbHandle`, без `.database`
     reach-through.
   - Добавлены regression-контракты: locked AgentTools plan-scope lock не эвиктится до
     release; Hindsight finalize extraction и dream extraction выполняются вне user lock,
     а retain/complete остаются под lock.
   - Static check: `rg` не нашёл `db_handle.database`/`.database` reach-through в
     `hindsight_memory.py`, `agent_tools.py`, `db_handle.py`, `mcp_server.py`.
   - Tests: `bot/tests/test_mcp_server.py`, `tests/test_db_handle.py`,
     `tests/test_agent_tools_plugin.py`, `tests/test_hindsight_finalize_worker.py`,
     `tests/test_hindsight_finalize_worker_integration.py`, `tests/test_hindsight_memory.py`
     под `-W error` — `119 passed, 1 skipped`.
   - Reviewer loop: `NO FINDINGS / NO WARNINGS`.

8. `P1-H` Runtime config and shutdown. **STATUS: DONE (2026-07-03).**
   - Единая мягкая политика numeric env parsing в `__main__.py`: malformed и non-finite
     float значения (`nan`/`inf`) fallback-ят с warning, включая list-prices и
     `MODEL_CONTEXT_WINDOWS`.
   - Дополнительно закрыты прямые runtime numeric env parses вне `__main__.py`:
     SQLite timeout/busy-timeout и `PLUGIN_MENU_PAGE_SIZE` теперь fallback-ят с warning.
   - `VOICE_REPLY_PROMPTS` парсится как semicolon-list без пустого `[""]`, включая
     unset/explicit empty env.
   - `cleanup()` отменяет только owned tasks, вызывает `db.shutdown()` после OpenAI/plugin
     close и остаётся idempotent при `post_shutdown` + defensive `finally`.
   - Tests: `tests/test_telegram_builder_config.py`, `tests/test_database.py`,
     `tests/test_plugin_handlers_registration.py` под `-W error` — `75 passed`.
   - Static check: нет прямых `int/float(os.getenv(...))` и `asyncio.all_tasks()` в
     P1-H runtime-файлах; `bot/plugins/skills.py` без diff.
   - Reviewer loop: `NO FINDINGS / NO WARNINGS`.

### Волна 3 — P2: поддерживаемость, cleanup, UI/infra

1. Разбить god-модули инкрементами: streaming helpers, Telegram session UI, OpenAI vision,
   summarization, budgeting, plugin stores/workers.
   - `P2-1.1` Telegram session-mode display lookup: ручные `prompt_start` loops в
     `stats()`/`reset()` заменены на общий `_session_mode_display_name()` с приоритетом
     `mode_key`, fallback на `prompt_start` и `prompt_markers`; malformed `mode_key` не
     роняет UI.
   - Tests: `tests/test_group_session_flow.py`, `tests/test_chat_modes_registry.py`,
     `tests/test_callback_authorization.py` под `-W error` — `58 passed`.
   - Reviewer loop: `NO FINDINGS / NO WARNINGS`.
2. Удалить проверенный dead code: устаревший отдельный oldest-session deletion API,
   старый markdown-конвертер, dead `html_utils` helpers и transient image attribute уже
   удалены; PlantUML live path починен через packaging/path; `conversations_vision`
   удалён после проверки.
3. Usage migration в SQLite: `usage_events`/aggregates, import old JSON, compatibility API
   — реализовано локальным SQLite store внутри `UsageTracker`.
   - `P2-3.1` Usage retention API: `_UsageSQLiteStore.prune_events()` удаляет старые
     raw events без удаления `usage_daily_aggregates`, чтобы сохранить all-time billing;
     `tts_characters` raw rows сохраняются, потому что это единственный источник
     model-level breakdown для compatibility snapshots.
   - `load_usage_history(history_days=...)` и `UsageTracker.prune_store(...)` позволяют
     записать bounded JSON snapshot, сохраняя all-time cost из aggregates.
   - Tests: `tests/test_usage_record_helpers.py` под `-W error` — `37 passed`.
   - Reviewer loop: первый review нашёл потерю TTS breakdown после raw-event pruning;
     после исправления re-review вернул `NO FINDINGS / NO WARNINGS`.
   - `P2-3.2` Retention lifecycle wiring: добавлен bot-owned background loop
     `retention_cleanup_loop()` и ручной `run_retention_cleanup_once()` для pruning
     `tool_call_events`, старых image rows, активных `UsageTracker` stores и session logs.
   - Env: `RETENTION_CLEANUP_INTERVAL_SECONDS`, `TOOL_CALL_EVENT_RETENTION_DAYS`,
     `IMAGE_RETENTION_DAYS`, `USAGE_RETENTION_DAYS`; отрицательные значения fallback'ятся
     на defaults, чтобы не превращаться в aggressive prune.
   - Resource fixes: usage SQLite connections теперь закрываются явно; pytest baseline
     event loop закрывается в `tests/conftest.py`; `SessionLogger._drain_summary_tasks()`
     даёт done-callback'ам удалить завершённые summary tasks.
   - Usage bounded JSON snapshot теперь durable после `prune_store(history_days=...)`:
     последующие usage writes сохраняют тот же history window; runtime trackers получают
     `usage_retention_days` через `make_usage_tracker()`.
   - Собственные compatibility snapshots помечаются в `usage_imports`, чтобы следующий
     `UsageTracker(...)` не импортировал их как legacy JSON и не удваивал usage/cost.
   - Tests: `tests/test_telegram_builder_config.py`, `tests/test_database.py`,
     `tests/test_usage_record_helpers.py`, `tests/test_session_logger.py`,
     `tests/test_session_logging_integration.py` под `-W error` — `150 passed`.
   - Reviewer loop: первый review нашёл negative retention env aggressive prune и
     недолговечный bounded usage snapshot; re-review нашёл negative
     `SESSION_LOG_RETENTION_DAYS`; второй re-review нашёл re-import собственных JSON
     snapshots; после исправлений третий re-review вернул `NO FINDINGS / NO WARNINGS`.
4. SessionLogger single queue writer + rotation/TTL. **STATUS: DONE (2026-07-03).**
   - `SessionLogger.record()` больше не создаёт `to_thread` task на каждое событие:
     JSONL-записи идут через один queue writer, `drain()` ставит barrier, `close()`
     останавливает writer после flush.
   - `record()` копирует event dict перед добавлением `ts`, чтобы caller payload не
     мутировался.
   - Добавлены size-based JSONL rotation и TTL cleanup для JSONL/summary files.
   - `OpenAIHelper.close()` предпочитает `session_logger.close()`, сохраняя fallback на
     legacy `drain()`.
   - Env/docs: `SESSION_LOG_MAX_BYTES`, `SESSION_LOG_RETENTION_DAYS`.
   - Scheduled summary flush coalesces one pending task per session, retrieves/logs task
     exceptions, no-op'ит empty summaries, and `close()`/`drain()` waits pending summary
     tasks. Если новые stats приходят во время активного flush, dirty/rerun path запускает
     следующий summary flush и не оставляет `_stats` в памяти.
   - Tests: `tests/test_session_logger.py`, `tests/test_session_logging_integration.py`,
     `tests/test_tool_handler_session_logging.py`, `tests/test_openai_helper_db_offload.py`,
     `tests/test_telegram_builder_config.py` под `-W error` — `67 passed`.
   - Reviewer loop: `NO FINDINGS / NO WARNINGS`.
5. `P2-5` PII-safe normal logging. **STATUS: DONE (2026-07-03).**
   - Обычные Python-логи для LLM request payloads, common args, assistant tool-call
     history, model tool calls, tool arguments/results, conversation reentry, direct
     results, transcript output и Telegram `file_id`/temp-path больше не печатают raw
     payload; вместо этого пишутся shape/count/length metadata.
   - `SessionLogger.record()` не редактировался: явные session JSONL events сохраняют
     full-fidelity payload для включённого trace.
   - Добавлен общий `log_value_shape()` / `log_json_shape()` / `log_exception_shape()`
     в `bot/utils.py`; нормальные exception-логи в touched runtime files переведены
     на shape-only без `logger.exception`/`exc_info`.
   - Regression tests покрывают happy path и failure paths: direct-result delivery,
     HTML-file read/remove, provider/TTS/session-memory errors, streaming APIError,
     transcribe failure, plugin-exchange mirror failure, busy-status/indicator/edit retry
     and Telegram error-handler paths.
   - Tests: `tests/test_pii_safe_logging.py`, `tests/test_openai_helper_tool_calls.py`,
     `tests/test_session_logging_integration.py`, `tests/test_tool_handler_session_logging.py`,
     `tests/test_plugin_direct_results.py`, `tests/test_utils_send_long_response_file.py`,
     `tests/test_telegram_transcribe.py`, `tests/test_telegram_streaming.py`,
     `tests/test_telegram_builder_config.py` под `-W error` — `211 passed`.
   - Static check: raw-log/exception-log паттерны `logger.exception`, `logging.exception`,
     `exc_info=`, raw `str(e)` logger patterns, raw `%s exc`, `payload=`, `arguments=`,
     `Function ... response:`, `Transcript output`, `artifact!r`, `artifact_payload!r`
     в touched runtime files не найдены; AST static guard ловит raw exception args after
     multiple logger args; `bot/plugins/skills.py` без diff.
   - Reviewer loop: `NO FINDINGS / NO WARNINGS` (`019f255f-38e8-7670-90ce-6ca4ae4d8ce9`).
6. Docker/compose: non-root user, explicit writable mounts, `env_file`, healthcheck after
   health endpoint decision. **STATUS: DONE (2026-07-03).**
   - `Dockerfile`: непривилегированный `bot` user (`10001:10001`), Docker-safe storage
     defaults, writable runtime dirs, legacy startup dirs для default-loaded plugins, и
     image `HEALTHCHECK` без добавления HTTP endpoint.
   - `docker-compose.yml`: удалён широкий `.:/app`, добавлен `env_file: .env`, named
     volumes для `/app/data`, `/app/output`, `/app/plots`, `/app/usage_logs`, `/app/log`,
     `/app/uploads`, tmpfs для `/app/bot/temp` и `/tmp`; Docker-safe `SKILLS_DIR`/
     `SKILLS_WORKDIR` перекрывают host-only `.env` paths.
   - `.dockerignore`, `.env.example`, README.md/README.ru.md и codebase-map nodes обновлены.
   - Tests: `tests/test_docker_runtime_config.py` под `-W error` — `3 passed`;
     `docker compose config --quiet` — passed; `bot/plugins/skills.py` без diff.
   - Reviewer loop: `NO FINDINGS / NO WARNINGS` (`019f2572-4eb9-7440-a462-ef62cac03514`).
7. `translations.json` chmod 644. **STATUS: DONE (2026-07-03).**
   - `stat -c '%a %n' translations.json` -> `644 translations.json`.
8. UI/HTML: CWD-independent output/data/plots roots, template extraction only after
   security fixes are green. **STATUS: DONE (2026-07-03).**
   - Добавлен `bot/runtime_paths.py`: repo-rooted defaults и env overrides
     `BOT_DATA_DIR`, `BOT_OUTPUT_DIR`, `BOT_PLOTS_DIR`; относительные env paths
     резолвятся от корня репозитория, не от CWD.
   - `HTMLVisualizer`, `send_long_response_as_file` и `codeinterpreter` используют
     runtime roots для HTML output, data copy/download, plots scan, cleanup, PlantUML
     и direct-result file path.
   - `bot/__main__.py`, Dockerfile/compose, `.env.example`, README.md/README.ru.md и
     codebase-map nodes обновлены под `BOT_*_DIR`.
   - Template extraction намеренно не смешан с root-fix: это отдельный DOM/CSS
     refactor, не требуемый для закрытия CWD риска.
   - Tests: `tests/test_runtime_paths.py`, `tests/test_html_utils_responsive.py`,
     `tests/test_codeinterpreter_plugin.py`, `tests/test_utils_send_long_response_file.py`,
     `tests/test_docker_runtime_config.py`, `tests/test_telegram_builder_config.py`
     под `-W error` — `67 passed`; `docker compose config --quiet` — passed;
     scoped literal scan — без filesystem path literals; `bot/plugins/skills.py` без diff.
   - Reviewer loop: `NO FINDINGS / NO WARNINGS` (`019f2583-8244-7b63-b568-38b0f4a565b1`).

## Параллелизация

- Можно параллелить: `P0-01`, `P0-02`, `P0-05`, `P0-06`, `P0-07`, `P0-10`.
- Нельзя параллелить без координации: `P0-03` с framework-args/stream tool-loop changes;
  все меняют `openai_tool_handler.py` tests.
- `P0-04` лучше делать отдельно от provider/model policy, чтобы retry/history regression
  был чистым.
- `P1-G` MCP и Hindsight P0 независимы; AgentTools clear/state лучше держать вместе.
- Docker/compose делать после CWD/writable-path cleanup, иначе non-root вскроет неготовые
  относительные записи.

## Per-task workflow

Для каждой задачи:

1. Subagent planner уже подготовил дорожку; при необходимости уточнить локально кодом.
2. Написать failing regression test.
3. Сделать минимальный patch.
4. Запустить targeted tests.
5. Запустить code-review subagent на сделанный diff этой задачи.
6. Исправить все ошибки и предупреждения review subagent.
7. Перейти к следующей задаче только после чистого targeted gate.

## Финальный gate

1. `.venv/bin/python -m pytest -q`
2. Targeted suites по всем изменённым зонам, если полный прогон падает на инфраструктурном
   pre-existing факторе, с явной фиксацией причины.
3. `git diff --check`
4. `rg` static checks по sync DB hot calls, hardcoded plugin refs, dangerous HTML injection
   patterns, если соответствующие зоны менялись.
5. Финальный reviewer-loop subagent: повторять review -> fix -> tests, пока нет ошибок и
   предупреждений.
