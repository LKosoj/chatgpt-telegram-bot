# План миграции плагинов на хуки

Документ-чеклист. Закрываем пункты по мере выполнения (`[ ]` → `[x]`).

## Контекст

Сейчас пять плагинов вшиты в ядро бота прямыми вызовами `plugin_manager.get_plugin('<имя>')`:
`conversation_analytics`, `reminders`, `skills`, `hindsight_memory`, `agent_tools`.
Часть из них также владеют таблицами в `bot/database.py` и логикой в `bot/openai_helper.py`.
Цель — перевести их на единый hook-контракт и убрать имена плагинов из ядра.

## Решения, принятые на ревью плана

- [x] Все хуки — `async def`. Гибридного диспетчера нет.
- [x] `Database` остаётся sync. `DbHandle` — async-фасад через `asyncio.to_thread` поверх текущего `Database`.
- [x] Перевод `Database` на `aiosqlite` — out of scope, отдельная задача после миграции (вариант (b) из обсуждения).
- [x] Политика сбоев blocking observer — (A) Skip & continue: исключение плагина логируется (`error`, с `plugin_id`/`event_name`/`exc_info`), плагин пропускается, остальные подписчики и операция ядра продолжаются. Единая политика для всех blocking-хуков, без параметра `strict`.

## Цели и инварианты

- [ ] Ни одного обращения к плагину по имени из `bot/telegram_bot.py`, `bot/openai_helper.py`, `bot/database.py`, кроме (а) generic `get_plugin(plugin_id)` для tool-роутинга и (б) UI-чтения публичных атрибутов плагина для меню настроек.
- [x] Сбой плагина в хуке не ломает ответ пользователю — обеспечивается dispatcher'ами в `PluginManager` (политика A для всех 4 типов).
- [x] Контракт хука — только `async def` (для observer/blocking/collector/mutator); декларативные методы (`register_schema`, `get_config_prefix`, `get_background_tasks`) sync.
- [x] Старые API `PluginManager.get_plugin / has_plugin / call_function` остаются работать (нужны для tool-вызовов и UI).
- [ ] После каждого этапа возможно откатить только этот этап, не каскадом.
- [x] Структурный лог при сбое в любом хуке: `plugin_hook_error` с полями `plugin_id, hook, event/slot, exc_class` (через `_log_hook_error`).

## Категории хуков

| Тип | Метод диспетчера | Семантика | Политика сбоя |
|---|---|---|---|
| Observer | `dispatch_observe` | `asyncio.gather(return_exceptions=True)`, fire-and-forget | Логируется, не блокирует |
| Blocking observer | `dispatch_blocking` | Последовательно, `await` существенен (happens-before) | (A) Skip & continue |
| Collector | `collect_fragments` | Сбор `list[str]` в алфавитном порядке по `plugin_id` | Логируется, фрагмент опускается |
| Mutator | `apply_mutators` | Последовательно, `value = await plugin.hook(value, payload)` | Логируется, возвращается последнее валидное значение |

Декларативные расширения `Plugin`:

- `register_schema() -> list[str]` — DDL-фрагменты с `IF NOT EXISTS`.
- `get_config_prefix() -> str | None` — префикс ключей конфига для плагина.
- `get_background_tasks() -> list[BackgroundTask]` — фоновые задачи с интервалами.

---

## Открытые вопросы до этапа 0

Закрыть до старта работ.

- [x] **FK у плагин-таблиц** — проверено: `agent_plan_contracts`, `agent_plan_tasks`, `hindsight_finalize_jobs` **без FK**. Schema-registry не зависит от порядка DDL. Очистка по `user_id` для `hindsight_finalize_jobs` — ручная через хук (см. этап 4C).
- [x] **Точки mutator'а в `OpenAIHelper`** — проверено: всего **3** точки (не 4):
  - `:896` в `__common_get_chat_response` (обслуживает и `get_chat_response`, и `get_chat_response_stream`).
  - `:1083` (`_retry_empty_response_after_tools`).
  - `:1129` (`_retry_empty_response_with_tools`).
  - `:847` (`_prepare_hindsight_session_memory_context`) — это **подготовка** (recall network call), не точка отправки. Logic переезжает **внутрь** mutator'а: плагин сам решает, делать ли recall.
- [x] **Политика сбоя клиента Hindsight в `on_before_chat_request`** — покрывается общим контрактом mutator'ов: плагин ловит HTTP-исключение внутри хука, возвращает unmodified `messages`, логирует. Отдельная политика не нужна.
- [x] **Per-user disabled в `PluginManager`** — проверено: сейчас живёт в `ChatGPTTelegramBot._disabled_plugins_for_user` (`telegram_bot.py:1385`) и дублируется в `OpenAIHelper._disabled_plugins_for_user` (`openai_helper.py:719`). Источник — таблица `user_settings`, JSON-поле `disabled_plugins`. Решение: переезд логики в `PluginManager` (с `set_db()` по аналогии с `set_openai()`) — выделено в первую задачу этапа 0 (раздел «Pre-задача»).
- [x] **Судьба `tool_call_events`** — out of scope этого плана (зафиксировано в разделе Out of scope).

---

## Этап 0 — Инфраструктура хуков ✅

~600-900 строк, средняя сложность, низкий риск. Никаких изменений поведения — только фреймворк.

### Pre-задача — переезд `_disabled_plugins_for_user` в `PluginManager` ✅

Выполнено. Чистка, упрощающая дизайн всех хуков.

- [x] В `PluginManager` добавлен `set_db(db)` по аналогии с `set_openai()`. Идемпотентен (документировано в docstring).
- [x] В `PluginManager` добавлен публичный `disabled_plugins_for_user(user_id: int | None) -> set[str]` (без подчёркивания — теперь часть публичного контракта).
- [x] В `PluginManager` добавлен публичный `is_plugin_disabled_for_user(plugin_name, user_id) -> bool`.
- [x] В `bot/__main__.py` после создания `Database` — вызов `plugin_manager.set_db(db)`.
- [x] В `bot/telegram_bot.py:89` — повторный `set_db(self.db)` для тестов, конструирующих бота напрямую.
- [x] `ChatGPTTelegramBot._disabled_plugins_for_user` и `_is_plugin_disabled_for_user` **удалены полностью**.
- [x] `OpenAIHelper._disabled_plugins_for_user` **удалён полностью**. Делегация: `self.plugin_manager.disabled_plugins_for_user(user_id)`.
- [x] Обновлены все 9 call-сайтов: 8 в `telegram_bot.py` (`:682, :1405, :2879, :3373, :3395, :3528, :3604, :3634`) + 1 в `openai_helper.py:718`.
- [x] Удалены неиспользуемые импорты `USER_DISABLED_PLUGINS_SETTING`, `normalize_string_list` из `openai_helper.py`.
- [x] Добавлены 5 новых тестов в `tests/test_plugin_manager.py` (edge cases: db=None, user_id=None, нормализация мусора, неверный shape настроек).
- [x] Обновлены 7 тест-фейков: `DummyPluginManager` в `test_openai_helper_tool_calls.py` и `test_hindsight_memory.py`, `FakePluginManager` в `test_telegram_streaming.py`, `test_per_conversation_serialization.py`, `test_plugin_handlers_registration.py`, `test_telegram_builder_config.py`, `test_plugin_menu_force_reply.py`, `RacingPluginManager` в `test_concurrent_tool_state.py` — все получили `set_db`, `disabled_plugins_for_user`, `is_plugin_disabled_for_user`.
- [x] Production-нормализация в фейках через делегацию `bot.user_settings.{get_user_settings, normalize_string_list}` (нет дубликата логики).
- [x] **Тесты: 371 passed, 0 failed** (полный прогон `tests/` за исключением `test_text_document_qa_anythingllm.py`, требующего сетевого доступа).
- [x] `grep "_is_plugin_disabled_for_user\|_disabled_plugins_for_user" bot/` → **0** совпадений.
- [x] `grep "plugin_manager.is_plugin_disabled_for_user\|plugin_manager.disabled_plugins_for_user" bot/` → **9** совпадений.
- [x] Code-review от независимого сабагента: APPROVE после применения 5 should-fix (идемпотентность `set_db`, делегация нормализации в фейках, стабы в 3 stale-фейках, docstring, формулировка комментария).

### Новые модули

- [x] `bot/plugins/hooks.py` — `HookEvent` StrEnum + 6 payload-датаклассов (`@dataclass(frozen=True, slots=True)`).
- [x] `bot/plugins/background.py` — `BackgroundTask(name, interval_seconds, coroutine_factory)`.
- [x] `bot/plugins/db_handle.py` — узкий async-фасад к `Database` через `asyncio.to_thread`; `transaction()` — буферизованный батч.

### Расширение `Plugin` (bot/plugins/plugin.py)

Опциональные `async def`-методы (no-op по умолчанию):

- [x] `async on_user_message(payload)`
- [x] `async on_assistant_response(payload)`
- [x] `async on_session_reset(payload)`
- [x] `async on_session_before_delete(payload)`
- [x] `async on_before_chat_request(messages, payload) -> list[dict]`
- [x] `async contribute_prompt_fragment(slot, payload) -> str | None`
- [x] `get_background_tasks() -> list[BackgroundTask]`
- [x] `register_schema() -> list[str]`
- [x] `get_config_prefix() -> str | None`
- [x] `initialize` не расширен в базе — shim `PluginManager._call_initialize` фильтрует kwargs по `inspect.signature(plugin.initialize)`. Совместимость с 11 существующими плагинами без модификации; современные плагины опт-инят `db=None, plugin_config=None` в своей сигнатуре.

### Payload-датаклассы

- [x] `UserMessagePayload(chat_id, user_id, request_id, text, has_image, has_voice, is_command, ts)`
- [x] `AssistantResponsePayload(chat_id, user_id, request_id, text, tokens, model, ts)`
- [x] `SessionResetPayload(chat_id, user_id, reason, terminal_only)`
- [x] `SessionBeforeDeletePayload(user_id, session_id, messages: tuple)` — `tuple`, не `list` (frozen-compatible).
- [x] `BeforeChatRequestPayload(chat_id, user_id, request_id)` (массив `messages` — отдельный аргумент мутатора)
- [x] `PromptFragmentPayload(slot, chat_id, user_id, query)`

### Диспетчер в `PluginManager`

- [x] `await dispatch_observe(event_name, payload, *, user_id=None)` — `asyncio.gather` с `return_exceptions=True`.
- [x] `await dispatch_blocking(event_name, payload, *, user_id=None)` — последовательно, плагин со сбоем пропускается (политика A).
- [x] `await collect_fragments(slot, payload, *, user_id=None) -> list[str]` — алфавитный порядок по `plugin_id`.
- [x] `await apply_mutators(event_name, payload, value, *, user_id=None)` — последовательно, на исключении возвращаем последнее валидное значение; `None` от плагина = «без изменений».
- [x] `start_background_tasks(application)` / `await stop_background_tasks(timeout)`.
- [x] Все 4 dispatcher'а уважают per-user disabled через `_active_plugin_instances(user_id)`. Покрыто отдельным тестом для каждого типа.

### DbHandle

Узкий контракт:

- [x] `await execute(sql, params=())`
- [x] `await executemany(sql, params_seq)`
- [x] `await fetch_one(sql, params=()) -> dict | None`
- [x] `await fetch_all(sql, params=()) -> list[dict]`
- [x] `transaction()` — async context manager (буферизованный батч: операции копятся в `TransactionScope`, исполняются единым `to_thread` на успешном `__aexit__`; при исключении буфер отбрасывается без `BEGIN/ROLLBACK`; reads внутри scope не поддерживаются).
- [x] Реализация через `asyncio.to_thread` поверх `Database.get_connection()`. Сериализация через существующий `_op_lock`.

### Schema-registry

- [x] Точка вызова — отдельный метод `register_plugin_schemas()` в `PluginManager`, выполняется из `bot/__main__.py:199` **после** `set_db(db)` и **до** `set_openai(openai)`. Создаёт инстансы плагинов **без** `initialize` (`_get_or_create_bare_instance`), чтобы `set_openai` затем вызвал `initialize` ровно один раз для каждого плагина. На этапе 0 registry пустой — DDL не выполняется, но точка есть. Покрыто 2 тестами: `test_register_plugin_schemas_does_not_invoke_initialize`, `test_register_plugin_schemas_then_set_openai_initializes_once`.

### Config-segments

- [x] `PluginManager._plugin_config_segment(plugin)` — фильтрация `self.config` по `plugin.get_config_prefix()`. Ключи отдаются **с префиксом** (минимизация diff для будущей миграции hindsight). Передаётся в `_call_initialize(plugin_config=...)` — плагин нарезает себе сам. По умолчанию `get_config_prefix() = None` → `{}`.

### Что НЕ делаем на этапе 0

- Ни один плагин не мигрирует.
- Ни один метод из `Database` не удаляется.
- Ни одна точка `get_plugin('name')` в ядре не убирается.

### Acceptance

- [x] Существующие тесты зелёные: **399 passed** (`PYTHONPATH=. python3 -m pytest tests/ --ignore=tests/test_text_document_qa_anythingllm.py`).
- [x] `tests/test_plugin_hooks.py` — **20 тестов**: каждый dispatcher работает; ошибка плагина изолируется (политика A для blocking — `test_dispatch_blocking_first_raises_second_runs` с явным `pytest.fail`-guard'ом); collector стабильно сортирует по `plugin_id`; mutator при `None`/исключении сохраняет последнее валидное значение; background task ретраится после исключения; per-user disabled покрыт для всех 4 типов dispatcher'ов; `_call_initialize` фильтрует kwargs; `_plugin_config_segment` сохраняет префикс.
- [x] `tests/test_db_handle.py` — **8 тестов**: execute/fetch/executemany/transaction commit+discard; параллельный execute через gather не корраптится; fetch возвращает plain `dict`, не `sqlite3.Row`.
- [x] Импорты модулей `bot.plugin_manager`, `bot.plugins.{hooks,background,db_handle}` резолвятся без ошибок; запуск бота не ломается (smoke import).

### Code-review summary

Code-review: **APPROVE** после применения 6 should-fix:
1. `register_plugin_schemas` перенесён ДО `set_openai` в `bot/__main__.py`.
2. Per-user disabled тесты добавлены для `dispatch_blocking`, `collect_fragments`, `apply_mutators`.
3. Тест политики A в `dispatch_blocking` обёрнут в `try/except → pytest.fail` для явности.
4. Background-task retry тест переписан на `asyncio.Event` (вместо timing-based sleep).
5. Docstring'и `transaction()` / `TransactionScope` приведены в соответствие реальной семантике (буферизация, не настоящий ROLLBACK).
6. Комментарий о редкости last-resort fallback в `_call_initialize` добавлен.

Дополнительно после code-review обнаружена и исправлена двойная инициализация (`register_plugin_schemas` → `get_plugin` → `_call_initialize(openai=None)`, потом `set_openai` → `_call_initialize` второй раз): добавлен `_get_or_create_bare_instance`, который создаёт инстанс без `initialize`. Покрыто 2 новыми тестами.

---

## Этап 1 — `conversation_analytics` → observer ✅

~200-300 строк, низкий риск.

### Изменения

- [x] Точка диспатча `on_assistant_response` в `bot/telegram_bot.py` после `get_chat_response`: в обеих ветках (`direct_result` и обычная). Заменяет оба `get_plugin('conversation_analytics')` вызова. `user_id=user_id` — теперь честно уважает per-user disabled (фикс латентного бага старого кода).
- [x] Точка диспатча `on_user_message` после `_try_handle_plugin_prompt` и до `try:` блока с `get_chat_response`. Поля `has_image/has_voice/is_command` есть в payload; `has_image/has_voice=False` пока (текстовый путь), `is_command` — `update.message.text.startswith("/")` с None-guard'ом.
- [x] `ConversationAnalyticsPlugin`: `async on_user_message` и `async on_assistant_response` через `asyncio.to_thread(self.update_stats, ...)` (`update_stats` — sync, файловая I/O). Сам `update_stats` не тронут.
- [x] Удалены оба обращения `get_plugin('conversation_analytics')` из `bot/telegram_bot.py` (`grep` → 0).

### Bonus

- [ ] **Отложено.** Баг `chat_id=hash(...)` (`bot/plugins/conversation_analytics.py:229, 249, 267, 285`) требует новой stateless-API на `OpenAIHelper`, что выходит за scope этапа 1. Открыть отдельный тикет после этапа 6.

### Что НЕ делаем

- БД не трогаем (analytics пишет в JSON-файл).
- topics/sentiment/cost-аналитика — feature work, отдельно.
- `text=analytics_prompt` в `AssistantResponsePayload` сохранён 1:1 со старым кодом (старый код хранил prompt пользователя в поле, которое логически должно быть assistant response text). Не фиксим в parity-миграции; follow-up.
- Voice/vision-пути не диспатчат `on_user_message` — паритет со старым кодом, который тоже не учитывал их в analytics.

### Acceptance

- [x] `grep "get_plugin\(['\"]conversation_analytics['\"]\)" bot/` → 0 совпадений.
- [x] `tests/test_plugin_chat_id_contract.py:189-211` обновлён: `await plugin.on_assistant_response(AssistantResponsePayload(...))` + явная ассерция str-coercion (`"1234" in plugin.conversation_stats`).
- [x] `tests/test_conversation_analytics_hooks.py` (новый, 107 строк, 5 тестов): user_message persist, assistant_response persist, dispatcher routing, **регресс на изоляцию исключений**, is_command propagation.
- [x] Регресс-тест: `update_stats` бросает `RuntimeError` → `dispatch_observe` не пробрасывает → `plugin_hook_error` логируется со структурными полями `plugin_id=conversation_analytics, event=on_assistant_response`.
- [x] **Полный прогон: 404 passed, 0 failed** (`PYTHONPATH=. python3 -m pytest tests/ --ignore=tests/test_text_document_qa_anythingllm.py`).
- [x] Стабы `dispatch_observe` добавлены в `FakePluginManager` 2 тест-файлов (`test_per_conversation_serialization.py`, `test_telegram_streaming.py`) — без них 8 streaming-тестов падали.
- [x] Code-review: APPROVE после применения 4 should-fix (унификация `import time`, явная str-coercion ассерция, комментарий про lazy-init в `_make_pm`, структурные поля в caplog-проверке).

---

## Этап 2 — `reminders` → `get_background_tasks()` ✅

~150-250 строк, низкий риск.

### Изменения

- [x] `RemindersPlugin.get_background_tasks()` возвращает 1 `BackgroundTask(name="check", interval_seconds=60.0, coroutine_factory=self._check_reminders_tick)`. `check_reminders`/`send_reminder`/`execute` — не тронуты.
- [x] Wrapper `async def _check_reminders_tick(self, *, application)` маппит framework-семантику (`application=` kwarg) на plugin-семантику (`application.bot` как `helper`). `application` НЕ кэшируется на `self`.
- [x] В `ChatGPTTelegramBot.post_init` ручной `asyncio.create_task(self.start_reminder_checker(...))` заменён на `await self.openai.plugin_manager.start_background_tasks(application)` внутри `if not self._background_tasks:` guard.
- [x] В `cleanup()` (`bot/telegram_bot.py:3836`) добавлен `await self.openai.plugin_manager.stop_background_tasks(timeout=10.0)` ДО блока ручной отмены задач, обёрнут `try/except Exception` — buggy plugin не аварит shutdown.
- [x] `start_reminder_checker` удалён (был на `:3907`, не `:3891` — план был устаревший).

### Что НЕ делаем

- `buffer_data_checker` и `hindsight_finalize_worker` остаются ручными — buffer это инфраструктура бота, не плагин-концерн; hindsight worker мигрирует в этапе 4C.
- `check_reminders` — параметр всё ещё называется `helper: Any`, хотя это `telegram.Bot`. Не переименовываем — это потребует касаний за пределами scope этапа.

### Acceptance

- [x] `rg "start_reminder_checker" bot/ tests/` → 0 совпадений.
- [x] `rg "asyncio.create_task.*reminder" bot/` → 0 совпадений.
- [x] Существующие reminder-тесты в `tests/test_concurrent_tool_state.py` (5 тестов) зелёные без изменений.
- [x] `tests/test_background_tasks.py` (новый, 4 теста): `get_background_tasks()` shape; tick с past-due reminder отправляет message; tick с future reminder не отправляет; lifecycle через `PluginManager` (register/stop в timeout 2s).
- [x] Поведенческая эквивалентность: framework `_background_task_loop` тикает → sleep, как старый `start_reminder_checker`. Первый тик при старте, тик каждые 60s после.
- [x] Per-user disable не применяется к bot-wide tasks (`start_background_tasks` итерирует `self.plugins` напрямую, не через `_active_plugin_instances`).
- [x] **Полный прогон: 408 passed, 0 failed** (`PYTHONPATH=. python3 -m pytest tests/ --ignore=tests/test_text_document_qa_anythingllm.py`).
- [x] Code-review: APPROVE после применения 3 should-fix (снят stage-tag из комментария, тест #4 переименован в `test_reminders_task_registers_and_stops_within_timeout` + sanity ассерция `get_plugin("reminders") is plugin`).

---

## Этап 3 — `agent_tools`: observer + первое применение schema-registry ✅

~400-600 строк, средняя сложность, средний риск.

### Часть A — событие `on_session_reset`

- [x] Вызов `_clear_agent_plan` в ядре заменён на `await self.openai.plugin_manager.dispatch_observe("on_session_reset", SessionResetPayload(..., terminal_only=False), user_id=...)` в 4 местах: `_handle_direct_result` (`telegram_bot.py:419`), voice/transcribe (`:2120`), text prompt (`:2653`), inline-query (`:3035`).
- [x] `AgentToolsPlugin.on_session_reset(payload)` — async; вычисляет scope через `compute_scope_key`; диспатчит на `_db_clear_terminal_tasks_async` или `_db_clear_all_tasks_async` в зависимости от `payload.terminal_only`. Defense-in-depth `try/except` + structured `logger.info`.
- [x] Удалены из `bot/telegram_bot.py`:
  - [x] `_get_agent_tools_plugin` — inline в `_build_plan_status_provider` (legitimate UI-чтение через generic `get_plugin("agent_tools")`).
  - [x] `_clear_agent_plan` — полностью удалён.
  - [x] Все обращения к `clear_plan_tasks` / `clear_terminal_plan_tasks` из ядра — `grep` → 0.

`cleanup_after_delivery` оставлен как есть — уже generic-механизм через `plugin_id` в директиве tool-результата.

**Поведенческое изменение**: после миграции пользователь с `agent_tools` в `disabled_plugins` не получит cleanup на session_reset (политика `_active_plugin_instances` фильтрует disabled). Старый код вызывал `_clear_agent_plan` безусловно. Считаем корректным новым поведением: disabled = inert.

### Часть B — миграция схемы

Из `bot/database.py` переехало в плагин (`bot/plugins/agent_tools.py`):

- [x] `CREATE TABLE agent_plan_contracts` → `register_schema()` (DDL byte-parity с оригиналом).
- [x] `CREATE TABLE agent_plan_tasks` → `register_schema()`.
- [x] `CREATE INDEX idx_agent_plan_tasks_scope_position` → `register_schema()`.
- [x] `save_agent_plan` (`database.py:332`) → `_db_save_plan` (sync, через `self.db.get_connection()`).
- [x] `get_agent_plan` (`:370`) → `_db_get_plan` (sync).
- [x] `clear_agent_plan` (`:413`) → `_db_clear_plan` (sync).
- [x] `prune_agent_plans` (`:425`) → `_db_prune_plans` (sync). **Doc drift**: метод реально называется `prune_agent_plans`, не `cleanup_old_agent_plans`.
- [x] Новые async-helper'ы для observer-хука: `_db_clear_terminal_tasks_async`, `_db_clear_all_tasks_async` (через `self.db_handle`).

**Dual surface**: sync DAO остаётся для существующих sync call-сайтов (`_build_plan_status_provider` через `BusyStatusMessage` — sync-контракт). Async DAO через `DbHandle` — только для нового `on_session_reset` хука. Перевод всего на async — out of scope (потребовал бы рефакторинг `BusyStatusMessage`).

**`transaction()` НЕ используется** — каждая операция либо single-statement, либо read-then-write (buffered txn не поддерживает чтения). Race window async-helper'ов идентичен sync-параметрам (тот же `_op_lock`, тот же raw two-call паттерн).

### Часть C — обновление тестов

- [x] `tests/test_database.py` — удалён obsolete `test_agent_plan_persists_tasks_contract_and_dependencies` (-36 строк).
- [x] `tests/test_agent_tools_plugin.py` — `agent_db` фикстура расширена: после создания `Database` прогоняет `AgentToolsPlugin().register_schema()` DDL'и. `_db_backed_agent_plugin` теперь передаёт `db=DbHandle(db)` в `initialize`.
- [x] `tests/test_agent_tools_schema_registry.py` (новый, 2 теста): `register_plugin_schemas()` создаёт таблицы и индекс на drop'нутой схеме; `register_schema()` идемпотентен на bare-инстансе без `initialize` (нет state mutation).
- [x] `tests/test_agent_tools_session_reset.py` (новый, 4 теста): full clear, terminal_only сохраняет open tasks, terminal_only чистит когда все closed, end-to-end через `pm.dispatch_observe`.
- [x] `tests/test_telegram_streaming.py` — `FakePluginManager.dispatch_observe` расширен: маршрутизирует `on_session_reset` к `agent_tools.clear_plan_tasks`/`clear_terminal_plan_tasks`. Существующие 3 streaming-теста асёртят `clear_calls`/`terminal_clear_calls` — работают без правок тел тестов.

### Acceptance

- [x] `rg "agent_plan_" bot/database.py` → 0 совпадений.
- [x] `rg "_get_agent_tools_plugin|_clear_agent_plan" bot/telegram_bot.py` → 0 совпадений.
- [x] `rg "self.db.(save|get|clear|prune)_agent_plan" bot/` → 0 совпадений.
- [x] Плановые операции работают: 38 тестов в `test_agent_tools_plugin.py` зелёные после миграции на новые DAO.
- [x] Сессия удалена → план плагина почищен: 4 теста в `test_agent_tools_session_reset.py` покрывают full clear, terminal_only, end-to-end routing.
- [x] **Полный прогон: 413 passed, 0 failed** (`PYTHONPATH=. python3 -m pytest tests/ --ignore=tests/test_text_document_qa_anythingllm.py`).
- [x] Code-review: APPROVE без must-fix. Применены 2 nice-to-have полировки: `_db_save_plan(contract: Optional[Dict[str, Any]])` (точный тип вместо `Any`), импорт `Optional` в typing.

---

## Этап 4 — `hindsight_memory`: полный переезд

~1000-1500 строк, высокая сложность, **высокий риск**. Разбит на 3 подкоммита; состояние рабочее после каждого.

### Подэтап 4A — клиент и конфиг ✅

- [x] `from .hindsight_client import HindsightClient` удалён из `bot/openai_helper.py:55`.
- [x] Конструктор клиента (`openai_helper.py:272-279`) уехал в `HindsightMemoryPlugin.initialize`.
- [x] `HindsightMemoryPlugin.get_config_prefix() = 'hindsight_'`.
- [x] 13 ключей `hindsight_*` (`openai_helper.py:256-271`; 12 настраиваемых + вычисляемый `hindsight_enabled`) переданы плагину через config-segment.
- [x] `setdefault`-блок (`:256-271`) удалён из `OpenAIHelper`. Дефолты ставит плагин в `initialize` **до** их чтения.
- [x] `self.hindsight_client` (`:272`) удалён как атрибут; превращён в `@property`-делегатор к `plugin.client` (Стратегия A — shim, удаляется в 4B/4C).
- [x] `is_hindsight_enabled` (`:1565`), `get_hindsight_bank_id` (`:1573`), `_hindsight_memory_types` (`:1576`) превращены в shim-делегаторы к плагину (тонкие, удаляются в 4B/4C вместе с потребителями `_prepare_hindsight_session_memory_context`/`finalize_hindsight_session_memory`).
- [x] Обращения внутри плагина (`bot/plugins/hindsight_memory.py:88, 95, 96, 102-104, 132, 167, 201, 216, 229, 231, 244, 246, 259, 260, 264, 272`) переключены на `self.client` / `self.is_active` / `self.bank_id_for(user_id)` / `self.memory_types`.
- [x] 5 потребителей в `bot/telegram_bot.py` (`:454`, `:489`, `:850`, `:1396`, `:3334`) переведены на чтение через плагин.
- [x] `bot/__main__.py`: `plugin_manager.config.update(openai_config)` после конструирования PluginManager — необходимое предусловие для config-segment.
- [x] Новый `tests/test_hindsight_plugin_init.py` (6 тестов): дефолты, overrides, conditional client, mirror в openai.config, bank_id_for, memory_types.
- [x] Адаптированы `tests/test_hindsight_memory.py`, `tests/test_hindsight_memory_plugin.py`, `tests/test_group_session_flow.py`, `tests/test_telegram_builder_config.py` под property `hindsight_client` и новый shape `DummyPluginManager.get_plugin('hindsight_memory')`.
- [x] Pytest: **419 passed, 0 failed** (база 413 + 6 новых).
- [x] Code-review: APPROVE, must-fix нет. Применены 2 nice-to-have: переименование теста `_14_keys` → `_13_keys` (фактическое число) и `setdefault` вместо unconditional assignment в mirror-блоке.
- [x] **Стратегия**: A — shim-делегаторы в helper'е. Альтернативы B (inline в helper'е) и C (helper читает плагин напрямую через `get_plugin('hindsight_memory')` в каждом call-сайте) отвергнуты ради «состояние рабочее после каждого подкоммита» и минимальной coupling.
- [x] **Backwards-compat mirror**: плагин в `initialize` зеркалирует свои 13 ключей в `openai.config` через `setdefault` — для совместимости с `_prepare_hindsight_session_memory_context` и `finalize_hindsight_session_memory`, которые остаются в helper'е до 4B/4C. Mirror снимается в 4C.

### Подэтап 4B — mutator-хук ✅

- [x] `_prepare_hindsight_session_memory_context` (`openai_helper.py:1584-1625`) полностью переехал в `HindsightMemoryPlugin.on_before_chat_request`. Recall (HTTP к Hindsight) делается внутри mutator'а: плагин получил `messages`, проверил маркер в messages (кэш), сделал recall, инжектнул memory-message, вернул новый список.
- [x] `_messages_with_hindsight_context` удалён из helper'а; его роль (`repair + language + recall + persist`) теперь у общего метода `OpenAIHelper._apply_before_chat_request_mutators`.
- [x] **3 точки вызова** в helper'е заменены на `messages = await self._apply_before_chat_request_mutators(...)`:
  - [x] `__common_get_chat_response` (`bot/openai_helper.py:850`) — `persist=True`.
  - [x] `_retry_empty_response_after_tools` (`:1049`) — `persist=False`.
  - [x] `_retry_empty_response_with_tools` (`:1102`) — `persist=False`.
- [x] **Bonus**: 2 дополнительные точки в `bot/openai_tool_handler.py` (`:310`, `:574`, retry в tool-call path) тоже переключены на `_apply_before_chat_request_mutators(..., persist=False)`. План их не упоминал, но без миграции рантайм сломался бы после удаления `_messages_with_hindsight_context`.
- [x] Отдельный вызов `_prepare_*` (`:821`) удалён, логика внутри mutator'а.
- [x] Кэш «уже сделали recall» — маркер в `messages` (через `is_hindsight_memory_message`). Плагин stateless по chat_id; helper persist'ит маркер в `self.conversations[chat_id]` через persist-ветку `_apply_before_chat_request_mutators` (Стратегия Z).
- [x] Удалены из `OpenAIHelper`: `_insert_hindsight_memory_context`, `_messages_with_hindsight_context`, `_has_hindsight_memory_context`, `_prepare_hindsight_session_memory_context`, константы `HINDSIGHT_MEMORY_MARKER`, `HINDSIGHT_CONTEXT_PROMPT`.
- [x] Константы и хелпер `is_hindsight_memory_message` переехали в `bot/plugins/hindsight_memory.py` (вместо импорта маркера helper'ом — публичный метод плагина).
- [x] Тест на изоляцию: `tests/test_hindsight_mutator.py::test_apply_mutators_isolates_plugin_exception` — при исключении в `on_before_chat_request` диспетчер возвращает оригинальный value, цепочка продолжается.
- [x] Тест на HTTP-сбой: `tests/test_hindsight_mutator.py::test_mutator_returns_none_on_http_failure` — плагин ловит `client.recall` exception, возвращает `None`, логирует warning.
- [x] Дополнительные тесты mutator'а: успешный recall+inject, кэш-hit, inactive, auto_recall=False, no user message, empty recall. Всего **8 новых тестов** в `tests/test_hindsight_mutator.py`.
- [x] Адаптированы: `tests/test_openai_helper_tool_calls.py` (тест `_messages_with_hindsight_context_repairs_*` переименован и переключён на `_repair_tool_call_history` + `_messages_with_language_instruction`; добавлена identity-заглушка `apply_mutators` в DummyPluginManager), `tests/test_hindsight_memory.py` (`DummyPluginManager.apply_mutators` делегирует на лениво-кэшированный плагин), `tests/test_plugin_chat_id_contract.py` (заглушка `_apply_before_chat_request_mutators` вместо `_messages_with_hindsight_context`).
- [x] Pytest: **427 passed, 0 failed** (база 419 + 8 новых). Контрольный `test_recalled_hindsight_memory_is_persisted_once_per_session` зелёный — Стратегия Z сохраняет инвариант «1 recall на сессию».
- [x] Code-review: APPROVE, must-fix нет. Применён 1 nice-to-have: позиция вставки маркера теперь «после **всех** leading system-сообщений», а не только после первого — устраняет тонкое поведенческое отличие на первом turn'е (раньше outbound был `[language, mode_system, MARKER, user...]`, в первой версии 4B стал `[language, MARKER, mode_system, user...]`, сейчас снова `[language, mode_system, MARKER, user...]`).
- [x] **Стратегия Z**: helper держит state (`self.conversations`, db.save), плагин stateless по chat_id. Альтернатива X (плагин пишет в `openai.conversations`) — анти-паттерн обратной связи. Альтернатива Y (полный stateless, recall каждый раз) — поведенческая регрессия в `_retry_*` (recall дублируется), потому что persisted маркер не сохранялся бы между запросами. Z — компромисс: чистый stateless-плагин, единый владелец state.

### Подэтап 4C — schema, jobs, worker

- [ ] `HindsightMemoryPlugin.register_schema()` возвращает DDL `hindsight_finalize_jobs` (`database.py:223-241`).
- [ ] Удалены из `Database`:
  - [ ] `CREATE TABLE hindsight_finalize_jobs` и индекс (`:223-241`)
  - [ ] `create_hindsight_finalize_job` (`:849`)
  - [ ] `create_hindsight_finalize_jobs_for_sessions` (`:863`)
  - [ ] `claim_hindsight_finalize_jobs` (`:894`)
  - [ ] `mark_hindsight_finalize_job_done` (`:956`)
  - [ ] `mark_hindsight_finalize_job_failed` (`:974`)
  - [ ] Ветка `hindsight_requires_async_finalize` в `delete_*_sessions` (`:1074-1081`).
  - [ ] Очистка `hindsight_finalize_jobs` в `delete_user_data` (`:672`) — переехала в плагин (нужно решить через какой механизм, см. open question).
- [ ] Ядро диспатчит `on_session_before_delete` (заменяет `_enqueue_*`):
  - [ ] `telegram_bot.py:4026` (точечное удаление).
  - [ ] `telegram_bot.py:1776, :3985` (batch по лимиту).
- [ ] Удалены из `telegram_bot.py`:
  - [ ] `_enqueue_hindsight_session_finalize_before_delete` (`:460`)
  - [ ] `_enqueue_hindsight_and_delete_oldest_sessions_for_limit` (`:491`)
  - [ ] `_process_pending_hindsight_finalize_jobs` (`:497`)
  - [ ] `hindsight_finalize_worker` (`:539`)
  - [ ] Старт воркера (`:3305`).
- [ ] `HindsightMemoryPlugin.get_background_tasks()` возвращает воркер (логика бывшего `_process_pending_hindsight_finalize_jobs`).
- [ ] `HindsightMemoryPlugin.on_session_before_delete` создаёт job через `self.db_handle.execute(...)`.
- [ ] `OpenAIHelper.finalize_hindsight_session_memory` переехал в плагин.

### Что НЕ делаем в этапе 4

- HTTP-контракт с внешним Hindsight-сервисом не меняем.
- `bank_id`, `namespace`, формат recall-результатов не трогаем.
- `on_session_before_delete` и `on_session_reset` не объединяем — разная семантика.

### Acceptance этапа 4

- [ ] `grep -i hindsight bot/openai_helper.py` пусто.
- [ ] `grep -i hindsight bot/telegram_bot.py` пусто (кроме UI-чтения статуса плагина, если решено оставить generic `get_plugin('hindsight_memory')`).
- [ ] `grep -i hindsight bot/database.py` пусто.
- [ ] `hindsight_*` имена живут только в `bot/plugins/hindsight_memory.py` и `bot/hindsight_client.py`.
- [ ] `tests/test_hindsight_mutator.py` (новый): mutator работает; сбой → unmodified messages.
- [ ] `tests/test_hindsight_session_lifecycle.py` (новый): `on_session_before_delete` доходит до плагина, job создан; при сбое плагина сессия всё равно удалена (политика A).
- [ ] `tests/test_hindsight_jobs.py` (новый): claim/process/mark_done/mark_failed через плагин.
- [ ] `tests/test_database.py` — ассертации про hindsight-таблицы и hindsight-методы удалены.
- [ ] Ручная проверка с реальным сервисом включён/выключен: оба сценария работают как раньше.

---

## Этап 5 — `skills` → collector

~200-300 строк, низкий риск.

### Изменения

- [ ] `_get_available_skills_summary` (`openai_helper.py:2441`) удалён.
- [ ] Блок «ПРИОРИТЕТНОЕ ПРАВИЛО» в `_build_auto_chat_mode_prompt` (`:2455+`) формируется плагином через `SkillsPlugin.contribute_prompt_fragment(slot='auto_mode_priority', payload)`.
- [ ] `_build_auto_chat_mode_prompt` вызывает `fragments = await plugin_manager.collect_fragments('auto_mode_priority', payload, user_id=user_id)` и склеивает с разделителем.
- [ ] `telegram_bot.py:1397` (`_available_skill_names`) **оставлен как есть** — UI-чтение через generic `get_plugin('skills')`, легитимное использование.

### Acceptance

- [ ] В `openai_helper.py` нет `get_plugin("skills")` и нет упоминания `available_skills`.
- [ ] Auto-mode prompt корректно рекомендует `skills_agent`, когда соответствующий skill установлен (smoke).
- [ ] `tests/test_skills_prompt_fragment.py` (новый): плагин возвращает фрагмент → промпт содержит; нет плагина → бот не падает; несколько фрагментов от разных плагинов — детерминированный порядок.

---

## Этап 6 — Финальная чистка

~100-200 строк, в основном тесты и документация.

- [ ] Линтер-тест `tests/test_no_hardcoded_plugin_refs.py`: grep по `bot/telegram_bot.py`, `bot/openai_helper.py`, `bot/database.py` на имена плагинов (`conversation_analytics`, `reminders`, `skills`, `hindsight_memory`, `agent_tools`). Падает при возврате хардкода. Исключения — white-list (generic `get_plugin(plugin_id)`, UI-меню настроек).
- [ ] `AGENTS.md` обновлён:
  - [ ] Раздел «Hooks: contract and lifecycle» (4 типа хуков, payload'ы, политика сбоев).
  - [ ] Раздел «Plugin-owned tables» (`register_schema()`).
  - [ ] Раздел «Plugin config segments».
  - [ ] Убраны упоминания вшитых вызовов.
- [ ] Решена судьба `tool_call_events` (отдельный заход, не часть миграции).

---

## Карта тестов по этапам

| Этап | Новое | Обновляемое |
|---|---|---|
| 0 | `test_plugin_hooks.py`, `test_db_handle.py` | — |
| 1 | `test_conversation_analytics_hooks.py` | `test_plugin_chat_id_contract.py` |
| 2 | `test_background_tasks.py` | reminders-тесты |
| 3 | `test_agent_tools_session_reset.py`, `test_agent_tools_schema_registry.py` | `test_agent_tools_plugin.py`, `test_database.py` |
| 4 | `test_hindsight_mutator.py`, `test_hindsight_session_lifecycle.py`, `test_hindsight_jobs.py` | `test_database.py`, `test_openai_helper_*` |
| 5 | `test_skills_prompt_fragment.py` | существующие skills-тесты |
| 6 | `test_no_hardcoded_plugin_refs.py` | `AGENTS.md` |

---

## Out of scope этого плана

- Перевод `Database` на `aiosqlite`. Отдельный план после этапа 6. Плагины не страдают — контракт `DbHandle` остаётся, меняется только реализация.
- Судьба `tool_call_events`.
- Реальные topics/sentiment в `conversation_analytics` (feature work).
- Очистка `hindsight_finalize_jobs` при удалении пользователя в `delete_user_data` — нужно решить через какой механизм плагин узнаёт об удалении пользователя (возможно, `on_user_delete` событие — добавлять только если потребуется по факту).

---

## Сводная оценка

| Этап | Срок | Сложность | Риск |
|---|---|---|---|
| 0 | 2-3 дня | средняя | низкий |
| 1 | 1 день | низкая | низкий |
| 2 | 1 день | низкая | низкий |
| 3 | 2-3 дня | средняя | средний |
| 4 | 5-7 дней | высокая | **высокий** |
| 5 | 1 день | низкая | низкий |
| 6 | 1 день | низкая | низкий |

Общее окно: 2-3 недели при последовательном выполнении.
Этап 4 — узкое место; этап 0 проектировать с особым вниманием к mutator-семантике.

---

## Точки хардкода в коде (для быстрой навигации)

Ссылки, чтобы не искать заново. Актуальны на момент составления плана (HEAD `main`).

### `bot/telegram_bot.py`
- `:349` — `_get_agent_tools_plugin` (этап 3)
- `:449` — generic `cleanup_after_delivery` через `plugin_id` (оставляем как образец)
- `:460` — `_enqueue_hindsight_session_finalize_before_delete` (этап 4C)
- `:491` — `_enqueue_hindsight_and_delete_oldest_sessions_for_limit` (этап 4C)
- `:497` — `_process_pending_hindsight_finalize_jobs` (этап 4C)
- `:539` — `hindsight_finalize_worker` (этап 4C)
- `:1397` — `_available_skill_names` — оставляем (UI, generic)
- `:1416` — `_hindsight_memory_plugin_for_user` (этап 4)
- `:1776` — `_enqueue_hindsight_and_delete_oldest_sessions_for_limit` (этап 4C)
- `:2801, :2846` — `get_plugin('conversation_analytics')` (этап 1)
- `:3305` — старт `hindsight_finalize_worker` (этап 4C)
- `:3891` — `start_reminder_checker` (этап 2)
- `:3985` — `_enqueue_hindsight_and_delete_oldest_sessions_for_limit` (этап 4C)
- `:4026` — `_enqueue_hindsight_session_finalize_before_delete` (этап 4C)

### `bot/openai_helper.py`
- `:55` — `from .hindsight_client import HindsightClient` (этап 4A)
- `:256-271` — 14 ключей `hindsight_*` setdefault (этап 4A)
- `:272-279` — конструктор `HindsightClient` (этап 4A)
- `:847` — `_prepare_hindsight_session_memory_context` (этап 4B)
- `:896, :1083, :1129` — `_messages_with_hindsight_context` (этап 4B)
- `:1565` — `is_hindsight_enabled` (этап 4A)
- `:1573` — `get_hindsight_bank_id` (этап 4A)
- `:1576` — `_hindsight_memory_types` (этап 4A)
- `:1584` — `_prepare_hindsight_session_memory_context` (этап 4B)
- `:1635` — `_messages_with_hindsight_context` (этап 4B)
- `:1737` — `_messages_with_hindsight_context` (этап 4B)
- `:2441` — `_get_available_skills_summary` (этап 5)
- `:2443` — `get_plugin("skills")` (этап 5)
- `:2455+` — блок «ПРИОРИТЕТНОЕ ПРАВИЛО» в `_build_auto_chat_mode_prompt` (этап 5)

### `bot/database.py`
- `:134, :143, :157` — DDL `agent_plan_*` (этап 3B)
- `:223-241` — DDL `hindsight_finalize_jobs` (этап 4C)
- `:332` — `save_agent_plan` (этап 3B)
- `:370` — `get_agent_plan` (этап 3B)
- `:413` — `clear_agent_plan` (этап 3B)
- `:425` — `cleanup_old_agent_plans` (этап 3B)
- `:672` — очистка `hindsight_finalize_jobs` в `delete_user_data` (этап 4C)
- `:849, :863, :894, :956, :974` — hindsight job-методы (этап 4C)
- `:1074-1081` — `hindsight_requires_async_finalize` ветка (этап 4C)
