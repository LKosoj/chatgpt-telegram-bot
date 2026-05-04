# ТЗ на устранение выбранных багов Telegram-бота

Дата: 2026-05-04

## 1. Цель

Устранить только перечисленные в задаче дефекты и связанные с ними улучшения в
боте, tool-call flow, плагинах и Telegram runtime.

Документ должен быть достаточным для разработки, ревью и приемки без повторного
расследования причин багов.

## 2. Границы работ

В scope входят только следующие группы:

1. Streaming-ответы Telegram.
2. Ограничения tools из chat modes после tool-call.
3. Авторизация callback flows.
4. Per-request state в `OpenAIHelper` при `concurrent_updates(True)`.
5. Несогласованный тип `chat_id` в tool-call и command/document paths.
6. Смешивание `chat_id` и `user_id` в group/session flow.
7. Падение plugin menu input из-за отсутствующего `ForceReply`.
8. Некорректный `ReactionPlugin` direct result.
9. Неработающий `ask_your_pdf.analyze_pdf`.
10. Улучшения, прямо перечисленные в задаче:
    RequestContext, per-chat/session serialization, JSON Schema validation,
    async-safe HTTP calls в выбранных плагинах, единая регистрация plugin
    message handlers, конфигурируемый Telegram Bot API local mode/base URL.

## 3. Вне scope

Не исправлять в рамках этого ТЗ:

- небезопасный `codeinterpreter`;
- старые SQLite-миграции и дефекты pruning sessions;
- любые новые пользовательские фичи;
- переименование плагинов и команд без необходимости;
- рефакторинг unrelated кода;
- изменение README, кроме минимального runtime-документирования новых env vars,
  если это потребуется для приемки.

## 4. Общие требования к реализации

1. Изменения должны быть хирургическими: каждая правка должна относиться к
   одному из пунктов scope.
2. Существующие публичные команды и plugin specs должны сохранять обратную
   совместимость, кроме явно исправляемых внутренних аргументов `chat_id`,
   `user_id`, `message_id`.
3. Runtime-поведение должно быть подтверждено тестами, а не только ручным
   осмотром.
4. Не вызывать внешние сервисы в тестах. HTTP, Telegram API, OpenAI и plugin
   network calls должны мокаться.
5. В местах, где раньше ошибки глотались через bare `except`, нужно логировать
   причину и возвращать управляемую ошибку либо fallback только для legacy flow.

## 5. Архитектурное решение

### 5.1 RequestContext

Ввести единый immutable request context для передачи per-request данных.

Рекомендуемая форма:

```python
@dataclass(frozen=True)
class RequestContext:
    chat_id: int
    user_id: int
    message_id: int | None = None
    session_id: str | None = None
    request_id: str | None = None

    @property
    def plugin_chat_id(self) -> str:
        return str(self.chat_id)
```

Требования:

- `chat_id` и `user_id` внутри Telegram/OpenAI flow остаются числовыми.
- Plugin-facing `chat_id` должен быть строкой, чтобы совпадать с текущими
  command/document paths.
- `RequestContext` передается в `OpenAIHelper`, `openai_tool_handler`,
  `PluginManager.call_function()` и плагины через явный аргумент
  `request_context`.
- Запрещено записывать request-local значения в `helper.user_id` и
  `helper.message_id` в новом flow.
- Для обратной совместимости можно временно оставить поля на `OpenAIHelper`, но
  они не должны использоваться обновленными плагинами.

### 5.2 Conversation/session key

Ввести единое правило:

- `actor_user_id` используется для авторизации, бюджета и admin checks.
- `conversation_key` используется для истории, active session, mode и DB
  context.
- В private chat `conversation_key == user_id == chat_id`.
- В group/supergroup `conversation_key == effective_chat.id`, а не
  `callback_query.from_user.id`.

Все session callbacks и prompt mode callbacks должны использовать тот же
`conversation_key`, что и обычные сообщения в `process_message()`.

### 5.3 Per-conversation serialization

Сохранить `concurrent_updates(True)` только если история сериализуется per
conversation key.

Требования:

- Добавить registry lock-ов по `conversation_key` или `(conversation_key,
  session_id)`.
- Операции "добавить user message -> вызвать модель/tool calls -> добавить
  assistant/tool result -> сохранить контекст" должны быть атомарны для одного
  conversation key.
- Разные чаты не должны блокировать друг друга.
- Если реализация lock-ов окажется слишком рискованной, допустим временный
  fallback: отключить `concurrent_updates(True)` и явно описать это в changelog.
  Но предпочтительный вариант - per-conversation locks.

## 6. Функциональные требования по багам

### 6.1 Streaming-ответы Telegram не отправляются

Проблема:

- `Database.get_conversation_context()` возвращает 5 значений.
- Streaming path в `ChatGPTTelegramBot.process_message()` распаковывает 3
  значения и глотает исключение.

Требования:

1. Исправить распаковку результата `get_conversation_context()` в streaming
   ветке.
2. Удалить bare `except` вокруг первичной отправки streaming-сообщения.
3. Если отправка первого сообщения не удалась, ошибка должна логироваться с
   `exc_info=True`, а пользователь должен получить понятное сообщение об ошибке
   либо поток должен корректно завершиться.
4. `i` не должен оставаться бесконечно равным `0` из-за скрытых ошибок.
5. `parse_mode` должен иметь fallback, если DB вернула `None`.

Критерии приемки:

- Первый chunk stream создает Telegram message.
- Последующие chunks редактируют созданное сообщение.
- Ошибка parse mode не приводит к молчаливой потере всего ответа.

Тесты:

- Fake stream из 2-3 chunks.
- Fake `get_conversation_context()` возвращает 5-tuple.
- Проверить, что `reply_text` вызван один раз для первого chunk.
- Проверить, что `edit_message_with_retry` вызван для последнего chunk.
- Проверить negative case: исключение в `reply_text` логируется и не зацикливает
  обработку.

### 6.2 Chat modes теряют ограничения tools после tool-call

Проблема:

- Initial model request вычисляет `allowed_plugins` из текущего mode.
- После tool-call `__handle_function_call()` вызывается без allow-list.
- `openai_tool_handler.handle_function_call()` имеет default `['All']` и при
  model re-entry снова передает все tools.

Требования:

1. Вынести вычисление allowed plugins в один метод, например
   `OpenAIHelper.resolve_allowed_plugins(chat_id, session_id)`.
2. Использовать этот метод и при initial request, и при tool-call handling.
3. `OpenAIHelper.__handle_function_call()` должен получать уже вычисленный
   `allowed_plugins`.
4. `openai_tool_handler.handle_function_call()` не должен default-ить к `All`,
   если caller уже знает ограничения mode.
5. Перед выполнением tool-call нужно проверять, что запрошенная функция
   принадлежит разрешенному плагину.
6. Проверка должна применяться и к legacy JSON tool request из assistant content.
7. На model re-entry `get_functions_specs()` должен получать тот же
   `allowed_plugins`, а не `All`.

Критерии приемки:

- Mode с `tools: ['weather']` не может выполнить `task_management.create_task`.
- После первого tool-call re-entry model видит только разрешенные tools.
- Legacy JSON tool request вне allow-list возвращает controlled error и не
  вызывает plugin.

Тесты:

- Unit-тест на `OpenAIHelper`/`openai_tool_handler` с fake plugin manager.
- Тест direct tool-call вне allow-list.
- Тест legacy JSON tool request вне allow-list.
- Тест re-entry: `get_functions_specs()` вызван с исходным allow-list.

### 6.3 Callback авторизация

Проблемы:

- `/reset` через callback отказывает обычным пользователям при
  `allowed_user_ids='*'`.
- `handle_prompt_selection()` и `handle_session_callback()` меняют режимы и
  сессии без нормального `is_allowed()`.

Требования:

1. В `reset()` для callback path заменить ad hoc проверку на общий
   `is_allowed(config, update, context)`.
2. В `handle_prompt_selection()` сразу после `query.answer()` выполнить
   `is_allowed()`.
3. В `handle_session_callback()` сразу после `query.answer()` выполнить
   `is_allowed()`.
4. Если пользователь не авторизован:
   - не создавать, не переключать, не удалять, не экспортировать сессии;
   - не менять mode;
   - показать локализованное сообщение отказа.
5. Поведение `allowed_user_ids='*'` должно означать allow-all во всех callback
   flows.

Критерии приемки:

- Callback reset разрешен обычному пользователю при allow-all config.
- Restricted config запрещает callback reset/session/mode mutation для
  неразрешенного пользователя.
- Admin по-прежнему разрешен.

Тесты:

- `reset()` callback with `allowed_user_ids='*'`.
- `reset()` callback restricted non-member.
- `handle_prompt_selection()` unauthorized: DB/OpenAI context не меняется.
- `handle_session_callback()` unauthorized: `create_session`,
  `switch_active_session`, `delete_session`, `export_sessions_to_yaml` не
  вызываются.

### 6.4 Убрать mutable per-request state с OpenAIHelper

Проблема:

- `openai_tool_handler` пишет `helper.user_id`.
- Telegram flow пишет/читает `helper.message_id`.
- Плагины `task_management`, `language_learning`, `reminders` читают эти поля.
- При `concurrent_updates(True)` возможна межпользовательская путаница.

Требования:

1. Заменить запись `helper.user_id = ...` на передачу `user_id` через kwargs и
   `request_context`.
2. Заменить `helper.message_id` на `request_context.message_id`.
3. Обновить плагины:
   - `task_management.py`: брать user id из `kwargs['user_id']` или
     `request_context.user_id`;
   - `language_learning.py`: то же для progress ownership;
   - `reminders.py`: брать `reply_to_message_id` из `request_context.message_id`.
4. Не использовать shared mutable поля для новых данных.
5. Для legacy совместимости допустим fallback чтения старых полей только если
   явного `request_context` нет; fallback должен быть помечен TODO/deprecation и
   покрыт тестом.

Критерии приемки:

- Два параллельных tool-call от разных пользователей не смешивают task/progress
  ownership.
- Reminder сохраняет `reply_to_message_id` текущего запроса, а не чужого.
- `helper.user_id` не используется обновленными плагинами как основной источник.

Тесты:

- Concurrent async test: два fake tool-call с разными `user_id`.
- Проверить task ids/owners.
- Проверить language progress owners.
- Проверить reminder `reply_to_message_id`.

### 6.5 Стандартизировать `chat_id` для plugins

Проблема:

- Tool-call path инжектит `chat_id` как `int`.
- Command/document paths передают `chat_id` как `str`.
- `text_document_qa` сравнивает строковый owner.
- `conversation_analytics` объявляет `chat_id` как string.

Требования:

1. Ввести правило: plugin-facing `chat_id` всегда `str`.
2. В `openai_tool_handler` инжектить `chat_id = str(request_context.chat_id)`.
3. В command/document paths сохранить строковый `chat_id`.
4. Внутренние поля `chat_id`, `user_id`, `message_id`, `update`,
   `request_context` не должны требоваться от модели.
5. Для plugin specs, где `chat_id` нужен только как injected arg, убрать его из
   model-visible `required`.
6. Если `chat_id` остается в schema, его тип должен совпадать с фактическим
   injected типом.

Критерии приемки:

- `text_document_qa` list/ask/delete работает одинаково через command path и
  tool-call path.
- `conversation_analytics` не отклоняется validation из-за int/string mismatch.
- Model не обязана сама придумывать `chat_id`.

Тесты:

- Tool-call для `text_document_qa.list_documents` с metadata owner as string.
- Tool-call для `conversation_analytics.analyze_conversation`.
- Проверка advertised schema не содержит internal-only required args.

### 6.6 Group/session flow: единый ключ

Проблема:

- Обычные group messages используют `chat_id`.
- Session callbacks используют `user_id`.
- Mode selection сохраняет context под `chat_id`, но берет active session по
  `user_id`.

Требования:

1. Ввести helper, например `get_conversation_key(update)`, который возвращает:
   - private: `effective_user.id`;
   - group/supergroup: `effective_chat.id`.
2. В session callbacks использовать `conversation_key` для:
   - `list_user_sessions`;
   - `create_session`;
   - `switch_active_session`;
   - `delete_session`;
   - `get_conversation_context`;
   - `openai.conversations[...]`.
3. `actor_user_id` оставить для auth/admin/budget.
4. Mode selection должен читать и писать active session по тому же
   `conversation_key`, по которому later prompt будет вести историю.
5. Не менять private chat behavior.

Критерии приемки:

- В группе `/reset -> change_mode -> prompt` работает на одной и той же истории.
- Переключение сессии в группе влияет на последующие group prompts.
- Private session flow не меняется.

Тесты:

- Fake group callback: `query.message.chat_id != query.from_user.id`.
- Проверить, что DB methods вызываются с group chat id.
- Проверить, что `openai.conversations[group_chat_id]` обновляется.
- Private regression test.

### 6.7 Plugin menu input: `ForceReply`

Проблема:

- `ForceReply` используется, но не импортирован.

Требования:

1. Импортировать `ForceReply` из `telegram`.
2. Добавить тест на callback `pluginmenu:input:*`.
3. Проверить, что создается reply prompt и `context.user_data["plugin_menu_pending"]`.

Критерии приемки:

- Callback не падает `NameError`.
- Пользователь получает force-reply prompt.

### 6.8 ReactionPlugin direct result

Проблема:

- `ReactionPlugin` возвращает kind `reaction` без `format`.
- `handle_direct_result()` читает `result['format']` до branch по kind и не
  поддерживает reaction.

Требования:

1. Изменить `handle_direct_result()` так, чтобы `format` читался только для
   branch-ей, которым он нужен.
2. Реализовать branch `kind == 'reaction'`.
3. Reaction должен применяться к reply target, если он есть.
4. Если target message отсутствует или Telegram API не поддерживает реакцию,
   пользователь должен получить controlled fallback message.
5. `ReactionPlugin` должен возвращать валидную shape для нового branch.

Критерии приемки:

- `react_with_emoji` не падает на отсутствии `format`.
- При наличии reply target вызывается Telegram reaction API.
- При отсутствии target возвращается понятный fallback.

Тесты:

- Unit-тест `handle_direct_result()` для `kind='reaction'`.
- Тест отсутствующего reply target.
- Regression test для `photo`, `gif`, `file`, `text`, если они уже покрыты или
  легко мокируются.

### 6.9 `ask_your_pdf.analyze_pdf`

Проблема:

- Используются undefined `analysis_prompt` и `result`.

Требования:

1. Реализовать извлечение текста из PDF:
   - сначала PyPDF2;
   - fallback через `textract`, если dependency доступна и это уже поддержано
     окружением;
   - controlled error, если текст извлечь нельзя.
2. Сформировать `analysis_prompt` из:
   - пользовательского `query`;
   - извлеченного текста;
   - имени файла;
   - явного ограничения максимального объема текста.
3. Вызвать `helper.get_chat_response()` или другой существующий helper
   консистентно с остальным кодом.
4. Сформировать `result` как JSON-serializable dict, например:
   `{"result": response, "file_hash": file_hash}`.
5. Сохранить в cache именно сформированный `result`.
6. При cache hit возвращать тот же contract.
7. Не возвращать произвольный `file_path` пользователю как direct file result,
   если путь не находится в контролируемой директории плагина.

Критерии приемки:

- `analyze_pdf` успешно отвечает на fake PDF/text extraction.
- Cache hit не вызывает helper повторно.
- Missing file возвращает controlled error.
- Undefined name exceptions отсутствуют.

Тесты:

- Happy path с monkeypatch extractor.
- Cache hit.
- Missing file.
- Oversized extracted text truncation.
- Path outside allowed storage rejected для `upload_pdf`.

## 7. Улучшения из scope

### 7.1 JSON Schema validation

Требования:

1. Заменить ручную проверку типов в `validation.py` на полноценную JSON Schema
   validation.
2. Добавить зависимость `jsonschema` в `requirements.txt`.
3. `bool` не должен проходить как `integer` или `number`.
4. Поддержать минимум:
   - `required`;
   - `type`;
   - `enum`;
   - nested `object`;
   - `array.items`;
   - `additionalProperties`, если оно задано в spec.
5. Ошибки validation должны быть короткими и пригодными для возврата модели.

Тесты:

- `true` против `integer` и `number`.
- Invalid `enum`.
- Missing required.
- Nested object.
- Array item type.
- Existing plugin arg validation regression.

### 7.2 Blocking HTTP calls в async plugins

Файлы scope: `weather.py`, `crypto.py`, `iplocation.py`.

Требования:

1. Убрать blocking `requests.get()` из async `execute()`.
2. Использовать `httpx.AsyncClient` с явным timeout или
   `asyncio.to_thread()` как минимальный fallback.
3. Любой network error должен возвращать controlled plugin error.
4. JSON parse errors должны обрабатываться отдельно.
5. Не вызывать реальные внешние API в тестах.

Тесты:

- Успешный mocked response.
- Timeout/error response.
- Проверка, что event loop не блокируется: параллельная async задача должна
  завершиться, пока HTTP call замокан как slow.

### 7.3 Единая регистрация plugin message handlers

Проблема:

- Plugin message handlers регистрируются и в `post_init()`, и в `run()`.

Требования:

1. Оставить один путь регистрации.
2. Предпочтительно регистрировать plugin message handlers в `post_init()`, где
   уже регистрируются plugin commands.
3. Удалить дублирующий блок из второго места без изменения порядка built-in
   handlers.
4. Добавить защиту от повторной регистрации, если `post_init()` будет вызван
   повторно.

Тесты:

- Fake application считает добавленные handlers.
- Каждый plugin message handler регистрируется один раз.
- Built-in text/document handlers сохраняют порядок относительно plugin
  handlers.

### 7.4 Telegram local Bot API config

Проблема:

- `.local_mode(True)` и `.base_url('http://localhost:8081/bot')` захардкожены.

Требования:

1. Добавить env/config:
   - `TELEGRAM_LOCAL_MODE`, bool;
   - `TELEGRAM_BASE_URL`, string optional.
2. Для обратной совместимости дефолт может сохранять текущее поведение, если это
   подтверждено deploy requirements. Если подтверждения нет, дефолт должен быть
   стандартный Telegram cloud API.
3. Builder должен вызывать `.local_mode()` и `.base_url()` только согласно
   config.
4. Невалидный base URL должен приводить к startup/config error, а не к позднему
   polling failure.

Тесты:

- Default config.
- Local mode enabled with custom base URL.
- Local mode disabled: base URL не устанавливается.
- Invalid URL rejected.

## 8. Сквозные критерии приемки

1. Все перечисленные баги имеют тесты, которые падают на старом коде и проходят
   после исправления.
2. `pytest -q` проходит в окружении с установленными зависимостями.
3. Skipped tests допустимы только для optional integrations и должны быть явно
   объяснены.
4. Не появляется новых внешних вызовов в тестах.
5. `git diff` не содержит unrelated форматирования или рефакторинга.
6. `README.md` не трогать, кроме минимальной документации новых env vars, если
   это нужно.
7. Все новые env vars имеют дефолты и тесты parsing.

## 9. Рекомендуемый порядок внедрения

### Этап 1. Быстрые изолированные фиксы

1. `ForceReply` import и тест.
2. `ReactionPlugin`/`handle_direct_result` branch и тесты.
3. `ask_your_pdf.analyze_pdf` undefined vars и тесты.
4. `validation.py` JSON Schema validation и тесты.

Причина: эти задачи локальны и уменьшают шум перед изменением core flow.

### Этап 2. Tool-call contract

1. Ввести `RequestContext`.
2. Стандартизировать plugin-facing `chat_id` как `str`.
3. Убрать `helper.user_id`/`helper.message_id` из обновленных плагинов.
4. Исправить allow-list для chat-mode tools.
5. Добавить tests на restricted tools и concurrent users.

Причина: эти изменения связаны общим контрактом передачи контекста.

### Этап 3. Telegram/session runtime

1. Исправить streaming path.
2. Исправить callback auth.
3. Ввести group/private `conversation_key`.
4. Добавить per-conversation locks или отключить concurrent updates.
5. Убрать duplicate plugin handler registration.
6. Сделать local Bot API config-driven.

Причина: эти изменения затрагивают ordering handlers, историю и session state.

### Этап 4. Async-safe network plugins

1. Перевести `weather.py`, `crypto.py`, `iplocation.py` на async-safe HTTP.
2. Добавить timeout/error tests.

Причина: это поведенчески отдельный слой, но нужен для устойчивости event loop.

## 10. Минимальный тестовый набор

Новые или расширенные файлы тестов:

- `tests/test_telegram_streaming.py`
- `tests/test_callback_authorization.py`
- `tests/test_openai_helper_tool_calls.py`
- `tests/test_plugin_arg_validation.py`
- `tests/test_plugin_direct_results.py`
- `tests/test_plugin_chat_id_contract.py`
- `tests/test_group_session_flow.py`
- `tests/test_async_plugins.py`
- `tests/test_plugin_handlers_registration.py`
- `tests/test_telegram_builder_config.py`

Если не хочется плодить файлы, допустимо расширять существующие тесты, но
названия тестов должны явно отражать баг:

- `test_streaming_unpacks_conversation_context_5_tuple`
- `test_mode_restrictions_survive_tool_reentry`
- `test_legacy_tool_request_respects_mode_allowlist`
- `test_callback_reset_allows_wildcard_allowed_users`
- `test_unauthorized_session_callback_does_not_mutate_db`
- `test_request_context_prevents_cross_user_tool_state`
- `test_tool_injected_chat_id_is_plugin_string`
- `test_group_session_callback_uses_group_conversation_key`
- `test_plugin_menu_input_imports_force_reply`
- `test_reaction_direct_result_without_format`
- `test_ask_your_pdf_analyze_pdf_defines_prompt_and_result`
- `test_jsonschema_rejects_bool_for_integer`
- `test_plugin_message_handlers_registered_once`

## 11. Риски и проверки ревью

1. RequestContext может затронуть много call sites. Ревьюер должен проверить,
   что старые поля `helper.user_id` и `helper.message_id` не остались основным
   источником истины.
2. Per-conversation lock может снизить throughput внутри одного чата. Это
   допустимо; межчатовая конкурентность должна сохраниться.
3. Исправление group/session key может изменить фактическую модель владения
   group sessions. Это ожидаемо: история и сессии должны использовать тот же
   ключ, что и prompts.
4. JSON Schema validation может начать отклонять tool args, которые раньше
   ошибочно принимались. Это желаемое поведение, но ошибки должны быть понятны
   модели.
5. Удаление duplicate handler registration может поменять порядок обработки
   сообщений. Нужен тест на порядок и ручная проверка `/plugins`, document
   upload и ordinary text prompt.

## 12. Definition of Done

Работа считается завершенной, когда:

1. Все требования из разделов 6 и 7 реализованы.
2. Все acceptance tests из разделов 6-10 проходят.
3. `pytest -q` проходит в полном dev environment.
4. Нет новых skipped tests без объяснения.
5. `git status --short` показывает только ожидаемые файлы.
6. В changelog или PR description явно перечислены:
   - измененный contract `RequestContext`;
   - canonical plugin-facing `chat_id: str`;
   - новые env vars Telegram builder;
   - любые legacy fallbacks.
