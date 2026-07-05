# Variant B: event/provider слой без замены ядра

Дата: 2026-07-05.

## Цель

Внедрить внутренний AI engine слой по мотивам Tau, но сохранить текущие сильные части
проекта как ядро:

- `PluginManager`, plugin hooks, validation и текущий tool-loop остаются источником истины.
- Telegram delivery, включая rich markdown delivery/drafts, остается в `telegram_bot.py`,
  `telegram_rich.py` и `utils.py`.
- OpenAI/PTB несовместимости закрываются адаптерами, а не переписыванием всего runtime.
- Новый слой включен по умолчанию; `CHAT_RUN_VARIANT_B_ENABLED=false` остается
  rollback-путем до DTO-миграции.

## Уже реализованный срез

1. `bot/ai_events.py`
   - frozen DTO для provider/run событий;
   - raw tool arguments сохраняются строкой;
   - model-visible tool name отделен от canonical plugin name через `model_name`.

2. `bot/ai_provider.py`
   - минимальный `AIProvider` protocol;
   - `AIProviderRequest` и `AIProviderResponse`;
   - collector для event stream в ответ.

3. `bot/ai_providers/fake.py`
   - deterministic fake provider для unit/integration tests;
   - fail-fast, если тест забыл положить ответ в очередь.

4. `bot/ai_providers/openai_compatible.py`
   - shell-адаптер для OpenAI-compatible response shape;
   - поддержка non-stream response и streaming text/tool-call aggregation;
   - split/interleaved streamed tool arguments не теряются.

5. `bot/plugin_tool_adapter.py`
   - тонкий bridge к `PluginManager`;
   - canonicalization выполняется перед execution;
   - raw provider arguments не парсятся в adapter-слое.

6. `bot/chat_run.py`
   - default compatibility run shell для text chat;
   - chat-completion requests, включая streaming, проходят через provider/event adapter;
   - raw SDK-shaped response временно сохраняется для текущего tool-loop;
   - пока вызывает существующие private helper methods, чтобы сохранить поведение;
   - не меняет image/audio, Telegram delivery и plugin command paths.

7. `CHAT_RUN_VARIANT_B_ENABLED=true`
   - production default включен;
   - `false` остается rollback-путем на legacy dispatcher.

## Архитектурный план

### Фаза 1. Provider/event foundation

Статус: сделано.

Acceptance criteria:

- AI events сериализуются в лог без потери вложенных dataclass/list/dict значений.
- Invalid JSON tool arguments доходят до tool-loop как raw string.
- Fake provider детерминирован и не скрывает пропущенные test fixtures.
- OpenAI-compatible adapter не теряет streamed tool calls.

### Фаза 2. Tool boundary adapter

Статус: сделано.

Acceptance criteria:

- `PluginManager.get_functions_specs()` остается единственным источником tool specs.
- `PluginManager.call_function()` остается единственным execution entrypoint.
- model-visible names преобразуются в canonical names только перед plugin execution.
- adapter не дублирует validation, disabled-plugin policy и request-context injection.

### Фаза 3. ChatRun default orchestration

Статус: compatibility wrapper включен для chat-completion paths, включая streaming.

Scope:

- `get_chat_response()` non-stream text path;
- `get_chat_response_stream()` streaming text path;
- low-level `chat_completion()` и tool re-entry;
- reply-intent classifier;
- vision chat-completion request, включая `interpret_image_stream()`;
- summary chat-completion request;
- без изменения image/audio endpoints;
- без изменения Telegram rich delivery.

Acceptance criteria:

- при `CHAT_RUN_VARIANT_B_ENABLED=false` используется legacy path;
- при `true` plain chat response совпадает с legacy контрактом;
- tool-call flow сохраняет injection `chat_id`/`user_id`;
- token accounting для initial response + tool reentry сохраняется;
- empty-response repair продолжает работать через существующие helper methods.

### Фаза 4. Transcript repair и event observability

Scope:

- вынести repair decisions в явные события:
  `AIRetry`, `AIToolExecutionStart`, `AIToolExecutionEnd`, `AIRunEnd`;
- подключить эти события к `SessionLogger` без расширения обычных debug payload rules;
- сохранить текущую crash-safe sanitization отдельно от normal debug visibility.

Acceptance criteria:

- пустой ответ до/после tool calls логируется как retry event;
- tool execution start/end содержит canonical name, provider name и duration;
- session log не редактирует caller payload in-place;
- существующие tests на normal logs и sanitized session events остаются зелеными.

### Фаза 5. Provider adapter в реальном request path

Scope:

- добавить adapter seam около `_timed_create`, не вокруг Telegram delivery;
- применять для chat-completion path, включая streaming, за rollback-флагом;
- сохранить OpenAI SDK response shape на выходе до завершения tool-loop migration.
- текущий срез считается compatibility wrapper, не финальной provider-neutral заменой
  tool-loop.

Acceptance criteria:

- legacy `_timed_create` parity покрыт тестами;
- provider errors получают структурированное событие и прежнее user-facing сообщение;
- no external service verification в тестах;
- rollback = выключить флаг.

### Фаза 6. Decision point: Pydantic v2 strict для provider-boundary DTO

Решение пока не принято и не должно блокировать текущий срез.

Когда рассматривать:

- появится второй реальный provider adapter с отличающимся response shape;
- будет повторяющийся класс багов из-за malformed provider payload;
- понадобится runtime validation внешнего provider payload до попадания в tool-loop.

Если включать:

- только на boundary DTO между provider adapter и core event stream;
- `strict=True`, `extra="forbid"`;
- без Pydantic в plugin DTO, Telegram delivery DTO и hot path внутренних событий;
- benchmark/тест на накладные расходы обязателен перед включением по умолчанию.

Если не включать:

- оставить frozen dataclasses и targeted normalization tests;
- документировать provider adapter contracts через tests.

### Фаза 7. Rollout

Scope:

- держать `CHAT_RUN_VARIANT_B_ENABLED=true` как default;
- использовать `CHAT_RUN_VARIANT_B_ENABLED=false` как быстрый rollback на legacy path;
- сравнивать legacy/new paths по token accounting, tool calls, retries и direct results;
- rich Telegram delivery остается управляемой текущими флагами:
  `TELEGRAM_RICH_MESSAGES=auto` и `TELEGRAM_RICH_DRAFTS=true`.

Acceptance criteria:

- targeted tests green;
- full `pytest -q` green перед production rollout;
- `git diff --check` clean;
- reviewer loop clean: no errors, no warnings.

## Порядок дальнейшей разработки

Сделано в текущем срезе:

1. Provider/event compatibility wrapper подключен к chat completions за
   `CHAT_RUN_VARIANT_B_ENABLED`.
2. Streaming chat completions также проходят через provider/event seam, сохраняя raw SDK
   async iterator contract для существующего Telegram/tool-loop кода.
3. Tool re-entry и empty-response retry для default path проходят через тот же
   provider/event seam.
4. Reply intent, vision и summary chat-completion calls также проходят через тот же
   rollback-флаг.
5. `SessionLogger` получает `ai_provider_response`, `provider_error`, `retry`,
   `tool_execution_start`, `tool_execution_end` и `run_end` events.
6. Targeted tests покрывают plain response, streaming response, tool re-entry, direct result,
   retry/run end, reply intent, vision, summary call-site, provider failure event и tool
   execution event payload.

Оставшаяся работа:

1. Провести Pydantic decision point отдельно, после второго provider adapter или реального
   malformed-payload кейса.
2. Мигрировать downstream tool-loop decisions с SDK-shaped response на provider-neutral DTO.
3. После DTO-миграции убрать compatibility `last_response` bridge из provider adapter.
4. Не удалять rollback-флаг до DTO-миграции и отдельного production window.

## Не цели

- Не переписывать `OpenAIHelper` целиком.
- Не переносить Telegram delivery в provider/event слой.
- Не менять plugin platform contract.
- Не добавлять Pydantic как общую зависимость до decision point.
- Не переносить image/audio endpoints в chat provider compatibility wrapper.
