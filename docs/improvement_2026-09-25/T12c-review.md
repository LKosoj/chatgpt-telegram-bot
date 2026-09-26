# T12c. `openai_helper.py` / `openai_tool_handler.py` / `chat_run.py` / `tool_result.py` — ревью

## Раунд 1

**Область ревью.** Сверено по `git diff HEAD` (база `08bc457`) с ручным разделением хантков
по волнам, т.к. T08–T11 трогали те же файлы раньше T12c:

- `bot/openai_helper.py` (503 изменённые строки в diff) — из них T12c: `_begin_turn`
  (новый метод, `:824-849`) + подключение в `get_chat_response` (`:877-879`) и
  `get_chat_response_stream` (`:959-961`); `_begin_simple_turn` (новый метод, `:2382-2389`) +
  подключение в `interpret_image` (`:2404`), `interpret_images` (`:2439`),
  `interpret_image_stream` (`:2557`); `_persist_conversation_context` (новый метод,
  `:684-702`) + подключение в `_maybe_apply_auto_chat_mode` (`:1266-1268`),
  `reset_chat_history` (`:3369-3371`), `_add_to_history` (`:3466-3468`),
  `record_plugin_exchange` (`:3510-3512`); `_interpret_image_text_response` (`:2534-2542`)
  заменено на вызов `finalize_chat_answer` (T12a, `bot/chat_response_utils.py:119-177`);
  `leading_system_count` (T12a) подключена в `_summarize_and_trim` (`:3746`) и
  `_fallback_trim_with_summary` (`:3824`). Остальные хантки файла (провайдер T10, обёртка
  untrusted-контента и `role: user` для tool-результатов T08, `ChatStateRegistry`/9
  compat-свойств/`_mutable_history`/`_chat_lock` T11, формулировка роутера T09) к T12c не
  относятся, по существу не проверялись — их владение в других Txx-review.md.
- `bot/chat_run.py` (133 изменённые строки) — весь diff T12c: новый метод
  `_retry_after_empty_response` (`:44-65`) заменяет два дословных повтора внутри
  `run_non_stream` (было `:101-126`/`:149-174` до правки); хвост `run_non_stream`
  (было `:200-254`) заменён на вызов `finalize_chat_answer`.
- `bot/openai_tool_handler.py` (74 изменённые строки) — из них T12c: удаление локального
  `_artifact_path` (было `:673-679`) и импорт `_artifact_path` из `.tool_result` (`:19`).
  Остальные хантки (`DANGEROUS_TOOL_NAMES`/`_tainted_plugin_ids` T08,
  `import openai`→`ProviderStreamError` T10, `_conversation_messages`→`helper._mutable_history`
  T11) к T12c не относятся.
- `bot/tool_result.py` — без изменений (diff пуст); `_artifact_path` (`:49-55`) остаётся
  канонической реализацией, к которой теперь обращается `openai_tool_handler.py`.
- Новые/затронутые тесты: `tests/test_t12c_chat_run_retry.py` (новый, 3 теста),
  `tests/test_t12c_artifact_path_dedup.py` (новый, 5 тестов),
  `tests/test_compat_state_views_guard.py` (T11-владение, счётчики понижены T12c-выносом
  `_persist_conversation_context` — 4 идентичных вызова `_save_conversation_context` слились
  в один сайт внутри нового метода).

**Прогон тестов и статического анализа.**
- `~/.venvs/ctb/bin/python -m pytest tests/test_compat_state_views_guard.py
  tests/test_t12c_artifact_path_dedup.py tests/test_t12c_chat_run_retry.py -q` — 11 passed.
- `~/.venvs/ctb/bin/python -m pytest tests/ bot/tests/ -q` — 1930 passed (база HEAD 08bc457 —
  1650; рост согласуется с волнами 1–12).
- `~/.venvs/ctb/bin/python -m ruff check bot/openai_helper.py bot/openai_tool_handler.py
  bot/chat_run.py bot/tool_result.py tests/test_t12c_*.py
  tests/test_compat_state_views_guard.py` — чисто.
- `python3 -m mypy bot/openai_helper.py bot/openai_tool_handler.py bot/chat_run.py
  bot/tool_result.py --python-executable ~/.venvs/ctb/bin/python --ignore-missing-imports`,
  сравнение по кодам ошибок с `/tmp/impl/mypy_before.keep`: `openai_helper.py` 46→44,
  `chat_run.py` 2→1 (оба уменьшения — из-за исчезнувших переменных/атрибутов при переносе
  кода, не из-за новых аннотаций), `openai_tool_handler.py` 7→7, `tool_result.py` 0→0.
  Ни одного нового кода ошибки не появилось — регрессий нет.

**Проверено построчно (не только сравнение diff, но и чтение итогового кода):**
- `_begin_turn`/`_begin_simple_turn` — тексты RuntimeError, порядок
  `_CHAT_STATE_KEY.set`/`set_trace`/`_TURN_STATS.set`/`slog.record` идентичны исходным двум
  копиям; оба вызывающих метода сохранили разный `finally`/пост-обработку (`get_chat_response`
  пишет `assistant_response` в `try` до `return`, `get_chat_response_stream` — в `finally`,
  т.к. это генератор) — экстракция не тронула эту асимметрию. Исключение из `_begin_turn`
  (реентрантность) происходит до `try:` в обоих местах, как и в оригинале — `finally` не
  теряет и не задваивает cleanup.
- Различие "max-age reset" между `_common_get_chat_response`
  (`state_key not in self.conversations or self.__max_age_reached(state_key) or
  session_changed`, `:1290`) и `_get_chat_response_stream_locked`
  (`state_key not in self.conversations or session_changed`, без `__max_age_reached`,
  `:1017`) — существовало до T12c и лежит вне вынесенного `_begin_turn`
  (тот покрывает только реентрантный чек + turn/trace/stats setup); экстракция не могла его
  задеть — подтверждено чтением обоих `_locked`-методов целиком.
  Тот же паттерн подтверждён и для третьего дубликата этого блока — `record_plugin_exchange`
  (`:3493-3497`, тоже с `__max_age_reached`) — все 3 места по-прежнему разные, что и
  объясняет, почему план (T12c C3/C10/C12a, "Session-load/reset-check") не был реализован
  (см. NIT ниже).
- `_persist_conversation_context` — 4 вызывающих места (`_maybe_apply_auto_chat_mode`,
  `reset_chat_history`, `_add_to_history`, `record_plugin_exchange`) передают аргументы
  в одном порядке, совпадающем с сигнатурой; проверены как единственные 4 места, где
  аргумент вызова буквально `{'messages': self.conversations[state_key]}` — три оставшихся
  прямых вызова `_save_conversation_context` (`:2712` внутри hindsight-хука с
  `try/except`+условным `persist`, `:2768` внутри другого хука с `conv.insert(...)`, `:3131`
  внутри `replace_system_message`) передают локальные переменные (`cleaned`/`conv`/
  `current_context`) и другую окружающую логику (try/except, условный persist) — правильно
  не включены в дубль-кластер, вызывающие места не забыты.
- `finalize_chat_answer` — ровно 2 потребителя (`chat_run.py:209`,
  `openai_helper.py:2535`), оба вызова совпадают по набору kwargs с сигнатурой
  `bot/chat_response_utils.py:119`. Тело функции побайтово соответствует прежнему тексту
  `chat_run.py`/`_interpret_image_text_response` (сверено построчно, включая порядок footer
  usage/plugins и формулу `total_tokens = sum(token_accumulator or []) or
  response_total_tokens(response)`).
- `leading_system_count` — единственные 2 вхождения старого цикла
  (`role == 'system'` head-scan) в репозитории замещены; других таких циклов в
  `bot/openai_helper.py`/`bot/chat_run.py`/`bot/openai_tool_handler.py`/`bot/tool_result.py`
  нет (проверено по всем вхождениям `== 'system'` в этих файлах).
- `_retry_after_empty_response` (`chat_run.py`) — прогнал вручную все три ветки нового теста
  (direct_result после ретрая на "after_tool_calls", direct_result после ретрая на
  "before_tool_calls", ретрай недоступен → `response` не подменяется, доходит до
  `finalize_chat_answer` и падает с `EMPTY_MODEL_RESPONSE_ERROR`, как раньше). Есть
  теоретическая асимметрия: раньше внешний код проверял `retry_response is not None` по
  результату `_retry_empty_response_with_tools` (до `_handle_function_call`), теперь —
  по результату `_handle_function_call` (после); совпадает при штатном контракте
  `_handle_function_call` (никогда не возвращает `None` как `response`) — не нашёл ни одного
  вызывающего или тестового места, где это не так, поэтому не поднимаю до WARNING.
- `_artifact_path` (`openai_tool_handler.py`↔`tool_result.py`) — удалённая копия и
  оставшаяся в `tool_result.py:49-55` побайтово идентичны; `os` в `openai_tool_handler.py`
  остаётся используемым (`os.getenv`, `:6`) — импорт не осиротел; `ARTIFACT_PATH_KEYS`
  (`tool_result.py`) и `_DIRECT_RESULT_ARTIFACT_KEYS` (`openai_tool_handler.py:720-734`) —
  два разных списка ключей не тронуты, слит только валидатор пути, как и требовал план.

## Находки

### WARNING: не выполнен пункт плана "Provider error log" (T12c C3/C10/C12a)
- **Файл:** `bot/openai_helper.py:1482-1490` (`_common_get_chat_response`) и
  `bot/openai_helper.py:2313-2321` (`__common_get_chat_response_vision`).
- План (`docs/improvement_2026-09-25/T12-plan.md`, раздел T12c C3/C10/C12a) явно требовал
  вынести общие ветки `except ProviderRateLimitError`/`except ProviderBadRequestError` в
  helper, оставив `ValueError` только в текстовом варианте. На деле оба except-блока
  `ProviderRateLimitError`/`ProviderBadRequestError` остались дословно продублированы
  (проверил построчно — тексты логов, `escape_markdown`, форматирование сообщения об
  ошибке совпадают на 100%). Это не поведенческий баг (тесты зелёные, дублирование —
  переживший T10 код), но явно запланированный пункт T12c тихо не сделан и нигде не
  зафиксирован как сознательно отклонённый (в отличие, например, от D5/D6/D9 в
  `T12-plan.md`, которые явно документируют решение не сливать).
- **Фикс:** либо вынести общий helper (`_raise_provider_rate_limit_or_bad_request(e,
  bot_language)` или похожий) и вызвать его в обоих except-блоках, либо явно отметить в
  `T12c-plan`/коде, что решено не сливать, и почему.

### WARNING: нет прямого регрессионного теста на `_persist_conversation_context`
- **Файл:** `bot/openai_helper.py:684-702` (новый метод), потребители — `:1266-1268`,
  `:3369-3371`, `:3466-3468`, `:3510-3512`.
- План (T12c C3/C10/C12a, "Персист контекста после изменения") явно требовал добавить в
  `tests/test_openai_helper_session_api.py` кейс, что все 4 вызывающих места одинаково
  персистят контекст (мок `_save_conversation_context`, проверка аргументов) — по мастер-
  плановому правилу "для групп без тестов — сначала тест, потом вынос". Ни один тестовый
  файл не упоминает `_persist_conversation_context` (`grep -rn _persist_conversation_context
  tests/ bot/tests/` — пусто). Корректность вынесенного метода проверена вручную (аргументы
  всех 4 сайтов совпадают по порядку с сигнатурой) и косвенно всем набором тестов (1930
  зелёных, многие из них многократно проходят через `_add_to_history`/`reset_chat_history`),
  но точечного теста, который целенаправленно ловит будущую регрессию именно в этой точке
  (например, перепутанный порядок `temperature`/`max_tokens_percent` при следующей правке),
  нет.
- **Фикс:** добавить кейс в `tests/test_openai_helper_session_api.py` (или
  `tests/test_per_conversation_serialization.py`) — мок `_save_conversation_context`,
  вызов каждого из 4 путей, проверка `call_args` на позиционные аргументы в ожидаемом
  порядке.

### NIT: пункты "Session-load/reset-check" и "Vision function-call handling" (T12c
  C3/C10/C12a) не реализованы и не задокументированы как сознательно отклонённые
- **Файл:** `bot/openai_helper.py:1003-1026` / `:2228-2239` / `:3484-3504` (3 копии
  session-load/reset-check) и `:2488-2507` / `:2596-2617` (2 копии vision
  function-call handling).
- План прямо предупреждал перечитать все копии целиком перед вырезкой, т.к. это граница с
  T11-стейт-кодом. При перечитывании обнаружил, что копии действительно не идентичны:
  `_get_chat_response_stream_locked` не содержит `self.__max_age_reached(state_key)` в
  условии перезагрузки (два других места содержат), а `_interpret_images_locked`/
  `_interpret_image_stream_locked` расходятся в способе учёта токенов и в
  return-vs-yield-протоколе после `is_direct_result`. Вынос общего helper в обоих случаях
  либо потребовал бы параметра "учитывать ли max_age"/callback для разного
  выхода из функции, либо реально изменил бы поведение одной из копий — то есть решение не
  сливать выглядит обоснованным, но нигде не написано, что это осознанный пропуск, а не
  недосмотр (в отличие от `T12-plan.md` D5/D6/D9, которые явно документируют такие решения).
- **Фикс (опционально):** короткий комментарий у одной из копий (например, у
  `_get_chat_response_stream_locked`) в духе "дубль с `__common_get_chat_response_vision`/
  `record_plugin_exchange`, не объединено — единственное расхождение в условии
  (max_age) достаточно для отдельного helper с параметром, не признано целесообразным
  для ~10 строк (T12c)". Риска нет, чисто трассируемость решения для будущих правок.

### NIT: избыточное строковое экранирование аннотации возврата
- **Файл:** `bot/chat_run.py:47`.
- `async def _retry_after_empty_response(...) -> "tuple[Any | None, tuple]":` — аннотация
  обёрнута в строку, хотя файл уже начинается с `from __future__ import annotations`
  (`bot/chat_run.py:1`), которое и так откладывает вычисление всех аннотаций. Двойное
  экранирование ничего не меняет в рантайме, просто не соответствует стилю остального
  файла (нигде больше в этих 4 файлах строковые аннотации возврата не используются).
- **Фикс:** `-> tuple[Any | None, tuple]:` без кавычек.

## Раунд 2

Проверены исправления обоих WARNING и обоих NIT из раунда 1.

### WARNING 1 — Provider error log — исправлено

`bot/openai_helper.py:1270-1283` добавляет `_raise_provider_rate_limit_or_bad_request(
self, e, bot_language) -> None`: `isinstance(e, ProviderRateLimitError)` →
`logger.warning("Rate limit error error=%s", ...)` + `text_key = 'error'`; иначе
(`ProviderBadRequestError`) → `logger.error("Bad request error error=%s", ...)` +
`text_key = 'openai_invalid'`; оба пути заканчиваются одинаковым
`raise Exception(f"⚠️ _{localized_text(text_key, bot_language)}._ ⚠️\n{error_message}")
from e`. Оба потребителя (`_common_get_chat_response:1500-1503`,
`__common_get_chat_response_vision:2327-2330`) заменили дублирующиеся except-блоки
на вызов этого метода.

Сверка с `git diff HEAD` (база `08bc457`, т.е. до T10 *и* T12c) и с описанием
T10-review/T10-plan подтверждает поведенческую идентичность обеим исходным копиям:
`T10-plan.md` явно предписывал T10 добавить **одинаковую** обёртку в обе функции —
"`_common_get_chat_response`... after: `except ProviderRateLimitError as e:
logger.warning(...); ...; raise Exception(...localized_text('error'...))`" и
"`__common_get_chat_response_vision`... same transformation — `except
openai.RateLimitError as e: raise e` gets the same wrap". То есть после T10 (и до
T12c) обе функции уже содержали побайтово одинаковые `RateLimitError`/
`BadRequestError`-блоки (тот же лог-текст, тот же уровень, тот же i18n-ключ) — это
подтверждено и T10-review.md ("Оба места обработки ошибок ... ловят
ProviderRateLimitError/ProviderBadRequestError ... у обоих теперь есть понятное
сообщение пользователю"). Слияние в `_raise_provider_rate_limit_or_bad_request`
корректно, поведение не поменялось ни для одного из двух call site.

Новые тесты в `tests/test_openai_helper_tool_calls.py` закрывают все 4 комбинации
(до раунда 1 была только реализация для non-stream rate-limit, без прямых тестов
на обёртку сообщения):
- `test_get_chat_response_bad_request_wraps_user_facing_message` (:823) — новый.
- `test_vision_chat_response_rate_limit_wraps_user_facing_message` (:840) — новый,
  и заодно закрывает пробел, о котором раунд-1 ревью не знало: до T12c у vision-пути
  `RateLimitError` **не было** `logger.warning` вовсе (`except openai.RateLimitError
  as e: raise e`, без лога) — план T10 явно требовал добавить туда тот же лог+wrap,
  так что этот тест впервые фиксирует, что оба лог-вызова (common и vision) реально
  идентичны, а не только по факту чтения кода.
- `test_vision_chat_response_bad_request_wraps_user_facing_message` (:861) — новый.
- Оба новых теста на `__common_get_chat_response_vision` используют реальные
  `openai.RateLimitError`/`openai.BadRequestError`, построенные через
  `_fake_rate_limit_error`/`_fake_bad_request_error` (:291-308, тоже новые
  хелперы) с настоящими `httpx.Response(429/400, ...)` — так же, как provider
  реально ловит ошибки через `isinstance`-перевод `OpenAICompatibleProvider`, не
  через мок класса ошибки.

### WARNING 2 — нет теста на `_persist_conversation_context` — исправлено

`tests/test_t12c_persist_conversation_context.py` (новый файл, 4 теста) мокает
`helper._save_conversation_context` и по очереди прогоняет все 4 вызывающих места:
`_maybe_apply_auto_chat_mode`, `reset_chat_history`, `_add_to_history`,
`record_plugin_exchange`; каждый тест проверяет `mock_save.assert_awaited_once_with(
chat_id, {"messages": ...}, parse_mode, temperature, max_tokens_percent, session_id)`
с позиционными аргументами в ожидаемом порядке — ровно то, что требовал план и
WARNING 2. Тесты также фиксируют не совсем очевидные детали поведения
(`reset_chat_history` использует переданный `session_id`, а `_add_to_history`/
`record_plugin_exchange` — session_id, вернувшийся из БД), что делает тест более
ценным регрессионным якорем, чем минимально требовалось.

### NIT — Session-load/reset-check / Vision function-call handling не
задокументированы как сознательный пропуск

Не тронуто (комментарий не добавлен) — это был опциональный NIT ("Фикс
(опционально)"), не блокирующий. Не считаю это невыполненным замечанием.

### NIT — строковая аннотация в `chat_run.py:47` — исправлено

`bot/chat_run.py:47`: `async def _retry_after_empty_response(...) -> tuple[Any |
None, tuple]:` — кавычки убраны, теперь соответствует стилю остального файла
(аннотация без кавычек, `from __future__ import annotations` уже отложенно
вычисляет её).

### Тесты и статический анализ

- `~/.venvs/ctb/bin/python -m pytest tests/test_t12c_artifact_path_dedup.py
  tests/test_t12c_chat_run_retry.py tests/test_t12c_persist_conversation_context.py
  tests/test_compat_state_views_guard.py tests/test_agent_tools_plugin.py
  tests/test_openai_helper_tool_calls.py -q` — 297 passed вместе с остальными
  T12-тестами (T12a/T12b/T12d в этом же прогоне, см. общий отчёт координатору).
- Полный `tests/ bot/tests/` — 1944 passed, 0 failed, 0 регрессий (было 1930 в
  раунде 1 — рост согласуется с добавленными в раунде 2 тестами по всем трём
  T12-веткам).
- `ruff check bot/openai_helper.py bot/openai_tool_handler.py bot/chat_run.py
  bot/tool_result.py tests/test_t12c_*.py` — чисто.
- mypy (`(файл, код)`-сравнение с `/tmp/impl/mypy_before.keep`): в области нового
  `_raise_provider_rate_limit_or_bad_request` (`bot/openai_helper.py:1270-1290`,
  `:1495-1510`, `:2320-2335`) и в `chat_run.py` рядом с `_retry_after_empty_response`
  ошибок нет; единственная оставшаяся ошибка `chat_run.py` (`var-annotated`,
  `usage_accumulator`) — тот же пред-существующий пункт, что был в базовой линии
  (`bot/chat_run.py:71` → сейчас `:92`, сдвиг строки из-за вставки нового метода
  раньше по файлу, не новая ошибка). По всем 4 owned-файлам T12c ни одного нового
  кода ошибки не появилось; наблюдаемые уменьшения — параллельная T13
  (аннотации), не T12c.

### Итог раунда 2

0 ERROR, 0 WARNING, 1 NIT остаётся не сделан (опциональный, явно помечен
"опционально" в раунде 1 — комментарий-трассировка решения не сливать
session-load/vision-copies; риска нет). Оба WARNING и второй NIT устранены
корректно, с тестами, без регрессий.
