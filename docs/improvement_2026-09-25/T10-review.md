# T10. Единый интерфейс провайдера — review

## Раунд 1

**Вердикт: ERROR 0, WARNING 2, NIT 0.** Реализация точно соответствует
`T10-plan.md` и мастер-плану (`00-master-plan.md:340-367`): новые классы ошибок,
перевод `openai.*` → `Provider*` на каждом пути (non-stream, стрим-создание,
mid-stream итерация, images/audio/models), понятное сообщение пользователю
после исчерпания ретраев SDK, `max_retries=3` на клиенте, удалён ручной
retry-цикл и константы `LLM_RATE_LIMIT_RETRY_*`, `plugin_tool_adapter.py` +
его тест удалены, AST-guard добавлен и реально ловит нарушения. Полный набор
тестов и целевые тесты зелёные, `ruff` чистый, новых `mypy`-ошибок нет.

### Проверенные файлы (diff HEAD, только хунки в зоне владения T10)

- `bot/ai_provider.py`: добавлены `ProviderError`/`ProviderRateLimitError`/
  `ProviderBadRequestError`/`ProviderStreamError` + 6 методов в `AIProvider`
  Protocol — соответствует плану §2.1.
- `bot/ai_providers/openai_compatible.py`: `build_openai_client` (перенесён
  дословно, `max_retries: 3` на месте), `_translate`, `_translate_stream_errors`
  (с пробросом `aclose()` в `finally` — важно для корректного закрытия
  `_AIProviderStreamProxy`), `raw_chat_completion(get_client)` (клиент читается
  через аксессор при каждом вызове — не захватывается один раз, что и
  требовалось для тестов, переопределяющих `helper.client` после
  конструктора). `OpenAICompatibleProvider` расширен `get_client`/
  `get_gateway_client` + `generate_image`/`list_models`/`speech`/`transcribe`
  (с переводом ошибок) и `edit_image`/`list_voices` (gateway-backed, без
  перевода — по плану не требуется). Существующие
  `stream_response`/`create_response`/`_response_events`/`_streaming_events`
  не тронуты.
- `bot/ai_providers/fake.py`: добавлены parity-методы
  (`generate_image`/`edit_image`/`speech`/`transcribe`/`list_models`/
  `list_voices` + `queue_*`), по образцу `queue_text`/`_event_batches`.
- `bot/openai_helper.py` (только хунки провайдера/обработки ошибок —
  остальные хунки в этом файле принадлежат T08/T09, не проверялись по
  существу):
  - `import openai` удалён, добавлены импорты `ProviderBadRequestError`,
    `ProviderRateLimitError`, `OpenAICompatibleProvider`, `build_openai_client`,
    `raw_chat_completion`.
  - `LLM_RATE_LIMIT_RETRY_ATTEMPTS`/`LLM_RATE_LIMIT_RETRY_WAIT_SECONDS`
    удалены; grep по всему дереву (кроме исторических доков) подтвердил ноль
    оставшихся ссылок.
  - `__init__`: `self.client = build_openai_client(...)`, `self._provider =
    OpenAICompatibleProvider(raw_chat_completion(lambda: self.client), ...,
    get_client=lambda: self.client, get_gateway_client=lambda:
    self.gateway_client)` — провайдер создаётся один раз, но клиент читается
    через замыкание (не снапшотится), поэтому ~90 тестов, переопределяющих
    `helper.client` после конструктора, не сломались (подтверждено прогоном).
  - `_create_chat_completion_with_rate_limit_retry` удалён целиком; ручного
    `asyncio.sleep` в LLM-пути больше нет (проверено grep'ом всех
    `asyncio.sleep` в `bot/` — остальные вхождения относятся к
    Telegram-поллингу/фоновым задачам/другим плагинам, не к LLM-запросам).
  - `_timed_create`: единственная замена — вызов через
    `self._provider.create_response(...)` вместо удалённого метода; логика
    таймингов/`llm_call`/`llm_error`/статистики не тронута.
  - `generate_image`/`get_available_tts_models`/`generate_speech`/
    `transcribe` переведены на `self._provider.*`; добавлен
    `raw_generate_image` — используется `stable_diffusion.py`, потому что
    `generate_image()` возвращает `(value, size)`, а не сырой
    url/b64_json-объект, нужный `extract_image_result()` (обоснование в
    T10-plan §2.6 подтверждено чтением обоих методов).
  - `edit_telegram_image`/`get_available_tts_voices` **оставлены** на прямом
    `self.gateway_client.*` — план явно разрешил это как отдельно взятую
    опцию (§2.2/§2.4/§6.1), не требуется AST-guard'ом (`gateway_client` не
    матчит `.client.`). Не находка.
  - Оба места обработки ошибок (`_common_get_chat_response:~1476-1484`,
    `__common_get_chat_response_vision:~2307-2315`) ловят
    `ProviderRateLimitError`/`ProviderBadRequestError` вместо `openai.*`; у
    обоих теперь есть понятное сообщение пользователю на исчерпание
    ретраев (раньше — голый `raise e` без обёртки, это и был баг из плана).
    Переиспользован существующий i18n-ключ `'error'` (подтверждено, что
    ключ есть в `translations.json`, отдельный ключ не нужен). Универсальный
    `except Exception` ниже по обоим методам подхватывает базовый
    `ProviderError` (не rate-limit/bad-request) тем же понятным сообщением —
    пробелов в покрытии 429/400/прочих `APIError` нет.
- `bot/openai_tool_handler.py` (только error-handling хунки — остальное,
  `DANGEROUS_TOOL_NAMES`/`_tainted_plugin_ids`, это T08, не проверялось): импорт
  `openai` заменён на `from .ai_provider import ProviderStreamError`; оба
  сайта (`:1246`, `:1306`) ловят `ProviderStreamError` вместо
  `openai.APIError`.
- `bot/plugins/stable_diffusion.py`: единственная строка —
  `helper.client.images.generate(...)` → `helper.raw_generate_image(...)`.
  `_edit_image` (`helper.gateway_client.image_edit`) не тронут — верно, это
  не `.client.` и не входит в guard.
- `bot/plugin_tool_adapter.py` + `tests/test_plugin_tool_adapter.py` удалены.
  Проверено по всему дереву (код, `AGENTS.md`, `README*`, codebase map
  `.cli-proxy/.codebase_map/*`): ссылки остались только в исторических
  доках (`docs/audit_remediation_plan_2026-09-04.md`,
  `docs/architecture_code_review_2026-09-04.md` — датированные документы вне
  зоны T10) и в докстринге `bot/ai_events.py:29`, который план explicitly
  оставляет как есть (не владение T10).

### AST-guard (`tests/test_ast_no_raw_openai_access.py`, новый файл)

Ловит `import openai`/`from openai...` и `X.client.Y`-паттерн везде под
`bot/`, кроме `bot/ai_providers/`; allow-list с обоснованием для
`OpenAIHelper.close()` (cleanup), `bot/net_safety.py` (`http.client.*` —
stdlib, не SDK) и `hindsight_memory.py` (`self.client` = `HindsightClient`).
Прогнан вручную по всему дереву `bot/` (инлайн-скрипт на `.client.`/`openai.`) —
подтверждено, что список найденных небезопасных мест совпадает с
allow-list'ом 1-в-1, никаких пропущенных нарушений. Тест проходит.

### Тесты и линт

```
~/.venvs/ctb/bin/python -m pytest tests/test_ai_provider.py \
  tests/test_openai_compatible_provider.py \
  tests/test_ast_no_raw_openai_access.py -q --no-header -p no:cacheprovider
# 22 passed

~/.venvs/ctb/bin/python -m pytest tests/test_openai_helper_tool_calls.py \
  -q --no-header -p no:cacheprovider
# 133 passed

~/.venvs/ctb/bin/python -m pytest tests/ bot/tests/ -q --no-header -p no:cacheprovider
# 1824 passed
```

`ruff check` по всем изменённым файлам — чисто. `mypy` на 6 файлах T10:
139 ошибок всего, но при сопоставлении с `/tmp/impl/mypy_before.keep` по
файлу+тексту сообщения (номера строк не совпадают из-за сдвигов от
параллельных задач в тех же файлах) — 29 ошибок в зоне T10 и до, и после,
28 «новых» строк 1-в-1 совпадают текстом с 28 «исчезнувшими» (тот же долг,
просто сдвинулся). **Новых mypy-ошибок от T10 нет.**

Целевые тесты из плана §5 присутствуют и корректны:
`test_timed_create_does_not_retry_rate_limit_manually` (1 вызов, без sleep,
`ProviderRateLimitError`), `test_get_chat_response_rate_limit_does_not_duplicate_user_message`
(обёрнутое сообщение, `client.calls == 1`), оба streaming-теста на
`ProviderStreamError` вместо monkeypatch `openai.APIError`, error-translation
тесты в `test_openai_compatible_provider.py` (rate-limit/bad-request/generic
API error + mid-stream), fake-provider round-trip тесты в
`test_ai_provider.py` для всех 6 новых методов.

### WARNING

1. **`tests/test_openai_compatible_provider.py` —
   `test_openai_compatible_provider_aggregates_streamed_tool_calls` потерял
   последнюю строку при правке.** В HEAD файл заканчивался строкой
   `assert response.tool_calls[0].arguments == '{"name":"pptx"}'` — она
   проверяла, что дельты аргументов стримингового tool-call'а
   (`'{"name"'` + `':"pptx"}'`) корректно склеиваются. В текущем diff эта
   строка удалена (`git diff HEAD` показывает `-    assert
   response.tool_calls[0].arguments == '{"name":"pptx"}'` без замены), и
   подстрока `'{"name":"pptx"}'` больше нигде в файле не встречается —
   подтверждено прямым поиском по файлу. Похоже на случайную потерю строки
   при дописывании новых тестов в конец файла (эта правка вообще не входит в
   заявленную T10-work — единственная причинно-следственная связь с задачей
   в том, что дописывались тесты error-translation в конец того же файла).
   Тест по-прежнему проходит (просто проверяет меньше), регрессии в проде
   нет, но защита от будущей регрессии в сборке аргументов стримингового
   tool-call'а пропала. **Исправление:** вернуть удалённую строку в конец
   `test_openai_compatible_provider_aggregates_streamed_tool_calls`.

2. **Нет теста на новую обёртку сообщения для vision-пути при исчерпании
   rate-limit.** `bot/openai_helper.py` (`__common_get_chat_response_vision`,
   `except ProviderRateLimitError`, ~строка 2307) получил точно такое же
   исправление, как и `_common_get_chat_response` (было — голый `raise e`
   без сообщения пользователю, стало — обёрнутое сообщение), но
   `test_get_chat_response_rate_limit_does_not_duplicate_user_message`
   (обновлён для не-vision пути) не имеет пары для vision-варианта — поиск
   по `tests/` не нашёл теста, комбинирующего vision-путь и rate-limit.
   Риск невысокий (код идентичен уже протестированному чат-пути, изменение
   симметричное и прочитано построчно), но это единственная часть плана
   §2.4 без прямого теста, а именно эта строка раньше была багом (голый
   `raise e`) — стоит закрыть тестом, чтобы будущий рефакторинг не вернул
   голый `raise e` незаметно для CI.

Обе находки не блокируют — ни одна не меняет наблюдаемое поведение
production-кода и не нарушает план; это пробелы в тестовом покрытии,
которые стоит закрыть на исправлении.

## Раунд 2

**Вердикт: ERROR 0, WARNING 0, NIT 0.** Обе находки раунда 1 закрыты тестами,
и обе меняющие тест-файлы проверены содержательно (reasoning, без правки
кода): каждая упала бы при откате соответствующего исправления. Область
проверки — только два файла: `tests/test_openai_compatible_provider.py`
(W1) и `tests/test_openai_helper_tool_calls.py` (W2). T11 параллельно
редактирует `bot/openai_helper.py` и др. — эти правки не рецензировались.

### W1 — восстановленная ассерция (`tests/test_openai_compatible_provider.py`)

`git diff HEAD -- tests/test_openai_compatible_provider.py` показывает
`test_openai_compatible_provider_aggregates_streamed_tool_calls` целиком как
контекст (без `-`/`+` внутри функции) — тело функции, включая последнюю
строку `assert response.tool_calls[0].arguments == '{"name":"pptx"}'`,
побайтово совпадает с `git show HEAD:tests/test_openai_compatible_provider.py`
(сверено напрямую). Строка на месте, потери больше нет.

Содержательность подтверждена чтением продакшен-кода
(`bot/ai_providers/openai_compatible.py:283-307`,
`_ToolCallStreamBuilder.add_delta`/`build`): дельты аргументов стримингового
tool-call'а копятся в `arguments_parts` и собираются через
`"".join(self.arguments_parts)`. Тест кормит два чанка —
`'{"name"'` и `':"pptx"}'` — и ассерция требует результат `'{"name":"pptx"}'`.
Если бы склейка была сломана (например, `build()` брал только последний
элемент списка вместо `"".join(...)`), результат был бы `':"pptx"}'` и
ассерция упала бы. Тест реально ловит регрессию в сборке аргументов, а не
проходит тривиально.

### W2 — новый тест vision-пути (`tests/test_openai_helper_tool_calls.py`,
`test_vision_chat_response_rate_limit_wraps_user_facing_message`, ~:811)

Прослежена вся цепочка вызовов, чтобы подтвердить, что тест действительно
доходит до строки, которая была багом (`except ProviderRateLimitError as e:`
в `__common_get_chat_response_vision`, `bot/openai_helper.py:2307-2310`):
`_create_chat_response_completion(kind='vision', ...)` (`:651`) →
`_timed_create_via_ai_provider` (`:578`, non-stream ветка, т.к. `stream`
не передан методу и по умолчанию `False`) → внутренний
`OpenAICompatibleProvider(create_chat_completion=self._timed_create, ...)`
→ `collect_ai_response(provider.stream_response(request))`. `stream_response`
(`bot/ai_providers/openai_compatible.py:111-119`) не оборачивает исключения
из `create_response` — они всплывают как есть. `self._timed_create` (`:467`)
зовёт `self._provider.create_response(...)` (реальный `_provider`,
собранный в `__init__` через `raw_chat_completion(lambda: self.client)`) и
перевыбрасывает исключение через голый `raise` (`:503`) после логирования.
`raw_chat_completion` (`bot/ai_providers/openai_compatible.py:71-86`) ловит
`openai.APIError` вокруг `client.chat.completions.create(**kwargs)` и
переводит через `_translate` в `ProviderRateLimitError` для
`openai.RateLimitError`. Тест переопределяет `helper.client` уже после
конструктора (`_make_helper`, `:487`) — это безопасно, т.к. `get_client`
читается лениво (`lambda: self.client`), что и подтверждено раундом 1 для
~90 других тестов с тем же паттерном. `_fake_rate_limit_error` строит
настоящий `openai.RateLimitError` (не выдуманный класс), поэтому
`isinstance(exc, openai.RateLimitError)` в `_translate` действительно
срабатывает — так же, как уже проверено для соседнего
`test_get_chat_response_rate_limit_does_not_duplicate_user_message`, тест
собран по тому же образцу.

Итог: `ProviderRateLimitError` доходит без изменений до
`__common_get_chat_response_vision`, где ловится и оборачивается в
`Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{error_message}")`.
Тест проверяет `not isinstance(exc_info.value, ProviderRateLimitError)` и
`"⚠️" in str(exc_info.value)`. Если бы исправление откатили к голому
`raise e` (баг, описанный в раунде 1), `exc_info.value` остался бы
`ProviderRateLimitError` — первая ассерция упала бы. Тест содержательный.

### Тесты и линт (только два файла раунда 2)

```
~/.venvs/ctb/bin/python -m pytest tests/test_openai_compatible_provider.py \
  tests/test_openai_helper_tool_calls.py -q --no-header -p no:cacheprovider
# 143 passed (один прогон, без флейков; падений в зоне T11 не наблюдалось,
# повторный прогон не потребовался)

~/.venvs/ctb/bin/python -m ruff check tests/test_openai_compatible_provider.py \
  tests/test_openai_helper_tool_calls.py
# All checks passed!
```

Код не менялся (ревьюер read-only для кода); правки не вносились.
