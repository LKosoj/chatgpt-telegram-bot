# T15. Один non-stream путь

Источники задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел T15, «Волна 4») и
`docs/architecture_code_review_2026-09-04.md` §5.3, пункт «П1. Закрыть двойной non-stream
путь». Термины по ходу: *non-stream* — режим, когда бот ждёт от модели весь ответ целиком и
только потом показывает его (в отличие от *stream* — показа по кусочкам, по мере генерации);
*feature-флаг* — переключатель в конфиге, который включает/выключает кусок логики без правки
кода; *name mangling* — механизм Python: метод с именем `__foo` внутри класса `OpenAIHelper`
на самом деле хранится под именем `_OpenAIHelper__foo`, и обратиться к нему снаружи класса
можно только по этому длинному имени; *protected-имя* — соглашение Python «имя с одним
подчёркиванием» (`_foo`): технически доступно откуда угодно, но означает «это внутренности
класса, не трогай снаружи без необходимости»; *шлюз (gateway)* — прокси-сервер перед реальным
API модели (например, OpenRouter), через который бот делает запросы.

## Цель

Сейчас нестримовый (non-stream) путь получения ответа от модели реализован дважды:

1. `ChatRun.run_non_stream` (`bot/chat_run.py`) — путь по умолчанию.
2. Копия той же логики внутри `OpenAIHelper._get_chat_response_locked`
   (`bot/openai_helper.py:947-1106`) — legacy-тело, живущее «на всякий случай».

Оба пути включаются одним и тем же флагом конфига `chat_run_variant_b_enabled` (по умолчанию
`True` = используется `ChatRun`). Цель — оставить один путь (`ChatRun`), убрать флаг и
legacy-тело, и одновременно избавить `ChatRun` от обращения к приватным методам `OpenAIHelper`
через name mangling, заменив их на protected-имена. Правки должны не менять наблюдаемое
поведение бота (кроме случаев, где сегодняшнее поведение уже расходится между путями — они
разобраны в разделе «Анализ» ниже).

## Анализ

### Находка 0 (важно): флаг управляет ДВУМЯ развилками, не одной

Аудит и remediation-план описывают `chat_run_variant_b_enabled` как переключатель между
`ChatRun.run_non_stream` и legacy-телом `_get_chat_response_locked`. Это верно, но неполно: тот
же флаг управляет ещё одной, более широкой развилкой — какой способ отправки запроса к модели
использовать вообще, независимо от stream/non-stream и от `ChatRun`:

```python
# bot/openai_helper.py:690-693
async def _create_chat_response_completion(self, *, kind, **kwargs):
    if self.config.get('chat_run_variant_b_enabled', True):
        return await self._timed_create_via_ai_provider(kind=kind, **kwargs)
    return await self._timed_create(kind=kind, **kwargs)
```

`_create_chat_response_completion` — это единственная точка входа, через которую ВСЕ вызовы
модели (не только non-stream-оркестрация) идут наружу: `chat_completion()` (низкоуровневый
метод, `bot/openai_helper.py:442-487`), `__common_get_chat_response` (non-stream и stream),
`__common_get_chat_response_vision` (vision), `ModelUtilities.classify_json`/`summarize_window`
(через `chat_completion`) и так далее. Значит флаг `chat_run_variant_b_enabled` при `False`
откатывает не только «ChatRun → legacy-тело», но и «обёртка AIProvider (`_timed_create_via_
ai_provider`, `bot/openai_helper.py:617-688`) → сырой SDK-вызов (`_timed_create`,
`bot/openai_helper.py:506-583`)» — причём это откатывается для streaming-пути
(`_get_chat_response_stream_locked`, `bot/openai_helper.py:1183-1330`), vision-путей
(`interpret_image`/`interpret_image_stream`) и `classify_reply_intent`/`_summarise_window`
тоже. Тесты `tests/test_openai_helper_tool_calls.py::test_interpret_image_can_roll_back_to_
legacy_timed_create` и `::test_chat_response_stream_can_roll_back_to_legacy_timed_create` это
прямо проверяют — и они вообще не трогают `ChatRun`/`_get_chat_response_locked`.

Практическое следствие для этой задачи: чтобы закрыть T15 «по-настоящему» (не оставить второй
мёртвый флаг), нужно убрать ОБЕ развилки за один проход — они управляются одной конфиг-опцией,
поэтому удаление ключа `chat_run_variant_b_enabled` закрывает обе сразу, но код в двух местах
(`bot/openai_helper.py:935-945` и `bot/openai_helper.py:691-693`) нужно исправить отдельно.
`_timed_create` как функция не удаляется — она остаётся внутренней «рабочей лошадкой» под
`_timed_create_via_ai_provider` (вызывается из неё на `bot/openai_helper.py:623`), пропадает
только прямой обход обёртки.

### Находка 1: построчное сравнение legacy-тела и `ChatRun.run_non_stream`

Оба тела вызывают один и тот же общий код — `__common_get_chat_response` (генерация ответа,
подрезка истории и т.д.) и `__handle_function_call` (тот же модуль `bot/openai_tool_handler.py`
за обёрткой) — так что «ядро» идентично. Расхождения есть только в обвязке
(`bot/openai_helper.py:947-1106` — legacy; `bot/chat_run.py:47-263` — `ChatRun`):

| # | Место | Legacy (`_get_chat_response_locked`) | `ChatRun.run_non_stream` | Кто «правее» |
|---|---|---|---|---|
| 1 | Определение `allowed_plugins` для `__handle_function_call` | `bot/openai_helper.py:967`: `resolve_allowed_plugins(chat_id, session_id, user_id)` — `user_id` без подстановки | `bot/chat_run.py:78-79`: `plugin_user_id = user_id or chat_id`, затем `resolve_allowed_plugins(chat_id, session_id, plugin_user_id)` | `ChatRun`. Первый (более ранний) вызов `resolve_allowed_plugins` внутри общего `__common_get_chat_response` уже использует подстановку `memory_user_id = kwargs.get('user_id') or chat_id` (`bot/openai_helper.py:1619`) — им строится список тулов, отправляемый модели. Legacy-тело при повторном резолве для реального вызова тула подстановку теряет, из-за чего при `user_id=None` список разрешённых плагинов для расчёта тулов и для их исполнения может отличаться (`disabled_plugins_for_user(None)` против `disabled_plugins_for_user(chat_id)`, см. `bot/plugin_manager.py:205-209`). На практике `user_id` из Telegram-апдейтов почти никогда `None` (все три вызывающих места в `bot/telegram_bot.py:2805, 4530, 4987` передают реальный `user_id`) — риск теоретический, но `ChatRun` уже ведёт себя как vision-путь (`bot/openai_helper.py:2735, 2872`, тоже с `user_id or chat_id`), то есть более единообразно. |
| 2 | Событие `AIRunEnd`/`AIRetry` в трейс сессии (`session_logger`) | Не пишет `run_end`/`retry` события вообще — только `turn_start`/`assistant_response` на уровне внешнего `get_chat_response` (`bot/openai_helper.py:882-908`, общий для обоих путей) | Пишет `AIRunEnd(reason=...)` на каждый выход (`direct_result`, `direct_result_after_retry`, `completed`, `error`) и `AIRetry` перед каждым retry-звонком (`bot/chat_run.py:33-45, 92-181, 257-263`) | `ChatRun` — это чистое дополнение (телеметрия для отладки), не влияет на возвращаемое значение. Тест `test_chat_run_variant_b_logs_retry_and_run_end` (`tests/test_openai_helper_tool_calls.py:2084`) закрепляет это поведение только для `ChatRun`. |
| 3 | Обработка `response.usage is None` в блоке `show_usage` | `bot/openai_helper.py:1093-1096`: `if total_tokens == usage_tokens: answer += f"...{response.usage.prompt_tokens}..."` — если `response.usage is None`, а `total_tokens == usage_tokens == 0`, будет `AttributeError` | `bot/chat_run.py:240-251`: та же проверка плюс `usage is not None and usage.prompt_tokens is not None and usage.completion_tokens is not None` | `ChatRun` — это фикс T06 (`docs/remediation_2026-09-04/T06-usage-none.md`), уже перенесённый в `ChatRun` и в vision-путь (`bot/openai_helper.py:2801-2807`), но не в legacy-тело. Отдельно переносить фикс в legacy не нужно — оно целиком удаляется этим тикетом. |
| 4 | Импорт вспомогательных функций (`response_has_message_text` и т.д.) | Модульные алиасы с подчёркиванием: `_response_has_message_text` и др. (`bot/openai_helper.py:63-67`, `from .chat_response_utils import ... as _response_has_message_text`) | Прямой импорт тех же функций без алиаса (`bot/chat_run.py:7-14`) | Разницы в поведении нет — это буквально одни и те же функции из `bot/chat_response_utils.py` под разными именами. |
| 5 | Всё остальное (порядок retry-веток, работа с `token_accumulator`/`usage_accumulator`, `n_choices > 1`, `show_plugins_used`) | идентично | идентично | — |

Итог: `ChatRun` уже является функциональным надмножеством legacy-тела (плюс телеметрия, плюс
уже перенесённый фикс T06, плюс более последовательная подстановка `user_id or chat_id`).
Переносить в `ChatRun` перед удалением legacy **нечего** — можно сразу удалять.

Дополнительно проверены `_repair_tool_call_history` и `record_plugin_exchange`, упомянутые в
задаче: они лежат **вне** развилки. `_repair_tool_call_history` вызывается из
`_apply_before_chat_request_mutators` (`bot/openai_helper.py:1465-1466`), которая, в свою
очередь, вызывается изнутри общего `__common_get_chat_response` — то есть выполняется
одинаково для обоих путей до того, как расходится обвязка. `record_plugin_exchange`
(`bot/openai_helper.py:3530`) — это вообще отдельный механизм для плагинов-обработчиков
промптов (RAG и т.п.), которые не проходят через `get_chat_response` в принципе; вызывается из
`bot/telegram_bot.py:4731`, и `ChatRun`/legacy-тело её не касаются.

### Находка 2: все ссылки на флаг `chat_run_variant_b_enabled`

Конфиг/код:
- `bot/__main__.py:214` — чтение `CHAT_RUN_VARIANT_B_ENABLED` в env (`parse_bool_env`,
  определена в `bot/__main__.py:23`), по умолчанию `True`.
- `bot/openai_helper.py:340` — `self.config.setdefault('chat_run_variant_b_enabled', True)`
  в конструкторе `OpenAIHelper`.
- `bot/openai_helper.py:691` — развилка в `_create_chat_response_completion` (Находка 0).
- `bot/openai_helper.py:935` — развилка в `_get_chat_response_locked` (`ChatRun` vs legacy-тело).

Документация:
- `README.md:293`, `README.ru.md:301` — строка таблицы env-переменных.
- `.env.example:44` — закомментированная строка `# CHAT_RUN_VARIANT_B_ENABLED=true`.

Прочее (не тесты, не в `tests`/`bot/tests`):
- `evals/judge/turn_runner.py:149` — жёстко прописывает `"chat_run_variant_b_enabled": True` в
  конфиге для LLM-judge прогонов. `evals/` не часть тестового прогона (см. AGENTS.md), но
  ключ, которого не будет в схеме конфига, стоит убрать для чистоты.
- `docs/ai_provider_variant_b_plan_2026-07-05.md` — исторический план внедрения. Не трогать
  (документ фиксирует историю решения, см. правило «Documentation Rules» в AGENTS.md).

Тесты — см. раздел «Тесты» ниже (полный список с классификацией «удалить/переписать/убрать
строку»).

### Находка 3: name-mangled приватные методы, к которым обращается `ChatRun`

`ChatRun` — не метод класса `OpenAIHelper`, а отдельный объект, которому передают `helper`
(`self.helper = helper`, `bot/chat_run.py:31`). Чтобы вызвать методы `OpenAIHelper`, у которых
имя начинается с двух подчёркиваний, `ChatRun` вынужден писать их «взорванные»
(name-mangled) имена:

| Приватный метод (объявлен в `OpenAIHelper`) | Определение | Вызовы из `ChatRun` (mangled-имя) | Вызовы внутри `OpenAIHelper` (`self.__x`, не mangled — компилятор сам взрывает имя) |
|---|---|---|---|
| `__common_get_chat_response` | `bot/openai_helper.py:1463` | `bot/chat_run.py:64` (`helper._OpenAIHelper__common_get_chat_response`) | `bot/openai_helper.py:953` (legacy non-stream, будет удалён), `:1219` (stream-путь, остаётся) |
| `__handle_function_call` | `bot/openai_helper.py:1721` (тонкая обёртка, делегирует в модульную `handle_function_call()` из `bot/openai_tool_handler.py`) | `bot/chat_run.py:80, 110, 158` | `bot/openai_helper.py:968, 993, 1032` (legacy non-stream, удаляются), `:1237` (stream), `:2741, 2878` (vision) |
| `__add_to_history` | `bot/openai_helper.py:3515` | `bot/chat_run.py:207, 218` | `bot/openai_helper.py:834, 845` (`ask()`), `:1076, 1082` (legacy non-stream, удаляются), `:1282` (stream), `:1508` (внутри `__common_get_chat_response`), `:2506` (vision), `:2785, 2791` (vision text), `:2913` (vision stream) |

Не входит в переименование: `__common_get_chat_response_vision` (`bot/openai_helper.py:2464`)
— `ChatRun` её не использует (vision — отдельный, не дублированный путь), трогать её вне
объявленного в задаче охвата не нужно (лишний диф).

Помимо трёх приватных методов, `ChatRun` также читает и пишет уже protected-поля helper'а
напрямую: `helper._gate_fired`, `helper._chat_request_extra_tokens`,
`helper._chat_request_models`, `helper._chat_request_usage_split`, `helper._chat_state_key(...)`
(`bot/chat_run.py:59-260`). Это НЕ name mangling (одно подчёркивание уже сегодня доступно
снаружи без «взрыва» имени) и не входит в переименование этой задачи — это отдельная,
более крупная архитектурная тема (публичный API состояния сессии, П4 в
`docs/architecture_code_review_2026-09-04.md`, идёт после П1 по плану волны 3). Здесь только
фиксирую как «замечено, не в этом тикете».

## Дизайн

Два отдельных шага, которые можно и нужно делать порознь (первый почти без риска, второй —
удаление кода и тестов):

**Шаг A — переименование приватных методов (protected вместо name-mangled).**
Просто убрать одно подчёркивание из тройки `__common_get_chat_response` → `_common_get_chat_
response`, `__handle_function_call` → `_handle_function_call`, `__add_to_history` → `_add_to_
history`. Это чисто механическая правка: Python сам не меняет поведение метода при переходе
с двух подчёркиваний на одно — меняется только то, что имя больше не «взрывается»
(`_OpenAIHelper__foo` → `_foo`). Нужно поправить:
1. Три `def` (объявления методов).
2. Все внутренние вызовы `self.__foo(...)` → `self._foo(...)` (10 сайтов вызова, таблица выше).
3. `ChatRun` — заменить `helper._OpenAIHelper__foo(...)` на `helper._foo(...)` (6 сайтов).
4. Тесты, которые сегодня обращаются к mangled-имени напрямую (полный список — раздел «Тесты»).

Эту правку можно сделать и оставить в отдельном коммите ДО удаления флага — тесты должны
пройти без единого изменения поведения, только имена меняются. Это и есть «выровнять
поведение» из формулировки задачи, применительно к именам методов: здесь выравнивать нечего
в логике (Находка 1 показала, что `ChatRun` уже строго не хуже legacy), но привести
именование к общему знаменателю нужно до того, как убирать флаг — иначе диф удаления флага
будет свален в одну кучу с переименованием и станет труднее ревьюить.

**Шаг B — убрать флаг и оба legacy-пути.**
1. `bot/openai_helper.py:935-945` — заменить
   ```python
   if self.config.get('chat_run_variant_b_enabled', True):
       from .chat_run import ChatRun
       return await ChatRun(self).run_non_stream(...)
   ```
   на безусловный вызов (тот же `return await ChatRun(self).run_non_stream(...)`, без `if`).
2. Удалить весь legacy-хвост `_get_chat_response_locked` — `bot/openai_helper.py:947-1106`
   (всё, что раньше выполнялось только при `chat_run_variant_b_enabled=False`).
3. `bot/openai_helper.py:690-693` — заменить `_create_chat_response_completion` на
   безусловный `return await self._timed_create_via_ai_provider(kind=kind, **kwargs)`, убрать
   `if`/`else`-ветку с `self._timed_create(kind=kind, **kwargs)`. Сам метод `_timed_create`
   (`bot/openai_helper.py:506-583`) не трогать — он остаётся нужен как внутренняя реализация
   `_timed_create_via_ai_provider` (`bot/openai_helper.py:623`).
4. `bot/openai_helper.py:340` — удалить `self.config.setdefault('chat_run_variant_b_enabled',
   True)`.
5. `bot/__main__.py:214` — удалить строку `'chat_run_variant_b_enabled':
   parse_bool_env('CHAT_RUN_VARIANT_B_ENABLED', True),`.
6. `bot/chat_run.py:22-28` — поправить docstring класса `ChatRun`: убрать фразу «The legacy
   SDK-shaped path remains behind the Variant B rollback flag» (пути для отката больше нет).
7. Документация: убрать строки про `CHAT_RUN_VARIANT_B_ENABLED` из `README.md:293`,
   `README.ru.md:301`, `.env.example:44`.
8. `evals/judge/turn_runner.py:149` — убрать ключ `"chat_run_variant_b_enabled": True,` из
   словаря конфига (косметика, вне тестового прогона).
9. Тесты — см. ниже.

Альтернативы, рассмотренные и отклонённые (как и в самом П1):
- *Legacy делегирует в `ChatRun`, флаг остаётся.* Не даёт закрыть саму задачу — дублирование
  формально исчезает из тела функции, но флаг и мёртвая ветка `_create_chat_response_completion`
  никуда не деваются, а `_get_chat_response_locked` продолжает делать нестандартный `if` там,
  где он больше не нужен.
- *Параметризовать тесты по флагу вместо удаления.* Закрепляет дублирование как «нормальное»
  и оставляет флаг в конфиге навсегда — именно то, от чего предостерегает П1
  (`docs/architecture_code_review_2026-09-04.md`, «Альтернативы»).

## Правки по file:line

Файл создаётся этим тикетом как план; ниже — точный список правок для исполнителя (порядок
соответствует шагам A и B из «Дизайна»).

**Шаг A (переименование, отдельный коммит):**
- `bot/openai_helper.py:1463` — `def __common_get_chat_response` → `def _common_get_chat_
  response`.
- `bot/openai_helper.py:953, 1219` — `self.__common_get_chat_response(` → `self._common_get_
  chat_response(`.
- `bot/openai_helper.py:1721` — `def __handle_function_call` → `def _handle_function_call`.
- `bot/openai_helper.py:968, 993, 1032, 1237, 2741, 2878` — `self.__handle_function_call(` →
  `self._handle_function_call(`.
- `bot/openai_helper.py:3515` — `def __add_to_history` → `def _add_to_history`.
- `bot/openai_helper.py:834, 845, 1076, 1082, 1282, 1508, 2506, 2785, 2791, 2913` —
  `self.__add_to_history(` → `self._add_to_history(`.
- `bot/chat_run.py:64` — `helper._OpenAIHelper__common_get_chat_response(` → `helper._common_
  get_chat_response(`.
- `bot/chat_run.py:80, 110, 158` — `helper._OpenAIHelper__handle_function_call(` → `helper.
  _handle_function_call(`.
- `bot/chat_run.py:207, 218` — `helper._OpenAIHelper__add_to_history(` → `helper._add_to_
  history(`.
- Тестовые файлы — см. «Тесты».

**Шаг B (удаление флага, отдельный коммит после того, как шаг A прошёл тесты):**
- `bot/openai_helper.py:935-945` → безусловный вызов `ChatRun(self).run_non_stream(...)`.
- `bot/openai_helper.py:947-1106` → удалить целиком.
- `bot/openai_helper.py:690-693` → безусловный `return await self._timed_create_via_ai_
  provider(kind=kind, **kwargs)`.
- `bot/openai_helper.py:340` → удалить строку.
- `bot/__main__.py:214` → удалить строку.
- `bot/chat_run.py:22-28` → поправить docstring (убрать упоминание флага отката).
- `README.md:293`, `README.ru.md:301`, `.env.example:44` → удалить строки/записи про
  `CHAT_RUN_VARIANT_B_ENABLED`.
- `evals/judge/turn_runner.py:149` → удалить ключ из словаря.

## Тесты

Полный список тестов, которые упоминают `chat_run_variant_b_enabled` или mangled-имена, с
решением по каждому.

### Переименование mangled-имён (шаг A, правка без удаления тестов)

- `tests/test_openai_helper_tool_calls.py` — 40+ обращений к `_OpenAIHelper__handle_function_
  call`, несколько к `_OpenAIHelper__common_get_chat_response` (включая
  `test_common_chat_response_methods_are_not_wrapped_in_method_level_retry`, где сравниваются
  исходники `_OpenAIHelper__common_get_chat_response` и `_OpenAIHelper__common_get_chat_
  response_vision` — переименовать только первое имя, второе не входит в охват задачи).
  Механическая замена `_OpenAIHelper__handle_function_call` → `_handle_function_call`,
  `_OpenAIHelper__common_get_chat_response` → `_common_get_chat_response` (без `_vision`).
- `tests/test_reset_chat_history_async.py:175-177, 330, 346, 362-363` —
  `_OpenAIHelper__add_to_history` → `_add_to_history`; заодно поправить комментарий на
  строке 175 («__add_to_history is name-mangled») — после переименования это уже не так.
- `tests/test_stream_usage.py:105` — `helper._OpenAIHelper__handle_function_call = ...` →
  `helper._handle_function_call = ...`.

Рекомендуемый способ — не руками, а точечной заменой строк (`sed` по точным подстрокам
`_OpenAIHelper__common_get_chat_response`, `_OpenAIHelper__handle_function_call`,
`_OpenAIHelper__add_to_history` во всех трёх файлах разом), с последующим прогоном тестов —
это чистое переименование, за пределами исходных 6+10+3 сайтов вызова в исходном коде диффа
быть не должно.

### Тесты на флаг (шаг B) — что делать с каждым

| Файл | Тест | Что проверяет | Действие |
|---|---|---|---|
| `tests/test_telegram_builder_config.py:688` | `test_main_enables_chat_run_variant_b_by_default` | что env-парсинг даёт `True` по умолчанию | **Удалить** — сам флаг пропадает из схемы конфига |
| `tests/test_telegram_builder_config.py:696` | `test_main_can_disable_chat_run_variant_b_flag` | что `CHAT_RUN_VARIANT_B_ENABLED=false` работает | **Удалить** — переключателя не остаётся |
| `tests/test_openai_helper_tool_calls.py:1970` | `test_chat_run_variant_b_false_uses_legacy_non_stream_path` | что при `False` вызывается legacy-тело, а не `ChatRun` | **Удалить** — сам факт существования legacy-тела проверяется, оно удаляется |
| `tests/test_openai_helper_tool_calls.py:830` | `test_reply_intent_can_roll_back_to_legacy_timed_create` | откат `classify_reply_intent` на `_timed_create` | **Удалить** — путь отката пропадает (Находка 0) |
| `tests/test_openai_helper_tool_calls.py:1102` | `test_interpret_image_can_roll_back_to_legacy_timed_create` | откат vision-пути на `_timed_create` | **Удалить** |
| `tests/test_openai_helper_tool_calls.py:1191` | `test_interpret_image_stream_can_roll_back_to_legacy_timed_create` | откат vision-стрима на `_timed_create` | **Удалить** |
| `tests/test_openai_helper_tool_calls.py:1889` | `test_chat_response_stream_can_roll_back_to_legacy_timed_create` | откат стрима на `_timed_create` | **Удалить** |
| `tests/test_openai_helper_summarize_trim.py:132` | `test_summarise_window_can_roll_back_to_legacy_timed_create` | откат `_summarise_window` на `_timed_create` | **Удалить** |
| `tests/test_openai_helper_tool_calls.py:583` | `test_create_chat_response_completion_gate` (параметризован `enabled × stream`) | что `_create_chat_response_completion` роутит по флагу | **Переписать**: убрать параметризацию по `enabled`/`expected_path`, оставить один тест, что `_create_chat_response_completion` всегда вызывает `_timed_create_via_ai_provider` с переданными `kind`/`kwargs` (смысл теста — что kwargs правильно прокидываются, а не что есть развилка) |
| `tests/test_openai_helper_tool_calls.py:924` | `test_chat_run_variant_b_provider_failure_logs_provider_error_event` | что ошибка провайдера пишет `provider_error` событие | **Оставить**, убрать строку 933 (`helper.config["chat_run_variant_b_enabled"] = True` — больше нет такого ключа); переименование теста (убрать `variant_b`) — на усмотрение исполнителя, не обязательно |
| `tests/test_openai_helper_tool_calls.py:1933` | `test_chat_run_variant_b_returns_plain_chat_response` | обычный ответ через `ChatRun` + событие `ai_provider_response` | **Оставить**, убрать строку 1940 (`assert helper.config["chat_run_variant_b_enabled"] is True`) |
| `tests/test_openai_helper_tool_calls.py:2003` | `test_chat_run_variant_b_preserves_tool_call_flow` | ChatRun корректно проводит tool-call | **Оставить**, убрать строку 2022 |
| `tests/test_openai_helper_tool_calls.py:2045` | `test_chat_run_variant_b_direct_result_short_circuits` | `direct_result` обрывает цепочку | **Оставить**, убрать строку 2064 |
| `tests/test_openai_helper_tool_calls.py:2084` | `test_chat_run_variant_b_logs_retry_and_run_end` | `AIRetry`/`AIRunEnd` события | **Оставить**, убрать строку 2104 |
| `tests/test_openai_helper_tool_calls.py:3008` | `test_chat_completion_normalizes_empty_tools_to_none` | нормализация `tools=[]` → `None` в `chat_completion()` | **Оставить**, убрать строку 3017 (`helper.config["chat_run_variant_b_enabled"] = False`) — эта нормализация происходит в `chat_completion()` до `_create_chat_response_completion` и от флага не зависит; строка была лишней уже сегодня |
| `tests/test_pricing.py:101` | `test_usage_split_reaches_record_chat_tokens_through_tool_call_round_trip` | usage split доезжает до `resolve_chat_cost` через `ChatRun` | **Оставить**, убрать строку 124 |
| `tests/test_pricing.py:154` | `test_usage_split_stays_unknown_when_gateway_omits_split` | T06: неизвестный split остаётся `None` | **Оставить**, убрать строку 166 |
| `tests/test_pricing.py:193` | `test_interpret_image_text_response_omits_split_detail_when_unknown` | то же для vision-пути | **Оставить без изменений в теле**, поправить только докстринг (строка 196 — убрать фразу «независимо от chat_run_variant_b_enabled», флага больше нет) |
| `evals/judge/turn_runner.py:149` | не тест, конфиг для judge-прогонов | — | Удалить ключ (см. «Правки по file:line»); не часть прогона `pytest` |

Итого по `tests/`: **6 тестов удалить целиком**, **1 переписать** (снять параметризацию),
**8 тестов** — убрать одну строку/утверждение про флаг, оставив тело теста как есть,
**1 докстринг** поправить. Плюс глобальная замена mangled-имён в трёх файлах (см. выше).

## Команды проверки

```bash
# Шаг A — после переименования методов, до удаления флага:
~/.venvs/ctb/bin/python -m pytest tests/test_openai_helper_tool_calls.py \
    tests/test_reset_chat_history_async.py tests/test_stream_usage.py -q

# Шаг B — после удаления флага/legacy-тела и правки тестов:
~/.venvs/ctb/bin/python -m pytest tests/test_openai_helper_tool_calls.py \
    tests/test_openai_helper_summarize_trim.py tests/test_pricing.py \
    tests/test_telegram_builder_config.py tests/test_reset_chat_history_async.py \
    tests/test_stream_usage.py -q

# Полный прогон (обязателен перед закрытием задачи):
~/.venvs/ctb/bin/python -m pytest -q

# Быстрая проверка, что флаг нигде не остался (должно быть пусто, кроме
# исторического docs/ai_provider_variant_b_plan_2026-07-05.md):
grep -rFn -e "chat_run_variant_b_enabled" -e "CHAT_RUN_VARIANT_B_ENABLED" \
    --include=*.py --include=*.md --include=*.example . \
    | grep -v docs/ai_provider_variant_b_plan_2026-07-05.md

# Проверка, что mangled-имена трёх методов больше нигде не встречаются
# (кроме _common_get_chat_response_vision — она не переименовывается):
grep -rFn -e "_OpenAIHelper__common_get_chat_response(" \
    -e "_OpenAIHelper__handle_function_call" \
    -e "_OpenAIHelper__add_to_history" \
    --include=*.py .
```

Примечание про инструменты в этой среде: обычный `grep`/`rg` в этом окружении иногда искажает
вывод при регэкспах с несколькими альтернативами через `\|` (спецсимволы вроде `(` трактуются
как часть регулярного выражения, а не буквально), а `-F` (буквальный поиск) сам по себе не
поддерживает `\|` как «или». Команды выше используют `-F` вместе с несколькими `-e` (по одному
шаблону на флаг) — так альтернативы ищутся буквально и результат не искажается.

## Риски

- **Пропустить одну из двух развилок.** Если поправить только `_get_chat_response_locked`
  (`:935-945`) и забыть про `_create_chat_response_completion` (`:691-693`), флаг формально
  уйдёт из конфига, но код продолжит на него ссылаться (`self.config.get(...,True)` с
  дефолтом — тихо продолжит работать как раньше, только уже без возможности прочитать значение
  из env). Митигация: команда проверки выше специально ищет обе строки.
- **Тест `test_create_chat_response_completion_gate` теряет часть смысла.** Он единственный
  прямо проверял факт существования развилки; переписанная версия проверяет только
  «прокидывание параметров», что менее ценно. Если появится третий провайдер/путь в будущем —
  тест не защитит от регрессии автоматически. Приемлемо: сама развилка исчезает, тестировать
  становится нечего, кроме прокидывания kwargs.
- **`user_id or chat_id` подстановка меняет набор разрешённых плагинов в редких сценариях**
  (Находка 1, пункт 1) — если где-то `get_chat_response` вызывается без `user_id` (сегодня не
  происходит ни в одном известном месте, см. `bot/telegram_bot.py:2805, 4530, 4987`), поведение
  после удаления legacy-тела чуть изменится (в сторону единообразия с vision-путём). Риск
  низкий: это не новое поведение, `ChatRun` уже отвечает за non-stream путь по умолчанию с
  июля 2026 (`chat_run_variant_b_enabled=True` по умолчанию), то есть это и есть текущее
  реальное поведение бота — удаление legacy лишь убирает второй, никогда не выбираемый по
  умолчанию код.
- **Большой диф по тестам (40+ замен mangled-имени в одном файле).** Механический риск —
  промахнуться и переименовать не то. Митигация: делать заменой по точной подстроке (`sed`
  с `-F`-эквивалентом или Python `str.replace`), не регэкспом, и прогонять полный набор тестов
  сразу после.
- **Docstring `ChatRun` и README могут разъехаться, если шаг B делается частями.** Низкий
  риск, чисто текстовые правки, не влияют на поведение.

## Критерии готовности

- [ ] Три метода переименованы (`_common_get_chat_response`, `_handle_function_call`,
      `_add_to_history`); проверочная команда `grep -rFn -e "_OpenAIHelper__common_get_chat_response(" -e "_OpenAIHelper__handle_function_call" -e "_OpenAIHelper__add_to_history"` по `bot/` и `tests/` пуста (кроме `_common_get_chat_response_vision`, которая не переименовывается).
- [ ] `chat_run_variant_b_enabled`/`CHAT_RUN_VARIANT_B_ENABLED` не встречаются в `bot/`,
      `tests/`, `evals/`, `README.md`, `README.ru.md`, `.env.example` (историчекий
      `docs/ai_provider_variant_b_plan_2026-07-05.md` не трогается).
- [ ] `bot/openai_helper.py:947-1106` (нумерация до правки) удалены; `_get_chat_response_locked`
      безусловно делегирует в `ChatRun`.
- [ ] `_create_chat_response_completion` безусловно вызывает `_timed_create_via_ai_provider`;
      `_timed_create` остаётся как внутренняя реализация, вызываемая из неё.
- [ ] 6 тестов из таблицы удалены, 1 переписан без параметризации по флагу, 8 — с одной убранной
      строкой/утверждением, докстринг `test_interpret_image_text_response_omits_split_detail_
      when_unknown` поправлен.
- [ ] `~/.venvs/ctb/bin/python -m pytest -q` проходит целиком (весь набор `tests/` и
      `bot/tests/`), без новых `xfail`/`skip`.
- [ ] Ручная сверка: в `bot/openai_helper.py` не осталось строк, читающих несуществующий ключ
      `chat_run_variant_b_enabled` из `self.config` (в том числе через `.get(..., default)` —
      такой код не упадёт при отсутствии ключа, но обязан быть удалён, а не просто перестать
      использоваться).

## Постскриптум после ревью

Ревью (Sonnet, 2026-09-04): ошибок нет. Ревьюер построчно сверил удалённое legacy-тело
`_get_chat_response_locked` (HEAD) с `ChatRun.run_non_stream` — все ветки покрыты; `_timed_create`
не стал мёртвым (вызывается из `_timed_create_via_ai_provider`); ссылок на старые mangled-имена и
на `chat_run_variant_b_enabled` в `bot/`, `tests/`, `evals/`, README, `.env.example`, AGENTS.md и
codebase map не осталось. Полный прогон: 1526 passed, 1 skipped.

Предупреждения:
1. Двойная пустая строка в `test_chat_run_variant_b_returns_plain_chat_response` после удаления
   assert — схлопнута.
2. Переписанный `test_create_chat_response_completion_gate` защищает только проброс kwargs, а не
   отсутствие второй развилки — осознанный компромисс, зафиксированный в разделе «Риски»; без
   действия.

Отклонение разработчика: в плане прозой «6 тестов удалить», в таблице — 8 строк «Удалить»;
удалены все 8 по таблице (каждый проверен по имени). Дополнительно владельцем убраны упоминания
флага в `README.md`, `README.ru.md`, `.env.example`, `evals/judge/turn_runner.py`.
