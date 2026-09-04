# T10. Пустые `tools` и жёсткий лимит re-entry

Источник задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел «T10. Пустые `tools` и
жёсткий лимит re-entry»), детали — `docs/architecture_code_review_2026-09-04.md` §4.1, первые
два пункта. Роль этого документа — план для разработчика; код не менялся, только прочитан и
процитирован (`file:line`). Все номера строк проверены `sed -n`/`python3` по текущему дереву
на момент написания плана (2026-09-04).

Кратко для не-программиста: когда бот работает как «агент» (сам решает вызвать инструмент,
получить результат, вызвать ещё один и т.д.), код каждый раз заново спрашивает у модели
(LLM, большая языковая модель — то, что генерирует ответ) «какие инструменты тебе сейчас можно
предложить». Если список инструментов на каком-то шаге оказывается пустым, но код всё равно
отправляет его в API (внешний сервис, к которому бот обращается за ответом модели) как
`tools: []` — OpenAI-совместимый сервер отвечает ошибкой 400 (запрос отклонён как некорректный)
вместо ответа модели. Ниже — три связанных бага в этой же логике «сколько ещё раз можно
переспросить модель» и план их исправления.

## Цель

1. Ни при каких условиях не отправлять в `chat_completion()` пустой список `tools` как есть —
   пустой список должен превращаться в `tools=None` (ключ не передаётся в запрос) плюс
   `tool_choice="none"` (модели явно запрещено вызывать инструменты), причём в одном месте,
   которое гарантированно проходят все вызовы модели.
2. `final_delivery_required` (флаг «модель обязана в конце вызвать `deliver_to_user` —
   инструмент „финальная сдача результата пользователю“ из плагина `agent_tools`») не должен
   взводиться, если этот инструмент фактически недоступен в текущем режиме чата — иначе код
   на следующем шаге сам себя загоняет в пустой список инструментов из п. 1.
3. Цикл повторных обращений к модели (re-entry — повторный запрос к модели с обновлённым
   списком инструментов после того, как предыдущий ответ содержал вызов инструмента) должен
   иметь жёсткую верхнюю границу по числу раундов даже тогда, когда `final_delivery_required`
   истинен — сейчас граница есть только для «обычного» случая, а «обязательная сдача
   результата» может продолжаться неограниченно долго, если инструмент `deliver_to_user`
   раз за разом отвечает неуспехом.

Все три пункта нужно решить так, чтобы имеющиеся тесты либо не изменили поведение, либо были
осознанно обновлены (список — в разделе «Тесты»), и чтобы новое поведение было покрыто новыми
тестами.

## Анализ путей (file:line)

### 1. Кто производит пустой `tools` и куда он утекает

Низкоуровневая точка, которая реально уходит в HTTP-запрос — `bot/openai_helper.py:444-472`,
метод `chat_completion()`. Сборка `kwargs`:

```python
# bot/openai_helper.py:466-469 (текущий код)
kwargs: dict = {"model": model, "messages": messages, "stream": stream}
if tools is not None:
    kwargs["tools"] = tools
if tool_choice is not None:
    kwargs["tool_choice"] = tool_choice
```

Условие — `tools is not None`. Пустой список `[]` этому условию удовлетворяет (`[] is not
None` — истина), поэтому `kwargs["tools"] = []` уходит в запрос как есть. Это единственная
функция, которая формирует kwargs для вызова модели во всех путях re-entry — все три
источника пустого списка ниже вызывают именно её.

Источники пустого списка в `bot/openai_tool_handler.py`:

- **`_retry_plain_text_tool_intent`** (`bot/openai_tool_handler.py:1007-1066`), строки
  1044-1048:

  ```python
  tools = helper.plugin_manager.get_functions_specs(helper, model_to_use, allowed_plugins)
  tools = _filter_tools_by_name(tools, suppressed_reentry_tools, helper.plugin_manager)
  if times >= max_consecutive_calls:
      tools = _filter_tools_to_names(tools, set(), helper.plugin_manager)
  tool_choice = "auto" if _has_tool_specs(tools) else "none"
  ```

  `_filter_tools_to_names(tools, set(), ...)` (`bot/openai_tool_handler.py:509-...`) оставляет
  только те спеки, чьё имя входит в `set()` — то есть ни одной. `tool_choice` здесь уже
  корректно вычисляется как `"none"`, но `tools` остаётся пустым списком, а не `None`, и
  именно этот пустой список долетает до `chat_completion()`.

- **Основная ветка `handle_function_call`** (`bot/openai_tool_handler.py:1086-...`), строки
  1656-1665:

  ```python
  tools = helper.plugin_manager.get_functions_specs(helper, model_to_use, allowed_plugins)
  tools = _filter_tools_by_name(tools, suppressed_reentry_tools, helper.plugin_manager)
  if final_delivery_required and times >= max_consecutive_calls:
      tools = _filter_tools_to_names(tools, {DELIVERY_TOOL_NAME}, helper.plugin_manager)
  tool_choice = _reentry_tool_choice(
      tools, times=times, max_consecutive_calls=max_consecutive_calls,
      final_delivery_required=final_delivery_required,
  )
  ```

  Если `DELIVERY_TOOL_NAME` (`"agent_tools.deliver_to_user"`, `bot/openai_tool_handler.py:289`)
  не входит в `tools` (потому что `agent_tools` не в allow-list текущего режима — явно
  указан всего в 1 из ~25 записей `bot/chat_modes.yml` (`project_manager`), ещё 2 режима
  получают его только через `tools: ['All']`; итого 3 из ~25 — см. ниже), после
  `_filter_tools_to_names(tools, {DELIVERY_TOOL_NAME}, ...)` список становится пустым.
  `_reentry_tool_choice` (см. следующий пункт) в этом случае и так возвращает `"none"`, но
  пустой `tools` всё равно уходит в `chat_completion()`.

- **`_retry_missing_delivery_tool`** (`bot/openai_tool_handler.py:913-1006`), строки 963-969 —
  тот же паттерн что и выше, тоже с инлайновым `tool_choice = "auto" if _has_tool_specs(tools)
  else "none"` (не через `_reentry_tool_choice`).

Вывод: `tool_choice="none"` уже вычисляется правильно во всех трёх местах, проблема только в
том, что пустой `tools` не приравнивается к `None` при отправке. Это подтверждает диагноз
задачи: правильное место фикса — `chat_completion()`, а не три места-источника (их не нужно
трогать вообще).

### 2. `final_delivery_required` взводится без проверки, что `deliver_to_user` вообще доступен

`bot/openai_tool_handler.py:1504-1505` (текущий код):

```python
if tool_result.success and (defer_direct_results or tool_result.artifacts):
    final_delivery_required = True
```

`tool_result.artifacts` — непустой кортеж, если в JSON-ответе **любого** инструмента (не
только `agent_tools`) нашёлся абсолютный путь в одном из ключей `ARTIFACT_PATH_KEYS =
("value", "file_path", "path", "output_path", "artifact_path")` (`bot/tool_result.py:8`,
функция `_artifact_path` на `bot/tool_result.py:47-53`, вызывается из
`artifact_entries_from_tool_response` — `bot/tool_result.py:55`). Это не зависит от того,
разрешён ли в текущем режиме плагин `agent_tools`.

Проверено по `bot/chat_modes.yml` (`python3 -c "import yaml; ..."`, поле `tools:` каждого
режима): `agent_tools` в allow-list явно есть только у `project_manager`; ещё у `assistant` и
`skills_agent` эффективно есть, потому что их `tools: ['All']`. Остальные ~22 режима имеют
явный список конкретных плагинов без `agent_tools` — например `code_interpreter` (там есть
`codeinterpreter`, который может вернуть путь к файлу как артефакт) или `content_creator`
(там есть `stable_diffusion`). В любом таком режиме успешный вызов, скажем,
`codeinterpreter`-инструмента с абсолютным путём в ответе взводит `final_delivery_required =
True`, хотя `agent_tools.deliver_to_user` в этом режиме не существует как спека вообще — и
следующий re-entry (см. пункт 1 выше, основная ветка `handle_function_call`) сам себя
загоняет в пустой `tools`.

Для проверки доступности инструмента уже есть готовая функция —
`_delivery_tool_is_allowed(helper, allowed_plugins)` (`bot/openai_tool_handler.py:886-889`):

```python
def _delivery_tool_is_allowed(helper, allowed_plugins) -> bool:
    is_allowed = getattr(helper.plugin_manager, "is_function_allowed", None)
    if not callable(is_allowed):
        return False
    return bool(is_allowed(DELIVERY_TOOL_NAME, allowed_plugins))
```

Она уже используется в двух местах: `_retry_missing_delivery_tool`
(`bot/openai_tool_handler.py:930`, как guard перед попыткой чинить отсутствие вызова) и
при вычислении `agent_delivery_workflow` (`bot/openai_tool_handler.py:1336-1338`):

```python
agent_delivery_workflow = (
    _agent_delivery_workflow_active(tool_calls, tools_used)
    and _delivery_tool_is_allowed(helper, allowed_plugins)
)
```

`allowed_plugins` в момент строки 1504 — это параметр `handle_function_call`, уже нормализован
через `helper.plugin_manager.filter_allowed_plugins(...)` на строке 1125-1126, то есть в
области видимости и в правильной, уже отфильтрованной форме. Единственное недостающее звено —
вызвать `_delivery_tool_is_allowed` в условии на строке 1504.

**Второе место, где взводится флаг** — `bot/openai_tool_handler.py:1627-1628`, внутри ветки
`repeated_failures` (несколько подряд неудачных вызовов одного инструмента):

```python
if skills_agent_mode:
    final_delivery_required = True
```

Проверено: `skills_agent_mode = _is_skills_agent_mode(helper, chat_id)` (импорт из
`.skill_script_routing`), а `skills_agent` режим в `bot/chat_modes.yml` имеет `tools: ['All']`
(проверено `python3 -c "import yaml; ...['skills_agent']['tools']"` → `['All']`). При
`allowed_plugins == ['All']` `_delivery_tool_is_allowed` возвращает `True` безусловно
(`is_function_allowed` в `bot/plugin_manager.py:626-631`: `if allowed_plugins == ['All']:
return True` — до проверки, есть ли спека фактически). Значит эта ветка уже безопасна по
построению (режим гарантирует `['All']`), добавлять сюда проверку не обязательно — это не
тот путь, который приводит к пустому `tools`. Отмечаю это как проверенный факт, а не
предположение, чтобб не трогать код, который не является источником бага (surgical-change
правило).

### 3. `_reentry_tool_choice` не имеет верхней границы при `final_delivery_required=True`

`bot/openai_tool_handler.py:905-910` (текущий код):

```python
def _reentry_tool_choice(tools, *, times: int, max_consecutive_calls: int, final_delivery_required: bool) -> str:
    if not _has_tool_specs(tools):
        return "none"
    if final_delivery_required:
        return "auto"
    return "auto" if times < max_consecutive_calls else "none"
```

Единственный вызов — `bot/openai_tool_handler.py:1660-1665`, внутри рекурсии
`handle_function_call` → `handle_function_call(..., times + 1, ...)`
(`bot/openai_tool_handler.py:1682-1699`), то есть `times` растёт на 1 каждый раунд без
верхней границы, заданной снаружи. Когда `final_delivery_required=True`, функция всегда
возвращает `"auto"`, независимо от `times`. Если `agent_tools.deliver_to_user`
(`bot/plugins/agent_tools.py`, обработка `success: False` при повторных вызовах —
`bot/plugins/agent_tools.py:2785-2810`, диагностировано в архитектурном ревью, не
перепроверено построчно в рамках этого плана, т.к. вне зоны правки T10) раз за разом отвечает
неуспехом, модель может вызывать его сколько угодно раз — единственная граница на практике
это «усталость» модели, а не код. Это противоречит правилу из `AGENTS.md` (раздел
«Deterministic Routing In Agent Plugins»): *«once `functions_max_consecutive_calls` is
exhausted the code forcibly narrows the tool set to the delivery tool»* — код обещает
принудительное ограничение, но фактической верхней границы раундов для этого случая нет,
только сужение набора инструментов.

Отдельно проверено: `_retry_missing_delivery_tool` (`bot/openai_tool_handler.py:913-1006`)
вычисляет свой `tool_choice` инлайново (строка 969: `"auto" if _has_tool_specs(tools) else
"none"`), не через `_reentry_tool_choice`, и это НЕ тот путь, который бесконечен — количество
входов в `_retry_missing_delivery_tool` уже ограничено параметром `delivery_repair_attempts`
и константой `DELIVERY_REPAIR_MAX_ATTEMPTS = 2` (`bot/openai_tool_handler.py:293`) через
проверку в `enforce_delivery_contract_if_needed` (`bot/openai_tool_handler.py:1129-1136`).
Неограничен именно путь «модель вызывает `deliver_to_user` как обычный tool call, получает
`success: False`, и код идёт по основной ветке `handle_function_call`» — там ограничения нет.
Фикс `_reentry_tool_choice` закрывает именно эту дыру; `_retry_missing_delivery_tool` в объём
правки T10 не входит (см. «Риски»).

## Дизайн (с кодом)

### Правка 1 — единая нормализация пустых `tools` в `chat_completion()`

Место: `bot/openai_helper.py:466-469`, метод `chat_completion()`. Это единственная точка,
через которую проходят все три источника из «Анализа», а также любые будущие вызовы —
исправление здесь защищает по построению, без необходимости синхронизировать три места и
без риска, что новый код в будущем повторит ту же ошибку.

```python
# было
kwargs: dict = {"model": model, "messages": messages, "stream": stream}
if tools is not None:
    kwargs["tools"] = tools
if tool_choice is not None:
    kwargs["tool_choice"] = tool_choice

# стало
kwargs: dict = {"model": model, "messages": messages, "stream": stream}
if tools is not None and not tools:
    # Пустой список инструментов нельзя отправлять как tools=[] — некоторые
    # OpenAI-совместимые шлюзы отвечают 400 "Invalid 'tools': empty array"
    # независимо от tool_choice. "Нет инструментов" всегда означает tools=None.
    tools = None
    tool_choice = "none"
if tools is not None:
    kwargs["tools"] = tools
if tool_choice is not None:
    kwargs["tool_choice"] = tool_choice
```

Разбор условия `tools is not None and not tools`: `None` пропускается (как и раньше —
поведение для явного «инструментов не передавали» не меняется), непустой список не
затрагивается, `[]` — единственный случай, который меняет поведение.

**Рекомендация:** ограничиться проверкой списка (`not tools`), не обрабатывать форму
`{"function_declarations": [...]}` (Google-стиль). Обоснование: по `AGENTS.md`
(«Google model tool specs...») ветка `{"function_declarations": specs}` сейчас недостижима
в рантайме, потому что `GOOGLE_MODELS` — алиас пустого кортежа `GOOGLE`
(`bot/plugin_manager.py:20`, `bot/model_constants.py:24`); добавлять код под мёртвую ветку —
нарушение правила «No features beyond what was asked» / «No speculative code».

**Альтернатива (если Google-ветку решат реактивировать):** тогда потребуется то же самое
условие, что уже реализовано в `_has_tool_specs` (`bot/openai_tool_handler.py:600-603`) —
`not tools.get("function_declarations")` для `dict`. На тот момент можно либо скопировать
эту проверку сюда, либо (чище) вынести общую функцию `_tools_are_empty(tools)` в модуль,
общий для обоих файлов — но заводить её сейчас ради недостижимого кода избыточно.

### Правка 2 — `final_delivery_required` только если `deliver_to_user` доступен

Место: `bot/openai_tool_handler.py:1504-1505`.

```python
# было
if tool_result.success and (defer_direct_results or tool_result.artifacts):
    final_delivery_required = True

# стало
if (
    tool_result.success
    and (defer_direct_results or tool_result.artifacts)
    and _delivery_tool_is_allowed(helper, allowed_plugins)
):
    final_delivery_required = True
```

Никакой новой функции не требуется — `_delivery_tool_is_allowed` уже существует именно для
этой проверки и уже используется рядом (строка 1337) для аналогичного случая
(`agent_delivery_workflow`). `allowed_plugins` уже в области видимости и уже нормализован
(строка 1125-1126). Изменение — три добавленные строки в одном условии.

Ветку `bot/openai_tool_handler.py:1627-1628` (`if skills_agent_mode: final_delivery_required
= True`) не трогаю — см. «Анализ», п. 2: она безопасна по построению, потому что
`skills_agent` режим всегда `tools: ['All']`.

### Правка 3 — жёсткая граница раундов для `_reentry_tool_choice`

Место: `bot/openai_tool_handler.py:905-910`, плюс новая константа рядом с
`DELIVERY_REPAIR_MAX_ATTEMPTS` (`bot/openai_tool_handler.py:293`).

```python
# было (константы, bot/openai_tool_handler.py:289-296)
DELIVERY_TOOL_NAME = "agent_tools.deliver_to_user"
DELIVERY_PLUGIN_PREFIX = DELIVERY_TOOL_NAME.rsplit(".", 1)[0] + "."
MANAGE_PLAN_TOOL_NAME = DELIVERY_PLUGIN_PREFIX + "manage_plan_tasks"
ASK_USER_TOOL_NAME = DELIVERY_PLUGIN_PREFIX + "ask_telegram_user"
DELIVERY_REPAIR_MAX_ATTEMPTS = 2
SUPPRESS_REENTRY_TOOLS_KEY = "suppress_reentry_tools"
RETRY_PLAIN_TEXT_TOOL_INTENT_KEY = "retry_plain_text_tool_intent"
TOOL_INTENT_REPAIR_MAX_ATTEMPTS = 1

# стало — добавить одну строку рядом с однотипными константами
DELIVERY_REPAIR_MAX_ATTEMPTS = 2
DELIVERY_GRACE_ROUNDS = 2  # доп. раунды re-entry сверх max_consecutive_calls, пока
                           # final_delivery_required=True и deliver_to_user ещё не сдал успех
```

```python
# было
def _reentry_tool_choice(tools, *, times: int, max_consecutive_calls: int, final_delivery_required: bool) -> str:
    if not _has_tool_specs(tools):
        return "none"
    if final_delivery_required:
        return "auto"
    return "auto" if times < max_consecutive_calls else "none"

# стало
def _reentry_tool_choice(tools, *, times: int, max_consecutive_calls: int, final_delivery_required: bool) -> str:
    if not _has_tool_specs(tools):
        return "none"
    if final_delivery_required:
        return "auto" if times < max_consecutive_calls + DELIVERY_GRACE_ROUNDS else "none"
    return "auto" if times < max_consecutive_calls else "none"
```

Значение `DELIVERY_GRACE_ROUNDS = 2` выбрано по образцу задачи («например 2») и симметрично
`DELIVERY_REPAIR_MAX_ATTEMPTS = 2`, которая уже задаёт похожий по смыслу лимит для соседнего
(текстового) пути повторов. Смысл: обычный лимит (`max_consecutive_calls`) даёт модели
`N` раундов на любые инструменты; если после этого всё ещё требуется сдать результат через
`deliver_to_user`, даём ещё `DELIVERY_GRACE_ROUNDS` раундов **специально на сдачу**, а не
бесконечно.

**Что происходит, когда граница исчерпана:** `tool_choice` становится `"none"`; вместе с
Правкой 1 пустой/несущественный список `tools` (если к этому моменту он сузился до
`{DELIVERY_TOOL_NAME}`, но именно на этом шаге появился в ответе) не является проблемой — при
`tool_choice="none"` передавать непустой `tools` не запрещено спецификацией API, это не тот
случай, который ловит Правка 1 (пустой). Модель вынуждена ответить текстом. Этот текстовый
ответ уходит в `handle_function_call` со следующего рекурсивного вызова, попадает в ветку
`if not tool_calls:` (`bot/openai_tool_handler.py:1300-1303`) и там снова проверяется
`enforce_delivery_contract_if_needed()` → `_delivery_contract_required` — которая, если режим
это требует, попробует `_retry_missing_delivery_tool` ещё раз, но это уже отдельный,
самостоятельно ограниченный контур (`DELIVERY_REPAIR_MAX_ATTEMPTS = 2`, см. «Анализ», п. 3).
То есть после исчерпания `DELIVERY_GRACE_ROUNDS` система не зависает, а переходит в уже
существующий ограниченный текстовый repair-контур и в худшем случае завершается
детерминированной ошибкой `_delivery_contract_error(helper)`
(`bot/openai_tool_handler.py:893-900`) вместо бесконечного цикла.

## Правки (файлы)

1. `bot/openai_helper.py:466-469` — нормализация пустых `tools` (Правка 1).
2. `bot/openai_tool_handler.py:1504-1505` — gate на `_delivery_tool_is_allowed` (Правка 2).
3. `bot/openai_tool_handler.py:289-296` — новая константа `DELIVERY_GRACE_ROUNDS`
   (Правка 3, часть 1).
4. `bot/openai_tool_handler.py:905-910` — жёсткая граница в `_reentry_tool_choice`
   (Правка 3, часть 2).

Больше никаких файлов менять не нужно — три источника пустого `tools` (п. 1 «Анализа»)
не трогаем: их `tool_choice` уже правильный, а `tools=[]` теперь безопасен на уровне
`chat_completion()`.

## Тесты

Базовая проверка перед правками (зелёная): `139 passed` для команды из раздела «Команды
проверки» (прогнано на текущем дереве без изменений).

### Существующие тесты, которые нужно обновить

- **`tests/test_openai_helper_tool_calls.py::test_plain_text_tool_intent_repair_after_limit_sends_no_tools`**
  (строка 2935). Сейчас:

  ```python
  assert helper.client.create_kwargs[0]["tool_choice"] == "none"
  assert helper.client.create_kwargs[0]["tools"] == []
  ```

  После Правки 1 `helper.client.create_kwargs[0]` больше не будет содержать ключ `"tools"`
  вообще (он не передаётся в `kwargs`, когда список пуст) — обращение `[...]["tools"]` упадёт
  с `KeyError`. Нужно заменить вторую строку на
  `assert "tools" not in helper.client.create_kwargs[0]` (или `.get("tools") is None`).
  Это единственный тест, который **обязательно** сломается без правки — он прямо
  проверяет старое (ошибочное) поведение как желаемое.

Остальные тесты, что нашлись по `deliver_to_user` / `final_delivery_required` /
`functions_max_consecutive_calls` (`test_final_delivery_reentry_after_limit_only_exposes_delivery_tool`
— 3740, `test_delivery_repair_after_limit_only_exposes_delivery_tool` — 3801,
`test_successful_tool_output_path_adds_manifest_for_delivery_reentry` — 3532,
`test_generic_successful_tool_does_not_bypass_consecutive_call_limit` — 3863, и другие с
`allowed_plugins=["All"]`) — проверены построчно и **не потребуют изменений**:

- Во всех них `allowed_plugins=["All"]`. `DummyPluginManager.is_function_allowed`
  (тестовый дублёр, `tests/test_openai_helper_tool_calls.py`, класс `DummyPluginManager`)
  воспроизводит поведение продакшена: `if allowed_plugins == ["All"]: return True` — то есть
  `_delivery_tool_is_allowed` в них всегда `True`, Правка 2 не меняет их исход.
- Значения `times`/`functions_max_consecutive_calls` во всех тестах, где встречается
  `_reentry_tool_choice` через `handle_function_call` — `times ∈ {0, 1}`,
  `functions_max_consecutive_calls ∈ {1, 5}` (проверено `grep` по всему файлу). При
  `DELIVERY_GRACE_ROUNDS = 2` порог `max_consecutive_calls + 2` везде ≥ 3, что больше любого
  использованного `times` — Правка 3 не меняет их исход.
- `test_allowed_tool_reentry_uses_original_allowlist` (единственный тест с ограниченным
  `allowed_plugins=["weather"]`) не производит артефактов и не взводит
  `final_delivery_required` — не затронут ни одной из трёх правок.

### Новые тесты

1. **`chat_completion` не отправляет `tools=[]`** (низкоуровневый, в
   `tests/test_openai_helper_tool_calls.py` или соседнем файле с юнит-тестами
   `OpenAIHelper.chat_completion`, если такой уже есть — проверить перед добавлением). Вызвать
   `helper.chat_completion(model=..., messages=[...], tools=[], tool_choice="auto")` и
   проверить, что в `kwargs`, дошедших до клиента: `"tools" not in kwargs` и
   `kwargs["tool_choice"] == "none"` (а не `"auto"`, которое было передано явно — фиксирует,
   что нормализация именно перезаписывает, а не просто пропускает `None`).
   Дополнительно — регрессия: `tools=None` по-прежнему не добавляет `"tools"` в kwargs, а
   непустой `tools=[{...}]` передаётся как есть без изменений `tool_choice`.

2. **`final_delivery_required` не взводится без `deliver_to_user` в allow-list** (в
   `tests/test_openai_helper_tool_calls.py`, по образцу
   `test_successful_tool_output_path_adds_manifest_for_delivery_reentry`, но с
   `allowed_plugins=["builder"]` вместо `["All"]`, и без `agent_tools.deliver_to_user` в
   `specs` — воспроизводит реальный режим вроде `code_interpreter` из `chat_modes.yml`, где
   есть артефакт-инструмент без `agent_tools`). Инструмент `builder.build` возвращает
   `output_path: "/tmp/out.pptx"` (артефакт). Ожидание: `final_delivery_required` не взводится,
   финальный ответ модели (текст) возвращается как есть, `helper.client.create_kwargs[-1]`
   не содержит `"tools"` с единственным `deliver_to_user` (в отличие от текущего
   `test_final_delivery_reentry_after_limit_only_exposes_delivery_tool`, где `deliver_to_user`
   разрешён и ожидаемо остаётся), и не возникает `KeyError`/400 в процессе (сам факт, что
   тест проходит без исключения — уже регрессионная проверка на пустой `tools`).

3. **Жёсткая граница `max_consecutive_calls + DELIVERY_GRACE_ROUNDS` даёт `"none"`**
   (юнит-тест на саму функцию `_reentry_tool_choice`, импортированную напрямую — по образцу
   того, как файл уже импортирует `_has_tool_specs`, `_filter_tools_by_name` и
   `_retry_plain_text_tool_intent` напрямую из `bot.openai_tool_handler`). Три случая:
   - `times = max_consecutive_calls + DELIVERY_GRACE_ROUNDS - 1`, `final_delivery_required=True`
     → `"auto"`.
   - `times = max_consecutive_calls + DELIVERY_GRACE_ROUNDS`, `final_delivery_required=True`
     → `"none"`.
   - Тот же случай без Правки 3 (для документирования регрессии, не для CI) вернул бы `"auto"`
     — можно не кодировать отдельным тестом, но описать в комментарии к тесту, что именно
     проверяется.
   При желании — сквозной тест через `handle_function_call` с фейковым `agent_tools.deliver_to_user`,
   который всегда возвращает `success: False`, и `functions_max_consecutive_calls=1`: после
   `1 + DELIVERY_GRACE_ROUNDS` раундов вызовов `handle_function_call` должен вернуть текстовый
   ответ (не зависнуть/не превысить лимит рекурсии), аналогично уже существующему
   `test_generic_successful_tool_does_not_bypass_consecutive_call_limit`, но с
   `deliver_to_user`, отвечающим неуспехом, вместо `weather.get_weather`.

## Команды проверки

```bash
python3 -m pytest tests/test_openai_helper_tool_calls.py tests/test_agent_tools_verify.py -q -p no:cacheprovider
```

`tests/test_agent_tools_verify.py` в правках T10 не участвует по коду (это тесты verify-шага
плана `agent_tools`, независимая от re-entry-логики область — `_apply_plan_runtime_effects`,
`on_before_chat_request`), но задан в команде проверки как регрессионный сосед, потому что
обе правки живут в `bot/openai_tool_handler.py`, а этот тест использует тот же модуль
`agent_tools`. Дополнительно, точечно после каждой правки:

```bash
python3 -m pytest tests/test_openai_helper_tool_calls.py -k "tool_intent_repair or final_delivery or delivery_repair or generic_successful_tool or reentry_tool_choice" -q -p no:cacheprovider
```

## Риски

- **`_retry_missing_delivery_tool` не получает жёсткую границу этим планом.** Его собственный
  `tool_choice` (строка 969) не проходит через `_reentry_tool_choice`, и его цикл повторов
  ограничен отдельно (`DELIVERY_REPAIR_MAX_ATTEMPTS = 2`) — это другой, уже ограниченный
  контур (текстовые ответы вместо вызова инструмента), не тот, что описан в задаче T10, п. 3
  (там речь о **вызовах** `deliver_to_user`, не о молчании модели). Если впоследствии
  окажется, что и этот контур должен использовать `DELIVERY_GRACE_ROUNDS` вместо собственной
  логики — это отдельная, более крупная правка (унификация двух похожих, но разных
  вычислений `tool_choice`), не входит в surgical-scope T10.
- **Точное значение `DELIVERY_GRACE_ROUNDS = 2`** — выбрано по аналогии с
  `DELIVERY_REPAIR_MAX_ATTEMPTS`, не выведено из данных о реальной частоте `success: False`
  у `deliver_to_user`. Если на практике 2 раунда мало (агент делает сложную многошаговую
  сдачу с несколькими артефактами) — потребуется поднять константу; тесты рассчитаны на
  переменную границу, а не захардкоженное число раундов, поэтому смена значения не потребует
  правки тестов из п. 3 «Новых тестов» (они используют `max_consecutive_calls +
  DELIVERY_GRACE_ROUNDS`, а не литерал).
- **Google `function_declarations` форма `tools` не обрабатывается Правкой 1** — осознанно
  (см. «Дизайн», Правка 1): ветка сейчас недостижима
  (`bot/plugin_manager.py:20`, `bot/model_constants.py:24`). Если её реактивируют без
  синхронной правки `chat_completion()`, `tools={"function_declarations": []}` пройдёт мимо
  новой проверки `not tools` (пустой `dict` с ключом — `bool({...})` истинен) и всё ещё сможет
  дойти до API как «пустой» с точки зрения фактических инструментов. Зафиксировано как
  известный, осознанно отложенный пробел, а не пропущенный случай.
- **`normalize_tool_result`/`ARTIFACT_PATH_KEYS`** (`bot/tool_result.py:8`) — Правка 2 не
  меняет то, что считается «артефактом» (любой абсолютный путь в ответе любого инструмента);
  она только не даёт этому факту требовать недоступный инструмент сдачи. Если у бота
  появится **второй** способ сдать артефакт кроме `deliver_to_user`, Правку 2 нужно будет
  расширить (сейчас не нужно — второго пути нет).

## Критерии готовности

1. `python3 -m pytest tests/test_openai_helper_tool_calls.py tests/test_agent_tools_verify.py -q -p no:cacheprovider`
   зелёный после всех правок и обновления
   `test_plain_text_tool_intent_repair_after_limit_sends_no_tools`.
2. Три новых теста (пустой `tools` в `chat_completion`, `final_delivery_required` без
   `deliver_to_user`, жёсткая граница `_reentry_tool_choice`) добавлены и зелёные.
3. Ни один тест не проверяет `tools == []` как ожидаемое значение, дошедшее до клиента API.
4. Grep по `bot/openai_helper.py` и `bot/openai_tool_handler.py` не находит новых мест, где
   пустой список `tools` передаётся в `chat_completion()` в обход нормализации (нормализация
   в одном месте, п. 1 «Дизайна», исключает такую возможность по построению).
5. `DELIVERY_GRACE_ROUNDS` объявлена один раз, читается `_reentry_tool_choice`, не
   продублирована как магическое число где-либо ещё.

## Предложение правки для AGENTS.md

В раздел «Deterministic Routing In Agent Plugins» добавить фразу: «`_reentry_tool_choice`
(`bot/openai_tool_handler.py:905`) forces `"none"` once `times >= max_consecutive_calls +
DELIVERY_GRACE_ROUNDS`, so the mandatory-delivery path (`final_delivery_required`) is bounded
the same way as the ordinary tool-call path.»

## Постскриптум после ревью (2026-09-04)

Реализовано по плану: при повторном входе в модель после tool-вызовов пустой список инструментов
больше не передаётся (`"tools" not in kwargs` вместо `tools == []`), три новых теста в
`tests/test_openai_helper_tool_calls.py`. Ревью (Sonnet, persona reviewer): ошибок и предупреждений
нет. Замечания: красный на момент ревью тест `..._rejects_nested_chat_response_even_with_same_chat_lock`
относился к параллельной задаче T09 и закрыт там; строка `chat_run_variant_b_enabled = False` в
`test_chat_completion_normalizes_empty_tools_to_none` — мёртвая настройка (boilerplate), не баг;
ссылка `bot/openai_tool_handler.py:868` на `_reentry_tool_choice` в `AGENTS.md` устарела ещё до
сегодняшних правок — уходит в T21.
