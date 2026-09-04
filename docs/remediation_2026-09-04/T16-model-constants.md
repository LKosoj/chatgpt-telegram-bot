# T16. Удалить мёртвый провайдер-слой (`model_constants`)

Источники задачи: `docs/audit_remediation_plan_2026-09-04.md`, раздел T16 («Волна 4»), и
`docs/architecture_code_review_2026-09-04.md`, пункт **П2** «Удалить мёртвый провайдер-слой
`model_constants`» (в задаче он назван «П3» — это ссылка на соседний пункт: П3 в текущей
редакции документа — про единый источник конфига, задача T17. Проверено построчным поиском по
файлу: пункты идут без повтора номеров, П2 — единственный, где речь про `model_constants`. Ниже
используется П2 как содержательно верный источник; расхождение в номере не влияет на объём
задачи, весь текст П2 процитирован ниже.)

Термины:
- **Мёртвый код (dead code) / недостижимая ветка** — код, который синтаксически существует, но
  условие для его выполнения никогда не становится истинным, поэтому он не может исполниться ни
  при каком реальном вводе.
- **Кортеж (tuple)** — неизменяемый список значений в Python, здесь: `()` значит «пустой список»,
  `model in ()` всегда `False`, а `model not in ()` всегда `True`.
- **Capability-таблица** — справочник вида «модель → что она умеет» (например, «поддерживает ли
  `tool_choice`», «использовать ли `max_completion_tokens` вместо `max_tokens`»), обычно
  задаваемый через конфиг/переменную окружения, а не зашитый в код.
- **Gateway-алиас** — здесь: `llmgateway/high`, `llmgateway/light_model` и т.п. — имя модели,
  которое бот передаёт в OpenAI-совместимый шлюз (gateway); реальный провайдер (OpenAI/Google/
  Anthropic/…) за этим именем выбирает сам шлюз, бот его не видит.

## Цель

`bot/model_constants.py:19-30` определяет 12 констант-«семейств моделей»
(`GPT_4_VISION_MODELS`, `GPT_4O_MODELS`, `GPT_5_MODELS`, `O_MODELS`, `ANTHROPIC`, `GOOGLE`,
`MISTRALAI`, `DEEPSEEK`, `LLAMA`, `PERPLEXITY`, `MOONSHOTAI`, `QWEN`) — и все они равны `()`.
Это статический факт исходного кода (не переменные окружения — значения зашиты в файл), поэтому
**каждая** ветка вида `model in (X + Y + …)` в проекте сейчас либо никогда не выполняется, либо
(через `not in`) выполняется **всегда**. Полный обход показал, что так ведут себя буквально все
15 использований в `bot/openai_helper.py`, `bot/telegram_bot.py` и `bot/plugin_manager.py` — без
исключений.

Задача:
1. Найти все использования каждой из 12 констант (сделано — см. таблицу ниже) и классифицировать
   каждую ветку: недостижима (`in (...)` с пустыми кортежами) или псевдо-условна (`not in (...)`
   всегда `True`, реальное поведение — только «положительная» ветка).
2. Для каждой ветки решить: удалить целиком (если она никогда не выполняется и не оставляет
   осиротевших переменных) или заменить веткой на то единственное поведение, которое уже
   исполняется сегодня.
3. Обосновать выбор между (а) точечным удалением мёртвого кода без новой абстракции и (б) единой
   функцией `model_capabilities(model) -> ModelCaps` с переопределением через конфиг — с учётом
   правила «Simplicity First» и явного предупреждения из П2: «Не наполнять кортежи из env
   ‘на всякий случай’: ветки два месяца не исполнялись».
4. Не трогать `LLMGATEWAY_*` и `MAX_OUTPUT_TOKENS` (`bot/model_constants.py:3-15`) — это не
   «семейства», это реальные, ненулевые, активно используемые значения (алиасы шлюза и лимит
   токенов), вне области T16.

## Таблица: константы → использования → достижимость → решение

Все 12 констант объявлены в `bot/model_constants.py:19-30` со значением `()`. Столбец
«Достижимость» — про конкретную ветку кода, использующую константу, а не про саму константу.

| Константа | Использования (`file:line`, актуально проверено) | Достижимость ветки | Что делает «включённая» ветка | Решение |
|---|---|---|---|---|
| `GPT_4_VISION_MODELS` | нигде, кроме определения (`bot/model_constants.py:19`) | н/п — не импортируется | — | Удалить константу |
| `GPT_5_MODELS` | нигде, кроме определения (`bot/model_constants.py:21`) | н/п — не импортируется | — | Удалить константу |
| `LLAMA` | нигде, кроме определения (`bot/model_constants.py:27`) | н/п — не импортируется | — | Удалить константу |
| `GPT_4O_MODELS` | импорт `bot/openai_helper.py:41`; `bot/openai_helper.py:1753` (`_uses_structured_tool_history`); `bot/openai_helper.py:3502` (`__add_function_call_to_history`); `bot/openai_helper.py:4269` (`get_max_tokens`) | `:1753`,`:3502` — тавтология: `X in () or Y` ≡ `Y`, реально достижима только `Y` (`model_to_use in self.get_model_choices()`); `:4269` — голая проверка `in GPT_4O_MODELS`, недостижима (кортеж всегда пуст, других веток нет) | `:1753`/`:3502` — переключают формат истории вызова функции («tool»/«assistant»-роль вместо «function»); `:4269` — жёстко ограничивает `max_generation_tokens = 32768` | `:1753`/`:3502` — убрать дизъюнкт `GPT_4O_MODELS`/`O_MODELS`, оставить `model_to_use in self.get_model_choices()`; `:4269` — удалить блок целиком (0 наблюдаемых эффектов) |
| `O_MODELS` | импорт `:42`; `:1589`,`:1622`,`:1636` (`__common_get_chat_response`); `:1753`? нет (см. выше только GPT_4O); `:2072`,`:2093` (`should_force_non_stream_first_turn[_async]`); `:2179` (`_retry_empty_response_after_tools`); `:2194` (`_retry_empty_response_with_tools`); `:2561` (`__common_get_chat_response_vision`); `:3502` (см. выше); `bot/telegram_bot.py:45,4224,4906` | `:1589`,`:2179` — `in (...)` с пустыми кортежами → недостижимы; `:1622`,`:1636`,`:2561`,`telegram_bot:4224/4906` — `not in (...)` всегда `True` → «положительная» ветка выполняется всегда; `:2072`,`:2093`,`:2194` — `in (...)` → `return False`/`return None` никогда не срабатывает | `:1589`/`:2179` — форсируют `stream=False` + `max_completion_tokens` вместо `max_tokens` (комментарий «o1 series only supports max_completion_tokens»); `:1622`/`:2561` — решают, прикреплять ли `tools`/`tool_choice`; `:1636` — флаг `gate_supported_model` для skills_agent-гейта; `:2072`/`:2093` — ранний `return False` (модель не поддерживает гейт); `:2194` — ранний `return None` (не ретраить с тулами); `telegram_bot:4224/4906` — решают, стримить ли ответ | Удалить `if`-блок `:1589`/`:2179` целиком (мёртвый код, 0 эффекта); упростить `:1622`/`:2561` до `if tools:`; удалить `gate_supported_model`/строку `:1636` и её употребление; удалить проверки `:2072`/`:2093`/`:2194` целиком (со следствиями — см. «Дизайн»/«Правки»); в `telegram_bot.py` убрать член `and model_to_use not in (...)`, оставить решение на `self.config['stream']` |
| `ANTHROPIC` | импорт `openai_helper.py:43`; `:1589`,`:2179` (см. `O_MODELS`); `:3492` (`__add_function_call_to_history`); `telegram_bot.py:45,4224,4906` | `:3492` — `in (ANTHROPIC + DEEPSEEK)`, пустой кортеж → недостижима | `:3492` — инжектит результат тула как `role: user` текст (для провайдеров без роли `function`) | Удалить ветку `:3492` целиком (мёртвая, 0 эффекта) |
| `GOOGLE` | импорт `openai_helper.py:44`; `:1589`,`:1622`,`:1636`,`:2072`,`:2093`,`:2179`,`:2194`,`:2561` (см. `O_MODELS`); `bot/plugin_manager.py:20` (`from .model_constants import GOOGLE as GOOGLE_MODELS`), `:357` (`_format_specs_for_model`); `telegram_bot.py:45,4224,4906` | `:357` — `if model_to_use in GOOGLE_MODELS:` — недостижима (кортеж пуст) | `:357` — переключает формат тул-спеков на `{"function_declarations": [...]}` (Google-стиль) вместо OpenAI-стиля `[{"type":"function","function":{...}}]` | Удалить импорт `GOOGLE_MODELS` (`plugin_manager.py:20`) и ветку `:357` целиком — `_format_specs_for_model` всегда возвращает OpenAI-стиль (уже фактическое поведение) |
| `MISTRALAI` | импорт `:45`; `:1589`,`:2179` (см. `O_MODELS`); `:3495` (`__add_function_call_to_history`); `telegram_bot.py:45,4224,4906` | `:3495` — `in (MISTRALAI + MOONSHOTAI)`, пустой кортеж → недостижима | `:3495` — инжектит результат тула как `role: tool` с полем `name` (Mistral/Moonshot-стиль) | Удалить ветку `:3495` целиком (мёртвая) |
| `DEEPSEEK` | импорт `:46`; `:3492` (см. `ANTHROPIC`); `telegram_bot.py:45,4224,4906` | недостижима (см. `ANTHROPIC`) | см. `ANTHROPIC` | см. `ANTHROPIC` |
| `PERPLEXITY` | импорт `:47`; `:1622`,`:1636`,`:2072`,`:2093`,`:2179`? нет — только в 5-членном списке `:1589`; отдельно узкий список `(O_MODELS+GOOGLE+PERPLEXITY)` в `:1622`,`:1636`,`:2072`,`:2093`,`:2194`,`:2561`; `telegram_bot.py:45,4224,4906` | см. `O_MODELS`/`GOOGLE` — те же ветки, тот же вывод | см. `O_MODELS`/`GOOGLE` | см. `O_MODELS`/`GOOGLE` |
| `MOONSHOTAI` | импорт `:48`; `:1589`,`:2179` (см. `O_MODELS`); `:3495` (см. `MISTRALAI`) | недостижима | см. `MISTRALAI` | см. `MISTRALAI` |
| `QWEN` | импорт `:49`; `:1589`,`:2179` (см. `O_MODELS`) | недостижима | входит только в форсирующую `max_completion_tokens`-ветку | удаляется вместе с `:1589`/`:2179` |

Компактно: **12 констант — все пустые, все 15 использующих их условий сегодня дают ровно один и
тот же результат независимо от модели.** Ни одна ветка не наблюдалась исполняющейся «два месяца»
(П2) — это структурное свойство кода (пустой кортеж), а не только наблюдение из логов.

## Дизайн

### Вариант (а): точечное удаление, без новой абстракции — выбран

Вариант (б) («ввести единый `model_capabilities(model) -> ModelCaps` с переопределением через
конфиг, по образцу `MODEL_CONTEXT_WINDOWS`») — рассмотрен и отклонён для текущего объёма задачи:

1. **Нет ни одного заполненного примера.** Все 12 констант — `()`, и это не значения по
   умолчанию из env (как `MODEL_CONTEXT_WINDOWS`, `bot/__main__.py:120-156`, которая читает CSV
   из окружения) — это буквально хардкод в `bot/model_constants.py`. Сегодня админ не может
   заполнить эти списки без правки исходного кода. Проектировать таблицу «на будущее» без единого
   реального кейса — это ровно то, от чего предостерегает П2 («не наполнять кортежи из env ‘на
   всякий случай’») и правило Simplicity First («No … ‘flexibility’ … that wasn't requested»).
2. **Архитектурная причина, почему кейсов и не было.** Комментарий в самом файле
   (`bot/model_constants.py:17-18`): «Provider groups are kept for compatibility with older
   helper checks. Runtime model selection is configured through `OPENAI_MODEL`» — модель всегда
   называется `llmgateway/high`/`llmgateway/light_model`/… (см. `README.md`, раздел
   `OPENAI_MODEL`), а какой физический провайдер стоит за этим именем — решает шлюз, а не бот.
   Поэтому «семейство по имени модели» в этой архитектуре структурно не может быть источником
   правды: бот не видит реального провайдера.
3. **Нечего сохранять.** Задача формулировки — «определить достижима ли ветка» — и ответ для
   всех 15 веток один: нет (или тождественно да). Нет ни одной ветки с реальной вариативностью,
   которую стоило бы «поднять» в capability-таблицу — только код, который эмулирует
   вариативность синтаксисом, но не поведением.
4. Если/когда появится реальный второй бэкенд не через шлюз (например, прямой вызов Google API в
   обход `llmgateway`), проектировать capability-конфиг стоит тогда — под конкретные, а не
   гипотетические требования этого бэкенда. Это ровно порядок, к которому подталкивает Simplicity
   First («No abstractions for single-use code»).

Итоговое правило замены — **для каждой ветки взять то единственное поведение, которое она уже
выдаёт сегодня для любого `model_to_use`, и оставить только его**, без переключателя. Это
поведенчески-нейтральный рефакторинг (эквивалент до/после для всех входов), а не смена логики.

### Побочный эффект удаления: два метода теряют мёртвую переменную

`should_force_non_stream_first_turn` и `should_force_non_stream_first_turn_async`
(`bot/openai_helper.py:2043-2095`) сегодня вычисляют `model_to_use` (через
`self.get_current_model(...)`/`self.get_current_model_async(...)`, с `try/except`-фолбэком на
`self.config.get("model", "")`) **только для того**, чтобы сравнить его с
`(O_MODELS + GOOGLE + PERPLEXITY)`. Больше `model_to_use` в этих двух функциях нигде не
используется. После удаления сравнения вся конструкция вычисления `model_to_use` (включая вызов
`get_current_model`/`get_current_model_async`, который может стоить чтения из БД/кэша сессии)
становится осиротевшей и должна быть удалена вместе с веткой — это не «дополнительная чистка
соседнего кода», а прямое следствие правки (AGENTS.md: «Remove imports/variables/functions that
YOUR changes made unused»). После удаления обе функции сводятся к трём проверкам без обращения к
текущей модели вообще — то есть T16 попутно убирает лишний (и потенциально небесплатный) вызов
`get_current_model[_async]` на пути, где раньше он тратился только на недостижимую/всегда-True
проверку.

`_retry_empty_response_with_tools` (`:2194`) — не тот случай: там `model_to_use` используется
и дальше (`get_functions_specs`, `get_max_tokens`), удаляется только сама проверка `if ... :
return None`, переменная остаётся жить.

### `_format_specs_for_model` (`bot/plugin_manager.py:349-359`) — Google-ветка

`AGENTS.md` уже фиксирует этот факт как известное: «the Google branch is currently unreachable
because `GOOGLE_MODELS` is an alias for `GOOGLE`… which is an empty tuple». T16 просто выполняет
удаление, о котором AGENTS.md предупреждал. После удаления функция всегда возвращает
`[{"type": "function", "function": spec} for spec in model_specs]` — параметр `model_to_use`
в `_format_specs_for_model` становится неиспользуемым и тоже удаляется из сигнатуры (нужно
проверить оба вызывающих места — см. «Правки»).

Внизу пайплайна (`bot/openai_tool_handler.py`) есть generic-обработчики формы
`{"function_declarations": [...]}` (`_filter_tools_by_name`, `_has_tool_specs`) и тест на них
(`tests/test_openai_helper_tool_calls.py:3382` `test_tool_suppression_filters_google_function_declarations`,
`tests/test_openai_compatible_provider.py:109-140`
`test_openai_compatible_provider_preserves_dict_tool_shape`) — они строят такой словарь **вручную
в тесте**, не через `_format_specs_for_model`, и не импортируют `GOOGLE`/`GOOGLE_MODELS`. Это
защитный код на случай словаря такой формы откуда угодно (например, будущего прямого Google-
провайдера в `bot/ai_providers/`, которого сегодня в дереве нет). Он вне области T16 — не
трогаем, тесты не меняются.

## Правки по `file:line`

### 1. `bot/model_constants.py:17-30` — удалить комментарий и все 12 констант-семейств

Было (`bot/model_constants.py`, полностью):
```python
from __future__ import annotations

LLMGATEWAY_HIGH_MODEL = "llmgateway/high"
LLMGATEWAY_LIGHT_MODEL = "llmgateway/light_model"
LLMGATEWAY_BIG_CONTEXT_MODEL = "llmgateway/big_context"

# Фолбэк-лимит max_tokens для одного запроса генерации, если в конфиге не задан
# ключ ``output_max_tokens``. Применяется как общий клампинг в get_max_tokens и
# как явное значение там, где параметр иначе не задан.
MAX_OUTPUT_TOKENS = 65535

LLMGATEWAY_WEB_SEARCH_MODEL = "llmgateway/web-search"
LLMGATEWAY_WEB_READ_MODEL = "llmgateway/web-read"
LLMGATEWAY_WEB_RESEARCH_MODEL = "llmgateway/web-research"
LLMGATEWAY_WEB_DEEP_RESEARCH_MODEL = "llmgateway/web-deep-research"

# Provider groups are kept for compatibility with older helper checks.
# Runtime model selection is configured through OPENAI_MODEL.
GPT_4_VISION_MODELS = ()
GPT_4O_MODELS = ()
GPT_5_MODELS = ()
O_MODELS = ()
ANTHROPIC = ()
GOOGLE = ()
MISTRALAI = ()
DEEPSEEK = ()
LLAMA = ()
PERPLEXITY = ()
MOONSHOTAI = ()
QWEN = ()
```

Стало:
```python
from __future__ import annotations

LLMGATEWAY_HIGH_MODEL = "llmgateway/high"
LLMGATEWAY_LIGHT_MODEL = "llmgateway/light_model"
LLMGATEWAY_BIG_CONTEXT_MODEL = "llmgateway/big_context"

# Фолбэк-лимит max_tokens для одного запроса генерации, если в конфиге не задан
# ключ ``output_max_tokens``. Применяется как общий клампинг в get_max_tokens и
# как явное значение там, где параметр иначе не задан.
MAX_OUTPUT_TOKENS = 65535

LLMGATEWAY_WEB_SEARCH_MODEL = "llmgateway/web-search"
LLMGATEWAY_WEB_READ_MODEL = "llmgateway/web-read"
LLMGATEWAY_WEB_RESEARCH_MODEL = "llmgateway/web-research"
LLMGATEWAY_WEB_DEEP_RESEARCH_MODEL = "llmgateway/web-deep-research"
```

### 2. `bot/openai_helper.py:36-50` — импорт

Было:
```python
from .model_constants import (
    LLMGATEWAY_BIG_CONTEXT_MODEL,
    LLMGATEWAY_HIGH_MODEL,
    LLMGATEWAY_LIGHT_MODEL,
    MAX_OUTPUT_TOKENS,
    GPT_4O_MODELS,
    O_MODELS,
    ANTHROPIC,
    GOOGLE,
    MISTRALAI,
    DEEPSEEK,
    PERPLEXITY,
    MOONSHOTAI,
    QWEN,
)
```
Стало:
```python
from .model_constants import (
    LLMGATEWAY_BIG_CONTEXT_MODEL,
    LLMGATEWAY_HIGH_MODEL,
    LLMGATEWAY_LIGHT_MODEL,
    MAX_OUTPUT_TOKENS,
)
```

### 3. `bot/openai_helper.py:1580-1605` (`__common_get_chat_response`) — форс `max_completion_tokens`

Было (даёт единственный реально исполняемый результат уже сегодня — `else`-ветка):
```python
            common_args = {
                'model': model_to_use,
                'messages': messages,
                'temperature': temperature,
                'n': 1, # several choices is not implemented yet
                'max_tokens': max_tokens,
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream,
                'extra_headers': { "X-Title": "tgBot" },
            }

            if model_to_use in (O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + PERPLEXITY + MOONSHOTAI + QWEN):
                stream = False
                common_args['stream'] = False

                #common_args['messages'] = [msg for msg in common_args['messages'] if msg['role'] != 'system']
                common_args['max_completion_tokens'] = max_tokens # o1 series only supports max_completion_tokens
                common_args.pop('max_tokens', None)
         
                # 'temperature', 'top_p', 'n', 'presence_penalty', 'frequency_penalty' are currently fixed and cannot be changed
            else:
                # Parameters for other models
                common_args.update({
                    'temperature': temperature,
                    'n': self.config['n_choices'],
                    'max_tokens': max_tokens,
                    'presence_penalty': self.config['presence_penalty'],
                    'frequency_penalty': self.config['frequency_penalty'],
                    'stream': stream,
                    'extra_headers': { "X-Title": "tgBot" },
                })
```
Стало (свернуто в единственный исполняемый путь — `n` сразу берётся из конфига, `if`/`else`
исчезают):
```python
            common_args = {
                'model': model_to_use,
                'messages': messages,
                'temperature': temperature,
                'n': self.config['n_choices'],
                'max_tokens': max_tokens,
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream,
                'extra_headers': { "X-Title": "tgBot" },
            }
```
Проверить: единственное различие исходного `common_args` (до `if`) и `else`-ветки — поле `n`
(`1` против `self.config['n_choices']`); `else` выполняется всегда (условие `if` недостижимо),
поэтому итоговое значение `n` в проде сегодня — всегда `self.config['n_choices']`. Правка это
воспроизводит.

### 4. `bot/openai_helper.py:1622` — прикрепление `tools`

Было:
```python
                if tools and model_to_use not in (O_MODELS + GOOGLE + PERPLEXITY):
                    common_args['tools'] = tools
                    common_args['tool_choice'] = 'auto'
```
Стало:
```python
                if tools:
                    common_args['tools'] = tools
                    common_args['tool_choice'] = 'auto'
```

### 5. `bot/openai_helper.py:1629-1645` — комментарий + `gate_supported_model`

Было:
```python
            # skills_agent first-turn planner gate. Only runs on non-streaming
            # first turns; the streaming dispatch path is expected to route to
            # non-streaming via OpenAIHelper.should_force_non_stream_first_turn
            # before reaching here. Excluded model families (O_MODELS, Google,
            # Perplexity) don't get function calling above and are skipped here too.
            # The gate is gated by the mode's ``force_non_stream_first_turn`` flag
            # in chat_modes.yml — this keeps the entire feature opt-in per-mode and
            # avoids surprising tests/users that rely on the legacy direct path.
            gate_supported_model = model_to_use not in (O_MODELS + GOOGLE + PERPLEXITY)
            gate_active = (
                not common_args.get('stream')
                and self.config.get('enable_functions', True)
                and bool(common_args.get('tools'))
                and gate_supported_model
                and self._is_skills_agent_mode(chat_id)
                and self._skills_agent_gate_enabled_for_mode()
                and not self._skills_agent_has_plan(chat_id, memory_user_id)
            )
```
Стало:
```python
            # skills_agent first-turn planner gate. Only runs on non-streaming
            # first turns; the streaming dispatch path is expected to route to
            # non-streaming via OpenAIHelper.should_force_non_stream_first_turn
            # before reaching here.
            # The gate is gated by the mode's ``force_non_stream_first_turn`` flag
            # in chat_modes.yml — this keeps the entire feature opt-in per-mode and
            # avoids surprising tests/users that rely on the legacy direct path.
            gate_active = (
                not common_args.get('stream')
                and self.config.get('enable_functions', True)
                and bool(common_args.get('tools'))
                and self._is_skills_agent_mode(chat_id)
                and self._skills_agent_gate_enabled_for_mode()
                and not self._skills_agent_has_plan(chat_id, memory_user_id)
            )
```

### 6. `bot/openai_helper.py:1753` — `_uses_structured_tool_history`

Было:
```python
    def _uses_structured_tool_history(self, model_to_use: str) -> bool:
        return model_to_use in GPT_4O_MODELS or model_to_use in self.get_model_choices()
```
Стало:
```python
    def _uses_structured_tool_history(self, model_to_use: str) -> bool:
        return model_to_use in self.get_model_choices()
```

### 7. `bot/openai_helper.py:2043-2095` — `should_force_non_stream_first_turn` / `_async`

Было:
```python
    def should_force_non_stream_first_turn(self, chat_id: int, user_id: int | None) -> bool:
        """Whether the dispatcher should route this request through non-streaming
        get_chat_response to give the skills_agent planner gate a chance to fire.

        Returns True iff: (1) current mode is skills_agent, (2) the mode has the
        ``force_non_stream_first_turn`` flag, (3) the active model belongs to a
        family that supports tool_choice (i.e. not O_MODELS / GOOGLE / PERPLEXITY —
        those skip tools above and would never fire the gate anyway), and (4) no
        plan exists yet for the scope. If any condition is False, the dispatcher
        streams as usual.
        """
        if not self._is_skills_agent_mode(chat_id):
            return False
        if not self._skills_agent_gate_enabled_for_mode():
            return False
        try:
            session_owner = user_id if user_id is not None else chat_id
            model_to_use = self.get_current_model(session_owner)
        except Exception as exc:
            logger.debug(
                "skills_agent gate: get_current_model failed error=%s",
                log_exception_shape(exc),
            )
            # Fall back to the configured default model so a future expansion of
            # the excluded-family lists still skips streaming overrides cleanly.
            try:
                model_to_use = self.config.get("model", "") or ""
            except Exception:
                model_to_use = ""
        if model_to_use in (O_MODELS + GOOGLE + PERPLEXITY):
            return False
        return not self._skills_agent_has_plan(chat_id, user_id)

    async def should_force_non_stream_first_turn_async(self, chat_id: int, user_id: int | None) -> bool:
        if not self._is_skills_agent_mode(chat_id):
            return False
        if not self._skills_agent_gate_enabled_for_mode():
            return False
        try:
            session_owner = user_id if user_id is not None else chat_id
            model_to_use = await self.get_current_model_async(session_owner)
        except Exception as exc:
            logger.debug(
                "skills_agent gate: get_current_model_async failed error=%s",
                log_exception_shape(exc),
            )
            try:
                model_to_use = self.config.get("model", "") or ""
            except Exception:
                model_to_use = ""
        if model_to_use in (O_MODELS + GOOGLE + PERPLEXITY):
            return False
        return not self._skills_agent_has_plan(chat_id, user_id)
```
Стало:
```python
    def should_force_non_stream_first_turn(self, chat_id: int, user_id: int | None) -> bool:
        """Whether the dispatcher should route this request through non-streaming
        get_chat_response to give the skills_agent planner gate a chance to fire.

        Returns True iff: (1) current mode is skills_agent, (2) the mode has the
        ``force_non_stream_first_turn`` flag, and (3) no plan exists yet for the
        scope. If any condition is False, the dispatcher streams as usual.
        """
        if not self._is_skills_agent_mode(chat_id):
            return False
        if not self._skills_agent_gate_enabled_for_mode():
            return False
        return not self._skills_agent_has_plan(chat_id, user_id)

    async def should_force_non_stream_first_turn_async(self, chat_id: int, user_id: int | None) -> bool:
        if not self._is_skills_agent_mode(chat_id):
            return False
        if not self._skills_agent_gate_enabled_for_mode():
            return False
        return not self._skills_agent_has_plan(chat_id, user_id)
```
Примечание: `user_id` в синхронной версии раньше не участвовал в вычислении `model_to_use`
(участвовал только `session_owner`, локальная переменная) — убеждаемся, что `user_id` остаётся
использованным в `self._skills_agent_has_plan(chat_id, user_id)`, поэтому сигнатуру функции не
трогаем.

### 8. `bot/openai_helper.py:2179` (`_retry_empty_response_after_tools`) — тот же форс `max_completion_tokens`

Было:
```python
        common_args = {
            'model': model_to_use,
            'messages': messages,
            'temperature': self.config['temperature'],
            'n': 1,
            'max_tokens': max_tokens,
            'presence_penalty': self.config['presence_penalty'],
            'frequency_penalty': self.config['frequency_penalty'],
            'stream': False,
            'extra_headers': { "X-Title": "tgBot" },
        }
        if model_to_use in (O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + PERPLEXITY + MOONSHOTAI + QWEN):
            common_args['max_completion_tokens'] = max_tokens
            common_args.pop('max_tokens', None)
        return await self._create_empty_response_retry_completion("after_tools", **common_args)
```
Стало:
```python
        common_args = {
            'model': model_to_use,
            'messages': messages,
            'temperature': self.config['temperature'],
            'n': 1,
            'max_tokens': max_tokens,
            'presence_penalty': self.config['presence_penalty'],
            'frequency_penalty': self.config['frequency_penalty'],
            'stream': False,
            'extra_headers': { "X-Title": "tgBot" },
        }
        return await self._create_empty_response_retry_completion("after_tools", **common_args)
```

### 9. `bot/openai_helper.py:2194` (`_retry_empty_response_with_tools`) — ранний `return None`

Было:
```python
        model_owner = chat_id if session_id else (user_id if user_id is not None else chat_id)
        model_to_use = model_to_use or await self.get_current_model_async(model_owner, session_id=session_id)
        if model_to_use in (O_MODELS + GOOGLE + PERPLEXITY):
            return None

        tools = self.plugin_manager.get_functions_specs(self, model_to_use, allowed_plugins)
```
Стало:
```python
        model_owner = chat_id if session_id else (user_id if user_id is not None else chat_id)
        model_to_use = model_to_use or await self.get_current_model_async(model_owner, session_id=session_id)

        tools = self.plugin_manager.get_functions_specs(self, model_to_use, allowed_plugins)
```

### 10. `bot/openai_helper.py:2561` (`__common_get_chat_response_vision`) — прикрепление `tools`

Было:
```python
                if tools and vision_model not in (O_MODELS + GOOGLE + PERPLEXITY):
                    common_args['tools'] = tools
                    common_args['tool_choice'] = 'auto'
```
Стало:
```python
                if tools:
                    common_args['tools'] = tools
                    common_args['tool_choice'] = 'auto'
```

### 11. `bot/openai_helper.py:3466-3506` (`__add_function_call_to_history`) — недостижимые ветки роли

Было:
```python
        # Some providers either don't support (or inconsistently support) the legacy "function" role.
        # For those, we inject tool results as regular user text so the model reliably sees them.
        if model_to_use in (ANTHROPIC + DEEPSEEK):
            function_result = f"Function {function_name} returned: {content}"
            self.conversations[state_key].append({"role": "user", "content": function_result})
        elif model_to_use in (MISTRALAI + MOONSHOTAI):
            # Mistral и Moonshot используют роль "tool" вместо "function"
            self.conversations[state_key].append({
                "role": "tool",
                "name": model_function_name,
                "content": content,
            })
        elif model_to_use in (O_MODELS + GPT_4O_MODELS) or model_to_use in self.get_model_choices():
            # For all other models (OpenAI-style), use the assistant role instead of deprecated function role
            # The 'function' role is no longer supported in OpenAI API as of 2025
            function_result = f"Function {function_name} returned: {content}"
            self.conversations[state_key].append({"role": "assistant", "content": function_result})
        else:
            # For OpenAI-style models, use the function role
            self.conversations[state_key].append({
                "role": "function",
                "name": model_function_name,
                "content": content,
            })
```
Стало:
```python
        if model_to_use in self.get_model_choices():
            # For all other models (OpenAI-style), use the assistant role instead of deprecated function role
            # The 'function' role is no longer supported in OpenAI API as of 2025
            function_result = f"Function {function_name} returned: {content}"
            self.conversations[state_key].append({"role": "assistant", "content": function_result})
        else:
            # For OpenAI-style models, use the function role
            self.conversations[state_key].append({
                "role": "function",
                "name": model_function_name,
                "content": content,
            })
```
Наблюдаемое поведение не меняется: первые две ветки (`ANTHROPIC+DEEPSEEK`, `MISTRALAI+MOONSHOTAI`)
никогда не выполнялись (пустые кортежи); третья тождественно сводится к
`model_to_use in self.get_model_choices()`; финальный `else` — без изменений.

### 12. `bot/openai_helper.py:4269` (`get_max_tokens`) — недостижимый кап `32768`

Было:
```python
        if model_to_use in GPT_4O_MODELS:
            max_generation_tokens = 32768
```
Стало: строки удаляются, следующий код (жёсткая верхняя граница из конфига) остаётся без
изменений.

### 13. `bot/telegram_bot.py:45` — импорт

Было:
```python
from .openai_helper import OpenAIHelper, O_MODELS, ANTHROPIC, GOOGLE, MISTRALAI, DEEPSEEK, PERPLEXITY
```
Стало:
```python
from .openai_helper import OpenAIHelper
```
Проверить перед правкой: `rg`/`grep`-поиском (лучше — маленьким `python3`-скриптом, как в этом
плане) подтвердить, что `O_MODELS`/`ANTHROPIC`/`GOOGLE`/`MISTRALAI`/`DEEPSEEK`/`PERPLEXITY` в
`bot/telegram_bot.py` больше нигде не используются, кроме строк `:4224` и `:4906` (проверено на
момент написания плана — да, только там).

### 14. `bot/telegram_bot.py:4223-4227` (`_process_message_locked` → `_edit`/`_describe`)

Было:
```python
            force_non_stream = await self._should_force_non_stream_first_turn(chat_id, user_id)
            if (
                self.config['stream']
                and model_to_use not in (O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + DEEPSEEK + PERPLEXITY)
                and not force_non_stream
            ):
```
Стало:
```python
            force_non_stream = await self._should_force_non_stream_first_turn(chat_id, user_id)
            if self.config['stream'] and not force_non_stream:
```
Это прямая реализация формулировки из T16: «Решение о стриме — из конфига, не из семейства
модели» — после правки решение буквально и есть только `self.config['stream']` (env `STREAM`,
`bot/__main__.py`, default `true`) плюс гейт `force_non_stream`.

### 15. `bot/telegram_bot.py:4905-4909` (`handle_callback_inline_query`)

Было:
```python
                inline_force_non_stream = await self._should_force_non_stream_first_turn(user_id, user_id)
                if (
                    self.config['stream']
                    and model_to_use not in (O_MODELS + ANTHROPIC + GOOGLE + MISTRALAI + DEEPSEEK + PERPLEXITY)
                    and not inline_force_non_stream
                ):
```
Стало:
```python
                inline_force_non_stream = await self._should_force_non_stream_first_turn(user_id, user_id)
                if self.config['stream'] and not inline_force_non_stream:
```

### 16. `bot/plugin_manager.py:20` — импорт-алиас

Было:
```python
from .model_constants import GOOGLE as GOOGLE_MODELS
```
Стало: строка удаляется.

### 17. `bot/plugin_manager.py:349-359` (`_format_specs_for_model`)

Было:
```python
    def _format_specs_for_model(self, specs, model_to_use):
        """
        Wrap function specs in the envelope expected by the target provider.
        Google models use {"function_declarations": [...]}, OpenAI-compatible
        models use the [{"type": "function", "function": {...}}] form.
        """
        model_specs = [self._spec_for_model(spec) for spec in specs]
        if model_to_use in GOOGLE_MODELS:
            return {"function_declarations": model_specs}
        return [{"type": "function", "function": spec} for spec in model_specs]
```
Стало:
```python
    def _format_specs_for_model(self, specs):
        """
        Wrap function specs in the envelope expected by the target provider.
        Currently only the OpenAI-compatible [{"type": "function", "function": {...}}]
        form is produced; every model reaches this gateway as an OpenAI-compatible
        alias (see bot/model_constants.py).
        """
        model_specs = [self._spec_for_model(spec) for spec in specs]
        return [{"type": "function", "function": spec} for spec in model_specs]
```
Требуется найти и поправить единственный вызов `self._format_specs_for_model(all_specs,
model_to_use)` (внутри `get_functions_specs`, тот же файл) — убрать второй аргумент. Проверить
командой из раздела «Команды проверки», нет ли других вызовов (`_format_specs_for_model(` по
`bot/`).

## Тесты

### Удалить целиком (тестируют функциональность, которую этот план убирает)

- `tests/test_openai_helper_tool_calls.py:516-536` —
  `test_provider_family_branch_forces_non_streaming_request`. Патчит
  `openai_helper_module.O_MODELS` (несуществующий после правки 2 атрибут — `monkeypatch.setattr`
  на несуществующий атрибут кидает `AttributeError`). Тестирует ветку из правки 3, которая
  становится недостижимой и удаляется.
- `tests/test_openai_helper_tool_calls.py:3111-3128` —
  `test_empty_response_after_tools_provider_family_uses_completion_tokens`. То же самое, для
  ветки из правки 8.
- `tests/test_skills_agent_gate.py:628-644` — `test_gate_e2e_excluded_model_family_skipped`.
  Импортирует `PERPLEXITY` из `bot.openai_helper` и сам себя скипает, если кортеж пуст
  (`pytest.skip("PERPLEXITY tuple is empty in this build")`) — то есть тест сегодня **всегда**
  скипается и уже сейчас не даёт покрытия. После удаления константы импорт сломается
  (`ImportError`), а сама концепция «excluded model family» перестаёт существовать в коде.

### Править (используют константы как реализацию, не как предмет теста)

- `tests/test_skills_agent_gate.py:435-475` — вспомогательная `_invoke_common`, которая
  дублирует gate-блок из `__common_get_chat_response`, чтобы тестировать только гейт в изоляции.
  Убрать `from bot.openai_helper import O_MODELS, GOOGLE, PERPLEXITY` (:451) и строку
  `gate_supported_model = model not in (O_MODELS + GOOGLE + PERPLEXITY)` (:453), убрать член
  `and gate_supported_model` из `gate_active` (:457) — зеркалит правку 5. Проверить остальные
  тесты этого файла, которые вызывают `_invoke_common(...)` — они не передают `model=` с расчётом
  на исключение по семейству (кроме уже удаляемого `test_gate_e2e_excluded_model_family_skipped`),
  так что после правки должны продолжать проходить без изменений в самих проверках.

### Подтверждено — не требуют изменений

- `tests/test_openai_helper_tool_calls.py:3382` `test_tool_suppression_filters_google_function_declarations`
  и `tests/test_openai_compatible_provider.py:109-140`
  `test_openai_compatible_provider_preserves_dict_tool_shape` — строят словарь
  `{"function_declarations": [...]}` вручную, не через `_format_specs_for_model`/`GOOGLE_MODELS`.
  Не импортируют ничего из `model_constants`. Не затрагиваются правкой 17.
- `tests/test_agent_tools_plugin.py:21` (`from bot.model_constants import MAX_OUTPUT_TOKENS`) и
  `tests/test_llm_gateway_routing.py:5` (`from bot.model_constants import
  LLMGATEWAY_LIGHT_MODEL`) — импортируют константы вне области T16 (не «семейства»), файл
  `model_constants.py` эти имена сохраняет без изменений.
- Любые тесты, которые просто передают `model="gpt-4o"` / другие имена как строковый параметр, не
  импортируя константы — их поведение не зависит от содержимого мёртвых кортежей и не меняется.

## Команды проверки

```bash
# 0. Окружение — venv проекта уже даёт нужный python; rg/grep иногда искажают вывод в этой
#    среде, поэтому для точечных грепов в этом плане использовался python3, не rg/grep.
PY=~/.venvs/ctb/bin/python

# 1. baseline — тесты, которые правка затронет напрямую, зелёные до правок
$PY -m pytest tests/test_openai_helper_tool_calls.py tests/test_skills_agent_gate.py \
  tests/test_plugin_manager.py -q -p no:cacheprovider

# 2. после удаления констант — быстрый структурный чек: убедиться, что в bot/ не осталось
#    ссылок на удалённые имена (ожидается пустой вывод)
$PY - <<'EOF'
import re, pathlib
names = ["GPT_4_VISION_MODELS","GPT_4O_MODELS","GPT_5_MODELS","O_MODELS","ANTHROPIC",
         "GOOGLE_MODELS","MISTRALAI","DEEPSEEK","LLAMA","PERPLEXITY","MOONSHOTAI","QWEN"]
pat = re.compile(r'\b(' + '|'.join(names) + r')\b')
hits = []
for root in ("bot", "tests", "bot/tests"):
    p = pathlib.Path(root)
    if not p.exists():
        continue
    for f in p.rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if pat.search(line):
                hits.append(f"{f}:{i}: {line.strip()}")
print("\n".join(hits) or "OK: ссылок не найдено")
EOF

# 3. GOOGLE не должен встречаться отдельно от удалённого импорта в telegram_bot/openai_helper —
#    отдельная проверка, т.к. GOOGLE короче и легче ложно матчится (например, в комментариях)
$PY -c "import re, pathlib
for f in ['bot/openai_helper.py','bot/telegram_bot.py','bot/plugin_manager.py']:
    text = pathlib.Path(f).read_text(encoding='utf-8')
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(r'\bGOOGLE\b', line):
            print(f'{f}:{i}: {line.strip()}')"

# 4. модуль всё ещё импортируется и рабочий (быстрый smoke)
$PY -c "from bot import model_constants; print(sorted(n for n in dir(model_constants) if n.isupper()))"

# 5. целевые тесты после правок (с новыми/удалёнными тестами)
$PY -m pytest tests/test_openai_helper_tool_calls.py tests/test_skills_agent_gate.py \
  tests/test_plugin_manager.py -q -p no:cacheprovider

# 6. более широкий прогон — всё, что трогает стриминг/тулы/vision/историю функций
$PY -m pytest tests/test_telegram_streaming.py tests/test_openai_helper_db_offload.py \
  tests/test_group_session_flow.py tests/test_callback_authorization.py \
  tests/test_openai_compatible_provider.py -q -p no:cacheprovider

# 7. полный прогон без evals/ (см. AGENTS.md Testing And Verification)
$PY -m pytest -q -p no:cacheprovider
```

## Риски

- **Ложное ощущение «мультипровайдерности».** Удаление веток убирает единственное место в
  коде, где было явно написано «вот так бы вело себя для Anthropic/Google/...». Если в
  будущем реально подключат нешлюзовой бэкенд, этот код придётся писать заново — но заново
  писать его нужно будет всё равно, так как сегодняшние ветки никогда не были протестированы
  на реальном трафике (значения зашиты как `()`, никто их не наполнял через конфиг ни разу за
  всё время существования файла).
- **`should_force_non_stream_first_turn` перестаёт трогать `get_current_model`.** Раньше функция
  «нащупывала» текущую модель на каждый вызов (правка 7). Если где-то до этого неявно
  рассчитывали на побочный эффект этого вызова (например, ленивую инициализацию
  сессии/кэша) — эффект пропадёт. Проверено чтением тела `get_current_model`/`_async`
  (`bot/openai_helper.py`) на предмет побочных эффектов помимо чтения — на момент анализа
  побочных эффектов, важных для остального кода, не найдено, но стоит перепроверить при
  ревью правки, а не полагаться только на этот план.
- **Тест `test_gate_e2e_excluded_model_family_skipped` удаляется, а не чинится.** Он и
  сегодня не даёт покрытия (всегда skip), так что риск регрессии от удаления — нулевой; риск
  в другом: если ревьюер решит, что тест документирует *намерение* («когда-нибудь допилим
  реальный список исключений») и должен остаться как TODO — стоит явно обсудить перед
  удалением, а не удалять молча.
- **`_format_specs_for_model` меняет сигнатуру** (правка 17: убирается параметр
  `model_to_use`). Нужно грепом подтвердить единственность вызывающего места
  (`get_functions_specs`, тот же файл) — если есть ещё вызовы (например, в тестах,
  монки-патчащих сам метод), их тоже нужно поправить; в текущей проверке по `tests/` не найдено,
  но стоит перепроверить на актуальном дереве перед мержем, так как номера строк в этом плане
  уже могут сместиться другими правками.
- **Комментарий `bot/model_constants.py:17-18`** («Provider groups are kept for compatibility…»)
  удаляется вместе с константами, которые он описывал — если кто-то отдельно ссылался на этот
  комментарий как на документацию решения (например, в другом plan-документе), ссылка устареет;
  в проверенных `docs/*.md` таких ссылок на текст комментария не найдено.

## Критерии готовности

- `bot/model_constants.py` содержит только `LLMGATEWAY_*` и `MAX_OUTPUT_TOKENS`; ни одной из 12
  констант-семейств (`GPT_4_VISION_MODELS`, `GPT_4O_MODELS`, `GPT_5_MODELS`, `O_MODELS`,
  `ANTHROPIC`, `GOOGLE`, `MISTRALAI`, `DEEPSEEK`, `LLAMA`, `PERPLEXITY`, `MOONSHOTAI`, `QWEN`)
  нет ни в файле, ни в импортах `bot/openai_helper.py`, `bot/telegram_bot.py`,
  `bot/plugin_manager.py`.
- Команда проверки №2 из раздела «Команды проверки» (грep по именам констант в `bot/`, `tests/`,
  `bot/tests/`) возвращает `OK: ссылок не найдено`.
- `bot/plugin_manager.py::_format_specs_for_model` всегда возвращает
  `[{"type": "function", "function": spec} for spec in model_specs]`; вызывающий код обновлён под
  новую сигнатуру (без `model_to_use`).
- `bot/telegram_bot.py`: оба места принятия решения о стриминге (`_process_message_locked`,
  `handle_callback_inline_query`) читают только `self.config['stream']` и соответствующий
  `force_non_stream`/`inline_force_non_stream` — без упоминания семейств моделей.
- `should_force_non_stream_first_turn`/`_async` не вызывают `get_current_model`/`_async` и не
  объявляют `model_to_use`.
- Удалённые тесты (`test_provider_family_branch_forces_non_streaming_request`,
  `test_empty_response_after_tools_provider_family_uses_completion_tokens`,
  `test_gate_e2e_excluded_model_family_skipped`) отсутствуют в дереве; `_invoke_common` в
  `tests/test_skills_agent_gate.py` не импортирует `O_MODELS`/`GOOGLE`/`PERPLEXITY`.
- Все команды из раздела «Команды проверки» (шаги 4-7) проходят зелёным; полный
  `pytest -q -p no:cacheprovider` (без `evals/`) зелёный.
- Диф не содержит правок вне файлов, перечисленных в разделе «Правки» и «Тесты» (surgical-правило
  из `AGENTS.md`/`CLAUDE.md`) — в частности, `LLMGATEWAY_*`/`MAX_OUTPUT_TOKENS` и их потребители
  не тронуты.

## Постскриптум после ревью

Ревью (Sonnet, персона reviewer) ошибок не нашло: для каждой свёрнутой ветки доказано, что
удалённая часть была недостижима (`x in ()` всегда `False`), словарь `common_args`,
`get_max_tokens` и структура записей `__add_function_call_to_history` не изменились ни для
одного достижимого случая; удалённые тесты покрывали только мёртвые ветки; единственный вызов
`_format_specs_for_model` обновлён. Полный прогон зелёный, `ruff` чист.

Замечания и их судьба:
- Синхронная `should_force_non_stream_first_turn` отсутствует в дереве — она была удалена ранее
  задачей T15 (единственный вызывающий код и на HEAD шёл через `*_async`). План T16 писался до
  этого; отклонение принято.
- Устаревший докстринг `tests/test_skills_agent_gate.py` («the model is not in the excluded
  families») — исправлен.
- `AGENTS.md` (описание Google-ветки в `_format_specs_for_model`) и
  `docs/tutorial/09_менеджер_плагинов.md:233` (пример с `GOOGLE_MODELS`) устарели — правка
  документации перенесена в T21.
- Параметр `model_to_use` у `get_functions_specs` больше не читается внутри, но сохранён:
  метод вызывается позиционно из многих мест, удаление — вне периметра.
