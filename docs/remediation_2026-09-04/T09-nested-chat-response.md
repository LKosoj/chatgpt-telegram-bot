# T09. Вложенный `get_chat_response` из плагинов

Источник задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел «T09»), находка
`docs/architecture_code_review_2026-09-04.md` §3.2 (HIGH, уверенность CERTAIN). Роль этого
документа — план для разработчика; код не менялся, только прочитан и воспроизведён репро-скриптом.

Термины: **ContextVar** — переменная, которая «едет» вместе с текущей async-задачей и
наследуется вложенными вызовами (`asyncio.create_task`/`asyncio.gather` копируют её значение
в момент создания новой задачи, а не по ссылке — более поздний сброс в исходной задаче
скопированную задачу уже не затрагивает). **deadlock** (взаимная блокировка) — сторона ждёт
ресурс, который может освободить только она сама. **guard** — проверка в начале функции,
которая отклоняет вызов, если предусловие нарушено, вместо того чтобы продолжать в
непредусмотренном состоянии.

## Цель

1. Сделать вложенный (nested) вызов `helper.get_chat_response()`/`get_chat_response_stream()`
   изнутри tool-call (вызова плагина моделью) невозможным: вместо зависания бота или порчи
   истории разговора — понятная ошибка сразу.
2. Перевести четыре плагина (`ask_your_pdf`, `language_learning`, `conversation_analytics`,
   `show_me_diagrams`), которые сегодня используют `get_chat_response` для одноразового
   («не показывать в истории») обращения к модели, на `helper.ask()` — метод, который для этого
   и предназначен и уже используется в `chief.py`, `movie_info.py` и в этом же
   `show_me_diagrams.py` (двумя строками ниже проблемных мест).
3. Обновить тесты: убрать мок на `get_chat_response` в `tests/test_ask_your_pdf.py`, заменить на
   мок `ask`; переписать тест, который сегодня *закрепляет* багу (`test_tool_execution_allows_
   nested_chat_response_with_same_chat_lock`), на тест, что guard её отклоняет; добавить
   прямые unit-тесты на сам guard.

## Репро

Скрипт `/tmp/hunt/repro_nested_lock.py` (в репозитории не хранится, лежит в `/tmp/hunt/`)
эмулирует реальный `OpenAIHelper`: внешний ход держит `_chat_lock(state_key)` и выставил
`_CHAT_STATE_KEY`, вложенный вызов идёт с другим `chat_id` (как в `ask_your_pdf`) и с тем же
`chat_id` (как в `show_me_diagrams`). Прогон (2026-09-04, без изменений в коде):

```
inner state_key resolves to: 100 (inner chat_id = -7195345581777275897 )
bypass enabled for inner: False
RESULT: DEADLOCK — nested get_chat_response never acquired the lock
CONTROL (bypass set to outer chat, inner chat differs): DEADLOCK
CONTROL (same chat id as bypass): completed, state_key used = 100
```

Это подтверждает оба сценария из §3.2 буквально:
- **Разный `chat_id`** (`ask_your_pdf`/`language_learning`/`conversation_analytics`, все — через
  `hash(...)`): `state_key = self._chat_state_key(chat_id)` (определение `_chat_state_key` —
  `bot/openai_helper.py:3170-3171`, `return _CHAT_STATE_KEY.get() or chat_id`)
  игнорирует переданный `chat_id`, потому что `_CHAT_STATE_KEY.get()` уже не `None` (истинно) —
  вложенный вызов лочит **тот же** `asyncio.Lock`, что уже держит внешний ход
  (`bot/openai_helper.py:893` / `:1141`, `_chat_lock`), и ждёт сам себя. Обход через
  `_chat_lock_bypass_enabled` (`:883`, `:1129`, определение `:3221-3223`) не срабатывает: bypass
  выставлен `_call_function_bounded` (читает `chat_id` из аргументов тула на
  `bot/openai_tool_handler.py:136`, применяет как bypass на `:158-159`) на **внешний**
  `chat_id` (тот, что инжектирован в аргументы тула), а сравнивается с синтетическим
  `hash(file_path)` — строки не совпадают.
- **Тот же `chat_id`** (`show_me_diagrams`): bypass совпадает, `async with lock` не блокирует —
  но `state_key`, под которым пишется ответ, разрешается в **внешний** `chat_id` (100 в
  контроле выше), то есть вложенный «служебный» вопрос модели («какой тип диаграммы?») попадает
  в историю **пользователя**, а не остаётся служебным. `ask()` (`:788-841`) от этой проблемы
  защищён собственным guard `_in_active_turn = _CHAT_STATE_KEY.get() is not None`
  (`bot/openai_helper.py:796`), который просто не пишет в историю внутри активного хода;
  `get_chat_response` такого guard не имеет вовсе — он обязан войти в
  `_get_chat_response_locked` и записать историю, потому что это его единственная задача.

Второй эффект «того же `chat_id`» — порча истории вызовов инструментов: вложенный вызов
вставляет `user`/`assistant` **между** уже записанным `assistant(tool_calls)` и ещё не
дописанным `tool`-ответом (тот появится в истории только после того, как внешний `asyncio.gather`
дождётся результата тула). На следующем запросе `_apply_before_chat_request_mutators`
(`bot/openai_helper.py:2914-2932`) вызывает `_repair_tool_call_history` (определение `:3039`,
тело `:3084-3138`), которая сканирует историю строго последовательно: сразу после `assistant`
с `tool_calls` (`:3085`) ожидает подряд идущие `tool`-сообщения, начиная цикл на `:3090` и
выходя из него, как только следующее сообщение — не `tool` (`break` на `:3092`); наткнувшись
на вставленный `user`, выходит из этого цикла раньше времени, помечает ожидаемые
`tool_call_id` как «потерянные» (`missing_ids` на `:3103`, заполняются
`INTERRUPTED_TOOL_RESULT_NOTICE` на `:3111-3120`), а настоящий `tool`-ответ, когда до него
доходит позже (уже вне ветки `assistant`-lookahead), обрабатывается веткой `:3129-3137`
(`"Dropping orphan tool result"`) как «сирота» и удаляется.

## Анализ: кто и как вызывает `get_chat_response` изнутри tool-call

| Плагин:строка | `chat_id` в вызове | Что делает | Есть ли `user_id` в `kwargs` | Замена |
|---|---|---|---|---|
| `ask_your_pdf.py:364-366` | `hash(file_path)` — синтетический, не совпадает с реальным | Одноразовый анализ текста PDF по `query`; результат = текст ответа, кладётся в `result` и в файловый кэш | Нет (не читается вовсе — баг сам по себе, `get_chat_response` получал `user_id=None`) | `helper.ask(analysis_prompt, user_id)` |
| `language_learning.py:187-190` | `hash(f"{language}_{level}_{exercise_type}")` — синтетический | Генерация текста упражнения; результат = текст, кладётся в `exercise["content"]` | Нет (не читается; в файле уже есть `_get_owner_user_id(kwargs)`, используется в соседней ветке `track_progress`) | `helper.ask(prompt, self._get_owner_user_id(kwargs))` |
| `conversation_analytics.py:233-236, 253-256, 271-274, 289-292` (4 вызова, один паттерн) | `hash(f"{chat_id}_topics"/"_learning"/"_format"/"_style")` — синтетический | Генерация текста рекомендаций по одной из 4 тем; результат — текст, добавляется в список `recommendations` | Нет (не читается; `kwargs['chat_id']` читается, `kwargs['user_id']` — нет) | `helper.ask(prompt, kwargs.get('user_id'))`, один раз на каждый из 4 вызовов |
| `show_me_diagrams.py:161-164` | `chat_id` (реальный, из `kwargs.get('chat_id') or 0`) | Уточняющий вопрос «какой тип диаграммы?», когда модель не передала `type`; результат — `.strip().lower()` сравнивается с `enum` типов | Да, `user_id = kwargs.get('user_id')` уже есть на `:149` | `helper.ask(type_prompt, user_id)` |
| `show_me_diagrams.py:186-189` | тот же `chat_id` | Уточняющий вопрос «опишите диаграмму подробнее», когда модель не передала `description` | Да (тот же `user_id`) | `helper.ask(description_prompt, user_id)` |

Ни один из шести вызовов не рассчитывает на память диалога, tool-calling внутри ответа или
конкретную модель — все передают самодостаточный текст в `query`/`prompt` и используют только
текст ответа (`response`, без usage/декораций). Это ровно то, для чего существует `ask()`.

**Почему не `ModelUtilities`.** `ModelUtilities` (`bot/model_utilities.py:16`) не выставлена на
`helper` как атрибут — она создаётся `ModelUtilities(self)` только внутри самого
`OpenAIHelper` (`bot/openai_helper.py:378, 3759, 4095`) для его собственных нужд (классификация
интента, сжатие истории, генерация заголовка). Ни один плагин в дереве её не использует; принятый
для плагинов способ одноразового вызова модели — `helper.ask(...)` (`chief.py:151,310`,
`movie_info.py:290`, и уже здесь же — `show_me_diagrams.py:244,292`). `ask()` — это уже принятый
в проекте локальный паттерн для такого вызова, а не новая абстракция.

**Почему не `record_plugin_exchange`.** `record_plugin_exchange` (`bot/openai_helper.py:3521`)
предназначен для обратного случая — когда плагин сам является основным обработчиком хода
(например, RAG-режим) и должен **вписать** свой обмен в историю сессии, чтобы следующий обычный
ход его видел. Здесь наоборот: вызовы внутри `execute()` — служебные, их не должно быть видно ни
в истории, ни в БД. Не подходит.

**Поведенческие отличия `ask()` от `get_chat_response()`, которые сознательно принимаются:**
- Модель: `ask()` по умолчанию берёт `self.config.get('model')` (тот же дефолт, что и у
  `get_chat_response`), но не умеет автоматически переключаться на `big_model_to_use` при
  нехватке контекстного окна (эта эскалация — часть `_get_chat_response_locked`/`ChatRun`,
  которой у `ask()` нет). Для коротких служебных промптов (анализ PDF, диаграммы,
  рекомендации) контекст маленький, переключение почти никогда не происходило бы и раньше.
- Температура: `ask()` жёстко использует `0.6` (`bot/openai_helper.py:836`), тогда как
  `get_chat_response` брал бы настроенную для чата температуру (по умолчанию в проде обычно
  `0.7-1.0`, см. `openai.config['temperature']`). Разница признаётся и не компенсируется —
  вводить в `ask()` параметр `temperature` ради четырёх плагинов было бы избыточной
  абстракцией под текущую задачу (нарушает «Simplicity First»); если разработчик решит иначе —
  можно добавить `temperature: float | None = None` в `ask()` отдельным, самостоятельно
  обсуждаемым изменением.
- Системный промпт: `ask()` без явного `assistant_prompt` подставляет собственный дефолт
  («Ты помошник, который отвечает на вопросы пользователя...», `bot/openai_helper.py:816-817`)
  вместо `self.config['assistant_prompt']` (обычно — системный промпт бота/режима). Для всех
  шести случаев весь смысл запроса и так целиком в `query`/`prompt` (инструкция самодостаточна),
  поэтому смена системного промпта не должна заметно менять результат; не передаём
  `assistant_prompt` явно, чтобы не плодить новые константы под одноразовые вызовы.
- Декорации ответа: `get_chat_response` мог дописывать к ответу footer с usage
  (`show_usage`) и списком использованных плагинов (`show_plugins_used`,
  `bot/openai_helper.py:1072-1086`, только в legacy-ветке `chat_run_variant_b_enabled=False`,
  которая по умолчанию выключена — см. T15). `ask()` этого никогда не делал. В проде с
  дефолтным `show_usage=False`/`show_plugins_used=False` разницы нет; при нестандартной
  конфигурации четыре плагина перестанут получать этот footer внутри своего текста — это
  правильно (footer должен быть виден пользователю в основном ответе, а не в служебном тексте
  внутри PDF-анализа или упражнения).
- Безвредный побочный эффект: `ask()` вызывает `self.get_max_tokens(model_to_use, 60, user_id)`
  (`bot/openai_helper.py:831`), а внутри — `self._chat_state_key(chat_id)` с `chat_id=user_id`
  (`bot/openai_helper.py:4218`); во вложенном вызове это резолвится в **внешний** `state_key`
  (то же наследование ContextVar, что и в основном баге) и просто читает размер **внешней**
  истории, чтобы посчитать бюджет `max_tokens` для служебного ответа. Чтения без записи —
  не баг, только neat-to-know: бюджет одноразового ответа слегка зависит от длины текущего
  разговора пользователя.
- `ask()` не оборачивает вызов в таймаут (`asyncio.wait_for`), в отличие от
  `ModelUtilities.one_shot`. Это уже так для `chief.py`/`movie_info.py`/`show_me_diagrams.py`
  сегодня — не регрессия этой задачи, но стоит знать: если апстрим модель зависнет, вложенный
  `ask()`-вызов зависнет вместе с ней (без дедлока на локе, но с зависшим tool-call). Отдельная
  тема, не в этой задаче.

## Дизайн guard

Guard ставится в начало `get_chat_response` и `get_chat_response_stream`, сразу после того, как
`chat_id`/`user_id`/`session_id` разрешены через `request_context` (чтобы в сообщении об ошибке
был финальный `chat_id`), и до `state_key = conversation_state_key or self._chat_state_key(chat_id)`
— то есть до того, как метод успеет что-либо тронуть (лок, ContextVar, историю).

```python
if _CHAT_STATE_KEY.get() is not None:
    raise RuntimeError(
        "get_chat_response() called re-entrantly from inside an active chat "
        "turn (chat_id=%r). A nested call either deadlocks on the per-chat "
        "lock (different chat_id) or corrupts the outer conversation history "
        "(same chat_id, lock bypassed). Plugins making a one-off model call "
        "from execute() must use helper.ask() instead."
        % (chat_id,)
    )
```

Почему безусловно (без исключения для bypass/совпадающего `chat_id`): именно «совпадающий
`chat_id`» — самый опасный случай (тихая порча истории вместо явного зависания), его нельзя
оставлять как «разрешённый». `_chat_lock_bypass_enabled`/`_without_chat_lock`
(`bot/openai_helper.py:3221-3231`) после этой правки становятся мёртвым кодом **для
`get_chat_response`/`get_chat_response_stream`** конкретно (это единственные два места, где
`_chat_lock_bypass_enabled` вообще вызывается — проверено `grep`), но их не убираю: это выходит
за рамки T09 (описано в §5.1 обзора как общий пункт про `_without_chat_lock`, не связанный с
этой задачей напрямую), и `_call_function_bounded` продолжает выставлять bypass для любых
других (гипотетических) top-level методов, которые тоже могут на него смотреть.

Почему guard безопасен для всех **сегодняшних легитимных** вызовов `get_chat_response`/
`get_chat_response_stream` — проверено `grep -n` по всему дереву, других мест нет:

| Место | Контекст вызова | `_CHAT_STATE_KEY` установлен? |
|---|---|---|
| `bot/telegram_bot.py:2805, 4233, 4530, 4910, 4988` | Top-level обработчики апдейтов Telegram (обычное сообщение, callback, inline-запрос) | Нет — новый апдейт, свежий контекст |
| `bot/plugins/agent_cron.py:217` (`_run_job`) | Запускается из `_checker_loop` (`:59`, периодическая фоновая задача, `application.create_task` при регистрации плагина) либо из `/agent_cron ... run` (`handle_cron_command:172`, Telegram-команда) | Нет — оба пути стартуют вне активного хода |
| `bot/plugins/agent_tools.py:1113` (`_run_background_job`) | Запускается из `handle_background_command` (`:944`, Telegram-команда `/background`) | Нет — команда, не tool-call |
| `bot/plugins/agent_tools.py:1501` (`_run_goal_run`) | Запускается из `_goal_runs_tick` (`:1438`, периодическая фоновая задача, читает `queued`-строки из БД) | Нет — `manage_goal_runs` (tool, `:1218-1605`) только пишет строку в БД со статусом `queued` и возвращается; сам запуск происходит позже отдельной задачей |
| `bot/plugins/ask_your_pdf.py:364`, `language_learning.py:187`, `conversation_analytics.py:233,253,271,289`, `show_me_diagrams.py:161,186` | Изнутри `execute()`, вызванного через `asyncio.gather` из активного хода | **Да** — это и есть баг, guard должен их отклонять |

Отдельно проверено: ни один вызов `self.get_chat_response`/`self.get_chat_response_stream`
внутри `bot/openai_helper.py` или `bot/chat_run.py` не существует (`ChatRun.run_non_stream`
ходит в приватный `__common_get_chat_response`, ретраи — в свои приватные хелперы) — то есть
сам `OpenAIHelper` никогда не вызывает себя рекурсивно через публичные методы, guard ничего
внутри не сломает.

`interpret_image`/`interpret_images`/`interpret_image_stream` (`bot/openai_helper.py:2625,
2661, 2803`) используют тот же `_CHAT_STATE_KEY`, но guard на них не ставится — они не входят в
область T09 (аудит и задача называют только `get_chat_response`/`get_chat_response_stream`), и
ни один плагин их не вызывает (`grep -rnE "helper\.interpret_image"` по `bot/plugins/*.py` — 0
совпадений). Если в будущем появится плагин, вызывающий `interpret_image*` изнутри `execute()`,
он словит тот же класс бага — стоит завести аналогичный guard тогда, а не сейчас (не защищаем
от сценария с нулём вхождений — «Simplicity First»).

## Правки по `file:line`

### 1. `bot/openai_helper.py:872-874` (внутри `get_chat_response`)

Сейчас:
```python
            if session_id is None:
                session_id = request_context.session_id

        state_key = conversation_state_key or self._chat_state_key(chat_id)
```

После:
```python
            if session_id is None:
                session_id = request_context.session_id

        if _CHAT_STATE_KEY.get() is not None:
            raise RuntimeError(
                "get_chat_response() called re-entrantly from inside an active chat "
                "turn (chat_id=%r). A nested call either deadlocks on the per-chat "
                "lock (different chat_id) or corrupts the outer conversation history "
                "(same chat_id, lock bypassed). Plugins making a one-off model call "
                "from execute() must use helper.ask() instead."
                % (chat_id,)
            )

        state_key = conversation_state_key or self._chat_state_key(chat_id)
```

### 2. `bot/openai_helper.py:1117-1119` (внутри `get_chat_response_stream`)

Тот же guard, тем же текстом, вставляется перед `state_key = conversation_state_key or
self._chat_state_key(chat_id)` этого метода (строка `:1119` до правки). Async-генератор:
`raise` до первого `yield` — исключение всплывёт на первой итерации (`async for ... in
get_chat_response_stream(...)`), это стандартное поведение и не требует отдельной обработки на
стороне вызывающих.

### 3. `bot/plugins/ask_your_pdf.py:364-366`

Сейчас:
```python
                response, _ = await helper.get_chat_response(
                    chat_id=hash(file_path),
                    query=analysis_prompt,
                )
```
После (добавить чтение `user_id` перед вызовом — в текущем коде `analyze_pdf` его не читает
вовсе):
```python
                user_id = kwargs.get("user_id")
                response, _ = await helper.ask(analysis_prompt, user_id)
```
`user_id` можно объявить раньше, сразу после `file_path = kwargs.get("file_path")` (:338) —
как сделано в остальных трёх плагинах ниже; конкретное место — на усмотрение разработчика,
семантически не важно.

### 4. `bot/plugins/language_learning.py:187-190`

Сейчас:
```python
            response, _ = await helper.get_chat_response(
                chat_id=hash(f"{language}_{level}_{exercise_type}"),
                query=prompt
            )
```
После:
```python
            user_id = self._get_owner_user_id(kwargs)
            response, _ = await helper.ask(prompt, user_id)
```
`_get_owner_user_id` — уже существующий в этом файле хелпер (`:99-104`), используется в ветке
`track_progress` (`:203`); переиспользуем тот же локальный паттерн вместо `kwargs.get('user_id')`
напрямую.

### 5. `bot/plugins/conversation_analytics.py:233-236, 253-256, 271-274, 289-292`

Один и тот же паттерн правки в четырёх местах (различаются только уже существующей веткой
`if/elif` и текстом `prompt`, который не меняется). Пример для `topics` (:233-236):

Сейчас:
```python
                response, _ = await helper.get_chat_response(
                    chat_id=hash(f"{chat_id}_topics"),
                    query=prompt
                )
```
После:
```python
                response, _ = await helper.ask(prompt, kwargs.get('user_id'))
```
Аналогично для `_learning` (:253-256), `_format` (:271-274), `_style` (:289-292) — только
заменяется вызов, `prompt` и последующий `recommendations.append(...)` не трогаются.
`user_id` здесь читается напрямую из `kwargs` (в этом файле нет своего `_get_owner_user_id`,
локальный стиль — прямые `kwargs['...']`/`kwargs.get('...')`, см. `chat_id = str(kwargs['chat_id'])`
на `:201`).

### 6. `bot/plugins/show_me_diagrams.py:149-190`

Сейчас (фрагмент, :149-165):
```python
        user_id = kwargs.get('user_id')
        # chat_id инжектится PluginManager-ом в kwargs перед execute (см.
        # openai_tool_handler.py). Раньше тут читался несуществующий ключ
        # helper.conversations['last_chat_id'], который всегда возвращал 0
        # и слал follow-up-запрос «в никуда».
        chat_id = kwargs.get('chat_id') or 0
        if not diagram_type:
            type_prompt = (
                "Выберите тип диаграммы из следующих:\n"
                f"{', '.join(self.diagram_types.keys())}\n"
                "Какой тип диаграммы вы хотите создать?"
            )
            diagram_type_response, _ = await helper.get_chat_response(
                chat_id=chat_id,
                query=type_prompt
            )
```
После (убираем `chat_id` целиком — после правки у него не остаётся других читателей в файле,
см. проверку в разделе «Анализ»; убираем и комментарий про `last_chat_id`, он объяснял именно
`chat_id`, который исчезает):
```python
        user_id = kwargs.get('user_id')
        if not diagram_type:
            type_prompt = (
                "Выберите тип диаграммы из следующих:\n"
                f"{', '.join(self.diagram_types.keys())}\n"
                "Какой тип диаграммы вы хотите создать?"
            )
            diagram_type_response, _ = await helper.ask(type_prompt, user_id)
```
И далее (:186-189), сейчас:
```python
            description_response, _ = await helper.get_chat_response(
                chat_id=chat_id,
                query=description_prompt
            )
```
После:
```python
            description_response, _ = await helper.ask(description_prompt, user_id)
```
Убедиться, что `chat_id` больше нигде в `execute()`/остальном файле не читается (проверено —
не читается); если разработчик найдёт ещё одно использование, которое я пропустил, `chat_id`
нужно оставить и убрать только из этих двух вызовов.

## Тесты

### A. `tests/test_ask_your_pdf.py` — переключить мок с `get_chat_response` на `ask`

Затронутые строки: `_helper()` (:51-54), `_helper_prompt()` (:57-61) и 9 обращений к
`helper.get_chat_response.*` (:87, 116, 125, 145, 165, 228, 250, 296).

```python
def _helper(answer="PDF analysis result"):
    return SimpleNamespace(
        ask=AsyncMock(return_value=(answer, 123)),
    )


def _helper_prompt(helper):
    call = helper.ask.await_args
    if call.args:
        return call.args[0]
    return call.kwargs["prompt"]
```
Остальные 9 строк — механическая замена `helper.get_chat_response` → `helper.ask` (метод
вызова — `assert_awaited_once()`/`assert_not_called()`/`.side_effect = [...]`/`.await_count` —
не меняется, только имя атрибута). `user_id` в тестовые вызовы `plugin.execute(...)` добавлять
не обязательно: `ask` в этих тестах полностью замокан (`SimpleNamespace` + `AsyncMock`), реальная
ветка `_in_active_turn`/`get_max_tokens` внутри `OpenAIHelper.ask()` не выполняется, а
`analyze_pdf` спокойно передаёт `user_id=None`, если `kwargs` его не содержит — это не упадёт.

Проверить итог: `python3 -m pytest tests/test_ask_your_pdf.py -q -p no:cacheprovider` — было
9 passed, должно остаться 9 passed (тесты те же самые по количеству, меняется только то, что
они мокают).

### B. `tests/test_openai_helper_tool_calls.py:2122-2144` — перезаписать закрепляющий баг тест

Текущий тест `test_tool_execution_allows_nested_chat_response_with_same_chat_lock` **проверяет
именно то поведение, которое пункт 3.2 называет багом** (совпадающий `chat_id`, лок обойдён,
ответ «тихо» пишется в чужую историю) и называет его «allows» — то есть фиксирует баг как
контракт. После правки guard это поведение больше не будет «allow», а будет `RuntimeError`.
Переписать тест на новое имя/поведение, переиспользуя ту же обвязку:

```python
@pytest.mark.asyncio
async def test_tool_execution_rejects_nested_chat_response_even_with_same_chat_lock():
    class NestedPluginManager(DummyPluginManager):
        async def call_function(self, name, helper, arguments, request_context=None):
            return await helper.get_chat_response(chat_id=1, query="nested", user_id=1)

    helper = _make_helper(
        NestedPluginManager({}),
        client=DummyClient([FakeResponse(content="nested done")]),
    )
    lock = await helper._chat_lock(1)

    async with lock:
        with pytest.raises(RuntimeError, match="get_chat_response"):
            await asyncio.wait_for(
                _call_function_bounded(
                    helper,
                    "prompt_perfect.optimize_prompt",
                    json.dumps({"chat_id": 1}),
                    None,
                    asyncio.Semaphore(1),
                ),
                timeout=2,
            )

    # Guard срабатывает до захвата лока — лок не должен остаться удержанным
    # где-то во вложенном вызове после того, как исключение всплыло.
    assert not lock.locked()
```
`asyncio.wait_for(..., timeout=2)` здесь — не защита от реального зависания (guard кидает
исключение до какого-либо `await` на локе), а регрессионный барьер: если кто-то в будущем
передвинет guard ниже захвата лока, тест упадёт по таймауту, а не повиснет сам.

### C. Прямые unit-тесты на guard (без прогона через `_call_function_bounded`)

Добавить в тот же файл, рядом с B:

```python
@pytest.mark.asyncio
async def test_get_chat_response_rejects_call_from_active_turn():
    helper = _make_helper(DummyPluginManager({}), client=DummyClient([FakeResponse(content="x")]))
    token = openai_helper_module._CHAT_STATE_KEY.set(1)
    try:
        with pytest.raises(RuntimeError, match="get_chat_response"):
            await asyncio.wait_for(
                helper.get_chat_response(chat_id=999, query="q", user_id=1),
                timeout=2,
            )
    finally:
        openai_helper_module._CHAT_STATE_KEY.reset(token)


@pytest.mark.asyncio
async def test_get_chat_response_stream_rejects_call_from_active_turn():
    helper = _make_helper(DummyPluginManager({}), client=DummyClient([_fake_stream([])]))
    token = openai_helper_module._CHAT_STATE_KEY.set(1)
    try:
        with pytest.raises(RuntimeError, match="get_chat_response"):
            async for _ in helper.get_chat_response_stream(chat_id=999, query="q", user_id=1):
                pass
    finally:
        openai_helper_module._CHAT_STATE_KEY.reset(token)
```
`openai_helper_module` уже импортирован в файле (`import bot.openai_helper as
openai_helper_module`, см. верх файла) — новых импортов не требуется.

### D. Плагины без сегодняшнего покрытия этого вызова

`tests/` не содержит тестов на `language_learning.py`, `show_me_diagrams.py` вообще, и
`conversation_analytics.py` тестируется только по хукам (`tests/test_conversation_analytics_
hooks.py`, не про `get_chat_response`) — то есть для трёх из четырёх плагинов сегодня нет теста,
который упадёт/не упадёт от этой правки. Явно не в счастливом смысле «ничего не сломаем», а в
смысле «некому подтвердить, что не сломали»: рекомендую разработчику при реализации добавить
хотя бы по одному smoke-тесту на изменённую ветку (`daily_practice`,
`get_personalized_recommendations` с любым `recommendation_type`, `generate_diagram` без
`type`/`description`) с замоканным `helper.ask`, по образцу `_helper()`/`_helper_prompt()` из
`tests/test_ask_your_pdf.py`. Это расширение покрытия, а не обязательное условие T09 (в
исходной задаче не запрошено), поэтому оставляю на усмотрение разработчика/ревью.

## Команды проверки

Выполнять из корня репозитория `/srv/git_projects/chatgpt-telegram-bot`, системный `python3`.

```bash
# Базовый прогон до правок (зафиксировано в этом плане — 2026-09-04):
python3 -m pytest tests/test_ask_your_pdf.py tests/test_openai_helper_tool_calls.py \
  -q -p no:cacheprovider
# -> 138 passed

# Репро (до правки должен показывать DEADLOCK/порчу истории, после — что nested-вызовы
# из execute() у четырёх плагинов больше не происходят вовсе, т.к. они на helper.ask()):
python3 /tmp/hunt/repro_nested_lock.py

# После правок:
python3 -m pytest tests/test_ask_your_pdf.py tests/test_openai_helper_tool_calls.py \
  -q -p no:cacheprovider -x
python3 -m pytest tests/test_conversation_analytics_hooks.py -q -p no:cacheprovider -x
python3 -m pytest tests/test_agent_tools_plugin.py tests/test_agent_cron_plugin.py \
  -q -p no:cacheprovider -x   # agent_tools/agent_cron тоже вызывают get_chat_response —
                               # убедиться, что guard их не задел (ожидание: без изменений)
python3 -m pytest -q -p no:cacheprovider   # полный прогон, если есть время/CI это позволяет
```

## Риски

- **`test_tool_execution_allows_nested_chat_response_with_same_chat_lock` — единственный явный
  тест на текущее (ошибочное) поведение.** Если его просто удалить, а не переписать (тест B),
  регресс останется незамеченным при повторном появлении такого паттерна в новом плагине.
  Переписан, а не удалён — см. тест B.
- **Поведенческий дрейф в четырёх плагинах** (модель/температура/системный промпт/footer,
  подробности — в «Анализ»). Риск низкий: все шесть вызовов — служебные однострочные ответы,
  используемые только как текст; ни один тест сегодня не проверяет точный текст ответа модели
  (все моки возвращают фиксированную строку). Основной наблюдаемый эффект — чуть другая
  температура (0.6 вместо настроенной) для этих конкретных генераций; решение принято явно, не
  скрыто.
- **`show_me_diagrams.py`: удаление `chat_id`/комментария.** Затрагивает только эти два места
  использования (проверено `grep`); если ревью найдёт третье использование, которое я
  пропустил, — вернуть переменную и убрать только из вызовов `ask()`.
- **guard и `_chat_lock_bypass_enabled`/`_without_chat_lock` становятся частично мёртвым кодом**
  для двух методов (см. «Дизайн guard»). Не удаляю в этой задаче — вне её рамок, `_without_chat_
  lock` используется и вне `get_chat_response` (`_call_function_bounded` выставляет его для
  любого тула с `chat_id` в аргументах, не только для тех, что зовут `get_chat_response`).
  Если owner захочет — можно завести отдельную задачу на аудит оставшихся потребителей bypass
  после того, как этот guard проживёт в проде.
- **`interpret_image*` не защищены тем же guard** (см. «Дизайн guard» — сознательно, 0 текущих
  вызовов из плагинов). Если появится такой плагин в будущем, он воспроизведёт тот же класс
  бага; стоит держать это в голове при код-ревью новых plugin.execute(), вызывающих
  `helper.interpret_image*`.
- **`agent_tools.py`/`agent_cron.py` тоже вызывают `get_chat_response`.** Прослежены оба пути
  планирования (`_run_background_job`, `_run_goal_run`) — оба стартуют вне активного хода
  (Telegram-команда или периодическая фоновая задача, не `execute()` тула). Риск того, что
  guard их сломает, оценён как отсутствующий, но именно поэтому в «Команды проверки» отдельно
  включены `test_agent_tools_plugin.py`/`test_agent_cron_plugin.py` — если там что-то красное,
  это сигнал, что анализ пропустил путь, где `create_task` создаётся уже внутри активного хода.

## Критерии готовности

- `bot/openai_helper.py`: `get_chat_response` и `get_chat_response_stream` кидают `RuntimeError`
  при `_CHAT_STATE_KEY.get() is not None`, до захвата лока и до записи в историю.
- Четыре плагина (`ask_your_pdf.py`, `language_learning.py`, `conversation_analytics.py` (4
  места), `show_me_diagrams.py` (2 места)) не содержат вызовов `helper.get_chat_response`;
  все шесть заменены на `helper.ask(...)`.
- `show_me_diagrams.py`: локальная переменная `chat_id` и поясняющий её комментарий убраны
  (не остаётся неиспользуемого кода, оставшегося от правки).
- `tests/test_ask_your_pdf.py`: 9 тестов проходят, все моки — на `helper.ask`, ни одного
  упоминания `get_chat_response`.
- `tests/test_openai_helper_tool_calls.py`: тест на «nested + тот же chat_id» теперь проверяет
  `RuntimeError` (не «allows»); добавлены прямые unit-тесты на guard для обоих методов
  (`get_chat_response`, `get_chat_response_stream`).
- `python3 -m pytest tests/test_ask_your_pdf.py tests/test_openai_helper_tool_calls.py -q
  -p no:cacheprovider` — все тесты зелёные (ожидаемое количество — не меньше 138 + количество
  новых тестов из пунктов B/C, то есть 138 + 2 = 140 минимум).
- `python3 /tmp/hunt/repro_nested_lock.py`, адаптированный на вызов через `execute()` четырёх
  плагинов (или просто прямой вызов `get_chat_response` из-под установленного
  `_CHAT_STATE_KEY`), показывает мгновенный `RuntimeError`, а не зависание и не «completed» с
  чужим `state_key`.
- `git status --short` показывает изменения только в перечисленных файлах (`bot/openai_helper.py`,
  четыре файла плагинов, `tests/test_ask_your_pdf.py`, `tests/test_openai_helper_tool_calls.py`)
  и ничего лишнего.

## Постскриптум после ревью

Ревью (Sonnet, персона reviewer): ошибок и предупреждений нет. Проверен полный обход всех
вызовов `get_chat_response`/`get_chat_response_stream` в `bot/` на предмет ложных срабатываний
guard через наследование ContextVar в `asyncio.create_task`: фоновые задачи `agent_tools`
(`_run_background_job`, `_run_goal_run`, subagent-loop), `agent_cron._run_job`,
`_ensure_session_name_with_llm` — все идут через `helper.chat_completion` либо создаются вне
активного хода, guard не задет. Тест
`test_tool_execution_rejects_nested_chat_response_even_with_same_chat_lock` явно выставляет
`_CHAT_STATE_KEY` — осознанное отклонение от псевдокода плана, иначе баг не воспроизводится.
