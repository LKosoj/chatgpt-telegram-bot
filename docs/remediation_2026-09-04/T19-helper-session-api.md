# T19. Публичный API сессии в `OpenAIHelper`

Источник задачи: `docs/audit_remediation_plan_2026-09-04.md` (раздел «T19», Волна 4), находка
`docs/architecture_code_review_2026-09-04.md` §5.3 «П4» (`Публичный API состояния сессии в
helper`). Для контекста также прочитаны «П2» (мёртвый провайдер-слой `model_constants`) и «П6»
(`UsageTracker` на loop) — обе не пересекаются по файлам с T19 и в этот план не входят; П4 —
единственный пункт §5.3, который дословно описывает задачу T19 (`load_session`,
`replace_system_message`, `evict`, мутации `openai.conversations` из
`bot/telegram_bot.py:2436-2481, 6149-6174`). Роль документа — план для разработчика; код не
менялся, только прочитан.

## Термины

- **Приватный метод/атрибут** — имя, начинающееся с `_` (например, `_clear_chat_state`).
  По соглашению Python это сигнал «трогать только изнутри своего класса»; ничего физически не
  запрещает вызвать его снаружи, но так теряется гарантия, что автор класса не сломает внешний
  код при следующей правке.
- **`getattr(obj, "имя", None)`** — «безопасно прочитать атрибут `obj.имя`, а если его нет —
  вернуть `None`» вместо падения с ошибкой `AttributeError`. В коде бота это используется как
  «проверка на всякий случай»: а вдруг `self.openai` — тестовая заглушка без этого метода.
- **AST (Abstract Syntax Tree, абстрактное синтаксическое дерево)** — представление исходного
  кода Python в виде дерева объектов вместо текста; позволяет находить «обращение к атрибуту
  `x.y`» надёжнее, чем поиском текста, потому что видит структуру, а не буквы.
  `ast.unparse(node)` превращает кусок дерева обратно в читаемый код-текст.
  `ast.dump(node)` печатает дерево как текст для отладки.
  Тест ниже — линтер (см. следующий пункт) на AST, а не на регулярных выражениях.
  Модуль `ast` — часть стандартной библиотеки Python, `import ast` без установки пакетов.
- **Линтер-тест** — обычный `pytest`-тест, который не проверяет поведение программы, а
  сканирует исходный код на запрещённый паттерн (здесь: «обращение к приватности helper'а») и
  падает, если паттерн найден без явного разрешения. В проекте уже есть образец:
  `tests/test_no_hardcoded_plugin_refs.py`.
- **Контекстный менеджер** (`with ...:`) — конструкция, которая гарантированно выполняет код
  «после» блока, даже если внутри было исключение (как `try/finally`, но короче).
  `_with_chat_state`/`chat_state_scope` ниже — именно такой менеджер.
- **ContextVar** — переменная, которая «едет» вместе с текущей асинхронной задачей
  (`asyncio.Task`) и не видна другим параллельным задачам. Используется, чтобы во время
  обработки одного запроса временно подменить «эффективный ключ чата» без глобальной переменной.

## Цель

1. Убрать из `bot/telegram_bot.py` прямые чтения/записи словарей `OpenAIHelper.conversations` /
   `loaded_conversation_sessions` и обращения к приватным методам `OpenAIHelper` через
   `getattr(self.openai, "_...", None)`.
2. Дать взамен маленький публичный API на `OpenAIHelper`: `history_snapshot`, `load_session`,
   `replace_system_message`, `evict` (плюс один служебный метод `chat_state_scope`, см.
   «Дизайн API» — он не входит в исходный список из четырёх имён, но нужен, чтобы закрыть ещё
   одно найденное обращение). Публичные методы — тонкие обёртки/переименования существующих
   приватных методов, семантика не меняется (кроме одного намеренного исправления
   несогласованности — см. «Поведенческий эффект» в «Дизайне API»).
3. Проверить `bot/plugins/*.py` на такие же обращения к `helper` и включить найденное в общую
   таблицу и в исправления.
4. Добавить линтер-тест `tests/test_no_private_helper_access.py` по образцу
   `tests/test_no_hardcoded_plugin_refs.py`, который ловит регресс (кто-то в будущем снова
   полезет в `self.openai._...`/`helper._...`).

Не входит в T19 (явно вне рамок, зафиксировано для прозрачности):
- `bot/chat_run.py:62-218` — обращения к `helper._OpenAIHelper__common_get_chat_response`,
  `__handle_function_call`, `__add_to_history`, `_gate_fired`, `_chat_request_*` через
  name-mangling. Это находка §5.1 / «П1», закрывается в T15 вместе с удалением legacy-пути
  (`chat_run_variant_b_enabled`). Пересекается с T19 только тем, что оба трогают
  `bot/openai_helper.py`, но не по конкретным строкам — файловый конфликт маловероятен, но
  порядок мержа стоит согласовать (в `audit_remediation_plan_2026-09-04.md` Волна 4 явно
  указан порядок «П1 → П2 → П3 → П4»; T19 = П4).
- `bot/plugin_manager.py:5398` `getattr(self.openai.plugin_manager, "plugin_instances", {})` —
  это `plugin_manager` (публичный атрибут), а не приватность `OpenAIHelper`; не относится к
  задаче.

## Таблица обращений

Найдено скриптом (`python3 -c` с `ast`/`re` по `bot/telegram_bot.py` и `bot/plugins/*.py`, не
`rg`/`grep` — в проекте отмечено, что они искажают вывод на этом окружении). Номера строк —
актуальные на 2026-09-04, проверены `sed -n`.

### `bot/telegram_bot.py`

| Строки | Обращение | Что делает | Целевой публичный метод |
|---|---|---|---|
| `2436` | `self.openai.conversations.get(conversation_key)` | Читает тёплый кеш истории чата, чтобы решить — грузить ли из БД (холодный кеш) или взять как есть. Внутри `_handle_prompt_selection_locked` (обработчик выбора режима `/prompt`). | `history_snapshot(chat_id)` |
| `2445` | `getattr(self.openai, '_messages_without_image_payloads', None)` | При холодной загрузке из БД сжимает vision-сообщения (списки `content` с картинками) в текстовый плейсхолдер `[image_file_id: ...]` перед тем, как класть в кеш — так же, как это делают ещё 6 мест внутри `openai_helper.py` при каждой загрузке в `self.conversations`. | Поглощается внутрь `load_session(...)` (вызывающий код больше не должен об этом знать) |
| `2450-2451` | `self.openai.conversations[conversation_key] = ...` / `self.openai.loaded_conversation_sessions[conversation_key] = session_id` | Кладёт холодно загруженную историю в кеш и помечает, какая сессия сейчас загружена. | `load_session(chat_id, session_id, messages)` |
| `2466-2467` | То же самое, после вставки/замены системного сообщения | Финальная запись кеша + указателя сессии (безусловно перезаписывается ещё раз, даже если шаг 2450 не выполнялся — тёплый кеш). | `load_session(...)` (вызывается один раз для обеих веток, тёплой и холодной — см. «Правки») |
| `2470-2490` | `getattr(self.openai, "_save_conversation_context", None)` + `if callable(...): await save_context(...) else: await self._db_call("save_conversation_context", ..., self.openai)` | Сохраняет новый системный промпт режима в БД; при отсутствии приватного метода (тестовые заглушки) — обходной путь через публичный `Database.save_conversation_context_async`. | Поглощается внутрь `replace_system_message(...)` |
| `3701` (в `_cleanup_parallel_session_state`, `def` на `3694`) | `getattr(self.openai, "_clear_chat_state", None)` + `if callable(...): clear_chat_state(session_key)` | Выселяет из памяти всё состояние параллельной («busy») сессии после того, как она обработана в фоне: `conversations`, `last_updated`, `loaded_conversation_sessions`, `_chat_request_extra_tokens`, `_chat_request_models`, `_chat_request_usage_split`, `last_image_file_ids`, персональный лок чата (см. докстринг `_clear_chat_state`, `bot/openai_helper.py:3262-3276`). | `evict(chat_id)` |
| `4067` (в `process_message`, вложенная `_run_locked`) | `getattr(self.openai, '_with_chat_state', None)` + `if ...and callable(...): with chat_state_scope(conversation_state_key): ...` | Временно подменяет «эффективный ключ чата» (через `ContextVar`) на ключ параллельной сессии на время обработки одного запроса — так, чтобы код внутри `openai_helper.py`, читающий `self._chat_state_key(chat_id)`, увидел не «основной» chat_id, а pinned session key. Используется только при параллельной обработке «busy»-сообщений в новой сессии. | `chat_state_scope(state_key)` (см. ниже — 5-й метод, не входил в исходный список задачи) |
| `6139-6156` (`action == "switch"`, обработчик `session:` callback) | `self.openai.conversations[conversation_key] = current_context['messages']` / `self.openai.loaded_conversation_sessions[conversation_key] = session_id` | После переключения активной сессии кладёт в кеш её историю из БД. **Не** пропускает через `_messages_without_image_payloads` (в отличие от строки 2445) — несогласованность, см. «Поведенческий эффект». | `load_session(chat_id, session_id, messages)` |
| `6158-6174` (`action == "delete"`) | То же самое | После удаления текущей сессии кладёт в кеш историю новой активной сессии. Та же несогласованность (нет стрипа картинок). | `load_session(chat_id, session_id, messages)` |

Дополнительно проверено и **не** найдено в `bot/telegram_bot.py`: обращений к
`self.openai.last_updated`, `self.openai._OpenAIHelper__...` (name-mangled), алиасов вида
`x = self.openai; x._foo(...)` (единственные локальные присваивания из `self.openai` — это
именованные аргументы вызовов вида `openai_helper=self.openai`/`helper=self.openai`,
передаваемые в `Database`/DB-хелперы, которые читают только публичный `.config['model']`).
Гипотеза задачи про `conversations_vision` не подтвердилась — такого атрибута в коде нет
(видимо, устаревшее имя из более раннего состояния кода).

### `bot/plugins/*.py`

Скрипт проверил все `.py` в `bot/plugins/` на: `helper._имя`, `self.openai._имя`/`self.helper._имя`
(на случай, если плагин сохранил `helper` в атрибут — см. ниже, один плагин так делает),
`getattr(<то же>, "_имя", ...)`, а также точные имена `conversations`, `loaded_conversation_sessions`,
`last_updated` без подчёркивания.

| Файл:строка | Обращение | Что делает | Рекомендация |
|---|---|---|---|
| `bot/plugins/agent_tools.py:3720-3721` | `def _tool_result_content(self, helper, content): formatter = getattr(helper, "_tool_result_content", None); if callable(formatter): return formatter(content); ...` (иначе — свой инлайн `json.dumps`) | Форматирует результат тула в строку для истории диалога; сначала пытается переиспользовать `OpenAIHelper._tool_result_content`, которая сама — тонкая обёртка (`bot/openai_helper.py:1772-1773`, `return tool_result_content(content)`) вокруг свободной функции `bot/tool_result.py:33`. | Импортировать `tool_result_content` из `bot.tool_result` напрямую и вызывать её вместо обращения к приватности `helper`. Убирает единственное найденное обращение плагина к приватному методу helper'а; поведение не меняется для JSON-сериализуемых значений (расхождение только в обработке несериализуемых объектов — см. «Риски»). |
| `bot/plugins/haiper_image_to_video.py:557` | `self.openai = helper` (плагин хранит helper в атрибуте) | Далее (`:229, :232, :436`) читает только `self.openai.db` и `self.openai.api_key` — оба публичные. | Не требует правок — приватных обращений через этот алиас нет. Упомянуто, потому что без проверки алиасов линтер мог бы пропустить будущую регрессию именно в этом файле. |

Обращений `helper.conversations`/`helper.loaded_conversation_sessions`/`helper.last_updated` (в
любой форме — `helper.`, `self.openai.`, `self.helper.`) в `bot/plugins/*.py` не найдено.

## Дизайн API

Все новые методы — на классе `OpenAIHelper` (`bot/openai_helper.py`), размещаются рядом с
существующими тонкими публичными геттерами `get_last_chat_model`/`get_last_chat_usage_split`
(`bot/openai_helper.py:3202-3216`, прямо перед `_with_chat_state`, `:3220`) — то есть образуют
один блок «публичный API состояния чата» перед приватными методами, которые они оборачивают.
Приватные методы (`_messages_without_image_payloads`, `_save_conversation_context`,
`_clear_chat_state`, `_with_chat_state`, `_chat_state_key`) **не переименовываются и не
удаляются** — каждый уже используется внутри `openai_helper.py` в 4-13 местах (проверено
`grep`), переименование потребовало бы правки всех этих мест ради задачи, которая просит только
«бот перестаёт трогать приватность». Единственное исключение — `_clear_chat_state`: у него ровно
один вызывающий во всём проекте (сам `bot/telegram_bot.py:3701`), поэтому для него тонкая
обёртка и прямое переименование эквивалентны по риску; оставляю обёртку для единообразия со
остальными.

```python
# bot/openai_helper.py — новый блок public session-state API,
# перед def _chat_state_key (:3199) / после get_last_chat_usage_split (:3216)

def history_snapshot(self, chat_id) -> list[dict] | None:
    """Публичное read-only чтение тёплого кеша истории chat_id.
    None — кеш холодный, вызывающий код должен прочитать историю из БД
    и передать её в load_session().
    """
    return self.conversations.get(self._chat_state_key(chat_id))

def load_session(self, chat_id, session_id: str | None, messages: list[dict]) -> list[dict]:
    """(Пере)заполняет кеш истории chat_id сообщениями messages (обычно — только что
    прочитанными из БД) и запоминает session_id как загруженную для chat_id сессию.
    Картиночные payload'ы стрипаются в текстовый плейсхолдер перед кешированием —
    как и во всех остальных местах этого класса, где self.conversations заполняется
    из БД (_messages_without_image_payloads, 6 вызовов в этом файле).
    Возвращает закешированный (уже стрипнутый) список, чтобы вызывающий код мог
    сразу передать его дальше (например, в replace_system_message через кеш).
    """
    state_key = self._chat_state_key(chat_id)
    cached = self._messages_without_image_payloads(list(messages))
    self.conversations[state_key] = cached
    self.loaded_conversation_sessions[state_key] = session_id
    return cached

async def replace_system_message(
    self,
    chat_id,
    content: str,
    *,
    mode_key: str | None = None,
    parse_mode: str = 'HTML',
    temperature: float | None = None,
    max_tokens_percent: int = 80,
) -> str | None:
    """Вставляет/заменяет первое системное сообщение в кеше chat_id и сохраняет
    результат в БД. Требует, чтобы load_session() уже был вызван в этом же ходе
    (session_id берётся из loaded_conversation_sessions, не передаётся явно —
    так же, как исходный код bot/telegram_bot.py безусловно перезаписывал его
    после вставки системного сообщения).
    temperature=None -> берётся self.config['temperature'] (тот же дефолт,
    что и mode_data.get('temperature', self.openai.config['temperature'])
    в исходном коде бота).
    Возвращает session_id, под которым сохранено (или None, если сохранить
    не удалось) — как _save_conversation_context.
    """
    state_key = self._chat_state_key(chat_id)
    current_context = self.conversations.get(state_key) or []
    session_id = self.loaded_conversation_sessions.get(state_key)
    system_message: dict = {"role": "system", "content": content}
    if mode_key is not None:
        system_message["mode_key"] = mode_key
    if current_context and current_context[0].get('role') == 'system':
        current_context[0] = system_message
    else:
        current_context.insert(0, system_message)
    self.conversations[state_key] = current_context
    if temperature is None:
        temperature = self.config['temperature']
    return await self._save_conversation_context(
        chat_id, {'messages': current_context}, parse_mode, temperature,
        max_tokens_percent, session_id,
    )

def evict(self, chat_id) -> None:
    """Публичная обёртка _clear_chat_state: убирает из памяти всё состояние
    per-chat для chat_id (conversations, last_updated, loaded_conversation_sessions,
    per-turn usage/model bookkeeping, last_image_file_ids, персональный лок).
    """
    self._clear_chat_state(chat_id)

def chat_state_scope(self, state_key):
    """Публичный алиас _with_chat_state: контекстный менеджер, временно
    подменяющий эффективный ключ чата (см. _chat_state_key) на state_key —
    для параллельной обработки «отложенных» сообщений в новой сессии.
    Не входит в исходный список из 4 методов задачи T19, добавлен, чтобы
    закрыть обращение bot/telegram_bot.py:4067 (getattr(self.openai,
    '_with_chat_state', None)) — иначе бот продолжил бы трогать приватность
    в одном месте. Если ревьюер считает это отдельной задачей — можно
    оставить :4067 как задокументированное исключение линтера вместо
    добавления метода (см. «Риски»).
    """
    return self._with_chat_state(state_key)
```

`history_snapshot`/`load_session`/`evict` — синхронные (как и обёртываемые приватные методы);
`replace_system_message` — `async def`, потому что вызывает `await self._save_conversation_context(...)`.

### Поведенческий эффект (осознанное изменение, не баг в этом плане)

`load_session` **всегда** стрипает картинки через `_messages_without_image_payloads`, как это
уже делают 6 других мест в `openai_helper.py`, где `self.conversations` заполняется из БД
(`bot/openai_helper.py:436, 813, 1210, 1488, 2490, 3579`). Строки `6154-6155` и `6172-6173` в
`bot/telegram_bot.py` (переключение/удаление сессии) сегодня **не** стрипают — кладут в кеш
сырые сообщения из БД, включая (потенциально) полные vision-payload'ы. После перевода на
`load_session` это исправится «бесплатно», как побочный эффект унификации через один метод.
Это выравнивание с уже установленным инвариантом класса, а не расширение задачи — но
поведенчески это правка (меньше памяти/токенов в модели после переключения сессии с картинками
в истории), и я не могу проверить её тестом (нет теста на переключение сессии с
vision-историей — см. «Риски»). Если владелец хочет чистого 1:1 переноса без побочных
исправлений — альтернатива: `load_session(chat_id, session_id, messages, *, strip_images=True)`
и на строках `6154-6155/6172-6173` явно передавать `strip_images=False`. Я рекомендую **не**
делать так (некому будет объяснить в будущем, почему эти два места — особые), но озвучиваю как
вариант, раз меняется наблюдаемое поведение.

## Правки по file:line

### `bot/telegram_bot.py:2434-2493` (`_handle_prompt_selection_locked`)

Было (см. «Таблица обращений» выше для точных строк 2436-2490) — читает тёплый кеш напрямую,
при холодном кеше вручную стрипает картинки и пишет в оба словаря, затем вручную
вставляет/заменяет системное сообщение и сохраняет через `getattr`-нащупывание приватного
метода с фолбэком на `_db_call`.

Станет:

```python
mode_data = chat_modes[mode]
current_context = self.openai.history_snapshot(conversation_key)
if current_context is None:
    saved_ctx, _, _, _, _ = await self._db_call(
        "get_conversation_context", conversation_key, session_id,
    )
    messages = saved_ctx['messages'] if saved_ctx and 'messages' in saved_ctx else []
else:
    messages = current_context
self.openai.load_session(conversation_key, session_id, messages)

await self.openai.replace_system_message(
    conversation_key,
    mode_data.get('prompt_start', ''),
    mode_key=mode,
    parse_mode=mode_data.get('parse_mode', 'HTML'),
    temperature=mode_data.get('temperature'),
    max_tokens_percent=mode_data.get('max_tokens_percent', 80),
)

await self.reset(update, context)
```

Убирает 3 `getattr`-проверки и ветвление `if callable(...): ... else: ...` (2 раза), сокращает
~55 строк исходного блока до ~15. `load_session` вызывается один раз для обеих веток (тёплой и
холодной) — сохраняет исходное поведение «`loaded_conversation_sessions` безусловно
перезаписывается в конце» (строки 2466-2467 в старом коде).

### `bot/telegram_bot.py:3694-3702` (`_cleanup_parallel_session_state`)

Было: `getattr(self.openai, "_clear_chat_state", None)` + `if callable(...): clear_chat_state(session_key)`.

Станет:
```python
self.openai.evict(session_key)
```

### `bot/telegram_bot.py:4059-4070` (`process_message`, вложенная `_run_locked`)

Было: `getattr(self.openai, '_with_chat_state', None)` + `if conversation_state_key is not None and callable(chat_state_scope): with chat_state_scope(conversation_state_key): ...`.

Станет:
```python
if conversation_state_key is not None:
    with self.openai.chat_state_scope(conversation_state_key):
        return await _run_locked()
return await _run_locked()
```

### `bot/telegram_bot.py:6139-6156` (`action == "switch"`)

Было: см. таблицу (строки `6154-6155`).

Станет:
```python
current_context, parse_mode, temperature, max_tokens_percent, _ = await self._db_call(
    "get_conversation_context", conversation_key, session_id,
)
if current_context and 'messages' in current_context:
    self.openai.load_session(conversation_key, session_id, current_context['messages'])
await self.reset(update, context)
```

### `bot/telegram_bot.py:6158-6174` (`action == "delete"`)

Симметрично — заменить строки `6172-6173` на
`self.openai.load_session(conversation_key, session_id, current_context['messages'])`.

### `bot/plugins/agent_tools.py:3720-3728` (`_tool_result_content`)

Добавить в начало файла (рядом с прочими `from bot....` импортами):
```python
from bot.tool_result import tool_result_content
```
Заменить тело метода:
```python
def _tool_result_content(self, helper, content: Any) -> str:
    return tool_result_content(content)
```
`helper` в сигнатуре становится неиспользуемым — либо убрать параметр и обновить единственный
вызывающий (`:3479`, `self._tool_result_content(helper, tool_response or "")` →
`self._tool_result_content(tool_response or "")`), либо оставить параметр ради минимального
диффа (плагин не единственный, у кого сигнатура private-метода не совпадает с использованием —
на усмотрение разработчика). Рекомендую убрать параметр — иначе линтер ниже придётся объяснять
ревьюеру, зачем `helper` всё ещё передаётся, если внутри не используется.

## Тесты

### Новый линтер `tests/test_no_private_helper_access.py`

По образцу `tests/test_no_hardcoded_plugin_refs.py`, но ищет не строковые литералы, а обращения
к атрибутам через `ast`, потому что паттерн — это структура кода (`x.y` или
`getattr(x, "y", ...)`), а не текст внутри строк.

```python
"""Linter test: catch new private-attribute access on the OpenAIHelper instance from
bot/telegram_bot.py and bot/plugins/*.py.

T19 replaced direct dict mutation (self.openai.conversations[...]) and
getattr(self.openai, "_private_method", None) probing with a small public API
(history_snapshot/load_session/replace_system_message/evict/chat_state_scope). This
test scans for regressions: any future `self.openai._foo` / `helper._foo` / the
explicit shared-state dict names, direct or via getattr(). If a new access is
legitimate, add it to ALLOWED with a reason -- do not silently raise the threshold.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TELEGRAM_BOT_FILE = REPO_ROOT / "bot" / "telegram_bot.py"
PLUGINS_DIR = REPO_ROOT / "bot" / "plugins"

# Names that are not underscore-prefixed but are still considered private
# per-chat state, mirroring OpenAIHelper._clear_chat_state's docstring.
DENYLISTED_STATE_ATTRS = {"conversations", "loaded_conversation_sessions", "last_updated"}

# (file_relpath, attr_name) -> (expected_count, reason)
ALLOWED: Dict[Tuple[str, str], Tuple[int, str]] = {}


def _is_private_attr(name: str) -> bool:
    return name.startswith("_") or name in DENYLISTED_STATE_ATTRS


def _matches_target(node: ast.AST, targets: set[str]) -> bool:
    try:
        return ast.unparse(node) in targets
    except Exception:
        return False


def _find_violations(tree: ast.AST, targets: set[str]) -> list[tuple[int, str]]:
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _matches_target(node.value, targets):
            if _is_private_attr(node.attr):
                violations.append((node.lineno, node.attr))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and _matches_target(node.args[0], targets)
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            attr = node.args[1].value
            if _is_private_attr(attr):
                violations.append((node.lineno, attr))
    return violations


def _check_file(path: Path, targets: set[str]) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _find_violations(tree, targets)


def _assert_against_allowlist(rel: str, violations: list[tuple[int, str]]) -> list[str]:
    from collections import Counter
    counts = Counter(attr for _lineno, attr in violations)
    failures = []
    for attr, actual in counts.items():
        allowed_count, _reason = ALLOWED.get((rel, attr), (0, ""))
        if actual > allowed_count:
            lines = [lineno for lineno, a in violations if a == attr]
            failures.append(
                f"  {rel}: found {actual} private access(es) to '{attr}' at lines {lines}, "
                f"allowed {allowed_count}"
            )
    return failures


def test_telegram_bot_does_not_touch_openai_privates() -> None:
    violations = _check_file(TELEGRAM_BOT_FILE, {"self.openai"})
    failures = _assert_against_allowlist("bot/telegram_bot.py", violations)
    assert not failures, (
        "bot/telegram_bot.py reaches into OpenAIHelper privates:\n" + "\n".join(failures)
        + "\n\nUse the public session API (history_snapshot/load_session/"
        "replace_system_message/evict/chat_state_scope) instead, or add a "
        "documented ALLOWED entry."
    )


@pytest.mark.parametrize("plugin_path", sorted(PLUGINS_DIR.glob("*.py")), ids=lambda p: p.name)
def test_plugins_do_not_touch_helper_privates(plugin_path: Path) -> None:
    violations = _check_file(plugin_path, {"helper", "self.openai", "self.helper"})
    rel = plugin_path.relative_to(REPO_ROOT).as_posix()
    failures = _assert_against_allowlist(rel, violations)
    assert not failures, (
        f"{rel} reaches into the OpenAIHelper instance's privates:\n" + "\n".join(failures)
        + "\n\nUse a public accessor, or add a documented ALLOWED entry."
    )
```

Ограничение (озвучить в PR/ревью, не скрывать): это эвристика по буквальному тексту выражения
перед `.атрибут` (`self.openai`, `helper`, `self.helper`) — обходится присваиванием в
промежуточную переменную (`x = self.openai; x._foo()`), как и текстовый линтер
`test_no_hardcoded_plugin_refs.py` обходится f-строками/конкатенацией. Сегодня в дереве такого
алиасинга для приватных обращений нет (проверено — единственный найденный алиас,
`haiper_image_to_video.py:557`, использует только публичные атрибуты). Тот же класс
ограничений уже задокументирован в проекте для `bot/command_policy.py` («эвристика над текстом
команды, а не барьер песочницы») — здесь то же самое: линтер ловит очевидное, не гарантирует
полноту.

### Обновить существующие тесты

- **`tests/test_group_session_flow.py`** — `_make_openai()` (`:159-167`) возвращает голый
  `SimpleNamespace(conversations={}, loaded_conversation_sessions={}, ...)` без
  `load_session`/`replace_system_message`/`evict`/`history_snapshot`. Тест
  `test_group_prompt_selection_uses_group_conversation_key_for_session_db` (`:273-284`) вызывает
  `bot.handle_prompt_selection(...)` и затем проверяет `assert -100123 in bot.openai.conversations`
  — это единственный тест, который реально доходит до переписываемого кода (остальные
  `handle_session_callback`-тесты в этом файле проверяют `switch`/`delete`/`new`, тоже дойдут до
  `load_session`). После правки бот вызывает `self.openai.load_session(...)` напрямую (без
  `getattr`-проверки), значит `SimpleNamespace` без этого метода упадёт с `AttributeError`.
  Нужно добавить в `_make_openai()` минимальные реализации, которые мутируют те же словари, что
  тест проверяет напрямую:
  ```python
  def load_session(chat_id, session_id, messages):
      conversations[chat_id] = list(messages)
      loaded_conversation_sessions[chat_id] = session_id
      return conversations[chat_id]
  ns.load_session = load_session
  ```
  (и так же для любых других новых методов, до которых реально доходит выполнение в тестах
  этого файла — проверить по каждому `test_group_session_*`/`test_private_session_switch_*`).
- **`tests/test_callback_authorization.py`** — `_make_openai()` (`:189-206`) тоже
  `SimpleNamespace`, но оба теста, которые вызывают `handle_prompt_selection`/
  `handle_session_callback` (`test_unauthorized_prompt_selection_does_not_mutate_mode_or_context`,
  `test_unauthorized_session_callback_does_not_mutate_db`), намеренно проверяют **отказ**
  неавторизованному пользователю — код возвращается раньше, чем доходит до
  `load_session`/`replace_system_message`. Риск низкий, но стоит перепроверить после правки
  (см. «Команды проверки») на случай, если в `_ensure_allowed`/ранних `return` что-то не так
  прочитано мной.
- **`tests/test_agent_tools_plugin.py`** — 78 тестов (базовый прогон подтверждён, см. ниже), ни
  один не упоминает `_tool_result_content` по имени; используется косвенно через
  `deliver_to_user`/публикацию артефактов. Прогнать целиком после правки строки `3720-3728` и
  добавления импорта.
- Прямых тестов на `_cleanup_parallel_session_state` (эвикшен `evict`) и на
  `process_message`'s `chat_state_scope`-путь (кроме `tests/test_openai_helper_tool_calls.py`,
  который тестирует `_with_chat_state` напрямую на **настоящем** `OpenAIHelper`, а не через
  бота — этот тест не меняется, приватный метод остаётся) в репозитории нет. Это дыра в покрытии
  до T19, не создаваемая им — но раз рефактор именно в этом месте, стоит добавить хотя бы один
  smoke-тест на `evict`/`chat_state_scope` через реальный вызов `_cleanup_parallel_session_state`
  с реальным (не `SimpleNamespace`) `OpenAIHelper`, чтобы было что запускать в регрессии.
  Явно не обязательно для T19 (не запрошено в исходной задаче) — оставляю на усмотрение
  разработчика/ревью, как это сделано в T09 для аналогичного случая.

## Команды проверки

Выполнять из корня репозитория `/srv/git_projects/chatgpt-telegram-bot`, `~/.venvs/ctb/bin/python`.

```bash
# Базовый прогон до правок (зафиксировано в этом плане, 2026-09-04):
~/.venvs/ctb/bin/python -m pytest tests/test_group_session_flow.py \
  tests/test_callback_authorization.py tests/test_no_hardcoded_plugin_refs.py \
  -q -p no:cacheprovider
# -> 59 passed

~/.venvs/ctb/bin/python -m pytest tests/test_agent_tools_plugin.py -q -p no:cacheprovider
# -> 78 passed

# После правок в bot/openai_helper.py, bot/telegram_bot.py, bot/plugins/agent_tools.py:
~/.venvs/ctb/bin/python -m pytest tests/test_group_session_flow.py \
  tests/test_callback_authorization.py tests/test_no_hardcoded_plugin_refs.py \
  tests/test_no_private_helper_access.py tests/test_agent_tools_plugin.py \
  -q -p no:cacheprovider -x

# helper._with_chat_state по-прежнему используется напрямую в тестах (не через бота) —
# убедиться, что приватный метод не переименован и не удалён:
~/.venvs/ctb/bin/python -m pytest tests/test_openai_helper_tool_calls.py -q -p no:cacheprovider -x

# Полный прогон:
~/.venvs/ctb/bin/python -m pytest -q -p no:cacheprovider

# Ручная проверка, что новый линтер ловит регресс (временно, не коммитить):
python3 -c "
import re
p = 'bot/telegram_bot.py'
s = open(p).read()
s = s.replace('self.openai.evict(session_key)', 'self.openai._clear_chat_state(session_key)')
open(p, 'w').write(s)
"
~/.venvs/ctb/bin/python -m pytest tests/test_no_private_helper_access.py -q -p no:cacheprovider
# ожидание: FAILED — затем откатить строку обратно
git checkout -- bot/telegram_bot.py
```

## Риски

- **`chat_state_scope` — пятый метод, не входивший в исходный список задачи (`load_session`,
  `replace_system_message`, `evict`, `history_snapshot`).** Добавлен, чтобы закрыть
  `bot/telegram_bot.py:4067` — иначе бот продолжил бы трогать `self.openai._with_chat_state`.
  Альтернатива: не трогать `:4067` в этой задаче, задокументировать как известное исключение в
  `ALLOWED` линтера с явной причиной («относится к параллельным сессиям, отдельная задача»).
  Я рекомендую добавить метод (он тривиален и без него линтер придётся сразу заводить с
  исключением) — но раз задача явно называла 4 метода, а не 5, это решение стоит подтвердить на
  код-ревью, а не считать согласованным по умолчанию.
- **Поведенческий эффект `load_session` всегда стрипает картинки** (см. «Дизайн API» —
  затрагивает `switch`/`delete` ветки `6154-6155`/`6172-6173`, которые раньше не стрипали).
  Нет теста на переключение/удаление сессии с vision-историей — риск тихой регрессии не в
  сторону поломки, а в сторону «теперь после переключения сессии модель видит меньше деталей о
  прежней картинке, чем раньше» до первого реального использования. Решение явное (задокументировано
  в «Дизайне API»), не скрытое; альтернатива с `strip_images=False` описана там же.
- **Тестовые заглушки `SimpleNamespace` в `tests/test_group_session_flow.py` должны получить
  реализации новых методов**, иначе `AttributeError` вместо прохождения теста (см. «Тесты»).
  Список выше не гарантированно полон — при реализации стоит прогнать файл целиком и добавить
  метод в фейк для каждого `AttributeError`, а не только для перечисленных здесь.
- **`bot/plugins/agent_tools.py`: `_tool_result_content` меняет источник данных** с
  `getattr(helper, "_tool_result_content", None)` (либо инлайн-фолбэк) на прямой импорт
  `tool_result_content` из `bot.tool_result`. Для JSON-сериализуемых значений и строк —
  идентичное поведение. Для несериализуемых объектов есть небольшая разница:
  `tool_result_content` использует `json.dumps(value, default=str, ...)` (несериализуемые куски
  превращаются в строки внутри JSON), старый инлайн-фолбэк на `TypeError` возвращал весь объект
  через `str(content)` целиком (не JSON). На практике оба пути — редкий edge case; 78 тестов
  `test_agent_tools_plugin.py` — единственная защита, ни один явно не бьёт по этому случаю
  (проверено `grep` на имя метода — ни одного прямого упоминания).
- **Синхронизация с T15/П1 (Волна 4, порядок «П1 → П2 → П3 → П4»).** T19 не трогает
  `bot/chat_run.py`, но обе задачи редактируют `bot/openai_helper.py`. Построчных пересечений не
  найдено (T19 — новый блок перед `_with_chat_state`/`:3199-3220`; T15 — тело
  `:934-1091` и окрестности `chat_run_variant_b_enabled`, `:342, 688-690, 922`), но порядок
  мержа стоит соблюсти, как задано в исходном плане волны.
- **Эвристика линтера обходится алиасингом** (`x = self.openai; x._foo()`). Сегодня в дереве
  такого нет (проверено), но это не защита от будущего обхода — тот же класс ограничений уже
  принят для `test_no_hardcoded_plugin_refs.py` (строковые литералы) и `command_policy.py`
  (эвристика над текстом команды).

## Критерии готовности

- `bot/openai_helper.py`: добавлены `history_snapshot`, `load_session`,
  `replace_system_message`, `evict`, `chat_state_scope` рядом с
  `get_last_chat_model`/`get_last_chat_usage_split`; ни один существующий приватный метод
  (`_messages_without_image_payloads`, `_save_conversation_context`, `_clear_chat_state`,
  `_with_chat_state`, `_chat_state_key`) не переименован и не удалён, их существующие
  внутренние вызывающие (`bot/openai_helper.py`, `bot/chat_run.py` тесты) не тронуты.
- `bot/telegram_bot.py`: ни одного обращения `self.openai.conversations`,
  `self.openai.loaded_conversation_sessions`, `getattr(self.openai, "_...", ...)` не осталось
  (кроме, возможно, задокументированного исключения на `chat_state_scope`, если ревью решит не
  добавлять этот метод — см. «Риски»).
- `bot/plugins/agent_tools.py`: `_tool_result_content` вызывает `bot.tool_result.tool_result_content`
  напрямую, без `getattr(helper, "_tool_result_content", None)`.
- `tests/test_no_private_helper_access.py` создан, проходит на текущем дереве после правок, и
  ловит намеренно внесённый регресс (см. «Команды проверки», ручная проверка).
- `tests/test_group_session_flow.py`, `tests/test_callback_authorization.py` обновлены (фейки
  `_make_openai()` получили нужные методы) и проходят без `AttributeError`.
- `~/.venvs/ctb/bin/python -m pytest -q -p no:cacheprovider` — полный прогон зелёный.
- `git status --short` показывает изменения только в перечисленных файлах
  (`bot/openai_helper.py`, `bot/telegram_bot.py`, `bot/plugins/agent_tools.py`,
  `tests/test_no_private_helper_access.py`, `tests/test_group_session_flow.py`,
  `tests/test_callback_authorization.py`) и ничего лишнего.

---

## Постскриптум после ревью

Ревьюер (Sonnet, persona reviewer, read-only) прошёл построчно по всем пяти новым методам
(`bot/openai_helper.py:2967-3050`), по пяти местам их использования в `bot/telegram_bot.py`
и по правке `bot/plugins/agent_tools.py`, сверяя старое и новое поведение. Отдельно он
проверил сам сторож: в песочнице подменил `self.openai.evict(...)` на
`self.openai._clear_chat_state(...)` и убедился, что линтер ловит регресс.

**Вердикт: `## Ошибки` — нет.** Фактический прогон ревьюера: 1616 passed, ruff чист.

### Предупреждения — что сделано

1. **`history_snapshot` отдавал живую ссылку на внутренний список, а не копию.**
   Имя («snapshot») и докстринг («read-only access») обещали безопасность, которой в коде
   не было: любой вызывающий мог добавить/удалить элемент и молча испортить кеш
   `OpenAIHelper` в обход `load_session`. Практического бага сегодня не было (единственный
   вызывающий сразу передаёт результат в `load_session`), но контракт обманывал будущих
   вызывающих.
   **Исправлено:** метод возвращает поверхностную копию (`list(cached)`) — так же, как это
   делает соседний приватный `_snapshot_chat_state` (`bot/openai_helper.py:2360`).
   Различие «холодный кеш → `None`» и «пустая история → `[]`» сохранено. В докстринге явно
   написано, что словари сообщений внутри списка всё ещё общие с кешем, поэтому менять
   сообщение «на месте» тоже нельзя — только через `load_session`.

2. **Ни один из пяти методов не имел прямого теста на настоящем `OpenAIHelper`.**
   Существующие тесты (`tests/test_group_session_flow.py`,
   `tests/test_per_conversation_serialization.py`) проверяли только то, что бот **зовёт**
   эти имена у самодельной заглушки, а не то, что реализация работает правильно; `evict`
   вообще не встречался в тестах по имени, а заглушка `chat_state_scope` не вызывалась,
   потому что в том тесте `process_message` подменён на `AsyncMock`.
   **Исправлено:** добавлен `tests/test_openai_helper_session_api.py` — 19 тестов на
   настоящих реализациях (helper собирается через `object.__new__`, как в
   `tests/test_openai_helper_summarize_trim.py`):
   - `history_snapshot`: холодный кеш → `None`; тёплый → содержимое; изменение результата
     не портит кеш; пустая история отличается от холодного кеша;
   - `load_session`: картинки стрипаются (`[image]`-плейсхолдер), id сессии запоминается,
     возвращается именно закешированный список, список вызывающего не мутируется,
     перезапись предыдущей истории, `session_id=None` допустим;
   - `replace_system_message`: вставка при отсутствии system-сообщения, замена
     существующего, `mode_key` не добавляется, если не передан, температура берётся из
     конфига при `temperature=None`, id сессии читается из карты загруженных сессий;
   - `evict`: очищает все восемь пер-чатовых словарей, идемпотентен на неизвестном ключе,
     не задевает другие чаты;
   - `chat_state_scope`: подменяет эффективный ключ, восстанавливает его после выхода,
     восстанавливает и при исключении, корректно вкладывается.

3. **`load_session` теперь стрипает картинки и в ветках `switch`/`delete`** (раньше эти две
   ветки картинки не стрипали) — осознанная правка плана, но без теста.
   **Закрыто пунктом 2**: механизм стрипа теперь проверяется напрямую
   (`test_load_session_strips_image_payloads_and_records_session`), включая то, что
   исходный список вызывающего остаётся с оригинальным vision-контентом.

4. **`_tool_result_content` в `agent_tools`: изменился только краевой случай** с
   несериализуемыми объектами (раньше инлайн-фолбэк на `TypeError` возвращал `str(content)`,
   теперь всё идёт через `tool_result_content` с `default=str`). Ревьюер сам проверил, что в
   боевом пути расхождения нет — `OpenAIHelper._tool_result_content` и раньше был тонкой
   обёрткой над той же функцией. **Действий не требуется.**

5. **Сторож видел не все файлы.** Он сканировал только `bot/telegram_bot.py` и
   `bot/plugins/*.py`, а обращения к приватностям хелпера есть ещё в
   `bot/openai_tool_handler.py` (16 штук) и `bot/skill_script_routing.py` (2). Это не регресс
   T19 — строки не тронуты диффом, — но пробел остался бы незамеченным при будущих правках.
   **Решено явно, оба варианта из рекомендации ревьюера применены по месту:**
   - `bot/skill_script_routing.py` **добавлен в сканирование** (`test_skill_script_routing_
     does_not_add_helper_privates`) с двумя задокументированными записями в `ALLOWED`:
     `conversations` (:20) и `_mode_from_system_message` (:31) — оба read-only-проба через
     `getattr()`, нужны, чтобы роутинг работал и против минимальных тестовых заглушек.
     Любое **новое** обращение теперь падает.
   - `bot/openai_tool_handler.py` **сознательно оставлен вне сторожа**, причина записана в
     докстринге теста: это не потребитель хелпера, а часть его же машинерии запроса,
     вынесенная в отдельный модуль; сторож там заморозил бы внутренности `OpenAIHelper`, а
     не границу. Два обращения, которые действительно мутируют общий кеш извне класса
     (`helper.conversations.setdefault` на `:255`, `helper.conversations.get` на `:1658`),
     названы в докстринге как известный оставшийся пробел вне периметра T19.

### Замечания без действия (подтверждены ревьюером, правок не требуют)

- `chat_state_scope`/`_with_chat_state` — `@contextmanager` с `try/finally`, исключения не
  глотаются, контекст освобождается гарантированно.
- `evict` идемпотентен: `_clear_chat_state` везде использует `.pop(key, None)`.
- Имя параметра `evict(self, chat_id)` унаследовано от `_clear_chat_state`, хотя по факту
  туда чаще передают готовый state-key (кортеж) — переименование не входило в план.
- Часть докстринга `load_session` описывает сегодня неиспользуемую возможность (передать
  возвращённый список дальше в `replace_system_message`).

### Проверка

`pytest -q tests bot/tests` — **1636 passed**, 3 warnings (deprecation из библиотеки
`telegram`, к задаче не относятся). `ruff check bot tests bot/tests` — All checks passed.

**Файлы, изменённые постскриптумом:** `bot/openai_helper.py` (копия в `history_snapshot`),
`tests/test_no_private_helper_access.py` (расширение сторожа), `tests/test_openai_helper_
session_api.py` (новый), `AGENTS.md` (раздел «Helper Session API» — уточнена формулировка
про копию).
