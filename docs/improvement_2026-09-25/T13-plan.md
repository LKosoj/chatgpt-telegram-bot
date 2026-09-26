# T13 — mypy до нуля (волна 8): план

Базовая ревизия: рабочее дерево на момент планирования (после T02–T12, T12 частично на
ревью — номера строк в файлах T12 могут слегка сместиться, группировка ниже поэтому идёт
по файлу/коду ошибки/паттерну, не только по номеру строки).

Команда (та же, что использует `scripts/mypy_baseline.py`, см. `run_mypy()` в этом файле):

```
python3 -m mypy --config-file pyproject.toml --python-executable ~/.venvs/ctb/bin/python
```

(конфиг `[tool.mypy]` в `pyproject.toml`: `files = ["bot"]`, `ignore_missing_imports = true`,
`exclude = ["bot/tests/", "bot/skills/"]`, `warn_unused_ignores = true` — отдельный
`--ignore-missing-imports` в CLI не нужен, уже в конфиге).

**Текущий факт (не мастер-плановская цифра 585 — часть уже снята T02–T12):**
**567 ошибок** в 51 файле. Разбивка по группам ниже в сумме даёт 567
(228 + 174 + 165) — группы дизъюнктны и покрывают все файлы с ошибками.

Правило поведения (общее правило + явно для T13): **фиксы не должны менять рантайм**.
Никаких бланкетных `# type: ignore`. `# type: ignore[код]` — только для обоснованных
проблем внешних библиотек с указанием кода. Порядок внутри каждой группы (из мастер-плана):
1) механика (`x: T = None` → `x: T | None`, `-> [Dict]` → `-> List[Dict]`, `any` → `Any`),
2) `None`-проверки, 3) остальное. Базовая линия (`mypy_baseline.json`) обновляется в конце
волны координатором, не здесь.

**Внешние библиотеки:** `ignore_missing_imports = true` уже глобальный — в текущих 567
ошибках **нет** `import-untyped`/`import-not-found`. Отдельных per-module оверрайдов или
`# type: ignore[import-untyped]` в рамках T13 не требуется, если не появятся новые при
правке (не встречалось при сканировании).

---

## Сквозные паттерны и канонические фиксы

Встречаются в нескольких группах — исполнитель применяет один и тот же фикс везде в своих
файлах, без изобретения новых форм.

1. **Implicit Optional** (`assignment` на `Incompatible default for parameter "X" (default
   has type "None", parameter has type "T")`): `x: T = None` → `x: T | None = None`.
   Примеры: `bot/database.py:1059,1170,1224` (`session_id: str = None`),
   `bot/utils.py:553` (`chat_action: ChatAction = ...` со str-default — уточнить, см. T13b).
   **Не менять** значение по умолчанию, только аннотацию.
2. **`-> [Dict]`** (`valid-type`, "Bracketed expression is not valid as a type"): заменить
   на `-> List[Dict]` (или `list[dict]`, если файл уже использует нижний регистр generics —
   смотреть по остальным аннотациям того же файла, не смешивать стиль). Единый источник —
   `bot/plugins/plugin.py:118` (`Plugin.get_spec`, абстрактный метод) — **14 файлов-плагинов
   в T13c** наследуют этот паттерн буквальным копированием сигнатуры в подклассах (каждый
   даёт ровно 1 ошибку `valid-type`). `bot/plugins/haiper_image_to_video.py` (T13b),
   `skills.py`/`agent_tools.py`/`hindsight_memory.py` (T13c) уже используют корректный
   `-> List[Dict]` — не трогать повторно.
3. **`any` (builtin) вместо `typing.Any`**: `bot/utils.py:1067,1076` (`response: any` в
   `is_direct_result`/`direct_result_inline_fallback_text`) — источник ошибок
   `bot/utils.py:1081,1222` ("any? has no attribute get"). Фикс: `response: any` →
   `response: Any` + `from typing import Any`, если ещё не импортирован (проверить импорт
   в начале `utils.py` перед добавлением, не дублировать).
4. **`update.message` / `update.callback_query` Optional** (`union-attr`, доминирующий
   паттерн в `telegram_bot.py`, `utils.py`, `haiper_image_to_video.py`) — единый дизайн
   помощников, см. раздел "Design: require_message/require_query" ниже. Реализуется в
   T13a (`bot/telegram_bot.py`), используется по аналогии в T13b (`utils.py`,
   `haiper_image_to_video.py`) и T13c без обязательного общего файла (см. риски).
5. **`Plugin.on_before_chat_request` базовая сигнатура** (T13c, `bot/plugins/plugin.py:64-66`):
   сейчас `-> List[Dict]`; 3 подкласса (`skills.py:219`, `agent_tools.py:356`,
   `hindsight_memory.py:2445`, все T13c) возвращают `list[dict[str, Any]] | None`
   (мутатор, `None` = "без изменений" — см. `AGENTS.md`, раздел про hooks). Фикс: сузить
   базовую сигнатуру до `-> list | None` (или точнее `-> List[Dict] | None`, чтобы не терять
   типизацию элементов) и **добавить `return None`/скорректировать `return messages`** тела
   по необходимости — базовая реализация (identity) может остаться `return messages`, это
   совместимо с `list | None`. Такое расширение — контравариантно безопасно (более широкий
   базовый возврат не ломает подклассы, которые уже уже `list | None`), новых ошибок в
   других плагинах не создаёт.

---

## Design: `require_message(update)` / `require_query(update)`

**Где:** `bot/telegram_bot.py`, как приватные функции модуля (не методы класса — вызываются
до/без доступа к `self` в части хендлеров, и это чистые преобразования типа, не состояние
бота). Разместить рядом с другими module-level helpers файла (проверить на месте, есть ли
уже такая секция; если нет — сразу над классом `ChatGPTTelegramBot`).

**Сигнатуры и поведение** (соответствует уже существующему в файле стилю раннего возврата,
см. `bot/telegram_bot.py:2696,3076,5614` — `if not update.message: return`):

```python
def require_message(update: Update) -> Message | None:
    """update.message может быть None (напр. edited_message-апдейт). Вызывающий код сам
    решает, что делать при None — как и сейчас, обычно ранний return."""
    return update.message

def require_query(update: Update) -> CallbackQuery | None:
    """update.callback_query может быть None для не-callback апдейтов."""
    return update.callback_query
```

Поведение при `None` **не меняется** — сами хелперы ничего не логируют и не бросают,
только сужают тип и делают точку доступа единообразной. Вызывающий код:

```python
message = require_message(update)
if message is None:
    return
...
await message.reply_text(...)
```

Это даёт mypy narrowing (после `if message is None: return`, дальше `message` имеет тип
`Message`, не `Message | None`) — убирает `union-attr` на `Message | None` (59 шт. в
`telegram_bot.py`) и на `CallbackQuery | None` (27 шт.) без изменения поведения, при условии,
что на каждом call site **уже есть** эквивалентная логика раннего возврата — это нужно
перепроверить по месту (`git diff`-подобно, читать каждый сайт), не предполагать вслепую.

**Отдельный случай — `MaybeInaccessibleMessage`** (`attr-defined`, 16+ мест: `.reply_text`,
`.delete` и т.д. отсутствуют на `MaybeInaccessibleMessage`, только на `Message`). Это
`query.message` (тип `MaybeInaccessibleMessage | None` в PTB — сообщение из callback может
быть "недоступным", если исходное сообщение удалено/устарело). `require_query` не решает
этот случай (это уже не Optional-проблема, а объединение типов). Нужен третий, более узкий
helper **только там, где реально вызывается `.reply_text`/`.delete` и т.п. на
`query.message`**:

```python
def require_accessible_message(query: CallbackQuery) -> Message | None:
    """query.message бывает MaybeInaccessibleMessage (устаревшее сообщение без методов
    отправки) или None. Возвращает Message только когда он пригоден для reply_text/delete."""
    msg = query.message
    return msg if isinstance(msg, Message) else None
```

Поведение: `isinstance`-сужение — рантайм не меняется (раньше код либо падал бы на
`AttributeError` при реальном `MaybeInaccessibleMessage`, либо — что вероятнее по счётчику
ошибок — на практике `query.message` почти всегда полноценный `Message` и код работал;
новая проверка добавляет `if msg is None: return`/аналог только как **типобезопасность**,
не меняя обработку реального `None`/`MaybeInaccessibleMessage` случая, если такой ветки в
коде раньше не было — **если в коде такой ветки нет, T13a не добавляет новую ветку
поведения**, а использует `# type: ignore[union-attr]` с обоснованием ИЛИ добавляет
`assert isinstance(msg, Message)` (эквивалент прежнего неявного допущения, тоже без
логического изменения, просто явный) — выбор между `isinstance`-guard-с-return и `assert`
оставлен исполнителю T13a по месту, критерий: не менять поведение при уже-`Message` входе,
и не проглатывать реальный `MaybeInaccessibleMessage` тихо новым `return`, если текущий код
не ожидает такого пути (проверить обработчики `CallbackQueryHandler` — PTB передаёт
`MaybeInaccessibleMessage` только когда `Application` собран с
`Defaults(block=False)`+старым апдейтом вне TTL; в этом проекте, если такого сценария нет,
`assert` безопаснее чем тихий `return`).

Дублирует ли `MaybeInaccessibleMessage`-паттерн `bot/utils.py` (T13b) и
`bot/plugins/haiper_image_to_video.py` (T13b) — да (`utils.py:1302,1307,1317,1320,1329,1332`;
`haiper_image_to_video.py:856,941`). **Не выносить в общий файл** (`bot/telegram_bot.py`
хелперы приватны модулю, T13b не должен импортировать из T13a — это создало бы
кросс-групповую зависимость файла в разгар параллельной работы). T13b пишет свой локальный
`isinstance(msg, Message)`-guard на месте каждого сайта (или локальную copy хелпера в своём
файле, если сайтов много — на усмотрение исполнителя, естественный локальный helper
предпочтительнее copy-paste на 2+ местах, `AGENTS.md`: "абстракции только когда убирают
реальное дублирование").

**Дублирующийся `_edit`** (упомянуто в мастер-плане, T13a): не буквальный `no-redef`
(mypy не жалуется на переопределение имени — это разные вложенные `async def _edit`/`_edit()`
в разных функциях, разные scope) — реальная проблема в том, что у `_edit` на
`bot/telegram_bot.py:4060` (`async def _edit():` — **0 аргументов**) отличается сигнатура от
трёх других локальных `_edit(message_id, text, markdown)` (`:3218, :4159, :4824` — **3
аргумента**), и это расхождение сигнатур вызывает `arg-type` на `:4188` (`Callable[[], Any]`
передан туда, где ожидается `Callable[[Any, str, bool], Awaitable[None]]`, при вызове
`stream_to_telegram(..., edit=...)`). Фикс по мастер-плану — переименовать
0-аргументный `_edit` на `:4060` в отдельное имя (например `_edit_no_args` или
осмысленное по контексту — прочитать функцию целиком перед переименованием, чтобы
подобрать понятное имя) и обновить единственный call site внутри той же функции. Это не
трогает 3-аргументные `_edit` на других строках.

---

## T13a — `bot/telegram_bot.py` (228 ошибок)

**Владение файлами:** только `bot/telegram_bot.py`.

По коду: `union-attr` 165 (`Message | None` 59, `CallbackQuery | None` 27,
`User | Any | None` 26, `Chat | None` 10, `User | None` 6, `MaybeInaccessibleMessage`-related
~7, `dict/str | Any | None` ~10, остальное — прочие Optional-члены Telegram-объектов, разбор
`message.effective_attachment` и т.п.), `arg-type` 24, `attr-defined` 16 (в основном
`MaybeInaccessibleMessage.reply_text`, плюс 2 на `authorized_callback`/
`plugin_authorized_callback` — динамический атрибут `_chatgpt_auth_wrapped`, навешиваемый
декоратором; фикс — типизировать через `Protocol`/`cast`, не убирать проверку), `assignment`
3, `misc` 2, `index` 2, `operator` 1, `return-value` 1, `str-unpack` 1.

**Шаги (в порядке мастер-плана):**
1. Механика: `var-annotated` (13 шт., в основном атрибуты `__init__` типа
   `self.message_buffer = {}` без аннотации, `:168-222` и далее) — добавить явные
   аннотации по фактическому использованию (прочитать, чем наполняется каждый словарь/
   список, прежде чем писать тип — не гадать `dict[Any, Any]` бездумно, если тип виден из
   контекста).
2. `require_message`/`require_query`/`require_accessible_message` (см. дизайн выше) —
   добавить один раз, применить на всех call site с `Message | None` / `CallbackQuery | None`
   / `MaybeInaccessibleMessage` в этом файле. Каждый сайт: перепроверить, что там уже
   фактически есть ранний return/эквивалент — если нет, **не изобретать** новую ветку
   поведения молча, вынести в отчёт как найденный потенциальный баг (не чинить как часть T13,
   если это меняет поведение при реально возможном None).
3. `User | Any | None` (26) — обычно `update.effective_user` или `query.from_user`,
   иногда через `getattr(..., 'from_user', None)` (даёт `Any` в объединении) — там где это
   `getattr` с `Any`-фоллбэком, заменить на прямой атрибут, если объект гарантированно
   типизирован (убирает лишний `Any` из union), иначе оставить и добавить `is None`-guard.
4. Переименовать `_edit` на `:4060` (см. выше), поправить `arg-type` на `:4188`.
5. `authorized_callback`/`plugin_authorized_callback` `_chatgpt_auth_wrapped` (2
   `attr-defined`) — если это декоратор, навешивающий маркер-атрибут на функцию: завести
   `Protocol` с этим атрибутом или использовать `setattr`+`getattr` с явным `# type:
   ignore[attr-defined]` только если типизация через `Protocol` несоразмерно сложна для
   разового маркера — предпочесть `Protocol`, ignore — крайний случай.
6. Оставшиеся `arg-type`/`assignment`/`misc`/`index`/`str-unpack`/`operator`/`return-value`
   (по 1-3 шт. каждый) — точечно, читать каждую строку, не паттерн.

**Тесты.** Не создавать новых поведенческих тестов ради типов (T13 не про баги). После
правок прогнать существующий набор, трогающий `telegram_bot.py`, полностью зелёным:
`~/.venvs/ctb/bin/python -m pytest tests/ -q --no-header -p no:cacheprovider -k
"telegram or callback or streaming or plugin_menu or session"` (широкий грэп, ужать по
факту после первого прогона) плюс полный `tests/` в конце (координатор).

**Готово, когда.** `mypy --config-file pyproject.toml --python-executable
~/.venvs/ctb/bin/python` даёт 0 ошибок для `bot/telegram_bot.py`; ни одного нового
`# type: ignore` без кода ошибки и обоснования; весь существующий pytest зелёный.

---

## T13b — `bot/database.py`, `bot/plugins/db_handle.py`, `bot/session_otel.py`,
`bot/session_logger.py`, `bot/utils.py`, `bot/plugins/haiper_image_to_video.py`,
`bot/validation.py`, `bot/skill_script_routing.py` (174 ошибки)

(Мастер-план пишет `bot/session_*.py` — в дереве это ровно два файла, оба перечислены явно.)

**Владение файлами:** ровно перечисленный список, ничего больше.

### `bot/database.py` (42: `attr-defined` 23, `assignment` 13, `has-type` 3, `return-value` 3)

- `attr-defined` на `self._local`/`self.db_path` (23 шт., все от `hasattr(self._local, ...)`/
  `self._local.connection = ...` начиная с `:329`) — `_local` создаётся динамически в
  `__new__`/`instance._local = threading.local()` (`:260`), не объявлен как атрибут класса.
  Фикс: добавить аннотацию класса `_local: threading.local` (и `db_path: str`, если тоже
  так создаётся — проверить) в теле класса `Database` (не менять логику `__new__`/thread-local
  паттерн, только добавить объявление типа).
- `has-type` на `self._executor` (`:419,420,427`) + `return-value` `:431` (`None` вместо
  `ThreadPoolExecutor`) — вероятно `self._executor = None` где-то и позже
  `self._executor = ThreadPoolExecutor(...)`; фикс — implicit Optional паттерн (#1 выше):
  `_executor: ThreadPoolExecutor | None = None`, дальше код уже (по описанию `AGENTS.md`
  про `Database.shutdown()`/executor) должен сам проверять на `None` перед использованием —
  если не проверяет, это потенциальный баг вне объёма T13 (задокументировать в отчёте, не
  чинить рантайм здесь, только типизацию).
- `assignment` 13, включая implicit-Optional `session_id: str = None` (`:1059,1170,1224`,
  паттерн #1) и `str | None` → `str`-переменные (`:1065,1068,1250` и т.п.) — на каждом сайте
  либо расширить тип переменной на `| None` (если реально может быть None дальше по коду),
  либо (частый случай в этом файле) добавить `if x is None: raise`/`assert x is not None`
  сразу после генерации `x`, если инвариант "здесь уже не None" верен по факту логики
  (`get_active_session_id`/`create_session` могут возвращать `None` при ошибке — проверить
  вызывающий код, что он это уже обрабатывает через `raise ValueError` — да, видно в
  `save_conversation_context` — тогда просто сузить тип через `assert`/локальную проверку,
  не копировать `raise` заново).
- `return-value` `:1332` (`int | None` вместо `int`) и `:2001` (`None` вместо `str`) —
  читать сигнатуру функции; либо расширить возвращаемый тип на `| None`, либо (если
  вызывающий код весь этот путь трактует "не найдено" как ошибку) добавить явную проверку
  перед `return`.

### `bot/plugins/db_handle.py` (5: `attr-defined` 4, `assignment` 1)

- `attr-defined` `"None" has no attribute "execute"/"executemany"` (`:41,50,61,76`) — вероятно
  `self._db: Database | None = None`, задаётся позже через `contextvar`/`ContextVar` (см.
  `assignment` на `:122`, "Token[Any]" в переменную типа None — похоже на
  `contextvars.Token`). Тот же implicit-Optional паттерн: аннотировать поле как
  `Database | None`, добавить `assert`/проверку перед `.execute` там, где вызывающий код
  уже гарантирует, что `db_handle` инициализирован (это `DbHandle`-фасад из `AGENTS.md`
  — "Plugin-owned tables" — использовать существующий шаблон, если он уже есть в файле
  для похожих полей).

### `bot/session_otel.py` (6: `operator` 2, `index` 2, `arg-type` 2)

- `Any | dict[Any, Any] | None` не индексируется/не поддерживает `in` (`:110,111,123,124`)
  — сузить тип перед `in`/`[...]` (`isinstance(x, dict)`-guard или `x = x or {}` если пустой
  dict эквивалентен None по семантике — проверить по коду, не менять поведение при
  реальном `None`).
- `arg-type` `:154,274` (`Any | None` там, где ждут `str`) — добавить `str(...)`/None-guard
  по месту, смотреть на вызываемую функцию (`_add_event_on_parent` и `.get`).

### `bot/session_logger.py` (5: `union-attr` 3, `return-value` 2)

- `Queue[Any] | None` (`:272,280`), `TraceContext | None` (`:342`) — implicit-Optional +
  None-guard паттерн, как везде.
- `return-value` `:357,365` "No return value expected" — функция аннотирована `-> None`, но
  содержит `return <value>`; либо это баг (лишний return value, безопасно убрать `return`
  на `return None`/просто `return`, если значение никуда не используется — проверить, не
  теряется ли реально нужное значение), либо аннотация должна быть шире — читать функцию
  целиком перед выбором.

### `bot/utils.py` (43: `attr-defined` 19, `assignment` 10, `arg-type` 5, `valid-type` 4,
`union-attr` 3, `var-annotated` 1, `misc` 1)

- `valid-type` 4 — `-> [Dict]`-паттерн или похожий (проверить по месту — utils.py не
  плагин, но может иметь похожие сигнатуры; если это другой случай "Bracketed expression",
  применить тот же фикс #2).
- `attr-defined` 19: 2 от `any`-паттерна (#3 выше, `:1081,1222`), остальные ~17 от
  `MaybeInaccessibleMessage` (`:1302,1307,1317,1320,1329,1332,...`) — см. раздел про
  `MaybeInaccessibleMessage` выше: локальный `isinstance(msg, Message)`-guard на каждом
  сайте (не импортировать хелпер из `telegram_bot.py`).
- `assignment` 10: implicit Optional (`chat_action: ChatAction = ...` с default `str` на
  `:553` — если default реально строка типа `"typing"` вместо enum `ChatAction.TYPING`,
  **не менять сам default** (это может быть намеренная PTB-совместимость), а расширить
  тип параметра на `ChatAction | str` — проверить, что PTB API принимает строку как алиас,
  прежде чем выбрать этот путь, иначе типизировать default как `ChatAction` и оставить
  ошибку как находку); `User | None` → `User`-переменные (`:672,674,767,769,820`) — обычная
  None-guard последовательность.
- `arg-type` 5, `union-attr` 3, `var-annotated` 1, `misc` 1 — точечно.

### `bot/plugins/haiper_image_to_video.py` (69: `union-attr` 54, `return-value` 7,
`assignment` 3, `arg-type` 3, `attr-defined` 1, `func-returns-value` 1)

Самый плотный файл после `telegram_bot.py`. `get_spec(self) -> List[Dict]` (`:248`) уже
корректен — паттерн #2 здесь не нужен.

- `union-attr` 54: тот же набор, что в T13a (`Message | None`, `CallbackQuery | None`,
  `User | None`, `Document | None`, `str | None`, 2×`MaybeInaccessibleMessage`) — применять
  **тот же дизайн** (`isinstance`/None-guard на каждом сайте, локально в этом файле, не
  импортировать из `telegram_bot.py` — файлы в разных группах правятся параллельно).
- `return-value`/`func-returns-value` (`:1283,1295,1312,1334,1342,1401,1409`) — функция(и)
  на этом диапазоне (похоже одна область ~1270-1420, тот же метод, что фигурировал в T12
  как `handle_prompt_constructor`/соседи, но T13 не трогает структуру, только типы) —
  "No return value expected"/"Return value expected" вперемешку означает, что часть путей
  функции возвращает значение, часть — нет (`return`/`return None`/`return x` смешаны) при
  фиксированной аннотации `-> int`/`-> None`. Читать функцию целиком перед фиксом: либо
  все пути должны возвращать одно и то же (унифицировать `return`-ы под объявленный тип,
  **не меняя, какие ветки исполняются**), либо аннотация должна стать `-> int | None`.
- `assignment` 3, `arg-type` 3 (включая `VideoTask(user_id=..., chat_id=...)` с
  `Any | None` на `:581,582` — None-guard перед конструктором), `attr-defined` 1
  (`_TemporaryFileWrapper`/`None` на `:71,72` — implicit Optional на файловом хендле).

### `bot/validation.py` (2: `arg-type` 1, `var-annotated` 1)

- `:26` `isinstance(x, <object>)` — второй аргумент `isinstance` не тип; читать код,
  вероятно передаётся переменная вместо класса/tuple классов — исправить на актуальный
  `_ClassInfo`.
- `:78` `Need type annotation for "parts"` — добавить `parts: list[str] = []` (по факту
  использования, не гадать).

### `bot/skill_script_routing.py` (2: `assignment` 2)

- `:76,78` — переменная объявлена/выведена как `Sequence[str]`, но присваивается
  `list[dict]`/`dict[str, Collection[Any]]` — читать функцию, вероятно нужно либо
  переименовать/перетипизировать переменную (если это два разных по смыслу значения под
  одним именем), либо это реальная логическая ошибка вне объёма T13 — если тесты `# not
  covered`, зафиксировать в отчёте, не чинить поведение молча.

**Тесты.** `tests/test_database.py`, `tests/test_db_handle.py`, `tests/test_haiper_*.py`
(любые файлы, упомянутые в `AGENTS.md` про haiper), `tests/test_validation*.py` (если есть)
— прогнать полностью после правок; для `utils.py`/`session_otel.py`/`session_logger.py`
искать соответствующие тесты по имени модуля.

**Готово, когда.** 0 ошибок mypy по всем 8 файлам группы; pytest своей области зелёный;
никаких новых веток поведения без явной пометки в отчёте.

---

## T13c — `bot/openai_helper.py`, `bot/plugins/plugin.py`, все остальные плагины и файлы
не из T13a/T13b (165 ошибок, список файлов — полный, ничего не подразумевается)

**Владение файлами (точный список из текущего скана, 0-ошибочные файлы не перечислены —
если появятся новые ошибки в файле этой группы вне списка, он всё равно принадлежит T13c
по правилу "остальные плагины"):** `bot/openai_helper.py`, `bot/plugins/plugin.py`,
`bot/agent_delivery.py`, `bot/ai_events.py`, `bot/chat_run.py`, `bot/conversation_key.py`,
`bot/html_utils.py`, `bot/openai_tool_handler.py`, `bot/plugins/agent_cron.py`,
`bot/plugins/agent_tools.py`, `bot/plugins/ask_your_pdf.py`, `bot/plugins/auto_tts.py`,
`bot/plugins/chief.py`, `bot/plugins/codeinterpreter.py`,
`bot/plugins/conversation_analytics.py`, `bot/plugins/crypto.py`,
`bot/plugins/ddg_image_search.py`, `bot/plugins/ddg_translate.py`,
`bot/plugins/ddg_web_search.py`, `bot/plugins/github_analysis.py`,
`bot/plugins/google_web_search.py`, `bot/plugins/hindsight_memory.py`,
`bot/plugins/iplocation.py`, `bot/plugins/language_learning.py`, `bot/plugins/mcp_server.py`,
`bot/plugins/prompt_perfect.py`, `bot/plugins/reaction.py`, `bot/plugins/reminders.py`,
`bot/plugins/show_me_diagrams.py`, `bot/plugins/skills.py`, `bot/plugins/spotify.py`,
`bot/plugins/stable_diffusion.py`, `bot/plugins/task_management.py`,
`bot/plugins/terminal.py`, `bot/plugins/text_document_qa.py`,
`bot/plugins/text_summarizer.py`, `bot/plugins/weather.py`, `bot/plugins/webshot.py`,
`bot/plugins/website_content.py`, `bot/plugins/wolfram_alpha.py`,
`bot/plugins/youtube_audio_extractor.py`, `bot/plugins/youtube_transcript.py`.

### 0. `bot/plugins/plugin.py` — базовые сигнатуры (делать первым в T13c, до остального —
см. "Кросс-групповые риски" ниже)

- `get_spec(self) -> [Dict]:` (`:118`) → `-> List[Dict]:` (паттерн #2). `List`/`Dict` уже
  импортированы в файле (используются в других сигнатурах, например
  `on_before_chat_request`) — проверить импорт, не дублировать.
- `on_before_chat_request(...) -> List[Dict]:` (`:64-66`) → `-> list | None:` (паттерн #5).
  Тело (`return messages`) не меняется — `list` совместим с `list | None`.
- Это снимает 3 `override`-ошибки разом: `skills.py:219`, `agent_tools.py:356`,
  `hindsight_memory.py:2445`.

### 1. `bot/openai_helper.py` (25: `assignment` 10, `arg-type` 10, `index` 3, `valid-type` 1,
`attr-defined` 1)

Точечно, файл большой — не читать целиком заново без необходимости (уже описан в
`AGENTS.md`/T09-T11 планах); смотреть конкретные строки из mypy-вывода на этом заходе.
Implicit-Optional (#1) и None-guard, где `assignment`/`arg-type` — обычный паттерн этого
проекта (Optional-параметры конфигурации/сессии). `valid-type` 1 — вероятно ещё один
`-> [Dict]`-подобный случай, проверить сначала, не факт что тот же паттерн.

### 2. Остальные плагины — `valid-type` "Bracketed expression" (по 1 на файл, 22 файла
всего в дереве считая T13b/T13a — здесь: `agent_tools.py` не в списке (уже `List[Dict]`),
но `auto_tts.py, crypto.py, ddg_image_search.py, ddg_translate.py, ddg_web_search.py,
github_analysis.py, google_web_search.py, iplocation.py, language_learning.py,
prompt_perfect.py, reaction.py, show_me_diagrams.py(нет — там return-value, проверить),
spotify.py, stable_diffusion.py, task_management.py, text_summarizer.py, weather.py,
webshot.py, website_content.py, wolfram_alpha.py, youtube_audio_extractor.py,
youtube_transcript.py`) — все одинаковый фикс #2: `def get_spec(self) -> [Dict]:` →
`-> List[Dict]:`, добавить `from typing import List, Dict` если файл их ещё не
импортирует (частый случай в мелких плагинах — проверять индивидуально, не предполагать).
**Это чисто механическая правка на ~20 файлов — кандидат делегировать
`code-generator`-агенту одним заданием с точным списком файлов, паттерном "до/после" и
критерием готовности "0 valid-type ошибок в каждом файле, никаких других правок", если
исполнитель T13c решит так организовать работу.**

### 3. `hindsight_memory.py` (21: `union-attr` 17, `assignment` 2, `attr-defined` 1,
`override` 1 — override снят пунктом 0)

`union-attr` 17 — `HindsightClient | None` (методы `list_memories/stats/recall/clear_bank`,
`:2765,2774,3166,3182,3227,3320,3324,3354,3362`), `MaybeInaccessibleMessage | None`
(`:2835`), `Message | None` (`:3295,3303`). Клиент, вероятно, ленивая инициализация
(`self._client: HindsightClient | None = None`) — None-guard перед каждым вызовом метода,
не менять момент инициализации.

### 4. `reminders.py` (18: `union-attr` 17, `assignment` 1) и `skills.py` (16: `arg-type` 7,
`operator` 4, `union-attr` 3, `override` 1 [снят пунктом 0], `misc` 1) и `agent_tools.py`
(16: `arg-type` 6, `assignment` 4, `misc` 2, `no-redef` 2, `override` 1 [снят пунктом 0],
`union-attr` 1)

- `agent_tools.py` `no-redef` `:2647,2648` (`blocked_transitions`/`completed_transitions`
  "already defined on line 2579/2580") — две переменные с одинаковыми именами в разных
  ветках одной функции (`_manage_plan_tasks`, см. `AGENTS.md` "Deterministic Routing"
  раздел — строки дрейфуют, но функция named). Если ветки взаимоисключающие (`if
  action=='add': ... else: ...`) — mypy всё равно может жаловаться при определённой
  структуре (`for`/повторный проход) — читать код перед переименованием одной из пар
  (например `blocked_transitions_add`/`blocked_transitions_update`, если семантически
  разные, или объединить объявление, если это буквально одна и та же переменная в двух
  последовательных, не вложенных, блоках — тогда `no-redef` обычно значит другой *тип*
  выведен во второй раз, не просто имя, проверить это в первую очередь).
- `reminders.py` `union-attr` 17 — вероятно тот же Telegram Optional-паттерн
  (`Message | None`/`Chat | None` в reminder-хендлерах) — стандартный None-guard.
- `skills.py` `operator` 4, `arg-type` 7 — точечно, не паттерн из списка выше без проверки.

### 5. Остальные файлы с 1-9 ошибками (`agent_delivery.py, ai_events.py, chat_run.py,
conversation_key.py, html_utils.py, openai_tool_handler.py, agent_cron.py, ask_your_pdf.py,
chief.py, codeinterpreter.py, conversation_analytics.py, mcp_server.py, terminal.py,
text_document_qa.py`) — по 1-9 ошибок каждый, смешанные коды (`assignment`, `misc`,
`return`, `return-value`, `var-annotated`, `call-overload`, `operator`). Точечно, читать
каждую ошибку на месте перед фиксом — слишком мало вхождений на файл для общего паттерна
сверх #1/#3, кроме отдельно отмеченных.

`mcp_server.py:658` `var-annotated` ("Need type annotation for result") — единственная
ошибка файла (заметки про `--check-untyped-defs` на `:73-86` не ошибки, а info-notes про
непроверяемые untyped-функции — **не включать `--check-untyped-defs`**, это не входит в
задачу T13 и изменило бы объём проверки всего проекта, а не только 0-ошибок в
перечисленных файлах).

**Тесты.** `tests/test_agent_tools_*.py`, `tests/test_hindsight_mutator.py`,
`tests/test_chat_modes_registry.py` (проверка `plugin.py` override не ломает загрузку
режимов), `tests/test_plugin_manager.py`, `bot/tests/test_mcp_server.py` — прогнать после
правок; особое внимание `tests/test_plugin_manager.py`/`tests/test_chat_modes_registry.py`
после правки `plugin.py` (пункт 0), так как это меняет сигнатуру базового класса, от
которого наследуются все плагины проекта.

**Готово, когда.** 0 ошибок mypy по всем файлам списка; `override`-ошибки в
`skills.py`/`agent_tools.py`/`hindsight_memory.py` сняты через `plugin.py`, не через
локальные `# type: ignore`; pytest полного набора плагинов зелёный.

---

## Кросс-групповые риски и порядок

1. **`plugin.py` (T13c) меняет сигнатуры базового класса `Plugin`.** Каждый плагин во ВСЕХ
   трёх группах (T13b: `haiper_image_to_video.py`; T13a не содержит плагинов) наследует
   `Plugin`. Проверено заранее: `haiper_image_to_video.py` уже объявляет
   `get_spec(self) -> List[Dict]:` (корректно) и не переопределяет
   `on_before_chat_request` — при текущем состоянии кода правка `plugin.py` **не должна**
   породить новых ошибок в T13b. Тем не менее: **T13c должен внести правку в `plugin.py`
   одной из первых своих правок** (пункт 0 выше), затем сразу перезапустить полный `mypy`
   (не только на своих файлах) и сверить, не появилось ли новых ошибок в файлах T13a/T13b.
   Если появились — не чинить чужой файл, сообщить владельцу группы в отчёте (по общему
   правилу владения файлами); T13a/T13b, в свою очередь, должны перезапустить mypy на
   своих файлах ближе к концу работы (после того как T13c объявит, что `plugin.py` готов),
   а не полагаться только на снимок ошибок из этого плана.
2. **Порядок волны 8 — все три подзадачи параллельны** (мастер-план явно указывает
   "Параллельно по группам файлов"), но правка `plugin.py` логически должна произойти
   раньше остальной части T13c, чтобы её последствия (если будут) успели проявиться и
   попасть в отчёт до финального ревью волны 9.
3. Никакой другой известный кросс-групповой эффект не найден: `database.py`/`db_handle.py`
   (T13b) не имеют публичных сигнатур, от которых зависят T13a/T13c по типам (используются
   через `Database`/`DbHandle`, но найденные ошибки — все внутренние для файла, не в
   публичных сигнатурах, на которые ссылаются другие файлы). `telegram_bot.py` (T13a) не
   экспортирует типов, от которых зависят T13b/T13c.
