# T08 — Deadlock между транзакцией `DbHandle` и синхронным чтением настроек

Статус: план (только чтение кода, код не менялся). Автор: планировщик-архитектор.
Источники: `docs/audit_remediation_plan_2026-09-04.md` (раздел T08),
`docs/architecture_code_review_2026-09-04.md` §3.1, `AGENTS.md` (Database Rules, Hooks,
Plugin-owned tables).

Короткий словарь терминов, которые встречаются ниже (по просьбе — объяснять термины сразу):

- **event loop (цикл событий)** — один-единственный поток программы, в котором по очереди
  выполняются все `async`-обработчики бота. Пока в этом потоке выполняется *синхронный*
  (блокирующий) код, цикл не может переключиться ни на что другое — бот не отвечает вообще
  никому, пока блокировка не снята.
- **deadlock (взаимная блокировка)** — ситуация, когда сторона A ждёт, пока освободится ресурс,
  который держит сторона B, а сторона B, в свою очередь, не может продолжить и освободить его,
  пока не получит что-то от A. Оба стоят вечно.
- **`threading.RLock`** — обычный «замок» уровня операционной системы. Его держит конкретный
  поток (thread); другой поток, пытающийся его взять, блокируется (зависает) физически, пока
  владелец не отпустит. Реентерабельный — значит, тот же поток может взять его повторно без
  блокировки самого себя.
- **`asyncio.Lock`** — «лёгкий» замок для корутин (async-функций) внутри одного event loop. В
  отличие от `threading.RLock`, ожидание на нём НЕ блокирует поток физически — другие задачи
  продолжают выполняться. Но он **не реентерабельный**: если корутина, уже держащая
  `asyncio.Lock`, попробует взять его второй раз сама у себя — она зависнет навсегда (сама у
  себя же).
- **`ContextVar`** — переменная, которая «привязана» к конкретной асинхронной задаче (task) и
  автоматически видна во всех функциях, которые эта задача вызывает вложенно (через `await`), но
  не видна в других, параллельно выполняющихся задачах.
- **DB-воркер (worker thread)** — единственный дополнительный поток, в котором физически
  выполняются все обращения к SQLite (через `ThreadPoolExecutor(max_workers=1)`), чтобы не
  блокировать event loop напрямую.


## 1. Цель

Убрать сценарий полного зависания бота, при котором event-loop-поток блокируется навсегда,
ожидая замок (`Database._op_lock`), который держит DB-воркер, открывший
`DbHandle.transaction()` — а DB-воркер, в свою очередь, не может продолжить и отпустить замок,
потому что следующий шаг транзакции должен запланировать сам event loop, а он заблокирован.

Дополнительно (по заданию T08):
- добавить понятную ошибку вместо зависания при вызове async-метода БД изнутри тела
  `db_handle.transaction()` (второй, отдельный сценарий самоблокировки);
- добавить хотя бы предупреждение (warning) при синхронном вызове `Database.*` из потока
  event loop;
- убрать два «мёртвых» синхронных двойника, которые всё ещё числятся в задаче T08, но по факту
  недостижимы в проде (см. §3.5);
- проверить `db.shutdown()` на event-loop-потоке.


## 2. Репро (подтверждено)

Скрипт уже лежит в `/tmp/hunt/repro_deadlock.py`. Механизм: задача A открывает
`async with db_handle.transaction()`, задача B параллельно делает синхронный
`db.get_user_settings(1)` (именно так это происходит в проде — см. §3).

Запуск:

```
timeout 10 python3 -u /tmp/hunt/repro_deadlock.py
```

Результат: **процесс не печатает вообще ничего** (даже с `-u`, без буферизации вывода) и
завершается только по `timeout` с кодом `124` (SIGTERM снаружи). Это подтверждает, что
зависание — не «медленно», а абсолютное: даже внутренний `asyncio.wait_for(..., timeout=5)`
внутри скрипта не срабатывает, потому что таймер `wait_for` тоже должен исполниться в event
loop, а event loop физически стоит на `self._op_lock.acquire()` (блокирующий вызов
`threading.RLock`, не связанный с asyncio). Без внешнего `timeout`/`kill` процесс не
восстановится сам никогда — ровно то, что написано в аудите: «Таймаутов нет; бот стоит до
рестарта».


## 3. Анализ путей

### 3.1. Первый сценарий (основной, воспроизведённый): sync-чтение настроек против открытой транзакции

Механизм пошагово:

1. Плагин открывает `async with self.db_handle.transaction() as tx:` (пример:
   `bot/plugins/agent_tools.py:1340, 1906, 1926`).
2. `_Transaction.__aenter__` (`bot/plugins/db_handle.py:96-113`) запускает `Database.transaction()`
   на DB-воркере (через `run_in_executor`). Внутри `Database.transaction()` (`bot/database.py:
   158-180`) вызывается `get_connection()` (`bot/database.py:106-156`), который берёт
   `self._op_lock.acquire()` (строка 132) — **на DB-воркер-потоке**. Этот `threading.RLock`
   принадлежит теперь DB-воркеру и не будет отпущен, пока не завершится `_end()` внутри
   `__aexit__` (тоже на воркере).
3. Между `__aenter__` и `__aexit__` DB-воркер **простаивает** (задача выполнена, воркер свободен),
   но лок остаётся захваченным именно этим потоком — воркер должен сам вызвать `release()`, а он
   это сделает только когда event loop продолжит корутину задачи A и дойдёт до `__aexit__`.
4. Если в этот момент **любой другой код на event-loop-потоке** вызывает синхронный метод
   `Database`, который тоже проходит через `get_connection()` (например
   `Database.get_user_settings()`), event-loop-поток вызывает `self._op_lock.acquire()`
   **напрямую, блокирующе** — и физически замирает, потому что лок держит другой поток (воркер).
5. Event loop теперь не может выполнить вообще ничего, включая callback, который должен
   продолжить корутину задачи A и дойти до `__aexit__`, чтобы отпустить лок. Тупик навсегда.

Кто вызывает синхронное чтение настроек на loop-потоке — `PluginManager._request_user_settings`
(`bot/plugin_manager.py:175-203`), конкретно строки 182 и 192:
`get_user_settings(self.db, user_id)` → `bot/database.py` `Database.get_user_settings` (строка
611) → `get_connection()`. Через него:

| # | Вызывающий код | file:line | Асинхронный контекст? | Обёрнут в `user_settings_scope`? |
|---|---|---|---|---|
| 1 | `PluginManager.disabled_plugins_for_user` | `bot/plugin_manager.py:205-210` | публичный sync-метод | зависит от вызывающего |
| 2 | `PluginManager._active_plugin_instances` → используется в `dispatch_observe`, `dispatch_blocking`, `collect_fragments`, `collect_objects`, `apply_mutators` | `bot/plugin_manager.py:996-1163` | все 5 диспетчеров уже `async def` | да, если снаружи есть активный `user_settings_scope` для этого `user_id`; иначе нет |
| 3 | `OpenAIHelper._apply_user_disabled_plugins` → `resolve_allowed_plugins` (на каждом сообщении) | `bot/openai_helper.py:1301-1311, 1349` | `resolve_allowed_plugins` — `async def` | да, если вызвано внутри `_process_message_locked` |
| 4 | `_apply_before_chat_request_mutators` (проверка hindsight_memory disabled) | `bot/openai_helper.py:2940` | `async def` (line 2914) | да/нет — зависит от вызывающего |
| 5 | `_plugin_help_text` (текст `/help`) | `bot/telegram_bot.py:1132-1141`, вызывается из `help()` (`:1106`, async) | `_plugin_help_text` сам **sync** | нет — `/help` не оборачивается в scope |
| 6 | `_try_handle_plugin_prompt` (гейт прямого перехвата сообщения плагином) | `bot/telegram_bot.py:4621-4640` | `async def` | нет отдельной обёртки — работает, только если вызвано изнутри `_process_message_locked` |
| 7 | `_ensure_plugin_handler_allowed` | `bot/telegram_bot.py:5095-5106` | `async def` | нет |
| 8 | `handle_plugins_menu`, `handle_plugin_menu_callback` (`/plugins`) | `bot/telegram_bot.py:5564-5680` | `async def` | нет |
| 9 | `skills.py:_disabled_skills_for_user` (используется в `contribute_prompt_fragment`, `on_before_chat_request`, `execute`) | `bot/plugins/skills.py:916-926` | все 3 вызывающих — `async def` | зависит |
| 10 | `agent_tools.py:_active_skill_context_for_subagents` → `disabled_lookup(...)` | `bot/plugins/agent_tools.py:3131-3152`, вызывается из `_run_subagents` (`:3006`, async) | да | нет |

Единственное текущее место, где `user_settings_scope` реально активируется —
`bot/telegram_bot.py:4043-4064` (`process_message` → `_run_locked`), которое оборачивает
**только** `_process_message_locked`. Внутри этой обёртки первый sync-вызов настроек кэшируется
(`_RequestUserSettingsCache.loaded=True`) и повторно БД не читает — но именно **первый** вызов
внутри неё всё равно синхронный (см. `_request_user_settings`, `bot/plugin_manager.py:180-189`):
кэш только уменьшает *количество* обращений к БД за один запрос, а не убирает сам факт
синхронного чтения.

Вне этой обёртки (список ниже) `user_settings_scope` не активен вообще, и каждый вызов —
отдельное синхронное чтение БД:

- `bot/telegram_bot.py:686-727` (`_handle_direct_result`, dispatch_observe)
- `bot/telegram_bot.py:772-785` (`_dispatch_assistant_response_observer` — общий хелпер для
  нескольких потоков обработки)
- `bot/telegram_bot.py:853-883` (`_dispatch_session_before_delete`, `dispatch_blocking`)
- `bot/telegram_bot.py:1288-1292` (`/stats`, `collect_fragments`)
- `bot/telegram_bot.py:1583-1587` (меню настроек, `collect_objects`)
- `bot/telegram_bot.py:2925-2988` (`_process_vision_media_group`)
- `bot/telegram_bot.py:3086-3175` (`vision` — обработчик фото)
- `bot/telegram_bot.py:2665-2795` (`transcribe` — голосовые, `dispatch_observe("on_session_reset")`)
- `bot/telegram_bot.py:4711-4736` (`_mirror_plugin_exchange`)
- `bot/telegram_bot.py:4848-4892` (`handle_callback_inline_query`)
- `bot/plugins/agent_cron.py:269-305` (`_maybe_dispatch_autonomous_response_hook` — cron-задачи
  вызывают `helper.get_chat_response()` напрямую, минуя `telegram_bot.py` целиком)

Важно: так как все обновления Telegram обрабатываются как параллельные задачи **на одном и том
же** event-loop-потоке, транзакция, открытая в контексте пользователя X, при зависании блокирует
event loop **для всех пользователей одновременно** — это не «зависнет один чат», это «зависнет
весь бот».

### 3.2. Второй сценарий (пока не воспроизводился, но конструктивно возможен): async-вызов БД изнутри тела транзакции

`_run_in_db_thread` (`bot/database.py:244-250`) сериализует все `DbHandle`-вызовы через
`asyncio.Lock` (`_db_handle_transaction_lock`), КРОМЕ вызовов изнутри самой открытой транзакции
(они помечены `_DB_HANDLE_TRANSACTION_LOCK_BYPASS`, `bot/plugins/db_handle.py:265-275`). Если
код внутри `async with db_handle.transaction() as tx:` вызовет **не** `tx.execute`/`tx.fetch_*`,
а обычный `db_handle.execute(...)` или любой `db.*_async(...)`, этот вызов попробует взять тот
же `asyncio.Lock`, который эта же задача уже держит с момента `__aenter__` — а `asyncio.Lock` не
реентерабельный, так что задача зависнет сама на себе. Сейчас нарушителей нет (в
`bot/plugins/agent_tools.py:1340, 1906, 1926` внутри `tx:`-блока используются только
`tx.execute`), но защиты от будущей ошибки — тоже нет.

### 3.3. `db.shutdown()` на event-loop-потоке

`bot/telegram_bot.py:5930-5934` вызывает `self.db.shutdown()` синхронно из `async def cleanup`.
`Database.shutdown()` (`bot/database.py:224-242`) тоже берёт `self._op_lock` (`with
self._op_lock:`, строка 227) — тот же самый замок из сценария 3.1. Если на момент остановки бота
где-то ещё не закрылась открытая `DbHandle.transaction()`, `shutdown()` зависнет на loop-потоке
точно так же.

### 3.4. Ограничение существующего тестового контракта (важно для выбора дизайна)

`PluginManager.disabled_plugins_for_user` / `is_plugin_disabled_for_user` /
`disabled_skills_for_user` — публичные **синхронные** методы, и это жёстко закреплено в тестах:

- `tests/test_plugin_manager.py:557-660` — 6 тестов, все синхронные (`def test_...`, без
  `async`), напрямую дергают `pm.disabled_plugins_for_user(...)` / `pm.is_plugin_disabled_for_user(...)`
  с `FakeDB`, у которой есть **только** синхронный `get_user_settings`.
- `tests/test_plugin_hooks.py:370` — `monkeypatch.setattr(pm, "disabled_plugins_for_user",
  fake_disabled)`, где `fake_disabled` — обычная синхронная функция
  (`lambda user_id: {"a"} if user_id == 42 else set()`, строки 395/415/435). Тест проверяет, что
  именно этот метод, вызванный именно так (синхронно), фильтрует плагины во всех пяти
  диспетчерах (`dispatch_observe`, `dispatch_blocking`, `collect_fragments`, `apply_mutators`,
  строки 372-438).

Если сделать `disabled_plugins_for_user` асинхронным (переименовать сигнатуру, требовать
`await`), эти ~9 тестов ломаются, а `test_plugin_hooks.py:370` вообще перестаёт быть валидным
способом подмены поведения — придётся переписывать сам механизм тестирования хуков, а не только
добавлять `await` в местах вызова. Это меняет публичный контракт, которым тесты явно владеют, и
```
Каждый changed line должен трассироваться к запросу пользователя
```
здесь бы не выполнялось — потребовалось бы менять код, который сам по себе не имеет отношения к
delоку, просто чтобы освободить сигнатуру.


## 4. Выбранный дизайн

Комбинация «пояс и подтяжки» (belt-and-suspenders): один структурный барьер, который гарантирует,
что зависание **не может быть вечным** ни при каких обстоятельствах (даже если что-то пропущено
ниже), плюс точечное уменьшение вероятности срабатывания на самом горячем пути, плюс два
целевых guard'а, которые просит задача T08 явно.

### 4.1. Барьер safety-net: таймаут на `Database._op_lock` (главная защита)

Заменить блокирующий `self._op_lock.acquire()` (`bot/database.py:132`) на
`self._op_lock.acquire(timeout=N)` с понятной ошибкой при неудаче. Это не убирает саму гонку
(конкуренцию за один и тот же лок), но превращает **вечное зависание всего бота** в **быстрый,
понятный отказ одного запроса** через N секунд — а событийный цикл тем временем продолжает
работать для всех остальных, потому что физическая блокировка потока снимается по таймауту, а не
висит бесконечно.

Почему это главный барьер, а не побочный: он не зависит от того, нашли ли мы все ~15 мест
синхронного чтения настроек (§3.1) — он страхует вообще любой будущий или пропущенный
синхронный вызов `Database.*`, а не только `disabled_plugins_for_user`.

### 4.2. Уменьшение вероятности: асинхронная предзагрузка кэша настроек на самом горячем пути

Не менять сигнатуру `disabled_plugins_for_user` (см. §3.4 — тесты владеют этим контрактом).
Вместо этого — заполнять `_RequestUserSettingsCache` **асинхронным** запросом ДО того, как
что-либо внутри `user_settings_scope` попробует прочитать её синхронно. Тогда синхронный fallback
внутри `_request_user_settings` (`bot/plugin_manager.py:180-189`) просто не будет достигнут:
кэш уже `loaded=True`.

Новый метод в `PluginManager` (`bot/plugin_manager.py`, рядом с `user_settings_scope`, после
строки 173):

```python
async def preload_user_settings_async(self, user_id: int | None) -> None:
    """Populate the active user_settings_scope cache via the async DB path.

    No-op if there is no db, no user_id, or no active scope for this task —
    callers that skip this still fall back to the existing sync read inside
    `_request_user_settings`, now bounded by `Database`'s op-lock timeout.
    """
    if self.db is None or user_id is None:
        return
    cache = _request_user_settings_cache.get()
    if cache is None or cache.manager_id != id(self) or cache.user_id != user_id:
        return
    if cache.loaded:
        return
    get_async = getattr(self.db, "get_user_settings_async", None)
    if not callable(get_async):
        return
    settings = get_async(user_id)
    if inspect.isawaitable(settings):
        settings = await settings
    settings = settings if isinstance(settings, dict) else {}
    cache.disabled_plugins = frozenset(
        normalize_string_list(settings.get(USER_DISABLED_PLUGINS_SETTING))
    )
    cache.disabled_skills = frozenset(
        normalize_string_list(settings.get(USER_DISABLED_SKILLS_SETTING))
    )
    cache.loaded = True

@asynccontextmanager
async def user_settings_scope_async(self, user_id: int | None):
    """`user_settings_scope` + `preload_user_settings_async` in one call."""
    with self.user_settings_scope(user_id):
        await self.preload_user_settings_async(user_id)
        yield
```

(добавить `inspect` к импортам — уже импортирован в файле, строка 6; добавить
`asynccontextmanager` к `from contextlib import contextmanager` на строке 2.)

Обязательное подключение (минимум для закрытия аудиторской находки — самый частый путь,
«на каждом сообщении»): `bot/telegram_bot.py:4043-4064` (`process_message` → `_run_locked`):

```python
async def _run_locked():
    user_id = getattr(getattr(update, 'effective_user', None), 'id', None)
    plugin_manager = getattr(self.openai, 'plugin_manager', None)
    scope_async = getattr(plugin_manager, 'user_settings_scope_async', None)
    if callable(scope_async):
        async with scope_async(user_id):
            return await self._process_message_locked(...)
    return await self._process_message_locked(...)
```

(заменяет текущий синхронный `with user_settings_scope(user_id):`, строки 4046-4048).

Рекомендуемое (не блокирующее, можно отдельным follow-up), тем же паттерном
(`async with plugin_manager.user_settings_scope_async(user_id):`) обернуть остальные точки входа
из таблицы §3.1: `_dispatch_session_before_delete` (:853), `/stats` (:1288), меню настроек
(:1583), `transcribe` (:2665), `_process_vision_media_group` (:2925), `vision` (:3086),
`handle_callback_inline_query` (:4848), и `agent_cron.py:_maybe_dispatch_autonomous_response_hook`
(:269, через `plugin_manager` из `helper`). Эти пути уже защищены барьером из §4.1 (не смогут
зависнуть вечно), поэтому их можно доводить постепенно, без риска для срока T08.

### 4.3. Guard: понятная ошибка при async-вызове БД изнутри тела транзакции (§3.2)

Новая `ContextVar` в `bot/database.py` (рядом с существующей `_DB_HANDLE_TRANSACTION_LOCK_BYPASS`,
строки 18-21):

```python
_DB_HANDLE_TRANSACTION_OPEN = contextvars.ContextVar(
    "db_handle_transaction_open",
    default=False,
)
```

`bot/plugins/db_handle.py:19` — импортировать вместе с существующим:
`from ..database import _DB_HANDLE_TRANSACTION_LOCK_BYPASS, _DB_HANDLE_TRANSACTION_OPEN`.

`_Transaction.__aenter__` (`bot/plugins/db_handle.py:96-113`) — установить маркер сразу после
успешного открытия, перед `return scope`:

```python
        self._scope = scope
        self._open_token = _DB_HANDLE_TRANSACTION_OPEN.set(True)
        return scope
```

`_Transaction.__aexit__` (`:115-136`) — снять маркер в начале, до любых `return`:

```python
    async def __aexit__(self, exc_type, exc, tb) -> bool:
        open_token = getattr(self, "_open_token", None)
        if open_token is not None:
            _DB_HANDLE_TRANSACTION_OPEN.reset(open_token)
            self._open_token = None
        scope = self._scope
        ...
```

`Database._run_in_db_thread` (`bot/database.py:244-250`) — проверить маркер до входа в
`async with lock:`:

```python
    async def _run_in_db_thread(self, func, *args, **kwargs):
        """Запускает sync-функцию в единственном db-worker потоке.

        ВНИМАНИЕ: нельзя вызывать `db`/`db_handle` методы (кроме
        `tx.execute`/`tx.fetch_one`/`tx.fetch_all`) изнутри тела
        `async with db_handle.transaction() as tx:` — тот же таск уже держит
        `_db_handle_transaction_lock`, а `asyncio.Lock` не реентерабелен;
        вместо зависания здесь вылетает понятная ошибка.
        """
        lock = getattr(self, "_db_handle_transaction_lock", None)
        if lock is not None and not _DB_HANDLE_TRANSACTION_LOCK_BYPASS.get():
            if _DB_HANDLE_TRANSACTION_OPEN.get():
                raise RuntimeError(
                    "Database call attempted from inside an open "
                    "DbHandle.transaction() body on the same task; use "
                    "tx.execute/tx.fetch_one/tx.fetch_all instead."
                )
            async with lock:
                return await self._run_in_db_thread_unlocked(func, *args, **kwargs)
        return await self._run_in_db_thread_unlocked(func, *args, **kwargs)
    ```

`ContextVar` привязана к задаче (task), поэтому это не даёт ложных срабатываний для ДРУГИХ,
параллельно выполняющихся задач, которые просто законно ждут своей очереди на `_db_handle_transaction_lock`
— сработает только если ТА ЖЕ задача попробует войти в БД второй раз, находясь внутри своей же
открытой транзакции.

### 4.4. Guard: предупреждение о синхронном вызове `Database.*` с loop-потока

`bot/database.py`, `get_connection()` (строка 106) — добавить перед `if not hasattr(self._local,
'connection'):` (строка 116):

```python
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            logger.debug(
                "Sync Database call from the event loop thread; prefer the "
                "*_async method to avoid blocking the loop.",
                stack_info=True,
            )
```

Точка вставки — единственный проход, через который идут ВСЕ синхронные обращения к БД (и
`get_user_settings`, и `list_user_sessions`, и всё остальное), поэтому одно место покрывает все
случаи. Уровень `debug` — как и просит задача («хотя бы warning в debug»); не `raise`, чтобы не
ломать легитимные синхронные пути (инициализация в `__new__`, синхронные юнит-тесты,
`run_sync_blocking`).

### 4.5. `db.shutdown()` — не блокировать loop

`bot/telegram_bot.py:5930-5934`:

```python
            if hasattr(self, 'db') and self.db is not None:
                try:
                    await asyncio.to_thread(self.db.shutdown)
                except Exception as e:
                    logger.warning("Error shutting down database error=%s", log_exception_shape(e))
```

(заменяет прямой синхронный `self.db.shutdown()`, строка 5932). `asyncio.to_thread` уже
используется в проекте (см. T14 в общем плане), паттерн знаком.

### 4.6. Мёртвые синхронные двойники (входят в T08 по тексту задачи)

Проверено чтением кода — оба недостижимы из прод-пути:

- `bot/telegram_bot.py:248-273` (`_get_user_language`, sync). Единственный вызывающий во всей
  кодовой базе — сам метод; продовый код всегда идёт через `_get_user_language_async`
  (`:275-300`, вызывается из `prepare_user_language` `:303`, и напрямую `:1349, :1371`). Прямых
  вызовов `self._get_user_language(` в проде нет.
- `bot/openai_helper.py:2020-2051` (`should_force_non_stream_first_turn`, sync — именно она
  внутри вызывает sync `get_current_model` на `:2037`, что и превращает
  `bot/openai_helper.py:4133/4147/4154` в «достижимо». `bot/telegram_bot.py:668-671`
  (`_should_force_non_stream_first_turn`) жёстко требует именно `_async`-двойник
  (`should_force_non_stream_first_turn_async`, `:2053-2072`) и бросает `RuntimeError`, если его
  нет — то есть прод физически не может дойти до sync-версии.

Удалить оба метода (действие — deletion, не conversion: они не нужны никому, кроме тестов
ниже). `get_current_model` (sync, `:4133`) **не удалять** — у него отдельные легитимные прямые
юнит-тесты (`tests/test_openai_helper_tool_calls.py:962-969`) и использование как публичного
sync API в `tests/test_per_conversation_serialization.py:182`; после удаления
`should_force_non_stream_first_turn` он просто перестаёт быть достижим с loop-потока.

Тесты, которые надо перенести/удалить вместе с этим (иначе `pytest` упадёт на `AttributeError`):

- `tests/test_callback_authorization.py:673-699` — `test_auto_language_detects_persists_and_caches_first_contact`
  и `test_explicit_bot_language_is_default_and_user_setting_can_override` напрямую зовут
  `bot._get_user_language(update)`. Переписать на `await bot._get_user_language_async(update)`
  (тест-функции сделать `async def` + `@pytest.mark.asyncio`), проверки `bot.db.get_user_settings`/
  `save_user_settings` заменить на `get_user_settings_async`/`save_user_settings_async` (или
  оставить как `AsyncMock`, если фикстура `_make_bot` уже так собирает `bot.db` — проверить перед
  правкой).
- `tests/test_skills_agent_gate.py:365-383` — 4 теста `test_should_force_non_stream_first_turn_*`
  дергают `helper.should_force_non_stream_first_turn(42, 7)` напрямую. Либо удалить (поведение уже
  покрыто асинхронным двойником — не проверено отдельно, нужно свериться, есть ли симметричные
  `*_async`-тесты рядом в этом же файле), либо переписать на `await
  helper.should_force_non_stream_first_turn_async(42, 7)`.
- `tests/test_callback_authorization.py:204` — `should_force_non_stream_first_turn=MagicMock(return_value=True)`
  в фикстуре `FakeHelper`; после удаления метода в реальном классе оставлять мёртвый kwarg в
  fake-объекте необязательно вредно (fake сам по себе не проверяет сигнатуру), но раз ключ больше
  ничему не соответствует — убрать вместе с правкой, если аналогичный `_async`-kwarg (`:205`) уже
  покрывает всё, что тест реально использует.


## 5. Отвергнутые альтернативы

**(a) «Тихий fallback»: если кэш не предзагружен — не ходить в БД, вернуть «ничего не отключено»
+ warning, без safety-net-таймаута.** Отклонено как *единственное* решение: на путях без scope
(таблица §3.1, 10+ мест) это означает, что отключённые пользователем плагины **тихо перестают
быть отключёнными** ровно там, где нет предзагрузки — то есть при не найденном ещё месте
(а найти все 15+ мест руками — риск само по себе) поведение молча перестаёт быть корректным,
без явного отказа. Элемент этой идеи (предзагрузка кэша) вошёл в финальный дизайн (§4.2), но без
«тихого возврата пустого множества» — вместо этого путь без предзагрузки просто идёт по старому
синхронному чтению, которое теперь ограничено таймаутом (§4.1), а не подменяет результат молча.

**(b) Полная конверсия `disabled_plugins_for_user` / `is_plugin_disabled_for_user` /
`disabled_skills_for_user` в `async` + перевод всех вызывающих.** Технически самый «чистый»
способ убрать синхронное чтение отовсюду разом. Отклонено как основной путь из-за §3.4: это
меняет **публичный, протестированный** контракт (`tests/test_plugin_manager.py:557-660`,
`tests/test_plugin_hooks.py:370`), причём один из тестов (`test_plugin_hooks.py`) использует
`monkeypatch.setattr` именно на синхронный метод как способ подмены поведения в пяти разных
диспетчерах хуков — то есть ломается не просто вызов, а сам способ тестирования хуков. Плюс
конверсия требует правок в ~15 местах (`bot/telegram_bot.py` в 8 местах, `bot/openai_helper.py`
в 2, `bot/plugins/skills.py` в 1, `bot/plugins/agent_tools.py` в 1, плюс промотирование
`_plugin_help_text` и `_active_skill_context_for_subagents` в `async`) в самом большом и хрупком
файле проекта — риск пропустить `await` где-то (что молча вернёт «правдивый» объект-корутину
вместо булева/множества) выше, чем выигрыш, если таймаут (§4.1) и предзагрузка на горячем пути
(§4.2) уже закрывают и вероятность, и худший исход. Оставлено как задокументированная опция на
будущее — если guard из §4.4 в проде/стейджинге покажет регулярные срабатывания на конкретных
путях, тогда точечно перевести именно их.

**(c) Переписать `DbHandle.transaction()` так, чтобы всё тело выполнялось одной синхронной
операцией на воркере (без удержания `_op_lock` через `await` обратно в loop).** Самый
структурно правильный фикс именно для сценария §3.1 — устраняет саму возможность держать замок
поперёк возврата в event loop. Отклонено как основной путь: меняет контракт `TransactionScope`
(`bot/plugins/db_handle.py:22-79`) с «await каждый `tx.execute` по отдельности» на «передай
список операций/коллбэк один раз», что требует переписать все 3 использования в
`bot/plugins/agent_tools.py:1340, 1906, 1926` (сами по себе они короткие DELETE-пакеты и
технически перекладываются легко, но переписывание базового класса `db_handle.py` — более
рискованная зона, чем добавление таймаута и одной ContextVar). Как и (b), не даёт защиты от
похожих проблем в других частях `Database` — таймаут из §4.1 покрывает более широкий класс
случаев за меньший риск. Задокументировано как будущий шаг усиления, если у какого-то плагина
появится генеральная потребность в длинной транзакции.


## 6. Новые тесты

1. **Регрессионный тест на сам deadlock** (перенос `/tmp/hunt/repro_deadlock.py` в
   `tests/test_db_handle.py`): реальная `Database` на временном sqlite-файле (см. фикстуру `db`/
   `handle` уже в `tests/test_db_handle.py:1-20`), задача A открывает
   `async with handle.transaction() as tx:` и внутри `await asyncio.sleep(0.05)` между двумя
   `tx.execute`, задача B параллельно вызывает синхронный `db.get_user_settings(1)` (через
   `asyncio.to_thread`, чтобы не блокировать сам тестовый event loop pytest'а). Обернуть весь
   `asyncio.gather(...)` в `asyncio.wait_for(timeout=5)`. **До фикса** — тест должен
   зафиксировать `DatabaseLockTimeoutError`/`TimeoutError` от самого `_op_lock` (после §4.1) —
   то есть тест проверяет, что вместо вечного зависания прилетает быстрая понятная ошибка (или,
   если задача B сама вызвана через `to_thread`, а не напрямую на loop — что весь `gather`
   завершается за разумное время без зависания процесса).
2. **`bot/database.py`: таймаут `_op_lock`.** Юнит-тест: два потока (`threading.Thread`),
   первый держит `get_connection()` открытым дольше настроенного таймаута (маленький, например
   `monkeypatch.setenv("DB_OP_LOCK_TIMEOUT_SECONDS", "0.2")`), второй пытается тоже войти в
   `get_connection()` — ожидается `DatabaseLockTimeoutError` (или выбранный класс исключения) за
   ~0.2 c, а не зависание теста.
3. **`bot/database.py`/`bot/plugins/db_handle.py`: guard на async-вызов внутри транзакции.**
   Внутри `async with handle.transaction() as tx:` вызвать `await handle.execute(...)` (не
   `tx.execute`) — ожидается немедленный `RuntimeError` с понятным текстом, а не зависание на
   `_db_handle_transaction_lock`. Обернуть в `asyncio.wait_for(..., timeout=2)` как
   защитную сетку самого теста.
4. **`bot/plugin_manager.py`: `preload_user_settings_async`.** С `FakeDB`, у которой
   `get_user_settings` (sync) считает вызовы и `get_user_settings_async` (async) тоже считает
   вызовы: внутри `user_settings_scope_async(42)` проверить, что `disabled_plugins_for_user(42)`
   возвращает корректное значение, а `FakeDB.get_user_settings` (sync) **ни разу** не вызван —
   вызван только `get_user_settings_async`. Дополнительно — тест на fallback: без обёртки
   `user_settings_scope_async` (только `user_settings_scope`, как раньше) поведение не меняется
   (используется существующий `test_user_settings_scope_reuses_disabled_plugin_and_skill_settings`,
   `tests/test_plugin_manager.py:634-660`, как регрессия «ничего не сломали»).
5. **`bot/telegram_bot.py`: `process_message` использует `user_settings_scope_async`.** Смок-тест
   (можно рядом с существующими тестами `process_message`/`_run_locked`, если такие есть — иначе
   через `FakeSettingsPluginManager`-подобный дубль из `tests/test_plugin_hooks.py`), что
   `preload_user_settings_async` реально вызывается до `_process_message_locked`.
6. **`db.shutdown()` не блокирует loop.** Юнит-тест на `bot/telegram_bot.py`: `cleanup()` вызывает
   `self.db.shutdown` через `asyncio.to_thread` (мокнуть `asyncio.to_thread` или сам `db.shutdown`
   и проверить, что вызов НЕ синхронный — например, через `AsyncMock`-обёртку и проверку, что
   awaited).
7. **Мёртвый код.** Убедиться, что `pytest --collect-only` после удаления
   `_get_user_language`/`should_force_non_stream_first_turn` не находит осиротевших ссылок (сам
   прогон тестов из raздела 7 это покажет).


## 7. Команды проверки

```bash
# точечные тесты, указанные в задаче + новые из §6 (добавятся в те же файлы)
python3 -m pytest tests/test_database.py tests/test_db_handle.py tests/test_plugin_hooks.py \
    tests/test_plugin_manager.py -q -p no:cacheprovider

# тесты, задетые удалением мёртвых sync-методов
python3 -m pytest tests/test_callback_authorization.py tests/test_skills_agent_gate.py \
    tests/test_openai_helper_tool_calls.py tests/test_per_conversation_serialization.py \
    -q -p no:cacheprovider

# сам репро-сценарий должен перестать зависать (после фикса §4.1 — быстрая ошибка/успех,
# не "тишина до kill by timeout")
timeout 10 python3 -u /tmp/hunt/repro_deadlock.py; echo "exit: $?"

# полный прогон — модуль общий (database.py, plugin_manager.py), задет широко
python3 -m pytest -q
```


## 8. Риски

- **Подбор таймаута `_op_lock` (§4.1).** Слишком маленький — ложные отказы под настоящей
  нагрузкой (несколько одновременных тяжёлых операций с БД). Слишком большой — медленное
  обнаружение реальных проблем. Сделать настраиваемым через env
  (`DB_OP_LOCK_TIMEOUT_SECONDS`, по аналогии с уже существующими `SQLITE_TIMEOUT`,
  `SQLITE_BUSY_TIMEOUT_MS` — `bot/database.py:118,127`) с разумным дефолтом (например, 10–15 c:
  заметно больше, чем любая одиночная sqlite-операция, но заметно меньше, чем терпение
  пользователя Telegram).
- **Неполное покрытие предзагрузкой (§4.2).** Из 11+ мест в таблице §3.1 в этом плане обязательно
  подключается только самое горячее (`process_message`). Остальные остаются с прежним
  поведением (синхронное чтение при первом обращении), но защищены таймаутом — то есть риск
  снижается с «весь бот виснет навсегда» до «единичный запрос иногда отказывает с понятной
  ошибкой при неудачном совпадении по времени с открытой транзакцией» (транзакции в дереве сейчас
  — только 3 коротких DELETE-пакета в `agent_tools.py`, так что окно конкуренции узкое). Довести
  остальные точки входа — рекомендуется отдельным follow-up тем же паттерном.
- **Удаление мёртвых sync-методов может задеть тесты, которые не были найдены этим проходом.**
  Перед удалением обязательно прогнать `python3 -m pytest -q` целиком (не только точечные файлы
  из §7), а не полагаться на список, найденный вручную.
- **`asyncio.get_running_loop()`-проверка в `get_connection()` (§4.4) — на каждый синхронный
  вызов БД добавляет один `try/except`.** Дешёво, но если этот путь окажется реально горячим
  (маловероятно — не должен быть, весь смысл фикса в том, чтобы убрать вызовы с loop-потока), при
  необходимости можно снизить частоту логирования (например, троттлинг раз в N секунд), но это
  не блокер для T08.
- **Guard из §4.3 (ContextVar-реентерабельность) — новое поведение, которого не было.** Если
  где-то в будущем плагине появится код, вызывающий `db_handle.execute()` (а не `tx.execute`)
  внутри тела транзакции, он теперь получит явную ошибку вместо зависания — это осознанное,
  документированное изменение поведения (fail fast вместо silent hang), не регрессия.


## 9. Критерии готовности

- `timeout 10 python3 -u /tmp/hunt/repro_deadlock.py` больше не молчит до принудительного
  завершения: скрипт либо печатает результат и завершается сам, либо ловит понятную ошибку —
  в обоих случаях `timeout` не должен требоваться, чтобы процесс закончился.
- Все команды из §7 зелёные (включая полный `python3 -m pytest -q`).
- Новые тесты из §6 добавлены и проходят; тест `test_plugin_hooks.py:370`
  (`monkeypatch.setattr(pm, "disabled_plugins_for_user", ...)`) продолжает проходить без правок
  — подтверждает, что публичный sync-контракт не тронут.
- `_get_user_language` (sync) и `should_force_non_stream_first_turn` (sync) отсутствуют в
  `bot/telegram_bot.py` / `bot/openai_helper.py`; их прямые тесты либо перенесены на `_async`-
  двойники, либо удалены с объяснением в diff'е, какой `_async`-тест их заменяет.
- `bot/plugins/db_handle.py`: docstring `transaction()` явно предупреждает про запрет вызывать
  другие `db`/`db_handle` методы внутри тела блока.
- Ни один из изменённых файлов не выходит за пределы списка: `bot/database.py`,
  `bot/plugins/db_handle.py`, `bot/plugin_manager.py`, `bot/telegram_bot.py`,
  `bot/plugins/agent_cron.py` (опционально, см. §4.2), `tests/test_database.py`,
  `tests/test_db_handle.py`, `tests/test_plugin_manager.py`, `tests/test_callback_authorization.py`,
  `tests/test_skills_agent_gate.py`.

## Постскриптум после ревью

Ревью (Sonnet, 2026-09-04): ошибок нет, два предупреждения — оба исправлены.

1. **Уровень лога синхронного вызова из event loop.** Было `logger.debug(...)` в
   `get_connection()`, а дефолтный уровень бота — `INFO` (`bot/__main__.py`), то есть сигнал
   в прод-логах не появлялся никогда. Теперь `Database._warn_if_sync_call_from_event_loop()`
   пишет `WARNING` с указанием call-site (первый кадр стека вне `bot/database.py` и
   `contextlib`), **один раз на каждый call-site** (реестр `_SYNC_CALL_WARNED_SITES` +
   `threading.Lock`), чтобы горячие пути (`/help`, `/stats`, ...) не засоряли лог. Тест:
   `tests/test_database.py::test_sync_call_from_event_loop_logs_warning_once_per_site`.
2. **Окно пересоздания executor после `shutdown()`.** После переноса `db.shutdown()` в
   `asyncio.to_thread` (в `cleanup()`) фоновая задача, не успевшая отмениться за 5 с, могла
   сделать async-вызов к БД и молча породить новый `ThreadPoolExecutor` с осиротевшим
   воркером. Добавлен флаг `_shutdown_started` (ставится в `shutdown()` под `_op_lock`);
   `_get_executor()` при установленном флаге бросает `RuntimeError("Database is shut down...")`.
   `test_shutdown_idempotent` переписан (раньше он закреплял именно старое поведение —
   пересоздание пула), добавлен
   `test_async_call_after_shutdown_raises_instead_of_spawning_executor`.

Отклонения разработчика от плана (маркер с `owner_task`, guard в фасаде `DbHandle`,
`try/except ValueError` вокруг `ContextVar.reset`, прямой sync-вызов в регрессионном тесте,
трёхуровневый выбор scope настроек) ревьюер проверил трассировкой и тестами и признал
необходимыми или безопасными; оставлены как есть.
