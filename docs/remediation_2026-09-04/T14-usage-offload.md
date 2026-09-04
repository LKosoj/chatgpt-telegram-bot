# T14. `UsageTracker` (SQLite + fsync) и PIL-декод на event loop

Небольшое пояснение терминов, которые дальше встречаются часто:
- **event loop** — единый поток, в котором бот параллельно обслуживает всех пользователей
  Telegram через `async`/`await`. Если внутри `async`-функции выполнить обычный
  (синхронный, блокирующий) вызов — например, запись в файл с `fsync` — весь бот на это
  время «замирает» для всех остальных пользователей, а не только для текущего.
- **`asyncio.to_thread(func, *args)`** — способ вынести синхронный вызов `func` в отдельный
  поток ОС (worker thread), чтобы event loop не блокировался. Сам `func` при этом
  выполняется как обычно, синхронно, просто не на главном потоке.
- **`threading.RLock`** — блокировка (замок) для потоков ОС: только один поток одновременно
  может находиться внутри `with lock:`. `RLock` (reentrant lock) отличается от обычного
  `Lock` тем, что один и тот же поток может брать его повторно без взаимной блокировки
  (deadlock) — это не используется в этой задаче напрямую, но важно знать при чтении кода.
- **fsync** — системный вызов, который заставляет ОС физически сбросить данные на диск,
  прежде чем продолжить. Это самая медленная часть записи файла.
- **read-modify-write гонка (race condition)** — ситуация, когда два потока одновременно
  читают общее состояние, оба вычисляют новое значение на основе старого и оба записывают
  результат — тогда одно из двух изменений «теряется». Ниже проверяется, что перенос кода в
  `to_thread` такую гонку не создаёт.

## Цель

Убрать синхронные (блокирующие event loop) операции из async-обработчиков
`bot/telegram_bot.py`:
1. Все точки записи/чтения `UsageTracker` (SQLite INSERT/UPDATE + JSON-снапшот с `fsync`),
   которые сегодня вызываются напрямую (без `to_thread`) из async-хендлеров.
2. Синхронный PIL-декод/конвертацию в `_telegram_image_as_png`.

Успех — все вызовы из таблицы ниже идут через `asyncio.to_thread`, конкурентные записи по
одному и тому же пользователю не теряют данные, существующие тесты остаются зелёными без
переписывания их логики (только точечные правки там, где это неизбежно — см. «Тесты»).

## Анализ

### Где лежит блокировка и что она защищает

- `UsageTracker._file_lock` — **атрибут класса** (`bot/usage_tracker.py:366`:
  `_file_lock = threading.RLock()`), то есть **один и тот же объект лока разделяют все
  экземпляры `UsageTracker`** (все пользователи, включая `guests`), а не только один
  пользователь. Он берётся на всё время критической секции «прочитать из SQLite → пересчитать
  → записать JSON-снапшот с `fsync`» в трёх местах:
  - `__init__` (импорт legacy JSON) — `bot/usage_tracker.py:414`
  - `prune_store` — `bot/usage_tracker.py:571`
  - `_record_usage_event` (главный путь записи любого события) — `bot/usage_tracker.py:579`
  - `_write_usage` (сам JSON+`fsync`+`os.replace`) — `bot/usage_tracker.py:596`
- `_record_usage_event` (`bot/usage_tracker.py:578-591`) — единственная точка входа для всех
  `add_*`: `self.store.record_event(...)` (SQLite INSERT в `usage_events` +
  `INSERT OR IGNORE`/`UPDATE` в `usage_daily_aggregates`, см. `_record_event`
  `bot/usage_tracker.py:162-229`) → `self._refresh_usage(...)` (читает свежие суммы обратно из
  SQLite) → `self._write_usage()` (JSON tempfile + `os.fsync` `bot/usage_tracker.py:609` +
  `os.replace` `bot/usage_tracker.py:616`). Всё — под одним `with self._file_lock:`.
- Геттеры (`get_current_cost`, `get_current_token_usage`, `get_current_image_count`,
  `get_current_vision_tokens`, `get_current_tts_usage`, `get_current_transcription_duration`,
  `bot/usage_tracker.py:649-856`) лок **не берут** — они читают только из SQLite
  (`_UsageSQLiteStore.load_costs` / `load_usage_history`, каждый вызов открывает свежее
  соединение `sqlite3.connect(...)` в `_connect()` `bot/usage_tracker.py:44-55`, с `PRAGMA
  busy_timeout`). Это уже сегодня согласованное чтение (SQLite отдаёт либо состояние «до»,
  либо «после» чужой записи, никогда не «середину» — см. «Дизайн» ниже), так что читатель без
  лока не ломает целостность, он может лишь увидеть чуть более старые данные.

Вывод: лок уже существует и уже правильно закрывает ровно ту гонку, о которой просят
проверить в задаче («не теряют данные при конкурентных `add_*`»). Перенос вызовов в
`to_thread` **не требует нового лока** — код внутри `to_thread` выполняется как обычно,
просто не на потоке event loop, а существующий `RLock` как был общим на все потоки ОС, так и
остаётся. Ниже (раздел «Почему `to_thread` не ломает атомарность») — подробное обоснование.

### Таблица call-sites (актуальные `file:line`, не из старого аудита)

| # | Метод/функция | Файл:строка вызова | Контекст (async?) | Что вызывает |
|---|---|---|---|---|
| 1 | `_record_chat_usage` (определение) | `bot/telegram_bot.py:799-811` | метод сейчас **sync** `def`, вызывается из async-хендлеров | `record_chat_tokens(...)` → `t.add_chat_tokens(...)` |
| 1a | вызов `_record_chat_usage` | `bot/telegram_bot.py:2813` | внутри `async def` | — |
| 1b | вызов `_record_chat_usage` | `bot/telegram_bot.py:4310` | внутри `async def` | — |
| 1c | вызов `_record_chat_usage` | `bot/telegram_bot.py:4604` | внутри `async def` | — |
| 1d | вызов `_record_chat_usage` | `bot/telegram_bot.py:4928` | внутри `async def` | — |
| 1e | вызов `_record_chat_usage` (используется возврат) | `bot/telegram_bot.py:5012` | внутри `async def` | `if not result: await self.reset(...)` |
| 2 | `record_vision_tokens(...)` | `bot/telegram_bot.py:1064` | async | `t.add_vision_tokens(...)` |
| 3 | `record_vision_tokens(...)` | `bot/telegram_bot.py:1082` | async | то же |
| 4 | `record_vision_tokens(...)` | `bot/telegram_bot.py:3044` | async | то же |
| 5 | `record_vision_tokens(...)` | `bot/telegram_bot.py:3067` | async | то же |
| 6 | `record_vision_tokens(...)` | `bot/telegram_bot.py:3254` | async | то же |
| 7 | `record_vision_tokens(...)` | `bot/telegram_bot.py:3364` | async | то же |
| 8 | `record_vision_tokens(...)` | `bot/telegram_bot.py:3400` | async | то же |
| 9 | `record_image_request(...)` | `bot/telegram_bot.py:2598` | async | `t.add_image_request(...)` |
| 10 | `record_tts_request(...)` | `bot/telegram_bot.py:2652` | async | `t.add_tts_request(...)` |
| 11 | `record_transcription_seconds(...)` | `bot/telegram_bot.py:2768` | async | `t.add_transcription_seconds(...)` |
| 12 | `self.usage[user_id].get_current_cost()` | `bot/telegram_bot.py:1216` (`/stats`) | async | прямой вызов метода трекера |
| 13 | `get_remaining_budget(...)` | `bot/telegram_bot.py:1218` (`/stats`) | async | `usage[...].get_current_cost()` внутри |
| 14 | `is_within_budget(...)` | `bot/telegram_bot.py:5050` (каждое сообщение, `check_allowed_and_within_budget`) | async | `get_remaining_budget(...)` внутри |
| 15 | `_telegram_image_as_png` (определение) | `bot/telegram_bot.py:1020-1025` | async, но `Image.open(...).save(...)` внутри — sync PIL | вызывается из `:1050` |

Примечание по строке 5051 из старого аудита (`docs/architecture_code_review_2026-09-04.md`) —
номер устарел, актуальная строка **5050** (проверено чтением файла).

### Кто ещё вызывает `UsageTracker`/`record_*` — проверено, не нашлось

- `bot/plugins/*.py` — ни один плагин не импортирует `UsageTracker` и не вызывает
  `record_chat_tokens`/`record_image_request`/`record_vision_tokens`/`record_tts_request`/
  `record_transcription_seconds`/`is_within_budget`/`get_remaining_budget` (проверено
  построчным поиском по всем файлам `bot/plugins/*.py`).
- `bot/openai_tool_handler.py` — не упоминает `UsageTracker`/`usage_tracker` вовсе.
- `UsageTracker.add_current_costs` (`bot/usage_tracker.py:815-818`) не вызывается нигде в
  дереве — существующий мёртвый код, не трогаем (см. правило «не удалять чужой мёртвый код
  без запроса»).
- `prune_store` (retention-джоба) уже обёрнут в `asyncio.to_thread` на месте вызова
  (`bot/telegram_bot.py:938-942`, внутри `run_retention_cleanup_once`) — это **не часть T14**,
  но важный факт: он уже сегодня дерёт тот же `_file_lock` из worker-потока параллельно с тем,
  что все `add_*`/геттеры сейчас выполняются синхронно на event loop-потоке. То есть
  `_file_lock` уже сегодня используется из разных потоков ОС одновременно, и это уже работает
  корректно — T14 просто расширяет этот же (уже провалидированный практикой) механизм на
  оставшиеся вызовы.
- Уже существующий прецедент **такого же** паттерна `to_thread`-обёртки над PIL-конвертацией
  — `bot/telegram_bot.py:3209-3212` (`_convert_image` для vision-загрузки из медиагруппы) и
  аналогичный `_convert_audio`/`_convert_media_group_image` рядом. `_telegram_image_as_png`
  чинится по этому же образцу, ничего нового не изобретаем.

### Существующие тесты (что уже есть)

- `tests/test_usage_record_helpers.py` (710 строк) — тестирует **sync** `record_chat_tokens`,
  `record_image_request`, `record_vision_tokens`, `record_tts_request`,
  `record_transcription_seconds` и сам `UsageTracker` напрямую, без `await`. Это самый большой
  риск: если превратить эти функции в `async def` «на месте», все ~700 строк тестов сломаются
  (вызов корутины вместо результата). Поэтому дизайн ниже **не трогает** сигнатуры этих
  функций — добавляет отдельные `_async`-версии рядом (см. «Дизайн»).
- `tests/test_usage_budget.py` (75 строк) — тестирует sync `get_remaining_budget` напрямую на
  реальном `UsageTracker`. Тоже не трогаем сигнатуру.
- `tests/test_telegram_transcribe.py` — **единственное место**, где сквозь `bot.transcribe(...)`
  используется рукописный `FakeUsageTracker` (`bot/tests/test_telegram_transcribe.py:58-70`
  — актуально `tests/test_telegram_transcribe.py:58-70`) с методами только
  `get_current_cost()` и `add_transcription_seconds(seconds)`. После перевода вызовов на
  `_async`-версии этому классу нужно добавить `add_transcription_seconds_async` и
  `get_current_cost_async` — иначе тест упадёт с `AttributeError` (см. «Тесты», это
  обязательная правка, не опциональная).
- `tests/test_telegram_streaming.py`, `tests/test_telegram_builder_config.py` — используют
  **настоящий** `UsageTracker` (не фейк) через `self.usage[user_id] = make_usage_tracker(...)`
  и после прогона проверяют `bot.usage[42].usage["usage_history"][...]`. Поскольку дизайн ниже
  не меняет итоговое состояние `UsageTracker.usage` (только переносит выполнение в поток), эти
  тесты должны остаться зелёными без изменений — это не гарантия «на бумаге», а прямое
  следствие того, что `add_chat_tokens_async` вызывает **тот же** `add_chat_tokens` без
  изменения его тела.
- `tests/test_utils_send_long_response_file.py:30-47` — образец теста-паттерна
  `fake_to_thread`, который переиспользуем для новых тестов (см. «Тесты»).

## Дизайн

### Почему `to_thread` не ломает атомарность (ответ на прямой вопрос задачи)

1. `UsageTracker._file_lock` — это `threading.RLock()`, объявленный на уровне класса, то есть
   **один общий замок на все объекты `UsageTracker` в процессе**, а не по одному на
   пользователя. Значит, даже два разных пользователя, пишущих одновременно, серилизуются
   через один и тот же замок — про «гонку между двумя записями одного и того же пользователя»
   можно не думать отдельно: защита сильнее, чем «per-user».
2. `RLock.acquire()` — блокировка на уровне ОС-потоков, а не asyncio-цикла. Если два
   `asyncio.to_thread(...)` вызова стартуют почти одновременно, они попадают в **два разных
   рабочих потока** пула по умолчанию (`loop.run_in_executor(None, ...)`, размер по умолчанию
   — `min(32, os.cpu_count() + 4)`, обычно достаточно). Один поток берёт `_file_lock` первым и
   доходит до конца критической секции (SQLite write → refresh → JSON write → `fsync` →
   `os.replace`), второй в это время **блокируется внутри своего рабочего потока**, ожидая
   освобождения замка. Event loop не блокируется ни на секунду в обоих случаях — ждёт только
   рабочий поток.
3. Поэтому итоговая последовательность операций (кто первый записал — тот первый) идентична
   сегодняшней синхронной версии, где вызовы на event loop-потоке и так шли строго по очереди
   (event loop — один поток, никакого параллелизма внутри него в принципе не было). Перенос в
   `to_thread` **не увеличивает** параллелизм внутри `UsageTracker` — он просто перестаёт
   занимать общий поток event loop, из-за чего *другие* пользователи и хендлеры перестают
   ждать чужую запись на диск.
4. Геттеры (`get_current_cost` и т.д.) не берут `_file_lock`, но это не новая проблема:
   SQLite (WAL, `busy_timeout=5000`, см. `bot/usage_tracker.py:47`) гарантирует, что читающее
   соединение внутри `_connect()` видит консистентный снимок — либо до чужого commit, либо
   после, никогда «половину» строки. Плюс `self.usage` в Python переприсваивается целиком
   (`self.usage = self._snapshot_usage(...)`, `bot/usage_tracker.py:563-568`), а не мутируется
   поточечно — присваивание атрибута атомарно на уровне GIL, так что конкурентный читатель
   либо видит старый словарь целиком, либо новый целиком, никогда «наполовину собранный».
   Порядок сохранения (что раньше запишется — SQLite-агрегат или JSON-снапшот) на корректность
   не влияет, потому что геттеры читают только из SQLite, а JSON — это только экспортный
   снепшот для совместимости (см. docstring `UsageTracker`, `bot/usage_tracker.py:333-342`).

### (а) `bot/usage_tracker.py` — новые async-методы на `UsageTracker`

Тонкие обёртки `asyncio.to_thread(self.<sync-метод>, ...)`, без новой логики блокировки —
существующий `_file_lock` внутри вызываемого sync-метода уже всё защищает (см. выше).
Добавляются в конец класса `UsageTracker`, после `initialize_all_time_cost`
(`bot/usage_tracker.py:857-888`, сейчас последний метод класса и конец файла):

```python
    # --- Async wrappers (T14) -------------------------------------------
    # Тонкие обёртки над уже существующими sync-методами: выполняют тот же
    # код в отдельном потоке (asyncio.to_thread), чтобы не блокировать
    # event loop бота. Новый lock не нужен: _file_lock (class-level
    # threading.RLock, см. :366) уже сериализует критическую секцию
    # внутри каждого вызываемого sync-метода.

    async def add_chat_tokens_async(self, *args, **kwargs):
        return await asyncio.to_thread(self.add_chat_tokens, *args, **kwargs)

    async def add_image_request_async(self, *args, **kwargs):
        return await asyncio.to_thread(self.add_image_request, *args, **kwargs)

    async def add_vision_tokens_async(self, *args, **kwargs):
        return await asyncio.to_thread(self.add_vision_tokens, *args, **kwargs)

    async def add_tts_request_async(self, *args, **kwargs):
        return await asyncio.to_thread(self.add_tts_request, *args, **kwargs)

    async def add_transcription_seconds_async(self, *args, **kwargs):
        return await asyncio.to_thread(self.add_transcription_seconds, *args, **kwargs)

    async def get_current_cost_async(self):
        return await asyncio.to_thread(self.get_current_cost)
```

Требуется добавить `import asyncio` в шапку файла (сейчас его там нет — импорты
`bot/usage_tracker.py:1-9`).

Вне периметра (сознательно не трогаем сейчас, см. «Риски»): `get_current_token_usage`,
`get_current_image_count`, `get_current_vision_tokens`, `get_current_tts_usage`,
`get_current_transcription_duration` — используются только в `/stats`
(`bot/telegram_bot.py:1210-1215`), это команда «по запросу», а не «на каждое сообщение», и в
явном списке методов из формулировки задачи их нет. `add_current_costs` — мёртвый код, не
трогаем.

### (б) `bot/utils.py` — async-версии функций-оркестраторов

Проблема: `record_chat_tokens`/`record_image_request`/`record_vision_tokens`/
`record_tts_request`/`record_transcription_seconds` не просто дергают один метод трекера —
они через `_charge_user_and_guest` могут списать стоимость **дважды**: с личного трекера
пользователя и (если он не в `allowed_user_ids`) ещё и с общего трекера `guests`. Эту
оркестрацию нужно продублировать в async-варианте, но без копирования логики валидации/цены
(чтобы не разъезжались через полгода) — общие куски выносятся в маленькие приватные
хелперы.

```python
# --- добавить рядом с _charge_user_and_guest (bot/utils.py:817) ---------
async def _charge_user_and_guest_async(usage, config, user_id, charge_fn_async):
    if user_id not in usage:
        logging.warning(f'No UsageTracker for user_id={user_id}; skipping charge.')
        return False
    try:
        await charge_fn_async(usage[user_id])
        allowed_user_ids = config['allowed_user_ids'].split(',')
        if str(user_id) not in allowed_user_ids and 'guests' in usage:
            await charge_fn_async(usage['guests'])
        return True
    except Exception as e:
        logging.warning("Failed to record usage error=%s", log_exception_shape(e))
        return False


# --- вынести из record_chat_tokens (bot/utils.py:849-873) чистую часть ---
def _resolve_chat_token_charge(config, model, used_tokens, prompt_tokens, completion_tokens):
    """Ценовая логика record_chat_tokens без обращения к трекеру -- общая
    для sync- и async-версии, чтобы не дублировать (в т.ч. мутацию
    _LEGACY_TOKEN_PRICE_FALLBACK_WARNED_MODELS)."""
    cost, price_source = resolve_chat_cost(
        model=model,
        total_tokens=used_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        fallback_price_per_1k=config.get('token_price'),
        table=config.get('model_token_prices') or {},
    )
    if price_source == 'legacy_fallback' and model not in _LEGACY_TOKEN_PRICE_FALLBACK_WARNED_MODELS:
        _LEGACY_TOKEN_PRICE_FALLBACK_WARNED_MODELS.add(model)
        logging.warning(
            'No per-model price configured for model=%s; falling back to legacy '
            'TOKEN_PRICE=%s for chat token cost',
            model, config.get('token_price'),
        )
    metadata = {
        'model': model,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'price_source': price_source,
    }
    return cost, metadata


def record_chat_tokens(usage, config, user_id, used_tokens, *, model=None,
                        prompt_tokens=None, completion_tokens=None):
    used_tokens = _positive_int_usage(used_tokens, 'chat tokens')
    if used_tokens is None:
        return False
    cost, metadata = _resolve_chat_token_charge(config, model, used_tokens, prompt_tokens, completion_tokens)
    return _charge_user_and_guest(
        usage, config, user_id,
        lambda t: t.add_chat_tokens(used_tokens, cost=cost, metadata=metadata),
    )


async def record_chat_tokens_async(usage, config, user_id, used_tokens, *, model=None,
                                    prompt_tokens=None, completion_tokens=None):
    used_tokens = _positive_int_usage(used_tokens, 'chat tokens')
    if used_tokens is None:
        return False
    cost, metadata = _resolve_chat_token_charge(config, model, used_tokens, prompt_tokens, completion_tokens)
    return await _charge_user_and_guest_async(
        usage, config, user_id,
        lambda t: t.add_chat_tokens_async(used_tokens, cost=cost, metadata=metadata),
    )


async def record_image_request_async(usage, config, user_id, image_size):
    return await _charge_user_and_guest_async(
        usage, config, user_id,
        lambda t: t.add_image_request_async(image_size),
    )


async def record_vision_tokens_async(usage, config, user_id, used_tokens):
    used_tokens = _positive_int_usage(used_tokens, 'vision tokens')
    if used_tokens is None:
        return False
    return await _charge_user_and_guest_async(
        usage, config, user_id,
        lambda t: t.add_vision_tokens_async(used_tokens),
    )


async def record_tts_request_async(usage, config, user_id, text_length, tts_model):
    text_length = _positive_int_usage(text_length, 'TTS characters')
    if text_length is None:
        return False
    return await _charge_user_and_guest_async(
        usage, config, user_id,
        lambda t: t.add_tts_request_async(text_length, tts_model),
    )


async def record_transcription_seconds_async(usage, config, user_id, seconds):
    seconds = _positive_int_usage(seconds, 'transcription seconds')
    if seconds is None:
        return False
    return await _charge_user_and_guest_async(
        usage, config, user_id,
        lambda t: t.add_transcription_seconds_async(seconds),
    )
```

`record_image_request`/`record_vision_tokens`/`record_tts_request`/
`record_transcription_seconds` (sync, `bot/utils.py:880-912`) остаются как есть — не трогаем
их тела вообще, только добавляем `_async`-соседей.

Бюджет — та же логика (маленький общий хелпер для выбора `user`/`user_id`/`name` из
`update`, чтобы не дублировать ~10 строк):

```python
# --- вынести общий кусок выбора пользователя (bot/utils.py:759-767) -----
_BUDGET_COST_MAP = {
    "monthly": "cost_month",
    "weekly": "cost_week",
    "daily": "cost_today",
    "all-time": "cost_all_time",
    "total": "cost_all_time",
}


def _budget_user_and_name(update: Update, is_inline: bool):
    if is_inline and update.inline_query:
        user = update.inline_query.from_user
    elif update.callback_query:
        user = update.callback_query.from_user
    elif update.message:
        user = update.message.from_user
    else:
        user = update.effective_user
    return (user.id if user else None), (user.name if user else "unknown")


async def get_remaining_budget_async(config, usage, update: Update, is_inline=False) -> float:
    user_id, name = _budget_user_and_name(update, is_inline)
    if user_id is None:
        return 0.0
    if user_id not in usage:
        # Создание трекера (DDL + импорт legacy JSON) остаётся синхронным --
        # см. "Риски", это разовая операция на процесс/пользователя.
        usage[user_id] = make_usage_tracker(config, user_id, name)

    user_budget = get_user_budget(config, user_id)
    budget_period = config['budget_period']
    if user_budget is not None:
        current_cost = await usage[user_id].get_current_cost_async()
        return user_budget - current_cost[_BUDGET_COST_MAP.get(budget_period, "cost_month")]

    if 'guests' not in usage:
        usage['guests'] = make_usage_tracker(config, 'guests', 'all guest users in group chats')
    current_cost = await usage['guests'].get_current_cost_async()
    return config['guest_budget'] - current_cost[_BUDGET_COST_MAP.get(budget_period, "cost_month")]


async def is_within_budget_async(config, usage, update: Update, is_inline=False) -> bool:
    remaining_budget = await get_remaining_budget_async(config, usage, update, is_inline=is_inline)
    return remaining_budget > 0
```

`get_remaining_budget`/`is_within_budget` (sync, `bot/utils.py:742-800`) можно по желанию
переписать на использование новых `_BUDGET_COST_MAP`/`_budget_user_and_name` (убирает
дублирование словаря и if/elif), либо оставить как есть — они не в критическом пути после
перевода `bot/telegram_bot.py:1218,5050` на `_async`-версии и продолжают явно тестироваться
`tests/test_usage_budget.py`. Рекомендация: переиспользовать `_BUDGET_COST_MAP` и
`_budget_user_and_name` и в sync-версиях тоже — тело функции не меняется по смыслу, только
две группы строк заменяются на вызов уже существующих (после этой правки) констант/хелпера,
риск для тестов нулевой (нет изменения поведения, только источник данных для тех же значений).

### (в) `bot/telegram_bot.py` — PIL в `_telegram_image_as_png`

По образцу уже существующего `_convert_image` (`bot/telegram_bot.py:3209-3212`):

```python
    async def _telegram_image_as_png(self, file_id: str) -> io.BytesIO:
        image_bytes = await self.openai.download_file_as_bytes(file_id)
        temp_file_png = io.BytesIO()

        def _convert():
            Image.open(io.BytesIO(image_bytes)).save(temp_file_png, format='PNG')

        await asyncio.to_thread(_convert)
        temp_file_png.seek(0)
        return temp_file_png
```

`asyncio` уже импортирован в `bot/telegram_bot.py` (используется в `:938, 2738, 3012, 3212,
6237, 6244`), новых импортов не требуется.

## Правки по `file:line`

| # | Файл:строка | Правка |
|---|---|---|
| 1 | `bot/usage_tracker.py:1-9` | добавить `import asyncio` |
| 2 | `bot/usage_tracker.py:888` (конец класса `UsageTracker`) | добавить блок из 6 `_async`-методов (см. «Дизайн (а)») |
| 3 | `bot/utils.py:742-799` | (опционально, без изменения поведения) вынести `_BUDGET_COST_MAP` и `_budget_user_and_name`; добавить `get_remaining_budget_async`, `is_within_budget_async` |
| 4 | `bot/utils.py:817-828` (после `_charge_user_and_guest`) | добавить `_charge_user_and_guest_async` |
| 5 | `bot/utils.py:849-873` | выделить `_resolve_chat_token_charge`; `record_chat_tokens` вызывает её; добавить `record_chat_tokens_async` |
| 6 | `bot/utils.py:880-912` | добавить `record_image_request_async`, `record_vision_tokens_async`, `record_tts_request_async`, `record_transcription_seconds_async` рядом со sync-версиями |
| 7 | `bot/telegram_bot.py:31-33` | добавить в импорт из `.utils`: `get_remaining_budget_async`, `is_within_budget_async`, `record_chat_tokens_async`, `record_image_request_async`, `record_vision_tokens_async`, `record_tts_request_async`, `record_transcription_seconds_async` |
| 8 | `bot/telegram_bot.py:799-811` | `_record_chat_usage` → `async def`, `return await record_chat_tokens_async(...)` |
| 9 | `bot/telegram_bot.py:2813, 4310, 4604, 4928` | добавить `await` перед `self._record_chat_usage(...)` |
| 10 | `bot/telegram_bot.py:5012` | `result = await self._record_chat_usage(...)` |
| 11 | `bot/telegram_bot.py:1064, 1082, 3044, 3067, 3254, 3364, 3400` | `record_vision_tokens(...)` → `await record_vision_tokens_async(...)` |
| 12 | `bot/telegram_bot.py:2598` | `record_image_request(...)` → `await record_image_request_async(...)` |
| 13 | `bot/telegram_bot.py:2652` | `record_tts_request(...)` → `await record_tts_request_async(...)` |
| 14 | `bot/telegram_bot.py:2768` | `record_transcription_seconds(...)` → `await record_transcription_seconds_async(...)` |
| 15 | `bot/telegram_bot.py:1216` | `self.usage[user_id].get_current_cost()` → `await self.usage[user_id].get_current_cost_async()` |
| 16 | `bot/telegram_bot.py:1218` | `get_remaining_budget(...)` → `await get_remaining_budget_async(...)` |
| 17 | `bot/telegram_bot.py:5050` | `is_within_budget(...)` → `await is_within_budget_async(...)` |
| 18 | `bot/telegram_bot.py:1020-1025` | `_telegram_image_as_png` — PIL через `asyncio.to_thread` (см. «Дизайн (в)») |

Строки-обёртки (`record_chat_tokens`, `record_image_request`, `record_vision_tokens`,
`record_tts_request`, `record_transcription_seconds`, `get_remaining_budget`,
`is_within_budget`, `add_chat_tokens` и т.д. на `UsageTracker`) **не удаляются** — они остаются
единственной проверяемой сегодня реализацией в `tests/test_usage_record_helpers.py` и
`tests/test_usage_budget.py`, и не становятся мёртвым кодом: сигнатуры публичные, поведение не
меняется.

## Тесты

1. **`tests/test_telegram_transcribe.py`** (правка обязательна, иначе тест ломается после
   правки 13) — в `FakeUsageTracker` (сейчас `get_current_cost`/`add_transcription_seconds`,
   строки ~58-70) добавить:
   ```python
   async def add_transcription_seconds_async(self, seconds):
       return self.add_transcription_seconds(seconds)

   async def get_current_cost_async(self):
       return self.get_current_cost()
   ```

2. **Новый файл `tests/test_usage_tracker_async.py`** — юнит-тесты на `UsageTracker`:
   - `test_add_chat_tokens_async_routes_through_to_thread` — паттерн `fake_to_thread` как в
     `tests/test_utils_send_long_response_file.py:30-47`, только монки патчатся
     `bot.usage_tracker.asyncio.to_thread`; проверить, что вызывается именно
     `self.add_chat_tokens` с теми же аргументами и что итоговое состояние `tracker.usage`
     совпадает с прямым синхронным вызовом.
   - По одному аналогичному тесту на `add_image_request_async`, `add_vision_tokens_async`,
     `add_tts_request_async`, `add_transcription_seconds_async`, `get_current_cost_async`.
   - **Тест на конкурентность (прямой ответ на требование задачи)**: на одном реальном
     `UsageTracker(tmp_path)` запустить `await asyncio.gather(*[tracker.add_chat_tokens_async(100) for _ in range(20)])`
     (без моков `to_thread` — реальные потоки), затем проверить
     `sum(tracker.usage["usage_history"]["chat_tokens"].values()) == 2000` и через прямой
     SQLite-запрос к `usage.sqlite3` (`SELECT SUM(amount) FROM usage_events WHERE
     event_type='chat_tokens'`) — тоже `2000`. Это ловит именно «потерянное обновление»,
     если бы лок не работал.

3. **Новый файл `tests/test_usage_record_helpers_async.py`** (по образцу
   `tests/test_usage_record_helpers.py`, но не копия — минимальный набор): по одному тесту на
   `record_chat_tokens_async` (списывает и юзера, и гостя как sync-версия),
   `record_image_request_async`, `record_vision_tokens_async` (включая
   `_positive_int_usage`-guard на 0/отрицательные), `record_tts_request_async`,
   `record_transcription_seconds_async`, `get_remaining_budget_async`,
   `is_within_budget_async`. Каждый тест — copy-paste соответствующего sync-теста из
   `tests/test_usage_record_helpers.py`/`tests/test_usage_budget.py` с `async def test_...` +
   `await`.

4. **`bot/telegram_bot.py`-хендлеры** — добавить в существующий/новый файл
   (`tests/test_telegram_usage_offload.py`) проверки, что при реальном прогоне хендлера
   (`bot.vision(...)`, генерация изображения, TTS, транскрипция, финальный ответ чата,
   `check_allowed_and_within_budget`) синхронные методы `UsageTracker` вызываются **не
   напрямую с event loop**, а через `to_thread` — монки патчим `bot.usage_tracker.asyncio.to_thread`
   аналогично `fake_to_thread` и утверждаем, что список перехваченных имён функций содержит
   ожидаемое (`add_chat_tokens`, `add_vision_tokens`, `add_image_request`, `add_tts_request`,
   `add_transcription_seconds`, `get_current_cost`) для соответствующего сценария.
   `tests/test_telegram_streaming.py`/`test_telegram_builder_config.py` не переписываются —
   они и так проверяют итоговое состояние `UsageTracker.usage`, которое не меняется.

5. **`_telegram_image_as_png`** — новый тест (можно в том же
   `tests/test_telegram_usage_offload.py`): монки патчить `telegram_bot.asyncio.to_thread`
   через `fake_to_thread`, вызвать `bot._telegram_image_as_png(file_id)` с замоканным
   `self.openai.download_file_as_bytes`, проверить, что `to_thread` вызван ровно один раз для
   функции конвертации и что возвращённый `BytesIO` действительно содержит валидный PNG
   (например, читается обратно через `PIL.Image.open`).

## Команды проверки

```bash
# точечные тесты по изменённым файлам
~/.venvs/ctb/bin/python -m pytest tests/test_usage_tracker_async.py -v
~/.venvs/ctb/bin/python -m pytest tests/test_usage_record_helpers_async.py -v
~/.venvs/ctb/bin/python -m pytest tests/test_usage_record_helpers.py tests/test_usage_budget.py -v
~/.venvs/ctb/bin/python -m pytest tests/test_telegram_transcribe.py -v
~/.venvs/ctb/bin/python -m pytest tests/test_telegram_usage_offload.py -v
~/.venvs/ctb/bin/python -m pytest tests/test_telegram_streaming.py tests/test_telegram_builder_config.py -v

# полный прогон -- убедиться, что перевод call-site'ов на await ничего не сломал
# в соседних хендлерах (vision/image/tts/transcribe/chat)
~/.venvs/ctb/bin/python -m pytest tests -q
```

## Риски

- **Пропущенный `await` где-то из 18 правок** превращает вызов в «создали корутину и не
  дождались» — Python не упадёт сразу, просто событие использования не запишется (тихая
  потеря данных об использовании/стоимости). Смягчение: `pytest.ini` не включает
  `-W error::RuntimeWarning`, поэтому "coroutine was never awaited" уйдёт в stderr, а не в
  падение теста — стоит явно грепнуть по `record_.*_async(` / `_record_chat_usage(` /
  `is_within_budget_async(`/`get_remaining_budget_async(` в `bot/telegram_bot.py` и убедиться,
  что перед каждым вызовом стоит `await`, до/после правки.
- **`make_usage_tracker`/`UsageTracker.__init__` остаются синхронными** (DDL `CREATE TABLE IF
  NOT EXISTS` + разовый импорт legacy JSON, `bot/usage_tracker.py:368-416`) — это по-прежнему
  блокирует event loop, но только один раз на пользователя за время жизни процесса (после
  первого обращения трекер живёт в `self.usage[user_id]` до перезапуска бота). В явном списке
  методов из формулировки задачи `__init__`/`make_usage_tracker` не значился — сознательно
  оставлено вне периметра этой правки, чтобы не расширять её за пределы «минимального
  решения». Если это важно закрыть — понадобится либо `make_usage_tracker_async`, либо
  ленивая инициализация трекера в фоне; отдельная небольшая задача.
- **`/stats`-геттеры кроме `get_current_cost`** (`get_current_token_usage`,
  `get_current_image_count`, `get_current_vision_tokens`, `get_current_tts_usage`,
  `get_current_transcription_duration`, `bot/telegram_bot.py:1210-1215`) остаются
  синхронными — `/stats` вызывается реже, чем «каждое сообщение», поэтому риск ниже, но
  внутренне непоследовательно: строка 1216 (`get_current_cost`) станет `await`, а соседние
  1210-1215 — нет. Если нужна полная консистентность `/stats`, можно по аналогии добавить
  `get_current_token_usage_async` и т.д. тем же однострочным паттерном — это чистое
  расширение того же списка в «Дизайн (а)», без изменения архитектуры.
- **Дублирование sync/async функций в `bot/utils.py`** — осознанный компромисс ради
  совместимости с `tests/test_usage_record_helpers.py` (710 строк). Общая логика (валидация,
  расчёт цены, выбор пользователя из `update`) вынесена в приватные хелперы
  (`_resolve_chat_token_charge`, `_budget_user_and_name`, `_BUDGET_COST_MAP`,
  `_positive_int_usage` уже был общим), поэтому дублируется только «форма вызова» (sync/await),
  а не бизнес-логика — риск расхождения низкий.
- **Конкурентность на уровне ОС-потоков** — `asyncio.to_thread` использует общий executor по
  умолчанию (`min(32, os.cpu_count() + 4)` потоков). При очень высокой нагрузке возможна
  временная очередь на `_file_lock` (задержка ответа одному пользователю), но не дедлок и не
  потеря данных — см. «Дизайн: почему `to_thread` не ломает атомарность».

## Критерии готовности

- [ ] `bot/usage_tracker.py`: добавлен `import asyncio` и 6 `_async`-методов на `UsageTracker`.
- [ ] `bot/utils.py`: добавлены `_charge_user_and_guest_async`, `_resolve_chat_token_charge`,
      `record_chat_tokens_async`, `record_image_request_async`, `record_vision_tokens_async`,
      `record_tts_request_async`, `record_transcription_seconds_async`,
      `get_remaining_budget_async`, `is_within_budget_async`; старые sync-функции не изменили
      сигнатуру и поведение.
- [ ] `bot/telegram_bot.py`: все 18 позиций из таблицы «Правки по file:line» переведены на
      `await ..._async(...)`; `_record_chat_usage` — `async def`; `_telegram_image_as_png`
      PIL-часть — через `asyncio.to_thread`.
- [ ] `tests/test_telegram_transcribe.py`: `FakeUsageTracker` дополнен `_async`-методами.
- [ ] Новые тесты (`tests/test_usage_tracker_async.py`,
      `tests/test_usage_record_helpers_async.py`, `tests/test_telegram_usage_offload.py`)
      зелёные, включая тест на конкурентные `add_chat_tokens_async` (20 параллельных вызовов
      не теряют данные).
- [ ] `tests/test_usage_record_helpers.py`, `tests/test_usage_budget.py`,
      `tests/test_telegram_streaming.py`, `tests/test_telegram_builder_config.py` остаются
      зелёными без изменений их кода (кроме отмеченной правки в `test_telegram_transcribe.py`).
- [ ] Полный `pytest tests -q` зелёный.
- [ ] Грепом по `bot/telegram_bot.py` подтверждено, что ни один из новых `..._async(`/
      `_record_chat_usage(` вызовов не остался без `await`.

## Постскриптум после ревью (2026-09-04)

Реализовано по плану: async-обёртки `UsageTracker.*_async` через `asyncio.to_thread`
(`bot/usage_tracker.py`), `_charge_user_and_guest_async`, `_resolve_chat_token_charge`,
`record_*_async`, `get_remaining_budget_async`/`is_within_budget_async` (`bot/utils.py`), все
вызовы учёта в `bot/telegram_bot.py` переведены на них. Новые тесты:
`tests/test_usage_tracker_async.py`, `tests/test_usage_record_helpers_async.py`,
`tests/test_telegram_usage_offload.py`.

**Инцидент при верификации.** Тест `test_vision_streaming_final_edit_failure_falls_back_to_new_message`
после T14 подвисал навсегда. Причина: тест подменял `asyncio.sleep` на голый `AsyncMock()`, а
`BusyStatusMessage._run()` (`bot/utils.py:183`) крутится в цикле именно на `asyncio.sleep`. Пока
все `await` в тесте были мгновенными мок-корутинами, фоновая задача просто не успевала стартовать;
появившийся в T14 настоящий `await asyncio.to_thread(...)` вернул управление циклу событий — и
фоновый цикл, ничего не уступая, занял его целиком, из-за чего результат из рабочего потока
уже никогда не мог быть доставлен. Исправлено в тестах: `_instant_sleep` (`await
_REAL_ASYNCIO_SLEEP(0)`) вместо `AsyncMock()` в трёх местах `tests/test_telegram_streaming.py`.
Продакшен-код не при чём: с настоящим `asyncio.sleep` цикл статуса засыпает как положено.

**Ревью (Sonnet, persona reviewer).** Одна ошибка, исправлена:
- `_edit_image_from_context` (`bot/telegram_bot.py:1016`) остался на синхронном
  `record_image_request` — это вызов, добавленный параллельной задачей T13, поэтому его не было в
  таблице call-site'ов плана T14. Заменён на `await record_image_request_async(...)`, осиротевший
  импорт синхронной версии убран; тест `test_edit_image_from_context_records_usage`
  (`tests/test_telegram_streaming.py`) переведён на `AsyncMock`-двойник, чтобы не фиксировать
  старое поведение молча (обычный `Mock()` не awaitable, запись бы тихо провалилась).

Предупреждения:
- Правка в `bot/openai_tool_handler.py` (`DELIVERY_GRACE_ROUNDS`, гейт `_delivery_tool_is_allowed`)
  относится к T10, а не к T14 — учтено в постскриптуме T10.
- Не была покрыта гостевая ветка `get_remaining_budget_async` — добавлен тест
  `test_remaining_budget_async_initializes_user_and_guest_trackers` (зеркало синхронного из
  `tests/test_usage_budget.py`).
- Проверено отдельно: `UsageTracker._file_lock` (class-level `RLock`) нигде не удерживается
  основным потоком одновременно с ожиданием `to_thread` — взаимоблокировки нет.

