# T04 — agent_cron и reminders: JSON → SQLite с атомарным захватом

Владение файлами (по мастер-плану): `bot/plugins/agent_cron.py`, `bot/plugins/reminders.py`,
`tests/test_agent_cron_plugin.py`, `tests/test_reminders_fixes.py` (плюс новые файлы по маске
`tests/test_agent_cron*.py` / `tests/test_reminders*.py`). Никакие другие файлы не трогать.

## 0. Важное отклонение, которое нужно решить ДО реализации

Помимо двух тестовых файлов, перечисленных в мастер-плане, `RemindersPlugin` напрямую
используется ещё в двух тестовых файлах вне владения этой задачи:

- `tests/test_background_tasks.py` — 8 тестов, из них 6 (`test_reminders_tick_*`,
  `test_send_failure_increments_attempts_no_duplicate`,
  `test_poisoned_reminder_removed_after_max_attempts`, `test_bad_time_does_not_kill_tick`,
  `test_successful_send_persists_and_failed_persists_attempts`,
  `test_save_reminders_no_tmp_file_after_success`) напрямую пишут в `plugin.reminders` и
  зовут `plugin.save_reminders()` / читают `plugin.reminders_file` — методы и атрибуты,
  которые в SQLite-версии перестают существовать.
- `tests/test_concurrent_tool_state.py` — 4 теста создают `RemindersPlugin()` через
  `plugin.initialize(storage_root=...)` **без** `db=`, затем зовут `plugin.execute("set_reminder", …)`
  и читают `plugin.reminders[...]`. Без `db_handle` `execute()` не сможет писать в БД.

Это реальный конфликт владения файлами, а не гипотетический. Рекомендация: координатор
расширяет владение этой задачи на `tests/test_background_tasks.py` и
`tests/test_concurrent_tool_state.py` **только в объёме тестов, ссылающихся на internals
`RemindersPlugin`** (переписать на DB-фикстуру), либо переносит эту правку отдельной
задачей сразу вслед за T04. Альтернатива — не удалять `self.reminders`/`save_reminders()`,
а оставить их как мёртвый back-compat слой — отклонена: это прямо противоречит цели задачи
(мастер-план требует перехода на SQLite, а не дублирования хранилища) и создаёт двух
источников истины. **Если координатор не расширит владение, разработчик обязан
остановиться и сообщить об этом, а не молча чинить чужие файлы** (правило AGENTS.md:
не трогать файлы вне владения). Ниже план считает, что владение будет расширено; если
нет — шаги 5 и 7 (reminders) блокируются.

## 1. Мэппинг JSON → колонки: обоснование выбора

Мастер-план прямо говорит: «Поля — все поля текущего JSON-объекта задачи + status,
locked_at, locked_by» — т.е. просит колоночную схему, а не blob. Это отличается от
`hindsight_finalize_jobs.messages` (JSON-колонка), потому что там платёж — это список
сообщений переменной длины, который всегда читается/пишется целиком. Здесь наоборот:
и `agent_cron`-задача, и `reminder` — плоский словарь скалярных полей, часть которых
(`status`, `paused`, `next_run_at`, `fire_at_utc`, `time`) участвует в WHERE/SET по
отдельности при захвате. JSON-колонка потребовала бы разбора/пересборки всего блоба на
каждое точечное обновление (`job["status"]="running"`) — колоночная схема дешевле и прямее
отражает мастер-план. Решение: **полностью колоночная схема** для обеих таблиц.

## 2. Единое правило времени (риск, который иначе тихо всё сломает)

`next_run_at`/`created_at`/`last_started_at` и т.д. в agent_cron, а также `time` в
reminders — это `datetime.now().isoformat()`, т.е. **наивное локальное время сервера**, не
UTC. `fire_at_utc` в reminders — единственное поле, которое реально UTC-наивное
(`datetime.now(timezone.utc).replace(tzinfo=None)`, `reminders.py:241`). SQLite
`CURRENT_TIMESTAMP`/`datetime('now')` — это всегда UTC. Если сравнивать
`next_run_at <= CURRENT_TIMESTAMP` (как в примере `hindsight_finalize_jobs`,
`hindsight_memory.py:980`), due-время будет тихо сдвигаться на смещение часового пояса
сервера — регрессия, которую сложно заметить в тестах (CI обычно в UTC).

**Правило для обеих таблиц**: все сравнения времени — через параметры, вычисленные в
Python (`datetime.now().isoformat(timespec="seconds")` для локального/next_run_at/time;
`datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")` для
fire_at_utc), никогда не через `CURRENT_TIMESTAMP`/`datetime(...)` в SQL. Колонки времени —
`TEXT`, а не `TIMESTAMP`-affinity. `locked_at` — новое поле, полностью наше — тоже
хранится как ISO-строка локального времени (`datetime.now()`), чтобы не заводить третий
источник времени в одной таблице.

## 3. agent_cron: DDL

```sql
CREATE TABLE IF NOT EXISTS agent_cron_jobs (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    schedule TEXT NOT NULL,
    prompt TEXT NOT NULL,
    schedule_type TEXT NOT NULL,
    next_run_at TEXT,
    interval_seconds INTEGER,
    hour INTEGER,
    minute INTEGER,
    weekday INTEGER,
    status TEXT NOT NULL DEFAULT 'active',
    paused INTEGER NOT NULL DEFAULT 0,
    reply_to_message_id INTEGER,
    message_thread_id INTEGER,
    created_at TEXT NOT NULL,
    last_started_at TEXT,
    last_finished_at TEXT,
    last_error TEXT,
    last_tokens INTEGER,
    locked_at TEXT,
    locked_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_cron_jobs_due
    ON agent_cron_jobs(paused, next_run_at);
CREATE INDEX IF NOT EXISTS idx_agent_cron_jobs_scope
    ON agent_cron_jobs(scope);
```

Мэппинг 1:1 с текущим `job` dict (`agent_cron.py:355-368` `_create_job`, плюс поля,
добавляемые `_advance_job`/`_run_job`: `next_run_at`, `interval_seconds`, `hour`,
`minute`, `weekday`, `last_started_at`, `last_finished_at`, `last_error`, `last_tokens`).
Новые поля — `status` (уже было полем dict, просто закрепляется колонкой), `locked_at`,
`locked_by` (`str(os.getpid())`, без доп. `uuid` — один процесс держит один event loop, этого
достаточно, чтобы отличить чужой lease после рестарта; `os` уже импортирован).

Реализация: `AgentCronPlugin.register_schema(self) -> List[str]` возвращает эти три
`CREATE ...` операторами (по образцу `hindsight_memory.py:1179-1202`).

## 4. agent_cron: захват (claim)

Лизинг: `AGENT_CRON_JOB_LEASE_SECONDS = 1800` (30 мин) — новая module-level константа.
Обоснование: задача — это полный агентный ход (`helper.get_chat_response`, возможны
раунды tool-calls), потенциально дольше, чем hindsight-экстракция (у которой лиз 900с),
поэтому взят больший запас, а не то же число вслепую. Это допущение, не требование
мастер-плана — вынесено как риск ниже.

Массовый захват (проверяется раз в 60с из `_checker_loop`, заменяет
`_check_due_jobs`, `agent_cron.py:187-198`):

```python
def _claim_due_jobs_sync(self, db, now_iso: str, lease_cutoff_iso: str, worker_id: str) -> List[Dict[str, Any]]:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(
            '''
            SELECT id FROM agent_cron_jobs
            WHERE paused = 0
              AND next_run_at IS NOT NULL
              AND next_run_at <= ?
              AND (status != 'running' OR locked_at IS NULL OR locked_at <= ?)
            ORDER BY next_run_at ASC
            ''',
            (now_iso, lease_cutoff_iso),
        )
        job_ids = [row[0] for row in cursor.fetchall()]
        if not job_ids:
            return []
        placeholders = ",".join("?" for _ in job_ids)
        cursor.execute(
            f'''UPDATE agent_cron_jobs
                SET status='running', locked_at=?, locked_by=?, last_started_at=?
                WHERE id IN ({placeholders})''',
            (now_iso, worker_id, now_iso, *job_ids),
        )
        cursor.execute(f"SELECT * FROM agent_cron_jobs WHERE id IN ({placeholders})", job_ids)
        rows = [dict(r) for r in cursor.fetchall()]
    order = {jid: i for i, jid in enumerate(job_ids)}
    rows.sort(key=lambda r: order.get(r["id"], 0))
    return rows
```

Без `LIMIT` — специально: старый код обрабатывал в одном тике ВСЕ due-задачи
(`agent_cron.py:190-198` — двойной цикл без ограничения), лимит был бы поведенческим
изменением, которого мастер-план не просит.

Точечный захват по id (для `/cron run`, `_handle_job_action`, `agent_cron.py:172-175`,
не фильтрует `paused`/`next_run_at` — как и раньше, ручной запуск игнорирует паузу):

```python
def _claim_job_by_id_sync(self, db, job_id: str, scope: str, now_iso: str, lease_cutoff_iso: str, worker_id: str) -> Dict[str, Any] | None:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(
            '''UPDATE agent_cron_jobs
               SET status='running', locked_at=?, locked_by=?, last_started_at=?
               WHERE id = ? AND scope = ?
                 AND (status != 'running' OR locked_at IS NULL OR locked_at <= ?)''',
            (now_iso, worker_id, now_iso, job_id, scope, lease_cutoff_iso),
        )
        if cursor.rowcount == 0:
            return None
        cursor.execute("SELECT * FROM agent_cron_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
```

`scope`-фильтр сохраняет текущую изоляцию: pause/resume/remove/run сейчас ищут задачу
через `self.jobs.get(scope, {}).get(job_id)` (`agent_cron.py:149`, `:201`) — пользователь
не может тронуть чужую задачу по угаданному id. Это поведение обязано остаться.

`_run_job(self, bot, scope, job_id, *, manual=False)` — сигнатура не меняется (тесты
зовут её напрямую с `job["scope"], job["id"]`). Внутри: если `manual`, сперва
`row = await self.db_handle.run_sync(self._claim_job_by_id_sync, job_id, scope, ...)`;
`row is None` → лог + (для ручного пути) можно расширить `_handle_job_action`, чтобы он
ответил «job is already running», но это уже в зоне chat-команды, не самого `_run_job` —
`_run_job` в этом случае просто возвращается (не бросает, симметрично текущему
`if not job: return`, `agent_cron.py:202-203`). Для автоматического пути `job` уже пришёл
захваченным из `_claim_due_jobs_sync`, повторный захват не нужен.

`_advance_job(job: Dict) -> None` (`agent_cron.py:373-405`) — **не трогается**, остаётся
чистой функцией над dict (мутирует `paused`/`next_run_at`). После run: собрать
финальный dict полей для UPDATE (`status`, `last_finished_at`/`last_error`, `last_tokens`,
`paused`, `next_run_at`, `locked_at=NULL`, `locked_by=NULL`) одним UPDATE по `id`.
Это сохраняет 100% текущей логики `_advance_job` без изменений и минимизирует диф.

## 5. reminders: DDL

```sql
CREATE TABLE IF NOT EXISTS reminders (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    target_chat_id TEXT NOT NULL,
    time TEXT NOT NULL,
    fire_at_utc TEXT,
    message TEXT NOT NULL,
    integration TEXT NOT NULL,
    reply_to_message_id INTEGER,
    send_attempts INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    locked_at TEXT,
    locked_by TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reminders_owner ON reminders(owner_id);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status, fire_at_utc, time);
```

Поле `user_id` внутри старого reminder-dict всегда равно `owner_id` (`reminders.py:351`,
`"user_id": owner_id`) — отдельной колонки не нужно, восстанавливается при сборке
transient-dict для `send_reminder`. `created_at` — новая колонка, при обычном создании
`datetime.now().isoformat(timespec="seconds")`, при импорте старых записей (у них нет
created_at) — тоже текущее время на момент импорта (явное допущение, поле не участвует
ни в какой логике, только диагностика/сортировка).

## 6. reminders: захват «отправить и пометить»

`REMINDER_LEASE_SECONDS = 120` — с запасом над интервалом тика (60с, `reminders.py:226`),
но короткий, т.к. отправка — один HTTP-вызов Telegram, а не агентный ход.

```python
def _claim_due_reminders_sync(self, db, now_local_iso, now_utc_iso, lease_cutoff_iso, worker_id) -> List[Dict]:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(
            '''
            SELECT id FROM reminders
            WHERE (status != 'processing' OR locked_at IS NULL OR locked_at <= ?)
              AND (
                    (fire_at_utc IS NOT NULL AND fire_at_utc <= ?)
                 OR (fire_at_utc IS NULL AND time <= ?)
              )
            ''',
            (lease_cutoff_iso, now_utc_iso, now_local_iso),
        )
        ids = [r[0] for r in cursor.fetchall()]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cursor.execute(
            f"UPDATE reminders SET status='processing', locked_at=?, locked_by=? WHERE id IN ({placeholders})",
            (now_local_iso, worker_id, *ids),
        )
        cursor.execute(f"SELECT * FROM reminders WHERE id IN ({placeholders})", ids)
        return [dict(r) for r in cursor.fetchall()]
```

`check_reminders()` (`reminders.py:234-299`) переписывается на: захватить due-записи одним
запросом → для каждой (сохраняя порядок и изоляцию ошибок текущего `try/except` внутри
цикла, `reminders.py:274-296`) вызвать `send_reminder`; успех → `DELETE FROM reminders
WHERE id=?`; ошибка при `attempts >= _MAX_SEND_ATTEMPTS` → тоже `DELETE` (сдаться, как
сейчас); иначе → `UPDATE reminders SET send_attempts=?, status='pending', locked_at=NULL,
locked_by=NULL WHERE id=?` (снять лок, следующий тик снова подхватит — ровно текущая
семантика ретраев, просто без промежуточного файла). Обработка «битого времени»
(`reminders.py:262-270`, `ValueError`/`KeyError` при парсинге `time`) остаётся — теперь это
часть SQL-фильтра? Нет: `time`/`fire_at_utc` — TEXT, сравниваются лексикографически с ISO-
строкой; если в БД когда-то окажется не-ISO `time` (руками через `/cron`-аналог не бывает,
но старый импорт может принести испорченную запись), лексикографическое сравнение может
дать неверный, но не падающий результат — в отличие от старого кода, который явно ловил
`ValueError` и логировал skip. Т.к. запись изначально пишется самим плагином через
`datetime.strptime(...).isoformat()`, мусор может попасть только через `_import` из старого
файла — при импорте валидировать `time`/`fire_at_utc` через `datetime.fromisoformat` и
не выбрасывать битые записи, а импортировать как есть, но это единственное место, где
нужна defensive-проверка (см. §8).

## 7. Одноразовый импорт + переименование в `*.migrated`

Общий паттерн для обеих таблиц, метод синхронный, вызывается из `initialize()`
(`agent_cron.py:51-55`, `reminders.py:187-191`) через
`self.db_handle.run_sync_blocking(self._import_..._sync)` (сигнатура
`run_sync_blocking(self, func, *args, **kwargs)`, `db_handle.py:261-264`, выполняется в
вызывающем потоке под `db._op_lock`, синхронно — подходит для sync `initialize()`).
Условие: только если `db_handle is not None` (тесты, вызывающие `initialize(storage_root=...)`
без `db=`, не должны падать — импорт просто не выполняется, как раньше «нет файла»).

```python
def _import_json_jobs_sync(self, db) -> None:
    with db.get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM agent_cron_jobs").fetchone()[0]
    if count > 0:
        return
    if not os.path.exists(self.jobs_file):
        return
    try:
        with open(self.jobs_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        logger.exception("Failed to read agent_cron_jobs.json for import")
        return
    if not isinstance(data, dict):
        return
    rows = []
    for scope, jobs in data.items():
        if not isinstance(jobs, dict):
            continue
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                continue
            status = job.get("status")
            rows.append((
                job_id, scope, job.get("chat_id"), job.get("user_id"),
                job.get("schedule", ""), job.get("prompt", ""),
                job.get("schedule_type", "once"), job.get("next_run_at"),
                job.get("interval_seconds"), job.get("hour"), job.get("minute"),
                job.get("weekday"), "active" if status == "running" else (status or "active"),
                int(bool(job.get("paused"))), job.get("reply_to_message_id"),
                job.get("message_thread_id"), job.get("created_at") or datetime.now().isoformat(timespec="seconds"),
                job.get("last_started_at"), job.get("last_finished_at"),
                job.get("last_error"), job.get("last_tokens"), None, None,
            ))
    with db.get_connection() as conn:
        conn.executemany(
            '''INSERT OR IGNORE INTO agent_cron_jobs (
                id, scope, chat_id, user_id, schedule, prompt, schedule_type, next_run_at,
                interval_seconds, hour, minute, weekday, status, paused, reply_to_message_id,
                message_thread_id, created_at, last_started_at, last_finished_at, last_error,
                last_tokens, locked_at, locked_by
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            rows,
        )
    os.replace(self.jobs_file, self.jobs_file + ".migrated")
```

`status == "running"` из старого файла принудительно нормализуется в `active`,
`locked_at`/`locked_by` — `None` — это прямой перенос старой логики восстановления
`_load_jobs()` (`agent_cron.py:319-323`: любая «running» запись при загрузке файла
считалась зависшей и сбрасывалась в «active», т.к. `_running_tasks` пустой после рестарта).
Переименование — только после успешной записи (import — best-effort: при ошибке чтения
файл остаётся на месте, будет предпринята повторная попытка при следующем `initialize()`,
т.к. таблица всё ещё пуста).

Для reminders — аналогично, `_import_json_reminders_sync`, но с валидацией `time`/
`fire_at_utc` через `datetime.fromisoformat` перед вставкой (записи с непарсящимся `time`
и без `fire_at_utc` — по-прежнему импортируются as-is: старый `check_reminders` тоже не
падал, а логировал skip на каждом тике, значит запись оставалась в файле вечно; в новой
версии она аналогично останется в таблице и будет каждый раз проигнорирована фильтром
`WHERE ... time <= ?` только если `time` лексикографически меньше текущего — если формат
битый и не ISO, сравнение может дать неожиданный результат один раз при импорте; это
существующий edge case из `test_bad_time_does_not_kill_tick`, разбирается в §8/тестах).

## 8. Замена in-memory структур

Убираются полностью: `self.jobs` (`agent_cron.py:43`), `self._load_jobs`/`self._save_jobs`
(`:307-340`); `self.reminders` (`reminders.py:18`), `self.load_reminders`/`self.save_reminders`
(`:193-220`). `self.jobs_file`/`self.reminders_file` остаются как путь к legacy-файлу
для одноразового импорта (переименовываются в путь после переезда). `self._running_tasks`
(`agent_cron.py:45`) и `self._checker_task` (`:44`) остаются без изменений — они про
asyncio-таски текущего процесса (graceful cancel в `close()`), не про хранение задач.

Чистые функции без изменений: `_parse_schedule`, `_unit_seconds`, `_parse_exact_datetime`,
`_parse_iso`, `_daily_schedule`, `_weekly_schedule`, `_advance_job`, `_usage`,
`_format_created`, `WEEKDAYS` — все они уже принимают/возвращают plain dict/str/None, не
знают про хранилище. `_format_jobs` (`agent_cron.py:407-420`) становится `async def`
(читает через `db_handle.fetch_all`), вызов в `handle_cron_command`
(`agent_cron.py:115`) получает `await`.

`delete_reminder`-путь (`reminders.py:397-399`) сейчас логирует
`logging.info(f"Список напоминаний для пользователя {owner_id}: {self.reminders}")` —
дамп ВСЕГО словаря всех пользователей на каждое удаление. Поле `self.reminders` исчезает,
поэтому эта строка технически обязана измениться — заменяется на лог только
результата точечного `SELECT` по `owner_id` (или убирается вовсе, если реализация решит,
что per-request debug-дамп бесполезен — на усмотрение разработчика, т.к. это не
поведенческая часть контракта, а debug-логирование).

## 9. Публичное поведение, которое обязано остаться идентичным

- Специфи инструментов (`get_spec()` обоих плагинов) — не трогаются вообще.
- Тексты ответов `direct_result` (`_format_created`, `_usage`, `reminders_*` через `self.t`)
  — идентичны.
- `create_cron_job`/`set_reminder`/`list_reminders`/`delete_reminder` — те же ключи
  kwargs, тот же owner/scope-based доступ (в группах `chat_id != user_id`,
  `target_chat_id` vs `owner_id`), тот же расчёт `fire_at_utc` из `current_time`.
  `/cron` handlers (`pause`/`resume`/`remove`/`run`/`list`/`add`) — тот же текст ответов,
  тот же scope-based доступ (нельзя тронуть чужую задачу по id).
  `handle_reminder_callback` (inline-кнопки) — тот же UX (view/delete/close).
- Напоминание отправляется **ровно один раз** (успешная отправка = удаление записи под
  локом), задача cron не может стартовать дважды параллельно (атомарный захват).
- Истёкший lease делает задачу/напоминание снова доступными для захвата — новое
  поведение ВМЕСТО старого «сброс running у файла на каждом `_load_jobs()`» (это и есть
  явно запрошенное мастер-планом изменение, не регрессия).

## 10. Тесты

### 10.1 Новый файл `tests/test_agent_cron_storage.py` (или расширение
`test_agent_cron_plugin.py` — решает разработчик; ниже — список сценариев)

Локальная fixture (не трогает `tests/conftest.py`, т.к. он вне владения):
```python
@pytest.fixture()
def cron_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    Database._reset_singleton()
    db = Database()
    with db.get_connection() as conn:
        for stmt in AgentCronPlugin().register_schema():
            conn.execute(stmt)
    yield db
    Database._reset_singleton()
```
(паттерн 1:1 с `agent_db` в `tests/conftest.py:27-34`, но локально в файле этой задачи).

- `test_register_schema_creates_table_and_indexes` — выполнить DDL на in-memory/tmp БД,
  проверить `sqlite_master` содержит таблицу и оба индекса.
- `test_import_json_jobs_moves_file_and_preserves_fields` — создать `agent_cron_jobs.json`
  с 2 задачами разных `schedule_type` (interval, weekly) в tmp_path, `initialize(db=..,
  storage_root=..)`, проверить: строки в таблице совпадают по всем полям с исходным dict,
  файл переименован в `*.migrated`, исходного файла больше нет.
- `test_import_skips_when_table_not_empty` — вставить строку вручную, положить JSON рядом,
  `initialize(...)`, проверить что JSON не тронут (не переименован), в таблице как было.
- `test_import_running_status_normalized_to_active` — JSON с `"status": "running"`, после
  импорта строка имеет `status='active'`, `locked_at IS NULL`.
- `test_claim_due_jobs_locks_and_returns_only_due` — 3 задачи в таблице (due, future,
  paused-due), захват возвращает только первую, её `status='running'`, `locked_at` заполнен.
- **`test_concurrent_claim_only_one_thread_wins`** (обязательный по мастер-плану) — одна
  due-задача в таблице; два `threading.Thread`, каждый вызывает
  `plugin._claim_due_jobs_sync(db, now, lease_cutoff, f"worker-{i}")` напрямую (по образцу
  `tests/test_database.py` `test_concurrent_access_smoke`, `threading.Thread` + `join`);
  собрать результаты в общий список под `threading.Lock`; assert ровно один список
  непустой (получил задачу), второй — пустой; после — в таблице ровно одна строка с
  `locked_by` одного из двух worker-id.
- `test_stale_lease_job_reclaimable` — задача со `status='running'`, `locked_at` = сейчас
  минус (`AGENT_CRON_JOB_LEASE_SECONDS`+10) сек → захват её возвращает.
- `test_fresh_lease_job_not_reclaimable` — тот же сетап, но `locked_at` = сейчас минус 5с →
  захват возвращает пусто.
- `test_manual_run_claims_ignoring_paused` — задача `paused=1`, `_claim_job_by_id_sync`
  всё равно захватывает (сохраняет текущее поведение `/cron run`).
- `test_manual_run_respects_scope_isolation` — задача принадлежит `scope="a"`, захват с
  `scope="b"` возвращает `None`.
- `test_advance_job_persists_next_run_at_after_success` — интеграционный: реальный
  `_run_job` с фейковым helper/bot поверх `cron_db`, после — в таблице обновлён
  `next_run_at` в будущем, `status='active'`, `locked_at IS NULL`.
- Перенести без изменений (чистые функции, БД не нужна):
  `test_agent_cron_parses_supported_natural_schedules`,
  `test_parse_schedule_every_0_minutes_returns_none`,
  `test_advance_job_zero_interval_pauses_job`.
- Переписать на `cron_db`-фикстуру (логика та же, хранилище другое):
  `test_agent_cron_manual_run_delivers_result`, `test_agent_cron_failure_uses_rich_config`,
  `test_agent_cron_default_does_not_dispatch_autonomous_hook`,
  `test_agent_cron_dispatches_autonomous_hook_when_enabled` — заменить прямое создание
  job-dict через `plugin._create_job(...)` на `await plugin._create_job(...)` с
  `plugin.db_handle` из `cron_db`, проверки `plugin.jobs[...]` заменить на
  `await plugin.db_handle.fetch_one("SELECT * FROM agent_cron_jobs WHERE id=?", (job["id"],))`.
- Удалить (стали неприменимы к новой архитектуре, обоснование — не заглушка, а
  структурное исчезновение бага, который они проверяли):
  `test_run_job_updates_live_dict_after_load_replaces_jobs` (проверял гонку вокруг
  `self.jobs`-словаря, которого больше нет — общий per-row UPDATE по id её не воспроизводит),
  `test_save_jobs_does_not_leave_tmp_file` (проверял tmp+replace для JSON-файла, которого
  больше нет для текущего хранения; аналог для файла импорта уже покрыт
  `test_import_json_jobs_moves_file_and_preserves_fields`).

### 10.2 `tests/test_reminders_fixes.py` — обновления

- `_make_plugin(tmp_path)` — добавить `db=` (реальный `DbHandle` поверх временной БД
  с зарегистрированной схемой reminders, тем же локальным fixture-паттерном что и cron).
- Все тесты, где сейчас проверяется `plugin.reminders[...]`, — заменить на
  `await plugin.db_handle.fetch_all("SELECT * FROM reminders WHERE owner_id=?", (owner_id,))`
  и проверку полей строки. Логика тестов (owner-keying, `target_chat_id`, `fire_at_utc`,
  legacy-путь без `fire_at_utc`) не меняется — меняется только способ чтения хранилища.
- `test_check_reminders_fires_by_utc`/`test_check_reminders_does_not_fire_future_utc`/
  `test_check_reminders_legacy_path_no_fire_at_utc` — убрать
  `patch.object(plugin, "save_reminders")`/`patch.object(plugin, "load_reminders")`
  (методов больше нет), вместо прямого присваивания `plugin.reminders = {...}` — INSERT
  строки в таблицу через `db_handle.execute(...)`.
- Новые:
  - `test_import_json_reminders_moves_file_and_preserves_fields`.
  - `test_import_skips_when_table_not_empty`.
  - **`test_concurrent_claim_reminder_only_one_thread_wins`** — тот же паттерн, что для
    cron, на `_claim_due_reminders_sync`.
  - `test_reminder_sent_exactly_once_under_claim` — заявленная в мастер-плане проверка:
    due-напоминание, `check_reminders` дважды подряд (или из двух параллельных вызовов) —
    `helper.send_message` вызван ровно 1 раз, вторая попытка видит пустой claim (запись уже
    удалена после первой).
  - `test_stale_processing_lease_reclaimable` / `test_fresh_processing_lease_not_reclaimable`.
  - `test_give_up_after_max_attempts_deletes_row` — перенос
    `test_poisoned_reminder_removed_after_max_attempts` из
    `tests/test_background_tasks.py` в это владение, на новом хранилище (если владение не
    расширят — см. §0, дублировать здесь достаточно для покрытия задачи T04, старый файл
    останется красным до отдельного фикса).

### 10.3 Тесты вне владения, которые следует ожидать красными без §0

`tests/test_background_tasks.py::test_reminders_tick_*`,
`::test_send_failure_increments_attempts_no_duplicate`,
`::test_poisoned_reminder_removed_after_max_attempts`, `::test_bad_time_does_not_kill_tick`,
`::test_successful_send_persists_and_failed_persists_attempts`,
`::test_save_reminders_no_tmp_file_after_success`;
`tests/test_concurrent_tool_state.py::test_reminder_uses_request_context_message_id_not_shared_helper`,
`::test_concurrent_reminder_calls_keep_reply_message_ids_separate`,
`::test_reminder_uses_explicit_message_id_without_request_context`,
`::test_reminder_list_tool_returns_saved_reminders` — эти 4 зовут `plugin.execute(...)`/
трогают `plugin.reminders` без `db=`. `test_reminder_list_tool_returns_empty_state` —
переживёт (не трогает хранилище). Разработчик обязан прогнать оба файла ПЕРЕД изменением
(зафиксировать какие красные уже сейчас — ожидается 0) и ПОСЛЕ, и явно перечислить эти
регрессии в отчёте разработчика/ревью, а не молчать о них.

## 11. Приёмочные команды

```
~/.venvs/ctb/bin/python -m pytest tests/test_agent_cron_plugin.py tests/test_reminders_fixes.py -q --no-header -p no:cacheprovider
# + новый файл(ы) cron storage, если вынесены отдельно
~/.venvs/ctb/bin/python -m pytest tests/test_plugin_manager.py tests/test_no_hardcoded_plugin_refs.py tests/test_no_private_helper_access.py -q --no-header -p no:cacheprovider
~/.venvs/ctb/bin/python -m ruff check bot/plugins/agent_cron.py bot/plugins/reminders.py tests/test_agent_cron_plugin.py tests/test_reminders_fixes.py
python3 -m mypy bot/plugins/agent_cron.py bot/plugins/reminders.py --python-executable ~/.venvs/ctb/bin/python --ignore-missing-imports
```
Полный прогон всего `tests/` (1650 baseline) — задача координатора в конце волны 1, не
единственной этой задачи (параллельно идут T01-T03 в том же дереве).

## 12. Риски / открытые допущения

1. **§0 — главный риск**: без расширения владения на 2 внешних тестовых файла задача
   технически не может считаться «existing tests green» в буквальном смысле мастер-плана
   для reminders. Нужно решение координатора до реализации.
2. Значения лизов (30 мин для cron, 2 мин для reminders) — инженерная оценка, не из
   мастер-плана; ревьюер может попросить обосновать/изменить.
3. `_op_lock` в `Database.get_connection()` (`database.py:310-318`) сериализует ВСЕ
   синхронные обращения к БД одним `threading.Lock` — конкурентный тест с двумя потоками
   детерминирован не из-за гонки внутри SQLite, а из-за этой сериализации на уровне
   Python; это ожидаемо и не делает тест бесполезным (он проверяет корректность SQL
   условия захвата, не сам факт блокировки).
4. Импорт при первом `initialize()` без `db=` (некоторые тесты вызывают
   `plugin.initialize(storage_root=...)` без `db=`) — импорт просто не запускается;
   такие тесты не видят персистентности между двумя `_make_plugin(tmp_path)` вызовами
   (раньше это работало через файл) — часть тестов в §10.3 именно на этом и падает.
5. `agent_cron`'s `on_startup`/`_checker_loop` (`agent_cron.py:57-59`, `:177-185`) —
   собственный `application.create_task`, НЕ через `get_background_tasks()`
   (`AGENTS.md`, «Background tasks»). Мастер-план не просит миграцию на этот механизм —
   оставляю как есть, чтобы не делать незапрошенный рефакторинг соседнего кода.

## Отклонение от плана (решение координатора, 2026-09-26)

Импорт старого `reminders.json`: записи без корректного `time` и без корректного `fire_at_utc`
не импортируются «как есть», а пропускаются с WARNING в логе. Иначе пустое `time=''` при
сравнении строк считалось бы наступившим, и напоминание сработало бы сразу после обновления.
В HEAD такие записи никогда не срабатывали — пропуск сохраняет это поведение
(найдено ревью W9, группа B, раунд 2).
