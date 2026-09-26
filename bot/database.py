import asyncio
import concurrent.futures
import contextvars
import sqlite3
from typing import Dict, Any, Optional, List, Generator, NamedTuple
from contextlib import contextmanager
import json
import sys
import threading
import os
import logging
import math
import hashlib
import uuid
from datetime import datetime
from types import FrameType
import yaml

logger = logging.getLogger(__name__)
_DB_HANDLE_TRANSACTION_LOCK_BYPASS = contextvars.ContextVar(
    "db_handle_transaction_lock_bypass",
    default=False,
)


class _TransactionMarker:
    """Tracks one open ``DbHandle.transaction()``: whether it is still active,
    and which task actually opened it.

    Both fields exist to defeat two different false-positive traps caused by
    ``asyncio.create_task()`` copying the current ``contextvars.Context`` at
    task-creation time:

    - Without ``active``: a task spawned *inside* an open transaction body
      keeps its own copy of the ContextVar entry for as long as it runs —
      including after the owning task's ``__aexit__`` has already returned
      (resetting a ContextVar only affects the context that performed the
      reset, not copies already forked off it). That spawned task would look
      "inside" the transaction forever. ``__aexit__`` flips ``active = False``
      on this shared object in place, so every context-copy observes the
      transaction as closed, regardless of how many task copies exist.
    - Without ``owner_task``: a task spawned *inside* an open transaction body
      (e.g. `asyncio.create_task(db._run_in_db_thread(...))`, a real, tested
      pattern — see `test_direct_database_async_call_waits_outside_dbhandle_transaction`
      in tests/test_db_handle.py) inherits a context-copy where the ContextVar
      is still set to this *active* marker, even though it is a genuinely
      independent task that is supposed to simply wait its turn on the
      handle-wide transaction lock, not be treated as the same logical call
      re-entering a lock it already holds. Comparing against
      ``asyncio.current_task()`` at guard-check time (a live runtime lookup,
      not something propagated via context-copying) distinguishes "the exact
      task that opened this transaction, awaiting inline" from "some other
      task that happens to have inherited a copy of the same ContextVar
      value".
    """

    __slots__ = ("active", "owner_task")

    def __init__(self, owner_task) -> None:
        self.active = True
        self.owner_task = owner_task


_DB_HANDLE_TRANSACTION_OPEN: contextvars.ContextVar = contextvars.ContextVar(
    "db_handle_transaction_open",
    default=None,
)


def _is_transaction_open_on_current_task() -> bool:
    """True iff the task currently executing is the one that opened the
    still-active `DbHandle.transaction()` visible through
    `_DB_HANDLE_TRANSACTION_OPEN` in this context. See `_TransactionMarker`."""
    marker = _DB_HANDLE_TRANSACTION_OPEN.get()
    if marker is None or not marker.active:
        return False
    try:
        current_task = asyncio.current_task()
    except RuntimeError:
        return False
    return current_task is marker.owner_task


# Call-sites, для которых уже выдано предупреждение о синхронном вызове из event loop.
_SYNC_CALL_WARNED_SITES: set[str] = set()
_SYNC_CALL_WARNED_SITES_LOCK = threading.Lock()


class DatabaseLockTimeoutError(RuntimeError):
    """``Database._op_lock`` could not be acquired within the configured timeout.

    Raised instead of blocking the calling thread forever. The main scenario this
    guards against: a `DbHandle.transaction()` opened on the DB-worker thread is
    suspended between two `await`s (control has returned to the event loop), and
    the event-loop thread itself makes a *synchronous* `Database.*` call that
    contends for the same `_op_lock` — see
    docs/remediation_2026-09-04/T08-db-deadlock.md.
    """

SQLITE_JOURNAL_MODES = frozenset({"DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"})

# Слепки prompt_start каждого режима из bot/chat_modes.yml ДО правок T06 (2026-09-25,
# HEAD 08bc457). Нужны, чтобы миграция 3 могла определить mode_key старых сессий, чей
# system-контент совпадает с ТЕКСТОМ ДО правки промптов, а не с текущим YAML.
# sha256(prompt_start.strip()) -> mode_key
LEGACY_PROMPT_FINGERPRINTS = {
    "b2c1c8abd73b2c73e540f0373f48303f1c09370ea0f73bc120fec97ab1eaef08": "assistant",
    "9be4c9b324b6a539d3ff61a4b081ba5e4110c3fedf1d4867e181b5abb6062e98": "text_improver",
    "7933d561247b29bb3010001a6bba9901b0c7b701c163797d666eee00ebf0ee87": "travel_guide",
    "b5d714aed40a91e8d6764d640c95e6ed5ae9ef912d42aaf46b177e2b66a5391a": "content_creator",
    "6d7202e82a2ea741d7f4be75afa5eb23a24bf85f443251a534bdd99ed36c1bd6": "technical_writer",
    "7ee9d354b4d76eb1169705ec95eb6b2d489d1d24923bc5caf8d9449bdd9ba312": "summary_assistant",
    "b6ca32103b07a5eca6b3319387679f3d831fb0f82b49bd16e4b33ed3f38208ab": "primitives",
    "f025f6641ea8c5a20238971c27e0ce66da6cf9fda8100f27d7bf5d7d4573775d": "chief_assistant",
    "065d4af6dba74620b9793a2eaacd0af408600db3f9f653d8638e43821a2594af": "medical_assistant",
    "cbe60fa1973fa9447eb97acf1a0ef269736aa31b50cb0f56d14add122556d0c4": "legal_assistant_ru",
    "3fb6127506423b9afe41c530021d8ce7c2ea263132ab8b9fec9271a1f41d0b47": "code_assistant",
    "c8656f9a7e3552eabfdb76569a657d08fa536943968f70562eb4b9da37267074": "sql_assistant",
    "64f4189c68d50ea0230beb85535a3521c1e880114e53792f0079c0f093d601e4": "artist",
    "f405f6e049ce7228ab0cc2a55e956d63703b5d93d7d1b9e073e836276c097ba9": "english_tutor",
    "9e124562a5fb93c1d01fae5f877f417df0991f6bffc2d2065fbb3c730a732dc7": "psychologist",
    "3c232872069309f3726f10a64a14b76fe88c231b510558f299363fe8013557c4": "movie_expert",
    "db5685b99655ca1eab5f9527f75aa62f37ce81a513a69c22610f57daeb613c0a": "school_tutor",
    "210dd566b31af11655af47fe77a7d30a5e637b5ef70ec759f3570a749b3b553d": "code_interpreter",
    "a7d0a0741577c59037d23411a3e7e51ef15327226ec6428f5b1e693ee2d51d9e": "startup_idea_generator",
    "ac8baf5213845478928347370f322a342b2edd7499996281e8c6ca3eefbd869b": "money_maker",
    "79f72e51feb8707d25d01919084c59d75656d57e28f2d9990261a31f83bd1921": "accountant",
    "8f61cbd329dd4429c00457424e991084a2bc972b0fcbb5caccaad43f1e8276a9": "project_manager",
    "c776c0d7598ed7a35bc43f80f1b3f3f2e8d12e5c5608d393ef132ea6a210ddc3": "meta_writer",
    "7697f97a7fbafaa604efafe8d8a9316b2b920eaa9fd2b0e40a1ced84563096e2": "personal_finance_planner",
    "9367c6a4cccb4cc6ab3b6f8d1f55fca35f03a1fa400eb4ca34edf99cae0ffaca": "skills_agent",
}


def _first_openai_model_from_env() -> str:
    return next(
        (model.strip() for model in os.getenv("OPENAI_MODEL", "").split(",") if model.strip()),
        "",
    )


def _normalize_journal_mode(raw_mode: str) -> str:
    mode = (raw_mode or "").strip().upper()
    if mode not in SQLITE_JOURNAL_MODES:
        logger.warning("Invalid SQLITE_JOURNAL_MODE=%r; falling back to WAL", raw_mode)
        return "WAL"
    return mode


def _sqlite_journal_mode_from_env() -> str:
    return _normalize_journal_mode(os.getenv("SQLITE_JOURNAL_MODE", "WAL"))


def _numeric_env(name, default, cast, *, minimum=None):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    raw_value = raw_value.strip()
    if raw_value == "":
        return default
    try:
        value = cast(raw_value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; falling back to %r", name, raw_value, default)
        return default
    if isinstance(value, float) and not math.isfinite(value):
        logger.warning("Invalid %s=%r; falling back to %r", name, raw_value, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("Invalid %s=%r; falling back to %r", name, raw_value, default)
        return default
    return value


class ConversationContextResult(NamedTuple):
    """Структурированный результат ``Database.get_conversation_context``.

    Оставлен как tuple (не dataclass), чтобы позиционная распаковка на местах
    вызова (``context, parse_mode, temperature, max_tokens_percent, session_id
    = ...``) продолжала работать без изменений.
    """

    context: Optional[Dict[str, Any]]
    parse_mode: str
    temperature: float
    max_tokens_percent: int
    session_id: Optional[str]


class ConversationContextError(RuntimeError):
    """Не удалось загрузить контекст разговора из-за отказа хранилища
    (заблокированная БД, неожиданная ошибка драйвера, либо сама сессия не
    смогла создаться). Не путать с легитимным «загружать ещё нечего» —
    это по-прежнему обычный ConversationContextResult с context=None.
    Вызывающий код не должен в ответ на это исключение создавать новую сессию.
    """


class ConversationContextCorruptError(ConversationContextError):
    """conversation_context.context не парсится как JSON. В отличие от
    заблокированной БД это проблема данных, а не временная — повтор не
    поможет. Отдельный подкласс — чтобы находить именно эти случаи по типу
    исключения для ручного разбора одной строки.
    """


class Database:
    _instance = None
    _lock = threading.Lock()
    _connection_lock = threading.Lock()
    _configured: Dict[str, Any] = {}  # explicit overrides set via configure(), empty by default
    _local: threading.local
    db_path: str
    _op_lock: threading.RLock
    _executor: Optional[concurrent.futures.ThreadPoolExecutor]

    # Текущая целевая версия схемы. Миграция 1 — переход с legacy-таблицы
    # `conversation_context` без session_id на новую схему с сессиями.
    # Миграция 2 — добавление колонки version (монотонный счётчик ревизий).
    # Миграция 3 — чисто дата-миграция (без изменения колонок): backfill
    # mode_key внутрь messages[0] системного сообщения в JSON `context` для
    # сессий, сохранённых до появления mode_key. См. LEGACY_PROMPT_FINGERPRINTS.
    TARGET_SCHEMA_VERSION = 3

    @classmethod
    def configure(
        cls,
        *,
        db_path: Optional[str] = None,
        max_sessions: Optional[int] = None,
        journal_mode: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Explicit config for values otherwise read from env inside this module.

        Call before the first ``Database()`` construction (``bot/__main__.py``
        does this right before ``db = Database()``). Any argument left as
        ``None`` keeps today's env-var fallback for that value untouched —
        this is what the tests that ``monkeypatch.setenv(...)`` and call
        ``_reset_singleton()`` without ever calling ``configure()`` rely on.
        """
        if db_path is not None:
            cls._configured['db_path'] = db_path
        if max_sessions is not None:
            cls._configured['max_sessions'] = int(max_sessions)
        if journal_mode is not None:
            cls._configured['journal_mode'] = _normalize_journal_mode(journal_mode)
        if default_model is not None:
            cls._configured['default_model'] = default_model

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                instance = super(Database, cls).__new__(cls)
                # Используем путь к текущему файлу для создания базы данных
                current_dir = os.path.dirname(os.path.abspath(__file__))
                instance.db_path = (
                    cls._configured.get('db_path')
                    or os.getenv("DB_PATH")
                    or os.path.join(current_dir, 'user_data.db')
                )
                instance._op_lock = threading.RLock()
                instance._executor = None
                instance._shutdown_started = False
                instance._local = threading.local()
                try:
                    instance.init_db()
                except Exception:
                    instance.shutdown()
                    instance._close_db_thread_connection()
                    raise
                cls._instance = instance
            return cls._instance

    def __init__(self):
        """Этот метод может быть вызван несколько раз, поэтому здесь не должно быть инициализации"""
        pass

    @classmethod
    def _reset_singleton(cls) -> None:
        """Reset the cached singleton (intended for tests)."""
        with cls._lock:
            instance = cls._instance
            cls._instance = None
            cls._configured = {}
            if instance is not None:
                instance.shutdown()
                instance._close_db_thread_connection()

    @staticmethod
    def _warn_if_sync_call_from_event_loop() -> None:
        """Логирует WARNING (один раз на каждое место вызова), если синхронный
        `Database.*` вызван из потока event loop.

        Такой вызов блокирует loop и может упереться в `_op_lock`, который держит
        приостановленная `DbHandle.transaction()` (см. `DatabaseLockTimeoutError`).
        Уровень именно WARNING: дефолтный уровень бота — INFO (`bot/__main__.py`),
        и DEBUG-сообщение в прод-логах никогда бы не появилось. Ограничение
        «один раз на call-site» — чтобы горячие пути (`/help`, `/stats`, ...)
        не засоряли лог на каждом запросе.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        frame: FrameType | None = sys._getframe(1)
        while frame is not None and (
            frame.f_code.co_filename == __file__ or frame.f_code.co_filename.endswith("contextlib.py")
        ):
            frame = frame.f_back
        site = f"{frame.f_code.co_filename}:{frame.f_lineno}" if frame is not None else "<unknown>"
        with _SYNC_CALL_WARNED_SITES_LOCK:
            if site in _SYNC_CALL_WARNED_SITES:
                return
            _SYNC_CALL_WARNED_SITES.add(site)
        logger.warning(
            "Sync Database call from the event loop thread at %s; prefer the "
            "*_async method to avoid blocking the loop (logged once per call site).",
            site,
            stack_info=True,
        )

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Контекстный менеджер для потокобезопасного доступа к соединению.

        Реентерабельный: вложенные `with get_connection()` в одном потоке
        переиспользуют одно соединение и коммитят/откатывают только на самом
        внешнем выходе. Это нужно, чтобы `transaction()` мог обернуть несколько
        существующих `with get_connection()`-блоков в одну атомарную единицу.
        """
        self._warn_if_sync_call_from_event_loop()
        if not hasattr(self._local, 'connection'):
            with self._connection_lock:
                timeout = _numeric_env("SQLITE_TIMEOUT", 5.0, float, minimum=0.0)
                self._local.connection = sqlite3.connect(self.db_path, timeout=timeout)
                self._local.connection.row_factory = sqlite3.Row
                self._local.connection.execute("PRAGMA foreign_keys = ON")
                journal_mode = self._configured.get('journal_mode') or _sqlite_journal_mode_from_env()
                try:
                    self._local.connection.execute(f"PRAGMA journal_mode = {journal_mode}")
                except sqlite3.Error:
                    self._local.connection.execute("PRAGMA journal_mode = WAL")
                busy_timeout = _numeric_env("SQLITE_BUSY_TIMEOUT_MS", 5000, int, minimum=0)
                self._local.connection.execute(f"PRAGMA busy_timeout = {busy_timeout}")
        if not hasattr(self._local, 'depth'):
            self._local.depth = 0

        lock_timeout = _numeric_env("DB_OP_LOCK_TIMEOUT_SECONDS", 15.0, float, minimum=0.0)
        if not self._op_lock.acquire(timeout=lock_timeout):
            raise DatabaseLockTimeoutError(
                f"Timed out after {lock_timeout}s waiting for Database._op_lock; "
                "an open DbHandle.transaction() may be holding it while suspended "
                "on the event loop, or a slow query is running concurrently. See "
                "docs/remediation_2026-09-04/T08-db-deadlock.md."
            )
        self._local.depth += 1
        is_outer = self._local.depth == 1
        try:
            yield self._local.connection
        except BaseException:
            if is_outer:
                try:
                    self._local.connection.rollback()
                except sqlite3.Error:
                    pass
            raise
        else:
            if is_outer:
                try:
                    self._local.connection.commit()
                except sqlite3.Error:
                    try:
                        self._local.connection.rollback()
                    except sqlite3.Error:
                        pass
                    raise
        finally:
            self._local.depth -= 1
            self._op_lock.release()

    @contextmanager
    def transaction(self, immediate: bool = True) -> Generator[sqlite3.Connection, None, None]:
        """Атомарная транзакция поверх (возможно нескольких) `get_connection()`.

        На самом внешнем уровне открывает `BEGIN IMMEDIATE`, чтобы захватить
        write-lock сразу и не зависеть от sqlite3 deferred-режима. Вложенные
        вызовы переиспользуют ту же транзакцию (тогда BEGIN не повторяется и
        управляющий commit/rollback делает внешний `transaction()`).
        """
        with self.get_connection() as conn:
            started = False
            if self._local.depth == 1 and not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
                started = True
            try:
                yield conn
            except BaseException:
                if started:
                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        pass
                raise

    def __del__(self):
        """Закрываем соединение текущего треда при сборке объекта."""
        try:
            self.shutdown()
        except Exception:
            pass
        try:
            self._close_db_thread_connection()
        except Exception:
            pass

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Лениво создаёт и возвращает ThreadPoolExecutor с одним воркером.

        max_workers=1 гарантирует ровно одно thread-local соединение воркера
        и сериализацию async-операций БД без дополнительных локов.
        """
        if self._executor is not None:
            return self._executor
        with self._op_lock:
            if self._shutdown_started:
                # После shutdown() новый пул не создаём: иначе «зависшая» фоновая
                # задача, сделавшая async-вызов к БД во время остановки, молча
                # породила бы осиротевший воркер-поток и ещё одно соединение.
                raise RuntimeError("Database is shut down; async DB access is no longer available")
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="db-worker"
                )
        return self._executor

    def _close_db_thread_connection(self) -> None:
        """Закрывает thread-local соединение текущего потока (вызывается из воркера)."""
        local = getattr(self, '_local', None)
        if local is None:
            return
        conn = getattr(local, 'connection', None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            try:
                del local.connection
            except AttributeError:
                pass

    def shutdown(self) -> None:
        """Закрывает воркер-соединение и останавливает executor (best-effort)."""
        try:
            with self._op_lock:
                self._shutdown_started = True
                executor = self._executor
                if executor is None:
                    return
                self._executor = None
            try:
                future = executor.submit(self._close_db_thread_connection)
                future.result(timeout=5)
            except Exception:
                pass
            try:
                executor.shutdown(wait=True)
            except Exception:
                pass
        except Exception:
            pass

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
            if _is_transaction_open_on_current_task():
                raise RuntimeError(
                    "Database call attempted from inside an open "
                    "DbHandle.transaction() body on the same task; use "
                    "tx.execute/tx.fetch_one/tx.fetch_all instead."
                )
            async with lock:
                return await self._run_in_db_thread_unlocked(func, *args, **kwargs)
        return await self._run_in_db_thread_unlocked(func, *args, **kwargs)

    async def _run_in_db_thread_unlocked(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            self._get_executor(),
            lambda: ctx.run(func, *args, **kwargs),
        )

    async def _run_db_method(self, method_name: str, *args, **kwargs):
        return await self._run_in_db_thread(getattr(self, method_name), *args, **kwargs)

    def init_db(self):
        """Инициализация базы данных и создание необходимых таблиц"""
        try:
            logger.info(f'Initializing database at {self.db_path}')
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Версионирование схемы. Раньше миграция запускалась по
                # «сниффингу» наличия колонки session_id; при крэше посередине
                # миграции БД могла остаться в состоянии без conversation_context,
                # но с conversation_context_old — последующий init_db не понимал
                # этого. Теперь храним версию явно и восстанавливаем _old.
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                self._recover_from_failed_migration(cursor)

                # Таблица для пользовательских настроек
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_settings (
                        user_id INTEGER PRIMARY KEY,
                        settings TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chat_settings (
                        chat_id TEXT PRIMARY KEY,
                        settings TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tool_call_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT,
                        chat_id TEXT,
                        user_id TEXT,
                        plugin_name TEXT,
                        function_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        duration_ms INTEGER NOT NULL DEFAULT 0,
                        error TEXT,
                        direct_result INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_tool_call_events_chat_created
                    ON tool_call_events(chat_id, created_at)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_tool_call_events_created_at
                    ON tool_call_events(created_at)
                ''')
                
                # Таблица для контекста разговора с поддержкой сессий
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_context (
                        user_id INTEGER,
                        context TEXT NOT NULL,
                        model TEXT NOT NULL,
                        parse_mode TEXT NOT NULL,
                        temperature FLOAT NOT NULL,
                        max_tokens_percent INTEGER DEFAULT 100,
                        session_id TEXT,
                        session_name TEXT DEFAULT NULL,
                        is_active INTEGER DEFAULT 0,
                        message_count INTEGER DEFAULT 0,
                        version INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, session_id)
                    )
                ''')
                
                # Таблица для хранения информации об изображениях
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        chat_id INTEGER NOT NULL,
                        file_id TEXT NOT NULL,
                        file_id_hash TEXT NOT NULL,
                        file_path TEXT,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES user_settings(user_id)
                    )
                ''')

                # Добавляем индекс для быстрого поиска по хешу
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_file_id_hash ON images(file_id_hash)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_images_user_created
                    ON images(user_id, created_at)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_images_user_chat_created
                    ON images(user_id, chat_id, created_at)
                ''')

                self._apply_schema_migrations(cursor)

                # Индекс для быстрого поиска сессий (создаем после миграции)
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_conversation_context_session 
                    ON conversation_context(user_id, session_id, is_active)
                ''')

                self._deduplicate_active_sessions(cursor)
                cursor.execute('''
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_context_one_active
                    ON conversation_context(user_id)
                    WHERE is_active = 1
                ''')

                conn.commit()
                logger.info('Database initialized successfully')
        except Exception as e:
            logger.error(f'Error initializing database: {e}', exc_info=True)
            raise

    def _schema_version(self, cursor: sqlite3.Cursor) -> int:
        cursor.execute('SELECT COALESCE(MAX(version), 0) FROM schema_version')
        return int(cursor.fetchone()[0])

    def _schema_migrations(self):
        return (
            (1, self._migrate_conversation_context_to_sessions),
            (2, self._migrate_conversation_context_version_column),
            (3, self._migrate_conversation_context_backfill_mode_key),
        )

    def _apply_schema_migrations(self, cursor: sqlite3.Cursor) -> None:
        self._reconcile_schema_version_with_shape(cursor)
        current_version = self._schema_version(cursor)
        for version, migration in self._schema_migrations():
            if current_version >= version:
                continue
            migration(cursor)
            cursor.execute(
                'INSERT OR IGNORE INTO schema_version (version) VALUES (?)',
                (version,),
            )
            current_version = version

    # Миграции 1 и 2 меняют колонки conversation_context — их можно проверить по форме
    # таблицы. Миграция 3+ — чисто дата-миграции (mode_key живёт в JSON, не в колонке),
    # форма таблицы их не отражает. Поэтому reconcile не пытается судить о версиях выше
    # этого потолка — иначе завершённая миграция 3 выглядела бы «убежавшей вперёд формы»
    # и откатывалась бы на каждом старте.
    SHAPE_VERIFIABLE_SCHEMA_VERSION = 2

    def _reconcile_schema_version_with_shape(self, cursor: sqlite3.Cursor) -> None:
        columns = set(self._conversation_context_columns(cursor))
        actual_version = 0
        if 'session_id' in columns:
            actual_version = 1
        if 'version' in columns:
            actual_version = 2
        recorded_version = self._schema_version(cursor)
        if recorded_version <= actual_version:
            return
        if recorded_version > self.SHAPE_VERIFIABLE_SCHEMA_VERSION:
            return
        logger.warning(
            "schema_version=%s is ahead of conversation_context shape=%s; resetting to %s",
            recorded_version,
            sorted(columns),
            actual_version,
        )
        cursor.execute('DELETE FROM schema_version WHERE version > ?', (actual_version,))

    def _conversation_context_columns(self, cursor: sqlite3.Cursor) -> list[str]:
        cursor.execute("PRAGMA table_info(conversation_context)")
        return [row[1] for row in cursor.fetchall()]

    def _migrate_conversation_context_to_sessions(self, cursor: sqlite3.Cursor) -> None:
        columns = self._conversation_context_columns(cursor)
        if 'session_id' in columns:
            logger.info('Migration 1: conversation_context already has session_id')
            return

        logger.warning('Migration 1: adding session support to conversation_context')
        cursor.execute('ALTER TABLE conversation_context RENAME TO conversation_context_old')
        cursor.execute('''
            CREATE TABLE conversation_context (
                user_id INTEGER,
                context TEXT NOT NULL,
                model TEXT NOT NULL,
                parse_mode TEXT NOT NULL,
                temperature FLOAT NOT NULL,
                max_tokens_percent INTEGER DEFAULT 100,
                session_id TEXT,
                session_name TEXT DEFAULT NULL,
                is_active INTEGER DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, session_id)
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_conversation_context_session
            ON conversation_context(user_id, session_id, is_active)
        ''')

        model_expr = "COALESCE(model, ?)" if 'model' in columns else "?"
        default_context = '{"messages": []}'
        cursor.execute(f'''
            INSERT INTO conversation_context
            (user_id, context, model, parse_mode, temperature, max_tokens_percent,
             session_id, session_name, is_active, message_count, created_at, updated_at)
            SELECT
                user_id,
                COALESCE(context, ?),
                {model_expr},
                COALESCE(parse_mode, 'HTML'),
                COALESCE(temperature, 0.8),
                COALESCE(max_tokens_percent, 100),
                hex(randomblob(16)) as session_id,
                'Первоначальная сессия' as session_name,
                1 as is_active,
                0 as message_count,
                COALESCE(created_at, CURRENT_TIMESTAMP),
                COALESCE(updated_at, CURRENT_TIMESTAMP)
            FROM conversation_context_old
        ''', (default_context, self._configured.get('default_model') or _first_openai_model_from_env()))

        cursor.execute('DROP TABLE conversation_context_old')
        logger.info('Migration 1: conversation_context session migration complete')

    def _migrate_conversation_context_version_column(self, cursor: sqlite3.Cursor) -> None:
        columns = self._conversation_context_columns(cursor)
        if 'version' in columns:
            logger.info('Migration 2: conversation_context already has version')
            return
        logger.info('Migration 2: adding version column to conversation_context')
        cursor.execute(
            'ALTER TABLE conversation_context ADD COLUMN version INTEGER NOT NULL DEFAULT 0'
        )

    def _migrate_conversation_context_backfill_mode_key(self, cursor: sqlite3.Cursor) -> None:
        """Миграция 3: чисто данные, без изменения колонок. Для сессий без mode_key
        в system-сообщении находит режим по sha256 старого (до правок T06) prompt_start
        и проставляет mode_key. Идемпотентна: пропускает строки, где mode_key уже есть
        или контент не совпал ни с одним слепком."""
        cursor.execute("SELECT user_id, session_id, context FROM conversation_context")
        rows = cursor.fetchall()
        updated = 0
        for user_id, session_id, context_json in rows:
            try:
                context = json.loads(context_json)
            except (TypeError, ValueError):
                continue
            messages = context.get("messages") if isinstance(context, dict) else None
            if not isinstance(messages, list) or not messages:
                continue
            first = messages[0]
            if not isinstance(first, dict) or first.get("role") != "system":
                continue
            if first.get("mode_key"):
                continue
            content = first.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            fingerprint = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
            mode_key = LEGACY_PROMPT_FINGERPRINTS.get(fingerprint)
            if not mode_key:
                continue
            first["mode_key"] = mode_key
            cursor.execute(
                "UPDATE conversation_context SET context = ?, version = version + 1 "
                "WHERE user_id = ? AND session_id = ?",
                (json.dumps(context, ensure_ascii=False), user_id, session_id),
            )
            updated += 1
        logger.info("Migration 3: backfilled mode_key for %d session(s)", updated)

    def _recover_from_failed_migration(self, cursor: sqlite3.Cursor) -> None:
        """Восстанавливает БД, если предыдущая миграция упала между RENAME и DROP."""
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_context_old'"
        )
        if cursor.fetchone() is None:
            return
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_context'"
        )
        has_new = cursor.fetchone() is not None
        if not has_new:
            logger.warning('Recovering conversation_context from _old after crashed migration')
            cursor.execute('ALTER TABLE conversation_context_old RENAME TO conversation_context')
            cursor.execute('DELETE FROM schema_version WHERE version >= 1')
            return
        # Why: DROP _old делаем только если new — это уже мигрированная схема
        # (имеет session_id). Иначе мы рискуем потерять данные пользователей.
        cursor.execute("PRAGMA table_info(conversation_context)")
        new_columns = {row[1] for row in cursor.fetchall()}
        if 'session_id' not in new_columns:
            logger.error(
                'Refusing to drop conversation_context_old: new table lacks session_id, '
                'manual intervention required.'
            )
            return
        # Compare row counts: if _old has more rows than new, the migration was partial
        # and data is safer in _old — restore it rather than losing rows.
        cursor.execute('SELECT COUNT(*) FROM conversation_context_old')
        old_count = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM conversation_context')
        new_count = cursor.fetchone()[0]
        if old_count > new_count:
            logger.warning(
                'conversation_context_old has more rows (%d) than new table (%d); '
                'migration was partial — dropping incomplete new table and restoring _old',
                old_count, new_count,
            )
            cursor.execute('DROP TABLE conversation_context')
            cursor.execute('ALTER TABLE conversation_context_old RENAME TO conversation_context')
            cursor.execute('DELETE FROM schema_version WHERE version >= 1')
            return
        logger.warning('Dropping stale conversation_context_old leftover')
        cursor.execute('DROP TABLE conversation_context_old')

    def _deduplicate_active_sessions(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute('''
            SELECT user_id
            FROM conversation_context
            WHERE is_active = 1
            GROUP BY user_id
            HAVING COUNT(*) > 1
        ''')
        user_ids = [row[0] for row in cursor.fetchall()]
        for user_id in user_ids:
            cursor.execute('''
                SELECT rowid
                FROM conversation_context
                WHERE user_id = ? AND is_active = 1
                ORDER BY updated_at DESC, created_at DESC, rowid DESC
            ''', (user_id,))
            rows = cursor.fetchall()
            if not rows:
                continue
            keep_rowid = rows[0][0]
            cursor.execute('''
                UPDATE conversation_context
                SET is_active = 0
                WHERE user_id = ? AND is_active = 1 AND rowid != ?
            ''', (user_id, keep_rowid))
    
    def _ensure_user(self, cursor: sqlite3.Cursor, user_id: int) -> None:
        """Гарантирует наличие строки в user_settings для user_id.

        Раньше тот же INSERT OR IGNORE инлайнился в save_image (FK requirement)
        и нигде больше, из-за чего conversation_context/save_user_model могли
        работать без user_settings. Helper унифицирует паттерн.
        """
        cursor.execute(
            'INSERT OR IGNORE INTO user_settings (user_id, settings) VALUES (?, ?)',
            (user_id, '{}'),
        )

    def save_user_settings(self, user_id: int, settings: Dict[str, Any]) -> None:
        """Сохранение пользовательских настроек"""
        try:
            logger.info(f'Saving settings for user_id={user_id}')
            with self.get_connection() as conn:
                cursor = conn.cursor()
                settings_json = json.dumps(settings, ensure_ascii=False)
                cursor.execute('''
                    INSERT INTO user_settings (user_id, settings)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET 
                    settings = excluded.settings,
                    updated_at = CURRENT_TIMESTAMP
                ''', (user_id, settings_json))
        except Exception as e:
            logger.error(f'Error saving user settings: {e}', exc_info=True)
            raise
    
    def get_user_settings(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Получение пользовательских настроек"""
        try:
            logger.info(f'Getting settings for user_id={user_id}')
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT settings FROM user_settings WHERE user_id = ?', (user_id,))
                result = cursor.fetchone()
                if result:
                    logger.info(f'Settings found for user_id={user_id}')
                    return json.loads(result[0])
                logger.info(f'No settings found for user_id={user_id}')
                return None
        except Exception as e:
            logger.error(f'Error getting user settings: {e}', exc_info=True)
            raise

    def save_chat_settings(self, chat_id: int | str, settings: Dict[str, Any]) -> None:
        """Сохранение настроек чата"""
        try:
            logger.info(f'Saving settings for chat_id={chat_id}')
            with self.get_connection() as conn:
                cursor = conn.cursor()
                settings_json = json.dumps(settings, ensure_ascii=False)
                cursor.execute('''
                    INSERT INTO chat_settings (chat_id, settings)
                    VALUES (?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                    settings = excluded.settings,
                    updated_at = CURRENT_TIMESTAMP
                ''', (str(chat_id), settings_json))
        except Exception as e:
            logger.error(f'Error saving chat settings: {e}', exc_info=True)
            raise

    def get_chat_settings(self, chat_id: int | str) -> Optional[Dict[str, Any]]:
        """Получение настроек чата"""
        try:
            logger.info(f'Getting settings for chat_id={chat_id}')
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT settings FROM chat_settings WHERE chat_id = ?', (str(chat_id),))
                result = cursor.fetchone()
                if result:
                    logger.info(f'Settings found for chat_id={chat_id}')
                    return json.loads(result[0])
                logger.info(f'No settings found for chat_id={chat_id}')
                return None
        except Exception as e:
            logger.error(f'Error getting chat settings: {e}', exc_info=True)
            raise

    def record_tool_call_event(
        self,
        *,
        function_name: str,
        plugin_name: str | None = None,
        status: str,
        duration_ms: int = 0,
        error: str | None = None,
        direct_result: bool = False,
        chat_id: int | str | None = None,
        user_id: int | str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Persist one tool-call telemetry event."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO tool_call_events
                    (request_id, chat_id, user_id, plugin_name, function_name, status, duration_ms, error, direct_result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    request_id,
                    str(chat_id) if chat_id is not None else None,
                    str(user_id) if user_id is not None else None,
                    plugin_name,
                    function_name,
                    status,
                    int(duration_ms),
                    error,
                    1 if direct_result else 0,
                ))
        except Exception as e:
            logger.error(f'Error recording tool call event: {e}', exc_info=True)
            raise

    def list_tool_call_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent tool-call telemetry events."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT request_id, chat_id, user_id, plugin_name, function_name,
                           status, duration_ms, error, direct_result, created_at
                    FROM tool_call_events
                    ORDER BY id DESC
                    LIMIT ?
                ''', (int(limit),))
                return [
                    {
                        'request_id': row['request_id'],
                        'chat_id': row['chat_id'],
                        'user_id': row['user_id'],
                        'plugin_name': row['plugin_name'],
                        'function_name': row['function_name'],
                        'status': row['status'],
                        'duration_ms': int(row['duration_ms']),
                        'error': row['error'],
                        'direct_result': bool(row['direct_result']),
                        'created_at': row['created_at'],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f'Error listing tool call events: {e}', exc_info=True)
            raise

    def prune_tool_call_events(self, days: int = 30) -> int:
        """Delete tool-call telemetry older than ``days`` and return rowcount."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM tool_call_events
                    WHERE created_at < datetime('now', '-' || ? || ' days')
                ''', (int(days),))
                return cursor.rowcount
        except Exception as e:
            logger.error(f'Error pruning tool call events: {e}', exc_info=True)
            raise
    
    def get_active_session_id(self, user_id: int) -> Optional[str]:
        """Получение ID активной сессии"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT session_id
                FROM conversation_context
                WHERE user_id = ? AND is_active = 1
                ORDER BY updated_at DESC, created_at DESC
            ''', (user_id,))
            result = cursor.fetchone()
            return result[0] if result else None

    @staticmethod
    def _session_name_source_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(str(item.get('text', '')))
            return ' '.join(part.strip() for part in text_parts if part and part.strip())
        if content is None:
            return ''
        return str(content)

    def save_conversation_context(self, user_id: int, context: Dict[str, Any], parse_mode: str, temperature: float, max_tokens_percent: int = 100, session_id: Optional[str] = None, openai_helper = None) -> str:
        """Сохранение контекста разговора с поддержкой сессий"""
        try:
            context_json = json.dumps(context, ensure_ascii=False)

            if not session_id:
                session_id = self.get_active_session_id(user_id)
            if not session_id:
                logger.info(f"Создаем новую сессию для пользователя {user_id}")
                session_id = self.create_session(user_id, openai_helper=openai_helper)
                if not session_id:
                    raise ValueError(f"Не удалось создать сессию для пользователя {user_id}")

            # Считаем количество пользовательских сообщений в сессии
            message_count = len([msg for msg in context.get('messages', []) if msg.get('role') == 'user'])

            # Атомарная транзакция: UPDATE → (если rowcount=0) deactivate+INSERT →
            # SELECT session_name. На partial index idx_conversation_context_one_active
            # без BEGIN IMMEDIATE возможна гонка с параллельным save из соседней
            # корутины — IntegrityError. transaction() поднимает write-lock сразу.
            with self.transaction() as conn:
                cursor = conn.cursor()
                self._ensure_user(cursor, user_id)

                # transaction() уже держит write-lock (BEGIN IMMEDIATE), поэтому
                # SELECT существования и UPDATE атомарны относительно других
                # писателей — отдельный CAS-retry не нужен. version поднимаем
                # монотонно как счётчик ревизий записи.
                cursor.execute(
                    'SELECT 1 FROM conversation_context WHERE user_id = ? AND session_id = ?',
                    (user_id, session_id),
                )
                row = cursor.fetchone()

                if row is not None:
                    cursor.execute('''
                        UPDATE conversation_context
                        SET context = ?,
                            parse_mode = ?,
                            temperature = ?,
                            max_tokens_percent = ?,
                            message_count = ?,
                            version = version + 1,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND session_id = ?
                    ''', (context_json, parse_mode, temperature, max_tokens_percent,
                          message_count, user_id, session_id))
                else:
                    # Строки нет — обычная вставка с version=0.
                    logger.info(f"Создаем новую запись для сессии {session_id}")
                    model = (
                        openai_helper.config['model'] if openai_helper
                        else (self._configured.get('default_model') or _first_openai_model_from_env())
                    )
                    cursor.execute('''
                        UPDATE conversation_context
                        SET is_active = 0
                        WHERE user_id = ?
                    ''', (user_id,))
                    cursor.execute('''
                        INSERT INTO conversation_context
                        (user_id, session_id, context, model, parse_mode, temperature,
                         max_tokens_percent, is_active, message_count, version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 0)
                    ''', (user_id, session_id, context_json, model, parse_mode, temperature,
                          max_tokens_percent, message_count))

                cursor.execute('''
                    SELECT session_name FROM conversation_context WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))
                result = cursor.fetchone()
                session_name_current = result[0] if result else None

            if session_name_current == "...":
                session_name = self._short_session_name_from_context(context)
                if session_name:
                    self.set_session_name(user_id, session_id, session_name)

            return session_id

        except Exception as e:
            logger.error(f'Ошибка сохранения контекста сессии: {e}', exc_info=True)
            raise

    def _short_session_name_from_context(self, context: Dict[str, Any]) -> str | None:
        user_message = next(
            (msg['content'] for msg in context.get('messages', [])
             if msg.get('role') == 'user'),
            None
        )
        user_message = self._session_name_source_text(user_message)
        if user_message and len(user_message) <= 20:
            return user_message.strip()[:20]
        return None

    def set_session_name(self, user_id: int, session_id: str, session_name: str) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE conversation_context
                SET session_name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND session_id = ?
            ''', (session_name.strip()[:50], user_id, session_id))

    async def save_conversation_context_async(
        self,
        user_id: int,
        context: Dict[str, Any],
        parse_mode: str,
        temperature: float,
        max_tokens_percent: int = 100,
        session_id: Optional[str] = None,
        openai_helper = None,
    ) -> str:
        saved_session_id = await self._run_in_db_thread(
            self.save_conversation_context,
            user_id,
            context,
            parse_mode,
            temperature,
            max_tokens_percent,
            session_id,
            openai_helper,
        )
        await self.ensure_session_name_async(
            user_id,
            saved_session_id,
            context,
            openai_helper=openai_helper,
        )
        return saved_session_id

    async def ensure_session_name_async(
        self,
        user_id: int,
        session_id: str,
        context: Dict[str, Any],
        openai_helper = None,
    ) -> None:
        """Если имя сессии всё ещё «...», ставит короткий fallback из первого
        сообщения. LLM-генерация имени теперь выполняется в OpenAIHelper
        (бывшая семантическая цикличная зависимость Database→OpenAIHelper).
        Параметр openai_helper оставлен для обратной совместимости — игнорируется.
        """
        if not session_id:
            return
        session = await self._run_in_db_thread(self.get_session_details, user_id, session_id)
        if not session or session.get("session_name") != "...":
            return
        user_message = next(
            (msg['content'] for msg in context.get('messages', [])
             if msg.get('role') == 'user'),
            None
        )
        user_message = self._session_name_source_text(user_message)
        if not user_message:
            return
        if len(user_message) <= 20:
            await self._run_in_db_thread(self.set_session_name, user_id, session_id, user_message[:20])
            return
        # Для длинных сообщений имя осталось "..." — OpenAIHelper.\
        # ensure_session_name_with_llm в фоне сгенерирует осмысленное имя
        # и сам вызовет db.set_session_name. БД больше не вызывает LLM.
    
    def get_conversation_context(
        self, user_id: int, session_id: Optional[str] = None, openai_helper = None
    ) -> ConversationContextResult:
        """Получение контекста разговора с поддержкой сессий.

        Бросает ConversationContextError (или её подкласс
        ConversationContextCorruptError для битого JSON) вместо того, чтобы
        подменять отказ чтения тем же результатом, что и «контекста ещё нет».
        Вызывающий код не должен создавать новую сессию в ответ на это
        исключение. Легитимное «загружать нечего» (свежий пользователь,
        либо session_id без единой строки) по-прежнему возвращается как
        обычный ConversationContextResult с context=None — это не ошибка.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Проверяем наличие активной сессии
                cursor.execute('''
                    SELECT session_id FROM conversation_context
                    WHERE user_id = ? AND is_active = 1
                ''', (user_id,))
                result = cursor.fetchone()

                # Если нет активной сессии и не указан session_id, создаем новую
                if not result and not session_id:
                    logger.info(f"Создаем новую сессию для пользователя {user_id}")
                    session_id = self.create_session(user_id, openai_helper=openai_helper)
                    if not session_id:
                        # create_session сама ловит все свои исключения и в этом
                        # случае уже залогировала причину — здесь это не «данных
                        # нет», а замаскированный отказ записи.
                        raise ConversationContextError(
                            f"Не удалось создать сессию для пользователя {user_id}"
                        )
                elif not session_id and result:
                    session_id = result[0]

                # Защитная ветка: практически недостижима — к этой точке
                # session_id уже гарантированно взят из create_session, из
                # активной строки, либо это исходный аргумент вызывающего кода.
                if not session_id:
                    logger.warning(f"Не удалось определить сессию для пользователя {user_id}")
                    return ConversationContextResult(None, 'HTML', 0.8, 100, None)

                cursor.execute('''
                    SELECT context, parse_mode, temperature, max_tokens_percent
                    FROM conversation_context
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))

                result = cursor.fetchone()
                if result:
                    try:
                        context = json.loads(result[0]) if result[0] is not None else {'messages': []}
                    except (TypeError, json.JSONDecodeError) as e:
                        # Ошибка данных, не временная — отдельный подкласс, чтобы
                        # её можно было найти по типу исключения и поправить
                        # строку вручную, а не тихо подменять контекст пустым.
                        raise ConversationContextCorruptError(
                            f"Повреждён JSON контекста user_id={user_id}, session_id={session_id}: {e}"
                        ) from e
                    parse_mode = result[1] if result[1] is not None else 'HTML'
                    temperature = round(result[2], 2) if result[2] is not None else 0.8
                    max_tokens_percent = result[3] if result[3] is not None else 100

                    return ConversationContextResult(
                        context, parse_mode, temperature, max_tokens_percent, session_id
                    )

                logger.info(f"Контекст не найден для сессии {session_id}, возвращаем значения по умолчанию")
                return ConversationContextResult(None, 'HTML', 0.8, 100, None)

        except ConversationContextError:
            raise
        except Exception as e:
            logger.error(
                f'Ошибка получения контекста сессии user_id={user_id} session_id={session_id}: {e}',
                exc_info=True,
            )
            raise
    
    def save_user_model(self, user_id: int, model_name: str) -> None:
        """Сохранение выбранной модели пользователя"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Обновляем модель в активной сессии
                cursor.execute('''
                    UPDATE conversation_context SET model = ? WHERE user_id = ? AND is_active = 1
                ''', (model_name, user_id))
        except Exception as e:
            logger.error(f'Error saving user model: {e}', exc_info=True)
            raise
    
    def save_image(self, user_id: int, chat_id: int, file_id: str, file_path: Optional[str] = None, status: str = 'active') -> int:
        """Сохранение информации об изображении"""
        try:
            # Генерируем хеш для file_id
            hash_object = hashlib.md5(file_id.encode())
            file_id_hash = hash_object.hexdigest()[:8]

            with self.get_connection() as conn:
                cursor = conn.cursor()
                self._ensure_user(cursor, user_id)
                cursor.execute('''
                    INSERT INTO images (user_id, chat_id, file_id, file_id_hash, file_path, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, chat_id, file_id, file_id_hash, file_path, status))
                assert cursor.lastrowid is not None
                return cursor.lastrowid
        except Exception as e:
            logger.error(f'Error saving image: {e}', exc_info=True)
            raise

    def get_user_images(self, user_id: int, chat_id: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение списка изображений пользователя"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if chat_id is not None:
                    cursor.execute('''
                        SELECT id, file_id, file_id_hash, file_path, status, created_at, updated_at
                        FROM images 
                        WHERE user_id = ? AND chat_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    ''', (user_id, chat_id, limit))
                else:
                    cursor.execute('''
                        SELECT id, file_id, file_id_hash, file_path, status, created_at, updated_at
                        FROM images 
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    ''', (user_id, limit))
                
                rows = cursor.fetchall()
                return [
                    {
                        'id': row[0],
                        'file_id': row[1],
                        'file_id_hash': row[2],
                        'file_path': row[3],
                        'status': row[4],
                        'created_at': row[5],
                        'updated_at': row[6]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f'Error getting user images: {e}', exc_info=True)
            raise

    def update_image_status(self, image_id: int, status: str) -> None:
        """Обновление статуса изображения"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE images 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status, image_id))
        except Exception as e:
            logger.error(f'Error updating image status: {e}', exc_info=True)
            raise

    def prune_old_images(self, days: int = 7) -> int:
        """Delete image records older than ``days`` and return rowcount."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM images 
                    WHERE created_at < datetime('now', '-' || ? || ' days')
                ''', (int(days),))
                return cursor.rowcount
        except Exception as e:
            logger.error(f'Error cleaning up old images: {e}', exc_info=True)
            raise

    def cleanup_old_images(self, days: int = 7) -> int:
        """Очистка устаревших данных об изображениях"""
        return self.prune_old_images(days)

    def count_user_sessions(self, user_id: int) -> int:
        """
        Подсчет количества сессий пользователя
        
        :param user_id: Идентификатор пользователя
        :return: Количество активных сессий
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM conversation_context 
                    WHERE user_id = ?
                """, (user_id,))
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Ошибка при подсчете сессий пользователя: {e}")
            return 0

    def get_oldest_session_ids_for_limit(
        self,
        user_id: int,
        max_sessions: Optional[int] = None,
        exclude_session_ids: Optional[List[str]] = None,
    ) -> List[str]:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                return self._oldest_session_ids_for_limit(
                    cursor,
                    user_id,
                    max_sessions=max_sessions,
                    exclude_session_ids=exclude_session_ids,
                )
        except sqlite3.Error as e:
            logger.error(f"Ошибка при получении старых сессий: {e}")
            return []

    def _oldest_session_ids_for_limit(
        self,
        cursor: sqlite3.Cursor,
        user_id: int,
        max_sessions: Optional[int] = None,
        exclude_session_ids: Optional[List[str]] = None,
    ) -> List[str]:
        max_sessions = self._coerce_max_sessions_limit(max_sessions)

        cursor.execute("SELECT COUNT(*) FROM conversation_context WHERE user_id = ?", (user_id,))
        total = cursor.fetchone()[0]
        if total < max_sessions:
            return []

        to_delete = total - (max_sessions - 1)
        if to_delete <= 0:
            return []

        excluded = [session_id for session_id in (exclude_session_ids or []) if session_id]
        exclude_clause = ""
        params: List[Any] = [user_id]
        if excluded:
            placeholders = ",".join("?" for _ in excluded)
            exclude_clause = f" AND session_id NOT IN ({placeholders})"
            params.extend(excluded)
        params.append(to_delete)

        cursor.execute(f"""
            SELECT session_id
            FROM conversation_context
            WHERE user_id = ?{exclude_clause}
            ORDER BY created_at ASC
            LIMIT ?
        """, params)
        return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def _coerce_max_sessions_limit(max_sessions: Optional[int] = None) -> int:
        if max_sessions is None:
            max_sessions = Database._configured.get('max_sessions')
        raw_value = os.getenv('MAX_SESSIONS', 5) if max_sessions is None else max_sessions
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            logger.warning("Invalid MAX_SESSIONS=%r; falling back to 5", raw_value)
            value = 5
        return max(1, value)

    def delete_sessions_by_ids(self, user_id: int, session_ids: List[str]) -> bool:
        if not session_ids:
            return True
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                placeholders = ",".join("?" for _ in session_ids)
                cursor.execute(
                    f"DELETE FROM conversation_context WHERE user_id = ? AND session_id IN ({placeholders})",
                    (user_id, *session_ids)
                )
                return cursor.rowcount == len(session_ids)
        except sqlite3.Error as e:
            logger.error(f"Ошибка при удалении сессий: {e}")
            return False

    def get_mode_from_context(self, context: Dict[str, Any]) -> Optional[Dict]:
        """
        Получает системное сообщение из контекста сессии
        
        :param context: Контекст сессии
        :return: Системное сообщение или None
        """
        try:
            if isinstance(context, dict) and 'messages' in context:
                system_messages = [
                    msg for msg in context.get('messages', []) 
                    if msg.get('role') == 'system'
                ]
                                
                if system_messages:
                    return system_messages[0]
                
                return None
            
            logger.warning(f"Некорректный формат контекста: {type(context)}")
            return None
        
        except Exception as e:
            logger.error(f'Ошибка получения системного сообщения из контекста: {e}', exc_info=True)
            return None

    def create_session(
        self,
        user_id: int,
        session_name: Optional[str] = None,
        max_sessions: Optional[int] = None,
        first_message: Optional[str] = None,
        openai_helper = None,
        prune_old_sessions: bool = True,
    ) -> Optional[str]:
        """
        Создание новой сессии с сохранением режима из активной сессии
        
        :param user_id: Идентификатор пользователя
        :param session_name: Название сессии (опционально)
        :param max_sessions: Максимальное количество активных сессий
        :param first_message: Первое сообщение для генерации названия
        :param openai_helper: Экземпляр OpenAIHelper для генерации названия
        :return: Идентификатор новой сессии или None
        """
        try:
            parse_mode = 'HTML'
            temperature = 0.8
            max_tokens_percent = 100
            system_message = None
            new_session_id = str(uuid.uuid4())
            model = (
                openai_helper.config['model'] if openai_helper
                else (self._configured.get('default_model') or _first_openai_model_from_env())
            )
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            with self.transaction() as conn:
                cursor = conn.cursor()
                self._ensure_user(cursor, user_id)
                cursor.execute('''
                    SELECT context, parse_mode, temperature, max_tokens_percent
                    FROM conversation_context
                    WHERE user_id = ? AND is_active = 1
                    ORDER BY updated_at DESC, created_at DESC, rowid DESC
                    LIMIT 1
                ''', (user_id,))
                active_row = cursor.fetchone()
                if active_row:
                    parse_mode = active_row['parse_mode']
                    temperature = active_row['temperature']
                    max_tokens_percent = active_row['max_tokens_percent']
                    try:
                        active_context = json.loads(active_row['context'])
                    except (TypeError, json.JSONDecodeError):
                        active_context = {}
                    system_message = self.get_mode_from_context(active_context)

                if prune_old_sessions:
                    # Plugin subscribers receive ``on_session_before_delete`` from
                    # the bot before ``create_session`` is invoked; here we just prune.
                    for old_session_id in self._oldest_session_ids_for_limit(
                        cursor,
                        user_id,
                        max_sessions=max_sessions,
                    ):
                        cursor.execute(
                            "DELETE FROM conversation_context WHERE user_id = ? AND session_id = ?",
                            (user_id, old_session_id),
                        )

                cursor.execute("""
                    UPDATE conversation_context
                    SET is_active = 0
                    WHERE user_id = ?
                """, (user_id,))

                context = {
                    'messages': [system_message] if system_message else [],
                }
                cursor.execute("""
                    INSERT INTO conversation_context
                    (user_id, context, parse_mode, temperature, max_tokens_percent,
                     session_id, session_name, created_at, is_active, message_count, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?)
                """, (
                    user_id,
                    json.dumps(context, ensure_ascii=False),
                    parse_mode,
                    temperature,
                    max_tokens_percent,
                    new_session_id,
                    session_name or "...",
                    now,
                    model,
                ))

            logger.info(f"Создана сессия {new_session_id} для пользователя {user_id}")
            return new_session_id
                
        except Exception as e:
            logger.error(f"Ошибка при создании сессии: {e}", exc_info=True)
            return None

    def list_user_sessions(self, user_id: int, is_active: int = 0) -> List[Dict[str, Any]]:
        """Получение списка сессий пользователя"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        session_id, 
                        session_name, 
                        is_active, 
                        message_count, 
                        updated_at,
                        context,
                        parse_mode,
                        temperature,
                        max_tokens_percent,
                        model
                    FROM conversation_context 
                    WHERE user_id = ? AND session_id IS NOT NULL AND is_active = case when ? = 1 then 1 else is_active end
                    ORDER BY updated_at DESC
                ''', (user_id, is_active))
                sessions = cursor.fetchall()
                return [
                    {
                        'session_id': session[0],
                        'session_name': session[1],
                        'is_active': bool(session[2]),
                        'message_count': session[3],
                        'updated_at': session[4],
                        'context': json.loads(session[5]) if session[5] else {},
                        'parse_mode': session[6],
                        'temperature': session[7],
                        'max_tokens_percent': session[8],
                        'model': session[9]
                    } for session in sessions
                ]
        except Exception as e:
            logger.error(f'Ошибка получения списка сессий: {e}', exc_info=True)
            return []

    async def list_user_sessions_async(self, user_id: int, is_active: int = 0) -> List[Dict[str, Any]]:
        """Async-обёртка над list_user_sessions для вызова из event loop."""
        return await self._run_in_db_thread(self.list_user_sessions, user_id, is_active)

    async def save_user_settings_async(self, user_id: int, settings: Dict[str, Any]) -> None:
        return await self._run_db_method("save_user_settings", user_id, settings)

    async def get_user_settings_async(self, user_id: int) -> Optional[Dict[str, Any]]:
        return await self._run_db_method("get_user_settings", user_id)

    async def save_chat_settings_async(self, chat_id: int | str, settings: Dict[str, Any]) -> None:
        return await self._run_db_method("save_chat_settings", chat_id, settings)

    async def get_chat_settings_async(self, chat_id: int | str) -> Optional[Dict[str, Any]]:
        return await self._run_db_method("get_chat_settings", chat_id)

    async def record_tool_call_event_async(self, **kwargs) -> None:
        return await self._run_db_method("record_tool_call_event", **kwargs)

    async def list_tool_call_events_async(self, limit: int = 50) -> List[Dict[str, Any]]:
        return await self._run_db_method("list_tool_call_events", limit)

    async def prune_tool_call_events_async(self, days: int = 30) -> int:
        return await self._run_db_method("prune_tool_call_events", days)

    async def get_active_session_id_async(self, user_id: int) -> Optional[str]:
        return await self._run_db_method("get_active_session_id", user_id)

    async def set_session_name_async(self, user_id: int, session_id: str, session_name: str) -> None:
        return await self._run_db_method("set_session_name", user_id, session_id, session_name)

    async def get_conversation_context_async(
        self,
        user_id: int,
        session_id: Optional[str] = None,
        openai_helper = None,
    ) -> ConversationContextResult:
        """См. ``get_conversation_context`` — исключения (включая
        ConversationContextError/ConversationContextCorruptError) пробрасываются
        через ``_run_in_db_thread``/``run_in_executor`` без изменений."""
        return await self._run_db_method(
            "get_conversation_context",
            user_id,
            session_id,
            openai_helper,
        )

    async def save_user_model_async(self, user_id: int, model_name: str) -> None:
        return await self._run_db_method("save_user_model", user_id, model_name)

    async def save_image_async(
        self,
        user_id: int,
        chat_id: int,
        file_id: str,
        file_path: Optional[str] = None,
        status: str = 'active',
    ) -> int:
        return await self._run_db_method(
            "save_image",
            user_id,
            chat_id,
            file_id,
            file_path,
            status,
        )

    async def get_user_images_async(
        self,
        user_id: int,
        chat_id: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return await self._run_db_method("get_user_images", user_id, chat_id, limit)

    async def update_image_status_async(self, image_id: int, status: str) -> None:
        return await self._run_db_method("update_image_status", image_id, status)

    async def prune_old_images_async(self, days: int = 7) -> int:
        return await self._run_db_method("prune_old_images", days)

    async def cleanup_old_images_async(self, days: int = 7) -> int:
        return await self._run_db_method("cleanup_old_images", days)

    async def count_user_sessions_async(self, user_id: int) -> int:
        return await self._run_db_method("count_user_sessions", user_id)

    async def get_oldest_session_ids_for_limit_async(
        self,
        user_id: int,
        max_sessions: Optional[int] = None,
        exclude_session_ids: Optional[List[str]] = None,
    ) -> List[str]:
        return await self._run_db_method(
            "get_oldest_session_ids_for_limit",
            user_id,
            max_sessions,
            exclude_session_ids,
        )

    async def delete_sessions_by_ids_async(self, user_id: int, session_ids: List[str]) -> bool:
        return await self._run_db_method("delete_sessions_by_ids", user_id, session_ids)

    async def create_session_async(
        self,
        user_id: int,
        session_name: Optional[str] = None,
        max_sessions: Optional[int] = None,
        first_message: Optional[str] = None,
        openai_helper = None,
        prune_old_sessions: bool = True,
    ) -> Optional[str]:
        return await self._run_db_method(
            "create_session",
            user_id,
            session_name,
            max_sessions,
            first_message,
            openai_helper,
            prune_old_sessions,
        )

    async def switch_active_session_async(self, user_id: int, session_id: str) -> bool:
        return await self._run_db_method("switch_active_session", user_id, session_id)

    async def delete_session_async(self, user_id: int, session_id: str, openai_helper=None):
        return await self._run_db_method("delete_session", user_id, session_id, openai_helper)

    async def get_session_details_async(
        self,
        user_id: int,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        return await self._run_db_method("get_session_details", user_id, session_id)

    async def export_sessions_to_yaml_async(self, user_id: int) -> Optional[str]:
        return await self._run_db_method("export_sessions_to_yaml", user_id)

    def switch_active_session(self, user_id: int, session_id: str) -> bool:
        """Переключение активной сессии"""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 1
                    FROM conversation_context
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))
                if cursor.fetchone() is None:
                    return False

                # Деактивируем все сессии пользователя
                cursor.execute('''
                    UPDATE conversation_context 
                    SET is_active = 0 
                    WHERE user_id = ?
                ''', (user_id,))
                
                # Активируем выбранную сессию
                cursor.execute('''
                    UPDATE conversation_context 
                    SET is_active = 1 
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f'Ошибка переключения сессии: {e}', exc_info=True)
            raise

    def delete_session(self, user_id: int, session_id: str, openai_helper=None):
        """Удаление сессии"""
        try:
            # Всё удаление — одна транзакция. Внутренние create_session/
            # switch_active_session/list_user_sessions переиспользуют её через
            # reentrant get_connection(); это закрывает гонку, при которой
            # параллельная корутина видит промежуточное «активной сессии нет».
            with self.transaction() as conn:
                session_count = self.count_user_sessions(user_id)

                if session_count == 1:
                    new_session_id = self.create_session(user_id, openai_helper=openai_helper)
                    if not new_session_id:
                        raise RuntimeError(
                            f"Не удалось создать новую сессию для пользователя {user_id}"
                        )

                sessions = self.list_user_sessions(user_id, 1)
                active_session = next((s for s in sessions if s['is_active']), None)

                if active_session and active_session.get('session_id') == session_id:
                    new_session_id = self.create_session(
                        user_id,
                        openai_helper=openai_helper,
                        prune_old_sessions=False,
                    )
                    if not new_session_id:
                        raise RuntimeError(
                            f"Не удалось создать новую сессию для пользователя {user_id}"
                        )
                    self.switch_active_session(user_id, new_session_id)

                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM conversation_context
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))

        except Exception as e:
            logger.error(f'Ошибка удаления сессии: {e}', exc_info=True)
            raise

    def migrate_conversation_context(self):
        """Атомарная миграция conversation_context на схему с session_id.

        Раньше: ALTER+CREATE+INSERT+DROP делались внутри `with get_connection`
        самой `init_db`, что коммитило промежуточные DDL вместе с другими
        CREATE'ами init_db (ALTER RENAME уже виден другим коннекшнам). При
        крэше между RENAME и DROP БД оставалась без основной таблицы и теряла
        все сессии. Сейчас вся миграция — отдельный `transaction()` с BEGIN
        IMMEDIATE и финальным INSERT в schema_version. Восстановление _old
        выполняется в _recover_from_failed_migration на старте.
        """
        try:
            logger.info('Начало миграции conversation_context')
            with self.transaction() as conn:
                cursor = conn.cursor()
                if self._schema_version(cursor) >= 1:
                    return
                self._migrate_conversation_context_to_sessions(cursor)
                cursor.execute(
                    'INSERT OR IGNORE INTO schema_version (version) VALUES (1)',
                )
                logger.info('Миграция conversation_context завершена успешно')
        except Exception as e:
            logger.error(f'Ошибка миграции базы данных: {e}', exc_info=True)
            raise

    def get_session_details(self, user_id: int, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает детали конкретной сессии.
        
        :param user_id: ID пользователя
        :param session_id: ID сессии
        :return: Словарь с деталями сессии или None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        session_id, 
                        session_name, 
                        is_active, 
                        message_count, 
                        created_at, 
                        context
                    FROM conversation_context 
                    WHERE user_id = ? AND session_id = ?
                ''', (user_id, session_id))
                result = cursor.fetchone()
                
                if result:
                    session_details = {
                        'session_id': result[0],
                        'session_name': result[1],
                        'is_active': bool(result[2]),
                        'message_count': result[3],
                        'created_at': result[4],
                        'context': json.loads(result[5]) if result[5] else {}
                    }
                    return session_details
                return None
        except Exception as e:
            logger.error(f'Error getting session details: {e}', exc_info=True)
            return None

    def export_sessions_to_yaml(self, user_id: int) -> Optional[str]:
        """
        Экспорт всех сессий пользователя в YAML-файл

        :param user_id: Идентификатор пользователя
        :return: Путь к сгенерированному YAML-файлу
        """
        try:
            # Получаем список всех сессий пользователя
            sessions = self.list_user_sessions(user_id)

            # Подготавливаем данные для экспорта
            export_data: Dict[str, Any] = {
                'user_id': user_id,
                'total_sessions': len(sessions),
                'sessions': []
            }
            
            for session in sessions:
                session_export = {
                    'session_id': session['session_id'],
                    'session_name': session['session_name'],
                    'is_active': session['is_active'],
                    'message_count': session['message_count'],
                    'updated_at': str(session['updated_at']),
                    'parse_mode': session['parse_mode'],
                    'temperature': session['temperature'],
                    'max_tokens_percent': session['max_tokens_percent'],
                    'context': session['context']
                }
                export_data['sessions'].append(session_export)
            
            # Создаем директорию для экспорта, если она не существует
            export_dir = os.path.join(os.getcwd(), 'exports')
            os.makedirs(export_dir, exist_ok=True)
            
            # Генерируем имя файла с отметкой времени
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'sessions_export_{user_id}_{timestamp}.yaml'
            filepath = os.path.join(export_dir, filename)
            
            # Сохраняем в YAML
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.safe_dump(export_data, f, allow_unicode=True, default_flow_style=False)
            
            logger.info(f"Экспортировано сессий: {len(sessions)} в файл {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f'Ошибка экспорта сессий в YAML: {e}', exc_info=True)
            return None 
