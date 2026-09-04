"""Narrow async facade over the sync ``Database`` for plugin hooks.

Each ``DbHandle`` method delegates to ``Database._run_in_db_thread`` so
coroutines never block the event loop. Serialization is provided by the
single-worker executor in ``Database`` (max_workers=1).

The ``transaction()`` context manager opens the underlying
``Database.transaction()`` on that worker thread, so reads inside the scope see
prior writes and exceptions roll the whole unit back.
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import suppress
from typing import Any, Sequence

from ..database import (
    _DB_HANDLE_TRANSACTION_LOCK_BYPASS,
    _DB_HANDLE_TRANSACTION_OPEN,
    _is_transaction_open_on_current_task,
    _TransactionMarker,
)


class TransactionScope:
    """Operations inside a live ``Database.transaction()``."""

    def __init__(self, db: Any) -> None:
        self._db = db
        self._ctx = None
        self._conn = None
        self._closed = False

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self._ensure_open()
        params = tuple(params)

        def _run() -> None:
            self._conn.execute(sql, params)

        await _run_db_thread_shielded(self._db, _run)

    async def executemany(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        self._ensure_open()
        params_list = [tuple(p) for p in params_seq]

        def _run() -> None:
            self._conn.executemany(sql, params_list)

        await _run_db_thread_shielded(self._db, _run)

    async def fetch_one(
        self, sql: str, params: Sequence[Any] = ()
    ) -> dict | None:
        self._ensure_open()
        params = tuple(params)

        def _run() -> dict | None:
            cursor = self._conn.execute(sql, params)
            row = cursor.fetchone()
            if row is None:
                return None
            return _row_to_dict(row, cursor)

        return await _run_db_thread_shielded(self._db, _run)

    async def fetch_all(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[dict]:
        self._ensure_open()
        params = tuple(params)

        def _run() -> list[dict]:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
            return [_row_to_dict(row, cursor) for row in rows]

        return await _run_db_thread_shielded(self._db, _run)

    def _ensure_open(self) -> None:
        if self._closed or self._conn is None:
            raise RuntimeError("DbHandle transaction is not active")


class _Transaction:
    """Async context manager produced by ``DbHandle.transaction()``.

    Opens the real ``Database.transaction()`` on the database worker thread and
    keeps it open until ``__aexit__``. Calls on the returned ``TransactionScope``
    therefore see read-your-writes state and roll back already executed writes
    if the body raises.
    """

    def __init__(self, db: Any, lock: asyncio.Lock) -> None:
        self._db = db
        self._lock = lock
        self._scope: TransactionScope | None = None
        self._marker: _TransactionMarker | None = None
        self._open_token = None

    async def __aenter__(self) -> TransactionScope:
        _raise_if_transaction_open_on_this_task()
        await self._lock.acquire()
        scope = TransactionScope(self._db)

        def _begin() -> None:
            ctx = self._db.transaction()
            scope._ctx = ctx
            scope._conn = ctx.__enter__()

        try:
            await _run_db_thread_shielded(self._db, _begin)
        except BaseException:
            if scope._conn is not None:
                await self._close_scope_after_failed_enter(scope)
            self._release_lock()
            raise
        self._scope = scope
        self._marker = _TransactionMarker(asyncio.current_task())
        self._open_token = _DB_HANDLE_TRANSACTION_OPEN.set(self._marker)
        return scope

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        marker = self._marker
        self._marker = None
        if marker is not None:
            marker.active = False
        open_token = self._open_token
        self._open_token = None
        if open_token is not None:
            try:
                _DB_HANDLE_TRANSACTION_OPEN.reset(open_token)
            except ValueError:
                # __aenter__ and __aexit__ ran in different tasks/contexts (a
                # caller driving the raw async-context-manager protocol by hand
                # instead of `async with`, e.g. some tests) — resetting a token
                # from outside the context that created it always raises.
                # Harmless: marker.active=False above already makes the guard
                # treat the transaction as closed everywhere the marker is
                # reachable, independent of this ContextVar's own reset.
                pass

        scope = self._scope
        self._scope = None
        if scope is None:
            self._lock.release()
            return False

        def _end() -> bool:
            ctx = scope._ctx
            if ctx is None:
                return False
            try:
                return bool(ctx.__exit__(exc_type, exc, tb))
            finally:
                scope._closed = True
                scope._ctx = None
                scope._conn = None

        try:
            return await _run_db_thread_shielded(self._db, _end)
        finally:
            self._release_lock()

    async def _close_scope_after_failed_enter(self, scope: TransactionScope) -> None:
        def _end() -> None:
            ctx = scope._ctx
            if ctx is None:
                return
            try:
                ctx.__exit__(asyncio.CancelledError, None, None)
            finally:
                scope._closed = True
                scope._ctx = None
                scope._conn = None

        with suppress(BaseException):
            await _run_db_thread_shielded(self._db, _end)

    def _release_lock(self) -> None:
        if self._lock.locked():
            self._lock.release()


class DbHandle:
    """Async-safe handle over the project ``Database`` singleton.

    Construct once (per ``PluginManager.set_db``) and reuse. All methods are
    coroutines; there is no sync surface.
    """

    def __init__(self, db: Any) -> None:
        self._db = db
        lock = getattr(db, "_db_handle_transaction_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            setattr(db, "_db_handle_transaction_lock", lock)
        self._transaction_lock = lock

    @property
    def database(self) -> Any:
        """Access the underlying ``Database`` instance (for advanced cases)."""
        return self._db

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        _raise_if_transaction_open_on_this_task()
        params = tuple(params)
        db = self._db

        def _run() -> None:
            with db.get_connection() as conn:
                conn.execute(sql, params)

        async with self._transaction_lock:
            await _run_db_thread_shielded(db, _run)

    async def executemany(
        self, sql: str, params_seq: Sequence[Sequence[Any]]
    ) -> None:
        _raise_if_transaction_open_on_this_task()
        params_list = [tuple(p) for p in params_seq]
        db = self._db

        def _run() -> None:
            with db.get_connection() as conn:
                conn.executemany(sql, params_list)

        async with self._transaction_lock:
            await _run_db_thread_shielded(db, _run)

    async def fetch_one(
        self, sql: str, params: Sequence[Any] = ()
    ) -> dict | None:
        _raise_if_transaction_open_on_this_task()
        params = tuple(params)
        db = self._db

        def _run() -> dict | None:
            with db.get_connection() as conn:
                cursor = conn.execute(sql, params)
                row = cursor.fetchone()
                if row is None:
                    return None
                return _row_to_dict(row, cursor)

        async with self._transaction_lock:
            return await _run_db_thread_shielded(db, _run)

    async def run_sync(self, func, *args, **kwargs):
        """Run a small synchronous DB callback on the dedicated DB worker."""
        _raise_if_transaction_open_on_this_task()

        def _run():
            return func(self._db, *args, **kwargs)

        async with self._transaction_lock:
            return await _run_db_thread_shielded(self._db, _run)

    def run_sync_blocking(self, func, *args, **kwargs):
        """Run a synchronous DB callback for sync initialization paths."""
        with self._db._op_lock:
            return func(self._db, *args, **kwargs)

    async def fetch_all(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[dict]:
        _raise_if_transaction_open_on_this_task()
        params = tuple(params)
        db = self._db

        def _run() -> list[dict]:
            with db.get_connection() as conn:
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                return [_row_to_dict(row, cursor) for row in rows]

        async with self._transaction_lock:
            return await _run_db_thread_shielded(db, _run)

    def transaction(self) -> _Transaction:
        """Return an async context manager that batches writes and flushes atomically.

        Semantics:

        - ``execute``/``executemany``/``fetch_one``/``fetch_all`` run inside
          one live ``Database.transaction()`` on the DB worker thread.
        - Writes executed before an exception in the context body are rolled
          back by ``Database.transaction()``.
        - Other ``DbHandle`` calls wait on this handle-wide transaction lock
          instead of joining the open worker-thread transaction accidentally.

        Do NOT call any other ``db``/``db_handle`` method (``db_handle.execute``,
        ``db.get_user_settings_async``, etc.) from inside this block on the same
        task — only ``tx.execute``/``tx.fetch_one``/``tx.fetch_all`` on the scope
        yielded here. The handle-wide transaction lock above is not reentrant, so
        such a call would otherwise hang forever; instead it now raises a
        ``RuntimeError`` immediately (see ``Database._run_in_db_thread``).
        """
        return _Transaction(self._db, self._transaction_lock)


def _raise_if_transaction_open_on_this_task() -> None:
    """Guard against `DbHandle`-facade calls made from inside the body of a
    `DbHandle.transaction()` opened by the same task.

    `DbHandle.execute`/`executemany`/`fetch_one`/`fetch_all`/`run_sync` and
    `DbHandle.transaction()` itself all wait on the same handle-wide
    `_transaction_lock`, which `_Transaction.__aenter__` already holds for the
    full duration of an open transaction. Since `asyncio.Lock` is not
    reentrant, a second acquire attempt by that same task (e.g. calling plain
    `db_handle.execute(...)` instead of `tx.execute(...)` from inside the
    transaction body) would wait forever on a lock only that same task's own
    `__aexit__` can release — a self-deadlock. Checking the marker here turns
    that hang into an immediate, understandable error instead. Legitimate
    in-transaction calls go through `TransactionScope` (`tx.execute` etc.),
    which does not take this lock and is unaffected. A task merely spawned
    from inside the transaction body (e.g. via `asyncio.create_task`) is not
    "the same task" for this check — see `_is_transaction_open_on_current_task`.
    """
    if _is_transaction_open_on_current_task():
        raise RuntimeError(
            "DbHandle call attempted from inside an open DbHandle.transaction() "
            "body on the same task; use tx.execute/tx.fetch_one/tx.fetch_all "
            "instead of db_handle.execute/fetch_one/fetch_all/run_sync/transaction."
        )


async def _run_db_thread_shielded(db: Any, func):
    """Await a DB-worker callback without leaking locks on cancellation."""
    token = _DB_HANDLE_TRANSACTION_LOCK_BYPASS.set(True)
    try:
        task = asyncio.create_task(db._run_in_db_thread(func))
        try:
            return await asyncio.shield(task)
        except BaseException:
            with suppress(BaseException):
                await task
            raise
    finally:
        _DB_HANDLE_TRANSACTION_LOCK_BYPASS.reset(token)


def _row_to_dict(row: Any, cursor: sqlite3.Cursor) -> dict:
    """Convert a row to a plain ``dict``.

    ``Database.get_connection`` sets ``row_factory = sqlite3.Row`` so ``dict(row)``
    works. We keep a ``cursor.description`` fallback for callers that may pass a
    cursor without the row factory.
    """
    if isinstance(row, sqlite3.Row):
        return dict(row)
    description = cursor.description or ()
    return {col[0]: row[idx] for idx, col in enumerate(description)}
