"""Narrow async facade over the sync ``Database`` for plugin hooks.

Each ``DbHandle`` method delegates to ``Database._run_in_db_thread`` so
coroutines never block the event loop. Serialization is provided by the
single-worker executor in ``Database`` (max_workers=1).

The ``transaction()`` context manager buffers operations and flushes the entire
batch in a single worker-thread call. This keeps the SQLite lock held for the
shortest possible time and avoids holding the GIL across ``await`` points.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, Sequence


class _BufferedOp:
    __slots__ = ("kind", "sql", "params")

    def __init__(self, kind: str, sql: str, params: Any) -> None:
        self.kind = kind
        self.sql = sql
        self.params = params


class TransactionScope:
    """Buffers DB ops; the actual flush happens on ``__aexit__``.

    Only write ops (``execute``/``executemany``) are buffered. Reads
    (``fetch_one``/``fetch_all``) are **not supported** inside the scope — they
    would execute against pre-batch state and not see buffered writes. If you
    need read-your-writes within a transactional unit, call ``DbHandle`` methods
    outside the ``transaction()`` context.
    """

    def __init__(self) -> None:
        self._ops: list[_BufferedOp] = []

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self._ops.append(_BufferedOp("execute", sql, tuple(params)))

    async def executemany(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        # Materialize the iterable now so the caller can reuse generators safely.
        self._ops.append(_BufferedOp("executemany", sql, [tuple(p) for p in params_seq]))


class _Transaction:
    """Async context manager produced by ``DbHandle.transaction()``.

    Buffered batch: operations are queued during the ``async with`` body and
    executed as a single transaction on successful exit (one ``get_connection``
    acquisition, one ``COMMIT``). On exception inside the body, the buffer is
    discarded — operations are never sent to the DB. There is no explicit
    ``ROLLBACK`` because there is no preceding ``BEGIN``; nothing reached SQLite.
    """

    def __init__(self, db: Any) -> None:
        self._db = db
        self._scope: TransactionScope | None = None

    async def __aenter__(self) -> TransactionScope:
        self._scope = TransactionScope()
        return self._scope

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        scope = self._scope
        self._scope = None
        if exc_type is not None:
            # Exception inside the ``async with`` body — discard buffered ops.
            # No DB call was ever made, so there is nothing to roll back.
            return False
        if scope is None or not scope._ops:
            return False

        ops = scope._ops
        db = self._db

        def _flush() -> None:
            # ``get_connection`` already acquires ``_op_lock`` and commits on
            # successful exit, rolls back on exception. That gives us
            # COMMIT/ROLLBACK semantics for the whole batch in one acquisition.
            with db.get_connection() as conn:
                cursor = conn.cursor()
                for op in ops:
                    if op.kind == "execute":
                        cursor.execute(op.sql, op.params)
                    else:  # executemany
                        cursor.executemany(op.sql, op.params)

        await db._run_in_db_thread(_flush)
        return False


class DbHandle:
    """Async-safe handle over the project ``Database`` singleton.

    Construct once (per ``PluginManager.set_db``) and reuse. All methods are
    coroutines; there is no sync surface.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    @property
    def database(self) -> Any:
        """Access the underlying ``Database`` instance (for advanced cases)."""
        return self._db

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        params = tuple(params)
        db = self._db

        def _run() -> None:
            with db.get_connection() as conn:
                conn.execute(sql, params)

        await db._run_in_db_thread(_run)

    async def executemany(
        self, sql: str, params_seq: Sequence[Sequence[Any]]
    ) -> None:
        params_list = [tuple(p) for p in params_seq]
        db = self._db

        def _run() -> None:
            with db.get_connection() as conn:
                conn.executemany(sql, params_list)

        await db._run_in_db_thread(_run)

    async def fetch_one(
        self, sql: str, params: Sequence[Any] = ()
    ) -> dict | None:
        params = tuple(params)
        db = self._db

        def _run() -> dict | None:
            with db.get_connection() as conn:
                cursor = conn.execute(sql, params)
                row = cursor.fetchone()
                if row is None:
                    return None
                return _row_to_dict(row, cursor)

        return await db._run_in_db_thread(_run)

    async def fetch_all(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[dict]:
        params = tuple(params)
        db = self._db

        def _run() -> list[dict]:
            with db.get_connection() as conn:
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                return [_row_to_dict(row, cursor) for row in rows]

        return await db._run_in_db_thread(_run)

    def transaction(self) -> _Transaction:
        """Return an async context manager that batches writes and flushes atomically.

        Semantics:

        - Buffered batch: ``execute``/``executemany`` calls inside the
          ``async with`` body are queued and executed as a single transaction
          on successful exit (one connection acquisition, one ``COMMIT``).
        - On exception inside the context, the buffer is discarded — operations
          are never sent to the DB. There is no explicit ``ROLLBACK`` because
          nothing was ever sent; there is no preceding ``BEGIN``.
        - Reads (``fetch_one``/``fetch_all``) inside the buffered scope are
          **not supported**: the ``TransactionScope`` object only exposes write
          methods. Run reads outside the ``transaction()`` block.
        """
        return _Transaction(self._db)


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
