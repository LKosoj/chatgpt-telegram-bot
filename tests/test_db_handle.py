import asyncio

import pytest

from bot.database import Database
from bot.plugins.db_handle import DbHandle


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = tmp_path / "handle.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._reset_singleton()
    instance = Database()
    yield instance
    Database._reset_singleton()


@pytest.fixture()
def handle(db):
    return DbHandle(db)


class _FakeTransaction:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        self.db.open_transactions += 1
        return object()

    def __exit__(self, exc_type, exc, tb):
        self.db.open_transactions -= 1
        self.db.close_calls += 1
        return False


class _BlockingDb:
    def __init__(self, *, block_call: int):
        self._db_handle_transaction_lock = asyncio.Lock()
        self.block_call = block_call
        self.call_count = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.open_transactions = 0
        self.close_calls = 0

    async def _run_in_db_thread(self, func):
        self.call_count += 1
        if self.call_count == self.block_call:
            self.started.set()
            await self.release.wait()
        return func()

    def transaction(self):
        return _FakeTransaction(self)


async def _create_kv_table(handle: DbHandle) -> None:
    await handle.execute(
        "CREATE TABLE IF NOT EXISTS kv ("
        "id INTEGER PRIMARY KEY, name TEXT NOT NULL, value INTEGER NOT NULL)"
    )


async def test_execute_then_fetch_all(handle: DbHandle):
    await _create_kv_table(handle)
    await handle.execute(
        "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (1, "alpha", 10)
    )
    rows = await handle.fetch_all("SELECT id, name, value FROM kv ORDER BY id")
    assert rows == [{"id": 1, "name": "alpha", "value": 10}]


async def test_fetch_one_returns_none_when_empty(handle: DbHandle):
    await _create_kv_table(handle)
    row = await handle.fetch_one("SELECT * FROM kv WHERE id = ?", (404,))
    assert row is None


async def test_executemany_bulk_insert(handle: DbHandle):
    await _create_kv_table(handle)
    rows = [(i, f"name-{i}", i * 2) for i in range(100)]
    await handle.executemany(
        "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", rows
    )
    count = await handle.fetch_one("SELECT COUNT(*) AS c FROM kv")
    assert count == {"c": 100}


async def test_transaction_commits_all_on_success(handle: DbHandle):
    await _create_kv_table(handle)
    async with handle.transaction() as tx:
        await tx.execute(
            "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (1, "a", 1)
        )
        await tx.execute(
            "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (2, "b", 2)
        )
        await tx.executemany(
            "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)",
            [(3, "c", 3), (4, "d", 4)],
        )
    rows = await handle.fetch_all("SELECT id FROM kv ORDER BY id")
    assert [r["id"] for r in rows] == [1, 2, 3, 4]


async def test_transaction_rolls_back_on_exception(handle: DbHandle):
    await _create_kv_table(handle)
    await handle.execute(
        "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (1, "preexisting", 0)
    )
    with pytest.raises(RuntimeError, match="boom"):
        async with handle.transaction() as tx:
            await tx.execute(
                "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (2, "x", 1)
            )
            await tx.execute(
                "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (3, "y", 2)
            )
            raise RuntimeError("boom")
    rows = await handle.fetch_all("SELECT id FROM kv ORDER BY id")
    # Only the pre-existing row should remain; the live transaction rolled back.
    assert [r["id"] for r in rows] == [1]


async def test_transaction_reads_its_own_writes(handle: DbHandle):
    await _create_kv_table(handle)

    async with handle.transaction() as tx:
        await tx.execute(
            "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (1, "inside", 10)
        )
        row = await tx.fetch_one("SELECT name, value FROM kv WHERE id = ?", (1,))

    assert row == {"name": "inside", "value": 10}


async def test_transaction_rolls_back_after_successful_inner_fetch(handle: DbHandle):
    await _create_kv_table(handle)

    with pytest.raises(RuntimeError, match="after fetch"):
        async with handle.transaction() as tx:
            await tx.execute(
                "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (1, "inside", 10)
            )
            assert await tx.fetch_one("SELECT id FROM kv WHERE id = ?", (1,)) == {"id": 1}
            raise RuntimeError("after fetch")

    assert await handle.fetch_all("SELECT * FROM kv") == []


async def test_transaction_cancelled_error_rolls_back_real_database(handle: DbHandle):
    await _create_kv_table(handle)

    with pytest.raises(asyncio.CancelledError):
        async with handle.transaction() as tx:
            await tx.execute(
                "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)",
                (1, "cancelled", 10),
            )
            raise asyncio.CancelledError()

    await handle.execute(
        "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (2, "later", 20)
    )
    rows = await handle.fetch_all("SELECT id, name FROM kv ORDER BY id")
    assert rows == [{"id": 2, "name": "later"}]


async def test_direct_database_async_call_waits_outside_dbhandle_transaction(db, handle):
    await _create_kv_table(handle)
    external_started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def external_insert():
        loop.call_soon_threadsafe(external_started.set)
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)",
                (2, "external", 20),
            )

    with pytest.raises(RuntimeError, match="rollback plugin"):
        async with handle.transaction() as tx:
            await tx.execute(
                "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)",
                (1, "plugin", 10),
            )
            external_task = asyncio.create_task(db._run_in_db_thread(external_insert))
            await asyncio.sleep(0)
            assert not external_task.done()
            assert not external_started.is_set()
            raise RuntimeError("rollback plugin")

    await external_task
    rows = await handle.fetch_all("SELECT id, name FROM kv ORDER BY id")
    assert rows == [{"id": 2, "name": "external"}]


async def test_transaction_enter_cancellation_closes_opened_db_transaction():
    fake_db = _BlockingDb(block_call=1)
    handle = DbHandle(fake_db)
    cm = handle.transaction()
    task = asyncio.create_task(cm.__aenter__())

    await fake_db.started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    fake_db.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_db.close_calls == 1
    assert fake_db.open_transactions == 0
    assert not handle._transaction_lock.locked()


async def test_transaction_exit_cancellation_waits_for_db_close():
    fake_db = _BlockingDb(block_call=2)
    handle = DbHandle(fake_db)
    cm = handle.transaction()
    await cm.__aenter__()
    task = asyncio.create_task(cm.__aexit__(None, None, None))

    await fake_db.started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    fake_db.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_db.close_calls == 1
    assert fake_db.open_transactions == 0
    assert not handle._transaction_lock.locked()


async def test_concurrent_execute_serializes(handle: DbHandle):
    await _create_kv_table(handle)

    async def insert(i: int) -> None:
        await handle.execute(
            "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)",
            (i, f"n-{i}", i),
        )

    await asyncio.gather(*(insert(i) for i in range(50)))
    rows = await handle.fetch_all("SELECT id FROM kv ORDER BY id")
    assert [r["id"] for r in rows] == list(range(50))


async def test_fetch_returns_plain_dict(handle: DbHandle):
    await _create_kv_table(handle)
    await handle.execute(
        "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (1, "z", 7)
    )
    one = await handle.fetch_one("SELECT * FROM kv WHERE id = ?", (1,))
    all_rows = await handle.fetch_all("SELECT * FROM kv")
    assert isinstance(one, dict) and type(one) is dict
    assert isinstance(all_rows[0], dict) and type(all_rows[0]) is dict


async def test_handle_uses_provided_database_instance(db, handle):
    assert handle.database is db
    # Confirm a second handle wrapping the same Database singleton sees writes
    # made through the first.
    await _create_kv_table(handle)
    await handle.execute(
        "INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (1, "shared", 1)
    )
    other = DbHandle(db)
    row = await other.fetch_one("SELECT name FROM kv WHERE id = 1")
    assert row == {"name": "shared"}


# ---------------------------------------------------------------------------
# Новые тесты: WP-A — проверяем что DbHandle ходит через _run_in_db_thread
# ---------------------------------------------------------------------------

async def test_handle_routes_through_db_worker(db, handle):
    """execute/fetch_one/fetch_all/transaction используют единственный воркер-поток."""
    worker_threads: set[int] = set()

    import threading

    original = db._run_in_db_thread

    async def patched(func, *args):
        # Запоминаем tid потока, в котором выполняется func.

        def wrapper():
            worker_threads.add(threading.current_thread().ident)
            return func(*args)

        import asyncio
        loop = asyncio.get_running_loop()
        import contextvars
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            db._get_executor(),
            lambda: ctx.run(wrapper),
        )

    db._run_in_db_thread = patched

    try:
        await _create_kv_table(handle)
        await handle.execute("INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (10, "x", 1))
        await handle.fetch_one("SELECT id FROM kv WHERE id = ?", (10,))
        await handle.fetch_all("SELECT id FROM kv ORDER BY id")
        async with handle.transaction() as tx:
            await tx.execute("INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (11, "y", 2))
    finally:
        db._run_in_db_thread = original

    # Все операции шли через один воркер-поток.
    assert len(worker_threads) == 1


async def test_transaction_uses_db_worker(db, handle):
    """Батч из transaction() доставляется через _run_in_db_thread."""
    await _create_kv_table(handle)

    async with handle.transaction() as tx:
        await tx.execute("INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (1, "a", 10))
        await tx.execute("INSERT INTO kv(id, name, value) VALUES (?, ?, ?)", (2, "b", 20))

    rows = await handle.fetch_all("SELECT id FROM kv ORDER BY id")
    assert [r["id"] for r in rows] == [1, 2]
