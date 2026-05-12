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
    # Only the pre-existing row should remain — buffered ops were discarded.
    assert [r["id"] for r in rows] == [1]


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
