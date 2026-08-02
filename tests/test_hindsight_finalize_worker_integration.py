"""Integration tests for the Hindsight finalize worker against a real SQLite DB.

The unit tests in ``test_hindsight_finalize_worker.py`` mock the per-job SQL
helpers, so the actual ``BEGIN IMMEDIATE`` claim, lease re-claim window, attempt
counter, and pending/failed transition would go uncovered after Stage 4C-3+5.
This module exercises those paths end-to-end through ``Database`` + ``DbHandle``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("tiktoken")

from bot.database import Database
from bot.plugins.db_handle import DbHandle
from bot.plugins.hindsight_memory import HindsightMemoryPlugin
from bot.plugins.hooks import SessionBeforeDeletePayload


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._reset_singleton()
    database = Database()
    plugin_for_ddl = HindsightMemoryPlugin()
    with database.get_connection() as conn:
        cursor = conn.cursor()
        for stmt in plugin_for_ddl.register_schema():
            cursor.execute(stmt)
    yield database
    Database._reset_singleton()


def _make_plugin(db):
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={
        'hindsight_base_url': 'http://x',
        'hindsight_api_token': 't',
        'hindsight_auto_save': True,
    })
    plugin.client = SimpleNamespace(enabled=True)
    plugin.db_handle = DbHandle(db)
    return plugin


def _job_rows(db):
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, user_id, session_id, status, attempts, '
            'saved_count, last_error, locked_at, kind, autonomous '
            'FROM hindsight_finalize_jobs ORDER BY id'
        )
        return [dict(row) for row in cursor.fetchall()]


@pytest.mark.asyncio
async def test_hook_inserts_job_then_worker_marks_done(db):
    plugin = _make_plugin(db)
    plugin._extract_session_memory_items = AsyncMock(return_value=[{"content": "a"}, {"content": "b"}])
    plugin._retain_session_memory_items = AsyncMock(return_value=2)

    await plugin.on_session_before_delete(SessionBeforeDeletePayload(
        user_id=11, session_id="s-ok",
        messages=({"role": "user", "content": "hello"},),
    ))

    rows = _job_rows(db)
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["attempts"] == 0
    # session_close jobs mix live and any autonomous turns from the same session,
    # so they must never be marked autonomous (see _enqueue_finalize_job_sync).
    assert rows[0]["kind"] == "session_close"
    assert rows[0]["autonomous"] == 0

    await plugin._finalize_tick(application=None)

    plugin._extract_session_memory_items.assert_awaited_once_with(
        11,
        "s-ok",
        [{"role": "user", "content": "hello"}],
        raise_on_error=True,
    )
    plugin._retain_session_memory_items.assert_awaited_once_with(
        11,
        "s-ok",
        [{"content": "a"}, {"content": "b"}],
        async_store=False,
    )
    rows = _job_rows(db)
    assert rows[0]["status"] == "done"
    assert rows[0]["saved_count"] == 2
    assert rows[0]["last_error"] is None


@pytest.mark.asyncio
async def test_worker_marks_failed_when_finalize_raises(db):
    plugin = _make_plugin(db)
    plugin._extract_session_memory_items = AsyncMock(side_effect=RuntimeError("boom"))

    await plugin.on_session_before_delete(SessionBeforeDeletePayload(
        user_id=22, session_id="s-retry",
        messages=({"role": "user", "content": "x"},),
    ))
    await plugin._finalize_tick(application=None)

    rows = _job_rows(db)
    assert rows[0]["attempts"] == 1
    assert rows[0]["status"] == "pending"
    assert "boom" in (rows[0]["last_error"] or "")


@pytest.mark.asyncio
async def test_worker_retries_corrupt_payload_and_processes_valid_same_batch(db):
    plugin = _make_plugin(db)
    valid_messages = [{"role": "user", "content": "valid"}]
    plugin._extract_session_memory_items = AsyncMock(return_value=[{"content": "valid"}])
    plugin._retain_session_memory_items = AsyncMock(return_value=1)

    with db.get_connection() as conn:
        conn.execute(
            'INSERT INTO hindsight_finalize_jobs (user_id, session_id, messages) VALUES (?, ?, ?)',
            (55, "s-corrupt", "{not-json"),
        )
        conn.execute(
            'INSERT INTO hindsight_finalize_jobs (user_id, session_id, messages) VALUES (?, ?, ?)',
            (
                55,
                "s-valid",
                json.dumps({"messages": valid_messages, "clear_generation": 0}),
            ),
        )

    await plugin._finalize_tick(application=None)

    plugin._extract_session_memory_items.assert_awaited_once_with(
        55, "s-valid", valid_messages, raise_on_error=True,
    )
    plugin._retain_session_memory_items.assert_awaited_once_with(
        55, "s-valid", [{"content": "valid"}], async_store=False,
    )
    rows = _job_rows(db)
    assert rows[0]["session_id"] == "s-corrupt"
    assert rows[0]["status"] == "pending"
    assert rows[0]["attempts"] == 1
    assert rows[0]["locked_at"] is None
    assert "Invalid finalize job payload" in (rows[0]["last_error"] or "")
    assert rows[1]["session_id"] == "s-valid"
    assert rows[1]["status"] == "done"
    assert rows[1]["saved_count"] == 1


@pytest.mark.asyncio
async def test_worker_skips_claimed_job_after_memory_clear(db):
    plugin = _make_plugin(db)
    plugin._extract_session_memory_items = AsyncMock(return_value=[{"content": "old"}])
    plugin._retain_session_memory_items = AsyncMock(return_value=1)

    await plugin.on_session_before_delete(SessionBeforeDeletePayload(
        user_id=44, session_id="s-stale",
        messages=({"role": "user", "content": "old fact"},),
    ))
    claimed_jobs = plugin._claim_finalize_jobs_sync(db)
    assert len(claimed_jobs) == 1

    plugin.client = SimpleNamespace(
        enabled=True,
        clear_bank=AsyncMock(return_value={"success": True}),
    )
    await plugin._clear_memory(SimpleNamespace(), 44)
    plugin._claim_finalize_jobs_sync = lambda _db: claimed_jobs

    await plugin._finalize_tick(application=None)

    plugin._extract_session_memory_items.assert_not_awaited()
    plugin._retain_session_memory_items.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_status_becomes_failed_after_max_attempts(db, monkeypatch):
    # Lower the threshold so the test stays fast and intent-clear.
    monkeypatch.setattr(
        "bot.plugins.hindsight_memory.HINDSIGHT_FINALIZE_JOB_MAX_ATTEMPTS", 2,
    )
    # Eliminate retry backoff so re-claim sees the job immediately.
    monkeypatch.setattr(
        "bot.plugins.hindsight_memory.HINDSIGHT_FINALIZE_JOB_RETRY_SECONDS", 0,
    )
    plugin = _make_plugin(db)
    plugin._extract_session_memory_items = AsyncMock(side_effect=RuntimeError("boom"))

    await plugin.on_session_before_delete(SessionBeforeDeletePayload(
        user_id=33, session_id="s-fail",
        messages=({"role": "user", "content": "x"},),
    ))
    await plugin._finalize_tick(application=None)
    await plugin._finalize_tick(application=None)

    rows = _job_rows(db)
    assert rows[0]["attempts"] == 2
    assert rows[0]["status"] == "failed"
