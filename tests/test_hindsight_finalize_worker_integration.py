"""Integration tests for the Hindsight finalize worker against a real SQLite DB.

The unit tests in ``test_hindsight_finalize_worker.py`` mock the per-job SQL
helpers, so the actual ``BEGIN IMMEDIATE`` claim, lease re-claim window, attempt
counter, and pending/failed transition would go uncovered after Stage 4C-3+5.
This module exercises those paths end-to-end through ``Database`` + ``DbHandle``.
"""

from __future__ import annotations

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
            'SELECT id, user_id, session_id, status, attempts, saved_count, last_error '
            'FROM hindsight_finalize_jobs ORDER BY id'
        )
        return [dict(row) for row in cursor.fetchall()]


@pytest.mark.asyncio
async def test_hook_inserts_job_then_worker_marks_done(db):
    plugin = _make_plugin(db)
    plugin.finalize_session_memory = AsyncMock(return_value=2)

    await plugin.on_session_before_delete(SessionBeforeDeletePayload(
        user_id=11, session_id="s-ok",
        messages=({"role": "user", "content": "hello"},),
    ))

    rows = _job_rows(db)
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["attempts"] == 0

    await plugin._finalize_tick(application=None)

    plugin.finalize_session_memory.assert_awaited_once()
    rows = _job_rows(db)
    assert rows[0]["status"] == "done"
    assert rows[0]["saved_count"] == 2
    assert rows[0]["last_error"] is None


@pytest.mark.asyncio
async def test_worker_marks_failed_when_finalize_raises(db):
    plugin = _make_plugin(db)
    plugin.finalize_session_memory = AsyncMock(side_effect=RuntimeError("boom"))

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
    plugin.finalize_session_memory = AsyncMock(side_effect=RuntimeError("boom"))

    await plugin.on_session_before_delete(SessionBeforeDeletePayload(
        user_id=33, session_id="s-fail",
        messages=({"role": "user", "content": "x"},),
    ))
    await plugin._finalize_tick(application=None)
    await plugin._finalize_tick(application=None)

    rows = _job_rows(db)
    assert rows[0]["attempts"] == 2
    assert rows[0]["status"] == "failed"


