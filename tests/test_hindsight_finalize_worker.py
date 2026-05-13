"""Tests for the Hindsight finalize background worker.

Stage 4C-3+5: ``HindsightMemoryPlugin._finalize_tick`` claims pending jobs,
runs ``finalize_session_memory`` for each, and marks them done/failed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("tiktoken")

from bot.plugins.hindsight_memory import HindsightMemoryPlugin


def _make_plugin(*, enabled: bool = True, auto_save: bool = True):
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={
        'hindsight_base_url': 'http://x' if enabled else '',
        'hindsight_api_token': 't' if enabled else '',
        'hindsight_auto_save': auto_save,
    })
    # Hook db_handle so _finalize_tick has a Database instance via .database.
    plugin.db_handle = SimpleNamespace(database=object())
    return plugin


@pytest.mark.asyncio
async def test_finalize_tick_skips_when_disabled():
    plugin = _make_plugin(enabled=False)
    plugin._claim_finalize_jobs_sync = MagicMock()
    plugin.finalize_session_memory = AsyncMock()

    await plugin._finalize_tick(application=None)

    plugin._claim_finalize_jobs_sync.assert_not_called()
    plugin.finalize_session_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_tick_no_jobs_does_not_call_finalize():
    plugin = _make_plugin()
    plugin._claim_finalize_jobs_sync = MagicMock(return_value=[])
    plugin.finalize_session_memory = AsyncMock()
    plugin._mark_finalize_job_done_sync = MagicMock()
    plugin._mark_finalize_job_failed_sync = MagicMock()

    await plugin._finalize_tick(application=None)

    plugin._claim_finalize_jobs_sync.assert_called_once()
    plugin.finalize_session_memory.assert_not_awaited()
    plugin._mark_finalize_job_done_sync.assert_not_called()
    plugin._mark_finalize_job_failed_sync.assert_not_called()


@pytest.mark.asyncio
async def test_finalize_tick_marks_done_on_success():
    plugin = _make_plugin()
    job = {
        "id": 7,
        "user_id": 42,
        "session_id": "s-1",
        "messages": [{"role": "user", "content": "x"}],
        "attempts": 0,
    }
    plugin._claim_finalize_jobs_sync = MagicMock(return_value=[job])
    plugin.finalize_session_memory = AsyncMock(return_value=3)
    plugin._mark_finalize_job_done_sync = MagicMock()
    plugin._mark_finalize_job_failed_sync = MagicMock()

    await plugin._finalize_tick(application=None)

    plugin.finalize_session_memory.assert_awaited_once_with(
        42, "s-1", [{"role": "user", "content": "x"}],
        raise_on_error=True, async_store=False,
    )
    plugin._mark_finalize_job_done_sync.assert_called_once()
    args = plugin._mark_finalize_job_done_sync.call_args.args
    # signature: (db, job_id, saved_count)
    assert args[1] == 7
    assert args[2] == 3
    plugin._mark_finalize_job_failed_sync.assert_not_called()


@pytest.mark.asyncio
async def test_finalize_tick_marks_failed_on_exception():
    plugin = _make_plugin()
    job = {
        "id": 9, "user_id": 42, "session_id": "s-2",
        "messages": [{"role": "user", "content": "y"}], "attempts": 1,
    }
    plugin._claim_finalize_jobs_sync = MagicMock(return_value=[job])
    plugin.finalize_session_memory = AsyncMock(side_effect=RuntimeError("boom"))
    plugin._mark_finalize_job_done_sync = MagicMock()
    plugin._mark_finalize_job_failed_sync = MagicMock()

    await plugin._finalize_tick(application=None)

    plugin._mark_finalize_job_done_sync.assert_not_called()
    plugin._mark_finalize_job_failed_sync.assert_called_once()
    args = plugin._mark_finalize_job_failed_sync.call_args.args
    # signature: (db, job_id, error)
    assert args[1] == 9
    assert "boom" in args[2]


@pytest.mark.asyncio
async def test_finalize_tick_processes_multiple_jobs():
    plugin = _make_plugin()
    jobs = [
        {"id": 1, "user_id": 1, "session_id": "a", "messages": [], "attempts": 0},
        {"id": 2, "user_id": 2, "session_id": "b", "messages": [], "attempts": 0},
        {"id": 3, "user_id": 3, "session_id": "c", "messages": [], "attempts": 0},
    ]
    plugin._claim_finalize_jobs_sync = MagicMock(return_value=jobs)
    plugin.finalize_session_memory = AsyncMock(return_value=0)
    plugin._mark_finalize_job_done_sync = MagicMock()

    await plugin._finalize_tick(application=None)

    assert plugin.finalize_session_memory.await_count == 3
    assert plugin._mark_finalize_job_done_sync.call_count == 3


def test_get_background_tasks_returns_finalize_worker():
    plugin = _make_plugin()
    tasks = plugin.get_background_tasks()

    assert len(tasks) == 1
    task = tasks[0]
    assert task.name == "finalize_worker"
    assert task.interval_seconds == 30
    assert task.coroutine_factory == plugin._finalize_tick
