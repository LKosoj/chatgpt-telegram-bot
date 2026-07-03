"""Tests for the Hindsight finalize background worker.

Stage 4C-3+5: ``HindsightMemoryPlugin._finalize_tick`` claims pending jobs,
extracts memory outside the user lock, retains it under the lock, and marks
jobs done/failed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("tiktoken")

from bot.plugins.hindsight_memory import HindsightMemoryPlugin


class FakeDbHandle:
    async def run_sync(self, func, *args, **kwargs):
        return func(object(), *args, **kwargs)


def _make_plugin(*, enabled: bool = True, auto_save: bool = True):
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={
        'hindsight_base_url': 'http://x' if enabled else '',
        'hindsight_api_token': 't' if enabled else '',
        'hindsight_auto_save': auto_save,
    })
    plugin.db_handle = FakeDbHandle()
    plugin._current_clear_generation_sync = MagicMock(return_value=0)
    return plugin


@pytest.mark.asyncio
async def test_finalize_tick_skips_when_disabled():
    plugin = _make_plugin(enabled=False)
    plugin._claim_finalize_jobs_sync = MagicMock()
    plugin._extract_session_memory_items = AsyncMock()

    await plugin._finalize_tick(application=None)

    plugin._claim_finalize_jobs_sync.assert_not_called()
    plugin._extract_session_memory_items.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_tick_no_jobs_does_not_call_finalize():
    plugin = _make_plugin()
    plugin._claim_finalize_jobs_sync = MagicMock(return_value=[])
    plugin._extract_session_memory_items = AsyncMock()
    plugin._mark_finalize_job_done_sync = MagicMock()
    plugin._mark_finalize_job_failed_sync = MagicMock()

    await plugin._finalize_tick(application=None)

    plugin._claim_finalize_jobs_sync.assert_called_once()
    plugin._extract_session_memory_items.assert_not_awaited()
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
    plugin._extract_session_memory_items = AsyncMock(return_value=[{"content": "a"}])
    plugin._retain_session_memory_items = AsyncMock(return_value=3)
    plugin._mark_finalize_job_done_sync = MagicMock()
    plugin._mark_finalize_job_failed_sync = MagicMock()

    await plugin._finalize_tick(application=None)

    plugin._extract_session_memory_items.assert_awaited_once_with(
        42, "s-1", [{"role": "user", "content": "x"}],
        raise_on_error=True,
    )
    plugin._retain_session_memory_items.assert_awaited_once_with(
        42, "s-1", [{"content": "a"}], async_store=False,
    )
    plugin._mark_finalize_job_done_sync.assert_called_once()
    args = plugin._mark_finalize_job_done_sync.call_args.args
    # signature: (db, job_id, saved_count)
    assert args[1] == 7
    assert args[2] == 3
    plugin._mark_finalize_job_failed_sync.assert_not_called()


@pytest.mark.asyncio
async def test_finalize_tick_extracts_memory_outside_user_lock():
    plugin = _make_plugin()
    job = {
        "id": 8,
        "user_id": 42,
        "session_id": "s-lock",
        "messages": [{"role": "user", "content": "x"}],
        "attempts": 0,
    }
    plugin._claim_finalize_jobs_sync = MagicMock(return_value=[job])
    events = []

    async def extract(*args, **kwargs):
        events.append(("extract", plugin._memory_user_lock(42).locked()))
        return [{"content": "a"}]

    async def retain(*args, **kwargs):
        events.append(("retain", plugin._memory_user_lock(42).locked()))
        return 1

    plugin._extract_session_memory_items = AsyncMock(side_effect=extract)
    plugin._retain_session_memory_items = AsyncMock(side_effect=retain)
    plugin._mark_finalize_job_done_sync = MagicMock()
    plugin._mark_finalize_job_failed_sync = MagicMock()

    await plugin._finalize_tick(application=None)

    assert events == [("extract", False), ("retain", True)]


@pytest.mark.asyncio
async def test_finalize_tick_marks_failed_on_exception():
    plugin = _make_plugin()
    job = {
        "id": 9, "user_id": 42, "session_id": "s-2",
        "messages": [{"role": "user", "content": "y"}], "attempts": 1,
    }
    plugin._claim_finalize_jobs_sync = MagicMock(return_value=[job])
    plugin._extract_session_memory_items = AsyncMock(side_effect=RuntimeError("boom"))
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
    plugin._extract_session_memory_items = AsyncMock(return_value=[])
    plugin._retain_session_memory_items = AsyncMock(return_value=0)
    plugin._mark_finalize_job_done_sync = MagicMock()

    await plugin._finalize_tick(application=None)

    assert plugin._extract_session_memory_items.await_count == 3
    plugin._retain_session_memory_items.assert_not_awaited()
    assert plugin._mark_finalize_job_done_sync.call_count == 3


@pytest.mark.asyncio
async def test_dream_tick_extracts_documents_outside_user_lock():
    plugin = _make_plugin()
    plugin.config["hindsight_dream_enabled"] = True
    plugin.openai = SimpleNamespace(config={"model": "llmgateway/light_model"})
    plugin._users_with_dream_events = AsyncMock(return_value=[{
        "user_id": 42,
        "last_event_id": 0,
        "max_event_id": 2,
    }])
    plugin._fetch_dream_events = AsyncMock(return_value=[
        {"id": 1, "event_type": "message", "payload": {"text": "one"}},
        {"id": 2, "event_type": "message", "payload": {"text": "two"}},
    ])
    plugin._fetch_approved_memory_documents = AsyncMock(return_value=[])
    events = []

    async def run_sync(func, *args, **kwargs):
        if func == plugin._create_dream_run_sync:
            events.append(("create", plugin._memory_user_lock(42).locked()))
            return 99
        if func == plugin._complete_dream_run_sync:
            events.append(("complete", plugin._memory_user_lock(42).locked()))
            return None
        raise AssertionError(f"unexpected db function: {func}")

    async def extract(*args, **kwargs):
        events.append(("extract", plugin._memory_user_lock(42).locked()))
        return [{"path": "preferences/example", "kind": "memory", "content": "User likes concise answers."}]

    plugin._db_run_sync = AsyncMock(side_effect=run_sync)
    plugin._extract_dream_documents = AsyncMock(side_effect=extract)

    await plugin._dream_tick(application=None)

    assert events == [("create", True), ("extract", False), ("complete", True)]


def test_get_background_tasks_returns_finalize_worker():
    plugin = _make_plugin()
    tasks = plugin.get_background_tasks()

    assert len(tasks) == 1
    task = tasks[0]
    assert task.name == "finalize_worker"
    assert task.interval_seconds == 30
    assert task.coroutine_factory == plugin._finalize_tick
