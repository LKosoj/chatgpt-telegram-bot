"""Tests for HindsightMemoryPlugin.on_session_before_delete hook subscription.

Stage 4C-3+5: plugin owns the enqueue logic and writes via DbHandle.execute.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("tiktoken")

from bot.plugins.hindsight_memory import HindsightMemoryPlugin
from bot.plugins.hooks import SessionBeforeDeletePayload


def _make_plugin(*, enabled: bool = True, auto_save: bool = True):
    plugin = HindsightMemoryPlugin()
    cfg = {
        'hindsight_base_url': 'http://x' if enabled else '',
        'hindsight_api_token': 't' if enabled else '',
        'hindsight_auto_save': auto_save,
    }
    plugin.initialize(plugin_config=cfg)
    return plugin


@pytest.mark.asyncio
async def test_on_session_before_delete_inserts_finalize_job():
    plugin = _make_plugin()
    plugin.db_handle = AsyncMock()
    plugin.db_handle.execute = AsyncMock()

    messages = ({"role": "user", "content": "remember"}, {"role": "assistant", "content": "ok"})
    payload = SessionBeforeDeletePayload(user_id=42, session_id="s-1", messages=messages)

    await plugin.on_session_before_delete(payload)

    plugin.db_handle.execute.assert_awaited_once()
    sql, params = plugin.db_handle.execute.await_args.args
    assert "INSERT INTO hindsight_finalize_jobs" in sql
    assert params[0] == 42
    assert params[1] == "s-1"
    decoded = json.loads(params[2])
    assert decoded["messages"] == [dict(m) for m in messages]
    assert decoded["clear_generation"] == 0


@pytest.mark.asyncio
async def test_on_session_before_delete_skips_when_messages_empty():
    plugin = _make_plugin()
    plugin.db_handle = AsyncMock()
    plugin.db_handle.execute = AsyncMock()

    payload = SessionBeforeDeletePayload(user_id=42, session_id="s-1", messages=())

    await plugin.on_session_before_delete(payload)

    plugin.db_handle.execute.assert_not_called()


@pytest.mark.asyncio
async def test_on_session_before_delete_skips_when_session_id_missing():
    plugin = _make_plugin()
    plugin.db_handle = AsyncMock()
    plugin.db_handle.execute = AsyncMock()

    payload = SessionBeforeDeletePayload(
        user_id=42, session_id="", messages=({"role": "user", "content": "x"},),
    )

    await plugin.on_session_before_delete(payload)

    plugin.db_handle.execute.assert_not_called()


@pytest.mark.asyncio
async def test_on_session_before_delete_skips_when_hindsight_disabled():
    plugin = _make_plugin(enabled=False)
    plugin.db_handle = AsyncMock()
    plugin.db_handle.execute = AsyncMock()

    payload = SessionBeforeDeletePayload(
        user_id=42, session_id="s-1", messages=({"role": "user", "content": "x"},),
    )

    await plugin.on_session_before_delete(payload)

    plugin.db_handle.execute.assert_not_called()


@pytest.mark.asyncio
async def test_on_session_before_delete_skips_when_auto_save_off():
    plugin = _make_plugin(auto_save=False)
    plugin.db_handle = AsyncMock()
    plugin.db_handle.execute = AsyncMock()

    payload = SessionBeforeDeletePayload(
        user_id=42, session_id="s-1", messages=({"role": "user", "content": "x"},),
    )

    await plugin.on_session_before_delete(payload)

    plugin.db_handle.execute.assert_not_called()


@pytest.mark.asyncio
async def test_on_session_before_delete_no_db_handle_logs_and_skips(caplog):
    plugin = _make_plugin()
    plugin.db_handle = None

    payload = SessionBeforeDeletePayload(
        user_id=42, session_id="s-1", messages=({"role": "user", "content": "x"},),
    )

    with caplog.at_level("WARNING"):
        await plugin.on_session_before_delete(payload)

    assert any("db_handle is None" in record.message for record in caplog.records)
