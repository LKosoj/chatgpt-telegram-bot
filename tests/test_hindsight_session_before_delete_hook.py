"""Tests for HindsightMemoryPlugin.on_session_before_delete hook subscription."""

from __future__ import annotations

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
    plugin.db_handle = object()
    plugin._db_run_sync = AsyncMock(return_value=True)

    messages = ({"role": "user", "content": "remember"}, {"role": "assistant", "content": "ok"})
    payload = SessionBeforeDeletePayload(user_id=42, session_id="s-1", messages=messages)

    await plugin.on_session_before_delete(payload)

    plugin._db_run_sync.assert_awaited_once_with(
        plugin._enqueue_finalize_job_sync,
        42,
        "s-1",
        [dict(m) for m in messages],
    )


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
