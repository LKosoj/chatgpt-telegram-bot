"""Tests for tool_exec session logging in _call_function_bounded."""
from __future__ import annotations

import asyncio
import json

import pytest

from bot.openai_tool_handler import _call_function_bounded
from bot.session_logger import SessionLogger, clear_trace, set_trace


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------

class _FakePluginManager:
    def __init__(self, *, raise_exc=None):
        self._raise_exc = raise_exc

    async def call_function(self, name, helper, args, request_context=None):
        if self._raise_exc is not None:
            raise self._raise_exc
        return json.dumps({'result': 'ok'})


class _FakeHelper:
    def __init__(self, session_logger):
        self.plugin_manager = _FakePluginManager()
        self.session_logger = session_logger
        # no _without_chat_lock attribute


class _FakeHelperRaises:
    def __init__(self, session_logger, exc):
        self.plugin_manager = _FakePluginManager(raise_exc=exc)
        self.session_logger = session_logger


# ---------------------------------------------------------------------------
# Helpers to read written events
# ---------------------------------------------------------------------------

async def _drain_and_read_events(slog: SessionLogger, tmp_path) -> list[dict]:
    await slog.drain()
    events = []
    for jsonl_file in tmp_path.rglob('*.jsonl'):
        for line in jsonl_file.read_text().splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_success_records_tool_exec(tmp_path):
    slog = SessionLogger(enabled=True, base_dir=str(tmp_path))
    helper = _FakeHelper(slog)
    token = set_trace(user_id=1, session_id='s1', turn_id='t1')
    try:
        await _call_function_bounded(helper, 'my_tool', '{}', None, asyncio.Semaphore(1))
    finally:
        clear_trace(token)

    events = await _drain_and_read_events(slog, tmp_path)
    tool_events = [e for e in events if e.get('type') == 'tool_exec']
    assert len(tool_events) == 1
    evt = tool_events[0]
    assert evt['name'] == 'my_tool'
    assert evt['ok'] is True
    assert evt['error'] is None
    assert evt['duration_ms'] >= 0


async def test_failing_call_function_raises_and_records_failure(tmp_path):
    slog = SessionLogger(enabled=True, base_dir=str(tmp_path))
    exc = RuntimeError('boom')
    helper = _FakeHelperRaises(slog, exc)
    token = set_trace(user_id=2, session_id='s2', turn_id='t2')
    try:
        with pytest.raises(RuntimeError, match='boom'):
            await _call_function_bounded(helper, 'my_tool', '{}', None, asyncio.Semaphore(1))
    finally:
        clear_trace(token)

    events = await _drain_and_read_events(slog, tmp_path)
    tool_events = [e for e in events if e.get('type') == 'tool_exec']
    assert len(tool_events) == 1
    evt = tool_events[0]
    assert evt['ok'] is False
    assert 'boom' in evt['error']


async def test_no_trace_no_event(tmp_path):
    slog = SessionLogger(enabled=True, base_dir=str(tmp_path))
    helper = _FakeHelper(slog)
    # Ensure no trace is set
    clear_trace()
    await _call_function_bounded(helper, 'my_tool', '{}', None, asyncio.Semaphore(1))

    events = await _drain_and_read_events(slog, tmp_path)
    tool_events = [e for e in events if e.get('type') == 'tool_exec']
    assert tool_events == []


async def test_no_session_logger_does_not_raise(tmp_path):
    """Helper without session_logger attribute must not cause any exception."""

    class _NoLogHelper:
        plugin_manager = _FakePluginManager()

    helper = _NoLogHelper()
    token = set_trace(user_id=3, session_id='s3', turn_id='t3')
    try:
        result = await _call_function_bounded(helper, 'my_tool', '{}', None, asyncio.Semaphore(1))
        assert result is not None
    finally:
        clear_trace(token)
