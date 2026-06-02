"""Integration test for T2: session-logging events emitted by OpenAIHelper."""
from __future__ import annotations

import asyncio
import json
import os
import types
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.openai_helper import OpenAIHelper, _TURN_STATS
from bot.session_logger import SessionLogger, set_trace, clear_trace, get_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_response(content="Hello!", prompt_tokens=10, completion_tokens=5):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    response = MagicMock()
    response.usage = usage
    response.choices = [choice]
    return response


def _make_helper(tmp_path) -> OpenAIHelper:
    helper = object.__new__(OpenAIHelper)
    helper.config = {
        'model': 'test-model',
        'session_log_enabled': True,
        'session_log_dir': str(tmp_path / 'session_logs'),
    }
    helper.session_logger = SessionLogger(enabled=True, base_dir=str(tmp_path / 'session_logs'))
    # Minimal client mock
    fake_resp = _make_fake_response()
    client_mock = MagicMock()
    client_mock.chat = MagicMock()
    client_mock.chat.completions = MagicMock()
    client_mock.chat.completions.create = AsyncMock(return_value=fake_resp)
    helper.client = client_mock
    return helper


# ---------------------------------------------------------------------------
# Tests for _timed_create and _emit_turn_end (lightweight, no full init)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timed_create_writes_llm_call_event(tmp_path):
    """_timed_create records an llm_call event when trace is active."""
    helper = _make_helper(tmp_path)

    user_id = 42
    session_id = "sess-abc"
    turn_id = uuid.uuid4().hex

    trace_token = set_trace(user_id, session_id, turn_id)
    stats_token = _TURN_STATS.set({'round_trips': 0, 'llm_ms': 0.0, 'mutator_ms': 0.0, 'start': 0.0})
    try:
        await helper._timed_create(
            kind='test_kind',
            model='test-model',
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
    finally:
        _TURN_STATS.reset(stats_token)
        clear_trace(trace_token)

    await helper.session_logger.drain()

    log_path = os.path.join(str(tmp_path / 'session_logs'), '42', 'sess-abc.jsonl')
    assert os.path.exists(log_path), "jsonl log file was not created"
    lines = [json.loads(l) for l in open(log_path).read().splitlines() if l.strip()]
    llm_events = [e for e in lines if e.get('type') == 'llm_call']
    assert len(llm_events) == 1
    ev = llm_events[0]
    assert ev['kind'] == 'test_kind'
    assert ev['stream'] is False
    assert ev['prompt_tokens'] == 10
    assert ev['completion_tokens'] == 5
    assert ev['finish_reason'] == 'stop'
    assert ev['tool_calls_returned'] == 0


@pytest.mark.asyncio
async def test_emit_turn_end_writes_turn_end_event(tmp_path):
    """_emit_turn_end records a turn_end event."""
    import time

    helper = _make_helper(tmp_path)

    user_id = 99
    session_id = "sess-end"
    turn_id = uuid.uuid4().hex

    trace_token = set_trace(user_id, session_id, turn_id)
    stats_token = _TURN_STATS.set({
        'round_trips': 2,
        'llm_ms': 350.0,
        'mutator_ms': 12.0,
        'start': time.monotonic() - 0.5,
    })
    try:
        helper._emit_turn_end(stats_token, user_id, session_id)
    finally:
        clear_trace(trace_token)

    await helper.session_logger.drain()

    log_path = os.path.join(str(tmp_path / 'session_logs'), '99', 'sess-end.jsonl')
    assert os.path.exists(log_path)
    lines = [json.loads(l) for l in open(log_path).read().splitlines() if l.strip()]
    turn_end_events = [e for e in lines if e.get('type') == 'turn_end']
    assert len(turn_end_events) == 1
    ev = turn_end_events[0]
    assert ev['round_trips'] == 2
    assert ev['llm_ms_total'] == 350.0
    assert ev['mutator_ms_total'] == 12.0
    assert ev['wall_ms'] >= 0


@pytest.mark.asyncio
async def test_get_chat_response_emits_turn_lifecycle(tmp_path):
    """get_chat_response emits turn_start + assistant_response + turn_end end-to-end."""
    helper = _make_helper(tmp_path)
    # Override heavy internals so we exercise only the turn-lifecycle wrapper.
    helper._chat_state_key = lambda chat_id: f"k{chat_id}"
    helper._chat_lock_bypass_enabled = lambda chat_id: True
    helper._get_chat_response_locked = AsyncMock(return_value=("Hi there", "5"))

    answer, tokens = await helper.get_chat_response(
        chat_id=7, query="hello", session_id="sess-x", user_id=7,
    )
    assert answer == "Hi there"
    assert tokens == "5"

    await helper.session_logger.drain()

    log_path = os.path.join(str(tmp_path / 'session_logs'), '7', 'sess-x.jsonl')
    assert os.path.exists(log_path)
    lines = [json.loads(l) for l in open(log_path).read().splitlines() if l.strip()]
    by_type = {e['type']: e for e in lines}

    assert 'turn_start' in by_type
    assert by_type['turn_start']['user_message'] == 'hello'
    assert by_type['turn_start']['model_requested'] == 'test-model'
    assert by_type['turn_start']['chat_id'] == 7

    assert 'assistant_response' in by_type
    assert by_type['assistant_response']['text'] == 'Hi there'

    assert 'turn_end' in by_type


@pytest.mark.asyncio
async def test_mutators_event_recorded(tmp_path):
    """_apply_before_chat_request_mutators records a mutators event with injected_count."""
    helper = _make_helper(tmp_path)
    helper._chat_state_key = lambda chat_id: "k"
    helper._repair_tool_call_history = lambda state_key: None
    helper._messages_with_language_instruction = lambda msgs: list(msgs)
    helper.conversations = {"k": [{"role": "user", "content": "hi"}]}

    plugin_manager = MagicMock()
    plugin_manager.get_plugin.return_value = None
    # apply_mutators injects one extra message -> injected_count == 1
    plugin_manager.apply_mutators = AsyncMock(
        return_value=[{"role": "user", "content": "hi"}, {"role": "system", "content": "x"}]
    )
    helper.plugin_manager = plugin_manager

    trace_token = set_trace(5, "sess-m", uuid.uuid4().hex)
    stats_token = _TURN_STATS.set({'round_trips': 0, 'llm_ms': 0, 'mutator_ms': 0, 'start': 0.0})
    try:
        await helper._apply_before_chat_request_mutators(
            chat_id=1, user_id=5, session_id="sess-m", request_id="r", persist=False,
        )
    finally:
        _TURN_STATS.reset(stats_token)
        clear_trace(trace_token)

    await helper.session_logger.drain()

    log_path = os.path.join(str(tmp_path / 'session_logs'), '5', 'sess-m.jsonl')
    assert os.path.exists(log_path)
    lines = [json.loads(l) for l in open(log_path).read().splitlines() if l.strip()]
    mut_events = [e for e in lines if e.get('type') == 'mutators']
    assert len(mut_events) == 1
    assert mut_events[0]['injected_count'] == 1
    assert mut_events[0]['duration_ms'] >= 0


@pytest.mark.asyncio
async def test_no_crash_when_session_logger_absent(tmp_path):
    """_timed_create and _emit_turn_end are safe when session_logger is not set."""
    import time

    helper = object.__new__(OpenAIHelper)
    # Deliberately no session_logger attribute
    fake_resp = _make_fake_response()
    client_mock = MagicMock()
    client_mock.chat.completions.create = AsyncMock(return_value=fake_resp)
    helper.client = client_mock

    stats_token = _TURN_STATS.set({'round_trips': 0, 'llm_ms': 0.0, 'mutator_ms': 0.0, 'start': time.monotonic()})
    trace_token = set_trace(1, "s", "t")
    try:
        resp = await helper._timed_create(
            kind='k',
            model='m',
            messages=[],
            stream=False,
        )
        assert resp is fake_resp
        # No exception should be raised
        helper._emit_turn_end(stats_token, 1, "s")
    finally:
        clear_trace(trace_token)
