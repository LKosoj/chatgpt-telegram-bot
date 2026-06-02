"""Tests for bot/session_logger.py (T1 — session-logging foundation)."""
import json
import os

import pytest

from bot.session_logger import (
    SessionLogger,
    TraceContext,
    _sanitize_id,
    clear_trace,
    get_trace,
    set_trace,
)


# ---------------------------------------------------------------------------
# _sanitize_id (sync unit tests)
# ---------------------------------------------------------------------------

def test_sanitize_id_replaces_at():
    assert _sanitize_id('user@42') == 'user_42'


def test_sanitize_id_empty_returns_unknown():
    assert _sanitize_id('') == '_unknown_'


def test_sanitize_id_none_returns_unknown():
    assert _sanitize_id(None) == '_unknown_'


def test_sanitize_id_preserves_valid_chars():
    assert _sanitize_id('abc_123-XYZ') == 'abc_123-XYZ'


# ---------------------------------------------------------------------------
# set_trace / get_trace / clear_trace (sync)
# ---------------------------------------------------------------------------

def test_set_get_clear_trace():
    token = set_trace(1, 'sess1', 'turn1')
    t = get_trace()
    assert t == TraceContext(1, 'sess1', 'turn1')
    clear_trace(token)
    assert get_trace() is None


def test_clear_trace_without_token():
    set_trace(2, 'sess2', 'turn2')
    clear_trace()
    assert get_trace() is None


# ---------------------------------------------------------------------------
# disabled logger — no-op
# ---------------------------------------------------------------------------

async def test_disabled_no_files(tmp_path):
    logger = SessionLogger(enabled=False, base_dir=str(tmp_path))
    logger.record({'type': 'llm_call', 'user_id': 'u1', 'session_id': 's1'})
    await logger.drain()
    assert list(tmp_path.iterdir()) == []


async def test_disabled_flush_summary_noop(tmp_path):
    logger = SessionLogger(enabled=False, base_dir=str(tmp_path))
    logger.record({'type': 'llm_call', 'user_id': 'u1', 'session_id': 's1'})
    await logger.flush_summary('u1', 's1')
    await logger.drain()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# record — creates .jsonl with valid JSON + int ts
# ---------------------------------------------------------------------------

async def test_record_creates_jsonl(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    logger.record({'type': 'llm_call', 'user_id': 'u1', 'session_id': 's1'})
    await logger.drain()

    path = tmp_path / 'u1' / 's1.jsonl'
    assert path.exists()
    line = path.read_text(encoding='utf-8').strip()
    data = json.loads(line)
    assert data['type'] == 'llm_call'
    assert isinstance(data['ts'], int)


# ---------------------------------------------------------------------------
# multiple events append (3 lines)
# ---------------------------------------------------------------------------

async def test_record_appends_multiple_lines(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    for i in range(3):
        logger.record({'type': 'turn_end', 'user_id': 'u1', 'session_id': 's1', 'wall_ms': i * 10})
    await logger.drain()

    path = tmp_path / 'u1' / 's1.jsonl'
    lines = [l for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# summary: llm_call aggregation
# ---------------------------------------------------------------------------

async def test_summary_llm_call(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    logger.record({'type': 'llm_call', 'user_id': 'u1', 'session_id': 's1',
                   'kind': 'chat', 'duration_ms': 100,
                   'prompt_tokens': 10, 'completion_tokens': 5})
    logger.record({'type': 'llm_call', 'user_id': 'u1', 'session_id': 's1',
                   'kind': 'chat', 'duration_ms': 200,
                   'prompt_tokens': 20, 'completion_tokens': 15})
    await logger.flush_summary('u1', 's1')
    await logger.drain()

    path = tmp_path / 'u1' / 's1.summary.json'
    assert path.exists()
    data = json.loads(path.read_text(encoding='utf-8'))

    assert data['llm_calls']['total'] == 2
    assert data['llm_calls']['by_kind']['chat'] == 2
    assert data['llm_ms']['total'] == 300
    assert data['llm_ms']['max'] == 200
    assert data['llm_ms']['avg'] == pytest.approx(150.0)
    assert data['tokens']['prompt_total'] == 30
    assert data['tokens']['completion_total'] == 20


# ---------------------------------------------------------------------------
# summary: tool_exec aggregation (field: name)
# ---------------------------------------------------------------------------

async def test_summary_tool_exec(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    logger.record({'type': 'tool_exec', 'user_id': 'u1', 'session_id': 's1',
                   'name': 'search', 'duration_ms': 50})
    logger.record({'type': 'tool_exec', 'user_id': 'u1', 'session_id': 's1',
                   'name': 'search', 'duration_ms': 70})
    logger.record({'type': 'tool_exec', 'user_id': 'u1', 'session_id': 's1',
                   'name': 'calc', 'duration_ms': 30})
    await logger.flush_summary('u1', 's1')
    await logger.drain()

    data = json.loads((tmp_path / 'u1' / 's1.summary.json').read_text(encoding='utf-8'))
    assert data['tools']['total'] == 3
    assert data['tools']['ms_total'] == 150
    assert data['tools']['by_name']['search'] == 2
    assert data['tools']['by_name']['calc'] == 1


# ---------------------------------------------------------------------------
# summary: turn_end + mutators
# ---------------------------------------------------------------------------

async def test_summary_turn_end_and_mutators(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    logger.record({'type': 'turn_end', 'user_id': 'u1', 'session_id': 's1', 'wall_ms': 500})
    logger.record({'type': 'turn_end', 'user_id': 'u1', 'session_id': 's1', 'wall_ms': 300})
    logger.record({'type': 'mutators', 'user_id': 'u1', 'session_id': 's1', 'duration_ms': 20})
    await logger.flush_summary('u1', 's1')
    await logger.drain()

    data = json.loads((tmp_path / 'u1' / 's1.summary.json').read_text(encoding='utf-8'))
    assert data['turns'] == 2
    assert data['wall_ms_total'] == 800
    assert data['mutator_ms_total'] == 20


# ---------------------------------------------------------------------------
# trace fallback: event without user_id/session_id uses trace context
# ---------------------------------------------------------------------------

async def test_record_uses_trace_fallback(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    token = set_trace('trace_user', 'trace_sess', 'turn1')
    try:
        logger.record({'type': 'llm_call', 'duration_ms': 10})
        await logger.drain()
    finally:
        clear_trace(token)

    path = tmp_path / 'trace_user' / 'trace_sess.jsonl'
    assert path.exists()
    data = json.loads(path.read_text(encoding='utf-8').strip())
    assert data['type'] == 'llm_call'


# ---------------------------------------------------------------------------
# drain waits for background writes
# ---------------------------------------------------------------------------

async def test_drain_waits_for_writes(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    for i in range(5):
        logger.record({'type': 'turn_end', 'user_id': 'u2', 'session_id': 's2', 'wall_ms': i})
    await logger.drain()

    path = tmp_path / 'u2' / 's2.jsonl'
    lines = [l for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(lines) == 5
