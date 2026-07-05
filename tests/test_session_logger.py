"""Tests for bot/session_logger.py (T1 — session-logging foundation)."""
import asyncio
import json
import logging
import os
import threading
import warnings

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


def test_sync_fallback_does_not_leak_writer_coroutine(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        logger.record({'type': 'turn_end', 'user_id': 'u0', 'session_id': 's0', 'wall_ms': 1})

    assert not [
        warning for warning in caught
        if "was never awaited" in str(warning.message)
    ]
    assert (tmp_path / 'u0' / 's0.jsonl').exists()


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
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
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
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(lines) == 5


# ---------------------------------------------------------------------------
# WP-D: _stats не растёт — pop после успешного flush_summary
# ---------------------------------------------------------------------------

async def test_stats_key_removed_after_flush_summary(tmp_path):
    """После успешного flush_summary ключ удаляется из _stats."""
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    logger.record({'type': 'llm_call', 'user_id': 'u3', 'session_id': 's3',
                   'kind': 'chat', 'duration_ms': 50})
    key = ('u3', 's3')
    assert key in logger._stats
    await logger.flush_summary('u3', 's3')
    assert key not in logger._stats


async def test_concurrent_record_during_flush_not_lost(tmp_path):
    """record(), пришедший во время записи summary, не теряется (pop до await).

    flush_summary извлекает агрегатор синхронно (pop до await), поэтому на
    момент записи summary ключ уже отсутствует в _stats, а конкурентный
    record() создаёт свежий агрегатор, который не будет затёрт.
    """
    import unittest.mock as mock
    from bot import session_logger as sl

    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    logger.record({'type': 'llm_call', 'user_id': 'u6', 'session_id': 's6',
                   'kind': 'chat', 'duration_ms': 10})
    key = ('u6', 's6')

    def fake_write(path, summary):
        # Эмулируем record() во время записи summary: ключ уже извлечён
        # flush'ем (pop до await), значит создаётся свежий агрегатор.
        assert key not in logger._stats
        logger._stats.setdefault(key, sl._SessionStats()).llm_total += 1

    with mock.patch.object(sl, '_write_summary', side_effect=fake_write):
        await logger.flush_summary('u6', 's6')

    # «Параллельный» инкремент сохранился под свежим агрегатором.
    assert key in logger._stats
    assert logger._stats[key].llm_total == 1


async def test_flush_summary_no_prior_record_is_noop(tmp_path):
    """flush_summary для ключа без событий не падает (no-op)."""
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    # Нет предварительных record() — ключ отсутствует в _stats
    await logger.flush_summary('u5', 's5')
    assert ('u5', 's5') not in logger._stats
    assert not (tmp_path / 'u5' / 's5.summary.json').exists()


async def test_record_does_not_mutate_event_dict(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    event = {'type': 'turn_end', 'user_id': 'u7', 'session_id': 's7'}

    logger.record(event)
    await logger.drain()

    assert 'ts' not in event


async def test_record_uses_single_writer_task(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))

    for i in range(5):
        logger.record({'type': 'turn_end', 'user_id': 'u8', 'session_id': 's8', 'wall_ms': i})
    writer_task = logger._writer_task

    assert writer_task is not None
    await logger.drain()
    assert logger._writer_task is writer_task
    assert not writer_task.done()

    await logger.close()
    assert writer_task.done()


async def test_close_waits_for_scheduled_summary(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    logger.record({'type': 'llm_call', 'user_id': 'u10', 'session_id': 's10', 'duration_ms': 5})

    assert logger.schedule_flush_summary('u10', 's10') is True
    await logger.close()

    summary_path = tmp_path / 'u10' / 's10.summary.json'
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    assert data['llm_calls']['total'] == 1


async def test_schedule_flush_summary_coalesces_pending_session_task(tmp_path, monkeypatch):
    from bot import session_logger as sl

    original_write_summary = sl._write_summary
    started = threading.Event()
    release = threading.Event()

    def slow_write_summary(path, summary):
        started.set()
        assert release.wait(5)
        original_write_summary(path, summary)

    monkeypatch.setattr(sl, '_write_summary', slow_write_summary)
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    logger.record({'type': 'llm_call', 'user_id': 'u11', 'session_id': 's11', 'duration_ms': 5})

    assert logger.schedule_flush_summary('u11', 's11') is True
    assert await asyncio.to_thread(started.wait, 5)
    assert len(logger._summary_tasks) == 1

    assert logger.schedule_flush_summary('u11', 's11') is True
    assert len(logger._summary_tasks) == 1

    release.set()
    await logger.drain()

    summary_path = tmp_path / 'u11' / 's11.summary.json'
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    assert data['llm_calls']['total'] == 1


async def test_drain_summary_tasks_allows_finished_task_callback_to_run(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    key = ('u12', 's12')

    async def done():
        return True

    task = asyncio.create_task(done())
    await task
    logger._summary_tasks[key] = task
    task.add_done_callback(lambda _task: logger._summary_tasks.pop(key, None))

    await asyncio.wait_for(logger._drain_summary_tasks(), timeout=1)

    assert logger._summary_tasks == {}


async def test_schedule_flush_summary_reruns_when_stats_arrive_during_active_flush(tmp_path, monkeypatch):
    from bot import session_logger as sl

    original_write_summary = sl._write_summary
    started = threading.Event()
    release = threading.Event()
    writes = []

    def slow_write_summary(path, summary):
        writes.append(summary)
        if len(writes) == 1:
            started.set()
            assert release.wait(5)
        original_write_summary(path, summary)

    monkeypatch.setattr(sl, '_write_summary', slow_write_summary)
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    logger.record({'type': 'llm_call', 'user_id': 'u13', 'session_id': 's13', 'duration_ms': 5})

    assert logger.schedule_flush_summary('u13', 's13') is True
    assert await asyncio.to_thread(started.wait, 5)
    logger.record({'type': 'llm_call', 'user_id': 'u13', 'session_id': 's13', 'duration_ms': 7})
    assert logger.schedule_flush_summary('u13', 's13') is True

    release.set()
    await logger.close()

    assert len(writes) == 2
    assert logger._stats == {}
    summary_path = tmp_path / 'u13' / 's13.summary.json'
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    assert data['llm_calls']['total'] == 1
    assert data['llm_ms']['total'] == 7


async def test_scheduled_summary_exception_is_retrieved(tmp_path, monkeypatch, caplog):
    from bot import session_logger as sl

    def fail_write_summary(_path, _summary):
        raise RuntimeError("summary write failed")

    monkeypatch.setattr(sl, '_write_summary', fail_write_summary)
    caplog.set_level(logging.DEBUG)
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path))
    logger.record({'type': 'llm_call', 'user_id': 'u12', 'session_id': 's12', 'duration_ms': 5})

    assert logger.schedule_flush_summary('u12', 's12') is True
    await logger.drain()

    assert logger._summary_tasks == {}
    assert any("session summary flush failed" in record.getMessage() for record in caplog.records)


async def test_jsonl_rotates_when_size_limit_exceeded(tmp_path):
    logger = SessionLogger(enabled=True, base_dir=str(tmp_path), max_file_bytes=120)

    logger.record({
        'type': 'turn_end',
        'user_id': 'u9',
        'session_id': 's9',
        'wall_ms': 1,
        'payload': 'x' * 80,
    })
    logger.record({
        'type': 'turn_end',
        'user_id': 'u9',
        'session_id': 's9',
        'wall_ms': 2,
        'payload': 'y' * 80,
    })
    await logger.drain()

    session_dir = tmp_path / 'u9'
    assert (session_dir / 's9.jsonl').exists()
    assert list(session_dir.glob('s9.jsonl.*'))


async def test_retention_deletes_old_session_logs(tmp_path):
    old_dir = tmp_path / 'old'
    old_dir.mkdir()
    old_jsonl = old_dir / 's.jsonl'
    old_summary = old_dir / 's.summary.json'
    old_jsonl.write_text('{}\n', encoding='utf-8')
    old_summary.write_text('{}', encoding='utf-8')
    old_mtime = 1_700_000_000
    os.utime(old_jsonl, (old_mtime, old_mtime))
    os.utime(old_summary, (old_mtime, old_mtime))

    logger = SessionLogger(
        enabled=True,
        base_dir=str(tmp_path),
        retention_seconds=1,
        cleanup_interval_seconds=0,
    )

    logger.record({'type': 'turn_end', 'user_id': 'new', 'session_id': 's', 'wall_ms': 1})
    await logger.drain()

    assert not old_jsonl.exists()
    assert not old_summary.exists()
    assert (tmp_path / 'new' / 's.jsonl').exists()
