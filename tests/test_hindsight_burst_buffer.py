"""Tests for the mid-conversation burst turn-buffer in HindsightMemoryPlugin.

The buffer accumulates ``on_user_message``/``on_assistant_response`` turns per
(user_id, chat_id) and flushes them into the *same* finalize-job queue used by
``on_session_before_delete`` (see ``_enqueue_finalize_job_sync``), instead of
extracting memory itself. Extraction stays solely in ``_finalize_tick``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("tiktoken")

from bot.plugins.hindsight_memory import HindsightMemoryPlugin
from bot.plugins.hooks import (
    AssistantResponsePayload,
    SessionBeforeDeletePayload,
    SessionResetPayload,
    UserMessagePayload,
)


class FakeDbHandle:
    """Only ``fetch_one`` is exercised directly by the buffer (active-session
    lookup); ``_db_run_sync`` itself is mocked per-test so job inserts don't
    need a real connection."""

    def __init__(self, active_session_id: str | None = "sess-active"):
        self.active_session_id = active_session_id

    async def fetch_one(self, sql, params=()):
        if self.active_session_id is None:
            return None
        return {"session_id": self.active_session_id}


def _make_plugin(*, auto_save: bool = True, burst_enabled: bool = True, **overrides):
    plugin = HindsightMemoryPlugin()
    cfg = {
        'hindsight_base_url': 'http://x',
        'hindsight_api_token': 't',
        'hindsight_auto_save': auto_save,
        'hindsight_burst_enabled': burst_enabled,
        'hindsight_burst_max_turns': 3,
        'hindsight_burst_quiet_seconds': 100,
        **overrides,
    }
    plugin.initialize(plugin_config=cfg)
    plugin.db_handle = FakeDbHandle()
    plugin._db_run_sync = AsyncMock(return_value=True)
    return plugin


def _user_payload(user_id=1, chat_id=1, text="hi"):
    return UserMessagePayload(
        chat_id=chat_id, user_id=user_id, request_id="r1", text=text,
        has_image=False, has_voice=False, is_command=False, ts=0.0,
    )


def _assistant_payload(user_id=1, chat_id=1, text="hello"):
    return AssistantResponsePayload(
        chat_id=chat_id, user_id=user_id, request_id="r1", text=text,
        tokens=10, model="m", ts=0.0,
    )


async def _full_turn(plugin, user_id=1, chat_id=1, utext="hi", atext="hello"):
    await plugin.on_user_message(_user_payload(user_id, chat_id, utext))
    await plugin.on_assistant_response(_assistant_payload(user_id, chat_id, atext))


# --- Accumulation without reaching the threshold ---------------------------

@pytest.mark.asyncio
async def test_accumulates_without_flush_below_max_turns():
    plugin = _make_plugin()  # max_turns=3
    await _full_turn(plugin)
    await _full_turn(plugin)

    key = (1, 1, False)
    buf = plugin._turn_buffers[key]
    assert buf.turn_count == 2
    assert [m["role"] for m in buf.messages] == ["user", "assistant", "user", "assistant"]
    plugin._db_run_sync.assert_not_awaited()


# --- Immediate flush on max_turns -------------------------------------------

@pytest.mark.asyncio
async def test_flushes_immediately_on_max_turns():
    plugin = _make_plugin()  # max_turns=3
    await _full_turn(plugin)
    await _full_turn(plugin)
    await _full_turn(plugin)

    key = (1, 1, False)
    assert key not in plugin._turn_buffers
    plugin._db_run_sync.assert_awaited_once()
    call = plugin._db_run_sync.call_args
    assert call.args[0] == plugin._enqueue_finalize_job_sync
    assert call.args[1] == 1  # user_id
    assert call.args[2] == "sess-active"  # active session_id
    assert len(call.args[3]) == 6  # 3 turns * (user + assistant)
    assert call.kwargs == {"kind": "burst"}


# --- Quiet-timeout flush via sweep, not earlier -----------------------------

@pytest.mark.asyncio
async def test_sweep_flushes_only_after_quiet_seconds(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(
        "bot.plugins.hindsight_memory.time.monotonic", lambda: clock["now"],
    )
    plugin = _make_plugin(hindsight_burst_quiet_seconds=100, hindsight_burst_max_turns=100)
    await _full_turn(plugin)

    clock["now"] = 50.0
    await plugin._burst_sweep_tick(application=None)
    plugin._db_run_sync.assert_not_awaited()
    assert (1, 1, False) in plugin._turn_buffers

    clock["now"] = 150.0
    await plugin._burst_sweep_tick(application=None)
    plugin._db_run_sync.assert_awaited_once()
    assert (1, 1, False) not in plugin._turn_buffers


# --- auto_save_enabled=False disables buffering entirely --------------------

@pytest.mark.asyncio
async def test_no_buffering_when_auto_save_disabled():
    plugin = _make_plugin(auto_save=False)
    await _full_turn(plugin)
    await _full_turn(plugin)
    await _full_turn(plugin)  # would hit max_turns=3 if buffering were active

    assert plugin._turn_buffers == {}
    plugin._db_run_sync.assert_not_awaited()


# --- Concurrent turns for the same key: no lost messages, no double flush --

@pytest.mark.asyncio
async def test_concurrent_turns_same_key_no_lost_messages_no_double_flush():
    plugin = _make_plugin(hindsight_burst_max_turns=5)
    await asyncio.gather(*[_full_turn(plugin) for _ in range(5)])

    plugin._db_run_sync.assert_awaited_once()
    call = plugin._db_run_sync.call_args
    assert len(call.args[3]) == 10  # 5 turns * (user + assistant), nothing lost
    assert (1, 1, False) not in plugin._turn_buffers


# --- close_async drains every key -------------------------------------------

@pytest.mark.asyncio
async def test_close_async_drains_all_keys():
    plugin = _make_plugin(hindsight_burst_max_turns=100)
    await _full_turn(plugin, user_id=1, chat_id=1)
    await _full_turn(plugin, user_id=2, chat_id=1)
    plugin.client = SimpleNamespace(close=AsyncMock())

    await plugin.close_async()

    assert plugin._turn_buffers == {}
    assert plugin._db_run_sync.await_count == 2
    plugin.client.close.assert_awaited_once()


# --- on_session_before_delete drains that user's buffers --------------------

@pytest.mark.asyncio
async def test_on_session_before_delete_drains_user_buffers():
    plugin = _make_plugin(hindsight_burst_max_turns=100)
    await _full_turn(plugin, user_id=42, chat_id=7)

    payload = SessionBeforeDeletePayload(
        user_id=42, session_id="s-1",
        messages=({"role": "user", "content": "remember"},),
    )
    await plugin.on_session_before_delete(payload)

    assert (42, 7, False) not in plugin._turn_buffers
    # One call to enqueue the session-close job, one to enqueue the burst flush.
    assert plugin._db_run_sync.await_count == 2
    kinds = [c.kwargs.get("kind") for c in plugin._db_run_sync.call_args_list]
    assert "burst" in kinds


# --- Regression: on_session_reset must not touch the buffer ----------------

@pytest.mark.asyncio
async def test_on_session_reset_does_not_touch_buffer():
    plugin = _make_plugin(hindsight_burst_max_turns=100)
    await _full_turn(plugin)
    before = plugin._turn_buffers[(1, 1, False)]
    snapshot_messages = list(before.messages)
    snapshot_turn_count = before.turn_count

    await plugin.on_session_reset(
        SessionResetPayload(chat_id=1, user_id=1, reason="request_start", terminal_only=False)
    )
    await plugin.on_session_reset(
        SessionResetPayload(chat_id=1, user_id=1, reason="final_delivery", terminal_only=False)
    )

    after = plugin._turn_buffers[(1, 1, False)]
    assert after.messages == snapshot_messages
    assert after.turn_count == snapshot_turn_count
    plugin._db_run_sync.assert_not_awaited()


# --- No active session: flush retains the buffer instead of losing it ------

@pytest.mark.asyncio
async def test_flush_retains_buffer_when_no_active_session_and_retries_next_turn():
    plugin = _make_plugin(hindsight_burst_max_turns=3)
    plugin.db_handle = FakeDbHandle(active_session_id=None)

    await _full_turn(plugin)
    await _full_turn(plugin)
    await _full_turn(plugin)

    key = (1, 1, False)
    assert key in plugin._turn_buffers
    assert plugin._turn_buffers[key].turn_count == 3
    assert len(plugin._turn_buffers[key].messages) == 6
    plugin._db_run_sync.assert_not_awaited()

    # A session becomes active later; the next turn's flush attempt sends the
    # whole retained buffer — nothing lost.
    plugin.db_handle.active_session_id = "sess-active"
    await _full_turn(plugin)

    assert key not in plugin._turn_buffers
    plugin._db_run_sync.assert_awaited_once()
    call = plugin._db_run_sync.call_args
    assert len(call.args[3]) == 8  # 4 turns * (user + assistant), nothing lost


# --- Flush failure (DB exception): buffer retained and retried later -------

@pytest.mark.asyncio
async def test_sweep_retains_buffer_on_db_exception_and_retries_next_sweep(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(
        "bot.plugins.hindsight_memory.time.monotonic", lambda: clock["now"],
    )
    plugin = _make_plugin(hindsight_burst_quiet_seconds=100, hindsight_burst_max_turns=100)
    plugin._db_run_sync = AsyncMock(side_effect=[RuntimeError("boom"), True])
    await _full_turn(plugin)

    clock["now"] = 150.0
    await plugin._burst_sweep_tick(application=None)
    assert plugin._db_run_sync.await_count == 1
    key = (1, 1, False)
    assert key in plugin._turn_buffers
    assert plugin._turn_buffers[key].turn_count == 1  # nothing lost

    clock["now"] = 300.0
    await plugin._burst_sweep_tick(application=None)
    assert plugin._db_run_sync.await_count == 2
    assert key not in plugin._turn_buffers


# --- Blocker 1: _clear_memory discards buffered turns, no stale burst job --

@pytest.mark.asyncio
async def test_clear_memory_discards_buffered_turns_prevents_stale_burst_job(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(
        "bot.plugins.hindsight_memory.time.monotonic", lambda: clock["now"],
    )
    plugin = _make_plugin(hindsight_burst_max_turns=100, hindsight_burst_quiet_seconds=100)

    await _full_turn(plugin, user_id=1, chat_id=1)
    await _full_turn(plugin, user_id=1, chat_id=2)
    assert (1, 1, False) in plugin._turn_buffers
    assert (1, 2, False) in plugin._turn_buffers

    plugin.client = SimpleNamespace(clear_bank=AsyncMock(), enabled=True)
    await plugin._clear_memory(helper=None, user_id=1)

    assert plugin._turn_buffers == {}
    plugin.client.clear_bank.assert_awaited_once()
    plugin._db_run_sync.assert_awaited_once()  # only _clear_local_memory_sync

    # Sweep past the quiet window: the discarded pre-clear turns must not
    # resurface as a burst finalize job.
    clock["now"] = 250.0
    await plugin._burst_sweep_tick(application=None)
    plugin._db_run_sync.assert_awaited_once()  # still just the clear's DB call


# --- Defect 3: per-buffer message cap ---------------------------------------

@pytest.mark.asyncio
async def test_buffer_messages_trimmed_at_max_buffer_messages():
    plugin = _make_plugin(hindsight_burst_max_turns=1000, hindsight_burst_max_buffer_messages=4)
    await _full_turn(plugin, utext="t1a", atext="t1b")
    await _full_turn(plugin, utext="t2a", atext="t2b")
    await _full_turn(plugin, utext="t3a", atext="t3b")

    buf = plugin._turn_buffers[(1, 1, False)]
    assert len(buf.messages) == 4
    assert [m["content"] for m in buf.messages] == ["t2a", "t2b", "t3a", "t3b"]
    assert buf.turn_count == 3  # turn_count itself is not trimmed


# --- Defect 3: dict-size cap on number of distinct buffers ------------------

@pytest.mark.asyncio
async def test_dict_size_capped_at_max_buffers_evicts_oldest(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(
        "bot.plugins.hindsight_memory.time.monotonic", lambda: clock["now"],
    )
    plugin = _make_plugin(hindsight_burst_max_turns=1000, hindsight_burst_max_buffers=2)

    clock["now"] = 1.0
    await _full_turn(plugin, user_id=1, chat_id=1)
    clock["now"] = 2.0
    await _full_turn(plugin, user_id=2, chat_id=1)
    assert len(plugin._turn_buffers) == 2

    clock["now"] = 3.0
    await _full_turn(plugin, user_id=3, chat_id=1)  # dict full -> evicts oldest (user 1)

    assert len(plugin._turn_buffers) == 2
    assert (1, 1, False) not in plugin._turn_buffers
    assert (2, 1, False) in plugin._turn_buffers
    assert (3, 1, False) in plugin._turn_buffers


# --- Autonomous capture: on_assistant_response(autonomous=True) uses a buffer key
# separate from live turns, so a burst flush never mixes cron-triggered content
# with real chat history for the same (user_id, chat_id). ------------------

def _autonomous_assistant_payload(user_id=1, chat_id=1, text="auto-hello"):
    return AssistantResponsePayload(
        chat_id=chat_id, user_id=user_id, request_id="r1", text=text,
        tokens=10, model="m", ts=0.0, autonomous=True,
    )


@pytest.mark.asyncio
async def test_autonomous_response_uses_separate_buffer_key_from_live_turns():
    plugin = _make_plugin(hindsight_burst_max_turns=100)
    await _full_turn(plugin, user_id=1, chat_id=1, utext="hi", atext="hello")
    # agent_cron only dispatches on_assistant_response (no on_user_message), so
    # exercise the hook directly the way it is actually fired in production.
    await plugin.on_assistant_response(_autonomous_assistant_payload(user_id=1, chat_id=1))

    live_key = (1, 1, False)
    autonomous_key = (1, 1, True)
    assert live_key in plugin._turn_buffers
    assert autonomous_key in plugin._turn_buffers
    assert plugin._turn_buffers[live_key].messages == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
    ]
    assert plugin._turn_buffers[autonomous_key].messages == [
        {"role": "assistant", "content": "auto-hello"},
    ]


@pytest.mark.asyncio
async def test_autonomous_buffer_flush_enqueues_job_with_autonomous_flag():
    plugin = _make_plugin(hindsight_burst_max_turns=1)
    await plugin.on_assistant_response(_autonomous_assistant_payload(user_id=1, chat_id=1))

    assert (1, 1, True) not in plugin._turn_buffers  # flushed immediately, max_turns=1
    plugin._db_run_sync.assert_awaited_once()
    call = plugin._db_run_sync.call_args
    assert call.kwargs == {"kind": "burst", "autonomous": True}


# --- Integration: an autonomous burst flush reaches _finalize_tick, which passes
# autonomous=True to the real extractor and its addendum shows up in the prompt --

@pytest.mark.asyncio
async def test_autonomous_burst_flush_extraction_includes_addendum(tmp_path, monkeypatch):
    from bot.database import Database
    from bot.plugins.db_handle import DbHandle
    from bot.plugins.hindsight_memory import AUTONOMOUS_EXTRACTION_ADDENDUM

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._reset_singleton()
    database = Database()
    plugin_for_ddl = HindsightMemoryPlugin()
    with database.get_connection() as conn:
        cursor = conn.cursor()
        for stmt in plugin_for_ddl.register_schema():
            cursor.execute(stmt)

    calls = []

    async def _chat_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"items":[]}'))],
        )

    fake_openai = SimpleNamespace(config={'light_model': 'fake-light'}, chat_completion=_chat_completion)

    plugin = HindsightMemoryPlugin()
    plugin.initialize(
        openai=fake_openai,
        plugin_config={
            'hindsight_base_url': 'http://x',
            'hindsight_api_token': 't',
            'hindsight_auto_save': True,
            'hindsight_burst_max_turns': 1,
        },
    )
    plugin.client = SimpleNamespace(enabled=True)
    plugin.db_handle = DbHandle(database)
    session_id = database.create_session(9)

    await plugin.on_assistant_response(AssistantResponsePayload(
        chat_id=9, user_id=9, request_id="r1", text="did the scheduled thing",
        tokens=1, model="m", ts=0.0, autonomous=True,
    ))

    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT kind, autonomous, session_id FROM hindsight_finalize_jobs"
        ).fetchone()
    assert row is not None
    assert row["kind"] == "burst"
    assert row["autonomous"] == 1
    assert row["session_id"] == session_id

    await plugin._finalize_tick(application=None)

    assert len(calls) == 1
    system_content = calls[0]["messages"][0]["content"]
    assert AUTONOMOUS_EXTRACTION_ADDENDUM in system_content

    Database._reset_singleton()


# --- Integration: a real-sqlite burst flush is picked up by _finalize_tick --

@pytest.mark.asyncio
async def test_burst_flush_writes_kind_burst_row_and_finalize_tick_claims_it(tmp_path, monkeypatch):
    from bot.database import Database
    from bot.plugins.db_handle import DbHandle

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._reset_singleton()
    database = Database()
    plugin_for_ddl = HindsightMemoryPlugin()
    with database.get_connection() as conn:
        cursor = conn.cursor()
        for stmt in plugin_for_ddl.register_schema():
            cursor.execute(stmt)

    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={
        'hindsight_base_url': 'http://x',
        'hindsight_api_token': 't',
        'hindsight_auto_save': True,
        'hindsight_burst_max_turns': 2,
    })
    plugin.client = SimpleNamespace(enabled=True)
    plugin.db_handle = DbHandle(database)
    session_id = database.create_session(7)

    await _full_turn(plugin, user_id=7, chat_id=7, utext="hi", atext="hello")
    await _full_turn(plugin, user_id=7, chat_id=7, utext="bye", atext="see ya")

    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT id, kind, session_id, status FROM hindsight_finalize_jobs"
        ).fetchone()
    assert row is not None
    assert row["kind"] == "burst"
    assert row["session_id"] == session_id
    assert row["status"] == "pending"
    job_id = row["id"]

    plugin._extract_session_memory_items = AsyncMock(return_value=[{"content": "fact"}])
    plugin._retain_session_memory_items = AsyncMock(return_value=1)

    await plugin._finalize_tick(application=None)

    plugin._retain_session_memory_items.assert_awaited_once_with(
        7, session_id, [{"content": "fact"}],
        async_store=False, document_id_suffix=f"burst-{job_id}",
    )
    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT status, saved_count FROM hindsight_finalize_jobs WHERE id = ?", (job_id,),
        ).fetchone()
    assert row["status"] == "done"
    assert row["saved_count"] == 1

    Database._reset_singleton()
