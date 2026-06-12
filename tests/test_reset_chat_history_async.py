"""Tests for the async ``reset_chat_history`` and its ``prune_old_sessions`` flag.

Stage 4C-3+5: ``reset_chat_history`` became ``async`` and gained an explicit
``prune_old_sessions`` parameter. When True (default), it must pre-dispatch
``on_session_before_delete`` for sessions that ``create_session`` is about to
prune; when False (summarise-overflow path), it must skip the pre-dispatch
and propagate the flag into ``Database.create_session``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("tiktoken")

from bot.openai_helper import OpenAIHelper


def _make_helper():
    helper = object.__new__(OpenAIHelper)
    helper.conversations = {}
    helper.loaded_conversation_sessions = {}
    helper.config = {'max_sessions': 5}
    helper.plugin_manager = SimpleNamespace(dispatch_blocking=AsyncMock())
    helper.db = SimpleNamespace(
        create_session=MagicMock(return_value="session-new"),
        get_conversation_context=MagicMock(return_value=(
            {"messages": []}, "HTML", 0.1, 80, "session-new",
        )),
        get_mode_from_context=MagicMock(return_value={"role": "system", "content": "sys"}),
        save_conversation_context=MagicMock(),
        get_oldest_session_ids_for_limit=MagicMock(return_value=[]),
    )
    return helper


@pytest.mark.asyncio
async def test_reset_chat_history_is_async():
    import inspect
    assert inspect.iscoroutinefunction(OpenAIHelper.reset_chat_history)


@pytest.mark.asyncio
async def test_reset_chat_history_with_prune_true_calls_pre_dispatch():
    helper = _make_helper()
    helper._dispatch_before_create_session_prune = AsyncMock()

    await helper.reset_chat_history(42)

    helper._dispatch_before_create_session_prune.assert_awaited_once_with(42)
    # create_session called with prune_old_sessions=True
    helper.db.create_session.assert_called_once()
    assert helper.db.create_session.call_args.kwargs["prune_old_sessions"] is True


@pytest.mark.asyncio
async def test_reset_chat_history_with_prune_false_skips_pre_dispatch():
    helper = _make_helper()
    helper._dispatch_before_create_session_prune = AsyncMock()

    await helper.reset_chat_history(42, prune_old_sessions=False)

    helper._dispatch_before_create_session_prune.assert_not_awaited()
    # create_session called with prune_old_sessions=False
    assert helper.db.create_session.call_args.kwargs["prune_old_sessions"] is False


@pytest.mark.asyncio
async def test_reset_chat_history_with_explicit_session_skips_both():
    helper = _make_helper()
    helper._dispatch_before_create_session_prune = AsyncMock()

    # Providing session_id should bypass create_session entirely.
    await helper.reset_chat_history(42, session_id="existing")

    helper._dispatch_before_create_session_prune.assert_not_awaited()
    helper.db.create_session.assert_not_called()


@pytest.mark.asyncio
async def test_async_method_signatures():
    """get_conversation_stats, resolve_allowed_plugins, __add_to_history are async."""
    import inspect
    assert inspect.iscoroutinefunction(OpenAIHelper.get_conversation_stats)
    assert inspect.iscoroutinefunction(OpenAIHelper.resolve_allowed_plugins)
    # __add_to_history is name-mangled
    add = getattr(OpenAIHelper, "_OpenAIHelper__add_to_history")
    assert inspect.iscoroutinefunction(add)


# ---------------------------------------------------------------------------
# Fix 1c: get_conversation_stats cold-cache behaviour
# ---------------------------------------------------------------------------

def _make_helper_stats(saved_context, session_id="s1"):
    """Helper with configurable DB context for stats tests."""
    helper = object.__new__(OpenAIHelper)
    helper.conversations = {}
    helper.loaded_conversation_sessions = {}
    helper.config = {'max_sessions': 5, 'model': 'llmgateway/light_model'}
    helper.plugin_manager = SimpleNamespace(dispatch_blocking=AsyncMock())
    helper.db = SimpleNamespace(
        create_session=MagicMock(return_value="session-new"),
        get_conversation_context=MagicMock(return_value=(
            saved_context, "HTML", 0.1, 80, session_id,
        )),
        get_mode_from_context=MagicMock(return_value={"role": "system", "content": "sys"}),
        save_conversation_context=MagicMock(),
        get_oldest_session_ids_for_limit=MagicMock(return_value=[]),
    )
    helper._save_conversation_context = AsyncMock()
    helper._dispatch_before_create_session_prune = AsyncMock()
    return helper


@pytest.mark.asyncio
async def test_get_conversation_stats_cold_loads_from_db():
    """Cold /stats: loads messages from DB, must NOT call create_session."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    helper = _make_helper_stats({"messages": messages}, session_id="existing-s")

    msg_count, _ = await helper.get_conversation_stats(42)

    # Should have loaded into cache
    state_key = helper._chat_state_key(42)
    assert state_key in helper.conversations
    assert helper.loaded_conversation_sessions[state_key] == "existing-s"
    # create_session must NOT have been called
    helper.db.create_session.assert_not_called()
    assert msg_count == len(messages)


@pytest.mark.asyncio
async def test_get_conversation_stats_cold_no_db_context_resets():
    """Cold /stats with no DB context: falls back to reset_chat_history."""
    helper = _make_helper_stats(None, session_id="session-new")
    helper._dispatch_before_create_session_prune = AsyncMock()

    await helper.get_conversation_stats(42)

    # create_session called because there was no saved context
    helper.db.create_session.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 1d: resolve_allowed_plugins must not reset a live session that has
#          user messages but no system message (auto_chat_modes=false)
# ---------------------------------------------------------------------------

def _make_helper_resolve():
    helper = object.__new__(OpenAIHelper)
    helper.conversations = {}
    helper.loaded_conversation_sessions = {}
    helper.config = {'max_sessions': 5}
    helper.plugin_manager = SimpleNamespace(
        dispatch_blocking=AsyncMock(),
        filter_allowed_plugins=MagicMock(side_effect=lambda x: x),
        disabled_plugins_for_user=MagicMock(return_value=set()),
    )
    helper.db = SimpleNamespace(
        create_session=MagicMock(return_value="session-new"),
        get_conversation_context=MagicMock(return_value=(
            {"messages": []}, "HTML", 0.1, 80, "session-new",
        )),
        get_mode_from_context=MagicMock(return_value={"role": "system", "content": "sys"}),
        save_conversation_context=MagicMock(),
        get_oldest_session_ids_for_limit=MagicMock(return_value=[]),
    )
    helper._save_conversation_context = AsyncMock()
    helper._dispatch_before_create_session_prune = AsyncMock()
    helper._mode_from_system_message = MagicMock(return_value=None)
    helper._apply_user_disabled_plugins = MagicMock(side_effect=lambda plugins, _uid: plugins)
    return helper


@pytest.mark.asyncio
async def test_resolve_allowed_plugins_no_system_but_has_user_skips_reset():
    """Session without system but with user messages: reset must NOT be called."""
    helper = _make_helper_resolve()
    messages = [{"role": "user", "content": "hi"}]
    helper.db.get_conversation_context = MagicMock(return_value=(
        {"messages": messages}, "HTML", 0.1, 80, "s1",
    ))

    result = await helper.resolve_allowed_plugins(99)

    helper.db.create_session.assert_not_called()
    assert result == ['All']


@pytest.mark.asyncio
async def test_resolve_allowed_plugins_empty_context_does_reset():
    """Session with no messages at all (empty/corrupt): reset MUST be called."""
    helper = _make_helper_resolve()
    sys_ctx = ({"messages": [{"role": "system", "content": "sys"}]}, "HTML", 0.1, 80, "s1")
    # Calls: (1) resolve start, (2) inside reset_chat_history, (3) resolve after reset
    helper.db.get_conversation_context = MagicMock(side_effect=[
        ({"messages": []}, "HTML", 0.1, 80, "s1"),
        sys_ctx,
        sys_ctx,
    ])

    await helper.resolve_allowed_plugins(99)

    helper.db.create_session.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 4g: ask() must not write to conversations when inside an active turn
# ---------------------------------------------------------------------------

def _make_helper_ask():
    helper = object.__new__(OpenAIHelper)
    helper.conversations = {}
    helper.loaded_conversation_sessions = {}
    helper.conversations_vision = {}
    helper.config = {'max_sessions': 5, 'light_model': 'test-model'}
    helper.plugin_manager = SimpleNamespace(dispatch_blocking=AsyncMock())
    helper.db = SimpleNamespace(
        create_session=MagicMock(return_value="s1"),
        get_conversation_context=MagicMock(return_value=(
            {"messages": []}, "HTML", 0.1, 80, "s1",
        )),
        get_mode_from_context=MagicMock(return_value={"role": "system", "content": "sys"}),
        save_conversation_context=MagicMock(),
        get_oldest_session_ids_for_limit=MagicMock(return_value=[]),
    )
    helper._save_conversation_context = AsyncMock()
    helper._dispatch_before_create_session_prune = AsyncMock()

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer", tool_calls=None, function_call=None))],
        usage=SimpleNamespace(total_tokens=5),
    )
    helper.chat_completion = AsyncMock(return_value=fake_response)
    helper.get_current_model = MagicMock(return_value="test-model")
    helper.get_max_tokens = MagicMock(return_value=100)
    # Stub __add_to_history so we can count calls
    helper._OpenAIHelper__add_to_history = AsyncMock()
    return helper


@pytest.mark.asyncio
async def test_ask_inside_active_turn_does_not_write_history():
    """ask() called with _CHAT_STATE_KEY set must not touch conversations."""
    from bot.openai_helper import _CHAT_STATE_KEY
    helper = _make_helper_ask()

    token = _CHAT_STATE_KEY.set("some-active-state-key")
    try:
        await helper.ask("test prompt", user_id=7)
    finally:
        _CHAT_STATE_KEY.reset(token)

    helper._OpenAIHelper__add_to_history.assert_not_called()
    # conversations dict should remain untouched
    assert 7 not in helper.conversations


@pytest.mark.asyncio
async def test_ask_outside_active_turn_writes_history_twice():
    """ask() with no active turn (_CHAT_STATE_KEY=None) writes user+assistant."""
    from bot.openai_helper import _CHAT_STATE_KEY
    helper = _make_helper_ask()

    # Ensure contextvar is clear
    assert _CHAT_STATE_KEY.get() is None

    await helper.ask("test prompt", user_id=7)

    assert helper._OpenAIHelper__add_to_history.await_count == 2
    calls = helper._OpenAIHelper__add_to_history.call_args_list
    assert calls[0].kwargs.get('role') == 'user' or calls[0].args[1] == 'user'
    assert calls[1].kwargs.get('role') == 'assistant' or calls[1].args[1] == 'assistant'
