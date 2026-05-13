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
