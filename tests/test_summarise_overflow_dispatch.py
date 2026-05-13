"""Tests for the pre-dispatch helpers used by summarise-overflow paths.

Stage 4C-3+5: before ``reset_chat_history`` wipes an in-memory conversation
during summarise-overflow, ``_dispatch_before_summarise_reset`` must fire
``on_session_before_delete`` so Hindsight can snapshot the outgoing window.
Similarly, ``_dispatch_before_create_session_prune`` fires the hook for any
old sessions that ``create_session`` is about to prune from SQLite.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("tiktoken")

from bot.openai_helper import OpenAIHelper
from bot.plugins.hooks import HookEvent, SessionBeforeDeletePayload


def _make_helper():
    helper = object.__new__(OpenAIHelper)
    helper.conversations = {}
    helper.config = {'max_sessions': 5}
    helper.plugin_manager = SimpleNamespace(dispatch_blocking=AsyncMock())
    helper.db = SimpleNamespace(
        get_oldest_session_ids_for_limit=MagicMock(return_value=[]),
        get_conversation_context=MagicMock(),
    )
    return helper


@pytest.mark.asyncio
async def test_dispatch_before_summarise_reset_fires_with_in_memory_snapshot():
    helper = _make_helper()
    helper.conversations[100] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    await helper._dispatch_before_summarise_reset(chat_id=100, user_id=42, session_id="s-x")

    helper.plugin_manager.dispatch_blocking.assert_awaited_once()
    args, kwargs = helper.plugin_manager.dispatch_blocking.await_args
    assert args[0] == HookEvent.ON_SESSION_BEFORE_DELETE
    payload = args[1]
    assert isinstance(payload, SessionBeforeDeletePayload)
    assert payload.user_id == 42
    assert payload.session_id == "s-x"
    assert list(payload.messages) == helper.conversations[100]
    assert kwargs == {"user_id": 42}


@pytest.mark.asyncio
async def test_dispatch_before_summarise_reset_skips_when_no_messages():
    helper = _make_helper()
    helper.conversations[100] = []

    await helper._dispatch_before_summarise_reset(chat_id=100, user_id=42, session_id="s-x")

    helper.plugin_manager.dispatch_blocking.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_before_summarise_reset_falls_back_to_chat_id():
    helper = _make_helper()
    helper.conversations[100] = [{"role": "user", "content": "x"}]

    # user_id=None must fall back to chat_id for the payload + dispatcher kwarg.
    await helper._dispatch_before_summarise_reset(chat_id=100, user_id=None, session_id=None)

    args, kwargs = helper.plugin_manager.dispatch_blocking.await_args
    payload = args[1]
    assert payload.user_id == 100
    assert payload.session_id == ""
    assert kwargs == {"user_id": 100}


@pytest.mark.asyncio
async def test_dispatch_before_summarise_reset_swallows_subscriber_failure(caplog):
    helper = _make_helper()
    helper.conversations[100] = [{"role": "user", "content": "x"}]
    helper.plugin_manager.dispatch_blocking.side_effect = RuntimeError("boom")

    with caplog.at_level("ERROR"):
        # Must not raise — summarisation must continue regardless.
        await helper._dispatch_before_summarise_reset(chat_id=100, user_id=42, session_id=None)


@pytest.mark.asyncio
async def test_dispatch_before_create_session_prune_fires_per_oldest_session():
    helper = _make_helper()
    helper.db.get_oldest_session_ids_for_limit.return_value = ["old-1", "old-2"]
    helper.db.get_conversation_context.side_effect = [
        ({"messages": [{"role": "user", "content": "a"}]}, "HTML", 0.1, 80, "old-1"),
        ({"messages": [{"role": "user", "content": "b"}]}, "HTML", 0.1, 80, "old-2"),
    ]

    await helper._dispatch_before_create_session_prune(42)

    assert helper.plugin_manager.dispatch_blocking.await_count == 2
    sessions = [call.args[1].session_id for call in helper.plugin_manager.dispatch_blocking.await_args_list]
    assert sessions == ["old-1", "old-2"]


@pytest.mark.asyncio
async def test_dispatch_before_create_session_prune_skips_when_no_old_sessions():
    helper = _make_helper()
    helper.db.get_oldest_session_ids_for_limit.return_value = []

    await helper._dispatch_before_create_session_prune(42)

    helper.plugin_manager.dispatch_blocking.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_before_create_session_prune_continues_on_per_session_failure(caplog):
    helper = _make_helper()
    helper.db.get_oldest_session_ids_for_limit.return_value = ["bad", "good"]

    def _ctx(user_id, session_id):
        if session_id == "bad":
            raise RuntimeError("missing")
        return ({"messages": [{"role": "user", "content": "g"}]}, "HTML", 0.1, 80, "good")

    helper.db.get_conversation_context.side_effect = _ctx

    with caplog.at_level("ERROR"):
        await helper._dispatch_before_create_session_prune(42)

    # Bad session is skipped, good session still dispatches.
    helper.plugin_manager.dispatch_blocking.assert_awaited_once()
    assert helper.plugin_manager.dispatch_blocking.await_args.args[1].session_id == "good"
