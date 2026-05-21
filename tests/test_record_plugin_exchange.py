"""Unit tests for ``OpenAIHelper.record_plugin_exchange``.

Covers the session-mismatch path: when the in-memory ``conversations`` cache
holds messages from a *different* session than the one passed in, the method
must reload from the DB before appending — otherwise the new pair gets
written under the wrong session.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("tiktoken")

from bot.openai_helper import OpenAIHelper


def _make_helper(db_context):
    helper = object.__new__(OpenAIHelper)
    helper.conversations = {}
    helper.conversations_vision = {}
    helper.loaded_conversation_sessions = {}
    helper.last_updated = {}
    helper.config = {'max_conversation_age_minutes': 60}
    helper._background_tasks = set()
    helper._closed = False
    helper.db = SimpleNamespace(
        get_conversation_context=MagicMock(return_value=db_context),
        save_conversation_context=MagicMock(return_value=None),
        save_conversation_context_async=AsyncMock(return_value=None),
    )
    return helper


@pytest.mark.asyncio
async def test_record_plugin_exchange_appends_pair_and_saves_once():
    helper = _make_helper((
        {"messages": [{"role": "system", "content": "sys"}]},
        "HTML", 0.8, 80, "session-A",
    ))

    await helper.record_plugin_exchange(
        chat_id=42, user_text="q", assistant_text="a", session_id="session-A",
    )

    assert helper.db.save_conversation_context_async.await_count == 1
    saved_args = helper.db.save_conversation_context_async.await_args.args
    saved_messages = saved_args[1]["messages"]
    assert saved_messages[-2:] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert helper.loaded_conversation_sessions[42] == "session-A"


@pytest.mark.asyncio
async def test_record_plugin_exchange_reloads_on_session_mismatch():
    """If the cache holds session-A messages and the call targets session-B,
    the method must reload session-B's history before appending — otherwise
    session-A content leaks into session-B's persisted context.
    """
    helper = _make_helper((
        {"messages": [{"role": "system", "content": "sys-B"}]},
        "HTML", 0.8, 80, "session-B",
    ))
    helper.conversations[42] = [
        {"role": "system", "content": "sys-A"},
        {"role": "user", "content": "old-A-question"},
        {"role": "assistant", "content": "old-A-answer"},
    ]
    helper.loaded_conversation_sessions[42] = "session-A"
    helper.last_updated[42] = datetime.datetime.now()

    await helper.record_plugin_exchange(
        chat_id=42, user_text="q-B", assistant_text="a-B", session_id="session-B",
    )

    saved_messages = helper.db.save_conversation_context_async.await_args.args[1]["messages"]
    assert all(msg.get("content") != "old-A-question" for msg in saved_messages)
    assert all(msg.get("content") != "old-A-answer" for msg in saved_messages)
    assert saved_messages == [
        {"role": "system", "content": "sys-B"},
        {"role": "user", "content": "q-B"},
        {"role": "assistant", "content": "a-B"},
    ]
    assert helper.loaded_conversation_sessions[42] == "session-B"


@pytest.mark.asyncio
async def test_record_plugin_exchange_bumps_last_updated():
    helper = _make_helper((
        {"messages": []}, "HTML", 0.8, 80, "session-A",
    ))

    assert 42 not in helper.last_updated
    await helper.record_plugin_exchange(
        chat_id=42, user_text="q", assistant_text="a", session_id="session-A",
    )
    assert isinstance(helper.last_updated[42], datetime.datetime)
