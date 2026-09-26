"""Direct regression coverage for OpenAIHelper._persist_conversation_context
(bot/openai_helper.py:1270 area), the shared tail T12c extracted out of
_maybe_apply_auto_chat_mode, reset_chat_history, _add_to_history, and
record_plugin_exchange -- all four previously called
self._save_conversation_context(chat_id, {'messages': ...}, parse_mode,
temperature, max_tokens_percent, session_id) verbatim. T12c-review.md W2
flagged that no test exercised _persist_conversation_context directly, only
indirectly via the wider suite; this pins what each of the four call sites
saves and the exact positional args passed to _save_conversation_context, so
a future edit that reorders or drops an argument in the extracted helper (or
in one call site but not the others) fails here instead of surfacing as a
silent context-persistence bug.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tests.test_openai_helper_tool_calls import DummyPluginManager, FakeResponse, _make_helper


def _mock_save(helper):
    mock = AsyncMock(return_value=None)
    helper._save_conversation_context = mock
    return mock


@pytest.mark.asyncio
async def test_maybe_apply_auto_chat_mode_persists_new_system_message():
    helper = _make_helper(DummyPluginManager({}))
    helper.config["auto_chat_modes"] = True
    helper._build_auto_chat_mode_prompt = AsyncMock(return_value="prompt text")
    helper.chat_completion = AsyncMock(return_value=FakeResponse(content="somemode"))
    helper.chat_modes_registry.get_mode_by_key = lambda key: {"prompt_start": "System prompt for mode."}
    mock_save = _mock_save(helper)

    await helper._maybe_apply_auto_chat_mode(
        1, "hi",
        session_id="session-x", parse_mode="Markdown", temperature=0.5, max_tokens_percent=70,
        user_id=1,
    )

    assert helper.conversations[1] == [
        {"role": "system", "content": "System prompt for mode.", "mode_key": "somemode"}
    ]
    mock_save.assert_awaited_once_with(
        1, {"messages": helper.conversations[1]}, "Markdown", 0.5, 70, "session-x",
    )


@pytest.mark.asyncio
async def test_reset_chat_history_persists_fresh_system_message():
    helper = _make_helper(DummyPluginManager({}))
    mock_save = _mock_save(helper)

    await helper.reset_chat_history(1, content="", session_id="session-y")

    assert helper.conversations[1] == [{"role": "system", "content": ""}]
    # DummyDB.get_conversation_context always answers (None, 0.1, 80,
    # "session-1"); reset_chat_history discards its own 5th (session_id)
    # element and keeps the caller-supplied session_id instead.
    mock_save.assert_awaited_once_with(
        1, {"messages": helper.conversations[1]}, None, 0.1, 80, "session-y",
    )


@pytest.mark.asyncio
async def test_add_to_history_persists_appended_message():
    helper = _make_helper(DummyPluginManager({}))
    mock_save = _mock_save(helper)

    await helper._add_to_history(1, role="user", content="hello", session_id="session-z")

    assert helper.conversations[1] == [{"role": "user", "content": "hello"}]
    # Unlike reset_chat_history, _add_to_history keeps the DB-returned
    # session_id ("session-1"), not the one it was called with.
    mock_save.assert_awaited_once_with(
        1, {"messages": helper.conversations[1]}, None, 0.1, 80, "session-1",
    )


@pytest.mark.asyncio
async def test_record_plugin_exchange_persists_both_messages():
    helper = _make_helper(DummyPluginManager({}))
    mock_save = _mock_save(helper)

    await helper.record_plugin_exchange(1, "user text", "assistant text", session_id="session-w")

    assert helper.conversations[1] == [
        {"role": "user", "content": "user text"},
        {"role": "assistant", "content": "assistant text"},
    ]
    mock_save.assert_awaited_once_with(
        1, {"messages": helper.conversations[1]}, None, 0.1, 80, "session-1",
    )
