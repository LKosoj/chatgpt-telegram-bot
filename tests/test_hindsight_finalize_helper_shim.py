from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _build_helper_with_plugin_present():
    """Build a minimal helper with _hindsight_plugin returning a mock plugin."""
    from bot.openai_helper import OpenAIHelper

    helper = OpenAIHelper.__new__(OpenAIHelper)
    helper.config = {}
    mock_plugin = SimpleNamespace(
        finalize_session_memory=AsyncMock(return_value=3),
    )
    helper._hindsight_plugin = lambda: mock_plugin
    helper.db = MagicMock()
    helper.db.get_conversation_context = MagicMock(
        return_value=({"messages": [{"role": "user", "content": "hi"}]}, None, None, None, None)
    )
    return helper, mock_plugin


async def test_helper_shim_returns_zero_when_no_plugin():
    from bot.openai_helper import OpenAIHelper
    helper = OpenAIHelper.__new__(OpenAIHelper)
    helper._hindsight_plugin = lambda: None
    result = await helper.finalize_hindsight_session_memory(1, "s")
    assert result == 0


async def test_helper_shim_returns_zero_when_no_session_id():
    helper, mock_plugin = _build_helper_with_plugin_present()
    result = await helper.finalize_hindsight_session_memory(1, "")
    assert result == 0
    mock_plugin.finalize_session_memory.assert_not_awaited()


async def test_helper_shim_passes_explicit_messages_through():
    helper, mock_plugin = _build_helper_with_plugin_present()
    msgs = [{"role": "user", "content": "x"}]
    result = await helper.finalize_hindsight_session_memory(42, "s1", messages=msgs)
    assert result == 3
    helper.db.get_conversation_context.assert_not_called()
    mock_plugin.finalize_session_memory.assert_awaited_once()
    call_args = mock_plugin.finalize_session_memory.call_args
    assert call_args.args[2] is msgs


async def test_helper_shim_reads_messages_from_db_when_missing():
    helper, mock_plugin = _build_helper_with_plugin_present()
    result = await helper.finalize_hindsight_session_memory(42, "s1")
    assert result == 3
    helper.db.get_conversation_context.assert_called_once()
    mock_plugin.finalize_session_memory.assert_awaited_once()
    call_args = mock_plugin.finalize_session_memory.call_args
    assert call_args.args[2] == [{"role": "user", "content": "hi"}]
