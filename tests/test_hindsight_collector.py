from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import InlineKeyboardButton

from bot.plugins.hindsight_memory import HindsightMemoryPlugin
from bot.plugins.hooks import SettingsMenuPayload, StatsBlockPayload


def _active_plugin():
    plugin = HindsightMemoryPlugin()
    plugin.initialize(
        plugin_config={
            'hindsight_base_url': 'http://x',
            'hindsight_api_token': 't',
        },
    )
    return plugin


def _inactive_plugin():
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={})
    return plugin


async def test_stats_block_returns_formatted_when_active():
    plugin = _active_plugin()
    plugin._memory_count_text = AsyncMock(return_value="42")
    result = await plugin.contribute_prompt_fragment(
        "stats_block",
        StatsBlockPayload(user_id=123, chat_id=999, bot_language="en"),
    )
    assert "Hindsight memory: enabled" in result
    assert "telegram-123" in result
    assert "42" in result


async def test_stats_block_handles_count_exception():
    plugin = _active_plugin()
    plugin._memory_count_text = AsyncMock(side_effect=RuntimeError("nope"))
    result = await plugin.contribute_prompt_fragment(
        "stats_block",
        StatsBlockPayload(user_id=123, chat_id=999, bot_language="en"),
    )
    assert "stats failed: nope" in result


async def test_stats_block_returns_none_when_inactive():
    plugin = _inactive_plugin()
    result = await plugin.contribute_prompt_fragment(
        "stats_block",
        StatsBlockPayload(user_id=123, chat_id=999, bot_language="en"),
    )
    assert result is None


async def test_stats_block_returns_none_without_user_id():
    plugin = _active_plugin()
    plugin._memory_count_text = AsyncMock(return_value="0")
    result = await plugin.contribute_prompt_fragment(
        "stats_block",
        StatsBlockPayload(user_id=None, chat_id=999, bot_language="en"),
    )
    assert result is None


async def test_settings_menu_buttons_returns_one_row_with_memory_button():
    plugin = _active_plugin()
    rows = await plugin.contribute_prompt_fragment(
        "settings_menu_buttons",
        SettingsMenuPayload(user_id=123, bot_language="en"),
    )
    assert isinstance(rows, list)
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, list) and len(row) == 1
    btn = row[0]
    assert isinstance(btn, InlineKeyboardButton)
    assert btn.text == "Hindsight memory"
    assert btn.callback_data == "memory:status"


async def test_settings_menu_buttons_returns_none_when_inactive():
    plugin = _inactive_plugin()
    rows = await plugin.contribute_prompt_fragment(
        "settings_menu_buttons",
        SettingsMenuPayload(user_id=123, bot_language="en"),
    )
    assert rows is None


async def test_unknown_slot_returns_none():
    plugin = _active_plugin()
    result = await plugin.contribute_prompt_fragment(
        "some_unknown_slot",
        SimpleNamespace(),
    )
    assert result is None
