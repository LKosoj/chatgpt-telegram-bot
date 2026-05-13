"""Regression tests for SkillsPlugin.contribute_prompt_fragment and the
auto_mode_priority collector slot in OpenAIHelper._build_auto_chat_mode_prompt."""

from __future__ import annotations

import types
from typing import Any, Dict

import pytest

from bot.plugins.hooks import PromptFragmentPayload
from bot.plugins.skills import SkillsPlugin


def _make_skills_plugin_with(available: Dict[str, Dict[str, Any]]) -> SkillsPlugin:
    plugin = SkillsPlugin()
    plugin.available_skills = available
    return plugin


@pytest.mark.asyncio
async def test_contribute_returns_none_for_unknown_slot():
    plugin = _make_skills_plugin_with({
        "fiction_writer": {"description": "Writes fiction stories."},
    })
    payload = PromptFragmentPayload(
        slot="something_else", chat_id=1, user_id=None, query="q"
    )

    assert await plugin.contribute_prompt_fragment("something_else", payload) is None


@pytest.mark.asyncio
async def test_contribute_returns_none_when_no_skills_installed():
    plugin = _make_skills_plugin_with({})
    payload = PromptFragmentPayload(
        slot="auto_mode_priority", chat_id=1, user_id=None, query="q"
    )

    assert await plugin.contribute_prompt_fragment("auto_mode_priority", payload) is None


@pytest.mark.asyncio
async def test_contribute_emits_priority_block_with_summary_lines():
    plugin = _make_skills_plugin_with({
        "fiction_writer": {"description": "Writes fiction stories and prose."},
        "code_reviewer": {"description": "Reviews code for bugs and style."},
    })
    payload = PromptFragmentPayload(
        slot="auto_mode_priority", chat_id=1, user_id=None, query="q"
    )

    fragment = await plugin.contribute_prompt_fragment("auto_mode_priority", payload)

    assert fragment is not None
    assert "ПРИОРИТЕТНОЕ ПРАВИЛО" in fragment
    assert "skills_agent" in fragment
    assert "- fiction_writer: Writes fiction stories and prose." in fragment
    assert "- code_reviewer: Reviews code for bugs and style." in fragment


@pytest.mark.asyncio
async def test_contribute_truncates_long_descriptions():
    long = "x" * 500
    plugin = _make_skills_plugin_with({
        "verbose_skill": {"description": long},
    })
    payload = PromptFragmentPayload(
        slot="auto_mode_priority", chat_id=1, user_id=None, query="q"
    )

    fragment = await plugin.contribute_prompt_fragment("auto_mode_priority", payload)

    assert fragment is not None
    assert "..." in fragment
    assert long not in fragment


@pytest.mark.asyncio
async def test_contribute_falls_back_to_id_when_description_empty():
    plugin = _make_skills_plugin_with({
        "no_desc_skill": {"description": ""},
    })
    payload = PromptFragmentPayload(
        slot="auto_mode_priority", chat_id=1, user_id=None, query="q"
    )

    fragment = await plugin.contribute_prompt_fragment("auto_mode_priority", payload)

    assert fragment is not None
    assert "- no_desc_skill" in fragment
    assert "- no_desc_skill: " not in fragment


@pytest.mark.asyncio
async def test_build_auto_chat_mode_prompt_includes_fragments_from_collector():
    """End-to-end: helper.collect_fragments → priority_block in prompt."""
    from tests.test_openai_helper_tool_calls import DummyPluginManager, _make_helper

    pm = DummyPluginManager({})

    async def fake_collect_fragments(slot, payload, *, user_id=None):
        if slot == "auto_mode_priority":
            return ["FRAGMENT_FROM_PLUGIN_A", "FRAGMENT_FROM_PLUGIN_B"]
        return []

    pm.collect_fragments = fake_collect_fragments

    helper = _make_helper(pm)
    helper.chat_modes_registry = types.SimpleNamespace(
        get_all_modes_list=lambda: ["name: assistant"]
    )

    prompt = await helper._build_auto_chat_mode_prompt(
        "test query", chat_id=1, user_id=None
    )

    assert "FRAGMENT_FROM_PLUGIN_A" in prompt
    assert "FRAGMENT_FROM_PLUGIN_B" in prompt
    assert "Остальные правила выбора" in prompt


@pytest.mark.asyncio
async def test_build_auto_chat_mode_prompt_works_without_fragments():
    """No fragments → no priority_block, no crash."""
    from tests.test_openai_helper_tool_calls import DummyPluginManager, _make_helper

    pm = DummyPluginManager({})
    helper = _make_helper(pm)
    helper.chat_modes_registry = types.SimpleNamespace(
        get_all_modes_list=lambda: ["name: assistant"]
    )

    prompt = await helper._build_auto_chat_mode_prompt(
        "test query", chat_id=1, user_id=None
    )

    assert "ПРИОРИТЕТНОЕ ПРАВИЛО" not in prompt
    assert "Остальные правила выбора" in prompt
