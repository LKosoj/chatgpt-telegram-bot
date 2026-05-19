"""T2 — agent_tools.on_before_chat_request plan-rule mutator behaviour."""
from typing import List

import pytest

from bot.plugins.agent_tools import (
    AgentToolsPlugin,
    _PLAN_RULE_MARKER,
    _PLAN_RULE_TEXT,
)
from bot.plugins.hooks import BeforeChatRequestPayload


class FakeHelper:
    """Minimal helper that satisfies on_before_chat_request's needs."""

    def __init__(self, allowed):
        self._allowed = allowed
        self.calls: List[tuple] = []

    async def resolve_allowed_plugins(self, chat_id, session_id=None, user_id=None):
        self.calls.append((chat_id, session_id, user_id))
        return self._allowed


def _make_plugin(allowed) -> AgentToolsPlugin:
    plugin = AgentToolsPlugin()
    plugin.openai = FakeHelper(allowed)
    return plugin


def _payload(chat_id=1, user_id=42) -> BeforeChatRequestPayload:
    return BeforeChatRequestPayload(chat_id=chat_id, user_id=user_id, request_id=None)


async def test_rule_injected_when_agent_tools_allowed():
    plugin = _make_plugin(["agent_tools", "skills"])
    messages = [
        {"role": "system", "content": "mode prompt without the magic word"},
        {"role": "user", "content": "Hello"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is not None
    assert new is not messages  # immutable-style copy
    # mode prompt is still first
    assert new[0]["content"] == "mode prompt without the magic word"
    # rule sits right after the leading system cluster
    assert new[1]["role"] == "system"
    assert new[1]["content"].startswith(_PLAN_RULE_MARKER)
    assert "manage_plan_tasks" in new[1]["content"]
    assert _PLAN_RULE_TEXT in new[1]["content"]
    # user message preserved
    assert new[-1] == {"role": "user", "content": "Hello"}


async def test_rule_injected_when_resolver_returns_all():
    plugin = _make_plugin(["All"])
    messages = [
        {"role": "system", "content": "mode prompt"},
        {"role": "user", "content": "q"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is not None
    assert any(
        isinstance(m.get("content"), str)
        and m["content"].startswith(_PLAN_RULE_MARKER)
        for m in new
    )


async def test_rule_absent_when_agent_tools_not_allowed():
    plugin = _make_plugin(["skills", "ddg_web_search"])
    messages = [
        {"role": "system", "content": "mode prompt"},
        {"role": "user", "content": "q"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is None


async def test_rule_idempotent_marker():
    plugin = _make_plugin(["agent_tools"])
    messages = [
        {"role": "system", "content": "mode prompt"},
        {"role": "system", "content": _PLAN_RULE_MARKER + "cached rule body"},
        {"role": "user", "content": "q"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is None


async def test_rule_skipped_when_mode_prompt_mentions_manage_plan_tasks():
    plugin = _make_plugin(["agent_tools"])
    messages = [
        {
            "role": "system",
            "content": (
                "You are skills_agent. Always call agent_tools.manage_plan_tasks "
                "before doing anything non-trivial."
            ),
        },
        {"role": "user", "content": "q"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is None


async def test_rule_inserted_after_leading_system_cluster():
    plugin = _make_plugin(["agent_tools"])
    messages = [
        {"role": "system", "content": "first system"},
        {"role": "system", "content": "second system"},
        {"role": "system", "content": "third system"},
        {"role": "user", "content": "q"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is not None
    # Three original systems are preserved at the front, in order
    assert [m["content"] for m in new[:3]] == [
        "first system",
        "second system",
        "third system",
    ]
    # Rule lands immediately after the leading system cluster
    assert new[3]["role"] == "system"
    assert new[3]["content"].startswith(_PLAN_RULE_MARKER)
    # User message stays last
    assert new[-1] == {"role": "user", "content": "q"}


async def test_rule_noop_when_helper_missing():
    plugin = AgentToolsPlugin()
    plugin.openai = None
    messages = [{"role": "user", "content": "q"}]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is None
