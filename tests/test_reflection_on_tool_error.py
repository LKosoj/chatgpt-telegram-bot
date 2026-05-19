"""Tests for the reflection-step (T3) injected after failed tool calls.

When one or more tools in a batch fail (returning either an error payload or a
validation error before execution), `handle_function_call` appends exactly one
short `user`-role reflection note to the conversation that names the failed
tools and instructs the model to articulate the failure before retrying.
The note must NOT appear when all tools succeed nor when a direct result
short-circuits the loop.
"""

import importlib.util
import json
import sys
import types
from types import SimpleNamespace

import pytest


_INSERTED_MODULES = []


def _install_module_if_missing(name, module):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = module
        _INSERTED_MODULES.append(name)


_markdown2 = types.ModuleType("markdown2")
_markdown2.markdown = lambda text, *args, **kwargs: text
_install_module_if_missing("markdown2", _markdown2)

from bot.openai_tool_handler import (  # noqa: E402
    REFLECTION_NOTE_PREFIX,
    handle_function_call,
)

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class FakeToolCall:
    def __init__(self, name, arguments, call_id=None):
        self.id = call_id or f"call_{name}"
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))


class FakeMessage:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls
        self.content = content


class FakeChoice:
    def __init__(self, tool_calls=None, content=""):
        self.message = FakeMessage(tool_calls=tool_calls, content=content)
        self.delta = None
        self.finish_reason = None


class FakeResponse:
    def __init__(self, tool_calls=None, content="done"):
        self.choices = [FakeChoice(tool_calls=tool_calls, content=content)]


class ScriptedPluginManager:
    """Returns pre-configured tool responses keyed by function name."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def filter_allowed_plugins(self, allowed_plugins):
        return allowed_plugins

    def is_function_allowed(self, function_name, allowed_plugins):
        return True

    async def call_function(self, function_name, helper, arguments, request_context=None):
        self.calls.append(function_name)
        payload = self.responses.get(function_name, {"result": "ok"})
        return json.dumps(payload, ensure_ascii=False)

    def get_functions_specs(self, helper, model_to_use, allowed_plugins):
        return []


class FakeDB:
    def list_user_sessions(self, user_id, is_active=1):
        return []


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        return FakeResponse(content="done")


class FakeHelper:
    def __init__(self, plugin_manager):
        self.plugin_manager = plugin_manager
        self.config = {"bot_language": "en", "functions_max_consecutive_calls": 1}
        self.db = FakeDB()
        self.client = FakeClient()
        self.conversations = {1234: []}
        self.history = []

    def get_current_model(self, user_id, session_id=None):
        return "gpt-test"

    async def chat_completion(self, **kwargs):
        return await self.client.chat.completions.create(**kwargs)

    def _add_function_call_to_history(self, chat_id, function_name, content):
        self.history.append((chat_id, function_name, content))

    async def _apply_before_chat_request_mutators(self, **kwargs):
        return []

    def get_max_tokens(self, model_to_use, max_tokens_percent, chat_id):
        return 100


def _reflection_messages(helper):
    return [
        m for m in helper.conversations[1234]
        if isinstance(m, dict)
        and m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].startswith(REFLECTION_NOTE_PREFIX)
    ]


async def _run(helper, tool_calls):
    return await handle_function_call(
        helper,
        chat_id=1234,
        response=FakeResponse(tool_calls=tool_calls),
        allowed_plugins=["All"],
        user_id=42,
    )


async def test_reflection_injected_on_failed_tool():
    pm = ScriptedPluginManager({"alpha.fail": {"error": "boom"}})
    helper = FakeHelper(pm)

    await _run(helper, [FakeToolCall("alpha.fail", {})])

    notes = _reflection_messages(helper)
    assert len(notes) == 1, helper.conversations[1234]
    content = notes[0]["content"]
    assert "alpha.fail" in content
    assert "следующим действием" in content
    assert len(content) <= 400


async def test_no_reflection_on_empty_success():
    pm = ScriptedPluginManager({"alpha.search": {"results": []}})
    helper = FakeHelper(pm)

    await _run(helper, [FakeToolCall("alpha.search", {})])

    assert _reflection_messages(helper) == []


async def test_no_reflection_on_direct_result():
    pm = ScriptedPluginManager({
        "alpha.deliver": {
            "direct_result": {
                "kind": "text",
                "format": "markdown",
                "value": "ok",
            }
        }
    })
    helper = FakeHelper(pm)

    await _run(helper, [FakeToolCall("alpha.deliver", {})])

    assert _reflection_messages(helper) == []


async def test_reflection_only_for_failed_in_batch():
    pm = ScriptedPluginManager({
        "alpha.ok1": {"result": "a"},
        "alpha.ok2": {"result": "b"},
        "alpha.ok3": {"result": "c"},
        "alpha.ok4": {"result": "d"},
        "alpha.broken": {"error": "kaboom"},
    })
    helper = FakeHelper(pm)

    await _run(
        helper,
        [
            FakeToolCall("alpha.ok1", {}),
            FakeToolCall("alpha.ok2", {}),
            FakeToolCall("alpha.broken", {}),
            FakeToolCall("alpha.ok3", {}),
            FakeToolCall("alpha.ok4", {}),
        ],
    )

    notes = _reflection_messages(helper)
    assert len(notes) == 1
    content = notes[0]["content"]
    assert "alpha.broken" in content
    for ok_name in ("alpha.ok1", "alpha.ok2", "alpha.ok3", "alpha.ok4"):
        assert ok_name not in content
