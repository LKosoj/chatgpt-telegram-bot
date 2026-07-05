import json
from types import SimpleNamespace

import pytest

from bot.ai_events import AIToolCall
from bot.plugin_tool_adapter import PluginToolAdapter
from bot.request_context import RequestContext


class RecordingPluginManager:
    def __init__(self):
        self.calls = []
        self.spec_calls = []
        self.allowed_calls = []
        self.model_to_canonical = {"skills_run": "skills.run"}
        self.canonical_to_model = {"skills.run": "skills_run"}

    def get_functions_specs(self, helper, model_to_use, allowed_plugins):
        self.spec_calls.append((helper, model_to_use, allowed_plugins))
        return [{"type": "function", "function": {"name": "skills_run"}}]

    def to_model_function_name(self, function_name):
        return self.canonical_to_model.get(function_name, function_name)

    def to_canonical_function_name(self, function_name):
        return self.model_to_canonical.get(function_name, function_name)

    def is_function_allowed(self, function_name, allowed_plugins):
        self.allowed_calls.append((function_name, allowed_plugins))
        return allowed_plugins == ["skills"]

    async def call_function(self, function_name, helper, arguments, request_context=None):
        self.calls.append((function_name, helper, arguments, request_context))
        return json.dumps({"result": "ok"}, ensure_ascii=False)


def _adapter():
    plugin_manager = RecordingPluginManager()
    helper = SimpleNamespace(plugin_manager=plugin_manager)
    return PluginToolAdapter(helper, "llmgateway/high", ["skills"]), plugin_manager, helper


def test_plugin_tool_adapter_delegates_tool_specs():
    adapter, plugin_manager, helper = _adapter()

    assert adapter.get_tools() == [{"type": "function", "function": {"name": "skills_run"}}]
    assert plugin_manager.spec_calls == [(helper, "llmgateway/high", ["skills"])]


def test_plugin_tool_adapter_preserves_name_mapping_and_allowed_check():
    adapter, plugin_manager, _helper = _adapter()

    assert adapter.to_model_name("skills.run") == "skills_run"
    assert adapter.to_canonical_name("skills_run") == "skills.run"
    assert adapter.is_allowed("skills_run") is True
    assert plugin_manager.allowed_calls == [("skills.run", ["skills"])]


def test_plugin_tool_adapter_canonicalizes_model_tool_call_before_execution():
    adapter, _plugin_manager, _helper = _adapter()
    provider_tool_call = AIToolCall(
        id="call_1",
        name="skills_run",
        arguments='{"name":"pptx"}',
    )

    assert adapter.to_canonical_tool_call(provider_tool_call) == AIToolCall(
        id="call_1",
        name="skills.run",
        model_name="skills_run",
        arguments='{"name":"pptx"}',
    )


@pytest.mark.asyncio
async def test_plugin_tool_adapter_calls_plugin_manager_with_raw_arguments_and_context():
    adapter, plugin_manager, helper = _adapter()
    request_context = RequestContext(chat_id=10, user_id=20, message_id=30)
    tool_call = AIToolCall(
        id="call_1",
        name="skills_run",
        arguments='{"bad"',
    )

    response = await adapter.call(tool_call, request_context=request_context)

    assert json.loads(response) == {"result": "ok"}
    assert plugin_manager.calls == [
        ("skills.run", helper, '{"bad"', request_context),
    ]


def test_plugin_tool_adapter_normalizes_direct_result_response():
    result = PluginToolAdapter.normalize_response(
        {
            "direct_result": {
                "kind": "text",
                "value": "hello",
            }
        },
        tool_name="skills.run",
    )

    assert result.success is True
    assert result.direct_result == {"kind": "text", "value": "hello"}
