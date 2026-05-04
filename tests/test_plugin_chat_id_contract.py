import json
import importlib.util
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

from bot.openai_tool_handler import handle_function_call  # noqa: E402
from bot.plugin_manager import PluginManager  # noqa: E402
from bot.plugins.conversation_analytics import ConversationAnalyticsPlugin  # noqa: E402
from bot.request_context import RequestContext  # noqa: E402
from bot.validation import validate_function_args  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class FakeToolCall:
    def __init__(self, name, arguments):
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
    def __init__(self, tool_name=None, arguments=None, content="done"):
        tool_calls = None
        if tool_name is not None:
            tool_calls = [FakeToolCall(tool_name, arguments or {})]
        self.choices = [FakeChoice(tool_calls=tool_calls, content=content)]


class RecordingPluginManager:
    def __init__(self):
        self.calls = []

    def filter_allowed_plugins(self, allowed_plugins):
        return allowed_plugins

    def is_function_allowed(self, function_name, allowed_plugins):
        return True

    async def call_function(self, function_name, helper, arguments, request_context=None):
        self.calls.append((function_name, json.loads(arguments), request_context))
        return json.dumps({
            "direct_result": {
                "kind": "text",
                "format": "markdown",
                "value": "ok",
            }
        })


class ValidatingAnalyticsPluginManager:
    def __init__(self, plugin):
        self.plugin = plugin
        self.calls = []
        self.plugin_calls = 0

    def filter_allowed_plugins(self, allowed_plugins):
        return allowed_plugins

    def is_function_allowed(self, function_name, allowed_plugins):
        return True

    async def call_function(self, function_name, helper, arguments, request_context=None):
        parsed_args = json.loads(arguments)
        self.calls.append(parsed_args)
        spec = self.plugin.get_spec()[0]
        errors = validate_function_args(spec, parsed_args)
        if errors:
            return json.dumps({"error": f"Invalid args for {function_name}: {errors}"})

        self.plugin_calls += 1
        result = await self.plugin.execute("analyze_conversation", helper, **parsed_args)
        return json.dumps(result, default=str, ensure_ascii=False)

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

    def get_current_model(self, user_id):
        return "gpt-test"

    def _add_function_call_to_history(self, chat_id, function_name, content):
        self.history.append((chat_id, function_name, content))

    def _messages_with_hindsight_context(self, chat_id):
        return []

    def get_max_tokens(self, model_to_use, max_tokens_percent, chat_id):
        return 100

    def _localized_text(self, key, bot_language):
        return key


@pytest.mark.asyncio
async def test_tool_injected_chat_id_is_plugin_string():
    plugin_manager = RecordingPluginManager()
    helper = FakeHelper(plugin_manager)
    request_context = RequestContext(chat_id=1234, user_id=42, message_id=99)

    await handle_function_call(
        helper,
        chat_id=1234,
        response=FakeResponse("text_document_qa.list_documents", {}),
        allowed_plugins=["All"],
        user_id=42,
        request_context=request_context,
    )

    function_name, arguments, call_context = plugin_manager.calls[0]
    assert function_name == "text_document_qa.list_documents"
    assert arguments["chat_id"] == "1234"
    assert isinstance(arguments["chat_id"], str)
    assert arguments["user_id"] == 42
    assert arguments["message_id"] == 99
    assert call_context is request_context


@pytest.mark.xfail(
    strict=True,
    reason="legacy tool flow still injects chat_id as int without RequestContext",
)
@pytest.mark.asyncio
async def test_legacy_tool_injected_chat_id_is_plugin_string():
    plugin_manager = RecordingPluginManager()
    helper = FakeHelper(plugin_manager)

    await handle_function_call(
        helper,
        chat_id=1234,
        response=FakeResponse("text_document_qa.list_documents", {}),
        allowed_plugins=["All"],
        user_id=42,
    )

    _function_name, arguments, _call_context = plugin_manager.calls[0]
    assert arguments["chat_id"] == "1234"
    assert isinstance(arguments["chat_id"], str)


@pytest.mark.asyncio
async def test_conversation_analytics_tool_call_uses_string_chat_id(tmp_path):
    plugin = ConversationAnalyticsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    plugin.update_stats("1234", {"text": "hello", "tokens": 3, "user_id": 42})
    plugin_manager = ValidatingAnalyticsPluginManager(plugin)
    helper = FakeHelper(plugin_manager)

    response, _tools_used = await handle_function_call(
        helper,
        chat_id=1234,
        response=FakeResponse(
            "conversation_analytics.analyze_conversation",
            {"time_period": "day", "analysis_type": "usage"},
        ),
        allowed_plugins=["All"],
        user_id=42,
        request_context=RequestContext(chat_id=1234, user_id=42),
    )

    assert isinstance(response, FakeResponse)
    assert plugin_manager.calls[0]["chat_id"] == "1234"
    assert plugin_manager.plugin_calls == 1
    assert not any("Invalid args" in item[2] for item in helper.history)


@pytest.mark.xfail(
    strict=True,
    reason="conversation_analytics still advertises injected chat_id as model-required",
)
def test_model_visible_required_args_exclude_internal_chat_id(tmp_path):
    plugin_manager = object.__new__(PluginManager)
    plugin_manager.plugins = {"conversation_analytics": ConversationAnalyticsPlugin}
    plugin_manager.plugin_instances = {}
    plugin_manager.openai = None
    plugin_manager.bot = None
    plugin_manager.storage_root = str(tmp_path)
    plugin_manager.strict_validation = False

    specs = plugin_manager.get_functions_specs(
        helper=None,
        model_to_use="gpt-test",
        allowed_plugins=["conversation_analytics"],
    )
    analyze_spec = next(
        item["function"]
        for item in specs
        if item["function"]["name"] == "conversation_analytics.analyze_conversation"
    )

    required = analyze_spec["parameters"].get("required", [])
    assert "chat_id" not in required
    assert "user_id" not in required
    assert "message_id" not in required
