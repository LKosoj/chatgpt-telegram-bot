import importlib.util
import json
import sys
import types

import pytest

_INSERTED_MODULES = []


def _install_module_if_missing(name, module):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = module
        _INSERTED_MODULES.append(name)


class _FakeEncoding:
    def encode(self, value):
        return list(value)


_tiktoken = types.ModuleType("tiktoken")
_tiktoken.encoding_for_model = lambda _model: _FakeEncoding()
_tiktoken.get_encoding = lambda _name: _FakeEncoding()
_install_module_if_missing("tiktoken", _tiktoken)

_markdown2 = types.ModuleType("markdown2")
_markdown2.markdown = lambda text, *args, **kwargs: text
_install_module_if_missing("markdown2", _markdown2)


def _retry(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


_tenacity = types.ModuleType("tenacity")
_tenacity.retry = _retry
_tenacity.stop_after_attempt = lambda *args, **kwargs: None
_tenacity.wait_fixed = lambda *args, **kwargs: None
_tenacity.retry_if_exception_type = lambda *args, **kwargs: None
_install_module_if_missing("tenacity", _tenacity)

from bot.openai_helper import OpenAIHelper, default_max_tokens
from bot.request_context import RequestContext

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class DummyDB:
    def __init__(self, context=None):
        self.context = context or {}
        self.saved_contexts = []

    def list_user_sessions(self, user_id, is_active=1):
        return []

    def get_conversation_context(self, *args, **kwargs):
        return self.context, None, 0.1, 80, "session-1"

    def save_conversation_context(self, *args, **kwargs):
        self.saved_contexts.append((args, kwargs))


class DummyPluginManager:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.call_contexts = []
        self.spec_calls = []
        self.filtered = []

    def filter_allowed_plugins(self, allowed_plugins):
        self.filtered.append(list(allowed_plugins or []))
        return allowed_plugins

    async def call_function(self, name, helper, arguments, request_context=None):
        self.calls.append((name, arguments))
        self.call_contexts.append(request_context)
        return self.responses[name]

    def get_functions_specs(self, helper, model_to_use, allowed_plugins):
        self.spec_calls.append(list(allowed_plugins or []))
        return []

    def get_plugin_source_name(self, function_name):
        return function_name.split(".", 1)[0]

    def has_plugin(self, plugin_name):
        return True

    def is_function_allowed(self, function_name, allowed_plugins):
        if allowed_plugins == ["All"]:
            return True
        if not allowed_plugins or allowed_plugins == ["None"]:
            return False
        plugin_name = function_name.split(".", 1)[0]
        return plugin_name in allowed_plugins


class DummyClient:
    def __init__(self, responses=None):
        self.calls = 0
        self.create_kwargs = []
        self.responses = list(responses or [])
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        self.create_kwargs.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse(tool_calls=None, content="done")


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = types.SimpleNamespace(name=name, arguments=arguments)


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
    def __init__(self, tool_calls=None, content=""):
        self.choices = [FakeChoice(tool_calls=tool_calls, content=content)]
        self.usage = types.SimpleNamespace(
            total_tokens=3,
            prompt_tokens=1,
            completion_tokens=2,
        )


def _make_helper(plugin_manager, db=None, client=None):
    config = {
        "openai_base": "",
        "api_key": "test",
        "show_usage": False,
        "stream": False,
        "proxy": None,
        "proxy_web": None,
        "max_history_size": 5,
        "max_conversation_age_minutes": 60,
        "assistant_prompt": "hi",
        "max_tokens": 100,
        "n_choices": 1,
        "temperature": 0.1,
        "image_model": "llmgateway/ai-klein-generation",
        "image_quality": "standard",
        "image_style": "vivid",
        "image_size": "512x512",
        "auto_chat_modes": False,
        "model": "llmgateway/high",
        "enable_functions": True,
        "functions_max_consecutive_calls": 2,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "bot_language": "en",
        "show_plugins_used": False,
        "whisper_prompt": "",
        "vision_model": "llmgateway/big_context",
        "enable_vision_follow_up_questions": True,
        "vision_prompt": "test",
        "vision_detail": "auto",
        "vision_max_tokens": 300,
        "tts_model": "llmgateway/silero-tts",
        "tts_voice": "kseniya",
        "tts_response_format": "wav",
        "transcription_model": "llmgateway/whisper-large-v3",
        "yandex_api_token": "",
        "assemblyai_api_key": "",
        "big_model_to_use": "llmgateway/big_context",
        "light_model": "llmgateway/light_model",
    }
    helper = OpenAIHelper(config=config, plugin_manager=plugin_manager, db=db or DummyDB())
    helper.client = client or DummyClient()
    helper.conversations[1] = []
    return helper


def test_high_model_default_context_is_256k():
    assert default_max_tokens("llmgateway/high") == 256_000


def test_vision_history_content_keeps_only_text():
    content = [
        {"type": "text", "text": "что на этой картинке?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
    ]

    assert OpenAIHelper._vision_history_content(content) == "что на этой картинке?"


def test_resolve_allowed_plugins_returns_mode_tools():
    saved_context = {
        "messages": [
            {"role": "system", "content": "weather-only"},
        ],
    }
    pm = DummyPluginManager({})
    helper = _make_helper(pm, db=DummyDB(saved_context))
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_system_prompt=lambda _content: {"tools": ["weather"]},
    )

    assert helper.resolve_allowed_plugins(chat_id=1, session_id="session-1") == ["weather"]
    assert pm.filtered[-1] == ["weather"]


def test_resolve_allowed_plugins_defaults_to_all_without_mode_tools():
    saved_context = {
        "messages": [
            {"role": "system", "content": "plain"},
        ],
    }
    pm = DummyPluginManager({})
    helper = _make_helper(pm, db=DummyDB(saved_context))
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_system_prompt=lambda _content: None,
    )

    assert helper.resolve_allowed_plugins(chat_id=1, session_id="session-1") == ["All"]
    assert pm.filtered[-1] == ["All"]


@pytest.mark.asyncio
async def test_initial_model_request_uses_resolved_allowed_plugins(monkeypatch):
    pm = DummyPluginManager({})
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content="done")]))
    monkeypatch.setattr(
        helper,
        "resolve_allowed_plugins",
        lambda chat_id, session_id=None: ["weather"],
    )

    answer, total_tokens = await helper.get_chat_response(
        chat_id=1,
        query="hello",
        user_id=1,
    )

    assert answer == "done"
    assert total_tokens == 3
    assert pm.spec_calls == [["weather"]]


@pytest.mark.asyncio
async def test_parallel_tool_calls_no_direct_result():
    responses = {
        "p1.do": {"result": "ok1"},
        "p2.do": {"result": "ok2"},
    }
    helper = _make_helper(DummyPluginManager(responses))
    response = FakeResponse(tool_calls=[
        FakeToolCall("p1.do", "{}"),
        FakeToolCall("p2.do", "{}"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert helper.client.calls == 1
    assert set(tools_used) == {"p1.do", "p2.do"}
    assert out.choices[0].message.tool_calls is None


@pytest.mark.asyncio
async def test_parallel_tool_calls_direct_result_short_circuit():
    responses = {
        "p1.do": {"direct_result": {"kind": "text", "format": "markdown", "value": "ok"}},
        "p2.do": {"result": "ok2"},
    }
    helper = _make_helper(DummyPluginManager(responses))
    response = FakeResponse(tool_calls=[
        FakeToolCall("p1.do", "{}"),
        FakeToolCall("p2.do", "{}"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert helper.client.calls == 0
    assert set(tools_used) == {"p1.do", "p2.do"}
    assert out["direct_result"]["value"] == "ok"


@pytest.mark.asyncio
async def test_legacy_tool_request_in_content_is_executed():
    responses = {
        "p1.do": {"result": "ok1"},
    }
    pm = DummyPluginManager(responses)
    helper = _make_helper(pm)
    response = FakeResponse(
        tool_calls=None,
        content='```json\n{"tool_name":"p1.do","x":1}\n```',
    )

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert helper.client.calls == 1
    assert set(tools_used) == {"p1.do"}
    assert pm.calls and pm.calls[0][0] == "p1.do"
    sent_args = json.loads(pm.calls[0][1])
    assert sent_args["x"] == 1
    assert sent_args["chat_id"] == 1
    assert sent_args["user_id"] == 1


@pytest.mark.asyncio
async def test_allowed_tool_reentry_uses_original_allowlist():
    responses = {
        "weather.get_weather": {"result": "sunny"},
    }
    pm = DummyPluginManager(responses)
    helper = _make_helper(pm)
    response = FakeResponse(tool_calls=[
        FakeToolCall("weather.get_weather", "{}"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["weather"], user_id=1
    )

    assert helper.client.calls == 1
    assert set(tools_used) == {"weather.get_weather"}
    assert out.choices[0].message.tool_calls is None
    assert pm.spec_calls == [["weather"]]


@pytest.mark.asyncio
async def test_all_allowlist_allows_any_tool_call():
    responses = {
        "task_management.create_task": {"result": "created"},
    }
    pm = DummyPluginManager(responses)
    helper = _make_helper(pm)
    response = FakeResponse(tool_calls=[
        FakeToolCall("task_management.create_task", "{}"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert helper.client.calls == 1
    assert set(tools_used) == {"task_management.create_task"}
    assert out.choices[0].message.tool_calls is None
    assert pm.calls and pm.calls[0][0] == "task_management.create_task"


@pytest.mark.asyncio
async def test_request_context_tool_flow_injects_context_without_shared_user_id():
    request_context = RequestContext(
        chat_id=77,
        user_id=42,
        message_id=123,
        session_id="session-1",
    )
    pm = DummyPluginManager({
        "p1.do": {"direct_result": {"kind": "text", "format": "markdown", "value": "ok"}},
    })
    helper = _make_helper(pm)
    helper.conversations[request_context.chat_id] = []
    helper.user_id = "legacy-user"
    response = FakeResponse(tool_calls=[
        FakeToolCall("p1.do", "{}"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=999,
        response=response,
        stream=False,
        allowed_plugins=["All"],
        user_id=999,
        request_context=request_context,
    )

    assert helper.user_id == "legacy-user"
    assert pm.call_contexts == [request_context]
    assert set(tools_used) == {"p1.do"}
    assert out["direct_result"]["value"] == "ok"
    sent_args = json.loads(pm.calls[0][1])
    assert sent_args["chat_id"] == "77"
    assert sent_args["user_id"] == 42
    assert sent_args["message_id"] == 123


@pytest.mark.asyncio
async def test_get_chat_response_with_request_context_keeps_legacy_message_id_state():
    request_context = RequestContext(
        chat_id=77,
        user_id=42,
        message_id=123,
        session_id="session-1",
    )
    pm = DummyPluginManager({})
    helper = _make_helper(
        pm,
        db=DummyDB({"messages": [{"role": "system", "content": "hi"}]}),
        client=DummyClient([FakeResponse(content="done")]),
    )
    helper.message_ids["request-1"] = 999
    helper.message_id = "legacy-message"

    answer, total_tokens = await helper.get_chat_response(
        chat_id=999,
        query="hello",
        request_id="request-1",
        user_id=999,
        request_context=request_context,
    )

    assert answer == "done"
    assert total_tokens == 3
    assert helper.message_id == "legacy-message"


@pytest.mark.asyncio
async def test_mode_allowlist_is_passed_to_tool_reentry():
    saved_context = {
        "messages": [
            {"role": "system", "content": "weather-only"},
        ],
    }
    pm = DummyPluginManager({
        "weather.get_weather": {"result": "sunny"},
    })
    first_response = FakeResponse(tool_calls=[
        FakeToolCall("weather.get_weather", "{}"),
    ])
    final_response = FakeResponse(tool_calls=None, content="done")
    helper = _make_helper(
        pm,
        db=DummyDB(saved_context),
        client=DummyClient([first_response, final_response]),
    )
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_system_prompt=lambda _content: {"tools": ["weather"]},
    )

    answer, total_tokens = await helper.get_chat_response(
        chat_id=1,
        query="weather",
        user_id=1,
    )

    assert answer == "done"
    assert total_tokens == 3
    assert pm.calls and pm.calls[0][0] == "weather.get_weather"
    assert pm.spec_calls == [["weather"], ["weather"]]


@pytest.mark.asyncio
async def test_mode_restrictions_survive_tool_reentry():
    saved_context = {
        "messages": [
            {"role": "system", "content": "weather-only"},
        ],
    }
    pm = DummyPluginManager({
        "task_management.create_task": {"result": "created"},
    })
    first_response = FakeResponse(tool_calls=[
        FakeToolCall("task_management.create_task", "{}"),
    ])
    final_response = FakeResponse(tool_calls=None, content="done")
    helper = _make_helper(
        pm,
        db=DummyDB(saved_context),
        client=DummyClient([first_response, final_response]),
    )
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_system_prompt=lambda _content: {"tools": ["weather"]},
    )

    await helper.get_chat_response(
        chat_id=1,
        query="create a task",
        user_id=1,
    )

    assert pm.calls == []
    assert pm.spec_calls
    assert all(call == ["weather"] for call in pm.spec_calls)
    assert any(
        "task_management.create_task" in message.get("content", "")
        and "not allowed in the current chat mode" in message.get("content", "")
        for message in helper.conversations[1]
    )


@pytest.mark.asyncio
async def test_legacy_tool_request_outside_allowlist_is_rejected():
    pm = DummyPluginManager({
        "task_management.create_task": {"result": "created"},
    })
    helper = _make_helper(pm)
    response = FakeResponse(
        tool_calls=None,
        content='```json\n{"tool_name":"task_management.create_task","title":"x"}\n```',
    )

    _out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["weather"], user_id=1
    )

    assert pm.calls == []
    assert "task_management.create_task" in tools_used
    assert any(
        "task_management.create_task" in message.get("content", "")
        and "not allowed in the current chat mode" in message.get("content", "")
        for message in helper.conversations[1]
    )
