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

from bot.openai_helper import OpenAIHelper, default_max_tokens  # noqa: E402
from bot.request_context import RequestContext  # noqa: E402

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


class MultiSessionDB(DummyDB):
    def __init__(self, contexts):
        super().__init__()
        self.contexts = contexts

    def get_conversation_context(self, chat_id, session_id=None, *args, **kwargs):
        return self.contexts[session_id], None, 0.1, 80, session_id


class DummyPluginManager:
    def __init__(self, responses, specs=None, plugins=None):
        self.responses = responses
        self.specs = specs or []
        self.plugins = plugins or {}
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
        return self.specs

    def get_plugin_source_name(self, function_name):
        return function_name.split(".", 1)[0]

    def has_plugin(self, plugin_name):
        return True

    def get_plugin(self, plugin_name):
        return self.plugins.get(plugin_name)

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


class FakeSkillsPlugin:
    active_skills = {"chat:1": {"pptx": {}}}
    available_skills = {"pptx": {"scripts": ["build.py"]}}


class FakeToolCall:
    def __init__(self, name, arguments, id=None):
        self.id = id or f"call_{name.replace('.', '_')}"
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


class FakeStreamChoice:
    def __init__(self, content):
        self.delta = types.SimpleNamespace(content=content, tool_calls=None)
        self.finish_reason = None


class FakeStreamItem:
    def __init__(self, content):
        self.choices = [FakeStreamChoice(content)]


async def _fake_stream(contents):
    for content in contents:
        yield FakeStreamItem(content)


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
        "max_tokens": 1000,
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
        "vision_max_tokens": 1000,
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


def test_hindsight_memory_parser_preserves_full_fields():
    helper = _make_helper(DummyPluginManager({}))
    content = "memory-" + ("x" * 2500)
    context = "context-" + ("y" * 700)
    tag = "tag-" + ("z" * 120)

    items = helper._parse_hindsight_memory_items(json.dumps({
        "items": [{
            "content": content,
            "context": context,
            "tags": [tag],
        }]
    }))

    assert items == [{
        "content": content,
        "context": context,
        "tags": [tag],
    }]


def test_auto_chat_mode_prompt_routes_by_complexity_not_keywords():
    helper = _make_helper(DummyPluginManager({}))
    helper.chat_modes_registry = types.SimpleNamespace(
        get_all_modes_list=lambda: [
            "name: assistant, welcome_message: simple assistant",
            "name: skills_agent, welcome_message: complex agent",
        ],
    )

    prompt = helper._build_auto_chat_mode_prompt("Подготовь план, выполни шаги и проверь результат")

    assert "skills_agent" in prompt
    assert "Если задача простая" in prompt
    assert "Сложная задача" in prompt
    assert "Не выбирай skills_agent по отдельным словам" in prompt
    assert "Верни assistant" in prompt or "верни assistant" in prompt


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
async def test_stream_without_tool_calls_preserves_first_chunk():
    helper = _make_helper(DummyPluginManager({}))

    response, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1,
        response=_fake_stream(["Hel", "lo"]),
        stream=True,
        allowed_plugins=["All"],
        user_id=1,
    )

    chunks = []
    async for item in response:
        chunks.append(item.choices[0].delta.content)
    assert chunks == ["Hel", "lo"]
    assert tools_used == ()


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
    assert total_tokens is not None
    assert pm.spec_calls == [["weather"]]


@pytest.mark.asyncio
async def test_get_chat_response_rejects_empty_model_content():
    pm = DummyPluginManager({})
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content=None)]))

    with pytest.raises(ValueError, match="Модель вернула пустой ответ"):
        await helper.get_chat_response(
            chat_id=1,
            query="hello",
            user_id=1,
        )


@pytest.mark.asyncio
async def test_llmgateway_tool_results_use_structured_tool_history():
    pm = DummyPluginManager({
        "skills.get_skill_status": {"success": True, "skill": {"active": True}},
    })
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content="done")]))
    response = FakeResponse(tool_calls=[
        FakeToolCall("skills.get_skill_status", "{}", id="call-status"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert set(tools_used) == {"skills.get_skill_status"}
    assert out.choices[0].message.content == "done"
    assert helper.conversations[1][0] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call-status",
            "type": "function",
            "function": {
                "name": "skills.get_skill_status",
                "arguments": "{}",
            },
        }],
    }
    assert helper.conversations[1][1] == {
        "role": "tool",
        "tool_call_id": "call-status",
        "content": '{"success": true, "skill": {"active": true}}',
    }
    assert not any(
        isinstance(message.get("content"), str)
        and message["content"].startswith("Function skills.get_skill_status returned")
        for message in helper.conversations[1]
    )


@pytest.mark.asyncio
async def test_empty_response_after_tool_calls_is_retried_for_final_answer():
    tool_spec = {
        "type": "function",
        "function": {
            "name": "skills.list_skills",
            "description": "List skills",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    pm = DummyPluginManager(
        {"skills.list_skills": {"success": True, "skills": []}},
        specs=[tool_spec],
    )
    client = DummyClient([
        FakeResponse(tool_calls=[FakeToolCall("skills.list_skills", "{}")], content=None),
        FakeResponse(content=None),
        FakeResponse(content="final answer"),
    ])
    helper = _make_helper(pm, client=client)

    answer, total_tokens = await helper.get_chat_response(
        chat_id=1,
        query="use skills",
        user_id=1,
    )

    assert answer == "final answer"
    assert total_tokens == 3
    assert client.calls == 3
    assert pm.calls[0][0] == "skills.list_skills"
    assert "Предыдущий ответ был пустым" in client.create_kwargs[-1]["messages"][-1]["content"]
    assert client.create_kwargs[-1]["tools"] == [tool_spec]


@pytest.mark.asyncio
async def test_get_chat_response_strips_think_markers():
    pm = DummyPluginManager({})
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content="<think>hidden</think>visible</think>done")]))

    answer, total_tokens = await helper.get_chat_response(
        chat_id=1,
        query="hello",
        user_id=1,
    )

    assert answer == "visible\ndone"
    assert total_tokens == 3


@pytest.mark.asyncio
async def test_raw_tool_result_response_is_retried_instead_of_sent():
    tool_spec = {
        "type": "function",
        "function": {
            "name": "skills.get_skill_status",
            "description": "Get skill status",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    pm = DummyPluginManager(
        {"skills.get_skill_status": {"success": True, "skill": {"active": True}}},
        specs=[tool_spec],
    )
    client = DummyClient([
        FakeResponse(tool_calls=[FakeToolCall("skills.get_skill_status", "{}")], content=None),
        FakeResponse(content='Function skills.get_skill_status returned: {"success": true}'),
        FakeResponse(content="final answer"),
    ])
    helper = _make_helper(pm, client=client)

    answer, total_tokens = await helper.get_chat_response(
        chat_id=1,
        query="use skills",
        user_id=1,
    )

    assert answer == "final answer"
    assert total_tokens == 3
    assert client.calls == 3


@pytest.mark.asyncio
async def test_empty_response_before_tool_calls_is_retried_with_tools():
    tool_spec = {
        "type": "function",
        "function": {
            "name": "skills.list_skills",
            "description": "List skills",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    pm = DummyPluginManager(
        {"skills.list_skills": {"success": True, "skills": []}},
        specs=[tool_spec],
    )
    client = DummyClient([
        FakeResponse(content=None),
        FakeResponse(tool_calls=[FakeToolCall("skills.list_skills", "{}")], content=None),
        FakeResponse(content="final answer"),
    ])
    helper = _make_helper(pm, client=client)

    answer, total_tokens = await helper.get_chat_response(
        chat_id=1,
        query="use skills",
        user_id=1,
    )

    assert answer == "final answer"
    assert total_tokens == 3
    assert client.calls == 3
    assert pm.calls[0][0] == "skills.list_skills"
    assert "Предыдущий ответ был пустым" in client.create_kwargs[1]["messages"][-1]["content"]
    assert client.create_kwargs[1]["tools"] == [tool_spec]


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
async def test_agent_mode_defers_direct_result_and_continues_tool_loop():
    responses = {
        "stable_diffusion.stable_diffusion": {
            "direct_result": {
                "kind": "photo",
                "format": "path",
                "value": "/tmp/image.png",
                "add_value": "generated",
            },
        },
    }
    helper = _make_helper(
        DummyPluginManager(responses),
        client=DummyClient([FakeResponse(content="presentation ready")]),
    )
    helper.conversations[1] = [{"role": "system", "content": "agent-mode", "mode_key": "skills_agent"}]
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_key=lambda key: {"defer_direct_results": True} if key == "skills_agent" else None,
        get_mode_by_system_prompt=lambda _content: None,
    )
    response = FakeResponse(tool_calls=[
        FakeToolCall("stable_diffusion.stable_diffusion", "{}", id="call-image"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert helper.client.calls == 1
    assert set(tools_used) == {"stable_diffusion.stable_diffusion"}
    # Intermediate direct_result tools (image generation etc.) are deferred only
    # into the tool history so the model can reference them. They must not be
    # re-delivered alongside the final answer, otherwise the user receives helper
    # images before the actual artifact (e.g. images before the final pptx).
    assert out.choices[0].message.content == "presentation ready"
    tool_messages = [message for message in helper.conversations[1] if message.get("role") == "tool"]
    assert tool_messages
    assert "/tmp/image.png" in tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_new_session_reloads_mode_before_deferring_direct_results():
    contexts = {
        "old-session": {
            "messages": [
                {"role": "system", "content": "plain", "mode_key": "assistant"},
                {"role": "user", "content": "old request"},
            ],
        },
        "new-session": {
            "messages": [
                {"role": "system", "content": "agent-mode", "mode_key": "skills_agent"},
            ],
        },
    }
    responses = {
        "stable_diffusion.stable_diffusion": {
            "direct_result": {
                "kind": "photo",
                "format": "path",
                "value": "/tmp/image.png",
                "add_value": "generated",
            },
        },
    }
    helper = _make_helper(
        DummyPluginManager(responses),
        db=MultiSessionDB(contexts),
        client=DummyClient([
            FakeResponse(tool_calls=[
                FakeToolCall("stable_diffusion.stable_diffusion", "{}", id="call-image"),
            ]),
            FakeResponse(content="presentation ready"),
        ]),
    )
    helper.conversations[1] = contexts["old-session"]["messages"]
    helper.loaded_conversation_sessions[1] = "old-session"
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_key=lambda key: (
            {"defer_direct_results": True} if key == "skills_agent"
            else {"defer_direct_results": False} if key == "assistant"
            else None
        ),
        get_mode_by_system_prompt=lambda _content: None,
    )

    answer, total_tokens = await helper.get_chat_response(
        chat_id=1,
        query="создай презентацию",
        user_id=1,
        session_id="new-session",
    )

    assert answer == "presentation ready"
    assert total_tokens == 3
    assert helper.client.calls == 2
    assert helper.conversations[1][0]["mode_key"] == "skills_agent"
    assert helper.loaded_conversation_sessions[1] == "new-session"


@pytest.mark.asyncio
async def test_agent_mode_sends_final_direct_result_when_defer_is_false():
    responses = {
        "agent_tools.deliver_to_user": {
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "ready",
                "artifacts": [
                    {
                        "kind": "file",
                        "format": "path",
                        "value": "/tmp/silver_analysis.pptx",
                    }
                ],
                "defer": False,
            },
        },
    }
    helper = _make_helper(
        DummyPluginManager(responses),
        client=DummyClient([FakeResponse(content="should not be called")]),
    )
    helper.conversations[1] = [{"role": "system", "content": "agent-mode", "mode_key": "skills_agent"}]
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_key=lambda key: {"defer_direct_results": True} if key == "skills_agent" else None,
        get_mode_by_system_prompt=lambda _content: None,
    )
    response = FakeResponse(tool_calls=[
        FakeToolCall("agent_tools.deliver_to_user", "{}", id="call-artifact"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert helper.client.calls == 0
    assert set(tools_used) == {"agent_tools.deliver_to_user"}
    assert out["direct_result"]["text"] == "ready"
    assert out["direct_result"]["artifacts"][0]["value"] == "/tmp/silver_analysis.pptx"


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

    assert not hasattr(helper, "user_id")
    assert pm.call_contexts == [request_context]
    assert set(tools_used) == {"p1.do"}
    assert out["direct_result"]["value"] == "ok"
    sent_args = json.loads(pm.calls[0][1])
    assert sent_args["chat_id"] == 77
    assert sent_args["user_id"] == 42
    assert sent_args["message_id"] == 123


@pytest.mark.asyncio
async def test_get_chat_response_with_request_context_does_not_create_legacy_message_id_state():
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
    answer, total_tokens = await helper.get_chat_response(
        chat_id=999,
        query="hello",
        request_id="request-1",
        user_id=999,
        request_context=request_context,
    )

    assert answer == "done"
    assert total_tokens == 3
    assert not hasattr(helper, "message_id")
    assert not hasattr(helper, "message_ids")


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
        "task_management.create_task" in (message.get("content") or "")
        and "not allowed in the current chat mode" in (message.get("content") or "")
        for message in helper.conversations[1]
    )


@pytest.mark.asyncio
async def test_skills_agent_routes_skill_scripts_away_from_codeinterpreter():
    pm = DummyPluginManager({
        "codeinterpreter.deep_analysis": {"result": "should not run"},
    })
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content="use run_skill_script")]))
    helper.conversations[1] = [{"role": "system", "content": "agent-mode", "mode_key": "skills_agent"}]
    response = FakeResponse(tool_calls=[
        FakeToolCall(
            "codeinterpreter.deep_analysis",
            json.dumps({
                "code_prompt": (
                    "import subprocess\n"
                    "subprocess.run(['python3', '/srv/chatgpt-telegram-bot/bot/skills/pptx/scripts/build.py'])"
                ),
            }),
            id="call-code",
        ),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert pm.calls == []
    assert set(tools_used) == {"codeinterpreter.deep_analysis"}
    assert out.choices[0].message.content == "use run_skill_script"
    assert any(
        "skills.run_skill_script" in (message.get("content") or "")
        and "codeinterpreter.deep_analysis" in (message.get("content") or "")
        for message in helper.conversations[1]
    )


@pytest.mark.asyncio
async def test_skills_agent_routes_active_skill_script_names_away_from_codeinterpreter():
    pm = DummyPluginManager(
        {"codeinterpreter.deep_analysis": {"result": "should not run"}},
        plugins={"skills": FakeSkillsPlugin()},
    )
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content="use run_skill_script")]))
    helper.conversations[1] = [{"role": "system", "content": "agent-mode", "mode_key": "skills_agent"}]
    response = FakeResponse(tool_calls=[
        FakeToolCall(
            "codeinterpreter.deep_analysis",
            json.dumps({"code_prompt": "import subprocess\nsubprocess.run(['python3', 'build.py'])"}),
            id="call-code",
        ),
    ])

    await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert pm.calls == []
    assert any(
        "skills.run_skill_script" in (message.get("content") or "")
        for message in helper.conversations[1]
    )


@pytest.mark.asyncio
async def test_skills_agent_rejects_ad_hoc_tmp_script_creation_via_codeinterpreter():
    pm = DummyPluginManager({"codeinterpreter.deep_analysis": {"result": "should not run"}})
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content="use skill scripts")]))
    helper.conversations[1] = [{"role": "system", "content": "agent-mode", "mode_key": "skills_agent"}]
    response = FakeResponse(tool_calls=[
        FakeToolCall(
            "codeinterpreter.deep_analysis",
            json.dumps({
                "code_prompt": (
                    "Write the following JavaScript code to the file /tmp/create_silver_pptx.js. "
                    "Use fs.writeFileSync to write it."
                ),
            }),
            id="call-code",
        ),
    ])

    await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert pm.calls == []
    assert any(
        "ad-hoc script files" in (message.get("content") or "")
        and "agent_tools.deliver_to_user" in (message.get("content") or "")
        for message in helper.conversations[1]
    )


@pytest.mark.asyncio
async def test_active_skill_script_reference_blocks_codeinterpreter_in_any_mode():
    pm = DummyPluginManager(
        {"codeinterpreter.deep_analysis": {"result": "should not run"}},
        plugins={"skills": FakeSkillsPlugin()},
    )
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content="use run_skill_script")]))
    helper.conversations[1] = [{"role": "system", "content": "assistant-mode"}]
    response = FakeResponse(tool_calls=[
        FakeToolCall(
            "codeinterpreter.deep_analysis",
            json.dumps({"code_prompt": "import subprocess; subprocess.run(['python3', 'build.py'])"}),
            id="call-code",
        ),
    ])

    await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert pm.calls == []
    assert any(
        "skills.run_skill_script" in (message.get("content") or "")
        and '"script_name": "build.py"' in (message.get("content") or "")
        and "suggested_tool_call" in (message.get("content") or "")
        for message in helper.conversations[1]
    )


@pytest.mark.asyncio
async def test_ordinary_mode_merges_multiple_direct_results_into_final():
    responses = {
        "stable_diffusion.first": {
            "direct_result": {"kind": "photo", "format": "path", "value": "/tmp/a.png"},
        },
        "stable_diffusion.second": {
            "direct_result": {"kind": "photo", "format": "path", "value": "/tmp/b.png"},
        },
    }
    helper = _make_helper(DummyPluginManager(responses))
    response = FakeResponse(tool_calls=[
        FakeToolCall("stable_diffusion.first", "{}", id="call-a"),
        FakeToolCall("stable_diffusion.second", "{}", id="call-b"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert helper.client.calls == 0
    assert set(tools_used) == {"stable_diffusion.first", "stable_diffusion.second"}
    assert out["direct_result"]["kind"] == "final"
    artifact_values = [a["value"] for a in out["direct_result"]["artifacts"]]
    assert artifact_values == ["/tmp/a.png", "/tmp/b.png"]


@pytest.mark.asyncio
async def test_deliver_to_user_carries_cleanup_directive_in_payload():
    responses = {
        "agent_tools.deliver_to_user": {
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "ready",
                "artifacts": [{"kind": "file", "format": "path", "value": "/tmp/out.pptx"}],
                "defer": False,
                "cleanup_skills": [
                    {"plugin_id": "skills", "scope": "chat:1", "skill_id": "pptx"},
                ],
            },
        },
    }
    helper = _make_helper(
        DummyPluginManager(responses),
        client=DummyClient([FakeResponse(content="should not be called")]),
    )
    helper.conversations[1] = [{"role": "system", "content": "agent-mode", "mode_key": "skills_agent"}]
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_key=lambda key: {"defer_direct_results": True} if key == "skills_agent" else None,
        get_mode_by_system_prompt=lambda _content: None,
    )
    response = FakeResponse(tool_calls=[
        FakeToolCall("agent_tools.deliver_to_user", "{}", id="call-final"),
    ])

    out, _tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert out["direct_result"]["cleanup_skills"][0]["skill_id"] == "pptx"
    assert out["direct_result"]["cleanup_skills"][0]["scope"] == "chat:1"


@pytest.mark.asyncio
async def test_active_skill_does_not_block_unrelated_codeinterpreter_calls():
    pm = DummyPluginManager(
        {"codeinterpreter.deep_analysis": {"result": "42"}},
        plugins={"skills": FakeSkillsPlugin()},
    )
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content="calculated")]))
    helper.conversations[1] = [{"role": "system", "content": "agent-mode", "mode_key": "skills_agent"}]
    response = FakeResponse(tool_calls=[
        FakeToolCall(
            "codeinterpreter.deep_analysis",
            json.dumps({"code_prompt": "print(6 * 7)"}),
            id="call-code",
        ),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert pm.calls and pm.calls[0][0] == "codeinterpreter.deep_analysis"
    assert set(tools_used) == {"codeinterpreter.deep_analysis"}
    assert out.choices[0].message.content == "calculated"
