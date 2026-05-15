import asyncio
import base64
import io
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

from bot.openai_helper import EMPTY_MODEL_RESPONSE_ERROR, OpenAIHelper, default_max_tokens  # noqa: E402
from bot.openai_tool_handler import (  # noqa: E402
    _call_function_bounded,
    _filter_tools_by_name,
    _has_tool_specs,
    handle_function_call,
)
from bot.i18n import reset_current_language, set_current_language  # noqa: E402
from bot.plugins.agent_tools import AgentToolsPlugin  # noqa: E402
from bot.request_context import RequestContext  # noqa: E402
from bot.user_settings import (  # noqa: E402
    USER_DISABLED_PLUGINS_SETTING,
    get_user_settings,
    normalize_string_list,
)

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class DummyDB:
    def __init__(self, context=None):
        self.context = context or {}
        self.saved_contexts = []
        self.user_settings = {}

    def list_user_sessions(self, user_id, is_active=1):
        return []

    def get_conversation_context(self, *args, **kwargs):
        return self.context, None, 0.1, 80, "session-1"

    def save_conversation_context(self, *args, **kwargs):
        self.saved_contexts.append((args, kwargs))

    def get_user_settings(self, user_id):
        return self.user_settings.get(user_id)


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
        self.db = None

    def set_db(self, db):
        self.db = db

    def disabled_plugins_for_user(self, user_id):
        if self.db is None or user_id is None:
            return set()
        settings = get_user_settings(self.db, user_id)
        return set(normalize_string_list(settings.get(USER_DISABLED_PLUGINS_SETTING)))

    def is_plugin_disabled_for_user(self, plugin_name, user_id):
        return bool(plugin_name) and plugin_name in self.disabled_plugins_for_user(user_id)

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

    async def apply_mutators(self, event_name, payload, value, *, user_id=None):
        return value

    async def collect_fragments(self, slot, payload, *, user_id=None):
        return []


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


class DummyModelsClient(DummyClient):
    def __init__(self, models):
        super().__init__()
        self.models = types.SimpleNamespace(list=self._list_models)
        self._models = models

    async def _list_models(self):
        return types.SimpleNamespace(data=[types.SimpleNamespace(id=model) for model in self._models])


class DummySpeechResponse:
    def read(self):
        return b"speech-bytes"


class DummySpeechClient(DummyClient):
    def __init__(self):
        super().__init__()
        self.speech_kwargs = []
        self.audio = types.SimpleNamespace(speech=types.SimpleNamespace(create=self._create_speech))

    async def _create_speech(self, **kwargs):
        self.speech_kwargs.append(kwargs)
        return DummySpeechResponse()


class DummyVoiceGateway:
    def __init__(self, voices):
        self.voices = voices
        self.calls = []

    async def audio_voices(self, model=None):
        self.calls.append(model)
        return [{"voice": voice} for voice in self.voices]


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


async def _collect_stream_content(response):
    content = ""
    async for item in response:
        if item.choices and item.choices[0].delta.content:
            content += item.choices[0].delta.content
    return content


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
    helper_db = db or DummyDB()
    helper = OpenAIHelper(config=config, plugin_manager=plugin_manager, db=helper_db)
    plugin_manager.set_db(helper_db)
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


@pytest.mark.asyncio
async def test_vision_follow_up_keeps_image_content_and_uses_vision_model():
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    client = DummyClient([
        FakeResponse(content="image answer"),
        FakeResponse(content="follow up"),
    ])
    helper = _make_helper(DummyPluginManager({}), client=client)

    answer, _tokens = await helper.interpret_image(1, io.BytesIO(png_1x1), prompt="what is shown?")
    assert answer == "image answer"
    assert helper.conversations_vision[1] is True
    assert isinstance(helper.conversations[1][0]["content"], list)
    assert any(
        item.get("type") == "image_url"
        for item in helper.conversations[1][0]["content"]
    )

    follow_up, _tokens = await helper.get_chat_response(1, "what color?", user_id=1)

    assert follow_up == "follow up"
    assert client.create_kwargs[-1]["model"] == "llmgateway/big_context"
    assert any(
        isinstance(message.get("content"), list)
        for message in client.create_kwargs[-1]["messages"]
    )


@pytest.mark.asyncio
async def test_interpret_image_retries_empty_vision_response_without_duplicate_history():
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    client = DummyClient([
        FakeResponse(content=""),
        FakeResponse(content=""),
        FakeResponse(content="image answer"),
    ])
    helper = _make_helper(DummyPluginManager({}), client=client)

    answer, _tokens = await helper.interpret_image(1, io.BytesIO(png_1x1), prompt="what is shown?", user_id=1)

    assert answer == "image answer"
    assert client.calls == 3
    assert sum(1 for message in helper.conversations[1] if message["role"] == "user") == 1


@pytest.mark.asyncio
async def test_interpret_image_handles_vision_tool_calls():
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    tool_spec = {
        "type": "function",
        "function": {
            "name": "skills.list_skills",
            "description": "List skills",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    pm = DummyPluginManager(
        {"skills.list_skills": {"skills": ["design"]}},
        specs=[tool_spec],
    )
    client = DummyClient([
        FakeResponse(tool_calls=[FakeToolCall("skills.list_skills", "{}")]),
        FakeResponse(content="Use the design skill."),
    ])
    helper = _make_helper(pm, client=client)

    answer, _tokens = await helper.interpret_image(1, io.BytesIO(png_1x1), prompt="what is shown?", user_id=1)

    assert answer == "Use the design skill."
    assert pm.calls == [("skills.list_skills", json.dumps({"chat_id": 1, "user_id": 1}))]
    assert client.create_kwargs[0]["tools"] == [tool_spec]
    assert client.create_kwargs[0]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_interpret_image_retries_plain_text_tool_intent():
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    stuck_text = (
        "Я вижу, вы хотите создать дизайн в светлых тонах.\n\n"
        "Давайте посмотрим, какие навыки у меня есть для работы с изображениями."
    )
    tool_spec = {
        "type": "function",
        "function": {
            "name": "skills.list_skills",
            "description": "List skills",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    client = DummyClient([
        FakeResponse(content=stuck_text),
        FakeResponse(content="Сформирую дизайн напрямую."),
    ])
    helper = _make_helper(DummyPluginManager({}, specs=[tool_spec]), client=client)

    answer, _tokens = await helper.interpret_image(1, io.BytesIO(png_1x1), prompt="create design", user_id=1)

    assert answer == "Сформирую дизайн напрямую."
    assert client.calls == 2
    assert client.create_kwargs[0]["tools"] == [tool_spec]
    assert client.create_kwargs[1]["tools"] == [tool_spec]
    assert "Undelivered assistant text" in helper.conversations[1][-2]["content"]
    assert stuck_text in helper.conversations[1][-2]["content"]


@pytest.mark.asyncio
async def test_vision_follow_up_can_use_tools():
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    tool_spec = {
        "type": "function",
        "function": {
            "name": "skills.list_skills",
            "description": "List skills",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    pm = DummyPluginManager(
        {"skills.list_skills": {"skills": ["design"]}},
        specs=[tool_spec],
    )
    client = DummyClient([
        FakeResponse(content="image answer"),
        FakeResponse(tool_calls=[FakeToolCall("skills.list_skills", "{}")]),
        FakeResponse(content="Use the design skill."),
    ])
    helper = _make_helper(pm, client=client)

    await helper.interpret_image(1, io.BytesIO(png_1x1), prompt="what is shown?", user_id=1)
    follow_up, _tokens = await helper.get_chat_response(1, "what skills can help?", user_id=1)

    assert follow_up == "Use the design skill."
    assert pm.calls == [("skills.list_skills", json.dumps({"chat_id": 1, "user_id": 1}))]
    assert client.create_kwargs[1]["model"] == "llmgateway/big_context"
    assert client.create_kwargs[1]["tools"] == [tool_spec]
    assert client.create_kwargs[1]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_edit_telegram_image_uses_configured_image_model():
    class FakeGateway:
        def __init__(self):
            self.calls = []

        async def image_edit_file(self, prompt, image_bytes, **kwargs):
            self.calls.append((prompt, image_bytes, kwargs))
            return {"data": [{"url": "https://example.com/edited.png"}]}

    helper = object.__new__(OpenAIHelper)
    helper.config = {
        "bot_language": "en",
        "image_model": "llmgateway/ai-klein-generation",
    }
    helper.gateway_client = FakeGateway()

    async def fake_download_file_as_bytes(file_id):
        assert file_id == "telegram-file-id"
        return b"image-bytes"

    helper.download_file_as_bytes = fake_download_file_as_bytes

    result = await OpenAIHelper.edit_telegram_image(helper, "add a hat", "telegram-file-id")

    assert result == ("https://example.com/edited.png", "url")
    [(prompt, image_bytes, kwargs)] = helper.gateway_client.calls
    assert prompt == "add a hat"
    assert image_bytes == b"image-bytes"
    assert kwargs["model"] == "llmgateway/ai-klein-generation"


def test_hindsight_memory_parser_preserves_full_fields():
    from bot.plugins.hindsight_memory import HindsightMemoryPlugin

    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={})
    content = "memory-" + ("x" * 2500)
    context = "context-" + ("y" * 700)
    tag = "tag-" + ("z" * 120)

    items = plugin._parse_hindsight_memory_items(json.dumps({
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


@pytest.mark.asyncio
async def test_auto_chat_mode_prompt_routes_by_complexity_not_keywords():
    helper = _make_helper(DummyPluginManager({}))
    helper.chat_modes_registry = types.SimpleNamespace(
        get_all_modes_list=lambda: [
            "name: assistant, welcome_message: simple assistant",
            "name: skills_agent, welcome_message: complex agent",
        ],
    )

    prompt = await helper._build_auto_chat_mode_prompt(
        "Подготовь план, выполни шаги и проверь результат",
        chat_id=1,
        user_id=None,
    )

    assert "skills_agent" in prompt
    assert "Если задача простая" in prompt
    assert "Сложная задача" in prompt
    assert "больше двух шагов" in prompt
    assert "Не выбирай skills_agent по отдельным словам" in prompt
    assert "Верни assistant" in prompt or "верни assistant" in prompt


@pytest.mark.asyncio
async def test_resolve_allowed_plugins_returns_mode_tools():
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

    assert await helper.resolve_allowed_plugins(chat_id=1, session_id="session-1") == ["weather"]
    assert pm.filtered[-1] == ["weather"]


@pytest.mark.asyncio
async def test_resolve_allowed_plugins_defaults_to_all_without_mode_tools():
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

    assert await helper.resolve_allowed_plugins(chat_id=1, session_id="session-1") == ["All"]
    assert pm.filtered[-1] == ["All"]


@pytest.mark.asyncio
async def test_resolve_allowed_plugins_removes_user_disabled_plugins():
    saved_context = {
        "messages": [
            {"role": "system", "content": "tools"},
        ],
    }
    db = DummyDB(saved_context)
    db.user_settings[42] = {"disabled_plugins": ["weather"]}
    pm = DummyPluginManager({})
    helper = _make_helper(pm, db=db)
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_system_prompt=lambda _content: {"tools": ["weather", "time"]},
    )

    assert await helper.resolve_allowed_plugins(chat_id=1, session_id="session-1", user_id=42) == ["time"]
    assert pm.filtered[-1] == ["time"]


@pytest.mark.asyncio
async def test_chat_request_tool_specs_use_request_user_for_disabled_plugins():
    saved_context = {
        "messages": [
            {"role": "system", "content": "tools"},
        ],
    }
    db = DummyDB(saved_context)
    db.user_settings[42] = {"disabled_plugins": ["weather"]}
    pm = DummyPluginManager(
        {},
        specs=[{
            "name": "time.now",
            "description": "x",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }],
    )
    client = DummyClient([FakeResponse(content="done")])
    helper = _make_helper(pm, db=db, client=client)
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_system_prompt=lambda _content: {"tools": ["weather", "time"]},
    )

    await helper._OpenAIHelper__common_get_chat_response(
        chat_id=1,
        query="hello",
        session_id="session-1",
        user_id=42,
    )

    assert pm.spec_calls[-1] == ["time"]


@pytest.mark.asyncio
async def test_subagent_parent_allowed_plugins_are_resolved_with_user_id():
    calls = []

    async def resolve_allowed_plugins(chat_id, session_id=None, user_id=None):
        calls.append((chat_id, session_id, user_id))
        return ["terminal"]

    helper = types.SimpleNamespace(resolve_allowed_plugins=resolve_allowed_plugins)
    request_context = RequestContext(
        chat_id=10,
        user_id=42,
        session_id="session-1",
    )

    allowed = await AgentToolsPlugin._resolve_parent_allowed_plugins(
        helper,
        request_context,
        {"chat_id": 99, "user_id": 99},
    )

    assert allowed == ["terminal"]
    assert calls == [(10, "session-1", 42)]


@pytest.mark.asyncio
async def test_tts_options_are_loaded_from_api():
    helper = _make_helper(DummyPluginManager({}), client=DummyModelsClient([
        "llmgateway/high",
        "llmgateway/silero-tts",
        "tts-1",
    ]))
    helper.gateway_client = DummyVoiceGateway(["alice", "bob"])

    assert await helper.get_available_tts_models() == ["llmgateway/silero-tts", "tts-1"]
    assert await helper.get_available_tts_voices("llmgateway/silero-tts") == ["alice", "bob"]
    assert helper.gateway_client.calls[0] == "llmgateway/silero-tts"


@pytest.mark.asyncio
async def test_generate_speech_uses_user_tts_settings():
    db = DummyDB()
    db.user_settings[42] = {"tts_model": "tts-1", "tts_voice": "bob"}
    client = DummySpeechClient()
    helper = _make_helper(DummyPluginManager({}), db=db, client=client)

    speech_file, text_length = await helper.generate_speech("hello", user_id=42)

    assert speech_file.getvalue() == b"speech-bytes"
    assert text_length == 5
    assert client.speech_kwargs[0]["model"] == "tts-1"
    assert client.speech_kwargs[0]["voice"] == "bob"


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

    async def _resolver(chat_id, session_id=None, user_id=None):
        return ["weather"]

    monkeypatch.setattr(helper, "resolve_allowed_plugins", _resolver)

    answer, total_tokens = await helper.get_chat_response(
        chat_id=1,
        query="hello",
        user_id=1,
    )

    assert answer == "done"
    assert total_tokens is not None
    assert pm.spec_calls == [["weather"]]


@pytest.mark.asyncio
async def test_tool_execution_allows_nested_chat_response_with_same_chat_lock():
    class NestedPluginManager(DummyPluginManager):
        async def call_function(self, name, helper, arguments, request_context=None):
            return await helper.get_chat_response(chat_id=1, query="nested", user_id=1)

    helper = _make_helper(
        NestedPluginManager({}),
        client=DummyClient([FakeResponse(content="nested done")]),
    )
    lock = await helper._chat_lock(1)

    async with lock:
        result = await asyncio.wait_for(
            _call_function_bounded(
                helper,
                "prompt_perfect.optimize_prompt",
                json.dumps({"chat_id": 1}),
                None,
                asyncio.Semaphore(1),
            ),
            timeout=1,
        )

    assert result == ("nested done", 3)


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
async def test_streaming_empty_response_is_not_saved_as_assistant_turn():
    helper = _make_helper(
        DummyPluginManager({}),
        client=DummyClient([_fake_stream([])]),
    )

    chunks = []
    async for content, _tokens in helper.get_chat_response_stream(
        chat_id=1,
        query="hello",
        user_id=1,
    ):
        chunks.append(content)

    assert chunks == [f"Error generating response: {EMPTY_MODEL_RESPONSE_ERROR}"]
    assert not any(
        message.get("role") == "assistant" and message.get("content") == ""
        for message in helper.conversations[1]
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


def test_repair_tool_call_history_emits_synthetic_tool_result():
    helper = _make_helper(DummyPluginManager({}))
    helper.conversations[1] = [
        {"role": "system", "content": "agent-mode", "mode_key": "skills_agent"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "chief_get_recipe:2",
                "type": "function",
                "function": {
                    "name": "chief.get_recipe",
                    "arguments": "{}",
                },
            }],
        },
        {"role": "user", "content": "continue"},
    ]

    helper._repair_tool_call_history(1)
    messages = helper._messages_with_language_instruction(helper.conversations[1])

    synthetic_tool_result = helper.conversations[1][2]
    assert synthetic_tool_result["role"] == "tool"
    assert synthetic_tool_result["tool_call_id"] == "chief_get_recipe:2"
    assert helper.conversations[1][3] == {"role": "user", "content": "continue"}

    outbound_assistant_index = next(
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    )
    outbound_tool_result = messages[outbound_assistant_index + 1]
    assert outbound_tool_result["role"] == "tool"
    assert outbound_tool_result["tool_call_id"] == "chief_get_recipe:2"
    assert "Tool result missing" in json.loads(outbound_tool_result["content"])["error"]


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
async def test_prompt_perfect_suppresses_itself_on_reentry():
    tool_spec = {
        "type": "function",
        "function": {
            "name": "prompt_perfect.optimize_prompt",
            "description": "Rewrite prompt",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    pm = DummyPluginManager(
        {
            "prompt_perfect.optimize_prompt": {
                "optimized_prompt": "write a detailed guide",
                "instruction": "Use optimized_prompt as the request.",
                "suppress_reentry_tools": ["prompt_perfect.optimize_prompt"],
            }
        },
        specs=[tool_spec],
    )
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content="final answer")]))
    response = FakeResponse(tool_calls=[
        FakeToolCall("prompt_perfect.optimize_prompt", json.dumps({"original_prompt": "guide"})),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert out.choices[0].message.content == "final answer"
    assert tools_used == ("prompt_perfect.optimize_prompt",)
    assert helper.client.create_kwargs[0]["tools"] == []
    assert helper.client.create_kwargs[0]["tool_choice"] == "none"
    assert len(pm.calls) == 1


@pytest.mark.asyncio
async def test_prompt_perfect_retries_plain_text_tool_intent_without_tools():
    stuck_text = (
        "Я вижу, вы хотите, чтобы я повторил запрос: создать дизайн в светлых "
        "тонах для плана бревенчатой бани (сруб).\n\n"
        "Давайте посмотрим, какие навыки у меня есть для работы с изображениями "
        "и дизайном."
    )
    tool_spec = {
        "type": "function",
        "function": {
            "name": "prompt_perfect.optimize_prompt",
            "description": "Rewrite prompt",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    pm = DummyPluginManager(
        {
            "prompt_perfect.optimize_prompt": {
                "optimized_prompt": "create a light-toned bathhouse design project",
                "instruction": "Use optimized_prompt as the request.",
                "suppress_reentry_tools": ["prompt_perfect.optimize_prompt"],
                "retry_plain_text_tool_intent": True,
            }
        },
        specs=[tool_spec],
    )
    helper = _make_helper(
        pm,
        client=DummyClient([
            FakeResponse(content=stuck_text),
            FakeResponse(content="Сформирую дизайн-проект напрямую."),
        ]),
    )
    response = FakeResponse(tool_calls=[
        FakeToolCall("prompt_perfect.optimize_prompt", json.dumps({"original_prompt": "guide"})),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert out.choices[0].message.content == "Сформирую дизайн-проект напрямую."
    assert tools_used == ("prompt_perfect.optimize_prompt",)
    assert helper.client.calls == 2
    assert helper.client.create_kwargs[0]["tools"] == []
    assert helper.client.create_kwargs[0]["tool_choice"] == "none"
    assert helper.client.create_kwargs[1]["tools"] == []
    assert helper.client.create_kwargs[1]["tool_choice"] == "none"
    assert len(pm.calls) == 1
    assert stuck_text != out.choices[0].message.content
    assert "Undelivered assistant text" in helper.conversations[1][-1]["content"]
    assert stuck_text in helper.conversations[1][-1]["content"]


@pytest.mark.asyncio
async def test_prompt_perfect_retries_streamed_plain_text_tool_intent():
    helper = _make_helper(
        DummyPluginManager({}, specs=[]),
        client=DummyClient([_fake_stream(["Сформирую ", "дизайн-проект напрямую."])]),
    )

    response, tools_used = await handle_function_call(
        helper,
        chat_id=1,
        response=_fake_stream([
            "Я вижу, что вам нужен дизайн-проект. ",
            "Позвольте мне загрузить подходящий skill.",
        ]),
        stream=True,
        times=1,
        tools_used=("prompt_perfect.optimize_prompt",),
        allowed_plugins=["All"],
        user_id=1,
        suppressed_reentry_tools={"prompt_perfect.optimize_prompt"},
        retry_plain_text_tool_intent=True,
    )

    assert await _collect_stream_content(response) == "Сформирую дизайн-проект напрямую."
    assert tools_used == ("prompt_perfect.optimize_prompt",)
    assert helper.client.calls == 1
    assert helper.client.create_kwargs[0]["stream"] is True
    assert "Undelivered assistant text" in helper.conversations[1][-1]["content"]


@pytest.mark.asyncio
async def test_prompt_perfect_suppression_survives_delivery_repair():
    responses = {
        "prompt_perfect.optimize_prompt": {
            "optimized_prompt": "write a detailed guide",
            "instruction": "Use optimized_prompt as the request.",
            "suppress_reentry_tools": ["prompt_perfect.optimize_prompt"],
        },
        "agent_tools.deliver_to_user": {
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "Готово",
                "artifacts": [],
                "defer": False,
            },
        },
    }
    specs = [
        {"type": "function", "function": {"name": "prompt_perfect.optimize_prompt"}},
        {"type": "function", "function": {"name": "agent_tools.deliver_to_user"}},
    ]
    helper = _make_helper(
        DummyPluginManager(responses, specs=specs),
        client=DummyClient([
            FakeResponse(content="plain text should be repaired"),
            FakeResponse(tool_calls=[
                FakeToolCall(
                    "agent_tools.deliver_to_user",
                    json.dumps({"text": "Готово"}),
                ),
            ]),
        ]),
    )
    helper.conversations[1] = [{"role": "system", "content": "agent-mode", "mode_key": "skills_agent"}]
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_key=lambda key: {"defer_direct_results": True} if key == "skills_agent" else None,
        get_mode_by_system_prompt=lambda _content: None,
    )
    response = FakeResponse(tool_calls=[
        FakeToolCall("prompt_perfect.optimize_prompt", json.dumps({"original_prompt": "guide"})),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    first_reentry_tools = helper.client.create_kwargs[0]["tools"]
    repair_tools = helper.client.create_kwargs[1]["tools"]
    assert [tool["function"]["name"] for tool in first_reentry_tools] == [
        "agent_tools.deliver_to_user",
    ]
    assert [tool["function"]["name"] for tool in repair_tools] == [
        "agent_tools.deliver_to_user",
    ]
    assert helper.client.create_kwargs[0]["tool_choice"] == "auto"
    assert helper.client.create_kwargs[1]["tool_choice"] == "auto"
    assert set(tools_used) == {"prompt_perfect.optimize_prompt", "agent_tools.deliver_to_user"}
    assert out["direct_result"]["text"] == "Готово"


def test_tool_suppression_filters_google_function_declarations():
    tools = {
        "function_declarations": [
            {"name": "prompt_perfect.optimize_prompt"},
            {"name": "agent_tools.deliver_to_user"},
        ]
    }

    filtered = _filter_tools_by_name(tools, {"prompt_perfect.optimize_prompt"})

    assert filtered == {
        "function_declarations": [
            {"name": "agent_tools.deliver_to_user"},
        ]
    }
    assert _has_tool_specs(filtered) is True
    assert _has_tool_specs(_filter_tools_by_name(filtered, {"agent_tools.deliver_to_user"})) is False


@pytest.mark.asyncio
async def test_non_object_tool_arguments_are_recoverable_tool_error():
    pm = DummyPluginManager({"p1.do": {"result": "should not run"}})
    helper = _make_helper(pm, client=DummyClient([FakeResponse(content="done")]))
    response = FakeResponse(tool_calls=[
        FakeToolCall("p1.do", "[]"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert pm.calls == []
    assert set(tools_used) == {"p1.do"}
    assert out.choices[0].message.content == "done"
    assert any(
        "Invalid arguments for p1.do" in (message.get("content") or "")
        for message in helper.conversations[1]
    )


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
    history_content = "\n".join(
        str(message.get("content") or "")
        for message in helper.conversations[1]
    )
    assert "Direct result returned to Telegram handler for delivery." in history_content
    assert "sent to the user" not in history_content


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
        "agent_tools.deliver_to_user": {
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "presentation ready",
                "artifacts": [],
                "defer": False,
            },
        },
    }
    helper = _make_helper(
        DummyPluginManager(
            responses,
            specs=[{"type": "function", "function": {"name": "agent_tools.deliver_to_user"}}],
        ),
        client=DummyClient([
            FakeResponse(content="presentation ready"),
            FakeResponse(tool_calls=[
                FakeToolCall("agent_tools.deliver_to_user", json.dumps({"text": "presentation ready"})),
            ]),
        ]),
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

    assert helper.client.calls == 2
    assert set(tools_used) == {"stable_diffusion.stable_diffusion", "agent_tools.deliver_to_user"}
    # Intermediate direct_result tools (image generation etc.) are deferred only
    # into the tool history so the model can reference them. They must not be
    # re-delivered alongside the final answer, otherwise the user receives helper
    # images before the actual artifact (e.g. images before the final pptx).
    assert out["direct_result"]["text"] == "presentation ready"
    assert out["direct_result"]["artifacts"] == []
    tool_messages = [message for message in helper.conversations[1] if message.get("role") == "tool"]
    assert tool_messages
    assert "/tmp/image.png" in tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_agent_tools_workflow_defers_intermediate_direct_results_without_mode_key():
    responses = {
        "agent_tools.manage_plan_tasks": {"success": True, "output": "plan added"},
        "stable_diffusion.stable_diffusion": {
            "direct_result": {
                "kind": "photo",
                "format": "path",
                "value": "/tmp/egg.png",
                "add_value": "generated",
            },
        },
        "codeinterpreter.download": {
            "direct_result": {
                "kind": "file",
                "format": "path",
                "value": "/tmp/intermediate.csv",
                "defer": False,
            },
        },
        "agent_tools.deliver_to_user": {
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "presentation ready",
                "artifacts": [{"kind": "file", "format": "path", "value": "/tmp/eggs.pptx"}],
                "defer": False,
            },
        },
    }
    specs = [
        {"type": "function", "function": {"name": "stable_diffusion.stable_diffusion"}},
        {"type": "function", "function": {"name": "codeinterpreter.download"}},
        {"type": "function", "function": {"name": "agent_tools.deliver_to_user"}},
    ]
    helper = _make_helper(
        DummyPluginManager(responses, specs=specs),
        client=DummyClient([
            FakeResponse(tool_calls=[
                FakeToolCall("stable_diffusion.stable_diffusion", "{}", id="call-image"),
                FakeToolCall("codeinterpreter.download", "{}", id="call-file"),
            ]),
            FakeResponse(tool_calls=[
                FakeToolCall(
                    "agent_tools.deliver_to_user",
                    json.dumps({
                        "text": "presentation ready",
                        "artifacts": [{"file_path": "/tmp/eggs.pptx"}],
                    }),
                    id="call-final",
                ),
            ]),
        ]),
    )
    helper.config["functions_max_consecutive_calls"] = 5
    helper.conversations[1] = [{"role": "system", "content": "memory-only system prompt"}]
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_key=lambda _key: None,
        get_mode_by_system_prompt=lambda _content: None,
    )
    response = FakeResponse(tool_calls=[
        FakeToolCall(
            "agent_tools.manage_plan_tasks",
            json.dumps({"action": "add", "tasks": [{"id": "T1", "content": "make deck"}]}),
            id="call-plan",
        ),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert helper.client.calls == 2
    assert [name for name, _args in helper.plugin_manager.calls] == [
        "agent_tools.manage_plan_tasks",
        "stable_diffusion.stable_diffusion",
        "codeinterpreter.download",
        "agent_tools.deliver_to_user",
    ]
    assert set(tools_used) == {
        "agent_tools.manage_plan_tasks",
        "stable_diffusion.stable_diffusion",
        "codeinterpreter.download",
        "agent_tools.deliver_to_user",
    }
    assert out["direct_result"]["kind"] == "final"
    assert out["direct_result"]["artifacts"][0]["value"] == "/tmp/eggs.pptx"
    tool_messages = [message for message in helper.conversations[1] if message.get("role") == "tool"]
    assert any("/tmp/egg.png" in message.get("content", "") for message in tool_messages)
    assert any("/tmp/intermediate.csv" in message.get("content", "") for message in tool_messages)
    manifest_messages = [
        message.get("content", "")
        for message in helper.conversations[1]
        if message.get("role") == "user" and "Current run artifact manifest" in message.get("content", "")
    ]
    assert manifest_messages
    assert "/tmp/egg.png" in manifest_messages[-1]
    assert "/tmp/intermediate.csv" in manifest_messages[-1]
    assert "Do not discover artifacts by broad-listing shared directories" in manifest_messages[-1]


@pytest.mark.asyncio
async def test_malformed_direct_result_reenters_model_instead_of_short_circuiting():
    helper = _make_helper(
        DummyPluginManager({"p1.do": {"direct_result": {"value": "missing kind"}}}),
        client=DummyClient([FakeResponse(content="done")]),
    )
    response = FakeResponse(tool_calls=[FakeToolCall("p1.do", "{}")])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert out.choices[0].message.content == "done"
    assert tools_used == ("p1.do",)


@pytest.mark.asyncio
async def test_skills_agent_retries_plain_text_final_response_through_delivery_tool():
    responses = {
        "terminal.terminal": {
            "success": True,
            "stdout": "Presentation created: /tmp/deck.pptx\n",
            "stderr": "",
        },
        "agent_tools.deliver_to_user": {
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "Готово",
                "artifacts": [{"kind": "file", "format": "path", "value": "/tmp/deck.pptx"}],
                "defer": False,
            },
        },
    }
    pm = DummyPluginManager(
        responses,
        specs=[{"type": "function", "function": {"name": "agent_tools.deliver_to_user"}}],
    )
    helper = _make_helper(
        pm,
        client=DummyClient([
            FakeResponse(content="<br>\n\n> paste"),
            FakeResponse(tool_calls=[
                FakeToolCall("agent_tools.deliver_to_user", json.dumps({
                    "text": "Готово",
                    "artifacts": [{"file_path": "/tmp/deck.pptx"}],
                }), id="call-final"),
            ]),
        ]),
    )
    helper.conversations[1] = [{"role": "system", "content": "agent-mode", "mode_key": "skills_agent"}]
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_key=lambda key: {"defer_direct_results": True} if key == "skills_agent" else None,
        get_mode_by_system_prompt=lambda _content: None,
    )
    response = FakeResponse(tool_calls=[
        FakeToolCall("terminal.terminal", json.dumps({"command": "node /tmp/create_pptx.js"}), id="call-terminal"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert helper.client.calls == 2
    assert helper.client.create_kwargs[1]["tool_choice"] == "auto"
    assert set(tools_used) == {"terminal.terminal", "agent_tools.deliver_to_user"}
    assert out["direct_result"]["kind"] == "final"
    assert out["direct_result"]["artifacts"][0]["value"] == "/tmp/deck.pptx"
    assert any(
        "Continue solving the original user task" in (message.get("content") or "")
        and "if the next step requires a tool, call that tool" in (message.get("content") or "")
        and "agent_tools.deliver_to_user" in (message.get("content") or "")
        for message in helper.conversations[1]
    )


@pytest.mark.asyncio
async def test_skills_agent_plain_status_after_tools_can_resume_tool_work():
    responses = {
        "terminal.terminal": {
            "success": True,
            "stdout": "ok\n",
            "stderr": "",
        },
        "agent_tools.deliver_to_user": {
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "Готово",
                "artifacts": [{"kind": "file", "format": "path", "value": "/tmp/deck.pptx"}],
                "defer": False,
            },
        },
    }
    specs = [
        {"type": "function", "function": {"name": "terminal.terminal"}},
        {"type": "function", "function": {"name": "agent_tools.deliver_to_user"}},
    ]
    helper = _make_helper(
        DummyPluginManager(responses, specs=specs),
        client=DummyClient([
            FakeResponse(content="PPTX создан. Проверяю содержимое через officecli."),
            FakeResponse(tool_calls=[
                FakeToolCall(
                    "terminal.terminal",
                    json.dumps({"command": "officecli view /tmp/deck.pptx text"}),
                    id="call-officecli",
                ),
            ]),
            FakeResponse(content="Проверил содержимое, готово."),
            FakeResponse(tool_calls=[
                FakeToolCall(
                    "agent_tools.deliver_to_user",
                    json.dumps({
                        "text": "Готово",
                        "artifacts": [{"file_path": "/tmp/deck.pptx"}],
                    }),
                    id="call-final",
                ),
            ]),
        ]),
    )
    helper.conversations[1] = [{"role": "system", "content": "agent-mode", "mode_key": "skills_agent"}]
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_key=lambda key: {"defer_direct_results": True} if key == "skills_agent" else None,
        get_mode_by_system_prompt=lambda _content: None,
    )
    response = FakeResponse(tool_calls=[
        FakeToolCall("terminal.terminal", json.dumps({"command": "node /tmp/create_pptx.js"}), id="call-build"),
    ])

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1, response=response, stream=False, allowed_plugins=["All"], user_id=1
    )

    assert helper.client.calls == 4
    assert [name for name, _args in helper.plugin_manager.calls] == [
        "terminal.terminal",
        "terminal.terminal",
        "agent_tools.deliver_to_user",
    ]
    officecli_args = json.loads(helper.plugin_manager.calls[1][1])
    assert "officecli view /tmp/deck.pptx text" in officecli_args["command"]
    assert set(tools_used) == {"terminal.terminal", "agent_tools.deliver_to_user"}
    assert out["direct_result"]["kind"] == "final"
    assert out["direct_result"]["artifacts"][0]["value"] == "/tmp/deck.pptx"
    assert helper.client.create_kwargs[1]["tool_choice"] == "auto"
    assert helper.client.create_kwargs[3]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_skills_agent_retries_initial_plain_text_response_through_delivery_tool():
    responses = {
        "agent_tools.deliver_to_user": {
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "Готово",
                "artifacts": [],
                "defer": False,
            },
        },
    }
    helper = _make_helper(
        DummyPluginManager(
            responses,
            specs=[{"type": "function", "function": {"name": "agent_tools.deliver_to_user"}}],
        ),
        client=DummyClient([
            FakeResponse(tool_calls=[
                FakeToolCall(
                    "agent_tools.deliver_to_user",
                    json.dumps({"text": "Готово"}),
                    id="call-final",
                ),
            ]),
        ]),
    )
    helper.conversations[1] = [{"role": "system", "content": "agent-mode", "mode_key": "skills_agent"}]
    helper.chat_modes_registry = types.SimpleNamespace(
        get_mode_by_key=lambda key: {"defer_direct_results": True} if key == "skills_agent" else None,
        get_mode_by_system_prompt=lambda _content: None,
    )

    out, tools_used = await helper._OpenAIHelper__handle_function_call(
        chat_id=1,
        response=FakeResponse(content="plain text should not pass through"),
        stream=False,
        allowed_plugins=["All"],
        user_id=1,
    )

    assert helper.client.calls == 1
    assert helper.client.create_kwargs[0]["tool_choice"] == "auto"
    assert set(tools_used) == {"agent_tools.deliver_to_user"}
    assert out["direct_result"]["text"] == "Готово"


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
        "agent_tools.deliver_to_user": {
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "presentation ready",
                "artifacts": [],
                "defer": False,
            },
        },
    }
    helper = _make_helper(
        DummyPluginManager(
            responses,
            specs=[{"type": "function", "function": {"name": "agent_tools.deliver_to_user"}}],
        ),
        db=MultiSessionDB(contexts),
        client=DummyClient([
            FakeResponse(tool_calls=[
                FakeToolCall("stable_diffusion.stable_diffusion", "{}", id="call-image"),
            ]),
            FakeResponse(content="presentation ready"),
            FakeResponse(tool_calls=[
                FakeToolCall("agent_tools.deliver_to_user", json.dumps({"text": "presentation ready"})),
            ]),
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

    assert answer["direct_result"]["text"] == "presentation ready"
    assert total_tokens == "0"
    assert helper.client.calls == 3
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
async def test_big_context_model_survives_tool_reentry():
    pm = DummyPluginManager({
        "weather.get_weather": {"result": "sunny"},
    })
    client = DummyClient([
        FakeResponse(tool_calls=[FakeToolCall("weather.get_weather", "{}")]),
        FakeResponse(tool_calls=None, content="done"),
    ])
    helper = _make_helper(pm, client=client)

    answer, _tokens = await helper.get_chat_response(
        chat_id=1,
        query="weather",
        user_id=1,
        big_context=True,
    )

    assert answer == "done"
    assert client.create_kwargs[0]["model"] == "llmgateway/big_context"
    assert client.create_kwargs[1]["model"] == "llmgateway/big_context"


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
                    "subprocess.run(['python3', '/srv/chatgpt-telegram-bot/bot/skills/META-SKILLS/pptx/scripts/build.py'])"
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
        and "terminal.terminal" in (message.get("content") or "")
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
        and "terminal.terminal" in (message.get("content") or "")
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
        and "terminal.terminal" in (message.get("content") or "")
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
        and "terminal.terminal" in (message.get("content") or "")
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
        {
            "codeinterpreter.deep_analysis": {"result": "42"},
            "agent_tools.deliver_to_user": {
                "direct_result": {
                    "kind": "final",
                    "format": "mixed",
                    "text": "calculated",
                    "artifacts": [],
                    "defer": False,
                },
            },
        },
        specs=[{"type": "function", "function": {"name": "agent_tools.deliver_to_user"}}],
        plugins={"skills": FakeSkillsPlugin()},
    )
    helper = _make_helper(pm, client=DummyClient([
        FakeResponse(content="calculated"),
        FakeResponse(tool_calls=[
            FakeToolCall("agent_tools.deliver_to_user", json.dumps({"text": "calculated"})),
        ]),
    ]))
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
    assert set(tools_used) == {"codeinterpreter.deep_analysis", "agent_tools.deliver_to_user"}
    assert out["direct_result"]["text"] == "calculated"


def test_model_messages_include_current_user_language_instruction():
    token = set_current_language("ru")
    try:
        messages = OpenAIHelper._messages_with_language_instruction([
            {"role": "system", "content": "base prompt"},
            {"role": "user", "content": "hello"},
        ])
    finally:
        reset_current_language(token)

    assert messages[0] == {"role": "system", "content": "base prompt"}
    assert messages[1]["role"] == "system"
    assert "Русский" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "hello"}
