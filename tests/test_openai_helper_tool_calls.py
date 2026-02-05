import types
import pytest

pytest.importorskip("tiktoken")

from bot.openai_helper import OpenAIHelper


class DummyDB:
    def list_user_sessions(self, user_id, is_active=1):
        return []

    def get_conversation_context(self, *args, **kwargs):
        return {}, None, None, None, None


class DummyPluginManager:
    def __init__(self, responses):
        self.responses = responses

    def filter_allowed_plugins(self, allowed_plugins):
        return allowed_plugins

    async def call_function(self, name, helper, arguments):
        return self.responses[name]

    def get_functions_specs(self, helper, model_to_use, allowed_plugins):
        return []


class DummyClient:
    def __init__(self):
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        return FakeResponse(tool_calls=None)


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = types.SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, tool_calls=None):
        self.message = FakeMessage(tool_calls=tool_calls)
        self.delta = None
        self.finish_reason = None


class FakeResponse:
    def __init__(self, tool_calls=None):
        self.choices = [FakeChoice(tool_calls=tool_calls)]


def _make_helper(plugin_manager):
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
        "image_model": "dall-e-2",
        "image_quality": "standard",
        "image_style": "vivid",
        "image_size": "512x512",
        "auto_chat_modes": False,
        "model": "openai/gpt-4.1",
        "enable_functions": True,
        "functions_max_consecutive_calls": 2,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "bot_language": "en",
        "show_plugins_used": False,
        "whisper_prompt": "",
        "vision_model": "gpt-4-vision-preview",
        "enable_vision_follow_up_questions": True,
        "vision_prompt": "test",
        "vision_detail": "auto",
        "vision_max_tokens": 300,
        "tts_model": "tts-1",
        "tts_voice": "alloy",
        "yandex_api_token": "",
        "assemblyai_api_key": "",
        "big_model_to_use": "",
    }
    helper = OpenAIHelper(config=config, plugin_manager=plugin_manager, db=DummyDB())
    helper.client = DummyClient()
    helper.conversations[1] = []
    return helper


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
