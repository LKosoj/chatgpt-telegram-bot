import types

import pytest

from bot.ai_provider import AIProviderRequest, collect_ai_response
from bot.ai_providers.openai_compatible import OpenAICompatibleProvider


class FakeToolCall:
    def __init__(self, name, arguments, id="call_1"):
        self.id = id
        self.function = types.SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, content=None, tool_calls=None, finish_reason="stop"):
        self.message = FakeMessage(content=content, tool_calls=tool_calls)
        self.delta = None
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [FakeChoice(content=content, tool_calls=tool_calls)]
        self.usage = types.SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        )


class FakeStreamFunctionDelta:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class FakeStreamToolCallDelta:
    def __init__(self, index=0, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = FakeStreamFunctionDelta(name=name, arguments=arguments)


class FakeStreamChoice:
    def __init__(self, content=None, finish_reason=None, tool_calls=None):
        self.message = None
        self.delta = types.SimpleNamespace(content=content, tool_calls=tool_calls)
        self.finish_reason = finish_reason


class FakeStreamChunk:
    def __init__(self, content=None, finish_reason=None, tool_calls=None):
        self.choices = [
            FakeStreamChoice(
                content=content,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )
        ]


async def fake_stream(chunks):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_openai_compatible_provider_collects_non_stream_text_and_usage():
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        return FakeResponse(content="hello")

    provider = OpenAICompatibleProvider(create)
    request = AIProviderRequest(
        model="m",
        messages=({"role": "user", "content": "hi"},),
        temperature=0.1,
        max_tokens=50,
        extra_headers={"X-Title": "tgBot"},
        extra={"n": 1},
    )

    response = await collect_ai_response(provider.stream_response(request))

    assert calls == [{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 50,
        "extra_headers": {"X-Title": "tgBot"},
        "n": 1,
    }]
    assert response.text == "hello"
    assert response.usage is not None
    assert response.usage.total_tokens == 3
    assert response.finish_reason == "stop"
    assert response.choices[0].message.content == "hello"


@pytest.mark.asyncio
async def test_openai_compatible_provider_preserves_raw_tool_arguments():
    async def create(**_kwargs):
        return FakeResponse(
            tool_calls=[
                FakeToolCall("skills_run", '{"bad"'),
            ],
        )

    provider = OpenAICompatibleProvider(create)
    response = await collect_ai_response(provider.stream_response(
        AIProviderRequest(model="m", messages=()),
    ))

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "skills_run"
    assert response.tool_calls[0].model_name == "skills_run"
    assert response.tool_calls[0].arguments == '{"bad"'


@pytest.mark.asyncio
async def test_openai_compatible_provider_preserves_dict_tool_shape():
    calls = []
    google_tools = {
        "function_declarations": [
            {"name": "skills_list", "description": "List skills"},
        ]
    }

    async def create(**kwargs):
        calls.append(kwargs)
        return FakeResponse(content="ok")

    provider = OpenAICompatibleProvider(create)
    await collect_ai_response(provider.stream_response(
        AIProviderRequest(model="m", messages=(), tools=google_tools),
    ))

    assert calls[0]["tools"] == google_tools
    assert calls[0]["tools"] is not google_tools


@pytest.mark.asyncio
async def test_openai_compatible_provider_collects_stream_deltas():
    async def create(**kwargs):
        assert kwargs["stream"] is True
        return fake_stream((
            FakeStreamChunk("hel"),
            FakeStreamChunk("lo", finish_reason="stop"),
        ))

    provider = OpenAICompatibleProvider(create)
    response = await collect_ai_response(provider.stream_response(
        AIProviderRequest(model="m", messages=(), stream=True),
    ))

    assert response.text == "hello"
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_compatible_provider_aggregates_streamed_tool_calls():
    async def create(**kwargs):
        assert kwargs["stream"] is True
        return fake_stream((
            FakeStreamChunk(tool_calls=[
                FakeStreamToolCallDelta(
                    index=0,
                    id="call_1",
                    name="skills_run",
                    arguments='{"name"',
                ),
            ]),
            FakeStreamChunk(
                tool_calls=[
                    FakeStreamToolCallDelta(index=0, arguments=':"pptx"}'),
                ],
                finish_reason="tool_calls",
            ),
        ))

    provider = OpenAICompatibleProvider(create)
    response = await collect_ai_response(provider.stream_response(
        AIProviderRequest(model="m", messages=(), stream=True),
    ))

    assert response.text == ""
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "skills_run"
    assert response.tool_calls[0].model_name == "skills_run"
    assert response.tool_calls[0].arguments == '{"name":"pptx"}'
