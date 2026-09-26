import types

import httpx2
import openai
import pytest

from bot.ai_provider import (
    AIProviderRequest,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderStreamError,
    collect_ai_response,
)
from bot.ai_providers.openai_compatible import OpenAICompatibleProvider, raw_chat_completion


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


def _rate_limit_error(message="limited"):
    request = httpx2.Request("POST", "https://example.com")
    response = httpx2.Response(429, request=request)
    return openai.RateLimitError(message, response=response, body=None)


def _bad_request_error(message="bad"):
    request = httpx2.Request("POST", "https://example.com")
    response = httpx2.Response(400, request=request)
    return openai.BadRequestError(message, response=response, body=None)


def _generic_api_error(message="broken"):
    request = httpx2.Request("POST", "https://example.com")
    return openai.APIConnectionError(message=message, request=request)


class _FakeSDKClient:
    """Minimal stand-in for openai.AsyncOpenAI: only chat.completions.create."""

    def __init__(self, *, exc=None, response=None):
        self._exc = exc
        self._response = response
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return self._response


@pytest.mark.asyncio
async def test_raw_chat_completion_translates_rate_limit_error():
    client = _FakeSDKClient(exc=_rate_limit_error())
    create = raw_chat_completion(lambda: client)

    with pytest.raises(ProviderRateLimitError):
        await create(model="m", messages=[])


@pytest.mark.asyncio
async def test_raw_chat_completion_translates_bad_request_error():
    client = _FakeSDKClient(exc=_bad_request_error())
    create = raw_chat_completion(lambda: client)

    with pytest.raises(ProviderBadRequestError):
        await create(model="m", messages=[])


@pytest.mark.asyncio
async def test_raw_chat_completion_translates_generic_api_error():
    client = _FakeSDKClient(exc=_generic_api_error())
    create = raw_chat_completion(lambda: client)

    with pytest.raises(ProviderError):
        await create(model="m", messages=[])


@pytest.mark.asyncio
async def test_raw_chat_completion_stream_translates_mid_iteration_api_error():
    async def raw_stream():
        yield FakeStreamChunk("hel")
        raise _generic_api_error("stream broke")

    client = _FakeSDKClient(response=raw_stream())
    create = raw_chat_completion(lambda: client)

    translated_stream = await create(model="m", messages=[], stream=True)

    chunks = []
    with pytest.raises(ProviderStreamError):
        async for chunk in translated_stream:
            chunks.append(chunk)
    assert len(chunks) == 1
