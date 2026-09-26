from types import SimpleNamespace

import pytest

from bot.ai_events import (
    AIMessage,
    AIProviderError,
    AIResponseEnd,
    AITextDelta,
    AIToolCall,
    AIToolCallReceived,
    AIUsage,
)
from bot.ai_provider import AIProviderRequest, collect_ai_response
from bot.ai_providers.fake import FakeAIProvider
from bot.ai_providers.openai_compatible import OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_fake_provider_records_request_and_collects_text():
    provider = FakeAIProvider.text("hello")
    request = AIProviderRequest(
        model="test-model",
        messages=({"role": "user", "content": "hi"},),
        stream=True,
    )

    response = await collect_ai_response(provider.stream_response(request))

    assert provider.requests == [request]
    assert response.text == "hello"
    assert response.tool_calls == ()
    assert response.errors == ()


@pytest.mark.asyncio
async def test_collect_ai_response_uses_final_message_over_delta_text():
    provider = FakeAIProvider((
        AITextDelta("hel"),
        AITextDelta("lo"),
        AIResponseEnd(
            message=AIMessage(role="assistant", content="normalized hello"),
            finish_reason="stop",
            usage=AIUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        ),
    ))

    response = await collect_ai_response(provider.stream_response(
        AIProviderRequest(model="m", messages=()),
    ))

    assert response.text == "normalized hello"
    assert response.finish_reason == "stop"
    assert response.usage == AIUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)


@pytest.mark.asyncio
async def test_collect_ai_response_keeps_tool_calls_and_errors():
    tool_call = AIToolCall(
        id="call_1",
        name="skills.run",
        arguments='{"name": "pptx"}',
    )
    error = AIProviderError(message="temporary", recoverable=True)
    provider = FakeAIProvider((
        AIToolCallReceived(tool_call),
        error,
        AIResponseEnd(message=AIMessage(role="assistant", tool_calls=(tool_call,))),
    ))

    response = await collect_ai_response(provider.stream_response(
        AIProviderRequest(model="m", messages=()),
    ))

    assert response.tool_calls == (tool_call,)
    assert response.errors == (error,)


@pytest.mark.asyncio
async def test_fake_provider_consumes_queued_responses_in_order():
    provider = FakeAIProvider()
    provider.queue_text("first")
    provider.queue_text("second")
    request = AIProviderRequest(model="m", messages=())

    first = await collect_ai_response(provider.stream_response(request))
    second = await collect_ai_response(provider.stream_response(request))

    assert first.text == "first"
    assert second.text == "second"
    provider.assert_no_pending_events()


@pytest.mark.asyncio
async def test_fake_provider_fails_loudly_when_no_response_is_queued():
    provider = FakeAIProvider()

    with pytest.raises(AssertionError, match="no queued event batch"):
        await collect_ai_response(provider.stream_response(
            AIProviderRequest(model="m", messages=()),
        ))


@pytest.mark.asyncio
async def test_fake_provider_generate_image_queue_round_trip():
    provider = FakeAIProvider()
    provider.queue_image("image-result-1")

    result = await provider.generate_image(prompt="a cat")

    assert result == "image-result-1"
    assert provider.image_calls == [{"prompt": "a cat"}]


@pytest.mark.asyncio
async def test_fake_provider_edit_image_queue_round_trip():
    provider = FakeAIProvider()
    provider.queue_image("edited-result-1")

    result = await provider.edit_image(prompt="add a hat", image_bytes=b"x")

    assert result == "edited-result-1"
    assert provider.image_calls == [{"prompt": "add a hat", "image_bytes": b"x"}]


@pytest.mark.asyncio
async def test_fake_provider_speech_queue_round_trip():
    provider = FakeAIProvider()
    provider.queue_speech("speech-result-1")

    result = await provider.speech(text="hello", voice="alice")

    assert result == "speech-result-1"
    assert provider.speech_calls == [{"text": "hello", "voice": "alice"}]


@pytest.mark.asyncio
async def test_fake_provider_transcribe_queue_round_trip():
    provider = FakeAIProvider()
    provider.queue_transcribe("transcript-1")

    result = await provider.transcribe(file=b"audio-bytes")

    assert result == "transcript-1"
    assert provider.transcribe_calls == [{"file": b"audio-bytes"}]


@pytest.mark.asyncio
async def test_fake_provider_list_models_queue_round_trip():
    provider = FakeAIProvider()
    provider.queue_models(["model-a", "model-b"])

    assert await provider.list_models() == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_fake_provider_list_voices_queue_round_trip():
    provider = FakeAIProvider()
    provider.queue_voices(["alice", "bob"])

    result = await provider.list_voices(model="tts-1")

    assert result == ["alice", "bob"]
    assert provider.voices_calls == [{"model": "tts-1"}]


@pytest.mark.asyncio
async def test_usage_keeps_missing_prompt_completion_as_none():
    """Шлюз прислал total_tokens, но не прислал prompt/completion_tokens.

    Regression test for T06: раньше _usage()/_int_or_zero превращали
    отсутствующие поля в 0, из-за чего resolve_chat_cost решал, что
    разбивка известна (model_split, цена 0.0), вместо честного
    model_blended.
    """
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="hi", tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(total_tokens=5, prompt_tokens=None, completion_tokens=None),
    )

    async def create_chat_completion(**kwargs):
        return fake_response

    provider = OpenAICompatibleProvider(create_chat_completion)
    response = await collect_ai_response(provider.stream_response(
        AIProviderRequest(model="m", messages=()),
    ))

    assert response.usage == AIUsage(prompt_tokens=None, completion_tokens=None, total_tokens=5)
