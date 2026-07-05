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
