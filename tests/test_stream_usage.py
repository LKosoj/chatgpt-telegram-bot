"""Streaming turns priced from a real usage chunk instead of the local tokenizer.

Without stream_options={'include_usage': True} an OpenAI-compatible stream
carries no token counts at all, so bot/openai_helper.py falls back to
_estimate_stream_tokens and bot.pricing can only apply the blended
(input+output)/2 rate. These tests cover the opt-in flag that closes that gap
and, just as importantly, the cases where the split must stay unknown.
"""
import types

import pytest

from tests.test_openai_helper_tool_calls import (
    DummyClient,
    DummyPluginManager,
    FakeClosableAsyncStream,
    FakeStreamItem,
    _make_helper,
)


class FakeUsageChunk:
    """Final choice-less chunk the API appends when include_usage is on."""

    def __init__(self, prompt_tokens, completion_tokens):
        self.choices = []
        self.usage = types.SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


def _stream_helper(stream_items, *, include_usage):
    pm = DummyPluginManager({}, specs=[])
    client = DummyClient([FakeClosableAsyncStream(stream_items)])
    helper = _make_helper(pm, client=client)
    helper.config["stream"] = True
    helper.config["stream_include_usage"] = include_usage
    helper.config["enable_functions"] = False
    return helper, client


async def _drain(helper, chat_id=1):
    chunks = []
    async for chunk in helper.get_chat_response_stream(chat_id=chat_id, query="hi"):
        chunks.append(chunk)
    return chunks


async def test_stream_options_absent_when_flag_disabled():
    helper, client = _stream_helper([FakeStreamItem(content="hello")], include_usage=False)

    await _drain(helper)

    assert client.create_kwargs, "expected the stream request to have been issued"
    assert "stream_options" not in client.create_kwargs[0]


async def test_stream_options_sent_when_flag_enabled():
    helper, client = _stream_helper([FakeStreamItem(content="hello")], include_usage=True)

    await _drain(helper)

    assert client.create_kwargs[0]["stream_options"] == {"include_usage": True}


async def test_usage_chunk_populates_split_and_token_count():
    helper, _client = _stream_helper(
        [FakeStreamItem(content="hello"), FakeUsageChunk(prompt_tokens=40, completion_tokens=7)],
        include_usage=True,
    )

    chunks = await _drain(helper)

    # Real counts win over the tokenizer estimate, so pricing can bill the
    # 40 prompt tokens and the 7 completion tokens at their separate rates.
    assert helper._chat_request_usage_split[helper._chat_state_key(1)] == (40, 7)
    assert chunks[-1][1] == "47"


async def test_split_stays_unknown_without_usage_chunk():
    helper, _client = _stream_helper([FakeStreamItem(content="hello")], include_usage=True)

    chunks = await _drain(helper)

    # A gateway that ignores stream_options must not be papered over: no split
    # means blended pricing, and the token count falls back to the estimate.
    assert helper._chat_request_usage_split[helper._chat_state_key(1)] is None
    assert chunks[-1][1] != "0"


async def test_split_stays_unknown_when_a_tool_call_ran_first():
    """A re-entry stream's usage covers only the last round trip.

    Recording it would price the turn as if the earlier tool-call round trips
    were free, so the split must stay unknown instead.
    """
    helper, _client = _stream_helper(
        [FakeStreamItem(content="hello"), FakeUsageChunk(prompt_tokens=40, completion_tokens=7)],
        include_usage=True,
    )

    async def fake_handle_function_call(chat_id, response, **kwargs):
        return response, ("some_plugin",)

    helper._OpenAIHelper__handle_function_call = fake_handle_function_call
    helper.config["enable_functions"] = True

    await _drain(helper)

    assert helper._chat_request_usage_split[helper._chat_state_key(1)] is None
