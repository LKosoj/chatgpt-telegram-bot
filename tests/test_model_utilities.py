"""Tests for ``ModelUtilities`` (bot/model_utilities.py): the shared thin
layer over ``OpenAIHelper.chat_completion`` used by the cheap one-shot model
calls (``classify_reply_intent``, ``generate_session_name``,
``_summarise_window``).

Exercised against a minimal hand-rolled helper double -- exactly the shape
the class contract promises to support: only ``chat_completion(**kwargs)``
and ``config`` are touched, nothing else on the helper.
"""

from __future__ import annotations

import asyncio
import logging
import types

import pytest

from bot.model_utilities import ModelUtilities


class FakeHelper:
    """Minimal helper double: only ``config`` + ``chat_completion(**kwargs)``."""

    def __init__(self, config: dict, *, response=None, delay: float = 0.0, error: Exception | None = None):
        self.config = config
        self._response = response
        self._delay = delay
        self._error = error
        self.calls: list[dict] = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._response


def _response(content: str, *, total_tokens: int = 7):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
        usage=types.SimpleNamespace(total_tokens=total_tokens),
    )


# --- model selection fallback ------------------------------------------------

@pytest.mark.asyncio
async def test_one_shot_falls_back_light_model_then_model():
    helper = FakeHelper({"light_model": "cheap", "model": "big"}, response=_response("ok"))
    mu = ModelUtilities(helper)

    await mu.one_shot(kind="k", messages=[], timeout_seconds=5)

    assert helper.calls[0]["model"] == "cheap"


@pytest.mark.asyncio
async def test_one_shot_falls_back_to_model_when_light_model_missing():
    helper = FakeHelper({"light_model": "", "model": "big"}, response=_response("ok"))
    mu = ModelUtilities(helper)

    await mu.one_shot(kind="k", messages=[], timeout_seconds=5)

    assert helper.calls[0]["model"] == "big"


@pytest.mark.asyncio
async def test_one_shot_explicit_model_overrides_fallback():
    helper = FakeHelper({"light_model": "cheap", "model": "big"}, response=_response("ok"))
    mu = ModelUtilities(helper)

    await mu.one_shot(kind="k", messages=[], timeout_seconds=5, model="explicit")

    assert helper.calls[0]["model"] == "explicit"


@pytest.mark.asyncio
async def test_summarize_window_uses_three_tier_fallback():
    helper = FakeHelper(
        {"summary_model": "", "light_model": "cheap", "model": "big"},
        response=_response("summary text"),
    )
    mu = ModelUtilities(helper)

    result = await mu.summarize_window(messages=[], timeout_seconds=5)

    assert result == "summary text"
    assert helper.calls[0]["model"] == "cheap"
    assert helper.calls[0]["kind"] == "summary"


# --- timeout / degradation ----------------------------------------------------

@pytest.mark.asyncio
async def test_one_shot_returns_none_on_timeout():
    helper = FakeHelper({"model": "m"}, delay=0.05)
    mu = ModelUtilities(helper)

    result = await mu.one_shot(kind="k", messages=[], timeout_seconds=0.01)

    assert result is None


@pytest.mark.asyncio
async def test_classify_json_returns_none_on_timeout():
    helper = FakeHelper({"model": "m"}, delay=0.05)
    mu = ModelUtilities(helper)

    result = await mu.classify_json(kind="k", messages=[], timeout_seconds=0.01)

    assert result is None


@pytest.mark.asyncio
async def test_generate_title_returns_none_and_zero_tokens_on_timeout():
    helper = FakeHelper({"model": "m"}, delay=0.05)
    mu = ModelUtilities(helper)

    text, tokens = await mu.generate_title(messages=[], timeout_seconds=0.01)

    assert text is None
    assert tokens == 0


@pytest.mark.asyncio
async def test_summarize_window_raises_on_timeout():
    helper = FakeHelper({"model": "m"}, delay=0.05)
    mu = ModelUtilities(helper)

    with pytest.raises(asyncio.TimeoutError):
        await mu.summarize_window(messages=[], timeout_seconds=0.01)


@pytest.mark.asyncio
async def test_summarize_window_raises_on_transport_error():
    helper = FakeHelper({"model": "m"}, error=RuntimeError("boom"))
    mu = ModelUtilities(helper)

    with pytest.raises(RuntimeError):
        await mu.summarize_window(messages=[], timeout_seconds=5)


@pytest.mark.asyncio
async def test_one_shot_returns_none_on_transport_error():
    helper = FakeHelper({"model": "m"}, error=RuntimeError("boom"))
    mu = ModelUtilities(helper)

    result = await mu.one_shot(kind="k", messages=[], timeout_seconds=5)

    assert result is None


# --- successful extraction shapes --------------------------------------------

@pytest.mark.asyncio
async def test_classify_json_returns_raw_content_text():
    helper = FakeHelper({"model": "m"}, response=_response('{"intent": "text_reply"}'))
    mu = ModelUtilities(helper)

    result = await mu.classify_json(kind="reply_intent", messages=[], timeout_seconds=5)

    assert result == '{"intent": "text_reply"}'
    assert helper.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_generate_title_returns_text_and_tokens():
    helper = FakeHelper({"model": "m"}, response=_response("My Title", total_tokens=42))
    mu = ModelUtilities(helper)

    text, tokens = await mu.generate_title(messages=[], timeout_seconds=5)

    assert text == "My Title"
    assert tokens == 42


@pytest.mark.asyncio
async def test_generate_title_returns_none_on_empty_content():
    helper = FakeHelper({"model": "m"}, response=_response(""))
    mu = ModelUtilities(helper)

    text, tokens = await mu.generate_title(messages=[], timeout_seconds=5)

    assert text is None
    assert tokens == 0


# --- logging: exactly one log line per error ---------------------------------

@pytest.mark.asyncio
async def test_one_shot_logs_exactly_once_on_timeout(caplog):
    helper = FakeHelper({"model": "m"}, delay=0.05)
    mu = ModelUtilities(helper)

    with caplog.at_level(logging.ERROR, logger="bot.model_utilities"):
        await mu.one_shot(kind="k", messages=[], timeout_seconds=0.01)

    assert len(caplog.records) == 1


@pytest.mark.asyncio
async def test_summarize_window_kind_is_always_summary_regardless_of_caller():
    helper = FakeHelper({"model": "m"}, response=_response("s"))
    mu = ModelUtilities(helper)

    await mu.summarize_window(messages=[], timeout_seconds=5)

    assert helper.calls[0]["kind"] == "summary"
