"""Tests for DDGTranslatePlugin after the DuckDuckGo backend was dropped.

``DDGS.translate()`` was removed from ``duckduckgo_search`` and never existed in
its successor package ``ddgs``, so every call to this tool used to raise
AttributeError. Translation now goes through ``helper.ask()`` (the only
sanctioned way for a plugin to make a one-off model call from ``execute()``).
"""

import asyncio

import pytest

from bot.plugins.ddg_translate import DDGTranslatePlugin


class FakeHelper:
    """Records ``ask()`` calls and replies with a canned translation."""

    def __init__(self, reply="ciao"):
        self.reply = reply
        self.calls = []

    async def ask(self, prompt, user_id, assistant_prompt=None, **kwargs):
        self.calls.append(
            {"prompt": prompt, "user_id": user_id, "assistant_prompt": assistant_prompt}
        )
        return self.reply, 7


class FailingHelper:
    async def ask(self, *args, **kwargs):
        raise RuntimeError("gateway down")


def test_plugin_no_longer_imports_the_dead_backend():
    import bot.plugins.ddg_translate as module

    assert not hasattr(module, "DDGS")


def test_spec_is_unchanged():
    [spec] = DDGTranslatePlugin().get_spec()
    assert spec["name"] == "translate"
    assert set(spec["parameters"]["properties"]) == {"text", "to_language"}
    assert spec["parameters"]["required"] == ["text", "to_language"]


@pytest.mark.asyncio
async def test_translate_asks_the_model_and_returns_the_translation():
    helper = FakeHelper(reply="  ciao  ")
    result = await DDGTranslatePlugin().execute(
        "translate", helper, text="hello", to_language="it", user_id=42
    )

    assert result == {"translation": "ciao", "to_language": "it"}
    [call] = helper.calls
    assert call["prompt"] == "hello"
    assert call["user_id"] == 42
    assert "it" in call["assistant_prompt"]


@pytest.mark.asyncio
async def test_translate_rejects_empty_input_without_calling_the_model():
    helper = FakeHelper()
    plugin = DDGTranslatePlugin()

    assert "error" in await plugin.execute("translate", helper, text="   ", to_language="it")
    assert "error" in await plugin.execute("translate", helper, text="hello", to_language="")
    assert helper.calls == []


@pytest.mark.asyncio
async def test_translate_reports_an_empty_model_answer_as_an_error():
    result = await DDGTranslatePlugin().execute(
        "translate", FakeHelper(reply="   "), text="hello", to_language="it"
    )
    assert "error" in result


@pytest.mark.asyncio
async def test_translate_reports_a_failing_model_call_as_an_error():
    result = await DDGTranslatePlugin().execute(
        "translate", FailingHelper(), text="hello", to_language="it"
    )
    assert "gateway down" in result["error"]


@pytest.mark.asyncio
async def test_translate_does_not_block_the_event_loop():
    """execute() must yield control: the model call is awaited, not run inline."""
    ticks = []

    async def ticker():
        for _ in range(3):
            ticks.append(1)
            await asyncio.sleep(0)

    task = asyncio.ensure_future(ticker())
    await DDGTranslatePlugin().execute(
        "translate", FakeHelper(), text="hello", to_language="it"
    )
    await task
    assert ticks == [1, 1, 1]
