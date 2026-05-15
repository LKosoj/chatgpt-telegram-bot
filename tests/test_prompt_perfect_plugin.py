import types
from unittest.mock import AsyncMock

import pytest

from bot.plugins.prompt_perfect import PromptPerfectPlugin


class FakeHelper:
    def __init__(self, content):
        self.config = {"light_model": "fake-light"}
        self.chat_completion = AsyncMock(return_value=self._response(content))
        self.get_chat_response = AsyncMock(side_effect=AssertionError("must not be called"))

    def _response(self, content):
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


def test_spec_keeps_tool_name_and_arguments_but_describes_next_response_rewrite():
    spec = PromptPerfectPlugin().get_spec()[0]

    assert spec["name"] == "optimize_prompt"
    assert set(spec["parameters"]["properties"]) == {"original_prompt", "context"}
    assert spec["parameters"]["required"] == ["original_prompt"]
    assert "next response" in spec["description"]
    assert "does not answer" in spec["description"]


@pytest.mark.asyncio
async def test_execute_uses_low_level_completion_without_model_response():
    plugin = PromptPerfectPlugin()
    helper = FakeHelper("Rewrite this as a precise request.")

    result = await plugin.execute(
        "optimize_prompt",
        helper,
        chat_id=123,
        original_prompt="make it better",
        context="short answer",
    )

    helper.get_chat_response.assert_not_called()
    helper.chat_completion.assert_awaited_once()
    call_kwargs = helper.chat_completion.await_args.kwargs
    assert call_kwargs["model"] == "fake-light"
    assert call_kwargs["stream"] is False
    assert "tools" not in call_kwargs
    assert "tool_choice" not in call_kwargs
    assert "make it better" in call_kwargs["messages"][1]["content"]
    assert result["optimized_prompt"] == "Rewrite this as a precise request."
    assert "instruction" in result
    assert result["suppress_reentry_tools"] == ["prompt_perfect.optimize_prompt"]
    assert result["retry_plain_text_tool_intent"] is True
    assert "model_response" not in result


@pytest.mark.asyncio
async def test_optimize_prompt_falls_back_to_original_prompt_for_empty_rewrite():
    plugin = PromptPerfectPlugin()
    helper = FakeHelper("   ")

    result = await plugin.execute(
        "optimize_prompt",
        helper,
        chat_id=123,
        original_prompt="original prompt",
    )

    helper.get_chat_response.assert_not_called()
    assert result["optimized_prompt"] == "original prompt"
    assert "model_response" not in result
