"""Regression coverage for ChatRun.run_non_stream's "retry after empty
response" duplicate (T12-plan.md T12c C-extra): the block that calls
helper._retry_empty_response_with_tools() then, if a retry response came
back, helper._handle_function_call() on it, is repeated verbatim in the
"plugins_used and not response_has_message_text" branch and the
"not plugins_used and not response_has_message_text" elif branch of
bot/chat_run.py:ChatRun.run_non_stream. Written before extracting the shared
_retry_after_empty_response() helper, to pin behavior (including the "no
retry available" case, where `response` must stay untouched rather than be
overwritten with a None sentinel) across the refactor.
"""

from __future__ import annotations

import types

import pytest

from bot.chat_response_utils import EMPTY_MODEL_RESPONSE_ERROR
from bot.chat_run import ChatRun


class FakeMessage:
    def __init__(self, content=""):
        self.content = content
        self.tool_calls = None


class FakeChoice:
    def __init__(self, content=""):
        self.message = FakeMessage(content)
        self.finish_reason = "stop"


class FakeResponse:
    def __init__(self, content="", total_tokens=3, prompt_tokens=1, completion_tokens=2):
        self.choices = [FakeChoice(content)]
        self.usage = types.SimpleNamespace(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class FakeHelper:
    """Minimal stand-in for OpenAIHelper, covering only what
    ChatRun.run_non_stream touches on the empty-response-retry path."""

    def __init__(self, *, initial_plugins_used):
        self.config = {
            "enable_functions": True,
            "n_choices": 1,
            "show_usage": False,
            "show_plugins_used": False,
            "bot_language": "en",
        }
        self._gate_fired = {}
        self._chat_request_extra_tokens = {}
        self._chat_request_models = {}
        self._chat_request_usage_split = {}
        self.session_logger = None
        self.plugin_manager = types.SimpleNamespace(get_plugin_source_name=lambda p: p)
        self.history = []

        self._initial_plugins_used = initial_plugins_used
        self.handle_function_call_calls = []
        self.retry_with_tools_calls = []
        self.retry_after_tools_calls = []
        # Queue of (response, plugins_used) pairs returned by successive
        # _handle_function_call calls, after the first (initial) call.
        self.handle_function_call_queue = []
        self.retry_with_tools_response = None
        self.retry_after_tools_response = None

    def _chat_state_key(self, chat_id):
        return chat_id

    async def _common_get_chat_response(self, chat_id, query, **kwargs):
        return FakeResponse(content="")

    async def resolve_allowed_plugins(self, chat_id, session_id, user_id):
        return ["some_plugin"]

    async def _add_to_history(self, chat_id, *, role, content, session_id):
        self.history.append((role, content))

    async def _handle_function_call(
        self, chat_id, response, *, allowed_plugins, user_id, request_context,
        model_to_use, token_accumulator, usage_accumulator,
    ):
        self.handle_function_call_calls.append(model_to_use)
        if not self.handle_function_call_calls[:-1]:
            # First call: the initial post-completion function-call handling.
            return FakeResponse(content=""), self._initial_plugins_used
        return self.handle_function_call_queue.pop(0)

    async def _retry_empty_response_with_tools(self, chat_id, user_id, session_id, allowed_plugins, *, model_to_use):
        self.retry_with_tools_calls.append(model_to_use)
        return self.retry_with_tools_response

    async def _retry_empty_response_after_tools(self, chat_id, user_id, session_id, *, model_to_use):
        self.retry_after_tools_calls.append(model_to_use)
        return self.retry_after_tools_response


def _direct_result(value):
    return {"direct_result": {"kind": "text", "value": value}}


@pytest.mark.asyncio
async def test_retry_after_tool_calls_direct_result_short_circuits():
    """Branch: plugins_used truthy, no text -> retry -> direct_result."""
    helper = FakeHelper(initial_plugins_used=("pluginA",))
    helper._chat_request_models["chat-1"] = "gpt-test"
    helper.retry_with_tools_response = FakeResponse(content="retry-in-flight")
    helper.handle_function_call_queue = [(_direct_result("done"), ("pluginB",))]

    run = ChatRun(helper)
    response, total_tokens = await run.run_non_stream(
        chat_id="chat-1", query="hi", session_id="s1", user_id=1, request_context=None,
    )

    assert response == _direct_result("done")
    assert helper.retry_with_tools_calls == ["gpt-test"]
    assert len(helper.handle_function_call_calls) == 2
    assert helper._chat_request_usage_split["chat-1"] is None


@pytest.mark.asyncio
async def test_retry_before_tool_calls_direct_result_short_circuits():
    """Branch: plugins_used empty, no text -> retry -> direct_result."""
    helper = FakeHelper(initial_plugins_used=())
    helper._chat_request_models["chat-1"] = "gpt-test"
    helper.retry_with_tools_response = FakeResponse(content="retry-in-flight")
    helper.handle_function_call_queue = [(_direct_result("done-2"), ("pluginC",))]

    run = ChatRun(helper)
    response, total_tokens = await run.run_non_stream(
        chat_id="chat-1", query="hi", session_id="s1", user_id=1, request_context=None,
    )

    assert response == _direct_result("done-2")
    assert helper.retry_with_tools_calls == ["gpt-test"]
    assert len(helper.handle_function_call_calls) == 2


@pytest.mark.asyncio
async def test_no_retry_available_leaves_response_untouched():
    """Branch: plugins_used empty, no text, and the retry call itself
    returns None (no retry response available). `response` must stay the
    original (still-empty) response rather than being overwritten -- this
    is exactly the guard a naive extraction could drop, so failure here
    would surface as an AttributeError on a None response instead of the
    expected "empty model response" ValueError.
    """
    helper = FakeHelper(initial_plugins_used=())
    helper.retry_with_tools_response = None

    run = ChatRun(helper)
    with pytest.raises(ValueError, match=EMPTY_MODEL_RESPONSE_ERROR):
        await run.run_non_stream(
            chat_id="chat-1", query="hi", session_id="s1", user_id=1, request_context=None,
        )

    assert helper.retry_with_tools_calls == [None]
    # No retry response -> _handle_function_call must not be called a second time.
    assert len(helper.handle_function_call_calls) == 1
