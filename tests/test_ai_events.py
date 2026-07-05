import pytest

from bot.ai_events import (
    AIMessage,
    AIResponseEnd,
    AIResponseStart,
    AIToolCall,
    AIUsage,
    event_to_log_dict,
)


def test_event_to_log_dict_serializes_nested_dataclasses():
    event = AIResponseEnd(
        message=AIMessage(
            role="assistant",
            content="done",
            tool_calls=(
                AIToolCall(
                    id="call_1",
                    name="search.query",
                    model_name="search_query",
                    arguments='{"q": "telegram"}',
                ),
            ),
        ),
        finish_reason="stop",
        usage=AIUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
    )

    assert event_to_log_dict(event) == {
        "message": {
            "role": "assistant",
            "content": "done",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "search.query",
                    "arguments": '{"q": "telegram"}',
                    "model_name": "search_query",
                }
            ],
            "tool_call_id": None,
            "name": None,
        },
        "type": "response_end",
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
    }


def test_response_start_metadata_defaults_are_not_shared():
    first = AIResponseStart(model="a")
    second = AIResponseStart(model="b")

    first.metadata["trace"] = "one"

    assert second.metadata == {}


def test_tool_call_keeps_raw_invalid_arguments_for_diagnostics():
    event = AIResponseEnd(
        message=AIMessage(
            role="assistant",
            content=None,
            tool_calls=(
                AIToolCall(
                    id="call_bad",
                    name="broken.tool",
                    arguments='{"unterminated"',
                ),
            ),
        )
    )

    data = event_to_log_dict(event)

    assert data["message"]["content"] is None
    assert data["message"]["tool_calls"][0]["arguments"] == '{"unterminated"'


def test_event_to_log_dict_rejects_non_event():
    with pytest.raises(TypeError):
        event_to_log_dict({"type": "response_end"})  # type: ignore[arg-type]
