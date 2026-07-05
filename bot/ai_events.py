from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, TypeAlias

JSONValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)


@dataclass(frozen=True, slots=True)
class AIUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class AIToolCall:
    """Tool call at the current boundary.

    Provider adapters use the model-visible tool name. PluginToolAdapter
    canonicalizes it before execution and preserves the raw provider name in
    model_name.
    """

    id: str
    name: str
    arguments: str = "{}"
    model_name: str | None = None


@dataclass(frozen=True, slots=True)
class AIToolResult:
    tool_call_id: str | None
    name: str
    ok: bool
    content: str
    error: str | None = None
    direct_result: dict[str, Any] | None = None
    artifacts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class AIMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = ""
    tool_calls: tuple[AIToolCall, ...] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class AIResponseStart:
    type: Literal["response_start"] = "response_start"
    model: str = ""
    provider: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AITextDelta:
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True, slots=True)
class AIThinkingDelta:
    delta: str
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(frozen=True, slots=True)
class AIToolCallReceived:
    tool_call: AIToolCall
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True, slots=True)
class AIRetry:
    attempt: int
    max_attempts: int
    delay_seconds: float
    message: str
    type: Literal["retry"] = "retry"
    data: dict[str, JSONValue] | None = None


@dataclass(frozen=True, slots=True)
class AIProviderError:
    message: str
    type: Literal["provider_error"] = "provider_error"
    recoverable: bool = False
    data: dict[str, JSONValue] | None = None


@dataclass(frozen=True, slots=True)
class AIResponseEnd:
    message: AIMessage | None = None
    type: Literal["response_end"] = "response_end"
    finish_reason: str | None = None
    usage: AIUsage | None = None


@dataclass(frozen=True, slots=True)
class AIToolExecutionStart:
    tool_call: AIToolCall
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass(frozen=True, slots=True)
class AIToolExecutionEnd:
    result: AIToolResult
    type: Literal["tool_execution_end"] = "tool_execution_end"
    duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class AIRunEnd:
    type: Literal["run_end"] = "run_end"
    reason: str | None = None


AIEvent: TypeAlias = (
    AIResponseStart
    | AITextDelta
    | AIThinkingDelta
    | AIToolCallReceived
    | AIRetry
    | AIProviderError
    | AIResponseEnd
    | AIToolExecutionStart
    | AIToolExecutionEnd
    | AIRunEnd
)


def event_to_log_dict(event: AIEvent) -> dict[str, Any]:
    """Return a JSON-serializable dict for session/runtime logging."""
    if not is_dataclass(event):
        raise TypeError("event_to_log_dict expects an AIEvent dataclass")
    return _to_plain_dict(asdict(event))


def _to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain_dict(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_dict(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value
