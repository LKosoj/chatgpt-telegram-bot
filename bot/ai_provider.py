from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .ai_events import (
    AIEvent,
    AIMessage,
    AIProviderError,
    AIResponseEnd,
    AITextDelta,
    AIToolCall,
    AIToolCallReceived,
    AIUsage,
)


@dataclass(frozen=True, slots=True)
class AIProviderRequest:
    model: str
    messages: tuple[Mapping[str, Any], ...]
    stream: bool = False
    tools: Any | None = None
    tool_choice: Any = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: Mapping[str, Any] | None = None
    extra_headers: Mapping[str, str] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AIProviderChoice:
    message: AIMessage
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AIProviderResponse:
    text: str = ""
    tool_calls: tuple[AIToolCall, ...] = ()
    usage: AIUsage | None = None
    finish_reason: str | None = None
    errors: tuple[AIProviderError, ...] = ()
    choices: tuple[AIProviderChoice, ...] = ()


class AIProvider(Protocol):
    def stream_response(
        self,
        request: AIProviderRequest,
    ) -> AsyncIterator[AIEvent]:
        ...


async def collect_ai_response(events: AsyncIterator[AIEvent]) -> AIProviderResponse:
    text_parts: list[str] = []
    tool_calls: list[AIToolCall] = []
    final_message: AIMessage | None = None
    usage: AIUsage | None = None
    finish_reason: str | None = None
    errors: list[AIProviderError] = []
    choices: list[AIProviderChoice] = []

    async for event in events:
        if isinstance(event, AITextDelta):
            text_parts.append(event.delta)
        elif isinstance(event, AIToolCallReceived):
            tool_calls.append(event.tool_call)
        elif isinstance(event, AIProviderError):
            errors.append(event)
        elif isinstance(event, AIResponseEnd):
            if final_message is None:
                final_message = event.message
                finish_reason = event.finish_reason
            usage = event.usage
            if event.message is not None:
                choices.append(AIProviderChoice(
                    message=event.message,
                    finish_reason=event.finish_reason,
                ))

    if final_message is not None:
        text = final_message.content or "".join(text_parts)
        final_tool_calls = final_message.tool_calls or tuple(tool_calls)
    else:
        text = "".join(text_parts)
        final_tool_calls = tuple(tool_calls)
    if not choices:
        choices.append(AIProviderChoice(
            message=AIMessage(
                role="assistant",
                content=text or None,
                tool_calls=tuple(final_tool_calls) or None,
            ),
            finish_reason=finish_reason,
        ))

    return AIProviderResponse(
        text=text,
        tool_calls=tuple(final_tool_calls),
        usage=usage,
        finish_reason=finish_reason,
        errors=tuple(errors),
        choices=tuple(choices),
    )
