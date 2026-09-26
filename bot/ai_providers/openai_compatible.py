from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import openai

from bot.ai_events import (
    AIEvent,
    AIMessage,
    AIResponseEnd,
    AIResponseStart,
    AITextDelta,
    AIToolCall,
    AIToolCallReceived,
    AIUsage,
)
from bot.ai_provider import (
    AIProviderRequest,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderStreamError,
)

CreateChatCompletion = Callable[..., Awaitable[Any]]


def build_openai_client(config: dict, http_client) -> "openai.AsyncOpenAI":
    """Moved from OpenAIHelper.__init__ verbatim (retries only in the SDK,
    max_retries=3)."""
    client_kwargs = {
        "api_key": config["api_key"],
        "http_client": http_client,
        "timeout": 300.0,
        "max_retries": 3,
    }
    if config["openai_base"]:
        client_kwargs["base_url"] = config["openai_base"]
    return openai.AsyncOpenAI(**client_kwargs)


def _translate(exc: Exception) -> Exception:
    if isinstance(exc, openai.RateLimitError):
        return ProviderRateLimitError(str(exc))
    if isinstance(exc, openai.BadRequestError):
        return ProviderBadRequestError(str(exc))
    if isinstance(exc, openai.APIError):
        return ProviderError(str(exc))
    return exc


async def _translate_stream_errors(raw_stream):
    """Wrap a raw SDK stream so mid-iteration openai.* errors surface as
    Provider* to every consumer (_AIProviderStreamProxy, openai_tool_handler).
    Forwards aclose() to raw_stream in `finally` so closing this wrapper
    (e.g. via _AIProviderStreamProxy.aclose()) still closes the underlying
    SDK stream, not just this generator."""
    try:
        async for chunk in raw_stream:
            yield chunk
    except openai.APIError as exc:
        raise ProviderStreamError(str(exc)) from exc
    finally:
        aclose = getattr(raw_stream, "aclose", None)
        if callable(aclose):
            await aclose()


def raw_chat_completion(get_client):
    """Build the CreateChatCompletion callable used by the production
    OpenAICompatibleProvider. `get_client` is called fresh on every request
    (not captured once) so tests that reassign `helper.client` after
    construction keep working unchanged. No retry here — SDK max_retries=3
    (build_openai_client) is the only retry layer."""
    async def _create(**kwargs):
        client = get_client()
        try:
            response = await client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise _translate(exc) from exc
        if kwargs.get("stream"):
            return _translate_stream_errors(response)
        return response
    return _create


@dataclass(frozen=True, slots=True)
class OpenAIStreamToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


class OpenAICompatibleProvider:
    def __init__(
        self,
        create_chat_completion: CreateChatCompletion,
        *,
        provider_name: str = "openai-compatible",
        get_client=None,
        get_gateway_client=None,
    ):
        self._create_chat_completion = create_chat_completion
        self.provider_name = provider_name
        self._get_client = get_client
        self._get_gateway_client = get_gateway_client

    async def stream_response(self, request: AIProviderRequest) -> AsyncIterator[AIEvent]:
        yield AIResponseStart(model=request.model, provider=self.provider_name)
        response = await self.create_response(request)
        if request.stream:
            async for event in _streaming_events(response):
                yield event
            return
        for event in _response_events(response):
            yield event

    async def create_response(self, request: AIProviderRequest) -> Any:
        return await self._create_chat_completion(**_request_to_kwargs(request))

    async def generate_image(self, **kwargs: Any) -> Any:
        try:
            return await self._get_client().images.generate(**kwargs)
        except openai.APIError as exc:
            raise _translate(exc) from exc

    async def list_models(self) -> Any:
        try:
            return await self._get_client().models.list()
        except openai.APIError as exc:
            raise _translate(exc) from exc

    async def speech(self, **kwargs: Any) -> Any:
        try:
            return await self._get_client().audio.speech.create(**kwargs)
        except openai.APIError as exc:
            raise _translate(exc) from exc

    async def transcribe(self, **kwargs: Any) -> Any:
        try:
            return await self._get_client().audio.transcriptions.create(**kwargs)
        except openai.APIError as exc:
            raise _translate(exc) from exc

    async def edit_image(self, **kwargs: Any) -> Any:
        return await self._get_gateway_client().image_edit_file(**kwargs)

    async def list_voices(self, **kwargs: Any) -> Any:
        return await self._get_gateway_client().audio_voices(**kwargs)


def _request_to_kwargs(request: AIProviderRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": list(request.messages),
        "stream": request.stream,
    }
    if request.tools is not None:
        if isinstance(request.tools, Mapping):
            kwargs["tools"] = dict(request.tools)
        elif isinstance(request.tools, (tuple, list)):
            kwargs["tools"] = list(request.tools)
        else:
            kwargs["tools"] = request.tools
    if request.tool_choice is not None:
        kwargs["tool_choice"] = request.tool_choice
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.response_format is not None:
        kwargs["response_format"] = dict(request.response_format)
    if request.extra_headers is not None:
        kwargs["extra_headers"] = dict(request.extra_headers)
    if request.extra:
        kwargs.update(dict(request.extra))
    return kwargs


def _response_events(response: Any) -> tuple[AIEvent, ...]:
    raw_choices = getattr(response, "choices", None) or []
    if not raw_choices:
        return (AIResponseEnd(message=AIMessage(role="assistant"), usage=_usage(response)),)

    events: list[AIEvent] = []
    for index, choice in enumerate(raw_choices):
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        tool_calls = _message_tool_calls(message)
        if index == 0:
            if isinstance(content, str) and content:
                events.append(AITextDelta(content))
            events.extend(AIToolCallReceived(tool_call) for tool_call in tool_calls)
        events.append(
            AIResponseEnd(
                message=AIMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=getattr(choice, "finish_reason", None),
                usage=_usage(response),
            )
        )
    return tuple(events)


async def _streaming_events(response: Any) -> AsyncIterator[AIEvent]:
    recorder = OpenAIStreamEventRecorder()
    async for chunk in response:
        for event in recorder.record_chunk(chunk):
            yield event
    for event in recorder.final_events():
        yield event


def stream_chunk_has_choice(chunk: Any) -> bool:
    return _first_choice(chunk) is not None


def stream_chunk_text_delta(chunk: Any) -> str:
    choice = _first_choice(chunk)
    if choice is None:
        return ""
    delta = getattr(choice, "delta", None)
    content = getattr(delta, "content", "")
    return content if isinstance(content, str) else ""


def stream_chunk_finish_reason(chunk: Any) -> str | None:
    choice = _first_choice(chunk)
    if choice is None:
        return None
    finish_reason = getattr(choice, "finish_reason", None)
    return finish_reason if isinstance(finish_reason, str) else None


def stream_chunk_tool_call_deltas(chunk: Any) -> tuple[OpenAIStreamToolCallDelta, ...]:
    choice = _first_choice(chunk)
    if choice is None:
        return ()
    delta = getattr(choice, "delta", None)
    result: list[OpenAIStreamToolCallDelta] = []
    for raw_tool_call in getattr(delta, "tool_calls", None) or ():
        function = getattr(raw_tool_call, "function", None)
        result.append(OpenAIStreamToolCallDelta(
            index=_int_or_zero(getattr(raw_tool_call, "index", 0)),
            id=getattr(raw_tool_call, "id", None),
            name=getattr(function, "name", None),
            arguments=getattr(function, "arguments", None),
        ))
    return tuple(result)


def _first_choice(response: Any) -> Any | None:
    choices = getattr(response, "choices", None) or []
    return choices[0] if choices else None


def _message_tool_calls(message: Any) -> tuple[AIToolCall, ...]:
    raw_tool_calls = getattr(message, "tool_calls", None) or []
    tool_calls: list[AIToolCall] = []
    for index, raw_call in enumerate(raw_tool_calls):
        function = getattr(raw_call, "function", None)
        name = getattr(function, "name", "") or ""
        arguments = getattr(function, "arguments", None)
        if not isinstance(arguments, str):
            arguments = "{}"
        tool_calls.append(
            AIToolCall(
                id=getattr(raw_call, "id", None) or f"tool_call_{index}",
                name=name,
                model_name=name,
                arguments=arguments,
            )
        )
    return tuple(tool_calls)


@dataclass
class _ToolCallStreamBuilder:
    id: str = ""
    name: str = ""
    arguments_parts: list[str] = field(default_factory=list)

    def add_delta(self, raw_tool_call: Any) -> None:
        call_id = getattr(raw_tool_call, "id", None)
        if isinstance(call_id, str) and call_id:
            self.id = call_id
        name = getattr(raw_tool_call, "name", None)
        if isinstance(name, str) and name:
            self.name = name
        arguments = getattr(raw_tool_call, "arguments", None)
        if isinstance(arguments, str):
            self.arguments_parts.append(arguments)

    def build(self, index: int) -> AIToolCall:
        arguments = "".join(self.arguments_parts) or "{}"
        return AIToolCall(
            id=self.id or f"tool_call_{index}",
            name=self.name,
            model_name=self.name,
            arguments=arguments,
        )


class OpenAIStreamEventRecorder:
    def __init__(self):
        self.content_parts: list[str] = []
        self.tool_call_builders: dict[int, _ToolCallStreamBuilder] = {}
        self.finish_reason: str | None = None

    def record_chunk(self, chunk: Any) -> tuple[AIEvent, ...]:
        if not stream_chunk_has_choice(chunk):
            return ()
        self.finish_reason = stream_chunk_finish_reason(chunk) or self.finish_reason
        content = stream_chunk_text_delta(chunk)
        events: list[AIEvent] = []
        if content:
            self.content_parts.append(content)
            events.append(AITextDelta(content))
        for raw_tool_call in stream_chunk_tool_call_deltas(chunk):
            index = raw_tool_call.index
            builder = self.tool_call_builders.setdefault(index, _ToolCallStreamBuilder())
            builder.add_delta(raw_tool_call)
        return tuple(events)

    def tool_calls(self) -> tuple[AIToolCall, ...]:
        return tuple(
            builder.build(index)
            for index, builder in sorted(self.tool_call_builders.items())
        )

    def final_events(self) -> tuple[AIEvent, ...]:
        tool_calls = self.tool_calls()
        events: list[AIEvent] = [
            AIToolCallReceived(tool_call) for tool_call in tool_calls
        ]
        events.append(
            AIResponseEnd(
                message=AIMessage(
                    role="assistant",
                    content="".join(self.content_parts) or None,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=self.finish_reason,
            )
        )
        return tuple(events)


def _usage(response: Any) -> AIUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return AIUsage(
        prompt_tokens=_int_or_none(getattr(usage, "prompt_tokens", None)),
        completion_tokens=_int_or_none(getattr(usage, "completion_tokens", None)),
        total_tokens=_int_or_zero(getattr(usage, "total_tokens", 0)),
    )


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _int_or_none(value: Any) -> int | None:
    """Как _int_or_zero, но не подменяет отсутствующее значение нулём.

    bot.pricing.resolve_chat_cost() читает None как "разбивка неизвестна" и
    считает по model_blended; 0 читается как "известно, что токенов не
    было" и уводит в model_split с нулевой ценой. См.
    docs/remediation_2026-09-04/T06-usage-none.md.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
