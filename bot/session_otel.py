"""Optional OpenTelemetry trace export for session logging.

This module is a second, optional sink on top of the JSONL session log
(`bot/session_logger.py`). JSONL remains the source of truth and this module
must never affect it: any failure here is caught and logged, never raised.

Design:
  - `build_otel_bridge(endpoint, ...)` returns a `_NullOtelBridge` (no-op) when
    tracing export is disabled/unavailable/misconfigured, or a real bridge
    backed by an OpenTelemetry `TracerProvider` held locally on the bridge
    instance (never registered globally via `trace.set_tracer_provider`).
  - The bridge turns a stream of session-log events (dicts, same shape as the
    ones written to JSONL) into spans: one root `bot.turn` span per turn_id,
    plus child spans for llm/tool/mutator work, plus `add_event` entries for
    everything else attached to the current turn span (if any).
  - Attributes copied onto spans are allow-listed per event type (default
    deny) to avoid leaking payload/arguments/response text that JSONL is
    allowed to carry but a wider-audience tracing backend should not receive.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SPAN_MAX_ATTR_CHARS = 2000
_TRUNCATION_SUFFIX = "...[truncated]"

# Event types that create/close their own span.
_TURN_START = "turn_start"
_TURN_END = "turn_end"
_LLM_TYPES = ("llm_call", "llm_error")
_TOOL_EXEC = "tool_exec"
_MUTATORS = "mutators"


def _truncate(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    cut = max(0, max_chars - len(_TRUNCATION_SUFFIX))
    return value[:cut] + _TRUNCATION_SUFFIX


# ---------------------------------------------------------------------------
# PII allow-list: per event type, extract only the safe subset of fields.
# Any event type not listed here (including future/unknown types, and
# tool_execution_start/tool_execution_end which are deliberately excluded to
# avoid duplicating tool_exec) falls back to the default-deny path: only
# `type`/`ts` are recorded.
# ---------------------------------------------------------------------------

def _extract_turn_start_attrs(event: dict) -> dict:
    user_message = event.get('user_message')
    return {
        'model_requested': event.get('model_requested'),
        'chat_id': event.get('chat_id'),
        'user_message_chars': len(user_message) if isinstance(user_message, str) else 0,
    }


def _extract_turn_end_attrs(event: dict) -> dict:
    return {
        key: event.get(key)
        for key in ('round_trips', 'llm_ms_total', 'mutator_ms_total', 'wall_ms')
        if key in event
    }


def _extract_llm_request_attrs(event: dict) -> dict:
    return {'kind': event.get('kind'), 'model': event.get('model')}


def _extract_llm_call_attrs(event: dict) -> dict:
    keys = (
        'kind', 'model', 'duration_ms', 'stream', 'prompt_tokens', 'completion_tokens',
        'total_tokens', 'finish_reason', 'tool_calls_returned', 'tools_count', 'error',
    )
    return {key: event.get(key) for key in keys if key in event}


def _extract_tool_exec_attrs(event: dict) -> dict:
    arguments = event.get('arguments')
    attrs = {
        'name': event.get('name'),
        'duration_ms': event.get('duration_ms'),
        'ok': event.get('ok'),
        'arguments_chars': len(arguments) if isinstance(arguments, str) else 0,
    }
    if event.get('error') is not None:
        attrs['error'] = event.get('error')
    return attrs


def _extract_mutators_attrs(event: dict) -> dict:
    return {'duration_ms': event.get('duration_ms'), 'injected_count': event.get('injected_count')}


def _extract_assistant_response_attrs(event: dict) -> dict:
    text = event.get('text')
    return {'text_chars': len(text) if isinstance(text, str) else 0}


def _extract_provider_error_attrs(event: dict) -> dict:
    data = event.get('data') if isinstance(event.get('data'), dict) else {}
    attrs = {'message': event.get('message'), 'recoverable': event.get('recoverable')}
    for key in ('kind', 'provider', 'stage'):
        if key in data:
            attrs[key] = data[key]
    return attrs


def _extract_retry_attrs(event: dict) -> dict:
    data = event.get('data') if isinstance(event.get('data'), dict) else {}
    attrs = {
        'message': event.get('message'),
        'attempt': event.get('attempt'),
        'max_attempts': event.get('max_attempts'),
        'delay_seconds': event.get('delay_seconds'),
    }
    if 'stage' in data:
        attrs['stage'] = data['stage']
    return attrs


def _extract_ai_provider_response_attrs(event: dict) -> dict:
    keys = ('kind', 'provider', 'text_chars', 'tool_call_count', 'error_count', 'finish_reason')
    return {key: event.get(key) for key in keys if key in event}


_EVENT_ATTR_EXTRACTORS = {
    _TURN_START: _extract_turn_start_attrs,
    _TURN_END: _extract_turn_end_attrs,
    'llm_request': _extract_llm_request_attrs,
    'llm_call': _extract_llm_call_attrs,
    'llm_error': _extract_llm_call_attrs,
    _TOOL_EXEC: _extract_tool_exec_attrs,
    _MUTATORS: _extract_mutators_attrs,
    'assistant_response': _extract_assistant_response_attrs,
    'provider_error': _extract_provider_error_attrs,
    'retry': _extract_retry_attrs,
    'ai_provider_response': _extract_ai_provider_response_attrs,
}


def _safe_attrs(event: dict, max_chars: int) -> dict:
    """Return the allow-listed, truncated attributes for one event.

    Unknown event types (not in `_EVENT_ATTR_EXTRACTORS`) yield an empty
    dict here — callers add `type`/`ts` themselves where relevant.
    """
    extractor = _EVENT_ATTR_EXTRACTORS.get(event.get('type'))
    if extractor is None:
        return {}
    raw = extractor(event) or {}
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, str):
            value = _truncate(value, max_chars)
        elif not isinstance(value, (bool, int, float)):
            value = _truncate(str(value), max_chars)
        result[key] = value
    return result


# ---------------------------------------------------------------------------
# No-op bridge
# ---------------------------------------------------------------------------

class _NullOtelBridge:
    """No-op bridge used when OTel export is disabled, unavailable, or failed to init."""

    def on_event(self, event: dict, user_id, session_id, turn_id) -> None:
        return None

    async def force_flush(self, timeout_ms: int = 5000) -> None:
        return None

    async def shutdown(self, timeout_ms: int = 5000) -> None:
        return None


_NULL_BRIDGE = _NullOtelBridge()


# ---------------------------------------------------------------------------
# Real bridge
# ---------------------------------------------------------------------------

class _OtelBridge:
    def __init__(
        self,
        *,
        provider,
        tracer,
        set_span_in_context,
        status_cls,
        status_code_cls,
        span_max_attr_chars: int = _DEFAULT_SPAN_MAX_ATTR_CHARS,
    ):
        self._provider = provider
        self._tracer = tracer
        self._set_span_in_context = set_span_in_context
        self._status_cls = status_cls
        self._status_code_cls = status_code_cls
        self._span_max_attr_chars = span_max_attr_chars
        self._turn_spans: dict[Any, Any] = {}
        self._shut_down = False

    # -- public API -----------------------------------------------------

    def on_event(self, event: dict, user_id, session_id, turn_id) -> None:
        try:
            self._handle_event(event, user_id, session_id, turn_id)
        except Exception:
            # Read the type defensively: this runs while already handling a
            # failure, so it must not raise a second one on a malformed event.
            event_type = event.get('type') if isinstance(event, dict) else None
            logger.debug(
                "session_otel: failed to process event type=%s", event_type, exc_info=True,
            )

    async def force_flush(self, timeout_ms: int = 5000) -> None:
        try:
            await asyncio.to_thread(self._provider.force_flush, timeout_ms)
        except Exception:
            logger.debug("session_otel: force_flush failed", exc_info=True)

    async def shutdown(self, timeout_ms: int = 5000) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        try:
            now_ns = time.time_ns()
            for turn_id, span in list(self._turn_spans.items()):
                try:
                    span.set_attribute('orphaned', True)
                    span.set_status(self._status_cls(self._status_code_cls.ERROR, 'turn_end never received'))
                    span.end(end_time=now_ns)
                except Exception:
                    logger.debug(
                        "session_otel: failed to close orphaned span turn_id=%s", turn_id, exc_info=True,
                    )
                finally:
                    self._turn_spans.pop(turn_id, None)
            await self.force_flush(timeout_ms)
            await asyncio.to_thread(self._provider.shutdown)
        except Exception:
            logger.debug("session_otel: shutdown failed", exc_info=True)

    # -- event routing ----------------------------------------------------

    def _handle_event(self, event: dict, user_id, session_id, turn_id) -> None:
        etype = event.get('type')
        now_ns = time.time_ns()
        if etype == _TURN_START:
            self._start_turn(event, user_id, session_id, turn_id, now_ns)
        elif etype == _TURN_END:
            self._end_turn(event, turn_id, now_ns)
        elif etype in _LLM_TYPES:
            self._add_llm_span(event, etype, turn_id, now_ns)
        elif etype == _TOOL_EXEC:
            self._add_tool_span(event, turn_id, now_ns)
        elif etype == _MUTATORS:
            self._add_mutators_span(event, turn_id, now_ns)
        else:
            # Includes tool_execution_start/tool_execution_end on purpose:
            # tool_exec already produces one span per tool call, so these do
            # not get their own spans (would double up 3 spans per call).
            self._add_event_on_parent(event, etype, turn_id, now_ns)

    # -- span builders ------------------------------------------------------

    def _start_turn(self, event: dict, user_id, session_id, turn_id, now_ns: int) -> None:
        if turn_id is None:
            return
        span = self._tracer.start_span('bot.turn', start_time=now_ns)
        span.set_attribute('openinference.span.kind', 'AGENT')
        if user_id is not None:
            span.set_attribute('user_id', str(user_id))
        if session_id is not None:
            span.set_attribute('session_id', str(session_id))
        for key, value in _safe_attrs(event, self._span_max_attr_chars).items():
            span.set_attribute(key, value)
        self._turn_spans[turn_id] = span

    def _end_turn(self, event: dict, turn_id, now_ns: int) -> None:
        if turn_id is None:
            return
        span = self._turn_spans.pop(turn_id, None)
        if span is None:
            return
        for key, value in _safe_attrs(event, self._span_max_attr_chars).items():
            span.set_attribute(key, value)
        span.end(end_time=now_ns)

    def _start_time_from_duration(self, event: dict, now_ns: int) -> int:
        duration_ms = event.get('duration_ms') or 0
        return now_ns - int(duration_ms * 1_000_000)

    def _add_llm_span(self, event: dict, etype: str, turn_id, now_ns: int) -> None:
        if turn_id is None:
            return
        parent = self._turn_spans.get(turn_id)
        start_ns = self._start_time_from_duration(event, now_ns)
        kind = event.get('kind') or 'unknown'
        ctx = self._set_span_in_context(parent) if parent is not None else None
        span = self._tracer.start_span(f'llm.{kind}', context=ctx, start_time=start_ns)
        span.set_attribute('openinference.span.kind', 'LLM')
        for key, value in _safe_attrs(event, self._span_max_attr_chars).items():
            span.set_attribute(key, value)
        if etype == 'llm_error':
            message = _truncate(str(event.get('error') or ''), self._span_max_attr_chars)
            span.set_status(self._status_cls(self._status_code_cls.ERROR, message))
        span.end(end_time=now_ns)

    def _add_tool_span(self, event: dict, turn_id, now_ns: int) -> None:
        if turn_id is None:
            return
        parent = self._turn_spans.get(turn_id)
        start_ns = self._start_time_from_duration(event, now_ns)
        name = event.get('name') or 'unknown'
        ctx = self._set_span_in_context(parent) if parent is not None else None
        span = self._tracer.start_span(f'tool.{name}', context=ctx, start_time=start_ns)
        span.set_attribute('openinference.span.kind', 'TOOL')
        for key, value in _safe_attrs(event, self._span_max_attr_chars).items():
            span.set_attribute(key, value)
        if event.get('ok') is False:
            message = _truncate(str(event.get('error') or ''), self._span_max_attr_chars)
            span.set_status(self._status_cls(self._status_code_cls.ERROR, message))
        span.end(end_time=now_ns)

    def _add_mutators_span(self, event: dict, turn_id, now_ns: int) -> None:
        if turn_id is None:
            return
        parent = self._turn_spans.get(turn_id)
        start_ns = self._start_time_from_duration(event, now_ns)
        ctx = self._set_span_in_context(parent) if parent is not None else None
        span = self._tracer.start_span('mutators', context=ctx, start_time=start_ns)
        span.set_attribute('openinference.span.kind', 'CHAIN')
        for key, value in _safe_attrs(event, self._span_max_attr_chars).items():
            span.set_attribute(key, value)
        span.end(end_time=now_ns)

    def _add_event_on_parent(self, event: dict, etype: str, turn_id, now_ns: int) -> None:
        if turn_id is None:
            return
        parent = self._turn_spans.get(turn_id)
        if parent is None:
            return
        attrs: dict[str, Any] = {'type': etype}
        if 'ts' in event:
            attrs['ts'] = event.get('ts')
        attrs.update(_safe_attrs(event, self._span_max_attr_chars))
        parent.add_event(etype or 'event', attributes=attrs, timestamp=now_ns)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_otel_bridge(
    endpoint: Optional[str],
    *,
    service_name: str,
    headers: Optional[dict] = None,
    insecure: bool = True,
    span_max_attr_chars: int = _DEFAULT_SPAN_MAX_ATTR_CHARS,
):
    """Build an OTel export bridge, or a no-op one if disabled/unavailable.

    `opentelemetry` is imported lazily, inside this function, and only after
    confirming `endpoint` is set — an empty/None endpoint returns the no-op
    bridge without importing the package at all.
    """
    if not endpoint:
        return _NULL_BRIDGE

    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.trace import set_span_in_context, Status, StatusCode
    except ImportError:
        logger.warning(
            "OTel export requested (endpoint=%s) but the opentelemetry packages are not "
            "installed; tracing export is disabled, JSONL session logging continues unaffected.",
            endpoint,
        )
        return _NULL_BRIDGE

    try:
        resource = Resource({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure, headers=headers)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        tracer = provider.get_tracer(service_name)
        return _OtelBridge(
            provider=provider,
            tracer=tracer,
            set_span_in_context=set_span_in_context,
            status_cls=Status,
            status_code_cls=StatusCode,
            span_max_attr_chars=span_max_attr_chars,
        )
    except Exception:
        logger.warning(
            "Failed to initialize OTel tracing bridge (endpoint=%s); tracing export is "
            "disabled, JSONL session logging continues unaffected.",
            endpoint,
            exc_info=True,
        )
        return _NULL_BRIDGE
