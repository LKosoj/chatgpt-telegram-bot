"""Tests for bot/session_otel.py — optional OTel export bridge for session logging.

`opentelemetry` packages are not assumed to be import-free in this test
process (another test, or the real package, might have already imported
them), so:
  - the "empty endpoint never imports opentelemetry" guarantee is checked in
    a fresh subprocess;
  - the "package genuinely missing" scenario is also checked in a fresh
    subprocess (poisoning `sys.modules['opentelemetry'] = None` only reliably
    blocks the import before anything under that name has been cached);
  - everything else fakes the opentelemetry module tree in-process via
    `sys.modules`, following the pattern used in test_agent_cron_plugin.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

from bot.session_logger import SessionLogger, clear_trace, set_trace

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fake opentelemetry object model — minimal, records calls for assertions.
# ---------------------------------------------------------------------------

class FakeSpan:
    def __init__(self, name, *, context=None, start_time=None):
        self.name = name
        self.context = context
        self.start_time = start_time
        self.attributes: dict = {}
        self.events: list[dict] = []
        self.status = None
        self.end_time = None
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def add_event(self, name, attributes=None, timestamp=None):
        self.events.append({"name": name, "attributes": attributes or {}, "timestamp": timestamp})

    def set_status(self, status):
        self.status = status

    def end(self, end_time=None):
        self.end_time = end_time
        self.ended = True


class FakeTracer:
    def __init__(self):
        self.spans: list[FakeSpan] = []
        self.raise_always = False

    def start_span(self, name, context=None, start_time=None, **kwargs):
        if self.raise_always:
            raise RuntimeError("boom: fake tracer failure")
        span = FakeSpan(name, context=context, start_time=start_time)
        self.spans.append(span)
        return span


class FakeTracerProvider:
    def __init__(self, resource=None):
        self.resource = resource
        self.processors = []
        self.tracer = FakeTracer()
        self.force_flush_calls: list[int] = []
        self.shutdown_calls = 0

    def add_span_processor(self, processor):
        self.processors.append(processor)

    def get_tracer(self, name, *args, **kwargs):
        return self.tracer

    def force_flush(self, timeout_millis=30000):
        self.force_flush_calls.append(timeout_millis)
        return True

    def shutdown(self):
        self.shutdown_calls += 1


class FakeBatchSpanProcessor:
    def __init__(self, exporter):
        self.exporter = exporter


class FakeResource:
    def __init__(self, attributes=None, schema_url=None):
        self.attributes = attributes


class FakeOTLPSpanExporter:
    def __init__(self, endpoint=None, insecure=None, headers=None, **kwargs):
        self.endpoint = endpoint
        self.insecure = insecure
        self.headers = headers


class FakeStatusCode:
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class FakeStatus:
    def __init__(self, status_code, description=None):
        self.status_code = status_code
        self.description = description


def _fake_set_span_in_context(span, context=None):
    return ("ctx-for", span)


@pytest.fixture
def fake_otel_modules(monkeypatch):
    """Install a fake `opentelemetry` package tree into sys.modules.

    Returns nothing; tests reach the fakes via `slog._otel._provider` /
    `slog._otel._tracer` after building a SessionLogger with an otel_endpoint.
    """
    otel = types.ModuleType("opentelemetry")
    sdk = types.ModuleType("opentelemetry.sdk")
    sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
    sdk_trace.TracerProvider = FakeTracerProvider
    sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
    sdk_trace_export.BatchSpanProcessor = FakeBatchSpanProcessor
    sdk_resources = types.ModuleType("opentelemetry.sdk.resources")
    sdk_resources.Resource = FakeResource
    exporter_pkg = types.ModuleType("opentelemetry.exporter")
    otlp_pkg = types.ModuleType("opentelemetry.exporter.otlp")
    proto_pkg = types.ModuleType("opentelemetry.exporter.otlp.proto")
    grpc_pkg = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc")
    trace_exporter_mod = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    trace_exporter_mod.OTLPSpanExporter = FakeOTLPSpanExporter
    otel_trace = types.ModuleType("opentelemetry.trace")
    otel_trace.set_span_in_context = _fake_set_span_in_context
    otel_trace.Status = FakeStatus
    otel_trace.StatusCode = FakeStatusCode

    modules = {
        "opentelemetry": otel,
        "opentelemetry.sdk": sdk,
        "opentelemetry.sdk.trace": sdk_trace,
        "opentelemetry.sdk.trace.export": sdk_trace_export,
        "opentelemetry.sdk.resources": sdk_resources,
        "opentelemetry.exporter": exporter_pkg,
        "opentelemetry.exporter.otlp": otlp_pkg,
        "opentelemetry.exporter.otlp.proto": proto_pkg,
        "opentelemetry.exporter.otlp.proto.grpc": grpc_pkg,
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": trace_exporter_mod,
        "opentelemetry.trace": otel_trace,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent_name, _, attr = name.rpartition(".")
        if parent_name in modules:
            setattr(modules[parent_name], attr, module)
    return modules


# ---------------------------------------------------------------------------
# 1. Empty endpoint -> no-op bridge, opentelemetry never imported.
# ---------------------------------------------------------------------------

_CODE_EMPTY_ENDPOINT = """
import sys
from bot.session_otel import build_otel_bridge, _NullOtelBridge

bridge = build_otel_bridge('', service_name='x')
assert isinstance(bridge, _NullOtelBridge), type(bridge)

leaked = [m for m in sys.modules if m == 'opentelemetry' or m.startswith('opentelemetry.')]
assert not leaked, leaked
print('OK')
"""


def test_empty_endpoint_returns_noop_without_importing_opentelemetry():
    result = subprocess.run(
        [sys.executable, "-c", _CODE_EMPTY_ENDPOINT],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# 2. Endpoint set, package missing -> no-op bridge + warning, JSONL unaffected.
# ---------------------------------------------------------------------------

_CODE_MISSING_PACKAGE = """
import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

# Poison the import before anything has cached opentelemetry.* in this
# (fresh) process, simulating "package not installed".
sys.modules['opentelemetry'] = None

from bot.session_logger import SessionLogger
from bot.session_otel import _NullOtelBridge

base_dir = sys.argv[1]
slog = SessionLogger(enabled=True, base_dir=base_dir, otel_endpoint='http://collector:4317')
assert isinstance(slog._otel, _NullOtelBridge), type(slog._otel)

slog.record({'type': 'turn_end', 'user_id': 'u1', 'session_id': 's1', 'wall_ms': 1})
asyncio.run(slog.drain())

assert os.path.exists(os.path.join(base_dir, 'u1', 's1.jsonl'))
print('OK')
"""


def test_endpoint_set_but_package_missing_returns_noop_with_warning(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", _CODE_MISSING_PACKAGE, str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout
    assert "opentelemetry" in result.stderr.lower()


# ---------------------------------------------------------------------------
# 3. Fake span/exporter raises -> record() does not raise, JSONL still written.
# ---------------------------------------------------------------------------

async def test_span_exception_does_not_break_record_or_jsonl(tmp_path, fake_otel_modules, caplog):
    slog = SessionLogger(enabled=True, base_dir=str(tmp_path), otel_endpoint='http://collector:4317')
    slog._otel._tracer.raise_always = True

    caplog.set_level(logging.DEBUG)
    token = set_trace('u1', 's1', 'turn-x')
    try:
        slog.record({'type': 'turn_start', 'model_requested': 'gpt-5', 'chat_id': 1})
    finally:
        clear_trace(token)

    await slog.drain()

    path = tmp_path / 'u1' / 's1.jsonl'
    assert path.exists()
    line = json.loads(path.read_text(encoding='utf-8').strip())
    assert line['type'] == 'turn_start'


# ---------------------------------------------------------------------------
# 4. One tool call -> exactly one tool span (dedup against
#    tool_execution_start/tool_execution_end).
# ---------------------------------------------------------------------------

async def test_tool_exec_produces_exactly_one_span(tmp_path, fake_otel_modules):
    slog = SessionLogger(enabled=True, base_dir=str(tmp_path), otel_endpoint='http://collector:4317')
    tracer = slog._otel._tracer

    token = set_trace('u1', 's1', 'turn-tool')
    try:
        slog.record({'type': 'turn_start', 'model_requested': 'gpt-5', 'chat_id': 1})
        slog.record({
            'type': 'tool_execution_start',
            'tool_call': {'id': 'call1', 'name': 'search', 'arguments': '{"secret": "abc"}', 'model_name': 'search'},
        })
        slog.record({
            'type': 'tool_execution_end',
            'result': {'tool_call_id': 'call1', 'name': 'search', 'ok': True, 'content': 'result text', 'error': None},
            'duration_ms': 42,
        })
        slog.record({
            'type': 'tool_exec', 'name': 'search', 'arguments': '{"secret": "abc"}',
            'duration_ms': 42, 'ok': True, 'error': None,
        })
        slog.record({'type': 'turn_end', 'round_trips': 1, 'llm_ms_total': 0, 'mutator_ms_total': 0, 'wall_ms': 50})
    finally:
        clear_trace(token)

    await slog.drain()

    assert len(tracer.spans) == 2  # bot.turn + tool.search only
    tool_spans = [s for s in tracer.spans if s.name.startswith('tool.')]
    assert len(tool_spans) == 1
    assert tool_spans[0].name == 'tool.search'

    turn_span = next(s for s in tracer.spans if s.name == 'bot.turn')
    event_names = [evt['name'] for evt in turn_span.events]
    assert 'tool_execution_start' in event_names
    assert 'tool_execution_end' in event_names

    # These two types are absent from the extractor table, so default-deny must
    # strip everything but type/ts — no raw arguments or tool output on the span.
    for evt in turn_span.events:
        if evt['name'] in ('tool_execution_start', 'tool_execution_end'):
            dumped = json.dumps(evt['attributes'], default=str)
            assert 'secret' not in dumped
            assert 'result text' not in dumped
            assert set(evt['attributes']) <= {'type', 'ts'}


# ---------------------------------------------------------------------------
# 5. Concurrency: two turns under asyncio.gather do not mix spans/attrs.
# ---------------------------------------------------------------------------

async def test_concurrent_turns_do_not_mix_llm_spans(tmp_path, fake_otel_modules):
    slog = SessionLogger(enabled=True, base_dir=str(tmp_path), otel_endpoint='http://collector:4317')
    tracer = slog._otel._tracer

    async def run_turn(turn_id, user_id, session_id, model):
        token = set_trace(user_id, session_id, turn_id)
        try:
            slog.record({'type': 'turn_start', 'model_requested': model, 'chat_id': user_id})
            await asyncio.sleep(0)
            slog.record({'type': 'llm_call', 'kind': 'chat', 'model': model, 'duration_ms': 5})
            await asyncio.sleep(0)
            slog.record({
                'type': 'turn_end', 'round_trips': 1, 'llm_ms_total': 5,
                'mutator_ms_total': 0, 'wall_ms': 10,
            })
        finally:
            clear_trace(token)

    await asyncio.gather(
        run_turn('t1', 1, 's1', 'model-a'),
        run_turn('t2', 2, 's2', 'model-b'),
    )

    roots = [s for s in tracer.spans if s.name == 'bot.turn']
    llm_spans = [s for s in tracer.spans if s.name.startswith('llm.')]
    assert len(roots) == 2
    assert len(llm_spans) == 2

    for llm_span in llm_spans:
        marker, parent_span = llm_span.context
        assert marker == 'ctx-for'
        assert parent_span in roots
        # The llm span's model must match the model of *its own* turn's root
        # span, proving turn_id-keyed parenting kept the two turns separate.
        assert llm_span.attributes.get('model') == parent_span.attributes.get('model_requested')


# ---------------------------------------------------------------------------
# 6. Lost turn_end: close() still ends the span, marked orphaned.
# ---------------------------------------------------------------------------

async def test_close_ends_orphaned_turn_span(tmp_path, fake_otel_modules):
    slog = SessionLogger(enabled=True, base_dir=str(tmp_path), otel_endpoint='http://collector:4317')
    tracer = slog._otel._tracer

    token = set_trace('u1', 's1', 'turn-orphan')
    try:
        slog.record({'type': 'turn_start', 'model_requested': 'gpt-5', 'chat_id': 1})
    finally:
        clear_trace(token)

    assert 'turn-orphan' in slog._otel._turn_spans

    await slog.close()

    assert 'turn-orphan' not in slog._otel._turn_spans
    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.ended is True
    assert span.attributes.get('orphaned') is True
    assert span.status is not None
    assert span.status.status_code == FakeStatusCode.ERROR


async def test_close_shuts_down_bridge_even_when_logging_disabled(tmp_path, fake_otel_modules):
    """A bridge is built from otel_endpoint alone, so `enabled=False` must not
    leave the provider (and its exporter thread) running after close()."""
    slog = SessionLogger(enabled=False, base_dir=str(tmp_path), otel_endpoint='http://collector:4317')
    provider = slog._otel._provider

    await slog.close()

    assert provider.shutdown_calls == 1


# ---------------------------------------------------------------------------
# 7. force_flush runs off the event loop (does not stall other coroutines).
# ---------------------------------------------------------------------------

async def test_force_flush_runs_off_the_event_loop(tmp_path, fake_otel_modules):
    slog = SessionLogger(enabled=True, base_dir=str(tmp_path), otel_endpoint='http://collector:4317')
    provider = slog._otel._provider

    started = threading.Event()
    release = threading.Event()

    def blocking_force_flush(timeout_millis=5000):
        started.set()
        assert release.wait(5), "release was not set in time"
        return True

    provider.force_flush = blocking_force_flush

    token = set_trace('u1', 's1', 'turn-flush')
    try:
        slog.record({'type': 'llm_call', 'kind': 'chat', 'model': 'gpt-5', 'duration_ms': 5})
    finally:
        clear_trace(token)

    counter = {"n": 0}

    async def counter_loop():
        while not release.is_set():
            counter["n"] += 1
            await asyncio.sleep(0)

    async def release_when_started():
        await asyncio.to_thread(started.wait, 5)
        release.set()

    counter_task = asyncio.create_task(counter_loop())
    await asyncio.gather(slog.flush_summary('u1', 's1'), release_when_started())
    await counter_task

    # The event loop kept running other coroutines while the fake blocking
    # force_flush executed in a worker thread — proof it wasn't awaited
    # directly on the loop.
    assert counter["n"] > 3
    assert provider.force_flush_calls == []  # replaced by blocking_force_flush; nothing recorded there


# ---------------------------------------------------------------------------
# 8. PII: secret string reaches JSONL (unchanged, current behavior) but not
#    span attributes/events.
# ---------------------------------------------------------------------------

async def test_secret_in_arguments_not_leaked_into_span_attributes(tmp_path, fake_otel_modules):
    slog = SessionLogger(enabled=True, base_dir=str(tmp_path), otel_endpoint='http://collector:4317')
    tracer = slog._otel._tracer

    secret = 'sk-super-secret-token-999'
    token = set_trace('u1', 's1', 'turn-pii')
    try:
        slog.record({'type': 'turn_start', 'model_requested': 'gpt-5', 'chat_id': 1})
        slog.record({
            'type': 'tool_exec', 'name': 'search',
            'arguments': json.dumps({'api_key': secret}),
            'duration_ms': 10, 'ok': True, 'error': None,
        })
        slog.record({
            'type': 'turn_end', 'round_trips': 0, 'llm_ms_total': 0,
            'mutator_ms_total': 0, 'wall_ms': 5,
        })
    finally:
        clear_trace(token)

    await slog.drain()

    jsonl_path = tmp_path / 'u1' / 's1.jsonl'
    jsonl_text = jsonl_path.read_text(encoding='utf-8')
    assert secret in jsonl_text  # current JSONL behavior — must not regress

    for span in tracer.spans:
        assert secret not in json.dumps(span.attributes, default=str)
        for evt in span.events:
            assert secret not in json.dumps(evt.get('attributes') or {}, default=str)
