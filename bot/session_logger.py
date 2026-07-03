from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections import namedtuple
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trace context
# ---------------------------------------------------------------------------

TraceContext = namedtuple('TraceContext', ['user_id', 'session_id', 'turn_id'])
_TRACE: ContextVar[Optional[TraceContext]] = ContextVar('session_log_trace', default=None)


def set_trace(user_id, session_id, turn_id):
    """Set trace context; returns the token from ContextVar.set()."""
    return _TRACE.set(TraceContext(user_id, session_id, turn_id))


def get_trace() -> Optional[TraceContext]:
    return _TRACE.get()


def clear_trace(token=None) -> None:
    if token is not None:
        _TRACE.reset(token)
    else:
        _TRACE.set(None)


# ---------------------------------------------------------------------------
# File name sanitisation
# ---------------------------------------------------------------------------

_SAFE_CHARS_RE = re.compile(r'[^A-Za-z0-9_\-]')


def _sanitize_id(value) -> str:
    if value is None:
        return '_unknown_'
    result = _SAFE_CHARS_RE.sub('_', str(value))
    return result or '_unknown_'


# ---------------------------------------------------------------------------
# In-memory stats aggregator
# ---------------------------------------------------------------------------

@dataclass
class _SessionStats:
    turns: int = 0
    llm_total: int = 0
    llm_by_kind: dict = field(default_factory=dict)
    llm_ms_total: float = 0.0
    llm_ms_max: float = 0.0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    tools_total: int = 0
    tools_ms_total: float = 0.0
    tools_by_name: dict = field(default_factory=dict)
    mutator_ms_total: float = 0.0
    wall_ms_total: float = 0.0

    def to_dict(self, user_id, session_id) -> dict:
        return {
            'user_id': user_id,
            'session_id': session_id,
            'turns': self.turns,
            'llm_calls': {
                'total': self.llm_total,
                'by_kind': dict(self.llm_by_kind),
            },
            'llm_ms': {
                'total': self.llm_ms_total,
                'max': self.llm_ms_max,
                'avg': self.llm_ms_total / self.llm_total if self.llm_total else 0.0,
            },
            'tokens': {
                'prompt_total': self.prompt_tokens_total,
                'completion_total': self.completion_tokens_total,
            },
            'tools': {
                'total': self.tools_total,
                'ms_total': self.tools_ms_total,
                'by_name': dict(self.tools_by_name),
            },
            'mutator_ms_total': self.mutator_ms_total,
            'wall_ms_total': self.wall_ms_total,
            'updated_ts': int(time.time() * 1000),
        }


# ---------------------------------------------------------------------------
# Module-level file helpers (run in thread via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _rotated_path(path: str) -> str:
    base = f"{path}.{int(time.time() * 1000)}"
    candidate = base
    counter = 1
    while os.path.exists(candidate):
        candidate = f"{base}.{counter}"
        counter += 1
    return candidate


def _write_line(path: str, line: str, max_file_bytes: int = 0) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if max_file_bytes > 0 and os.path.exists(path):
        current_size = os.path.getsize(path)
        incoming_size = len(line.encode('utf-8'))
        if current_size > 0 and current_size + incoming_size > max_file_bytes:
            os.replace(path, _rotated_path(path))
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(line)


def _write_summary(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _cleanup_old_logs(base_dir: str, retention_seconds: int) -> int:
    if retention_seconds <= 0 or not os.path.isdir(base_dir):
        return 0
    cutoff = time.time() - retention_seconds
    deleted = 0
    for root, _dirs, files in os.walk(base_dir):
        for file_name in files:
            if '.jsonl' not in file_name and not file_name.endswith('.summary.json'):
                continue
            path = os.path.join(root, file_name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    deleted += 1
            except FileNotFoundError:
                continue
            except OSError:
                logger.debug("Failed to remove old session log %s", path, exc_info=True)
    return deleted


@dataclass
class _WriteItem:
    path: str
    line: str


@dataclass
class _DrainItem:
    future: asyncio.Future


@dataclass
class _StopItem:
    future: asyncio.Future


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SessionLogger:
    def __init__(
        self,
        enabled: bool,
        base_dir: str,
        max_file_bytes: int = 0,
        retention_seconds: int = 0,
        cleanup_interval_seconds: int = 3600,
    ):
        self.enabled = enabled
        self.base_dir = base_dir
        self._stats: dict = {}      # (safe_uid, safe_sid) -> _SessionStats
        self.max_file_bytes = max(0, int(max_file_bytes or 0))
        self.retention_seconds = max(0, int(retention_seconds or 0))
        self.cleanup_interval_seconds = max(0, int(cleanup_interval_seconds or 0))
        self._queue: asyncio.Queue | None = None
        self._writer_task: asyncio.Task | None = None
        self._summary_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._summary_dirty: set[tuple[str, str]] = set()
        self._last_cleanup_monotonic: float | None = None
        self._closed = False

    def _ensure_writer(self) -> asyncio.Queue:
        asyncio.get_running_loop()
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._writer_task is None or self._writer_task.done():
            self._writer_task = asyncio.create_task(
                self._writer_loop(),
                name="session_logger_writer",
            )
        return self._queue

    async def _maybe_cleanup_old_logs(self) -> None:
        if self.retention_seconds <= 0:
            return
        now = time.monotonic()
        if (
            self._last_cleanup_monotonic is not None
            and now - self._last_cleanup_monotonic < self.cleanup_interval_seconds
        ):
            return
        self._last_cleanup_monotonic = now
        await asyncio.to_thread(_cleanup_old_logs, self.base_dir, self.retention_seconds)

    async def _writer_loop(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            try:
                if isinstance(item, _WriteItem):
                    await self._maybe_cleanup_old_logs()
                    await asyncio.to_thread(_write_line, item.path, item.line, self.max_file_bytes)
                elif isinstance(item, _DrainItem):
                    if not item.future.done():
                        item.future.set_result(None)
                elif isinstance(item, _StopItem):
                    if not item.future.done():
                        item.future.set_result(None)
                    return
            except Exception as exc:
                logger.debug("session log writer failed", exc_info=True)
                if isinstance(item, (_DrainItem, _StopItem)) and not item.future.done():
                    item.future.set_exception(exc)
            finally:
                self._queue.task_done()

    def record(self, event: dict) -> None:
        if not self.enabled or self._closed:
            return

        event = dict(event)
        event.setdefault('ts', int(time.time() * 1000))

        # Resolve user_id / session_id from event, falling back to trace context
        user_id = event.get('user_id')
        session_id = event.get('session_id')
        if user_id is None or session_id is None:
            trace = get_trace()
            if trace is not None:
                if user_id is None:
                    user_id = trace.user_id
                if session_id is None:
                    session_id = trace.session_id

        safe_uid = _sanitize_id(user_id)
        safe_sid = _sanitize_id(session_id)

        # Update in-memory aggregator
        key = (safe_uid, safe_sid)
        if key not in self._stats:
            self._stats[key] = _SessionStats()
        stats = self._stats[key]
        etype = event.get('type')
        if etype == 'llm_call':
            stats.llm_total += 1
            kind = event.get('kind', 'unknown')
            stats.llm_by_kind[kind] = stats.llm_by_kind.get(kind, 0) + 1
            d = event.get('duration_ms', 0)
            stats.llm_ms_total += d
            if d > stats.llm_ms_max:
                stats.llm_ms_max = d
            stats.prompt_tokens_total += event.get('prompt_tokens', 0) or 0
            stats.completion_tokens_total += event.get('completion_tokens', 0) or 0
        elif etype == 'tool_exec':
            stats.tools_total += 1
            stats.tools_ms_total += event.get('duration_ms', 0)
            name = event.get('name', 'unknown')
            stats.tools_by_name[name] = stats.tools_by_name.get(name, 0) + 1
        elif etype == 'mutators':
            stats.mutator_ms_total += event.get('duration_ms', 0)
        elif etype == 'turn_end':
            stats.turns += 1
            stats.wall_ms_total += event.get('wall_ms', 0)
        # other types: no aggregation

        line = json.dumps(event, ensure_ascii=False, default=str) + '\n'
        path = os.path.join(self.base_dir, safe_uid, f'{safe_sid}.jsonl')

        try:
            queue = self._ensure_writer()
            queue.put_nowait(_WriteItem(path, line))
        except RuntimeError:
            # No running event loop — write synchronously as fallback
            _cleanup_old_logs(self.base_dir, self.retention_seconds)
            _write_line(path, line, self.max_file_bytes)

    async def flush_summary(self, user_id, session_id) -> None:
        if not self.enabled:
            return
        safe_uid = _sanitize_id(user_id)
        safe_sid = _sanitize_id(session_id)
        key = (safe_uid, safe_sid)
        # Снимаем агрегатор синхронно (pop до await): между to_dict() и await
        # конкурентный record() мог бы дописать инкременты в тот же объект
        # in-place, а последующий pop их потерял бы (lost update). Извлекая
        # объект сейчас, мы фиксируем снапшот; параллельный record() создаст
        # свежий агрегатор под этим ключом и попадёт в следующий summary.
        stats = self._stats.pop(key, None)
        if stats is None:
            return False
        summary = stats.to_dict(user_id, session_id)
        path = os.path.join(self.base_dir, safe_uid, f'{safe_sid}.summary.json')
        await asyncio.to_thread(_write_summary, path, summary)
        return True

    def schedule_flush_summary(self, user_id, session_id) -> bool:
        if not self.enabled or self._closed:
            return False
        safe_uid = _sanitize_id(user_id)
        safe_sid = _sanitize_id(session_id)
        key = (safe_uid, safe_sid)
        existing = self._summary_tasks.get(key)
        if existing is not None and not existing.done():
            self._summary_dirty.add(key)
            return True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._start_summary_task(loop, user_id, session_id, key)
        return True

    def _start_summary_task(self, loop, user_id, session_id, key) -> None:
        task = loop.create_task(self.flush_summary(user_id, session_id))
        self._summary_tasks[key] = task

        def _forget_done(done_task, *, summary_key=key):
            if self._summary_tasks.get(summary_key) is done_task:
                self._summary_tasks.pop(summary_key, None)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc is not None:
                logger.debug(
                    "session summary flush failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            if summary_key in self._summary_dirty and summary_key in self._stats and not self._closed:
                self._summary_dirty.discard(summary_key)
                self._start_summary_task(
                    loop,
                    user_id,
                    session_id,
                    summary_key,
                )
            else:
                self._summary_dirty.discard(summary_key)

        task.add_done_callback(_forget_done)

    async def _drain_summary_tasks(self) -> None:
        while self._summary_tasks:
            await asyncio.gather(*list(self._summary_tasks.values()), return_exceptions=True)
            await asyncio.sleep(0)

    async def drain(self) -> None:
        if not self.enabled or self._queue is None:
            await self._drain_summary_tasks()
            return
        writer = self._writer_task
        if writer is not None and not writer.done():
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            await self._queue.put(_DrainItem(future))
            await future
        await self._drain_summary_tasks()

    async def close(self) -> None:
        if not self.enabled:
            return
        if self._closed:
            await self.drain()
            return
        await self.drain()
        self._closed = True
        if self._queue is None:
            return
        writer = self._writer_task
        if writer is None or writer.done():
            return
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(_StopItem(future))
        await future
        await writer

    async def cleanup_old_logs(self) -> int:
        if not self.enabled or self.retention_seconds <= 0:
            return 0
        return await asyncio.to_thread(_cleanup_old_logs, self.base_dir, self.retention_seconds)
