from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import namedtuple
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

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

def _write_line(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(line)


def _write_summary(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SessionLogger:
    def __init__(self, enabled: bool, base_dir: str):
        self.enabled = enabled
        self.base_dir = base_dir
        self._stats: dict = {}      # (safe_uid, safe_sid) -> _SessionStats
        self._bg_tasks: set = set()

    def record(self, event: dict) -> None:
        if not self.enabled:
            return

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
            task = asyncio.create_task(asyncio.to_thread(_write_line, path, line))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        except RuntimeError:
            # No running event loop — write synchronously as fallback
            _write_line(path, line)

    async def flush_summary(self, user_id, session_id) -> None:
        if not self.enabled:
            return
        safe_uid = _sanitize_id(user_id)
        safe_sid = _sanitize_id(session_id)
        key = (safe_uid, safe_sid)
        stats = self._stats.get(key, _SessionStats())
        summary = stats.to_dict(user_id, session_id)
        path = os.path.join(self.base_dir, safe_uid, f'{safe_sid}.summary.json')
        await asyncio.to_thread(_write_summary, path, summary)

    async def drain(self) -> None:
        if self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)
