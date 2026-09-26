"""T11: per-chat state consolidated into one record per chat/session key.

Before this module, OpenAIHelper kept 9 parallel dicts (self.conversations,
self.loaded_conversation_sessions, self.last_updated, self._chat_request_models,
self._chat_request_usage_split, self._chat_request_extra_tokens, self._gate_fired,
self._last_summary_at, self.last_image_file_ids) plus a lazily-created
self._per_chat_locks dict, all keyed by the same chat/session key
(see OpenAIHelper._chat_state_key) and cleared together by _clear_chat_state.
Nothing enforced that "together" -- _gate_fired/_last_summary_at were never
actually cleared (a leak, fixed by this module: ChatStateRegistry.evict() drops
the whole record).

ChatStateRegistry replaces the 9 dicts with one OrderedDict of ConversationState
records. bot/openai_helper.py exposes the old attribute names again as
MutableMapping-backed properties (_FieldView, one per field) so that the ~250
existing read/write sites and the tests that build an OpenAIHelper via
object.__new__() and assign these dicts directly keep working unchanged.
"""

from __future__ import annotations

import asyncio
import datetime
import os
from collections import OrderedDict
from collections.abc import MutableMapping
from dataclasses import dataclass, field

_UNSET = object()  # sentinel: "this field was never assigned", distinct from a real None


def _positive_int_env(name: str, default: int) -> int:
    # mirrors bot/openai_tool_handler.py:58-62 (kept local, no new shared util)
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


DEFAULT_MAX_CHAT_STATES = _positive_int_env("MAX_CHAT_STATES", 1000)


@dataclass
class ConversationState:
    history: list | None = None
    session_id: object = _UNSET          # str | None once set; _UNSET means "never loaded"
    last_updated: datetime.datetime | None = None
    request_model: str | None = None
    # tuple[int, int] | None once set (None is itself a legitimate "unknown split"
    # value written at the top of every turn -- see openai_helper.py's
    # _chat_request_usage_split reset); _UNSET means "never touched this turn".
    usage_split: object = _UNSET
    # extra_tokens/gate_fired: 0/False double as both the default and the "unset"
    # sentinel below. Safe only because no code anywhere checks `key in` either of
    # the old dicts (verified across bot/ and tests/) -- if that ever changes, these
    # two need a real sentinel like session_id/usage_split above.
    extra_tokens: int = 0
    gate_fired: bool = False
    last_summary_at: int | None = None
    last_image_file_ids: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ChatStateRegistry:
    """One record per chat/session key, replacing 9 parallel dicts.

    `_states` is an OrderedDict for LRU: get_or_create()/peek() move an
    accessed key to the end; sweep() walks from the front (oldest first).
    """

    def __init__(self, *, max_states: int = DEFAULT_MAX_CHAT_STATES,
                 sweep_min_interval_seconds: float = 60.0):
        self._states: "OrderedDict[object, ConversationState]" = OrderedDict()
        self._max_states = max_states
        self._sweep_min_interval = sweep_min_interval_seconds
        self._last_swept_at: datetime.datetime | None = None

    def get_or_create(self, key) -> ConversationState:
        state = self._states.get(key)
        if state is None:
            state = ConversationState()
            self._states[key] = state
        self._states.move_to_end(key)
        return state

    def peek(self, key) -> ConversationState | None:
        state = self._states.get(key)
        if state is not None:
            self._states.move_to_end(key)
        return state

    def evict(self, key) -> None:
        self._states.pop(key, None)

    def maybe_sweep(self, now: datetime.datetime, *, max_age_minutes: float) -> int:
        """Throttled sweep -- cheap no-op unless sweep_min_interval has
        elapsed since the last real scan. Mirrors the existing
        self._tts_models_cache "(timestamp, value)" freshness-check pattern
        already used elsewhere in OpenAIHelper (openai_helper.py:289)."""
        if (self._last_swept_at is not None
                and (now - self._last_swept_at).total_seconds() < self._sweep_min_interval):
            return 0
        self._last_swept_at = now
        return self.sweep(now, max_age_minutes=max_age_minutes)

    def sweep(self, now: datetime.datetime, *, max_age_minutes: float) -> int:
        """Evict idle (older than max_age_minutes) and over-LRU-cap entries,
        skipping any entry whose lock is currently held. Returns count evicted."""
        evicted = 0
        max_age = datetime.timedelta(minutes=max_age_minutes)
        for key in list(self._states.keys()):  # oldest-first
            state = self._states.get(key)
            if state is None or state.lock.locked():
                continue
            idle = state.last_updated is not None and (now - state.last_updated) > max_age
            over_cap = len(self._states) > self._max_states
            if idle or over_cap:
                del self._states[key]
                evicted += 1
        return evicted

    def __len__(self) -> int:
        return len(self._states)


class _FieldView(MutableMapping):
    """dict-like view of one ConversationState field across a ChatStateRegistry.

    Backs the old dict-attribute names (conversations, _gate_fired, ...) that
    external code (bot/chat_run.py, tests building OpenAIHelper via
    object.__new__) still reads/writes as plain dicts.
    """

    def __init__(self, registry: ChatStateRegistry, field_name: str, unset):
        self._registry = registry
        self._field = field_name
        self._unset = unset

    def __getitem__(self, key):
        state = self._registry.peek(key)
        value = None if state is None else getattr(state, self._field)
        if state is None or value is self._unset:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value) -> None:
        setattr(self._registry.get_or_create(key), self._field, value)

    def __delitem__(self, key) -> None:
        state = self._registry.peek(key)
        if state is None or getattr(state, self._field) is self._unset:
            raise KeyError(key)
        setattr(state, self._field, self._unset)

    def __contains__(self, key) -> bool:
        state = self._registry.peek(key)
        return state is not None and getattr(state, self._field) is not self._unset

    def __iter__(self):
        # Materialized eagerly, not a lazy generator over the live OrderedDict:
        # Mapping mixins (dict(view), ==) interleave "pull next key" with
        # __getitem__ calls, and __getitem__ -> peek() -> move_to_end() mutates
        # _states mid-walk, which would raise "OrderedDict mutated during
        # iteration" on a lazy generator.
        return iter([k for k, s in self._registry._states.items()
                     if getattr(s, self._field) is not self._unset])

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def replace(self, new_mapping) -> None:
        """Backs `helper.<field-view> = {...}` whole-dict reassignment."""
        seen = set()
        for k, v in dict(new_mapping).items():
            self[k] = v
            seen.add(k)
        for k in list(self):
            if k not in seen:
                del self[k]
