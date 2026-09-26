# T11. Состояние чата (ConversationState + ChatStateRegistry) — implementation plan

Owner: developer implementing T11, wave 6. File ownership per master plan
(`docs/improvement_2026-09-25/00-master-plan.md:369-395`): `bot/conversation_state.py`
(new), `bot/openai_helper.py`, `bot/openai_tool_handler.py` (only the two `conversations`
touches), tests affected by the migration, `tests/test_no_private_helper_access.py`.
**Do not touch `bot/chat_run.py`, `bot/telegram_bot.py`, `bot/tool_result.py`, or any
plugin** — they are owned by other tasks/waves and read these dicts as private
attributes of a live `OpenAIHelper` instance; the compat-property layer exists
specifically so none of them need to change.

---

## 0. Inventory — every per-chat dict/attribute, current line, dataclass field

All line numbers verified by reading current `bot/openai_helper.py` (4177 lines) content,
not memory.

| current attribute | def line | type | new `ConversationState` field | presence semantics |
|---|---|---|---|---|
| `self.conversations` | `openai_helper.py:269` | `dict[int, list]` | `history: list \| None = None` | `None` = cold cache (checked via `in`/`not in` at 8+ sites, e.g. `:1006`, `:1286`, `:2228`, `:3400`) |
| `self.loaded_conversation_sessions` | `:270` | `dict[int, str\|None]` | `session_id: str\|None = _UNSET` | value `None` is a **legitimate loaded state** (`:2372/:2374` restore path), so needs a real sentinel, not `None`, to mean "unset" |
| `self.last_updated` | `:287` | `dict[int, datetime]` | `last_updated: datetime\|None = None` | `None` never a legitimate set value (`:3286` `__max_age_reached`, `:2355` snapshot) — safe sentinel |
| `self._chat_request_models` | `:272` | `dict[int, str]` | `request_model: str\|None = None` | no `in` checks anywhere in tree — `None` safe as both default and "unset" |
| `self._chat_request_usage_split` | `:278` | `dict[int, tuple[int,int]\|None]` | `usage_split: tuple[int,int]\|None = None` | `:1366` explicitly stores `None` as a real value, but **no code anywhere checks `in` this dict** (verified by grep) — `.get()`/`.pop()` return `None` either way, so no sentinel needed |
| `self._chat_request_extra_tokens` | `:279` | `dict[int, int]` | `extra_tokens: int = 0` | no presence checks |
| `self._gate_fired` | `:283` | `dict` (bool) | `gate_fired: bool = False` | no presence checks |
| `self._last_summary_at` | `:286` | `dict` (int) | `last_summary_at: int \| None = None` | no presence checks; **currently never cleared by `_clear_chat_state`/`evict` — leak, fixed by this task (master-plan step 3)** |
| `self.last_image_file_ids` | `:288` | `dict[int, str]` (singular value despite plural name — keep the name, don't "fix" it) | `last_image_file_ids: str \| None = None` | no presence checks |
| `self._per_chat_locks` (+ `self._chat_locks_guard`, both lazily created inside `_chat_lock()`, `:2929-2933`) | n/a (lazy) | `dict[int, asyncio.Lock]` | `lock: asyncio.Lock = field(default_factory=asyncio.Lock)` | `test_openai_helper_session_api.py:39/213/219` reads/writes `helper._per_chat_locks` directly — needs a compat view too, even though it's not in the master plan's explicit attribute list |

`_chat_locks_guard` (the double-checked-locking guard around `_per_chat_locks` creation,
`:2929-2932`) has **zero** external references (checked every test file and every
`bot/*.py`/`bot/plugins/*.py`) — it disappears entirely once `ChatStateRegistry.get_or_create`
is synchronous (a plain dict/`OrderedDict` operation has no `await` point, so the race it
guards against cannot occur; see §3).

**Read/write site counts** (attribute name substring match, so includes both reads and
writes; local variables named e.g. `extra_tokens` are included but harmless noise —
exact call sites for the risky ones are cited individually below):

| file | conversations | loaded_conversation_sessions | last_updated | _chat_request_models | usage_split | extra_tokens | _gate_fired | _last_summary_at | last_image_file_ids |
|---|---|---|---|---|---|---|---|---|---|
| `bot/openai_helper.py` | 78 | 26 | 15 | 8 | 11 | 23 | 4 | 5 | 7 |
| `bot/openai_tool_handler.py` | 2 (`:254`, `:1712`, via `_conversation_messages()` def `:253`) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `bot/chat_run.py` (**not owned — do not edit**) | 0 | 0 | 0 | 7 | 5 | 4 | 1 | 0 | 0 |
| `bot/tool_result.py` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `bot/telegram_bot.py` (**not owned**) | 3 (public API only) | 0 | 0 | 0 | 1 (public) | 0 | 0 | 0 | 0 |
| `bot/plugins/*.py` | 3 (`hindsight_memory.py`, unrelated local var + `evict` calls) | 0 | 2 (`task_management.py`, unrelated) | 0 | 0 | 0 | 0 | 0 | 0 |

`bot/chat_run.py:58,61,72,85,92,106,115,123,136,154,163,171,184,230,259` do direct
`helper._chat_request_models.get(...)`, `helper._chat_request_extra_tokens.pop(...)`,
`helper._chat_request_usage_split[key] = ...`, `helper._gate_fired.pop(...)`,
`helper._chat_state_key(chat_id)`. **This is the hard constraint that forces the
compat-view design** in §3 — `chat_run.py` is not in T11's file list, so these must keep
working unmodified through property-returned proxies supporting `.get`, `.pop(k, default)`,
`__setitem__`, exactly as today.

**Test file counts** (only files touching the **real** `OpenAIHelper` class; `SimpleNamespace`/
hand-rolled fake helpers are excluded below — see §7 classification):

| test file | how helper is built | relevant attrs touched |
|---|---|---|
| `tests/test_openai_helper_session_api.py` | `object.__new__(OpenAIHelper)` | all 9 dicts + `_per_chat_locks`, 82 total hits — **primary regression target** |
| `tests/test_openai_helper_tool_calls.py` | `object.__new__` (`_make_helper`) | `conversations` ×91, `loaded_conversation_sessions` ×3, `_with_chat_state` ×4 — **largest file, run first per commit** |
| `tests/test_openai_helper_summarize_trim.py` | `object.__new__` | `conversations` ×24, `_last_summary_at` ×2 |
| `tests/test_reset_chat_history_async.py` | `object.__new__` ×6 | `conversations` ×12, `loaded_conversation_sessions` ×6 |
| `tests/test_record_plugin_exchange.py` | `object.__new__` | `conversations`, `loaded_conversation_sessions`, `last_updated`, `max_conversation_age_minutes` |
| `tests/test_session_logging_integration.py` | `object.__new__` ×2 | `conversations` (whole-dict reassignment) |
| `tests/test_skills_agent_gate.py` | real helper | `conversations` ×2, `_gate_fired` ×7 |
| `tests/test_summarise_overflow_dispatch.py` | real helper | `conversations` ×6 |
| `tests/test_exemplar_summarize_failure_structure.py` | `object.__new__` | `conversations` ×7, `_last_summary_at` ×1 |
| `tests/test_exemplar_interrupted_tool_call_repair.py` | `object.__new__` | `conversations` ×7 |
| `tests/test_reflection_on_tool_error.py` | real helper | `conversations` ×6 |
| `tests/test_hindsight_memory.py` | real helper | `conversations` ×4 (direct `[1] = [...]`), `max_conversation_age_minutes` |
| `tests/test_stream_usage.py` | real helper | `helper._chat_request_usage_split[helper._chat_state_key(1)]` ×3 (direct subscript) |
| `tests/test_pricing.py` | real helper | `helper.get_last_chat_usage_split(1)` — public accessor only, low risk |
| `tests/test_no_private_helper_access.py` | AST scan, no instantiation | needs the extension described in §6 |

**Confirmed NOT affected** despite matching a substring grep — verified by reading each:
`tests/test_callback_authorization.py` (`SimpleNamespace` fake, `conversations={}` is a
plain dict kwarg, not real `OpenAIHelper`), `tests/test_plugin_chat_id_contract.py`
(`self.conversations` on a hand-written fake plugin/bot class), `tests/test_group_session_flow.py`
and `tests/test_per_conversation_serialization.py` (both use a `FakeHelper`/`SimpleNamespace`
exposing only `history_snapshot`/`load_session`/`replace_system_message`/`evict`/
`chat_state_scope` — confirmed by `test_openai_helper_session_api.py:3-6`'s own docstring —
so they only assert `bot/telegram_bot.py` *calls* these public names, unaffected by the
internal rewrite as long as signatures don't change), `tests/test_hindsight_burst_buffer.py`
and `tests/test_bounded_dicts_and_async_save.py` (the word "evict" in test names/docstrings,
unrelated dict: hindsight's own burst buffer and `telegram_bot.py`'s `_BoundedLRU`,
respectively — not `OpenAIHelper` state at all), `tests/test_telegram_streaming.py` and
`tests/test_telegram_usage_offload.py` (fake helpers defining their own
`get_last_chat_usage_split`).

---

## 1. Concurrency model (unchanged by this task — document, don't touch)

- **`chat_state_scope`/`_with_chat_state`** (`openai_helper.py:3037`, `:3045-3051`): a
  `contextvars.ContextVar` (`_CHAT_STATE_KEY`, module-level `:88`) overridden via a
  `@contextmanager`. `_chat_state_key(chat_id)` (`:2941-2942`) returns
  `_CHAT_STATE_KEY.get() or chat_id` — i.e. every dict lookup goes through this resolver, so
  the "key" used by the registry is never `chat_id` directly, it's this resolved
  `state_key`. This is exactly how "parallel processing of deferred messages in a new
  session" works (per docstring `:3038-3042`): a separate asyncio task can set the
  ContextVar to a synthetic session key for its own duration without colliding with the
  real `chat_id`'s cache. **No change needed** — `ChatStateRegistry` is keyed by whatever
  `_chat_state_key()` returns; it has no opinion on what a "key" means.
- **Per-chat lock** (`_chat_lock`, `:2916-2939`): today a manual double-checked-locking
  dance (`asyncio.Lock` guard around creating per-chat `asyncio.Lock`s) because two
  coroutines could race to create the first lock for a chat_id. Once creation is a
  `ChatStateRegistry.get_or_create()` call with **no `await` inside it**, this race is
  structurally impossible (asyncio is cooperative; a sync function without an `await`
  point can't be interleaved), so the guard is removed, not preserved (see §3, §5).
  `_chat_lock_bypass_enabled`/`_without_chat_lock` (`:3076-3086`, a *different* ContextVar,
  `_CHAT_LOCK_BYPASS_CHAT_ID`) are independent of storage and untouched.
- **`max_conversation_age_minutes`**: read fresh from `self.config['max_conversation_age_minutes']`
  at every call in `__max_age_reached` (`:3290`) — config is never mutated after
  `OpenAIHelper.__init__` (grepped for `self.config[...] = `, zero hits), but to stay
  byte-for-byte faithful, `OpenAIHelper` passes it into `sweep()`/`get_or_create()` **fresh
  on each call**, not snapshotted once into the registry at construction.
- **Core background loop**: there is **no** existing periodic task mechanism belonging to
  `OpenAIHelper` itself. `self._background_tasks` (`:271`) is a plain `set()` of
  fire-and-forget one-shot tasks (session-name generation, `:678-687`), drained on
  `close()` (`:4148-4153`) — not a scheduler. The only periodic-task framework in the tree
  is `PluginManager.start_background_tasks(application)`, invoked from
  `ChatGPTTelegramBot.post_init` (`bot/telegram_bot.py:5280`) for **plugin**-declared
  `BackgroundTask` entries — `telegram_bot.py` is not in T11's file ownership, so wiring a
  new core sweep loop through `post_init`/`_post_shutdown` is out of reach for this task
  even if it were otherwise desirable. See §4 for the resulting design decision.

---

## 2. `bot/conversation_state.py` (new file) — target design

```python
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
    usage_split: tuple[int, int] | None = None
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
```

Notes for the implementer:
- `sweep()` reads `len(self._states)` fresh inside the loop, so once the registry drops
  back under `max_states` it naturally stops evicting purely-for-capacity entries even
  mid-scan (idle entries keep being evicted regardless, since that check is independent).
- `max_age_minutes` and `max_states` are **parameters**, not attributes read from
  `OpenAIHelper.config` — keeps this file fully decoupled from `OpenAIHelper` (own tests
  don't need a helper instance at all, just a `ChatStateRegistry()`).
- `_UNSET` is exported (not prefixed unexported) only if `openai_helper.py`'s property
  setter for `loaded_conversation_sessions` needs to reference it directly (it does, to
  reset a key's `session_id` back to "unset" on whole-dict reassignment — see §3).

---

## 3. Compat view properties in `bot/openai_helper.py`

**The hard constraint**: 10 test files construct `OpenAIHelper` via
`object.__new__(OpenAIHelper)`, bypassing `__init__` entirely, then assign these
attributes directly as plain dicts (§0 table). Property getters/setters must therefore
**lazily** create the backing registry the same way `_chat_lock`'s `_chat_locks_guard`
and `set_last_image_file_id`'s `self.last_image_file_ids = getattr(self, 'last_image_file_ids', {})`
(`:3909`) already do — never assume `__init__` ran.

Add one small `_FieldView(MutableMapping)` in `bot/conversation_state.py` (justified by
real duplication: 9 near-identical dict-like proxies, not a single-use abstraction) plus a
tiny `_get_registry(helper)` lazy accessor, then 9 property pairs in `OpenAIHelper` (10th,
`_per_chat_locks`, is a bare property with no setter — no test reassigns the whole dict,
only individual keys). The sketch below is precise enough to implement directly but is not
literal final code — the implementer should write it, run `tests/test_conversation_state.py`
(§8) against it, and adjust (the `pop`/`get`/`setdefault`/`update`/`clear` mixins that
`MutableMapping` provides automatically only need `__getitem__`/`__setitem__`/
`__delitem__`/`__iter__`/`__len__` below; don't hand-write them unless a mixin's default
behavior turns out to be wrong for a specific field):

```python
# bot/conversation_state.py, added to the same module
class _FieldView(MutableMapping):
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

    def __setitem__(self, key, value):
        setattr(self._registry.get_or_create(key), self._field, value)

    def __delitem__(self, key):
        state = self._registry.peek(key)
        if state is None or getattr(state, self._field) is self._unset:
            raise KeyError(key)
        setattr(state, self._field, self._unset)

    def __contains__(self, key) -> bool:
        state = self._registry.peek(key)
        return state is not None and getattr(state, self._field) is not self._unset

    def __iter__(self):
        return (k for k, s in self._registry._states.items()
                if getattr(s, self._field) is not self._unset)

    def __len__(self):
        return sum(1 for _ in self)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key, *args):
        # MutableMapping's own pop() mixin already does exactly this via
        # __getitem__/__delitem__; listed here only to spell out that no
        # override is needed -- delete this method and rely on the mixin.
        return super().pop(key, *args)

    def replace(self, new_mapping) -> None:
        """Backs `helper.<field-view> = {...}` whole-dict reassignment."""
        seen = set()
        for k, v in dict(new_mapping).items():
            self[k] = v
            seen.add(k)
        for k in list(self):
            if k not in seen:
                del self[k]
```

`MutableMapping` gives `__eq__`/`__ne__` for free (compares as a mapping, item by item),
satisfying `assert helper.conversations == {}` (`test_openai_helper_session_api.py:228`,
`test_evict_is_idempotent_on_unknown_chat`) and `assert bot.openai.conversations == {}`-style
assertions without extra code. `pop`/`get`/`setdefault`/`update`/`clear` mixins from
`MutableMapping` cover every operation `chat_run.py` and `openai_helper.py` itself perform
(`.pop(key, default)`, `.get(key)`, `[key] = value`, `del d[key]`, `key in d`) — verified
site-by-site against every call listed in §0's `chat_run.py` line list.

In `OpenAIHelper`:

```python
def _get_chat_states(self) -> ChatStateRegistry:
    registry = self.__dict__.get('_chat_states')
    if registry is None:
        registry = ChatStateRegistry()
        self.__dict__['_chat_states'] = registry
    return registry

@property
def conversations(self):
    return _FieldView(self._get_chat_states(), 'history', None)

@conversations.setter
def conversations(self, value):
    _FieldView(self._get_chat_states(), 'history', None).replace(value)

# ... same pair for loaded_conversation_sessions (field 'session_id', unset=_UNSET),
# last_updated ('last_updated', None), _chat_request_models ('request_model', None),
# _chat_request_usage_split ('usage_split', None), _chat_request_extra_tokens
# ('extra_tokens', 0 -- see note below), _gate_fired ('gate_fired', False -- same note),
# _last_summary_at ('last_summary_at', None)

@property
def _per_chat_locks(self):
    return _FieldView(self._get_chat_states(), 'lock', None)  # 'unset' never occurs:
    # default_factory always creates a real Lock, so __contains__ reduces to "record exists"
```

**Note on `extra_tokens`/`gate_fired` "unset" value**: their natural defaults (`0`,
`False`) are also valid real values (a turn can legitimately have 0 extra tokens or a gate
that hasn't fired). Since **no code anywhere checks `key in self._chat_request_extra_tokens`
or `key in self._gate_fired`** (confirmed by grep across `bot/`, `tests/`), using the
default as the "unset" sentinel is safe here — `.get(key, 0)`/`.pop(key, False)` behave
identically whether the key was "never set" or "set to the default", exactly matching
current dict behavior (a missing key and an explicit `0`/`False` are already
indistinguishable through `.get(key, default)`, which is the only access pattern used).

`self._chat_states` becomes the module's actual private attribute name (matches
`AGENTS.md`'s "Helper Session API" section, which already names `_chat_states` alongside
`_clear_chat_state`/`_with_chat_state` when describing what external code must not touch —
that line currently describes attributes that don't all exist yet; this task makes it
accurate).

`__init__` (`openai_helper.py:269-288`): replace the 9 dict-literal assignments with a
single `self._chat_states = ChatStateRegistry()`, keep `self._background_tasks` (:271,
unrelated) as-is.

`_chat_lock` (`:2916-2939`) becomes:
```python
async def _chat_lock(self, chat_id) -> asyncio.Lock:
    """Per-chat asyncio.Lock guarding mutations of self.conversations. ...
    (same docstring content, minus the now-removed double-checked-locking note)."""
    return self._get_chat_states().get_or_create(self._chat_state_key(chat_id)).lock
```
Stays `async def` (unchanged signature — 5 call sites do `await self._chat_lock(...)`,
`:865,965,2392,2428,2576`) even though the body no longer awaits anything, purely to avoid
touching those 5 call sites' `await` keyword.

`_clear_chat_state` (`:3088-3103`) becomes a one-liner:
```python
def _clear_chat_state(self, chat_id) -> None:
    self._get_chat_states().evict(chat_id)
```
This is also where the master-plan step-3 fix lands: `evict()` on the registry drops the
**whole** `ConversationState` record, so `gate_fired`/`last_summary_at` — never cleared
today (§0) — are cleared along with everything else automatically, with no extra code.

`get_or_create`-triggering call sites (every place that currently does
`self.conversations[state_key] = ...` after a cold-cache DB load, e.g. `:1011`, `:1289`,
`:2230`, `:3405`) keep working unchanged through the `conversations` property's
`__setitem__` → `_FieldView.__setitem__` → `registry.get_or_create(key)` path — **and this
is where the lazy sweep hooks in** (see §4): trigger `maybe_sweep()` from inside
`_FieldView.__setitem__`/`ChatStateRegistry.get_or_create`, not from a dozen call sites.

---

## 4. Background cleanup: lazy sweep, not a new core loop

**Decision: lazy, throttled sweep triggered from `ChatStateRegistry.get_or_create()`.**

Justification (see §1's last bullet for the fact-finding):
1. `OpenAIHelper` has no existing periodic-task mechanism of its own — the only one in the
   tree (`PluginManager.start_background_tasks`) is plugin-scoped and wired from
   `ChatGPTTelegramBot.post_init` in `bot/telegram_bot.py`, a file **not owned by T11**.
   Adding a new lifecycle hook there is out of reach without violating file ownership.
2. Self-starting a loop inside `OpenAIHelper.__init__` (`asyncio.create_task(...)`) would
   require a running event loop at construction time. `bot/__main__.py:407` constructs
   `OpenAIHelper` in the same synchronous setup section as `PluginManager`/`Database`,
   before `Application.run_polling()` starts the loop — `asyncio.create_task` there raises
   `RuntimeError: no running event loop`. It would also break every test building
   `OpenAIHelper` synchronously outside an event loop (most of them).
3. A throttled lazy sweep needs zero new plumbing: `ChatStateRegistry.maybe_sweep(now,
   max_age_minutes=...)` is called once per `get_or_create()`, and is a cheap timestamp
   comparison (`(now - self._last_swept_at).total_seconds() < 60`) unless the interval has
   actually elapsed, at which point it's one O(n) pass over currently-in-memory chats
   (bounded by `MAX_CHAT_STATES`, so O(1000) worst case) — negligible next to an LLM round
   trip. This mirrors an existing in-file pattern: `self._tts_models_cache: tuple[float,
   list[str]] | None = None` (`openai_helper.py:289`) is exactly a "(timestamp, cached
   value), skip recompute if fresh" cache already used in this class.
4. Correctness: `sweep()` explicitly skips any entry whose `lock.locked()` is `True`
   (master-plan requirement), so a sweep can never evict state a concurrent turn is
   actively using, regardless of when in the request lifecycle it fires.

`get_or_create` calls `maybe_sweep(datetime.datetime.now(), max_age_minutes=self._max_age_minutes_getter())`
— practically, `OpenAIHelper` wraps this: after building `self._chat_states =
ChatStateRegistry(max_states=_positive_int_env("MAX_CHAT_STATES", 1000))`, its
`_get_chat_states()` accessor (or `_FieldView.__setitem__`, whichever ends up calling
`get_or_create`) passes `max_age_minutes=self.config['max_conversation_age_minutes']`
through on every call — cheap dict lookup, stays byte-for-byte faithful to
`__max_age_reached`'s current fresh-read behavior (§1).

Alternative considered and rejected: **eager LRU eviction on every insert** (like
`telegram_bot.py`'s `_BoundedLRU`, `:130-153`) instead of sweeping in `sweep()`. Rejected
because `_BoundedLRU` evicts unconditionally on overflow with no way to skip a
currently-locked entry (`OrderedDict.__delitem__` on `next(iter(self))` has no "is this one
busy" check) — reusing it as-is would violate the master plan's explicit "skip locked
entries" requirement, and forking it to add lock-awareness inside `__setitem__` on every
single write is less clear than doing both TTL and LRU-cap eviction together in one
explicit `sweep()` pass, which is also what the master plan names as the method to write.

---

## 5. `bot/openai_tool_handler.py` — the two `conversations` touches

`_conversation_messages` (def `:253-254`):
```python
# before
def _conversation_messages(helper, chat_id):
    return helper.conversations.setdefault(_chat_state_key(helper, chat_id), [])

# after
def _conversation_messages(helper, chat_id):
    state = helper._get_chat_states().get_or_create(_chat_state_key(helper, chat_id))
    if state.history is None:
        state.history = []
    return state.history
```
5 call sites (`:997,1075,1681,1692,1703`) all do `_conversation_messages(helper,
chat_id).append({...})` — unaffected, same object identity/mutate-in-place contract as
today (`self.conversations[key].append(...)`).

Logging line `:1712`:
```python
# before
_json_for_log(helper.conversations.get(_chat_state_key(helper, chat_id))),
# after
_json_for_log(
    (helper._get_chat_states().peek(_chat_state_key(helper, chat_id)) or ConversationState()).history
),
```
(or equivalently keep using the `conversations` property's `.get()` here since this one
site has no mutation requirement — either is fine; using the property is one line shorter
and doesn't require importing `ConversationState`/`_get_chat_states` into
`openai_tool_handler.py`. **Recommend keeping `helper.conversations.get(...)` unchanged at
`:1712`** — only `:254` needs the registry directly, because only `:254` mutates through
`setdefault`. This keeps the diff smaller and doesn't require touching the file's imports.)

Revised recommendation for `:253-254` only — same idea, phrased as a helper method on
`OpenAIHelper` instead of reaching into `_get_chat_states()` from outside the class (keeps
`openai_tool_handler.py` talking to one accessor, consistent with how it already reaches
`~10-14` other underscore-prefixed helper internals on purpose, documented in
`test_no_private_helper_access.py`'s own docstring):
```python
# bot/openai_helper.py, new method next to load_session/history_snapshot
def _mutable_history(self, chat_id) -> list:
    """Returns the live history list for chat_id, creating an empty one if
    this is the first write this turn. Used by openai_tool_handler.py's
    _conversation_messages(), which needs to .append() through the cache
    (unlike history_snapshot(), which returns a defensive copy)."""
    state = self._get_chat_states().get_or_create(self._chat_state_key(chat_id))
    if state.history is None:
        state.history = []
    return state.history

# bot/openai_tool_handler.py:253-254
def _conversation_messages(helper, chat_id):
    return helper._mutable_history(chat_id)
```
This is the preferred version — one new underscore-prefixed method, same shape as the
other ~10-14 already-allowed internals, instead of reaching two attributes deep
(`helper._get_chat_states().get_or_create(...)`) from a different module.

---

## 6. Extending `tests/test_no_private_helper_access.py`

Currently the module docstring (`:14-22`) explicitly excludes
`bot/openai_tool_handler.py` and documents the two gaps this task closes
(`helper.conversations.setdefault` at `:255`/actually `:254`, `.get` at `:1712`/now
unchanged per §5's recommendation — update this docstring reference if `:1712` stays as
`helper.conversations.get`).

Master-plan step 5 asks to extend the scan to `openai_tool_handler.py`. Add a fourth test
function following the exact shape of the existing three (`_check_file` +
`_assert_against_allowlist`, `:107-136`):

```python
OPENAI_TOOL_HANDLER_FILE = REPO_ROOT / "bot" / "openai_tool_handler.py"

def test_openai_tool_handler_does_not_touch_new_helper_privates() -> None:
    violations = _check_file(OPENAI_TOOL_HANDLER_FILE, {"helper"})
    rel = "bot/openai_tool_handler.py"
    failures = _assert_against_allowlist(rel, violations)
    assert not failures, (
        f"{rel} reaches into new OpenAIHelper privates:\n" + "\n".join(failures)
        + "\n\nOnly the already-allow-listed internals may be touched from here; add a "
        "documented ALLOWED entry if a new one is genuinely needed."
    )
```

Populate `ALLOWED` with the ~7 unique pre-existing names found by grep (occurrence counts
from the current tree — recount after §5's edit, since `_mutable_history` replaces the
`conversations` touch and is itself a **new** allowed name):
`_apply_before_chat_request_mutators` (3), `_add_function_call_to_history` (2),
`_tool_call_global_semaphore_bundle` (1), `_tool_call_global_semaphore` (1),
`_without_chat_lock` (1, via `getattr`), `_chat_state_key` (1, via `getattr` — note:
`openai_tool_handler.py` also has its own **module-level** `_chat_state_key(helper,
chat_id)` function at `:246` which calls `getattr(helper, "_chat_state_key", None)`; don't
double count), `_mode_from_system_message` (1, via `getattr`), `_uses_structured_tool_history`
(1, via `getattr`), `_add_assistant_tool_calls_to_history` (1, via `getattr`),
`_defer_direct_tool_results` (1, via `getattr`), plus the new `_mutable_history` (1) from
§5. Since `DENYLISTED_STATE_ATTRS` (`:40`) already includes `"conversations"` (checked via
`_is_private_attr`, not underscore-based), the guard automatically catches `helper.conversations`
in `openai_tool_handler.py` at `ALLOWED` count `0` once §5's rewrite removes the
`:254` touch — no separate entry needed, it just falls through to "0 allowed, 0 found" like
every other file already does for this name.

Update the module docstring (`:1-23`) to drop the "Deliberately NOT scanned" paragraph
about `openai_tool_handler.py` and replace it with one line noting it is now scanned with
its own allow-list, same as the other three targets.

---

## 7. Migration order (keep tests green at each step)

1. Write `bot/conversation_state.py` in full (§2, §3's `_FieldView`) as a standalone,
   framework-free module. No other file imports it yet — safe, independently testable
   step. Add a small new test file for it directly (not listed in master plan's explicit
   test list, but needed): `tests/test_conversation_state.py` covering `get_or_create`/
   `peek`/`evict`/`sweep` TTL+LRU+locked-skip behavior in isolation, no `OpenAIHelper`
   involved.
2. `bot/openai_helper.py`: add `from .conversation_state import ChatStateRegistry,
   ConversationState, _UNSET` (or re-export `_UNSET` under a clearer name if preferred),
   replace the 9-dict `__init__` block with `self._chat_states = ChatStateRegistry(...)`,
   add the 9 property pairs + `_per_chat_locks` property + `_get_chat_states()` + the new
   `_mutable_history()` method. **Do not yet delete any old direct-dict-literal code path
   inside methods** — the properties make `self.conversations[...]` etc. keep working
   exactly as before syntactically, so method bodies need zero changes in this step.
3. Rewrite `_chat_lock` (§3) and `_clear_chat_state`/`evict` (§3) to use
   `self._get_chat_states()` directly instead of the now-property-backed dict syntax
   (functionally equivalent either way, but this is where the guard-removal and the
   "`gate_fired`/`last_summary_at` now cleared" fix land — worth its own step so a test
   failure here is attributable).
4. Run `tests/test_openai_helper_session_api.py` first (smallest, most direct exercise of
   every property) — iterate here before touching anything else.
5. Run `tests/test_openai_helper_tool_calls.py` (largest, 91 `conversations` hits) —
   expect this to be the most likely place for a subtle proxy-semantics mismatch to
   surface (e.g. an iteration pattern not yet covered by `_FieldView`).
6. Run the rest of §0's "affected" test list as one batch.
7. `bot/openai_tool_handler.py`: apply §5's two-line change (`_conversation_messages` body
   only). Run its own light test coverage plus
   `tests/test_openai_helper_tool_calls.py` again (it exercises tool-call flows that hit
   this function).
8. Extend `tests/test_no_private_helper_access.py` per §6. Run it standalone.
9. Full targeted suite (§9) once, then full `tests/ bot/tests/` as a final gate.

---

## 8. Tests to add/update

**New:**
- `tests/test_conversation_state.py` — `ChatStateRegistry` in isolation: `get_or_create`
  creates once and reuses; `peek` never creates; `evict` removes; `sweep(now,
  max_age_minutes=...)` evicts entries older than the cutoff, evicts over-`max_states` LRU
  overflow, **never** evicts an entry whose `state.lock.locked()` is `True` even if it's
  both idle and over cap; `maybe_sweep` is a no-op inside the throttle window and a real
  sweep after it elapses. `_FieldView`: `replace()` whole-dict reassignment semantics
  (extra keys removed, new keys added, untouched fields on shared records preserved);
  `__eq__` against a plain dict; `.pop(key, default)`/`.get(key)`/`del`/`in` parity with a
  real `dict` for a field with `None`-as-unset and one with a real sentinel
  (`session_id`/`_UNSET`).

**Must pass unchanged (regression, real `OpenAIHelper` — see §0's classification table for
the full list and why each is affected):** every file in §0's "Test file counts" table.
Run in the order given in §7 step 4-6.

**Extend:** `tests/test_no_private_helper_access.py` per §6 — one new test function +
`ALLOWED` entries + docstring update.

**Update if `_UNSET` sentinel value is asserted anywhere by identity** — none currently
are (checked: no test imports or compares against a helper-internal sentinel today), so no
test literally needs to import `_UNSET` from the new module; flagging only as a thing to
re-check after implementation since it's new API surface.

---

## 9. Acceptance commands

```bash
~/.venvs/ctb/bin/python -m mypy bot/conversation_state.py bot/openai_helper.py \
  bot/openai_tool_handler.py --python-executable ~/.venvs/ctb/bin/python --ignore-missing-imports

~/.venvs/ctb/bin/python -m ruff check bot/conversation_state.py bot/openai_helper.py \
  bot/openai_tool_handler.py tests/test_conversation_state.py tests/test_no_private_helper_access.py

~/.venvs/ctb/bin/python -m pytest tests/test_conversation_state.py \
  tests/test_openai_helper_session_api.py tests/test_no_private_helper_access.py \
  -q --no-header -p no:cacheprovider

~/.venvs/ctb/bin/python -m pytest tests/test_openai_helper_tool_calls.py -q --no-header -p no:cacheprovider

~/.venvs/ctb/bin/python -m pytest tests/test_openai_helper_summarize_trim.py \
  tests/test_reset_chat_history_async.py tests/test_record_plugin_exchange.py \
  tests/test_session_logging_integration.py tests/test_skills_agent_gate.py \
  tests/test_summarise_overflow_dispatch.py tests/test_exemplar_summarize_failure_structure.py \
  tests/test_exemplar_interrupted_tool_call_repair.py tests/test_reflection_on_tool_error.py \
  tests/test_hindsight_memory.py tests/test_stream_usage.py tests/test_pricing.py \
  tests/test_group_session_flow.py tests/test_per_conversation_serialization.py \
  -q --no-header -p no:cacheprovider

~/.venvs/ctb/bin/python -m pytest tests/ bot/tests/ -q --no-header -p no:cacheprovider
```

`python3` inline grep (not `rg`/`grep` directly, per environment note) to confirm no
stray old-style dict literal remains: search `bot/openai_helper.py` for
`self\.conversations\s*:\s*dict`/`self\._gate_fired\s*:\s*dict`-style `__init__`
declarations — should return zero after step 2 of §7.

---

## 10. Risks / explicitly flagged design choices

1. **`_FieldView.replace()` semantics on shared records**: whole-dict reassignment
   (`helper.conversations = {...}`) only touches the `history` field of any record it
   creates/updates/clears — it must **not** wipe other fields (`request_model`,
   `gate_fired`, ...) of a record that also exists for that key from an earlier,
   independent assignment (e.g. a fixture that sets `helper._gate_fired = {1: True}` then
   separately `helper.conversations = {1: [...]}`). §3's design achieves this by scoping
   `replace()` to one field per call — verify with a dedicated test in
   `tests/test_conversation_state.py` (§8) since this is the single most likely place for
   a subtle regression if implemented as "wipe the whole record" instead.
2. **`extra_tokens`/`gate_fired` default-as-sentinel** (§3's note): safe today because no
   `in` check exists on those two dicts anywhere in the tree, but this is a soft invariant,
   not enforced by any guard — if a future change adds `if key in helper._gate_fired:`
   somewhere, it would silently misbehave for a key that was set to exactly `False`/`0`.
   Not worth a real sentinel today (adds complexity for a case that doesn't occur), but
   worth a one-line comment on the field definitions in `conversation_state.py` warning
   future editors.
3. **`MAX_CHAT_STATES` env var**: `.env.example` is not in T11's file ownership, so this
   plan does not add a line there — flagging as a follow-up the implementer or a later
   task should do (one line, `# MAX_CHAT_STATES=1000`, next to
   `MAX_CONVERSATION_AGE_MINUTES=18000` at `.env.example:63`).
4. **`openai_tool_handler.py:1712`** (`_json_for_log(helper.conversations.get(...))`):
   recommendation in §5 is to leave this one call site untouched (still goes through the
   `conversations` property, which still works read-only) rather than also routing it
   through `_get_chat_states()`/`_mutable_history()` — smaller diff, and it was never the
   mutation gap (only `:254`'s `setdefault` was). If a reviewer prefers **zero** remaining
   `helper.conversations` references in this file for symmetry, swap it for
   `helper.history_snapshot(chat_id)` instead (the existing public accessor, read-only,
   already used by `telegram_bot.py`) — either is fine, this plan defers the choice to
   whoever implements it.
5. **Sweep O(n) worst case**: bounded by `MAX_CHAT_STATES` (default 1000), so a real sweep
   pass is at most 1000 dict lookups plus `state.lock.locked()` checks — negligible, but
   confirm no test asserts an exact call count on `datetime.datetime.now()` that a new
   `maybe_sweep()` call inside `get_or_create` would perturb (none found in §0's grep, but
   re-check after implementation since `datetime.now` is easy to monkeypatch-assert on).
