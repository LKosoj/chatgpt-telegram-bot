"""Unit tests for bot/conversation_state.py in isolation (no OpenAIHelper).

Covers ChatStateRegistry (get_or_create/peek/evict/sweep -- TTL eviction, LRU-cap
eviction, locked entries are never evicted, maybe_sweep's throttle) and _FieldView
(the dict-like proxy that backs OpenAIHelper's old per-attribute dicts): replace()
whole-dict reassignment semantics, __eq__ against a plain dict, and get/pop/del/in
parity with a real dict for both a None-as-unset field and a real-sentinel
(_UNSET) field.
"""

from __future__ import annotations

import asyncio
import datetime

import pytest

from bot.conversation_state import (
    ChatStateRegistry,
    ConversationState,
    _FieldView,
    _positive_int_env,
    _UNSET,
)


# --- _positive_int_env: MAX_CHAT_STATES parsing --------------------------------

def test_positive_int_env_falls_back_to_default_on_invalid_value(monkeypatch):
    monkeypatch.setenv("MAX_CHAT_STATES", "abc")
    assert _positive_int_env("MAX_CHAT_STATES", 1000) == 1000


def test_positive_int_env_clamps_zero_to_one(monkeypatch):
    monkeypatch.setenv("MAX_CHAT_STATES", "0")
    assert _positive_int_env("MAX_CHAT_STATES", 1000) == 1


def test_positive_int_env_clamps_negative_to_one(monkeypatch):
    monkeypatch.setenv("MAX_CHAT_STATES", "-5")
    assert _positive_int_env("MAX_CHAT_STATES", 1000) == 1


def test_positive_int_env_parses_a_valid_value(monkeypatch):
    monkeypatch.setenv("MAX_CHAT_STATES", "42")
    assert _positive_int_env("MAX_CHAT_STATES", 1000) == 42


def test_positive_int_env_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("MAX_CHAT_STATES", raising=False)
    assert _positive_int_env("MAX_CHAT_STATES", 1000) == 1000


# --- ChatStateRegistry: get_or_create / peek / evict ------------------------

def test_get_or_create_creates_once_and_reuses():
    registry = ChatStateRegistry()
    state = registry.get_or_create(1)
    assert isinstance(state, ConversationState)
    assert registry.get_or_create(1) is state


def test_peek_never_creates():
    registry = ChatStateRegistry()
    assert registry.peek(1) is None
    assert len(registry) == 0


def test_peek_returns_existing_without_recreating():
    registry = ChatStateRegistry()
    state = registry.get_or_create(1)
    assert registry.peek(1) is state


def test_evict_removes_entry():
    registry = ChatStateRegistry()
    registry.get_or_create(1)
    registry.evict(1)
    assert registry.peek(1) is None
    assert len(registry) == 0


def test_evict_unknown_key_is_a_no_op():
    registry = ChatStateRegistry()
    registry.evict(999)  # must not raise
    assert len(registry) == 0


def test_evict_clears_the_whole_record_including_gate_fired_and_last_summary_at():
    """Regression for the leak _clear_chat_state had before T11: gate_fired and
    last_summary_at were never cleared by the old dict-pop list. Since eviction
    now drops the whole ConversationState record, every field disappears together."""
    registry = ChatStateRegistry()
    state = registry.get_or_create(1)
    state.history = [{"role": "user", "content": "hi"}]
    state.gate_fired = True
    state.last_summary_at = 12

    registry.evict(1)

    assert registry.peek(1) is None


# --- sweep: TTL eviction ------------------------------------------------------

def test_sweep_evicts_idle_entries_past_max_age():
    registry = ChatStateRegistry()
    state = registry.get_or_create(1)
    state.last_updated = datetime.datetime.now() - datetime.timedelta(minutes=120)

    evicted = registry.sweep(datetime.datetime.now(), max_age_minutes=60)

    assert evicted == 1
    assert registry.peek(1) is None


def test_sweep_keeps_entries_within_max_age():
    registry = ChatStateRegistry()
    state = registry.get_or_create(1)
    state.last_updated = datetime.datetime.now() - datetime.timedelta(minutes=10)

    evicted = registry.sweep(datetime.datetime.now(), max_age_minutes=60)

    assert evicted == 0
    assert registry.peek(1) is not None


def test_sweep_never_evicts_an_entry_with_no_last_updated():
    """A freshly created record (last_updated still None) must not be treated
    as idle -- only an explicit timestamp makes an entry eligible."""
    registry = ChatStateRegistry()
    registry.get_or_create(1)

    evicted = registry.sweep(datetime.datetime.now(), max_age_minutes=0)

    assert evicted == 0
    assert registry.peek(1) is not None


# --- sweep: LRU-cap eviction ---------------------------------------------------

def test_sweep_evicts_over_cap_oldest_first():
    registry = ChatStateRegistry(max_states=2)
    registry.get_or_create(1)
    registry.get_or_create(2)
    registry.get_or_create(3)  # now 3 entries, cap is 2

    evicted = registry.sweep(datetime.datetime.now(), max_age_minutes=99999)

    assert evicted == 1
    # oldest (1) was evicted, most recently created/touched survive
    assert registry.peek(1) is None
    assert registry.peek(2) is not None
    assert registry.peek(3) is not None


def test_get_or_create_touch_moves_entry_to_the_end_of_lru_order():
    registry = ChatStateRegistry(max_states=2)
    registry.get_or_create(1)
    registry.get_or_create(2)
    registry.get_or_create(1)  # touch 1 again -> 2 is now the oldest
    registry.get_or_create(3)  # over cap by one

    evicted = registry.sweep(datetime.datetime.now(), max_age_minutes=99999)

    assert evicted == 1
    assert registry.peek(2) is None
    assert registry.peek(1) is not None
    assert registry.peek(3) is not None


# --- sweep: locked entries are never evicted -----------------------------------

async def test_sweep_never_evicts_a_locked_entry_even_if_idle_and_over_cap():
    registry = ChatStateRegistry(max_states=1)
    locked_state = registry.get_or_create("locked")
    locked_state.last_updated = datetime.datetime.now() - datetime.timedelta(minutes=999)

    async with locked_state.lock:
        registry.get_or_create("other")  # pushes registry over the cap of 1
        evicted = registry.sweep(datetime.datetime.now(), max_age_minutes=1)

    # "other" is unlocked and the registry is over cap, so it is legitimately
    # evicted; "locked" is both idle and over cap but must survive regardless.
    assert evicted == 1
    assert registry.peek("locked") is not None
    assert registry.peek("other") is None


async def test_sweep_evicts_the_same_entry_once_its_lock_is_released():
    registry = ChatStateRegistry(max_states=1)
    state = registry.get_or_create("chat")
    state.last_updated = datetime.datetime.now() - datetime.timedelta(minutes=999)

    async with state.lock:
        assert registry.sweep(datetime.datetime.now(), max_age_minutes=1) == 0

    assert registry.sweep(datetime.datetime.now(), max_age_minutes=1) == 1
    assert registry.peek("chat") is None


# --- maybe_sweep: throttle ------------------------------------------------------

def test_maybe_sweep_is_a_no_op_inside_the_throttle_window():
    registry = ChatStateRegistry(sweep_min_interval_seconds=60.0)
    state = registry.get_or_create(1)
    state.last_updated = datetime.datetime.now() - datetime.timedelta(minutes=999)
    now = datetime.datetime.now()

    assert registry.maybe_sweep(now, max_age_minutes=1) == 1  # first call: real sweep
    # a second idle+expired entry created right after -- still inside the window
    state2 = registry.get_or_create(2)
    state2.last_updated = datetime.datetime.now() - datetime.timedelta(minutes=999)
    assert registry.maybe_sweep(now, max_age_minutes=1) == 0
    assert registry.peek(2) is not None


def test_maybe_sweep_runs_again_after_the_interval_elapses():
    registry = ChatStateRegistry(sweep_min_interval_seconds=60.0)
    now = datetime.datetime.now()
    registry.maybe_sweep(now, max_age_minutes=1)

    state = registry.get_or_create(1)
    state.last_updated = datetime.datetime.now() - datetime.timedelta(minutes=999)

    later = now + datetime.timedelta(seconds=61)
    assert registry.maybe_sweep(later, max_age_minutes=1) == 1
    assert registry.peek(1) is None


# --- _FieldView: dict parity ----------------------------------------------------

def test_field_view_setitem_getitem_roundtrip():
    registry = ChatStateRegistry()
    view = _FieldView(registry, "history", None)
    view[1] = ["a"]
    assert view[1] == ["a"]


def test_field_view_getitem_raises_key_error_when_never_set():
    registry = ChatStateRegistry()
    view = _FieldView(registry, "history", None)
    with pytest.raises(KeyError):
        view[1]


def test_field_view_get_pop_del_in_parity_with_none_as_unset():
    registry = ChatStateRegistry()
    view = _FieldView(registry, "last_updated", None)
    reference: dict = {}

    view[1] = "now"
    reference[1] = "now"
    assert (1 in view) == (1 in reference)
    assert view.get(1) == reference.get(1)
    assert view.get(2, "dflt") == reference.get(2, "dflt")
    assert view.pop(1, None) == reference.pop(1, None)
    assert (1 in view) == (1 in reference) is False
    with pytest.raises(KeyError):
        del view[1]
    with pytest.raises(KeyError):
        del reference[1]


def test_field_view_get_pop_del_in_parity_with_real_sentinel_for_a_legitimate_none_value():
    """session_id (and, since T11's fix, usage_split) use _UNSET rather than None
    as the 'never set' marker, because None is itself a legitimate stored value
    (e.g. load_session(chat_id, None, ...) or a reset-to-unknown usage split)."""
    registry = ChatStateRegistry()
    view = _FieldView(registry, "session_id", _UNSET)

    view[1] = None  # a real, legitimate value -- not "unset"
    assert view[1] is None
    assert 1 in view
    assert view.get(1, "missing") is None
    assert view.pop(1) is None
    assert 1 not in view
    with pytest.raises(KeyError):
        view[1]


def test_field_view_eq_against_plain_dict():
    registry = ChatStateRegistry()
    view = _FieldView(registry, "history", None)
    assert view == {}
    view[1] = ["a"]
    view[2] = ["b"]
    assert view == {1: ["a"], 2: ["b"]}
    assert dict(view) == {1: ["a"], 2: ["b"]}


def test_field_view_replace_adds_and_removes_keys():
    registry = ChatStateRegistry()
    view = _FieldView(registry, "history", None)
    view[1] = ["old"]
    view[2] = ["keep"]

    view.replace({2: ["keep"], 3: ["new"]})

    assert dict(view) == {2: ["keep"], 3: ["new"]}
    with pytest.raises(KeyError):
        view[1]


def test_field_view_replace_only_touches_its_own_field_on_a_shared_record():
    """Whole-dict reassignment of one field (e.g. helper.conversations = {...})
    must not wipe other fields (e.g. gate_fired) of a record that also exists
    for that key from an independent assignment -- the single most likely
    regression if replace() were implemented as 'wipe the whole record'."""
    registry = ChatStateRegistry()
    gate_view = _FieldView(registry, "gate_fired", False)
    history_view = _FieldView(registry, "history", None)

    gate_view[1] = True
    history_view.replace({1: ["hello"]})

    assert history_view[1] == ["hello"]
    assert gate_view[1] is True


async def test_field_view_lock_field_default_factory_creates_a_real_lock_per_entry():
    registry = ChatStateRegistry()
    state = registry.get_or_create(1)  # _FieldView.__getitem__ uses peek(), not
    # get_or_create() -- it never creates a record itself, same as a plain dict
    # never auto-vivifies a missing key. The record must already exist (as it
    # would in production, via OpenAIHelper._chat_lock's get_or_create call).
    view = _FieldView(registry, "lock", None)
    assert view[1] is state.lock
    assert isinstance(view[1], asyncio.Lock)
