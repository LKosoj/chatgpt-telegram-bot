"""Verify-step trigger tests for agent_tools plugin (variant 2+3).

Variant 2 (backstop): a plan task transitioning to status=completed (via add or
update) schedules a one-shot ``[verify-step-v1]`` system-message inject, delivered
by on_before_chat_request. Mirrors the re-plan trigger lifecycle.

Variant 3 (prompt): the plan-rule prefix instructs inline intent-before-tool and
assessment-after-tool, without extra round-trips.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.database import Database
from bot.plugins.agent_tools import (
    AgentToolsPlugin,
    _PLAN_RULE_TEXT,
    _VERIFY_TRIGGER_MARKER,
)
from bot.plugins.db_handle import DbHandle
from bot.plugins.hooks import BeforeChatRequestPayload, SessionResetPayload
from bot.utils import compute_scope_key


PLAN_CONTRACT = {
    "goal": "Finish verify test",
    "success_criteria": ["Verify trigger fires only when expected"],
    "verification": ["Inspected pending verify state"],
}


@pytest.fixture()
def agent_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "agent.db"))
    Database._reset_singleton()
    db = Database()
    with db.get_connection() as conn:
        for stmt in AgentToolsPlugin().register_schema():
            conn.execute(stmt)
    yield db
    Database._reset_singleton()


class FakeHelper:
    """Minimal helper that satisfies on_before_chat_request's needs."""

    def __init__(self, allowed=None):
        self._allowed = allowed if allowed is not None else ["All"]

    async def resolve_allowed_plugins(self, chat_id, session_id=None, user_id=None):
        return self._allowed


def _make_plugin(tmp_path, db, allowed=None) -> tuple[AgentToolsPlugin, SimpleNamespace]:
    helper = SimpleNamespace(user_id=42, db=db)
    plugin = AgentToolsPlugin()
    plugin.initialize(openai=helper, storage_root=str(tmp_path), db=DbHandle(db))
    plugin.openai = FakeHelper(allowed)
    plugin.openai.db = db  # type: ignore[attr-defined]
    return plugin, helper


def _payload(chat_id=10, user_id=42) -> BeforeChatRequestPayload:
    return BeforeChatRequestPayload(chat_id=chat_id, user_id=user_id, request_id=None)


# ---------- Variant 3: plan-rule prompt carries intent/assessment guidance ----

def test_plan_rule_text_includes_intent_and_assessment():
    assert "намерение" in _PLAN_RULE_TEXT
    assert "оцени" in _PLAN_RULE_TEXT


# ---------- Unit-level helpers (no DB) ---------------------------------------

def test_schedule_verify_idempotent_same_task():
    plugin = AgentToolsPlugin()
    scope = "chat:1"
    plugin._schedule_verify(scope, "T1")
    first = dict(plugin._pending_verify[scope])
    plugin._schedule_verify(scope, "T1")
    assert plugin._pending_verify[scope] == first
    # Different task overrides the entry (still single pending).
    plugin._schedule_verify(scope, "T2")
    assert plugin._pending_verify[scope] == {"task_id": "T2"}


def test_schedule_verify_ignores_empty_task():
    plugin = AgentToolsPlugin()
    plugin._schedule_verify("chat:1", "")
    assert "chat:1" not in plugin._pending_verify


# ---------- DB-backed plan transitions ---------------------------------------

@pytest.mark.asyncio
async def test_completed_status_transition_schedules_verify(tmp_path, agent_db):
    plugin, helper = _make_plugin(tmp_path, agent_db)
    seeded = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10, user_id=42,
        action="add",
        definition_of_done=PLAN_CONTRACT,
        tasks=[{"id": "T1", "content": "Work", "status": "in_progress"}],
    )
    assert seeded["success"] is True
    scope = compute_scope_key(10, 42)
    assert scope not in plugin._pending_verify

    updated = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10, user_id=42,
        action="update",
        tasks=[{"id": "T1", "status": "completed"}],
    )
    assert updated["success"] is True
    assert plugin._pending_verify[scope] == {"task_id": "T1"}


@pytest.mark.asyncio
async def test_completed_no_op_no_verify_when_already_completed(tmp_path, agent_db):
    plugin, helper = _make_plugin(tmp_path, agent_db)
    seeded = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10, user_id=42,
        action="add",
        definition_of_done=PLAN_CONTRACT,
        tasks=[{"id": "T1", "content": "Work", "status": "completed"}],
    )
    assert seeded["success"] is True
    scope = compute_scope_key(10, 42)
    # First completed-on-add schedules verify.
    assert plugin._pending_verify[scope] == {"task_id": "T1"}
    plugin._pending_verify.pop(scope, None)

    # No-op update to completed again should NOT re-schedule (status unchanged).
    again = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10, user_id=42,
        action="update",
        tasks=[{"id": "T1", "status": "completed"}],
    )
    assert again["success"] is True
    assert scope not in plugin._pending_verify


@pytest.mark.asyncio
async def test_clear_plan_clears_pending_verify(tmp_path, agent_db):
    plugin, helper = _make_plugin(tmp_path, agent_db)
    seeded = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10, user_id=42,
        action="add",
        definition_of_done=PLAN_CONTRACT,
        tasks=[{"id": "T1", "content": "Work", "status": "in_progress"}],
    )
    assert seeded["success"] is True
    scope = compute_scope_key(10, 42)
    plugin._schedule_verify(scope, "T1")
    assert scope in plugin._pending_verify

    cleared = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10, user_id=42,
        action="clear",
    )
    assert cleared["success"] is True
    assert scope not in plugin._pending_verify


@pytest.mark.asyncio
async def test_session_reset_clears_pending_verify(tmp_path, agent_db):
    plugin, helper = _make_plugin(tmp_path, agent_db)
    scope = compute_scope_key(10, 42)
    plugin._schedule_verify(scope, "T1")

    await plugin.on_session_reset(SessionResetPayload(
        chat_id=10, user_id=42, reason="request_start", terminal_only=False,
    ))
    assert scope not in plugin._pending_verify


@pytest.mark.asyncio
async def test_session_reset_terminal_only_preserves_pending_verify(tmp_path, agent_db):
    """terminal_only=True clears finished tasks but the session continues —
    a pending verify trigger must survive to be delivered next chat request."""
    plugin, helper = _make_plugin(tmp_path, agent_db)
    scope = compute_scope_key(10, 42)
    plugin._schedule_verify(scope, "T1")

    await plugin.on_session_reset(SessionResetPayload(
        chat_id=10, user_id=42, reason="request_start", terminal_only=True,
    ))
    assert scope in plugin._pending_verify


# ---------- Mutator delivery -------------------------------------------------

@pytest.mark.asyncio
async def test_mutator_injects_verify_message_when_pending(tmp_path, agent_db):
    plugin, _ = _make_plugin(tmp_path, agent_db, allowed=["agent_tools"])
    scope = compute_scope_key(10, 42)
    plugin._schedule_verify(scope, "T1")

    messages = [
        {"role": "system", "content": "mode prompt"},
        {"role": "user", "content": "hi"},
    ]
    new = await plugin.on_before_chat_request(messages, _payload())
    assert new is not None
    triggers = [
        m for m in new
        if isinstance(m, dict)
        and m.get("role") == "system"
        and isinstance(m.get("content"), str)
        and m["content"].startswith(_VERIFY_TRIGGER_MARKER)
    ]
    assert len(triggers) == 1
    assert "T1" in triggers[0]["content"]
    assert "completed" in triggers[0]["content"]
    # Pending entry was popped (one-shot).
    assert scope not in plugin._pending_verify


@pytest.mark.asyncio
async def test_mutator_pops_pending_verify_once_idempotent(tmp_path, agent_db):
    plugin, _ = _make_plugin(tmp_path, agent_db, allowed=["agent_tools"])
    scope = compute_scope_key(10, 42)
    plugin._schedule_verify(scope, "T1")

    messages = [
        {"role": "system", "content": "mode prompt"},
        {"role": "user", "content": "hi"},
    ]
    first = await plugin.on_before_chat_request(messages, _payload())
    assert first is not None
    triggers1 = [
        m for m in first
        if isinstance(m, dict)
        and isinstance(m.get("content"), str)
        and m["content"].startswith(_VERIFY_TRIGGER_MARKER)
    ]
    assert len(triggers1) == 1

    # Second invocation finds no pending verify entry.
    second = await plugin.on_before_chat_request(messages, _payload())
    if second is None:
        return
    triggers2 = [
        m for m in second
        if isinstance(m, dict)
        and isinstance(m.get("content"), str)
        and m["content"].startswith(_VERIFY_TRIGGER_MARKER)
    ]
    assert triggers2 == []
