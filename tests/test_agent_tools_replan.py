"""Re-plan trigger tests for agent_tools plugin.

Triggers:
  A. Consecutive tool failures (>=2) on the SAME in_progress task schedule a
     one-shot re-plan system-message inject.
  B. A task transitioning to status=blocked (via add or update) schedules the
     same kind of inject.

Delivery is via on_before_chat_request mutator; inject is idempotent per scope.
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
    TASK_STATUSES,
    _REPLAN_TRIGGER_MARKER,
)
from bot.plugins.db_handle import DbHandle
from bot.plugins.hooks import BeforeChatRequestPayload, SessionResetPayload
from bot.utils import compute_scope_key


PLAN_CONTRACT = {
    "goal": "Finish replan test",
    "success_criteria": ["Replan trigger fires only when expected"],
    "verification": ["Inspected pending replan state"],
}


@pytest.fixture()
def agent_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "agent.db"))
    Database._instance = None
    db = Database()
    with db.get_connection() as conn:
        for stmt in AgentToolsPlugin().register_schema():
            conn.execute(stmt)
    return db


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
    # FakeHelper replaces openai for mutator path (resolve_allowed_plugins) without
    # losing the SQLite-backed db assigned by initialize().
    plugin.openai = FakeHelper(allowed)
    plugin.openai.db = db  # type: ignore[attr-defined]
    return plugin, helper


def _payload(chat_id=10, user_id=42) -> BeforeChatRequestPayload:
    return BeforeChatRequestPayload(chat_id=chat_id, user_id=user_id, request_id=None)


# ---------- Unit-level helpers (no DB) ---------------------------------------

def test_task_statuses_includes_blocked():
    assert "blocked" in TASK_STATUSES


def test_manage_plan_tasks_schema_enum_includes_blocked():
    plugin = AgentToolsPlugin()
    spec = next(s for s in plugin.get_spec() if s["name"] == "manage_plan_tasks")
    enum = spec["parameters"]["properties"]["tasks"]["items"]["properties"]["status"]["enum"]
    assert "blocked" in enum


@pytest.mark.asyncio
async def test_record_tool_outcome_increments_on_repeated_failure_same_task():
    plugin = AgentToolsPlugin()
    scope = "chat:1"
    await plugin._record_tool_outcome(scope, "T1", False, "ddg.search")
    assert plugin._tool_error_streaks[scope] == {"task_id": "T1", "count": 1}
    # Second failure on same task reaches threshold, schedules replan,
    # and resets the streak to avoid trigger spam.
    await plugin._record_tool_outcome(scope, "T1", False, "ddg.search")
    assert scope not in plugin._tool_error_streaks
    assert plugin._pending_replan[scope] == {"reason": "errors", "task_id": "T1"}


@pytest.mark.asyncio
async def test_record_tool_outcome_resets_on_success():
    plugin = AgentToolsPlugin()
    scope = "chat:1"
    await plugin._record_tool_outcome(scope, "T1", False, "ddg.search")
    assert scope in plugin._tool_error_streaks
    await plugin._record_tool_outcome(scope, "T1", True, "ddg.search")
    assert scope not in plugin._tool_error_streaks
    assert scope not in plugin._pending_replan


@pytest.mark.asyncio
async def test_record_tool_outcome_resets_on_task_id_change():
    plugin = AgentToolsPlugin()
    scope = "chat:1"
    await plugin._record_tool_outcome(scope, "T1", False, "ddg.search")
    assert plugin._tool_error_streaks[scope]["task_id"] == "T1"
    await plugin._record_tool_outcome(scope, "T2", False, "ddg.search")
    # Task changed: streak resets to count=1 under new task id, no replan.
    assert plugin._tool_error_streaks[scope] == {"task_id": "T2", "count": 1}
    assert scope not in plugin._pending_replan


@pytest.mark.asyncio
async def test_consecutive_errors_threshold_resets_counter_to_zero_after_trigger():
    plugin = AgentToolsPlugin()
    scope = "chat:1"
    await plugin._record_tool_outcome(scope, "T1", False, "ddg.search")
    await plugin._record_tool_outcome(scope, "T1", False, "ddg.search")
    # Streak cleared, replan scheduled.
    assert scope not in plugin._tool_error_streaks
    # A third failure starts a new streak from count=1 (no immediate re-trigger).
    plugin._pending_replan.clear()
    await plugin._record_tool_outcome(scope, "T1", False, "ddg.search")
    assert plugin._tool_error_streaks[scope] == {"task_id": "T1", "count": 1}
    assert scope not in plugin._pending_replan


def test_schedule_replan_idempotent_same_reason():
    plugin = AgentToolsPlugin()
    scope = "chat:1"
    plugin._schedule_replan(scope, "errors", "T1")
    first = dict(plugin._pending_replan[scope])
    plugin._schedule_replan(scope, "errors", "T1")
    assert plugin._pending_replan[scope] == first
    # Different reason/task overrides the entry (still single pending).
    plugin._schedule_replan(scope, "blocked", "T1")
    assert plugin._pending_replan[scope] == {"reason": "blocked", "task_id": "T1"}


# ---------- DB-backed plan transitions ---------------------------------------

@pytest.mark.asyncio
async def test_blocked_status_transition_schedules_replan(tmp_path, agent_db):
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
    assert scope not in plugin._pending_replan

    updated = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10, user_id=42,
        action="update",
        tasks=[{"id": "T1", "status": "blocked"}],
    )
    assert updated["success"] is True
    assert plugin._pending_replan[scope] == {"reason": "blocked", "task_id": "T1"}


@pytest.mark.asyncio
async def test_blocked_status_no_op_no_replan_when_already_blocked(tmp_path, agent_db):
    plugin, helper = _make_plugin(tmp_path, agent_db)
    seeded = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10, user_id=42,
        action="add",
        definition_of_done=PLAN_CONTRACT,
        tasks=[{"id": "T1", "content": "Work", "status": "blocked"}],
    )
    assert seeded["success"] is True
    scope = compute_scope_key(10, 42)
    # First blocked-on-add schedules replan.
    assert plugin._pending_replan[scope] == {"reason": "blocked", "task_id": "T1"}
    plugin._pending_replan.pop(scope, None)

    # No-op update to blocked again should NOT re-schedule (status didn't change).
    again = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10, user_id=42,
        action="update",
        tasks=[{"id": "T1", "status": "blocked"}],
    )
    assert again["success"] is True
    assert scope not in plugin._pending_replan


@pytest.mark.asyncio
async def test_clear_plan_clears_streak_and_pending(tmp_path, agent_db):
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
    await plugin._record_tool_outcome(scope, "T1", False, "ddg.search")
    plugin._schedule_replan(scope, "errors", "T1")
    assert scope in plugin._tool_error_streaks or scope in plugin._pending_replan

    cleared = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10, user_id=42,
        action="clear",
    )
    assert cleared["success"] is True
    assert scope not in plugin._tool_error_streaks
    assert scope not in plugin._pending_replan


@pytest.mark.asyncio
async def test_session_reset_clears_streak_and_pending(tmp_path, agent_db):
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
    await plugin._record_tool_outcome(scope, "T1", False, "ddg.search")
    plugin._schedule_replan(scope, "errors", "T1")

    await plugin.on_session_reset(SessionResetPayload(
        chat_id=10, user_id=42, reason="request_start", terminal_only=False,
    ))
    assert scope not in plugin._tool_error_streaks
    assert scope not in plugin._pending_replan


@pytest.mark.asyncio
async def test_session_reset_terminal_only_preserves_pending(tmp_path, agent_db):
    """terminal_only=True clears finished tasks but the session continues —
    a pending re-plan trigger must survive to be delivered next chat request."""
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
    plugin._schedule_replan(scope, "errors", "T1")
    await plugin._record_tool_outcome(scope, "T1", False, "ddg.search")

    await plugin.on_session_reset(SessionResetPayload(
        chat_id=10, user_id=42, reason="request_start", terminal_only=True,
    ))
    # In-session terminal-only reset must NOT wipe re-plan trigger state.
    assert scope in plugin._pending_replan
    assert scope in plugin._tool_error_streaks


# ---------- Mutator delivery -------------------------------------------------

@pytest.mark.asyncio
async def test_mutator_injects_replan_message_when_pending(tmp_path, agent_db):
    plugin, _ = _make_plugin(tmp_path, agent_db, allowed=["agent_tools"])
    scope = compute_scope_key(10, 42)
    plugin._schedule_replan(scope, "errors", "T1")

    messages = [
        {"role": "system", "content": "mode prompt"},
        {"role": "user", "content": "hi"},
    ]
    new = await plugin.on_before_chat_request(messages, _payload())
    assert new is not None
    # Replan trigger message present somewhere as a system message.
    triggers = [
        m for m in new
        if isinstance(m, dict)
        and m.get("role") == "system"
        and isinstance(m.get("content"), str)
        and m["content"].startswith(_REPLAN_TRIGGER_MARKER)
    ]
    assert len(triggers) == 1
    assert "T1" in triggers[0]["content"]
    assert "consecutive errors" in triggers[0]["content"]
    # Pending entry was popped (one-shot).
    assert scope not in plugin._pending_replan


@pytest.mark.asyncio
async def test_mutator_pops_pending_once_idempotent(tmp_path, agent_db):
    plugin, _ = _make_plugin(tmp_path, agent_db, allowed=["agent_tools"])
    scope = compute_scope_key(10, 42)
    plugin._schedule_replan(scope, "blocked", "T1")

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
        and m["content"].startswith(_REPLAN_TRIGGER_MARKER)
    ]
    assert len(triggers1) == 1

    # Second invocation finds no pending entry. The plan-rule mutator may still
    # inject the plan-rule message, but no replan trigger message.
    second = await plugin.on_before_chat_request(messages, _payload())
    if second is None:
        return
    triggers2 = [
        m for m in second
        if isinstance(m, dict)
        and isinstance(m.get("content"), str)
        and m["content"].startswith(_REPLAN_TRIGGER_MARKER)
    ]
    assert triggers2 == []


@pytest.mark.asyncio
async def test_mutator_replan_message_for_blocked_reason(tmp_path, agent_db):
    plugin, _ = _make_plugin(tmp_path, agent_db, allowed=["agent_tools"])
    scope = compute_scope_key(10, 42)
    plugin._schedule_replan(scope, "blocked", "T7")

    messages = [
        {"role": "system", "content": "mode prompt"},
        {"role": "user", "content": "hi"},
    ]
    new = await plugin.on_before_chat_request(messages, _payload())
    triggers = [
        m for m in (new or [])
        if isinstance(m, dict)
        and isinstance(m.get("content"), str)
        and m["content"].startswith(_REPLAN_TRIGGER_MARKER)
    ]
    assert len(triggers) == 1
    assert "T7" in triggers[0]["content"]
    assert "blocked" in triggers[0]["content"]


# ---------- Regression: blocked behaves as "open" for delivery ---------------

@pytest.mark.asyncio
async def test_delivery_plan_error_treats_blocked_as_open(tmp_path, agent_db):
    plugin, helper = _make_plugin(tmp_path, agent_db)
    seeded = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10, user_id=42,
        action="add",
        definition_of_done=PLAN_CONTRACT,
        tasks=[{"id": "T1", "content": "Work", "status": "blocked"}],
    )
    assert seeded["success"] is True
    scope = compute_scope_key(10, 42)
    err = plugin._delivery_plan_error(scope, "completed", "summary")
    assert err is not None
    assert "T1" in err
