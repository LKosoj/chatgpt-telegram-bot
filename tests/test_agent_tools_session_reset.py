"""Stage 3 — agent_tools on_session_reset observer."""
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
from bot.plugin_manager import PluginManager
from bot.plugins.agent_tools import AgentToolsPlugin
from bot.plugins.db_handle import DbHandle
from bot.plugins.hooks import SessionResetPayload


PLAN_CONTRACT = {
    "goal": "Finish reset test",
    "success_criteria": ["Plan state matches reset policy"],
    "verification": ["Checked persisted plan state"],
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


def _make_plugin(tmp_path, db):
    helper = SimpleNamespace(user_id=42, db=db)
    plugin = AgentToolsPlugin()
    plugin.initialize(openai=helper, storage_root=str(tmp_path), db=DbHandle(db))
    return plugin, helper


@pytest.mark.asyncio
async def test_on_session_reset_clears_all_tasks(tmp_path, agent_db):
    plugin, helper = _make_plugin(tmp_path, agent_db)
    seeded = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        user_id=42,
        action="add",
        definition_of_done={
            "goal": "Finish review",
            "success_criteria": ["No stale plan contract remains"],
            "verification": ["Listed plan after reset"],
        },
        tasks=[
            {"id": "T1", "content": "Inspect", "status": "in_progress"},
        ],
    )
    assert seeded["success"] is True
    assert plugin.get_plan_tasks(chat_id=10, user_id=42)
    assert seeded["plan_tasks"]["definition_of_done"]["goal"] == "Finish review"

    await plugin.on_session_reset(SessionResetPayload(
        chat_id=10, user_id=42, reason="request_start", terminal_only=False,
    ))

    listed = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        user_id=42,
        action="list",
    )
    assert listed["plan_tasks"]["tasks"] == []
    assert listed["plan_tasks"]["definition_of_done"] is None


@pytest.mark.asyncio
async def test_on_session_reset_terminal_only_preserves_open_tasks(tmp_path, agent_db):
    plugin, helper = _make_plugin(tmp_path, agent_db)
    seeded = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        user_id=42,
        action="add",
        definition_of_done=PLAN_CONTRACT,
        tasks=[
            {"id": "T1", "content": "Open work", "status": "in_progress"},
        ],
    )
    assert seeded["success"] is True

    await plugin.on_session_reset(SessionResetPayload(
        chat_id=10, user_id=42, reason="final_delivery", terminal_only=True,
    ))

    remaining = plugin.get_plan_tasks(chat_id=10, user_id=42)
    assert [t["id"] for t in remaining] == ["T1"]
    assert remaining[0]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_on_session_reset_terminal_only_clears_when_all_closed(tmp_path, agent_db):
    plugin, helper = _make_plugin(tmp_path, agent_db)
    seeded = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        user_id=42,
        action="add",
        definition_of_done={
            "goal": "Finish delivery",
            "success_criteria": ["Closed plan is fully cleared"],
            "verification": ["Listed plan after terminal reset"],
        },
        tasks=[
            {"id": "T1", "content": "Done work", "status": "completed"},
        ],
    )
    assert seeded["success"] is True

    await plugin.on_session_reset(SessionResetPayload(
        chat_id=10, user_id=42, reason="final_delivery", terminal_only=True,
    ))

    listed = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        user_id=42,
        action="list",
    )
    assert listed["plan_tasks"]["tasks"] == []
    assert listed["plan_tasks"]["definition_of_done"] is None


@pytest.mark.asyncio
async def test_dispatch_observe_routes_to_agent_tools(tmp_path, agent_db):
    plugin_dir = tmp_path / "p"
    plugin_dir.mkdir()
    pm = PluginManager(config={"plugins": ["agent_tools"]}, plugins_directory=str(plugin_dir))
    pm.plugins["agent_tools"] = AgentToolsPlugin
    pm.set_db(agent_db)
    pm.register_plugin_schemas()
    fake_openai = SimpleNamespace(db=agent_db, config={}, bot=None, plugin_manager=pm)
    pm.set_openai(fake_openai)
    plugin = pm.get_plugin("agent_tools")
    helper = SimpleNamespace(user_id=42, db=agent_db)

    seeded = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        user_id=42,
        action="add",
        definition_of_done=PLAN_CONTRACT,
        tasks=[{"id": "T1", "content": "Work", "status": "in_progress"}],
    )
    assert seeded["success"] is True
    assert plugin.get_plan_tasks(chat_id=10, user_id=42)

    await pm.dispatch_observe(
        "on_session_reset",
        SessionResetPayload(
            chat_id=10, user_id=42, reason="request_start", terminal_only=False,
        ),
        user_id=42,
    )

    assert plugin.get_plan_tasks(chat_id=10, user_id=42) == []
