"""Stage 3 — agent_tools owns its schema via register_schema()."""
import importlib.machinery
import importlib.util
import sys
import types

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.database import Database
from bot.plugin_manager import PluginManager
from bot.plugins.agent_tools import AgentToolsPlugin


def test_register_plugin_schemas_creates_agent_plan_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "fresh.db"))
    Database._reset_singleton()
    try:
        db = Database()
        # Drop the tables (simulate fresh post-migration state)
        with db.get_connection() as conn:
            conn.execute("DROP TABLE IF EXISTS agent_plan_tasks")
            conn.execute("DROP TABLE IF EXISTS agent_plan_contracts")

        plugin_dir = tmp_path / "p"
        plugin_dir.mkdir()
        pm = PluginManager(config={"plugins": ["agent_tools"]}, plugins_directory=str(plugin_dir))
        pm.plugins["agent_tools"] = AgentToolsPlugin
        pm.set_db(db)
        pm.register_plugin_schemas()

        with db.get_connection() as conn:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "agent_plan_tasks" in names
        assert "agent_plan_contracts" in names
        assert "agent_working_checkpoints" in names
        assert "agent_goal_runs" in names
        assert "agent_goal_run_events" in names
        with db.get_connection() as conn:
            idxs = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()}
        assert "idx_agent_plan_tasks_scope_position" in idxs
    finally:
        Database._reset_singleton()


def test_register_schema_returns_ddl_without_initialize():
    plugin = AgentToolsPlugin()
    # bare — no initialize called
    stmts = plugin.register_schema()
    assert isinstance(stmts, list)
    assert len(stmts) == 9
    assert all("IF NOT EXISTS" in stmt for stmt in stmts)
    # plugin state untouched
    assert getattr(plugin, "db", None) is None
    assert getattr(plugin, "db_handle", None) is None
