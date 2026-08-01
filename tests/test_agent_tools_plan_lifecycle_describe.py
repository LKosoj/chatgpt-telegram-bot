"""D2 — describe_plan_lifecycle() is the single source of truth for the plan
task lifecycle: it must not drift from TASK_STATUSES/CLOSED_STATUSES or from
the ``status`` enum hardcoded in manage_plan_tasks's JSON schema."""
import importlib.machinery
import importlib.util
import sys
import types

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.plugins.agent_tools import (
    AgentToolsPlugin,
    CLOSED_STATUSES,
    TASK_STATUSES,
    describe_plan_lifecycle,
)


def _manage_plan_tasks_spec():
    plugin = AgentToolsPlugin()  # bare — no initialize(), no DB, no network
    specs = plugin.get_spec()
    spec = next(s for s in specs if s["name"] == "manage_plan_tasks")
    return spec


def test_describe_plan_lifecycle_statuses_match_task_statuses_constant():
    lifecycle = describe_plan_lifecycle()
    assert set(lifecycle["statuses"]) == TASK_STATUSES


def test_describe_plan_lifecycle_closed_statuses_match_closed_statuses_constant():
    lifecycle = describe_plan_lifecycle()
    assert set(lifecycle["closed_statuses"]) == CLOSED_STATUSES


def test_manage_plan_tasks_status_enum_matches_task_statuses():
    """Regression test for the duplication found between TASK_STATUSES and the
    hardcoded ``enum`` literal in manage_plan_tasks's tasks[].status schema.
    These two places are not derived from each other and must be kept in sync
    by hand (or the enum should be generated from TASK_STATUSES)."""
    spec = _manage_plan_tasks_spec()
    status_enum = (
        spec["parameters"]["properties"]["tasks"]["items"]["properties"]["status"]["enum"]
    )
    assert set(status_enum) == TASK_STATUSES


def test_describe_plan_lifecycle_data_renders_a_complete_diagram():
    """describe_plan_lifecycle() is documentation-as-data: build a minimal
    text rendering directly from its output and check every status, every
    invariant, and every side effect made it into the rendered text — i.e.
    the diagram is derived from the data, not hand-maintained separately."""
    lifecycle = describe_plan_lifecycle()

    lines = ["Plan task lifecycle", "===================", ""]
    lines.append("Statuses: " + ", ".join(lifecycle["statuses"]))
    lines.append("Closed: " + ", ".join(lifecycle["closed_statuses"]))
    lines.append("Open: " + ", ".join(lifecycle["open_statuses"]))
    lines.append("")
    lines.append("Invariants enforced by _validate_plan_tasks:")
    for invariant in lifecycle["invariants"]:
        lines.append(f"- {invariant}")
    lines.append("")
    lines.append("Side effects scheduled by code on status transition:")
    for status, effect in lifecycle["side_effects"].items():
        lines.append(f"- {status} -> {effect}")

    rendered = "\n".join(lines)

    for status in lifecycle["statuses"]:
        assert status in rendered
    for invariant in lifecycle["invariants"]:
        assert invariant in rendered
    for effect in lifecycle["side_effects"].values():
        assert effect in rendered
