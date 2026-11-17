"""Tests for mode selection and planner preflight decisions."""

from __future__ import annotations

import importlib.util
import sys
import types
from unittest.mock import AsyncMock

import pytest

_INSERTED_MODULES = []


def _install_module_if_missing(name, module):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = module
        _INSERTED_MODULES.append(name)


_tiktoken = types.ModuleType("tiktoken")
_tiktoken.encoding_for_model = lambda _model: FakeEncoding()
_tiktoken.get_encoding = lambda _name: FakeEncoding()
_install_module_if_missing("tiktoken", _tiktoken)

_markdown2 = types.ModuleType("markdown2")
_markdown2.markdown = lambda text, *args, **kwargs: text
_install_module_if_missing("markdown2", _markdown2)


def _retry(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


_tenacity = types.ModuleType("tenacity")
_tenacity.retry = _retry
_tenacity.stop_after_attempt = lambda *args, **kwargs: None
_tenacity.wait_fixed = lambda *args, **kwargs: None
_tenacity.retry_if_exception_type = lambda *args, **kwargs: None
_install_module_if_missing("tenacity", _tenacity)


from bot.openai_helper import (  # noqa: E402
    INFORMATION_ONLY_TOOLS,
    OpenAIHelper,
)
from tests.fakes import FakeChoice, FakeEncoding  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class DummyDB:
    """Minimal DB stub for context loading paths."""

    def __init__(self, context=None):
        self.context = context or {"messages": []}
        self.user_settings = {}

    def get_conversation_context(self, *args, **kwargs):
        return self.context, None, 0.1, 80, "session-1"

    def save_conversation_context(self, *args, **kwargs):  # pragma: no cover
        pass

    def list_user_sessions(self, user_id, is_active=1):  # pragma: no cover
        return []

    def get_user_settings(self, user_id):
        return self.user_settings.get(user_id)


class StubChatModesRegistry:
    """Mimics ChatModesRegistry returning a hard-coded mode dict by key.

    Identity comparison: every call to get_mode_by_key('skills_agent') returns
    the SAME dict instance, matching the production registry's behavior on a
    single request (data is loaded once from yaml).
    """

    def __init__(self, modes):
        self._modes = modes

    def get_mode_by_key(self, key):
        return self._modes.get(key)

    def get_mode_by_system_prompt(self, content):
        for data in self._modes.values():
            if isinstance(data, dict) and data.get("prompt_start", "") == content:
                return data
        return None

    def validate_tools(self, plugin_manager):
        return None

    def all_modes(self):
        return self._modes


class StubAgentToolsPlugin:
    plugin_id = "agent_tools"

    def __init__(self, tasks=None, raises=False):
        self._tasks = tasks or []
        self._raises = raises
        self.calls = []

    def get_plan_tasks(self, chat_id=None, user_id=None):
        self.calls.append((chat_id, user_id))
        if self._raises:
            raise RuntimeError("storage failure")
        return list(self._tasks)


class StubPluginManager:
    def __init__(self, plugins=None):
        self.plugins = plugins or {}

    def get_plugin(self, name):
        return self.plugins.get(name)


def _make_helper(
    *,
    mode_key="skills_agent",
    skills_agent_mode=None,
    tasks=None,
    plugin_raises=False,
):
    """Construct an OpenAIHelper-like object with the minimal state needed for
    gate unit tests. We do NOT call OpenAIHelper.__init__ — it requires an
    openai client. Instead we bind the bound methods we want to test to a
    SimpleNamespace-like proxy, then attach the helper attributes the gate
    methods read from."""

    helper = OpenAIHelper.__new__(OpenAIHelper)
    if skills_agent_mode is None:
        skills_agent_mode = {
            "key": "skills_agent",
            "force_non_stream_first_turn": True,
        }
    modes = {"skills_agent": skills_agent_mode}
    helper.chat_modes_registry = StubChatModesRegistry(modes)
    helper.plugin_manager = StubPluginManager(
        plugins={"agent_tools": StubAgentToolsPlugin(tasks=tasks, raises=plugin_raises)}
    )
    state_key = 42
    helper.conversations = {
        state_key: [
            {
                "role": "system",
                "mode_key": mode_key,
                "content": "skills agent system prompt",
            }
        ]
    }
    helper._gate_fired = {}
    # Patch _chat_state_key to a deterministic value for this synthetic chat.
    helper._chat_state_key = lambda chat_id: state_key  # type: ignore[method-assign]
    return helper, state_key


def _tools_list_with_planner():
    return [
        {
            "type": "function",
            "function": {
                "name": "agent_tools.manage_plan_tasks",
                "description": "Plan tasks",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "skills.list_skills",
                "description": "List skills",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminal.terminal",
                "description": "Run a shell command",
                "parameters": {},
            },
        },
    ]


@pytest.mark.asyncio
async def test_manual_mode_is_not_replaced_on_first_message():
    helper, _ = _make_helper(mode_key="skills_agent")
    helper.config = {"auto_chat_modes": True}
    helper.chat_completion = AsyncMock()
    await helper._maybe_apply_auto_chat_mode(
        42, "hello", session_id="session-1", parse_mode=None,
        temperature=None, max_tokens_percent=None, user_id=42,
    )
    helper.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_mode_failure_keeps_existing_mode():
    helper, key = _make_helper(mode_key=None)
    helper.config = {"auto_chat_modes": True, "model": "test"}
    helper._build_auto_chat_mode_prompt = AsyncMock(return_value="choose")
    helper.get_max_tokens = lambda *args: 20
    helper.chat_completion = AsyncMock(side_effect=RuntimeError("unavailable"))
    before = dict(helper.conversations[key][0])
    await helper._maybe_apply_auto_chat_mode(
        42, "hello", session_id="session-1", parse_mode=None,
        temperature=None, max_tokens_percent=None, user_id=42,
    )
    assert helper.conversations[key][0] == before


def test_preflight_checks_each_batch_after_information_tool():
    helper, _ = _make_helper(tasks=[])
    info = [{"name": "skills.list_skills"}]
    execution = [{"name": "terminal.terminal"}]
    assert not helper.plan_preflight_required(42, 42, None, info)
    assert helper.plan_preflight_required(42, 42, None, execution)
    assert not helper.plan_preflight_required(42, 42, None, [{"name": "agent_tools.deliver_to_user"}])


@pytest.mark.asyncio
async def test_prepare_mode_loads_pinned_session_before_stream_decision():
    helper, key = _make_helper(mode_key="default")
    helper.config = {"auto_chat_modes": True}
    helper.conversations = {}
    helper.loaded_conversation_sessions = {}
    messages = [{"role": "system", "mode_key": "skills_agent", "content": "prompt"}]
    helper._db_call = AsyncMock(return_value=({"messages": messages}, None, 0.1, 80, "session-1"))
    helper.chat_completion = AsyncMock()
    assert await helper.prepare_chat_mode(42, "work", 42, "session-1")
    assert helper.loaded_conversation_sessions[key] == "session-1"
    helper.chat_completion.assert_not_awaited()


# ---------------------------------------------------------------------------
# _skills_agent_gate_should_fire
# ---------------------------------------------------------------------------


def test_gate_skips_pure_text_response():
    helper, _ = _make_helper()
    # No tool_calls (model returned only text) -> gate stays quiet.
    assert helper._skills_agent_gate_should_fire([]) is False


def test_gate_fires_on_terminal_tool_call_when_no_plan_in_skills_agent():
    helper, _ = _make_helper()
    tool_calls = [{"name": "terminal.terminal"}]
    assert helper._skills_agent_gate_should_fire(tool_calls) is True


@pytest.mark.parametrize("tool_name", sorted(INFORMATION_ONLY_TOOLS))
def test_gate_skips_information_only_tools(tool_name):
    helper, _ = _make_helper()
    # A single information-only tool call must NOT trigger the gate.
    assert helper._skills_agent_gate_should_fire([{"name": tool_name}]) is False


def test_gate_skips_when_planner_already_called():
    """When the only tool the model called is manage_plan_tasks (the planner
    itself), the gate must not fire — the model is doing what we want."""
    helper, _ = _make_helper()
    assert (
        helper._skills_agent_gate_should_fire(
            [{"name": "agent_tools.manage_plan_tasks"}]
        )
        is False
    )


def test_gate_fires_on_mixed_calls_with_at_least_one_execution_tool():
    helper, _ = _make_helper()
    tool_calls = [
        {"name": "skills.list_skills"},
        {"name": "terminal.terminal"},
    ]
    assert helper._skills_agent_gate_should_fire(tool_calls) is True


def test_gate_ignores_malformed_entries():
    helper, _ = _make_helper()
    # None and dicts without 'name' should be silently ignored.
    assert helper._skills_agent_gate_should_fire([{"foo": "bar"}, {}]) is False


# ---------------------------------------------------------------------------
# _skills_agent_has_plan
# ---------------------------------------------------------------------------


def test_skills_agent_has_plan_false_when_no_tasks():
    helper, _ = _make_helper(tasks=[])
    assert helper._skills_agent_has_plan(42, 7) is False


def test_skills_agent_has_plan_true_with_tasks():
    helper, _ = _make_helper(tasks=[{"id": "t1", "content": "x", "status": "pending"}])
    assert helper._skills_agent_has_plan(42, 7) is True


def test_skills_agent_has_plan_fail_open_on_storage_error():
    helper, _ = _make_helper(plugin_raises=True)
    # Storage failure must NOT block delivery — gate fails open.
    assert helper._skills_agent_has_plan(42, 7) is False


def test_skills_agent_has_plan_false_when_plugin_missing():
    helper, _ = _make_helper()
    helper.plugin_manager = StubPluginManager(plugins={})  # no agent_tools
    assert helper._skills_agent_has_plan(42, 7) is False


# ---------------------------------------------------------------------------
# _is_skills_agent_mode
# ---------------------------------------------------------------------------


def test_is_skills_agent_mode_true_with_mode_key():
    helper, _ = _make_helper(mode_key="skills_agent")
    assert helper._is_skills_agent_mode(42) is True


def test_gate_inactive_in_other_modes():
    helper, _ = _make_helper(mode_key="default")
    assert helper._is_skills_agent_mode(42) is False


def test_is_skills_agent_mode_false_when_no_system_message():
    helper, state_key = _make_helper()
    helper.conversations[state_key] = []  # no system message
    assert helper._is_skills_agent_mode(42) is False


# ---------------------------------------------------------------------------
# should_force_non_stream_first_turn_async (telegram_bot dispatch helper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_should_force_non_stream_first_turn_true_in_skills_agent_without_plan():
    helper, _ = _make_helper(tasks=[])
    assert await helper.should_force_non_stream_first_turn_async(42, 7) is True


@pytest.mark.asyncio
async def test_should_force_non_stream_first_turn_returns_false_when_plan_exists():
    helper, _ = _make_helper(tasks=[{"id": "t1", "content": "x", "status": "pending"}])
    assert await helper.should_force_non_stream_first_turn_async(42, 7) is False


@pytest.mark.asyncio
async def test_should_force_non_stream_first_turn_false_in_other_modes():
    helper, _ = _make_helper(mode_key="default")
    assert await helper.should_force_non_stream_first_turn_async(42, 7) is False


@pytest.mark.asyncio
async def test_should_force_non_stream_first_turn_false_when_flag_off():
    helper, _ = _make_helper(skills_agent_mode={"key": "skills_agent"})
    # No force_non_stream_first_turn -> dispatcher streams as usual.
    assert await helper.should_force_non_stream_first_turn_async(42, 7) is False
