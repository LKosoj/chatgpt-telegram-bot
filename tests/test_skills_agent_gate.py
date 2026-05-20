"""Tests for the skills_agent first-turn planner gate in bot/openai_helper.py.

Gate contract (briefly):
- Active iff: current mode is skills_agent, function calling is enabled, the request
  has a non-empty `tools` list, the model is not in the excluded families, the
  request is non-streaming, and no plan task exists yet for the scope.
- If active and the model returns any execution-tool call (anything not in
  INFORMATION_ONLY_TOOLS) on the FIRST turn of the request, drop the response and
  retry with tool_choice pinned to manage_plan_tasks.
- Fires at most once per request (_gate_fired flag).
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

_INSERTED_MODULES = []


def _install_module_if_missing(name, module):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = module
        _INSERTED_MODULES.append(name)


class _FakeEncoding:
    def encode(self, value):
        return list(value)


_tiktoken = types.ModuleType("tiktoken")
_tiktoken.encoding_for_model = lambda _model: _FakeEncoding()
_tiktoken.get_encoding = lambda _name: _FakeEncoding()
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
# _build_force_planner_tool_choice
# ---------------------------------------------------------------------------


def test_gate_forced_retry_uses_correct_tool_choice_shape():
    helper, _ = _make_helper()
    tools = _tools_list_with_planner()
    choice = helper._build_force_planner_tool_choice(tools)
    assert choice == {
        "type": "function",
        "function": {"name": "agent_tools.manage_plan_tasks"},
    }


def test_build_force_planner_falls_back_to_auto_when_no_planner():
    helper, _ = _make_helper()
    tools = [
        {
            "type": "function",
            "function": {"name": "terminal.terminal", "parameters": {}},
        }
    ]
    assert helper._build_force_planner_tool_choice(tools) == "auto"


def test_build_force_planner_handles_empty_tools():
    helper, _ = _make_helper()
    assert helper._build_force_planner_tool_choice([]) == "auto"


def test_build_force_planner_handles_bare_name_suffix():
    """The lookup is suffix-based on .endswith('manage_plan_tasks') — defensive
    against unexpected prefixes from non-OpenAI gateways."""
    helper, _ = _make_helper()
    tools = [
        {
            "type": "function",
            "function": {"name": "weird_prefix/manage_plan_tasks"},
        }
    ]
    choice = helper._build_force_planner_tool_choice(tools)
    assert choice == {
        "type": "function",
        "function": {"name": "weird_prefix/manage_plan_tasks"},
    }


# ---------------------------------------------------------------------------
# should_force_non_stream_first_turn (telegram_bot dispatch helper)
# ---------------------------------------------------------------------------


def test_should_force_non_stream_first_turn_true_in_skills_agent_without_plan():
    helper, _ = _make_helper(tasks=[])
    assert helper.should_force_non_stream_first_turn(42, 7) is True


def test_should_force_non_stream_first_turn_returns_false_when_plan_exists():
    helper, _ = _make_helper(tasks=[{"id": "t1", "content": "x", "status": "pending"}])
    assert helper.should_force_non_stream_first_turn(42, 7) is False


def test_should_force_non_stream_first_turn_false_in_other_modes():
    helper, _ = _make_helper(mode_key="default")
    assert helper.should_force_non_stream_first_turn(42, 7) is False


def test_should_force_non_stream_first_turn_false_when_flag_off():
    helper, _ = _make_helper(skills_agent_mode={"key": "skills_agent"})
    # No force_non_stream_first_turn -> dispatcher streams as usual.
    assert helper.should_force_non_stream_first_turn(42, 7) is False


# ---------------------------------------------------------------------------
# End-to-end gate behavior through __common_get_chat_response
# ---------------------------------------------------------------------------


class FakeToolCall:
    def __init__(self, name, arguments="{}", id=None):
        self.id = id or f"call_{name.replace('.', '_')}"
        self.function = types.SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls
        self.content = content


class FakeChoice:
    def __init__(self, tool_calls=None, content=""):
        self.message = FakeMessage(tool_calls=tool_calls, content=content)
        self.delta = None
        self.finish_reason = None


class FakeResponse:
    def __init__(self, tool_calls=None, content="", total_tokens=3):
        self.choices = [FakeChoice(tool_calls=tool_calls, content=content)]
        self.usage = types.SimpleNamespace(
            total_tokens=total_tokens,
            prompt_tokens=1,
            completion_tokens=2,
        )


class RecordingClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return FakeResponse(content="done")


async def _invoke_common(helper, *, chat_id, tools, stream=False, model="gpt-4o"):
    """Bypass all session/history/summarization machinery and invoke just the
    gate-relevant logic from __common_get_chat_response. We replicate the
    create-call + gate block manually so the test focuses on the gate."""

    state_key = helper._chat_state_key(chat_id)
    memory_user_id = 7
    common_args = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        "tool_choice": "auto",
        "stream": stream,
    }

    # Mirror the gate block from __common_get_chat_response.
    from bot.openai_helper import O_MODELS, GOOGLE, PERPLEXITY

    gate_supported_model = model not in (O_MODELS + GOOGLE + PERPLEXITY)
    gate_active = (
        not common_args.get("stream")
        and bool(common_args.get("tools"))
        and gate_supported_model
        and helper._is_skills_agent_mode(chat_id)
        and helper._skills_agent_gate_enabled_for_mode()
        and not helper._skills_agent_has_plan(chat_id, memory_user_id)
    )
    gate_fired = helper._gate_fired.get(state_key, False)

    response = await helper.client.chat.completions.create(**common_args)

    if gate_active and not gate_fired:
        try:
            raw_tool_calls = getattr(response.choices[0].message, "tool_calls", None) or []
            tool_calls_normalized = [
                {"name": getattr(getattr(tc, "function", None), "name", None)}
                for tc in raw_tool_calls
            ]
        except Exception:
            tool_calls_normalized = []
        if helper._skills_agent_gate_should_fire(tool_calls_normalized):
            retry_args = dict(common_args)
            retry_args["tool_choice"] = helper._build_force_planner_tool_choice(
                retry_args.get("tools") or []
            )
            helper._gate_fired[state_key] = True
            response = await helper.client.chat.completions.create(**retry_args)

    return response


@pytest.mark.asyncio
async def test_gate_e2e_forces_retry_when_execution_tool_called_first():
    helper, _ = _make_helper(tasks=[])
    helper.client = RecordingClient(
        responses=[
            # First call: model jumps straight to terminal.terminal — bad.
            FakeResponse(tool_calls=[FakeToolCall("terminal.terminal")]),
            # Retry: model now calls manage_plan_tasks.
            FakeResponse(tool_calls=[FakeToolCall("agent_tools.manage_plan_tasks")]),
        ]
    )

    response = await _invoke_common(
        helper, chat_id=42, tools=_tools_list_with_planner()
    )

    assert len(helper.client.calls) == 2
    first, retry = helper.client.calls
    assert first["tool_choice"] == "auto"
    assert retry["tool_choice"] == {
        "type": "function",
        "function": {"name": "agent_tools.manage_plan_tasks"},
    }
    final_call_name = response.choices[0].message.tool_calls[0].function.name
    assert final_call_name == "agent_tools.manage_plan_tasks"


@pytest.mark.asyncio
async def test_gate_e2e_does_not_retry_for_information_only_tool():
    helper, _ = _make_helper(tasks=[])
    helper.client = RecordingClient(
        responses=[FakeResponse(tool_calls=[FakeToolCall("skills.list_skills")])]
    )
    await _invoke_common(helper, chat_id=42, tools=_tools_list_with_planner())
    assert len(helper.client.calls) == 1


@pytest.mark.asyncio
async def test_gate_e2e_skips_when_plan_exists():
    helper, _ = _make_helper(
        tasks=[{"id": "t1", "content": "x", "status": "pending"}]
    )
    helper.client = RecordingClient(
        responses=[FakeResponse(tool_calls=[FakeToolCall("terminal.terminal")])]
    )
    await _invoke_common(helper, chat_id=42, tools=_tools_list_with_planner())
    # Even though terminal.terminal was called, the plan already exists, so the
    # gate is inactive and no retry happens.
    assert len(helper.client.calls) == 1


@pytest.mark.asyncio
async def test_gate_e2e_inactive_in_other_modes():
    helper, _ = _make_helper(mode_key="default", tasks=[])
    helper.client = RecordingClient(
        responses=[FakeResponse(tool_calls=[FakeToolCall("terminal.terminal")])]
    )
    await _invoke_common(helper, chat_id=42, tools=_tools_list_with_planner())
    assert len(helper.client.calls) == 1


@pytest.mark.asyncio
async def test_gate_e2e_inactive_when_enable_functions_false():
    """If function calling is disabled, the request has no tools list (the
    helper never sets one) — the gate predicate excludes empty-tools requests."""
    helper, _ = _make_helper(tasks=[])
    helper.client = RecordingClient(
        responses=[FakeResponse(content="just text, no tools")]
    )
    # tools=[] simulates enable_functions=False / model without function support.
    await _invoke_common(helper, chat_id=42, tools=[])
    assert len(helper.client.calls) == 1


@pytest.mark.asyncio
async def test_gate_e2e_fires_only_once_per_request():
    """Even if the retry response somehow still calls an execution-tool, the
    gate must not retry again — _gate_fired is now True."""
    helper, _ = _make_helper(tasks=[])
    helper.client = RecordingClient(
        responses=[
            FakeResponse(tool_calls=[FakeToolCall("terminal.terminal")]),
            # Hypothetical second response still calls terminal (gateway/SDK
            # disrespected tool_choice).
            FakeResponse(tool_calls=[FakeToolCall("terminal.terminal")]),
        ]
    )

    await _invoke_common(helper, chat_id=42, tools=_tools_list_with_planner())
    # Exactly two calls: one original + one forced retry. No third call.
    assert len(helper.client.calls) == 2
    state_key = helper._chat_state_key(42)
    assert helper._gate_fired.get(state_key) is True

    # A second "_invoke_common" without resetting _gate_fired must NOT retry.
    helper.client._responses.append(
        FakeResponse(tool_calls=[FakeToolCall("terminal.terminal")])
    )
    await _invoke_common(helper, chat_id=42, tools=_tools_list_with_planner())
    assert len(helper.client.calls) == 3  # only the new initial call, no retry


@pytest.mark.asyncio
async def test_gate_e2e_skips_pure_text_response():
    helper, _ = _make_helper(tasks=[])
    helper.client = RecordingClient(
        responses=[FakeResponse(tool_calls=None, content="just a plain text reply")]
    )
    await _invoke_common(helper, chat_id=42, tools=_tools_list_with_planner())
    assert len(helper.client.calls) == 1


@pytest.mark.asyncio
async def test_gate_e2e_skips_when_planner_already_called():
    helper, _ = _make_helper(tasks=[])
    helper.client = RecordingClient(
        responses=[
            FakeResponse(
                tool_calls=[FakeToolCall("agent_tools.manage_plan_tasks")]
            )
        ]
    )
    await _invoke_common(helper, chat_id=42, tools=_tools_list_with_planner())
    assert len(helper.client.calls) == 1


@pytest.mark.asyncio
async def test_gate_e2e_inactive_when_mode_flag_off():
    """When the skills_agent mode dict lacks force_non_stream_first_turn, the
    gate is the no-op legacy path even on terminal tool calls. Keeps backward
    compatibility for tests/users that mock the registry without the flag."""
    helper, _ = _make_helper(
        skills_agent_mode={"key": "skills_agent"},  # no force_non_stream_first_turn
        tasks=[],
    )
    helper.client = RecordingClient(
        responses=[FakeResponse(tool_calls=[FakeToolCall("terminal.terminal")])]
    )
    await _invoke_common(helper, chat_id=42, tools=_tools_list_with_planner())
    assert len(helper.client.calls) == 1


@pytest.mark.asyncio
async def test_gate_e2e_excluded_model_family_skipped():
    """Models in the excluded families (e.g. Perplexity) do not get tools and
    the gate must not fire for them."""
    helper, _ = _make_helper(tasks=[])
    helper.client = RecordingClient(
        responses=[FakeResponse(tool_calls=[FakeToolCall("terminal.terminal")])]
    )
    # Pick a model that's in PERPLEXITY tuple — use the constant.
    from bot.openai_helper import PERPLEXITY

    if not PERPLEXITY:
        pytest.skip("PERPLEXITY tuple is empty in this build")
    await _invoke_common(
        helper, chat_id=42, tools=_tools_list_with_planner(), model=PERPLEXITY[0]
    )
    assert len(helper.client.calls) == 1
