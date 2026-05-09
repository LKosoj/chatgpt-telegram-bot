import asyncio
import importlib.util
import importlib.machinery
import json
import sys
import types
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.plugin_manager import PluginManager
from bot.i18n import localized_text
from bot.database import Database
from bot.plugins.agent_tools import AgentToolsPlugin
from bot.request_context import RequestContext


class FakeBot:
    def __init__(self):
        self.messages = []
        self.markup_edits = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.messages))

    async def edit_message_reply_markup(self, **kwargs):
        self.markup_edits.append(kwargs)


class FakeCompletions:
    def __init__(self, tool_calls=None):
        self.calls = []
        if tool_calls is None:
            tool_calls = [
                SimpleNamespace(
                    id="tool_1",
                    function=SimpleNamespace(
                        name="skills.list_skills",
                        arguments="{}",
                    ),
                )
            ]
        self.tool_calls = tool_calls

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools") and not any(message.get("role") == "tool" for message in kwargs["messages"]):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=self.tool_calls,
                        )
                    )
                ]
            )
        content = kwargs["messages"][-1]["content"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"done: {content.splitlines()[0]}", tool_calls=None)
                )
            ]
        )


class FakePluginManager:
    def __init__(self):
        self.calls = []

    def get_functions_specs(self, helper, model_to_use, allowed_plugins):
        return [
            {
                "type": "function",
                "function": {
                    "name": "skills.list_skills",
                    "description": "list skills",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "skills.get_skill",
                    "description": "get skill",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "agent_tools.deliver_to_user",
                    "description": "final delivery",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "agent_tools.run_subagents",
                    "description": "nested subagents",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "skills.run_skill_agent",
                    "description": "nested skill agent",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    async def call_function(self, function_name, helper, arguments, request_context=None):
        self.calls.append((function_name, json.loads(arguments)))
        return json.dumps({"success": True, "skills": ["demo"]}, ensure_ascii=False)

    def get_subagent_function_specs(
        self,
        helper,
        model_to_use,
        parent_allowed_plugins=None,
        blocked_function_names=None,
    ):
        blocked = set(blocked_function_names or ())
        specs = self.get_functions_specs(helper, model_to_use, parent_allowed_plugins or ["All"])
        filtered = []
        names = set()
        for tool in specs or []:
            name = (tool.get("function") or {}).get("name")
            if not name or name in blocked:
                continue
            filtered.append(tool)
            names.add(name)
        return filtered, names


class FakeLLMHelper:
    def __init__(self, completions=None, plugin_manager=None, model_name="llmgateway/high"):
        self.completions = completions or FakeCompletions()
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=self.completions))
        self.plugin_manager = plugin_manager or FakePluginManager()
        self.model_name = model_name

    def get_current_model(self, user_id):
        return self.model_name


class FakeBackgroundHelper(FakeLLMHelper):
    def __init__(self):
        super().__init__()
        self.requests = []

    async def get_chat_response(self, **kwargs):
        self.requests.append(kwargs)
        return "background done", 11


class ParallelToolPluginManager(FakePluginManager):
    def __init__(self):
        super().__init__()
        self.running = 0
        self.overlapped = False

    async def call_function(self, function_name, helper, arguments, request_context=None):
        self.running += 1
        if self.running > 1:
            self.overlapped = True
        try:
            await asyncio.sleep(0.01)
            self.calls.append((function_name, json.loads(arguments)))
            return json.dumps({"success": True}, ensure_ascii=False)
        finally:
            self.running -= 1


class SlowFakeCompletions(FakeCompletions):
    async def create(self, **kwargs):
        await asyncio.sleep(0.01)
        return await super().create(**kwargs)


class FakeMessage:
    def __init__(self, chat_id: int, text: str, user_id: int = 42):
        self.chat = SimpleNamespace(id=chat_id)
        self.from_user = SimpleNamespace(id=user_id)
        self.text = text
        self.replies = []

    async def reply_text(self, text: str):
        self.replies.append(text)


@pytest.fixture()
def agent_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "agent.db"))
    Database._instance = None
    return Database()


def _db_backed_agent_plugin(tmp_path, db):
    helper = SimpleNamespace(user_id=42, db=db)
    plugin = AgentToolsPlugin()
    plugin.initialize(openai=helper, storage_root=str(tmp_path))
    return plugin, helper


def test_agent_tools_registers_specs_and_handlers():
    pm = PluginManager(config={"plugins": ["agent_tools"]})

    specs = pm.get_functions_specs(helper=None, model_to_use="llmgateway/high", allowed_plugins=["agent_tools"])
    names = {spec["function"]["name"] for spec in specs}
    assert names == {
        "agent_tools.manage_plan_tasks",
        "agent_tools.ask_telegram_user",
        "agent_tools.cancel_pending_question",
        "agent_tools.run_subagents",
        "agent_tools.deliver_to_user",
    }
    ask_spec = next(spec["function"] for spec in specs if spec["function"]["name"] == "agent_tools.ask_telegram_user")
    assert ask_spec["parameters"]["required"] == ["question", "options"]
    assert ask_spec["parameters"]["properties"]["options"]["minItems"] == 1
    plan_spec = next(spec["function"] for spec in specs if spec["function"]["name"] == "agent_tools.manage_plan_tasks")
    task_props = plan_spec["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert "depends_on" in task_props
    assert "definition_of_done" in plan_spec["parameters"]["properties"]

    commands = pm.get_plugin_commands()
    assert any(command.get("command") == "background" for command in commands)
    assert any(command.get("callback_pattern") == "^agentask:" for command in commands)
    assert pm.get_message_handlers()


def test_agent_tools_preserves_full_option_text():
    plugin = AgentToolsPlugin()
    long_option = "A" * 120

    assert plugin._normalize_options([long_option]) == [long_option]


@pytest.mark.asyncio
async def test_manage_plan_tasks_tracks_progress(tmp_path, agent_db):
    plugin, helper = _db_backed_agent_plugin(tmp_path, agent_db)

    added = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        definition_of_done={
            "goal": "Improve agent planning",
            "success_criteria": ["Plan persists"],
            "verification": ["Run focused tests"],
            "constraints": ["Keep tool names stable"],
        },
        tasks=[
            {"id": "T1", "content": "Inspect implementation", "status": "completed"},
            {"id": "T2", "content": "Add Telegram ask tool", "status": "in_progress", "depends_on": ["T1"]},
        ],
    )

    assert added["success"] is True
    assert added["plan_tasks"]["progress"] == {"total": 2, "closed": 1, "open": 1}
    assert added["plan_tasks"]["definition_of_done"]["goal"] == "Improve agent planning"
    assert added["plan_tasks"]["tasks"][1]["depends_on"] == ["T1"]

    updated = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="update",
        tasks=[{"id": "T2", "status": "completed"}],
    )

    assert updated["success"] is True
    assert updated["plan_tasks"]["progress"] == {"total": 2, "closed": 2, "open": 0}

    cleared = await plugin.execute("manage_plan_tasks", helper, chat_id=10, action="clear")
    assert cleared["success"] is True
    assert cleared["plan_tasks"]["tasks"] == []


@pytest.mark.asyncio
async def test_manage_plan_tasks_persists_dag_and_contract_in_database(tmp_path, agent_db):
    plugin, helper = _db_backed_agent_plugin(tmp_path, agent_db)

    added = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        user_id=42,
        action="add",
        definition_of_done={
            "goal": "Ship SQLite planning",
            "success_criteria": ["DAG is persisted"],
            "verification": ["Reload plugin and list plan"],
        },
        tasks=[
            {"id": "T1", "content": "Add schema", "status": "completed"},
            {"id": "T2", "content": "Wire plugin", "status": "pending", "depends_on": ["T1"]},
        ],
    )
    assert added["success"] is True

    reloaded = AgentToolsPlugin()
    reloaded.initialize(openai=helper, storage_root=str(tmp_path))
    listed = await reloaded.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        user_id=42,
        action="list",
    )

    assert listed["plan_tasks"]["definition_of_done"]["goal"] == "Ship SQLite planning"
    assert listed["plan_tasks"]["tasks"][1]["depends_on"] == ["T1"]


@pytest.mark.asyncio
async def test_manage_plan_tasks_requires_database(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = SimpleNamespace(user_id=42)

    result = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="list",
    )

    assert result["success"] is False
    assert "requires database" in result["error"]


@pytest.mark.asyncio
async def test_manage_plan_tasks_rejects_open_dependency(tmp_path, agent_db):
    plugin, helper = _db_backed_agent_plugin(tmp_path, agent_db)

    result = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[
            {"id": "T1", "content": "Inspect", "status": "pending"},
            {"id": "T2", "content": "Implement", "status": "in_progress", "depends_on": ["T1"]},
        ],
    )

    assert result["success"] is False
    assert "dependencies are still open" in result["error"]


@pytest.mark.asyncio
async def test_manage_plan_tasks_rejects_dependency_cycle(tmp_path, agent_db):
    plugin, helper = _db_backed_agent_plugin(tmp_path, agent_db)

    result = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[
            {"id": "T1", "content": "One", "status": "pending", "depends_on": ["T2"]},
            {"id": "T2", "content": "Two", "status": "pending", "depends_on": ["T1"]},
        ],
    )

    assert result["success"] is False
    assert "dependency cycle" in result["error"]


@pytest.mark.asyncio
async def test_clear_plan_tasks_removes_open_plan(tmp_path, agent_db):
    plugin, helper = _db_backed_agent_plugin(tmp_path, agent_db)

    added = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[{"id": "T1", "content": "Create presentation", "status": "in_progress"}],
    )
    assert added["success"] is True

    assert plugin.clear_plan_tasks(chat_id=10, user_id=42) is True
    assert plugin.get_plan_tasks(chat_id=10, user_id=42) == []
    assert plugin.clear_plan_tasks(chat_id=10, user_id=42) is False


@pytest.mark.asyncio
async def test_clear_terminal_plan_tasks_preserves_open_plan(tmp_path, agent_db):
    plugin, helper = _db_backed_agent_plugin(tmp_path, agent_db)

    added = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[{"id": "T1", "content": "Create presentation", "status": "in_progress"}],
    )
    assert added["success"] is True

    assert plugin.clear_terminal_plan_tasks(chat_id=10, user_id=42) is False
    assert plugin.get_plan_tasks(chat_id=10, user_id=42) == [
        {"id": "T1", "content": "Create presentation", "status": "in_progress", "depends_on": []}
    ]


@pytest.mark.asyncio
async def test_clear_terminal_plan_tasks_removes_closed_plan(tmp_path, agent_db):
    plugin, helper = _db_backed_agent_plugin(tmp_path, agent_db)

    added = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[{"id": "T1", "content": "Create presentation", "status": "completed"}],
    )
    assert added["success"] is True

    assert plugin.clear_terminal_plan_tasks(chat_id=10, user_id=42) is True
    assert plugin.get_plan_tasks(chat_id=10, user_id=42) == []


@pytest.mark.asyncio
async def test_manage_plan_tasks_rejects_multiple_in_progress(tmp_path, agent_db):
    plugin, helper = _db_backed_agent_plugin(tmp_path, agent_db)

    result = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[
            {"id": "T1", "content": "Collect recipes", "status": "in_progress"},
            {"id": "T2", "content": "Create presentation", "status": "in_progress"},
        ],
    )

    assert result["success"] is False
    assert "Only one plan task may be in_progress" in result["error"]
    assert plugin.get_plan_tasks(chat_id=10) == []


@pytest.mark.asyncio
async def test_manage_plan_tasks_rejects_in_progress_when_earlier_tasks_are_open(tmp_path, agent_db):
    plugin, helper = _db_backed_agent_plugin(tmp_path, agent_db)

    added = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[
            {"id": "T1", "content": "Collect recipes", "status": "completed"},
            {"id": "T2", "content": "Generate images", "status": "pending"},
            {"id": "T3", "content": "Create presentation", "status": "pending"},
        ],
    )
    assert added["success"] is True

    result = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[{"id": "T4", "content": "Send file to user", "status": "in_progress"}],
    )

    assert result["success"] is False
    assert "earlier tasks are still open" in result["error"]
    assert [task["id"] for task in plugin.get_plan_tasks(chat_id=10)] == ["T1", "T2", "T3"]


@pytest.mark.asyncio
async def test_manage_plan_tasks_rejects_duplicate_delivery_task(tmp_path, agent_db):
    plugin, helper = _db_backed_agent_plugin(tmp_path, agent_db)

    added = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[
            {"id": "T1", "content": "Collect recipes", "status": "completed"},
            {"id": "T2", "content": "Create presentation", "status": "completed"},
            {"id": "T3", "content": "Check and deliver presentation to user", "status": "pending"},
        ],
    )
    assert added["success"] is True

    result = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[{"id": "T4", "content": "Send file to user", "status": "pending"}],
    )

    assert result["success"] is False
    assert "duplicate delivery task" in result["error"]
    assert [task["id"] for task in plugin.get_plan_tasks(chat_id=10)] == ["T1", "T2", "T3"]


@pytest.mark.asyncio
async def test_run_subagents_runs_tool_capable_workers(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = FakeLLMHelper()

    result = await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        shared_context="Shared facts",
        subagents=[
            {"id": "a1", "role": "reviewer", "task": "Check assumptions"},
            {"id": "a2", "role": "tester", "task": "Find test cases", "context": "Focus on files"},
        ],
    )

    assert result["success"] is True
    assert [item["id"] for item in result["subagents"]] == ["a1", "a2"]
    assert [item["status"] for item in result["subagents"]] == ["completed", "completed"]
    assert len(helper.completions.calls) == 4
    assert all(call["model"] == "llmgateway/high" for call in helper.completions.calls)
    assert all("max_tokens" not in call for call in helper.completions.calls)
    tool_names = {
        tool["function"]["name"]
        for call in helper.completions.calls
        for tool in call.get("tools", [])
    }
    assert "skills.list_skills" in tool_names
    assert "agent_tools.deliver_to_user" not in tool_names
    assert "agent_tools.run_subagents" not in tool_names
    assert "skills.run_skill_agent" not in tool_names
    assert [call[0] for call in helper.plugin_manager.calls] == ["skills.list_skills", "skills.list_skills"]
    assert any(
        "Shared facts" in message["content"]
        for message in helper.completions.calls[0]["messages"]
        if message.get("role") == "user"
    )


@pytest.mark.asyncio
async def test_run_subagents_applies_per_subagent_overrides(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = FakeLLMHelper()

    await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        max_rounds=4,
        subagents=[
            {"id": "a1", "role": "r", "task": "t", "temperature": 0.7, "max_rounds": 25},
        ],
    )

    assert helper.completions.calls
    assert helper.completions.calls[0]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_run_subagents_applies_per_subagent_model_override(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = FakeLLMHelper(model_name="llmgateway/high")

    await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        subagents=[
            {"id": "a1", "role": "r", "task": "t", "model": "llmgateway/light_model"},
        ],
    )

    assert helper.completions.calls
    assert helper.completions.calls[0]["model"] == "llmgateway/light_model"


@pytest.mark.asyncio
async def test_run_subagents_ignores_unsupported_model_override(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = FakeLLMHelper(model_name="llmgateway/high")

    await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        subagents=[
            {"id": "a1", "role": "r", "task": "t", "model": "openai/gpt-4o"},
        ],
    )

    assert helper.completions.calls
    assert helper.completions.calls[0]["model"] == "llmgateway/high"


@pytest.mark.asyncio
async def test_run_subagents_executes_same_round_tool_calls_in_parallel(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = FakeLLMHelper(
        completions=FakeCompletions([
            SimpleNamespace(
                id="tool_a",
                function=SimpleNamespace(name="skills.list_skills", arguments="{}"),
            ),
            SimpleNamespace(
                id="tool_b",
                function=SimpleNamespace(
                    name="skills.get_skill",
                    arguments='{"skill_name":"demo"}',
                ),
            ),
        ]),
        plugin_manager=ParallelToolPluginManager(),
    )

    result = await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        subagents=[{"id": "a1", "role": "worker", "task": "use two tools"}],
    )

    assert result["success"] is True
    assert helper.plugin_manager.overlapped is True
    assert {name for name, _args in helper.plugin_manager.calls} == {
        "skills.list_skills",
        "skills.get_skill",
    }
    tool_messages = [
        message for message in helper.completions.calls[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == ["tool_a", "tool_b"]


@pytest.mark.asyncio
async def test_run_subagents_can_schedule_background_workers(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = FakeLLMHelper(
        completions=SlowFakeCompletions(),
        model_name="llmgateway/env-main",
    )

    result = await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        background=True,
        subagents=[{"id": "reflect", "role": "skill_reflector", "task": "record reflection"}],
    )

    assert result == {
        "success": True,
        "background": True,
        "scheduled": 1,
        "subagents": ["reflect"],
    }
    tasks = list(plugin._background_subagent_tasks)
    assert len(tasks) == 1
    await asyncio.gather(*tasks)
    assert plugin._background_subagent_tasks == set()
    assert helper.completions.calls[0]["model"] == "llmgateway/env-main"
    assert [call[0] for call in helper.plugin_manager.calls] == ["skills.list_skills"]


@pytest.mark.asyncio
async def test_background_job_runs_and_delivers_to_chat(tmp_path):
    plugin = AgentToolsPlugin()
    helper = FakeBackgroundHelper()
    plugin.initialize(openai=helper, storage_root=str(tmp_path))
    bot = FakeBot()

    job = plugin._create_background_job(
        chat_id=100,
        user_id=42,
        prompt="do background work",
        reply_to_message_id=555,
        message_thread_id=None,
    )
    await plugin._run_background_job(bot, job["scope"], job["id"])

    stored = plugin.background_jobs[job["scope"]][job["id"]]
    assert stored["status"] == "completed"
    assert stored["tokens"] == 11
    assert helper.requests[0]["chat_id"] == 100
    assert helper.requests[0]["user_id"] == 42
    assert bot.messages[0]["chat_id"] == 100
    assert "Background job" in bot.messages[0]["text"]
    assert "background done" in bot.messages[0]["text"]


@pytest.mark.asyncio
async def test_run_subagents_floors_max_rounds_to_minimum(tmp_path):
    from bot.plugins.agent_tools import MIN_SUBAGENT_TOOL_ROUNDS, AgentToolsPlugin as _Plugin

    assert _Plugin._normalize_max_rounds(1) == MIN_SUBAGENT_TOOL_ROUNDS
    assert _Plugin._normalize_max_rounds(None) == MIN_SUBAGENT_TOOL_ROUNDS
    assert _Plugin._normalize_max_rounds(MIN_SUBAGENT_TOOL_ROUNDS + 5) == MIN_SUBAGENT_TOOL_ROUNDS + 5
    assert _Plugin._normalize_max_rounds("garbage") == MIN_SUBAGENT_TOOL_ROUNDS
    assert _Plugin._normalize_max_rounds(None, default=15) == 15
    assert _Plugin._normalize_max_rounds(5, default=15) == MIN_SUBAGENT_TOOL_ROUNDS
    assert _Plugin._normalize_max_rounds(20, default=15) == 20


@pytest.mark.asyncio
async def test_run_subagents_rejects_too_many_workers(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = FakeLLMHelper()

    result = await plugin.execute(
        "run_subagents",
        helper,
        subagents=[
            {"id": f"a{i}", "role": "worker", "task": "work"}
            for i in range(6)
        ],
    )

    assert result["success"] is False
    assert "At most" in result["error"]


@pytest.mark.asyncio
async def test_ask_telegram_user_resolves_from_text_answer(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    task = asyncio.create_task(
        plugin.execute(
            "ask_telegram_user",
            helper,
            chat_id=10,
            user_id=42,
            question="Pick an approach",
            options=["Minimal", "Full port"],
            timeout_seconds=1,
        )
    )

    for _ in range(10):
        if plugin.pending_by_chat:
            break
        await asyncio.sleep(0)

    assert bot.messages
    assert plugin.is_waiting_for_text(10, 42) is True

    other_user_message = FakeMessage(chat_id=10, text="Full port", user_id=99)
    await plugin.handle_text_answer(SimpleNamespace(effective_message=other_user_message), SimpleNamespace())
    assert task.done() is False

    message = FakeMessage(chat_id=10, text="Minimal", user_id=42)
    update = SimpleNamespace(effective_message=message)
    await plugin.handle_text_answer(update, SimpleNamespace())
    result = await task

    assert result["success"] is True
    assert result["answer"] == "Minimal"
    assert message.replies == ["Answer received."]
    assert plugin.pending_by_chat == {}


class FakeSkillsAwarePluginManager(FakePluginManager):
    def __init__(self):
        super().__init__()
        self._skills_plugin = SimpleNamespace(
            active_skills={"chat:10": {"pptx": {}}},
            available_skills={"pptx": {"scripts": ["build.py"]}},
        )

    def get_plugin(self, name: str):
        return self._skills_plugin if name == "skills" else None

    def get_functions_specs(self, helper, model_to_use, allowed_plugins):
        specs = super().get_functions_specs(helper, model_to_use, allowed_plugins)
        return specs + [
            {
                "type": "function",
                "function": {
                    "name": "codeinterpreter.deep_analysis",
                    "description": "run code",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]


class _DeepAnalysisCompletions(FakeCompletions):
    def __init__(self, code_prompt: str):
        super().__init__()
        self._code_prompt = code_prompt

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools") and not any(message.get("role") == "tool" for message in kwargs["messages"]):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="ci_call",
                    function=SimpleNamespace(
                        name="codeinterpreter.deep_analysis",
                        arguments=json.dumps({"code_prompt": self._code_prompt}),
                    ),
                )],
            ))])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="acknowledged", tool_calls=None),
        )])


class _DeepAnalysisHelper:
    def __init__(self, code_prompt: str):
        self.completions = _DeepAnalysisCompletions(code_prompt)
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=self.completions))
        self.plugin_manager = FakeSkillsAwarePluginManager()

    def get_current_model(self, user_id):
        return "llmgateway/high"


@pytest.mark.asyncio
async def test_subagent_routing_guard_blocks_active_skill_script_via_codeinterpreter(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = _DeepAnalysisHelper("import subprocess; subprocess.run(['python3', 'build.py'])")

    result = await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        subagents=[{"id": "a1", "role": "worker", "task": "do work"}],
    )

    assert result["success"] is True
    called = [name for name, _args in helper.plugin_manager.calls]
    assert "codeinterpreter.deep_analysis" not in called
    tool_messages = [
        message["content"]
        for message in helper.completions.calls[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert tool_messages
    assert any("skills.run_skill_script" in content for content in tool_messages)
    assert any("terminal.terminal" in content for content in tool_messages)
    assert any('"script_name": "build.py"' in content for content in tool_messages)


@pytest.mark.asyncio
async def test_subagent_routing_guard_allows_unrelated_codeinterpreter_call(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = _DeepAnalysisHelper("print(6 * 7)")

    result = await plugin.execute(
        "run_subagents",
        helper,
        chat_id=10,
        user_id=42,
        subagents=[{"id": "a1", "role": "worker", "task": "do work"}],
    )

    assert result["success"] is True
    called = [name for name, _args in helper.plugin_manager.calls]
    assert "codeinterpreter.deep_analysis" in called


@pytest.mark.asyncio
async def test_ask_telegram_user_timeout_clears_markup_and_notifies(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    result = await plugin.execute(
        "ask_telegram_user",
        helper,
        chat_id=10,
        user_id=42,
        question="Pick an approach",
        options=["Yes", "No"],
        allow_free_text=False,
        timeout_seconds=1,
    )

    assert result["success"] is False
    assert "Timed out" in result["error"]
    assert bot.markup_edits and bot.markup_edits[0]["reply_markup"] is None
    assert bot.markup_edits[0]["chat_id"] == 10
    timeout_notice = localized_text("agent_tools_question_timeout_notice", "en")
    notification = next(
        (m for m in bot.messages if timeout_notice == m.get("text")),
        None,
    )
    assert notification is not None
    assert plugin.pending_by_chat == {}


@pytest.mark.asyncio
async def test_close_clears_inline_markup_for_pending_questions(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    task = asyncio.create_task(
        plugin.execute(
            "ask_telegram_user",
            helper,
            chat_id=10,
            user_id=42,
            question="Pick an approach",
            options=["Yes", "No"],
            timeout_seconds=10,
        )
    )
    for _ in range(20):
        if plugin.pending_by_chat and any(
            pending.get("message_id") for pending in plugin.pending_questions.values()
        ):
            break
        await asyncio.sleep(0)
    assert plugin.pending_by_chat

    plugin.close()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await task

    assert bot.markup_edits, "Expected close() to schedule markup cleanup"
    assert bot.markup_edits[0]["reply_markup"] is None
    assert plugin.pending_by_chat == {}


@pytest.mark.asyncio
async def test_ask_telegram_user_requires_options(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    result = await plugin.execute(
        "ask_telegram_user",
        helper,
        chat_id=10,
        user_id=42,
        question=(
            "Как получить файл?\n"
            "- Попробовать другой способ отправки\n"
            "- Предоставить путь к файлу"
        ),
        allow_free_text=True,
        timeout_seconds=1,
    )

    assert result["success"] is False
    assert "options" in result["error"]
    assert bot.messages == []


@pytest.mark.asyncio
async def test_cancel_pending_question_clears_markup_and_resolves(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    task = asyncio.create_task(
        plugin.execute(
            "ask_telegram_user",
            helper,
            chat_id=10,
            user_id=42,
            question="Pick",
            options=["A", "B"],
            timeout_seconds=10,
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
        if plugin.pending_by_chat and any(
            pending.get("message_id") for pending in plugin.pending_questions.values()
        ):
            break
    assert plugin.pending_by_chat

    cancel_result = await plugin.execute(
        "cancel_pending_question",
        helper,
        chat_id=10,
        user_id=42,
        reason="Решили иначе",
    )
    assert cancel_result["success"] is True
    assert plugin.pending_by_chat == {}

    cancel_text = localized_text(
        "agent_tools_question_cancelled",
        "en",
    ).format(reason="Решили иначе")
    cancel_notice = next(
        (m for m in bot.messages if cancel_text == m.get("text")),
        None,
    )
    assert cancel_notice is not None
    assert any(edit.get("reply_markup") is None for edit in bot.markup_edits)

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancel_pending_question_without_pending_returns_error(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    result = await plugin.execute(
        "cancel_pending_question",
        helper,
        chat_id=10,
        user_id=42,
    )
    assert result["success"] is False
    assert "No pending question" in result["error"]


@pytest.mark.asyncio
async def test_ask_telegram_user_multi_select_force_disables_free_text(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    task = asyncio.create_task(
        plugin.execute(
            "ask_telegram_user",
            helper,
            chat_id=10,
            user_id=42,
            question="Pick all that apply",
            options=["A", "B", "C"],
            multi_select=True,
            allow_free_text=True,
            timeout_seconds=10,
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
        if plugin.pending_by_chat and any(
            pending.get("message_id") for pending in plugin.pending_questions.values()
        ):
            break
    assert plugin.pending_by_chat

    pending = next(iter(plugin.pending_questions.values()))
    assert pending["multi_select"] is True
    assert pending["allow_free_text"] is False

    markup = bot.messages[0]["reply_markup"]
    button_labels = [btn.text for row in markup.inline_keyboard for btn in row]
    assert any(label.startswith("☐ ") for label in button_labels)
    assert "Confirm" in button_labels

    plugin.close()
    await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await task


def test_subagent_internal_publish_collects_items_for_parent():
    plugin = AgentToolsPlugin()
    published: list = []

    response = plugin._handle_internal_publish(
        {"kind": "text", "value": "found something"},
        published,
    )
    assert "stored" in response
    response2 = plugin._handle_internal_publish(
        {"kind": "file", "value": "/tmp/result.csv", "caption": "csv"},
        published,
    )
    assert "stored" in response2

    assert len(published) == 2
    assert published[0] == {"kind": "text", "value": "found something"}
    assert published[1] == {"kind": "file", "value": "/tmp/result.csv", "caption": "csv"}


def test_internal_publish_outside_subagent_returns_error():
    plugin = AgentToolsPlugin()
    response = plugin._handle_internal_publish(
        {"kind": "text", "value": "hi"},
        None,
    )
    assert "only available inside a subagent" in response


def test_internal_publish_validates_kind():
    plugin = AgentToolsPlugin()
    published: list = []
    response = plugin._handle_internal_publish(
        {"kind": "weird", "value": "x"},
        published,
    )
    assert "kind must be one of" in response
    assert published == []


@pytest.mark.asyncio
async def test_deliver_to_user_returns_final_direct_result(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = SimpleNamespace()

    artifact_path = tmp_path / "report.txt"
    artifact_path.write_text("hello", encoding="utf-8")

    result = await plugin.execute(
        "deliver_to_user",
        helper,
        chat_id=10,
        user_id=42,
        text="Готово",
        artifacts=[{"file_path": str(artifact_path), "caption": "summary"}],
    )

    assert result["success"] is True
    direct_result = result["direct_result"]
    assert direct_result["kind"] == "final"
    assert direct_result["format"] == "mixed"
    assert direct_result["defer"] is False
    assert direct_result["text"] == "Готово"
    assert direct_result["artifacts"] == [
        {
            "kind": "file",
            "format": "path",
            "value": str(artifact_path),
            "file_size": len("hello"),
            "caption": "summary",
        }
    ]


@pytest.mark.asyncio
async def test_deliver_to_user_deduplicates_only_within_same_request(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = SimpleNamespace()

    first_context = RequestContext(chat_id=10, user_id=42, message_id=1, request_id="10_1")
    second_context = RequestContext(chat_id=10, user_id=42, message_id=2, request_id="10_2")

    first = await plugin.execute(
        "deliver_to_user",
        helper,
        chat_id=10,
        user_id=42,
        text="first",
        request_context=first_context,
    )
    duplicate = await plugin.execute(
        "deliver_to_user",
        helper,
        chat_id=10,
        user_id=42,
        text="duplicate",
        request_context=first_context,
    )
    second = await plugin.execute(
        "deliver_to_user",
        helper,
        chat_id=10,
        user_id=42,
        text="second",
        request_context=second_context,
    )

    assert first["success"] is True
    assert duplicate["success"] is True
    assert duplicate["skipped"] is True
    assert second["success"] is True
    assert "skipped" not in second


@pytest.mark.asyncio
async def test_deliver_to_user_requires_text_or_artifacts(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = SimpleNamespace()

    result = await plugin.execute("deliver_to_user", helper, chat_id=10, user_id=42)
    assert result["success"] is False
    assert "text or artifacts" in result["error"]


@pytest.mark.asyncio
async def test_deliver_to_user_rejects_missing_or_empty_files(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = SimpleNamespace()

    missing = await plugin.execute(
        "deliver_to_user",
        helper,
        chat_id=10,
        user_id=42,
        artifacts=[{"file_path": str(tmp_path / "missing.txt")}],
    )
    assert missing["success"] is False
    assert "does not exist" in missing["error"]

    empty_path = tmp_path / "empty.bin"
    empty_path.write_bytes(b"")
    empty = await plugin.execute(
        "deliver_to_user",
        helper,
        chat_id=10,
        user_id=42,
        artifacts=[{"file_path": str(empty_path)}],
    )
    assert empty["success"] is False
    assert "is empty" in empty["error"]


def test_deliver_to_user_blocked_for_subagents():
    from bot.plugins.agent_tools import SUBAGENT_BLOCKED_FUNCTIONS

    assert "agent_tools.deliver_to_user" in SUBAGENT_BLOCKED_FUNCTIONS


def test_question_markup_renders_selected_marks(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    markup = plugin._question_markup(
        "qid",
        ["A", "B", "C"],
        allow_free_text=False,
        multi_select=True,
        selected_indices={0, 2},
    )
    labels = [btn.text for row in markup.inline_keyboard for btn in row]
    assert labels == ["✅ A", "☐ B", "✅ C", "Confirm"]


@pytest.mark.asyncio
async def test_pending_question_persisted_and_recovered_on_startup(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    helper = SimpleNamespace(user_id=42, bot=bot)

    task = asyncio.create_task(
        plugin.execute(
            "ask_telegram_user",
            helper,
            chat_id=10,
            user_id=42,
            question="Pick",
            options=["A", "B"],
            timeout_seconds=10,
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
        if any(p.get("message_id") for p in plugin.pending_questions.values()):
            break

    pending_file = tmp_path / "agent_pending_questions.json"
    assert pending_file.exists()
    snapshot = json.loads(pending_file.read_text())
    assert len(snapshot) == 1
    assert snapshot[0]["chat_id"] == 10
    assert snapshot[0]["message_id"] == 1

    plugin.close()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await task

    bot.messages.clear()
    bot.markup_edits.clear()

    new_plugin = AgentToolsPlugin()
    new_plugin.initialize(storage_root=str(tmp_path))
    assert new_plugin._orphaned_pending and len(new_plugin._orphaned_pending) == 1

    await new_plugin.on_startup(SimpleNamespace(bot=bot))

    assert new_plugin._orphaned_pending == []
    assert any(edit.get("reply_markup") is None for edit in bot.markup_edits)
    orphaned_notice = localized_text("agent_tools_orphaned_question_inactive", "en")
    notice = next((m for m in bot.messages if orphaned_notice == m.get("text")), None)
    assert notice is not None
    assert not pending_file.exists()
