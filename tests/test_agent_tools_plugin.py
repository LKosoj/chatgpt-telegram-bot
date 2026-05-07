import asyncio
import json
from types import SimpleNamespace

import pytest

from bot.plugin_manager import PluginManager
from bot.plugins.agent_tools import AgentToolsPlugin


class FakeBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.messages))


class FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools") and not any(message.get("role") == "tool" for message in kwargs["messages"]):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="tool_1",
                                    function=SimpleNamespace(
                                        name="skills.list_skills",
                                        arguments="{}",
                                    ),
                                )
                            ],
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
                    "name": "skills.publish_result",
                    "description": "publish final result",
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
        ]

    async def call_function(self, function_name, helper, arguments, request_context=None):
        self.calls.append((function_name, json.loads(arguments)))
        return json.dumps({"success": True, "skills": ["demo"]}, ensure_ascii=False)


class FakeLLMHelper:
    def __init__(self):
        self.completions = FakeCompletions()
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=self.completions))
        self.plugin_manager = FakePluginManager()

    def get_current_model(self, user_id):
        return "llmgateway/high"


class FakeMessage:
    def __init__(self, chat_id: int, text: str, user_id: int = 42):
        self.chat = SimpleNamespace(id=chat_id)
        self.from_user = SimpleNamespace(id=user_id)
        self.text = text
        self.replies = []

    async def reply_text(self, text: str):
        self.replies.append(text)


def test_agent_tools_registers_specs_and_handlers():
    pm = PluginManager(config={"plugins": ["agent_tools"]})

    specs = pm.get_functions_specs(helper=None, model_to_use="llmgateway/high", allowed_plugins=["agent_tools"])
    names = {spec["function"]["name"] for spec in specs}
    assert names == {
        "agent_tools.manage_plan_tasks",
        "agent_tools.ask_telegram_user",
        "agent_tools.run_subagents",
    }
    ask_spec = next(spec["function"] for spec in specs if spec["function"]["name"] == "agent_tools.ask_telegram_user")
    assert ask_spec["parameters"]["required"] == ["question", "options"]
    assert ask_spec["parameters"]["properties"]["options"]["minItems"] == 1

    commands = pm.get_plugin_commands()
    assert any(command.get("callback_pattern") == "^agentask:" for command in commands)
    assert pm.get_message_handlers()


def test_agent_tools_preserves_full_option_text():
    plugin = AgentToolsPlugin()
    long_option = "A" * 120

    assert plugin._normalize_options([long_option]) == [long_option]


@pytest.mark.asyncio
async def test_manage_plan_tasks_tracks_progress(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    helper = SimpleNamespace(user_id=42)

    added = await plugin.execute(
        "manage_plan_tasks",
        helper,
        chat_id=10,
        action="add",
        tasks=[
            {"id": "T1", "content": "Inspect implementation", "status": "completed"},
            {"id": "T2", "content": "Add Telegram ask tool", "status": "in_progress"},
        ],
    )

    assert added["success"] is True
    assert added["plan_tasks"]["progress"] == {"total": 2, "closed": 1, "open": 1}

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
    assert "skills.publish_result" not in tool_names
    assert "agent_tools.run_subagents" not in tool_names
    assert [call[0] for call in helper.plugin_manager.calls] == ["skills.list_skills", "skills.list_skills"]
    assert any(
        "Shared facts" in message["content"]
        for message in helper.completions.calls[0]["messages"]
        if message.get("role") == "user"
    )


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
