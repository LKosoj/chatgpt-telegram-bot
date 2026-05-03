import asyncio
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
    assert names == {"agent_tools.manage_plan_tasks", "agent_tools.ask_telegram_user"}

    commands = pm.get_plugin_commands()
    assert any(command.get("callback_pattern") == "^agentask:" for command in commands)
    assert pm.get_message_handlers()


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
