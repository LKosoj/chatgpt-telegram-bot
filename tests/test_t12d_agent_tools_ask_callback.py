"""T12d D3: handle_ask_callback markup cleanup is shared via
_clear_ask_user_markup. No direct test existed for this method before the
extraction; this covers both the single-select and multi-select confirm
branches to prove the merged helper still clears the markup and resolves
the pending question in each case.
"""
import asyncio
from types import SimpleNamespace

import pytest

from bot.plugins.agent_tools import AgentToolsPlugin


class FakeBot:
    async def send_message(self, **kwargs):
        return SimpleNamespace(message_id=1)


class FakeQuery:
    def __init__(self, data):
        self.data = data
        self.from_user = None
        self.answer_calls = []
        self.markup_edits = []

    async def answer(self, text=None, show_alert=False):
        self.answer_calls.append({"text": text, "show_alert": show_alert})

    async def edit_message_reply_markup(self, reply_markup=None):
        self.markup_edits.append(reply_markup)


async def _start_pending_question(plugin, bot, *, multi_select=False, options=None):
    helper = SimpleNamespace(user_id=42, bot=bot)
    task = asyncio.create_task(
        plugin.execute(
            "ask_telegram_user",
            helper,
            chat_id=10,
            question="Pick one",
            options=options or ["Yes", "No"],
            multi_select=multi_select,
            timeout_seconds=10,
        )
    )
    for _ in range(20):
        if plugin.pending_questions:
            break
        await asyncio.sleep(0)
    question_id = next(iter(plugin.pending_questions))
    return task, question_id


@pytest.mark.asyncio
async def test_handle_ask_callback_single_select_clears_markup(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    task, question_id = await _start_pending_question(plugin, bot)

    query = FakeQuery(f"agentask:{question_id}:0")
    update = SimpleNamespace(callback_query=query)
    await plugin.handle_ask_callback(update, SimpleNamespace())

    result = await task
    assert result == {"success": True, "answer": "Yes", "output": "User answered: Yes"}
    assert query.answer_calls[-1]["text"] == plugin.t("agent_tools_answer_received")
    assert query.markup_edits[-1] is None


@pytest.mark.asyncio
async def test_handle_ask_callback_multi_select_confirm_clears_markup(tmp_path):
    plugin = AgentToolsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    bot = FakeBot()
    task, question_id = await _start_pending_question(
        plugin, bot, multi_select=True, options=["A", "B", "C"],
    )

    # Toggle one option on first (not the merged branch), then confirm.
    toggle_query = FakeQuery(f"agentask:{question_id}:0")
    await plugin.handle_ask_callback(SimpleNamespace(callback_query=toggle_query), SimpleNamespace())

    confirm_query = FakeQuery(f"agentask:{question_id}:confirm")
    await plugin.handle_ask_callback(SimpleNamespace(callback_query=confirm_query), SimpleNamespace())

    result = await task
    assert result == {"success": True, "answer": "A", "output": "User answered: A"}
    assert confirm_query.answer_calls[-1]["text"] == plugin.t("agent_tools_answer_received")
    assert confirm_query.markup_edits[-1] is None
