import importlib.util
import importlib.machinery
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.plugins.hindsight_memory import HindsightMemoryPlugin


class FakeHindsightClient:
    def __init__(self):
        self.cleared = []
        self.stats_payload = {"memory_count": 3}
        self.memories_payload = {"items": [{"id": "m1", "text": "Remember this"}]}

    async def stats(self, bank_id):
        return {**self.stats_payload, "bank_id": bank_id}

    async def list_memories(self, bank_id, **kwargs):
        return {**self.memories_payload, "bank_id": bank_id}

    async def clear_bank(self, bank_id):
        self.cleared.append(bank_id)
        return {"ok": True}

    async def recall(self, bank_id, query, **kwargs):
        return {"results": [{"text": f"match {query}", "type": "world"}], "bank_id": bank_id}


class FakeHelper:
    def __init__(self):
        self.hindsight_client = FakeHindsightClient()
        self.config = {"hindsight_recall_budget": "mid", "hindsight_recall_max_tokens": 4096}

    def is_hindsight_enabled(self):
        return True

    def get_hindsight_bank_id(self, user_id):
        return f"telegram-{user_id}"

    def _hindsight_memory_types(self):
        return ["world", "experience"]


@pytest.mark.asyncio
async def test_hindsight_memory_status_export_and_clear():
    plugin = HindsightMemoryPlugin()
    helper = FakeHelper()
    plugin.initialize(openai=helper)
    message = SimpleNamespace(
        reply_text=AsyncMock(),
        reply_document=AsyncMock(),
    )

    status = await plugin._memory_status_text(helper, 42)
    await plugin._send_memory_export(message, helper, 42)
    await plugin._clear_memory(helper, 42)

    assert "Hindsight memory is enabled" in status
    assert "Memories: `3`" in status
    message.reply_document.assert_awaited_once()
    assert helper.hindsight_client.cleared == ["telegram-42"]


@pytest.mark.asyncio
async def test_hindsight_memory_status_falls_back_to_list_count():
    plugin = HindsightMemoryPlugin()
    helper = FakeHelper()
    helper.hindsight_client.stats_payload = {"documents": 1}
    helper.hindsight_client.memories_payload = {
        "items": [
            {"id": "m1", "text": "Remember this"},
            {"id": "m2", "text": "Remember that"},
        ]
    }
    plugin.initialize(openai=helper)

    status = await plugin._memory_status_text(helper, 42)

    assert "Memories: `2`" in status


@pytest.mark.asyncio
async def test_hindsight_memory_search_uses_recall():
    plugin = HindsightMemoryPlugin()
    helper = FakeHelper()
    plugin.initialize(openai=helper)
    message = SimpleNamespace(reply_text=AsyncMock())

    await plugin._send_memory_search(message, helper, 42, "project")

    message.reply_text.assert_awaited_once()
    assert "match project" in message.reply_text.await_args.args[0]
