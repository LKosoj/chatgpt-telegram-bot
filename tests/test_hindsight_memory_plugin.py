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
    enabled = True

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


def _make_plugin_with_fake_client():
    """Build an initialized plugin with a FakeHindsightClient swapped in."""
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={
        'hindsight_base_url': 'http://x',
        'hindsight_api_token': 't',
    })
    plugin.client = FakeHindsightClient()
    return plugin


@pytest.mark.asyncio
async def test_hindsight_memory_status_export_and_clear():
    plugin = _make_plugin_with_fake_client()
    helper = SimpleNamespace()
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
    assert plugin.client.cleared == ["telegram-42"]


@pytest.mark.asyncio
async def test_hindsight_memory_status_falls_back_to_list_count():
    plugin = _make_plugin_with_fake_client()
    plugin.client.stats_payload = {"documents": 1}
    plugin.client.memories_payload = {
        "items": [
            {"id": "m1", "text": "Remember this"},
            {"id": "m2", "text": "Remember that"},
        ]
    }
    helper = SimpleNamespace()

    status = await plugin._memory_status_text(helper, 42)

    assert "Memories: `2`" in status


@pytest.mark.asyncio
async def test_hindsight_memory_search_uses_recall():
    plugin = _make_plugin_with_fake_client()
    helper = SimpleNamespace()
    message = SimpleNamespace(reply_text=AsyncMock())

    await plugin._send_memory_search(message, helper, 42, "project")

    message.reply_text.assert_awaited_once()
    assert "match project" in message.reply_text.await_args.args[0]
