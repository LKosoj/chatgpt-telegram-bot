"""Stage 4B — on_before_chat_request mutator behaviour."""
from typing import Any, Dict, List

import pytest

from bot.plugins.hindsight_memory import (
    HindsightMemoryPlugin,
    HINDSIGHT_MEMORY_MARKER,
)
from bot.plugins.hooks import BeforeChatRequestPayload


class FakeHindsight:
    enabled = True

    def __init__(self, recall_result=None):
        self.recall_calls: List[tuple] = []
        self._result = recall_result or {
            "results": [{"text": "user likes Python", "score": 0.9}]
        }

    async def recall(self, bank_id, query, **kwargs):
        self.recall_calls.append((bank_id, query, kwargs))
        return self._result


def _make_plugin(client, **cfg_overrides) -> HindsightMemoryPlugin:
    plugin = HindsightMemoryPlugin()
    cfg = {
        'hindsight_base_url': 'http://x',
        'hindsight_api_token': 't',
        'hindsight_auto_recall': True,
        **cfg_overrides,
    }
    plugin.initialize(plugin_config=cfg)
    plugin.client = client
    return plugin


def _payload(user_id=123) -> BeforeChatRequestPayload:
    return BeforeChatRequestPayload(chat_id=1, user_id=user_id, request_id=None)


async def test_mutator_recalls_and_injects_marker():
    plugin = _make_plugin(FakeHindsight())
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Hello"},
    ]
    new = await plugin.on_before_chat_request(messages, _payload())
    assert new is not None
    assert any(plugin.is_hindsight_memory_message(m) for m in new)
    assert new[0]["content"] == "system prompt"
    assert new[1]["content"].startswith(HINDSIGHT_MEMORY_MARKER)
    assert plugin.client.recall_calls[0][0] == "telegram-123"


async def test_mutator_cache_skips_recall_when_marker_present():
    plugin = _make_plugin(FakeHindsight())
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "system", "content": f"{HINDSIGHT_MEMORY_MARKER}\ncached"},
        {"role": "user", "content": "another query"},
    ]
    new = await plugin.on_before_chat_request(messages, _payload())
    assert new is None
    assert plugin.client.recall_calls == []


async def test_mutator_returns_none_on_http_failure(caplog):
    class FailingClient:
        enabled = True

        async def recall(self, *a, **kw):
            raise RuntimeError("boom")

    plugin = _make_plugin(FailingClient())
    messages = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "Q"},
    ]
    import logging
    with caplog.at_level(logging.WARNING, logger="bot.plugins.hindsight_memory"):
        new = await plugin.on_before_chat_request(messages, _payload())
    assert new is None
    assert any("Hindsight recall failed" in r.message for r in caplog.records)


async def test_mutator_noop_when_inactive():
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={})
    new = await plugin.on_before_chat_request(
        [{"role": "user", "content": "q"}], _payload(),
    )
    assert new is None


async def test_mutator_noop_when_auto_recall_disabled():
    plugin = _make_plugin(FakeHindsight(), hindsight_auto_recall=False)
    new = await plugin.on_before_chat_request(
        [{"role": "system", "content": "sp"}, {"role": "user", "content": "q"}],
        _payload(),
    )
    assert new is None
    assert plugin.client.recall_calls == []


async def test_apply_mutators_isolates_plugin_exception(tmp_path):
    from bot.plugin_manager import PluginManager

    class BrokenPlugin:
        plugin_id = "broken"
        function_prefix = "broken"

        async def on_before_chat_request(self, value, payload):
            raise RuntimeError("plugin crashed")

    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    pm = PluginManager(config={'plugins': []}, plugins_directory=str(plugin_dir))
    broken = BrokenPlugin()
    pm.plugins["broken"] = type(broken)
    pm.plugin_instances["broken"] = broken

    original = [{"role": "user", "content": "q"}]
    result = await pm.apply_mutators(
        'on_before_chat_request', _payload(), original, user_id=None,
    )
    assert result == original


async def test_mutator_noop_when_no_user_message():
    plugin = _make_plugin(FakeHindsight())
    new = await plugin.on_before_chat_request(
        [{"role": "system", "content": "sp"}], _payload(),
    )
    assert new is None
    assert plugin.client.recall_calls == []


async def test_mutator_noop_on_empty_recall():
    class EmptyClient:
        enabled = True

        async def recall(self, *a, **kw):
            return {"results": []}

    plugin = _make_plugin(EmptyClient())
    new = await plugin.on_before_chat_request(
        [{"role": "system", "content": "sp"}, {"role": "user", "content": "q"}],
        _payload(),
    )
    assert new is None
