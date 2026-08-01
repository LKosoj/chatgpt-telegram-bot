"""Stage 4B — on_before_chat_request mutator behaviour."""
import asyncio
import json
import logging
from types import SimpleNamespace
from typing import List


from bot.plugins.hindsight_memory import (
    HINDSIGHT_DYNAMIC_MEMORY_MARKER,
    HindsightMemoryPlugin,
    HINDSIGHT_MEMORY_MARKER,
)
from bot.plugins.hooks import BeforeChatRequestPayload
from bot.session_logger import clear_trace, set_trace


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


class FakeGateOpenAI:
    """Fake OpenAIHelper stand-in for retrieval-gate ``chat_completion`` calls."""

    def __init__(self, *, content=None, raise_exc=None, delay=None):
        self.config = {"light_model": "fake-light"}
        self.content = content
        self.raise_exc = raise_exc
        self.delay = delay
        self.calls: List[dict] = []
        self.session_logger = None

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raise_exc:
            raise self.raise_exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeSessionLogger:
    def __init__(self):
        self.events: List[dict] = []

    def record(self, event):
        self.events.append(event)


def _make_plugin(client, openai=None, **cfg_overrides) -> HindsightMemoryPlugin:
    plugin = HindsightMemoryPlugin()
    cfg = {
        'hindsight_base_url': 'http://x',
        'hindsight_api_token': 't',
        'hindsight_auto_recall': True,
        **cfg_overrides,
    }
    plugin.initialize(openai=openai, plugin_config=cfg)
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


async def test_dynamic_recall_injects_ephemeral_marker_when_baseline_exists():
    plugin = _make_plugin(FakeHindsight(), hindsight_dynamic_recall=True)
    messages = [
        {"role": "system", "content": "sp"},
        {"role": "system", "content": f"{HINDSIGHT_MEMORY_MARKER}\nbaseline"},
        {"role": "user", "content": "new query"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is not None
    assert any(
        isinstance(message.get("content"), str)
        and message["content"].startswith(HINDSIGHT_DYNAMIC_MEMORY_MARKER)
        for message in new
    )
    assert not any(
        plugin.is_hindsight_memory_message(message)
        and message["content"].startswith(HINDSIGHT_DYNAMIC_MEMORY_MARKER)
        for message in new
    )
    assert plugin.client.recall_calls[0][1] == "new query"


async def test_dynamic_recall_skips_non_persistent_retry_payload():
    plugin = _make_plugin(FakeHindsight(), hindsight_dynamic_recall=True)
    payload = BeforeChatRequestPayload(
        chat_id=1, user_id=123, request_id=None, allow_dynamic_recall=False,
    )
    messages = [
        {"role": "system", "content": f"{HINDSIGHT_MEMORY_MARKER}\nbaseline"},
        {"role": "user", "content": "retry query"},
    ]

    new = await plugin.on_before_chat_request(messages, payload)

    assert new is None
    assert plugin.client.recall_calls == []


# --- A1: retrieval gate ------------------------------------------------------

async def test_retrieval_gate_enabled_by_default_consults_gate():
    """Feature is on by default: the gate is asked, and a 'retrieve' verdict recalls as before."""
    gate_openai = FakeGateOpenAI(content=json.dumps({"retrieve": True, "query": "q", "reason": "asks about project"}))
    plugin = _make_plugin(FakeHindsight(), openai=gate_openai)
    messages = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "Hello"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is not None
    assert any(plugin.is_hindsight_memory_message(m) for m in new)
    assert len(gate_openai.calls) == 1


async def test_retrieval_gate_explicitly_disabled_skips_gate_call():
    """The off switch still works: zero chat_completion calls, recall runs unconditionally."""
    gate_openai = FakeGateOpenAI(content=json.dumps({"retrieve": False, "query": "q", "reason": "n/a"}))
    plugin = _make_plugin(FakeHindsight(), openai=gate_openai, hindsight_retrieval_gate_enabled=False)
    messages = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "Hello"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is not None
    assert any(plugin.is_hindsight_memory_message(m) for m in new)
    assert gate_openai.calls == []


async def test_retrieval_gate_skip_blocks_dynamic_recall():
    gate_openai = FakeGateOpenAI(content=json.dumps({"retrieve": False, "query": "q", "reason": "small talk"}))
    plugin = _make_plugin(
        FakeHindsight(), openai=gate_openai,
        hindsight_dynamic_recall=True, hindsight_retrieval_gate_enabled=True,
    )
    messages = [
        {"role": "system", "content": f"{HINDSIGHT_MEMORY_MARKER}\nbaseline"},
        {"role": "user", "content": "thanks"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is None
    assert plugin.client.recall_calls == []
    assert len(gate_openai.calls) == 1


async def test_retrieval_gate_retrieve_allows_dynamic_recall():
    gate_openai = FakeGateOpenAI(content=json.dumps({"retrieve": True, "query": "q", "reason": "asks about project"}))
    plugin = _make_plugin(
        FakeHindsight(), openai=gate_openai,
        hindsight_dynamic_recall=True, hindsight_retrieval_gate_enabled=True,
    )
    messages = [
        {"role": "system", "content": f"{HINDSIGHT_MEMORY_MARKER}\nbaseline"},
        {"role": "user", "content": "what did I say about my project"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is not None
    assert plugin.client.recall_calls
    assert any(
        isinstance(m.get("content"), str) and m["content"].startswith(HINDSIGHT_DYNAMIC_MEMORY_MARKER)
        for m in new
    )


async def test_retrieval_gate_skip_blocks_baseline_recall():
    gate_openai = FakeGateOpenAI(content=json.dumps({"retrieve": False, "query": "q", "reason": "greeting"}))
    plugin = _make_plugin(FakeHindsight(), openai=gate_openai, hindsight_retrieval_gate_enabled=True)
    messages = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "hi"},
    ]

    new = await plugin.on_before_chat_request(messages, _payload())

    assert new is None
    assert plugin.client.recall_calls == []
    assert len(gate_openai.calls) == 1


async def test_retrieval_gate_exception_fails_open(caplog):
    gate_openai = FakeGateOpenAI(raise_exc=RuntimeError("boom"))
    plugin = _make_plugin(FakeHindsight(), openai=gate_openai, hindsight_retrieval_gate_enabled=True)
    messages = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "hi"},
    ]

    with caplog.at_level(logging.WARNING, logger="bot.plugins.hindsight_memory"):
        new = await plugin.on_before_chat_request(messages, _payload())

    assert new is not None
    assert plugin.client.recall_calls
    assert any("Hindsight retrieval gate failed" in r.message for r in caplog.records)


async def test_retrieval_gate_invalid_json_fails_open(caplog):
    gate_openai = FakeGateOpenAI(content="not json")
    plugin = _make_plugin(FakeHindsight(), openai=gate_openai, hindsight_retrieval_gate_enabled=True)
    messages = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "hi"},
    ]

    with caplog.at_level(logging.WARNING, logger="bot.plugins.hindsight_memory"):
        new = await plugin.on_before_chat_request(messages, _payload())

    assert new is not None
    assert plugin.client.recall_calls
    assert any("Hindsight retrieval gate failed" in r.message for r in caplog.records)


async def test_retrieval_gate_timeout_fails_open(caplog):
    gate_openai = FakeGateOpenAI(content=json.dumps({"retrieve": False}), delay=0.2)
    plugin = _make_plugin(
        FakeHindsight(), openai=gate_openai,
        hindsight_retrieval_gate_enabled=True,
        hindsight_retrieval_gate_timeout_seconds=0.01,
    )
    messages = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "hi"},
    ]

    with caplog.at_level(logging.WARNING, logger="bot.plugins.hindsight_memory"):
        new = await plugin.on_before_chat_request(messages, _payload())

    assert new is not None
    assert plugin.client.recall_calls
    assert any("Hindsight retrieval gate failed" in r.message for r in caplog.records)


async def test_retrieval_gate_records_session_log_event():
    gate_openai = FakeGateOpenAI(
        content=json.dumps({"retrieve": True, "query": "q", "reason": "asks about memory"})
    )
    gate_openai.session_logger = FakeSessionLogger()
    plugin = _make_plugin(FakeHindsight(), openai=gate_openai, hindsight_retrieval_gate_enabled=True)
    messages = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "what do you remember about me"},
    ]

    token = set_trace(123, "sess-1", "turn-1")
    try:
        await plugin.on_before_chat_request(messages, _payload())
    finally:
        clear_trace(token)

    assert len(gate_openai.session_logger.events) == 1
    event = gate_openai.session_logger.events[0]
    assert event["decision"] == "retrieve"
    assert "reason" in event
    assert "duration_ms" in event
