"""Stage 1 — observer-hook migration for conversation_analytics."""
import asyncio
import json
import logging
from pathlib import Path

import pytest

from bot.plugin_manager import PluginManager
from bot.plugins.conversation_analytics import ConversationAnalyticsPlugin
from bot.plugins.hooks import AssistantResponsePayload, UserMessagePayload


def _make_plugin(tmp_path: Path) -> ConversationAnalyticsPlugin:
    plugin = ConversationAnalyticsPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    return plugin


def _make_pm(tmp_path: Path, plugin: ConversationAnalyticsPlugin) -> PluginManager:
    plugin_dir = tmp_path / "p"
    plugin_dir.mkdir(exist_ok=True)
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    # Bypass normal discovery: dispatcher's `_active_plugin_instances` resolves the
    # instance via `get_plugin`, which would re-run `_call_initialize(openai=None, ...)`.
    # `ConversationAnalyticsPlugin.initialize` only reacts to `storage_root`, so a
    # second call with `storage_root=None` is a no-op and the pre-seeded state stands.
    pm.plugins["conversation_analytics"] = ConversationAnalyticsPlugin
    pm.plugin_instances["conversation_analytics"] = plugin
    return pm


async def test_on_user_message_writes_to_stats(tmp_path):
    plugin = _make_plugin(tmp_path)
    await plugin.on_user_message(
        UserMessagePayload(
            chat_id=1, user_id=2, request_id="r", text="hi",
            has_image=False, has_voice=False, is_command=False, ts=0.0,
        )
    )
    stats_file = Path(plugin.analytics_file)
    assert stats_file.exists()
    data = json.loads(stats_file.read_text())
    # find the chat_id key (str-coerced by update_stats)
    chat_key = next(iter(data.keys()))
    assert chat_key == "1"
    messages = data["1"]["messages"]
    assert any(m.get("text") == "hi" for m in messages)


async def test_on_assistant_response_writes_to_stats(tmp_path):
    plugin = _make_plugin(tmp_path)
    await plugin.on_assistant_response(
        AssistantResponsePayload(
            chat_id=1, user_id=2, request_id="r", text="response",
            tokens=5, model="gpt", ts=0.0,
        )
    )
    data = json.loads(Path(plugin.analytics_file).read_text())
    messages = data["1"]["messages"]
    assert any(m.get("text") == "response" for m in messages)


async def test_dispatch_observe_routes_to_plugin(tmp_path):
    plugin = _make_plugin(tmp_path)
    pm = _make_pm(tmp_path, plugin)
    await pm.dispatch_observe(
        "on_assistant_response",
        AssistantResponsePayload(
            chat_id=99, user_id=7, request_id=None, text="via dispatcher",
            tokens=2, model="m", ts=0.0,
        ),
    )
    data = json.loads(Path(plugin.analytics_file).read_text())
    assert any(m.get("text") == "via dispatcher" for m in data["99"]["messages"])


async def test_update_stats_exception_does_not_block_dispatcher(tmp_path, caplog):
    plugin = _make_plugin(tmp_path)

    def boom(*_a, **_kw):
        raise RuntimeError("boom")
    plugin.update_stats = boom

    pm = _make_pm(tmp_path, plugin)
    payload = AssistantResponsePayload(
        chat_id=1, user_id=2, request_id="r", text="hi",
        tokens=3, model="m", ts=0.0,
    )
    with caplog.at_level(logging.ERROR, logger="bot.plugin_manager"):
        # Must not raise — that is the invariant.
        await pm.dispatch_observe("on_assistant_response", payload)

    matching = [r for r in caplog.records if r.getMessage() == "plugin_hook_error"]
    assert matching, "expected at least one plugin_hook_error log"
    assert any(
        getattr(r, "plugin_id", None) == "conversation_analytics"
        and getattr(r, "event", None) == "on_assistant_response"
        for r in matching
    )


async def test_is_command_flag_propagated(tmp_path):
    plugin = _make_plugin(tmp_path)
    await plugin.on_user_message(
        UserMessagePayload(
            chat_id=1, user_id=2, request_id=None, text="/start",
            has_image=False, has_voice=False, is_command=True, ts=0.0,
        )
    )
    data = json.loads(Path(plugin.analytics_file).read_text())
    user_msg = next(m for m in data["1"]["messages"] if m.get("role") == "user")
    assert user_msg["is_command"] is True


async def test_concurrent_writes_produce_valid_json(tmp_path):
    """Concurrent on_user_message/on_assistant_response must not corrupt JSON."""
    plugin = _make_plugin(tmp_path)

    user_payloads = [
        UserMessagePayload(
            chat_id=1, user_id=i, request_id=f"u{i}", text=f"msg {i}",
            has_image=False, has_voice=False, is_command=False, ts=float(i),
        )
        for i in range(20)
    ]
    assistant_payloads = [
        AssistantResponsePayload(
            chat_id=1, user_id=i, request_id=f"a{i}", text=f"reply {i}",
            tokens=i, model="m", ts=float(i),
        )
        for i in range(20)
    ]

    await asyncio.gather(
        *[plugin.on_user_message(p) for p in user_payloads],
        *[plugin.on_assistant_response(p) for p in assistant_payloads],
    )

    stats_file = Path(plugin.analytics_file)
    assert stats_file.exists(), "stats file must exist after concurrent writes"

    raw = stats_file.read_text(encoding="utf-8")
    data = json.loads(raw)  # raises if JSON is corrupted

    assert "1" in data, "chat_id '1' must be present"
    messages = data["1"]["messages"]
    assert len(messages) > 0, "messages list must be non-empty"


async def test_concurrent_writes_no_tmp_leftover(tmp_path):
    """After concurrent writes, no .tmp file must remain on disk."""
    plugin = _make_plugin(tmp_path)

    payloads = [
        UserMessagePayload(
            chat_id=2, user_id=i, request_id=f"t{i}", text=f"text {i}",
            has_image=False, has_voice=False, is_command=False, ts=float(i),
        )
        for i in range(15)
    ]

    await asyncio.gather(*[plugin.on_user_message(p) for p in payloads])

    tmp_file = Path(plugin.analytics_file + ".tmp")
    assert not tmp_file.exists(), ".tmp file must not remain after successful writes"
