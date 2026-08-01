"""A2 — /memory report: human-readable Markdown mirror of Hindsight memory."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("tiktoken")

from bot.database import Database
from bot.plugins.db_handle import DbHandle
from bot.plugins.hindsight_memory import HindsightMemoryPlugin
import bot.plugins.hindsight_memory as hindsight_memory_module


class FakeHindsightClient:
    enabled = True

    def __init__(self, memories_payload=None, list_memories_exc=None):
        self.memories_payload = memories_payload if memories_payload is not None else {"items": []}
        self.list_memories_exc = list_memories_exc

    async def list_memories(self, bank_id, **kwargs):
        if self.list_memories_exc:
            raise self.list_memories_exc
        return dict(self.memories_payload)


@pytest.fixture()
def plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "report.db"))
    Database._reset_singleton()
    db = Database()
    plugin = HindsightMemoryPlugin()
    with db.get_connection() as conn:
        for stmt in plugin.register_schema():
            conn.execute(stmt)
    plugin.initialize(plugin_config={
        "hindsight_base_url": "http://x",
        "hindsight_api_token": "t",
    })
    plugin.client = FakeHindsightClient()
    plugin.db_handle = DbHandle(db)
    yield plugin
    Database._reset_singleton()


async def _insert_approved_document(
    plugin,
    user_id=42,
    path="profile/preferences.md",
    kind="profile",
    content="User prefers concise answers.",
    version=1,
    status="approved",
):
    await plugin.db_handle.execute(
        '''
        INSERT INTO hindsight_memory_documents
        (user_id, path, kind, status, content, content_hash, version, approved_at)
        VALUES (?, ?, ?, ?, ?, 'hash', ?, CURRENT_TIMESTAMP)
        ''',
        (user_id, path, kind, status, content, version),
    )


def _fake_update(user_id=42):
    message = SimpleNamespace(
        is_topic_message=False,
        message_thread_id=None,
        message_id=123,
        reply_text=AsyncMock(),
    )
    return SimpleNamespace(
        effective_message=message,
        effective_chat=SimpleNamespace(type="private"),
        effective_user=SimpleNamespace(id=user_id),
        message=None,
        callback_query=None,
    )


# --- markdown rendering -------------------------------------------------------

async def test_report_groups_by_kind_with_version_and_date(plugin):
    await _insert_approved_document(plugin, path="tools/pytest.md", kind="tool", content="Use pytest -q.")
    await _insert_approved_document(
        plugin, path="profile/preferences.md", kind="profile", content="Likes concise answers.",
    )

    text = await plugin._render_memory_report_markdown(42)

    assert "## Profile" in text
    assert "## Tools" in text
    # Fixed display order: profile before tool, regardless of insertion order.
    assert text.index("## Profile") < text.index("## Tools")
    assert "### profile/preferences.md (v1," in text
    assert "### tools/pytest.md (v1," in text


async def test_report_shows_only_latest_version_per_path(plugin):
    await _insert_approved_document(
        plugin, path="profile/preferences.md", kind="profile", content="Old content", version=1,
    )
    await _insert_approved_document(
        plugin, path="profile/preferences.md", kind="profile", content="New content", version=2,
    )

    text = await plugin._render_memory_report_markdown(42)

    assert "New content" in text
    assert "Old content" not in text
    assert text.count("### profile/preferences.md") == 1


# --- PII second layer ----------------------------------------------------------

async def test_report_withholds_sensitive_document_content_inserted_directly(plugin, caplog):
    # Bypasses the write-time filter entirely: the row is inserted straight into
    # the DB, the way a bug or a manual DB edit could.
    await _insert_approved_document(plugin, path="tools/secrets.md", kind="tool", content="password: hunter2")

    with caplog.at_level(logging.WARNING, logger="bot.plugins.hindsight_memory"):
        text = await plugin._render_memory_report_markdown(42)

    assert "hunter2" not in text
    assert "[withheld: looked like sensitive content]" in text
    assert any("withheld sensitive content" in r.message for r in caplog.records)


async def test_report_withholds_sensitive_list_memories_item(plugin, caplog):
    plugin.client.memories_payload = {"items": [{"id": "m1", "text": "api_key: sk-abc123"}]}

    with caplog.at_level(logging.WARNING, logger="bot.plugins.hindsight_memory"):
        text = await plugin._render_memory_report_markdown(42)

    assert "sk-abc123" not in text
    assert "[withheld: looked like sensitive content]" in text


async def test_report_handles_list_memories_failure(plugin):
    plugin.client.list_memories_exc = RuntimeError("boom")

    text = await plugin._render_memory_report_markdown(42)

    assert "Individual memories are currently unavailable." in text


# --- delivery: short vs. long path ---------------------------------------------

async def test_send_memory_report_short_path_uses_reply_text(plugin, monkeypatch):
    monkeypatch.setattr(hindsight_memory_module, "should_send_text_as_file", lambda text, chunks=None: False)
    monkeypatch.setattr(
        hindsight_memory_module, "render_markdown_message_entities", lambda text: [(text, [])],
    )
    send_file_mock = AsyncMock()
    monkeypatch.setattr(hindsight_memory_module, "send_long_response_as_file", send_file_mock)

    update = _fake_update()
    await plugin._send_memory_report(update, 42)

    update.effective_message.reply_text.assert_awaited_once()
    send_file_mock.assert_not_awaited()


async def test_send_memory_report_long_path_uses_file_and_sets_enable_quoting(plugin, monkeypatch):
    monkeypatch.setattr(hindsight_memory_module, "should_send_text_as_file", lambda text, chunks=None: True)
    send_file_mock = AsyncMock()
    monkeypatch.setattr(hindsight_memory_module, "send_long_response_as_file", send_file_mock)

    update = _fake_update()
    await plugin._send_memory_report(update, 42)

    send_file_mock.assert_awaited_once()
    call_args = send_file_mock.await_args
    config_arg = call_args.args[0]
    assert config_arg["enable_quoting"] is True


# --- dispatch smoke -------------------------------------------------------------

async def test_handle_memory_command_dispatches_report_action(plugin, monkeypatch):
    monkeypatch.setattr(hindsight_memory_module, "message_text", lambda message: "report")
    sent = AsyncMock()
    monkeypatch.setattr(plugin, "_send_memory_report", sent)

    update = _fake_update()
    await plugin.handle_memory_command(update, SimpleNamespace())

    sent.assert_awaited_once_with(update, 42)


async def test_handle_memory_callback_dispatches_report_action(plugin, monkeypatch):
    sent = AsyncMock()
    monkeypatch.setattr(plugin, "_send_memory_report", sent)

    query = SimpleNamespace(data="memory:report", answer=AsyncMock(), message=SimpleNamespace())
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=42),
    )
    await plugin.handle_memory_callback(update, SimpleNamespace())

    sent.assert_awaited_once_with(update, 42)
