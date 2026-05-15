from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("tiktoken")

from bot.database import Database
from bot.plugins.db_handle import DbHandle
from bot.plugins.hindsight_memory import HindsightMemoryPlugin
from bot.plugins.hooks import AssistantResponsePayload, UserMessagePayload


@pytest.fixture()
def plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "events.db"))
    Database._reset_singleton()
    db = Database()
    plugin = HindsightMemoryPlugin()
    with db.get_connection() as conn:
        for stmt in plugin.register_schema():
            conn.execute(stmt)
    plugin.initialize(plugin_config={
        "hindsight_base_url": "http://x",
        "hindsight_api_token": "t",
        "hindsight_dream_enabled": True,
    })
    plugin.client = SimpleNamespace(enabled=True)
    plugin.db_handle = DbHandle(db)
    yield plugin
    Database._reset_singleton()


async def test_observer_hooks_append_user_and_assistant_events(plugin):
    await plugin.on_user_message(UserMessagePayload(
        chat_id=10, user_id=42, request_id="r1", text="remember this",
        has_image=False, has_voice=False, is_command=False, ts=1.0,
    ))
    await plugin.on_assistant_response(AssistantResponsePayload(
        chat_id=10, user_id=42, request_id="r1", text="ok",
        tokens=12, model="m", ts=2.0,
    ))

    rows = await plugin.db_handle.fetch_all(
        "SELECT event_type, role, text_preview, payload_json "
        "FROM hindsight_memory_events ORDER BY id"
    )

    assert [row["event_type"] for row in rows] == ["user_message", "assistant_response"]
    assert rows[0]["role"] == "user"
    assert rows[1]["role"] == "assistant"
    assert "remember this" in rows[0]["text_preview"]
    assert json.loads(rows[1]["payload_json"])["model"] == "m"


async def test_event_journal_redacts_sensitive_text_and_payload_keys(plugin):
    await plugin._append_memory_event(
        "manual",
        user_id=42,
        chat_id=10,
        text="GITHUB_TOKEN=abc123 and Bearer secret-token",
        payload={"api_key": "abc123", "safe": "value"},
    )

    row = await plugin.db_handle.fetch_one(
        "SELECT text_preview, payload_json, redaction_count FROM hindsight_memory_events"
    )

    assert "abc123" not in row["text_preview"]
    assert "secret-token" not in row["text_preview"]
    payload = json.loads(row["payload_json"])
    assert payload["api_key"] == "[REDACTED]"
    assert payload["safe"] == "value"
    assert row["redaction_count"] >= 2


async def test_event_journal_redacts_natural_language_secrets(plugin):
    await plugin._append_memory_event(
        "manual",
        user_id=42,
        chat_id=10,
        text="my password is hunter2. keep using the existing workflow.",
        payload={},
    )

    row = await plugin.db_handle.fetch_one(
        "SELECT text_preview, redaction_count FROM hindsight_memory_events"
    )

    assert "hunter2" not in row["text_preview"]
    assert "password" not in row["text_preview"].lower()
    assert "existing workflow" in row["text_preview"]
    assert row["redaction_count"] >= 1


async def test_event_journal_redacts_russian_secret_phrasing(plugin):
    await plugin._append_memory_event(
        "manual",
        user_id=42,
        chat_id=10,
        text="мой пароль hunter2. рабочий процесс остается прежним.",
        payload={"токен": "abc123"},
    )

    row = await plugin.db_handle.fetch_one(
        "SELECT text_preview, payload_json, redaction_count FROM hindsight_memory_events"
    )

    assert "hunter2" not in row["text_preview"]
    assert "пароль" not in row["text_preview"].lower()
    assert "рабочий процесс" in row["text_preview"]
    payload = json.loads(row["payload_json"])
    assert payload["токен"] == "[REDACTED]"
    assert row["redaction_count"] >= 2


async def test_event_journal_redacts_multiline_secret_values(plugin):
    await plugin._append_memory_event(
        "manual",
        user_id=42,
        chat_id=10,
        text="my password\nhunter2\nkeep this workflow.",
        payload={},
    )
    await plugin._append_memory_event(
        "manual",
        user_id=42,
        chat_id=10,
        text="мой пароль\nsekret123\nрабочий процесс прежний.",
        payload={},
    )

    rows = await plugin.db_handle.fetch_all(
        "SELECT text_preview FROM hindsight_memory_events ORDER BY id"
    )

    assert "hunter2" not in rows[0]["text_preview"]
    assert "password" not in rows[0]["text_preview"].lower()
    assert "workflow" in rows[0]["text_preview"]
    assert "sekret123" not in rows[1]["text_preview"]
    assert "пароль" not in rows[1]["text_preview"].lower()
    assert "рабочий процесс" in rows[1]["text_preview"]
