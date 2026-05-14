import os
import sqlite3
import threading
from pathlib import Path

import pytest

from bot.database import Database


class DummyOpenAI:
    def __init__(self):
        self.config = {"model": "llmgateway/high"}

    async def generate_session_name(self, *args, **kwargs):
        return ("ShortName", None)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._reset_singleton()
    return Database()


def test_message_count_saved(db):
    context = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u2"},
        ]
    }
    db.save_conversation_context(1, context, "HTML", 0.8, 80, openai_helper=DummyOpenAI())
    sessions = db.list_user_sessions(1, is_active=1)
    assert sessions[0]["message_count"] == 2


@pytest.mark.asyncio
async def test_session_name_uses_text_from_multimodal_content(db):
    context = {
        "messages": [
            {"role": "system", "content": "s"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "что на этой картинке изображено подробно?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                ],
            },
        ]
    }

    await db.save_conversation_context_async(
        1,
        context,
        "HTML",
        0.8,
        80,
        openai_helper=DummyOpenAI(),
    )

    sessions = db.list_user_sessions(1, is_active=1)
    assert sessions[0]["session_name"] == "ShortName"


def test_max_sessions_enforced(db):
    helper = DummyOpenAI()
    for _ in range(6):
        db.create_session(1, max_sessions=3, openai_helper=helper)
    sessions = db.list_user_sessions(1)
    assert len(sessions) <= 3


def test_save_context_with_missing_explicit_session_includes_model(db):
    context = {"messages": [{"role": "user", "content": "hello"}]}

    db.save_conversation_context(
        1,
        context,
        "HTML",
        0.8,
        80,
        session_id="missing-session",
        openai_helper=DummyOpenAI(),
    )

    session = db.get_session_details(1, "missing-session")
    sessions = db.list_user_sessions(1)
    assert session is not None
    assert sessions[0]["model"] == "llmgateway/high"


def test_save_context_with_missing_explicit_session_deactivates_previous(db):
    helper = DummyOpenAI()
    first = db.create_session(1, openai_helper=helper)
    assert db.get_active_session_id(1) == first

    db.save_conversation_context(
        1,
        {"messages": [{"role": "user", "content": "new explicit"}]},
        "HTML",
        0.8,
        80,
        session_id="missing-session",
        openai_helper=helper,
    )

    sessions = db.list_user_sessions(1, is_active=1)
    assert [session["session_id"] for session in sessions] == ["missing-session"]


def test_active_session_unique_index_rejects_duplicate_active_rows(db):
    with db.get_connection() as conn:
        conn.execute('''
            INSERT INTO conversation_context
            (user_id, session_id, context, model, parse_mode, temperature, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        ''', (1, "s1", '{"messages": []}', "llmgateway/high", "HTML", 0.8))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('''
                INSERT INTO conversation_context
                (user_id, session_id, context, model, parse_mode, temperature, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            ''', (1, "s2", '{"messages": []}', "llmgateway/high", "HTML", 0.8))


def test_switch_missing_session_keeps_current_active_session(db):
    helper = DummyOpenAI()
    first = db.create_session(1, openai_helper=helper)
    second = db.create_session(1, openai_helper=helper)
    assert db.get_active_session_id(1) == second

    assert db.switch_active_session(1, "missing-session") is False

    assert db.get_active_session_id(1) == second
    assert first != second


def test_create_session_copies_active_mode_before_pruning_oldest(db):
    helper = DummyOpenAI()
    first = db.create_session(1, max_sessions=3, openai_helper=helper)
    second = db.create_session(1, max_sessions=3, openai_helper=helper)
    third = db.create_session(1, max_sessions=3, openai_helper=helper)
    mode_context = {
        "messages": [{"role": "system", "content": "mode prompt", "mode_key": "mode"}],
    }
    db.save_conversation_context(
        1,
        mode_context,
        "Markdown",
        0.2,
        55,
        session_id=first,
        openai_helper=helper,
    )
    db.switch_active_session(1, first)

    new_session = db.create_session(1, max_sessions=3, openai_helper=helper)
    context, parse_mode, temperature, max_tokens_percent, _ = db.get_conversation_context(1, new_session)

    assert first not in {session["session_id"] for session in db.list_user_sessions(1)}
    assert {session["session_id"] for session in db.list_user_sessions(1)} == {second, third, new_session}
    assert context["messages"] == [{"role": "system", "content": "mode prompt", "mode_key": "mode"}]
    assert parse_mode == "Markdown"
    assert temperature == 0.2
    assert max_tokens_percent == 55


def test_deleting_active_session_at_limit_does_not_delete_extra_session(db):
    helper = DummyOpenAI()
    for _ in range(5):
        db.create_session(1, max_sessions=5, openai_helper=helper)
    active = db.get_active_session_id(1)

    db.delete_session(1, active, openai_helper=helper)

    sessions = db.list_user_sessions(1)
    session_ids = {session["session_id"] for session in sessions}
    assert len(sessions) == 5
    assert active not in session_ids
    assert db.get_active_session_id(1) in session_ids


def test_legacy_conversation_context_migrates_before_session_index(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._reset_singleton()
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE conversation_context (
                user_id INTEGER PRIMARY KEY,
                context TEXT NOT NULL,
                parse_mode TEXT NOT NULL,
                temperature FLOAT NOT NULL,
                max_tokens_percent INTEGER DEFAULT 100,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            INSERT INTO conversation_context
            (user_id, context, parse_mode, temperature, max_tokens_percent)
            VALUES (?, ?, ?, ?, ?)
        """, (1, '{"messages": []}', "HTML", 0.8, 80))

    migrated = Database()

    sessions = migrated.list_user_sessions(1)
    assert len(sessions) == 1
    assert sessions[0]["session_id"]
    assert sessions[0]["model"] == "llmgateway/high"
    with migrated.get_connection() as conn:
        indexes = conn.execute("PRAGMA index_list(conversation_context)").fetchall()
    assert any(row[1] == "idx_conversation_context_session" for row in indexes)


def test_pragmas_enabled(db):
    with db.get_connection() as conn:
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1


def test_concurrent_access_smoke(db):
    def worker(idx):
        db.save_user_settings(idx, {"x": idx})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_chat_settings_are_persisted_by_chat_id(db):
    db.save_chat_settings("-100123", {"text_document_qa_rag_enabled": True})

    assert db.get_chat_settings("-100123") == {"text_document_qa_rag_enabled": True}
    assert db.get_chat_settings("-100456") is None

    db.save_chat_settings("-100123", {"text_document_qa_rag_enabled": False, "other": "value"})
    assert db.get_chat_settings("-100123") == {
        "text_document_qa_rag_enabled": False,
        "other": "value",
    }


def test_tool_call_events_are_recorded(db):
    db.record_tool_call_event(
        request_id="r1",
        chat_id=10,
        user_id=42,
        plugin_name="demo",
        function_name="demo.run",
        status="success",
        duration_ms=12,
        direct_result=True,
    )

    events = db.list_tool_call_events()

    assert len(events) == 1
    assert events[0]["function_name"] == "demo.run"
    assert events[0]["plugin_name"] == "demo"
    assert events[0]["status"] == "success"
    assert events[0]["duration_ms"] == 12
    assert events[0]["direct_result"] is True


def test_save_image_creates_missing_user_settings(db):
    image_id = db.save_image(42, 42, "telegram-file-id")

    assert image_id
    assert db.get_user_settings(42) == {}
    images = db.get_user_images(42, 42, limit=1)
    assert images[0]["file_id"] == "telegram-file-id"
