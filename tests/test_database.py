import os
import sqlite3
import threading
from pathlib import Path

import pytest

from bot.database import Database


class DummyOpenAI:
    def __init__(self):
        self.config = {"model": "llmgateway/high"}

    def ask_sync(self, *args, **kwargs):
        return ("ShortName", None)


class DummyHindsightOpenAI(DummyOpenAI):
    def __init__(self):
        super().__init__()
        self.config["hindsight_auto_save"] = True

    def is_hindsight_enabled(self):
        return True


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._instance = None
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


def test_session_name_uses_text_from_multimodal_content(db):
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

    db.save_conversation_context(1, context, "HTML", 0.8, 80, openai_helper=DummyOpenAI())

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


def test_switch_missing_session_keeps_current_active_session(db):
    helper = DummyOpenAI()
    first = db.create_session(1, openai_helper=helper)
    second = db.create_session(1, openai_helper=helper)
    assert db.get_active_session_id(1) == second

    assert db.switch_active_session(1, "missing-session") is False

    assert db.get_active_session_id(1) == second
    assert first != second


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
    Database._instance = None
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


def test_create_session_prunes_hindsight_sessions_through_finalize_jobs(db):
    helper = DummyHindsightOpenAI()
    for _ in range(4):
        db.create_session(1, max_sessions=3, openai_helper=helper)
    sessions = db.list_user_sessions(1)
    assert len(sessions) <= 3
    with db.get_connection() as conn:
        job_count = conn.execute("SELECT COUNT(*) FROM hindsight_finalize_jobs").fetchone()[0]
    assert job_count == 1


def test_hindsight_finalize_job_claim_and_done(db):
    messages = [{"role": "user", "content": "remember this"}]
    job_id = db.create_hindsight_finalize_job(1, "session-1", messages)

    jobs = db.claim_hindsight_finalize_jobs(limit=5)

    assert len(jobs) == 1
    assert jobs[0]["id"] == job_id
    assert jobs[0]["user_id"] == 1
    assert jobs[0]["session_id"] == "session-1"
    assert jobs[0]["messages"] == messages

    assert db.mark_hindsight_finalize_job_done(job_id, saved_count=2) is True
    assert db.claim_hindsight_finalize_jobs(limit=5) == []


def test_hindsight_finalize_job_retries_then_fails(db):
    job_id = db.create_hindsight_finalize_job(1, "session-1", [{"role": "user", "content": "remember this"}])
    assert db.claim_hindsight_finalize_jobs(limit=5)[0]["id"] == job_id

    assert db.mark_hindsight_finalize_job_failed(
        job_id,
        "temporary error",
        retry_delay_seconds=0,
        max_attempts=2,
    ) is True
    assert db.claim_hindsight_finalize_jobs(limit=5)[0]["id"] == job_id

    assert db.mark_hindsight_finalize_job_failed(
        job_id,
        "final error",
        retry_delay_seconds=0,
        max_attempts=2,
    ) is True
    assert db.claim_hindsight_finalize_jobs(limit=5) == []


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


def test_save_image_creates_missing_user_settings(db):
    image_id = db.save_image(42, 42, "telegram-file-id")

    assert image_id
    assert db.get_user_settings(42) == {}
    images = db.get_user_images(42, 42, limit=1)
    assert images[0]["file_id"] == "telegram-file-id"


def test_delete_user_data_removes_images_before_settings(db):
    db.save_image(42, 42, "telegram-file-id")

    db.delete_user_data(42)

    assert db.get_user_settings(42) is None
    assert db.get_user_images(42, 42, limit=1) == []
