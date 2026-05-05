import os
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


def test_create_session_does_not_prune_hindsight_sessions_without_async_finalize(db):
    helper = DummyHindsightOpenAI()
    for _ in range(4):
        db.create_session(1, max_sessions=3, openai_helper=helper)
    sessions = db.list_user_sessions(1)
    assert len(sessions) == 4


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
