from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("tiktoken")

from bot.database import Database
from bot.plugins.db_handle import DbHandle
from bot.plugins.hindsight_memory import HindsightMemoryPlugin
from bot.plugins.hooks import UserMessagePayload


class FakeOpenAI:
    def __init__(self, content: str):
        self.config = {"light_model": "fake-light"}
        self.content = content
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


@pytest.fixture()
def plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "dream.db"))
    Database._reset_singleton()
    db = Database()
    plugin = HindsightMemoryPlugin()
    with db.get_connection() as conn:
        for stmt in plugin.register_schema():
            conn.execute(stmt)
    plugin.initialize(
        openai=FakeOpenAI('{"documents":[]}'),
        plugin_config={
            "hindsight_base_url": "http://x",
            "hindsight_api_token": "t",
            "hindsight_dream_enabled": True,
        },
    )
    plugin.client = SimpleNamespace(enabled=True)
    plugin.db_handle = DbHandle(db)
    yield plugin
    Database._reset_singleton()


async def _seed_user_events(plugin, user_id=42):
    await plugin.on_user_message(UserMessagePayload(
        chat_id=10, user_id=user_id, request_id="r1", text="I prefer concise answers.",
        has_image=False, has_voice=False, is_command=False, ts=1.0,
    ))
    await plugin.on_user_message(UserMessagePayload(
        chat_id=10, user_id=user_id, request_id="r2", text="Remember the project uses .venv.",
        has_image=False, has_voice=False, is_command=False, ts=2.0,
    ))


async def test_dream_tick_writes_candidate_and_advances_watermark(plugin):
    plugin.openai.content = json.dumps({
        "documents": [{
            "path": "profile/preferences.md",
            "kind": "profile",
            "content": "User prefers concise answers.",
        }]
    })
    await _seed_user_events(plugin)

    await plugin._dream_tick(application=None)

    doc = await plugin.db_handle.fetch_one(
        "SELECT path, kind, status, content FROM hindsight_memory_documents"
    )
    state = await plugin.db_handle.fetch_one(
        "SELECT last_event_id FROM hindsight_dream_state WHERE user_id = ?", (42,)
    )
    run = await plugin.db_handle.fetch_one("SELECT status FROM hindsight_dream_runs")

    assert doc["path"] == "profile/preferences.md"
    assert doc["kind"] == "profile"
    assert doc["status"] == "candidate"
    assert "concise" in doc["content"]
    assert state["last_event_id"] == 2
    assert run["status"] == "completed"


async def test_dream_tick_writes_lesson_as_candidate_only(plugin):
    plugin.openai.content = json.dumps({
        "documents": [{
            "path": "tools/pytest.md",
            "kind": "lesson",
            "content": "Use focused pytest commands for touched plugin behavior.",
        }]
    })
    await _seed_user_events(plugin)

    await plugin._dream_tick(application=None)

    doc = await plugin.db_handle.fetch_one(
        "SELECT path, kind, status, lesson_type, verified_at FROM hindsight_memory_documents"
    )

    assert doc["path"] == "tools/pytest.md"
    assert doc["kind"] == "lesson"
    assert doc["status"] == "candidate"
    assert doc["lesson_type"] == "lesson_candidate"
    assert doc["verified_at"] is None


async def test_dream_tick_leaves_watermark_when_model_output_is_invalid(plugin):
    plugin.openai.content = "not json"
    await _seed_user_events(plugin)

    await plugin._dream_tick(application=None)

    state = await plugin.db_handle.fetch_one(
        "SELECT last_event_id FROM hindsight_dream_state WHERE user_id = ?", (42,)
    )
    run = await plugin.db_handle.fetch_one("SELECT status, error FROM hindsight_dream_runs")
    docs = await plugin.db_handle.fetch_all("SELECT * FROM hindsight_memory_documents")

    assert state is None
    assert run["status"] == "failed"
    assert "no JSON" in run["error"]
    assert docs == []
