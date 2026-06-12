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
    dream_user_messages = [m for m in plugin.openai.calls[0]["messages"] if m["role"] == "user"]
    assert any("json" in m["content"].lower() for m in dream_user_messages)


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
        "SELECT last_event_id, fail_count, retry_after FROM hindsight_dream_state WHERE user_id = ?", (42,)
    )
    run = await plugin.db_handle.fetch_one("SELECT status, error FROM hindsight_dream_runs")
    docs = await plugin.db_handle.fetch_all("SELECT * FROM hindsight_memory_documents")

    # State is now created on failure: watermark stays at 0, fail_count incremented.
    assert state is not None
    assert state["last_event_id"] == 0
    assert state["fail_count"] == 1
    assert run["status"] == "failed"
    assert "no JSON" in run["error"]
    assert docs == []


async def test_dream_tick_backoff_after_failure(plugin):
    """After one failure, retry_after is set and a second tick does not call LLM."""
    plugin.openai.content = "not json"
    await _seed_user_events(plugin)

    await plugin._dream_tick(application=None)

    state = await plugin.db_handle.fetch_one(
        "SELECT fail_count, retry_after FROM hindsight_dream_state WHERE user_id = ?", (42,)
    )
    assert state["fail_count"] == 1
    assert state["retry_after"] is not None

    calls_before = len(plugin.openai.calls)
    await plugin._dream_tick(application=None)
    # No new LLM call because user is in backoff.
    assert len(plugin.openai.calls) == calls_before


async def test_dream_tick_advances_watermark_after_max_attempts(plugin):
    """After MAX_ATTEMPTS failures the watermark is advanced past the failing events."""
    from bot.plugins.hindsight_memory import HINDSIGHT_DREAM_MAX_ATTEMPTS
    plugin.openai.content = "not json"
    await _seed_user_events(plugin)

    # Force fail_count up to one below the limit, simulating prior failures
    # by directly writing state (last_event_id=0 so events are still pending).
    await plugin.db_handle.execute(
        '''
        INSERT INTO hindsight_dream_state(user_id, last_event_id, fail_count, retry_after, updated_at)
        VALUES (42, 0, ?, NULL, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
            fail_count = excluded.fail_count,
            retry_after = NULL,
            updated_at = CURRENT_TIMESTAMP
        ''',
        (HINDSIGHT_DREAM_MAX_ATTEMPTS - 1,),
    )

    await plugin._dream_tick(application=None)

    state = await plugin.db_handle.fetch_one(
        "SELECT last_event_id, fail_count, retry_after FROM hindsight_dream_state WHERE user_id = ?", (42,)
    )
    # Watermark advanced past the 2 seeded events; fail_count reset.
    assert state["last_event_id"] == 2
    assert state["fail_count"] == 0
    assert state["retry_after"] is None


async def test_dream_tick_resets_fail_count_on_success(plugin):
    """A successful run resets fail_count and retry_after to 0/NULL."""
    from bot.plugins.hindsight_memory import HINDSIGHT_DREAM_MAX_ATTEMPTS
    plugin.openai.content = json.dumps({
        "documents": [{
            "path": "profile/pref.md",
            "kind": "profile",
            "content": "Concise.",
        }]
    })
    await _seed_user_events(plugin)

    # Pre-seed a non-zero fail_count.
    await plugin.db_handle.execute(
        '''
        INSERT INTO hindsight_dream_state(user_id, last_event_id, fail_count, retry_after, updated_at)
        VALUES (42, 0, 2, NULL, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
            fail_count = excluded.fail_count,
            retry_after = NULL,
            updated_at = CURRENT_TIMESTAMP
        ''',
        (),
    )

    await plugin._dream_tick(application=None)

    state = await plugin.db_handle.fetch_one(
        "SELECT last_event_id, fail_count, retry_after FROM hindsight_dream_state WHERE user_id = ?", (42,)
    )
    assert state["last_event_id"] == 2
    assert state["fail_count"] == 0
    assert state["retry_after"] is None
