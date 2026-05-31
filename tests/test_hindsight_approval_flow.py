from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("tiktoken")

from bot.database import Database
from bot.plugins.db_handle import DbHandle
from bot.plugins.hindsight_memory import HindsightMemoryPlugin


class FakeHindsight:
    enabled = True

    def __init__(self):
        self.retained = []
        self.cleared = []

    async def retain_memories(self, bank_id, items, **kwargs):
        self.retained.append((bank_id, items, kwargs))
        return {"success": True}

    async def clear_bank(self, bank_id):
        self.cleared.append(bank_id)
        return {"success": True}


class FailingClearHindsight(FakeHindsight):
    async def clear_bank(self, bank_id):
        self.cleared.append(bank_id)
        raise RuntimeError("remote clear failed")


class BlockingHindsight(FakeHindsight):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.retain_started = asyncio.Event()
        self.release_retain = asyncio.Event()

    async def retain_memories(self, bank_id, items, **kwargs):
        self.calls.append("retain-start")
        self.retain_started.set()
        await self.release_retain.wait()
        self.calls.append("retain-end")
        return await super().retain_memories(bank_id, items, **kwargs)

    async def clear_bank(self, bank_id):
        self.calls.append("clear")
        return await super().clear_bank(bank_id)


@pytest.fixture()
def plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "approval.db"))
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
    plugin.client = FakeHindsight()
    plugin.db_handle = DbHandle(db)
    yield plugin
    Database._reset_singleton()


async def _insert_candidate(
    plugin,
    user_id=42,
    path="profile/preferences.md",
    content="User prefers concise answers.",
):
    await plugin.db_handle.execute(
        '''
        INSERT INTO hindsight_memory_documents
        (user_id, path, kind, status, content, content_hash, version)
        VALUES (?, ?, 'profile', 'candidate', ?, 'hash', 1)
        ''',
        (user_id, path, content),
    )
    row = await plugin.db_handle.fetch_one(
        "SELECT id FROM hindsight_memory_documents WHERE user_id = ?", (user_id,)
    )
    return int(row["id"])


async def test_approve_candidate_promotes_local_doc_and_retains_to_hindsight(plugin):
    document_id = await _insert_candidate(plugin)

    message = await plugin._approve_candidate_document(42, document_id)

    row = await plugin.db_handle.fetch_one(
        "SELECT status, approved_at FROM hindsight_memory_documents WHERE id = ?",
        (document_id,),
    )
    assert message == f"Approved memory candidate {document_id}."
    assert row["status"] == "approved"
    assert row["approved_at"] is not None
    assert plugin.client.retained[0][0] == "telegram-42"
    item = plugin.client.retained[0][1][0]
    assert item["content"] == "User prefers concise answers."
    assert item["document_id"].startswith("telegram-42-memory-")
    assert "-v" not in item["document_id"]
    assert "version:1" in item["tags"]
    assert plugin.client.retained[0][2]["async_store"] is False


async def test_discard_candidate_does_not_retain(plugin):
    document_id = await _insert_candidate(plugin)

    message = await plugin._discard_candidate_document(42, document_id)

    row = await plugin.db_handle.fetch_one(
        "SELECT status, discarded_at FROM hindsight_memory_documents WHERE id = ?",
        (document_id,),
    )
    assert message == f"Discarded memory candidate {document_id}."
    assert row["status"] == "discarded"
    assert row["discarded_at"] is not None
    assert plugin.client.retained == []


async def test_approval_rejects_candidate_for_other_user(plugin):
    document_id = await _insert_candidate(plugin, user_id=99)

    message = await plugin._approve_candidate_document(42, document_id)

    row = await plugin.db_handle.fetch_one(
        "SELECT status FROM hindsight_memory_documents WHERE id = ?",
        (document_id,),
    )
    assert message == "Memory candidate not found."
    assert row["status"] == "candidate"
    assert plugin.client.retained == []


async def test_approval_does_not_finalize_when_retention_drops_document(plugin):
    document_id = await _insert_candidate(
        plugin,
        content="password=super-secret",
    )

    message = await plugin._approve_candidate_document(42, document_id)

    row = await plugin.db_handle.fetch_one(
        "SELECT status FROM hindsight_memory_documents WHERE id = ?",
        (document_id,),
    )
    assert message == "Approve failed: memory document was not retained."
    assert row["status"] == "candidate"
    assert plugin.client.retained == []


async def test_approval_allows_safe_content_when_path_contains_sensitive_word(plugin):
    document_id = await _insert_candidate(
        plugin,
        path="tools/token-budget.md",
        content="Use a small budget for lightweight review tasks.",
    )

    message = await plugin._approve_candidate_document(42, document_id)

    row = await plugin.db_handle.fetch_one(
        "SELECT status FROM hindsight_memory_documents WHERE id = ?",
        (document_id,),
    )
    assert message == f"Approved memory candidate {document_id}."
    assert row["status"] == "approved"
    assert plugin.client.retained[0][1][0]["context"] == "Approved dream memory document."


async def test_clear_memory_deletes_local_memory_artifacts(plugin):
    candidate_id = await _insert_candidate(plugin)
    await plugin.db_handle.execute(
        '''
        INSERT INTO hindsight_memory_documents
        (user_id, path, kind, status, content, content_hash, version)
        VALUES (42, 'project/demo.md', 'project', 'approved', 'Project fact.', 'hash2', 1)
        '''
    )
    await plugin.db_handle.execute(
        '''
        INSERT INTO hindsight_memory_events(user_id, event_type, payload_json)
        VALUES (42, 'user_message', '{}')
        '''
    )
    await plugin.db_handle.execute(
        '''
        INSERT INTO hindsight_dream_state(user_id, last_event_id)
        VALUES (42, 7)
        '''
    )
    await plugin.db_handle.execute(
        '''
        INSERT INTO hindsight_dream_runs(user_id, status, input_summary, output_json)
        VALUES (42, 'completed', 'input', '{"documents":[]}')
        '''
    )
    await plugin.db_handle.execute(
        '''
        INSERT INTO hindsight_finalize_jobs(user_id, session_id, messages)
        VALUES (42, 's1', '{"messages":[]}')
        '''
    )

    await plugin._clear_memory(SimpleNamespace(), 42)

    assert plugin.client.cleared == ["telegram-42"]
    assert candidate_id
    for table in (
        "hindsight_memory_documents",
        "hindsight_memory_events",
        "hindsight_dream_state",
        "hindsight_dream_runs",
        "hindsight_finalize_jobs",
    ):
        row = await plugin.db_handle.fetch_one(
            f"SELECT COUNT(*) AS count FROM {table} WHERE user_id = 42"
        )
        assert row["count"] == 0


async def test_clear_memory_deletes_local_artifacts_when_remote_clear_fails(plugin):
    await _insert_candidate(plugin)
    plugin.client = FailingClearHindsight()

    with pytest.raises(RuntimeError, match="remote clear failed"):
        await plugin._clear_memory(SimpleNamespace(), 42)

    row = await plugin.db_handle.fetch_one(
        "SELECT COUNT(*) AS count FROM hindsight_memory_documents WHERE user_id = 42"
    )
    assert plugin.client.cleared == ["telegram-42"]
    assert row["count"] == 0


async def test_clear_waits_for_inflight_approval_and_removes_local_doc(plugin):
    document_id = await _insert_candidate(plugin)
    plugin.client = BlockingHindsight()

    approve_task = asyncio.create_task(plugin._approve_candidate_document(42, document_id))
    await plugin.client.retain_started.wait()
    clear_task = asyncio.create_task(plugin._clear_memory(SimpleNamespace(), 42))
    await asyncio.sleep(0)

    assert plugin.client.calls == ["retain-start"]

    plugin.client.release_retain.set()
    approve_message, _ = await asyncio.gather(approve_task, clear_task)

    docs = await plugin.db_handle.fetch_all(
        "SELECT * FROM hindsight_memory_documents WHERE user_id = 42"
    )
    assert approve_message == f"Approved memory candidate {document_id}."
    assert plugin.client.calls == ["retain-start", "retain-end", "clear"]
    assert docs == []


async def test_session_delete_after_clear_does_not_enqueue_finalize_job(plugin):
    await plugin._clear_memory(SimpleNamespace(), 42)

    await plugin.on_session_before_delete(SimpleNamespace(
        user_id=42,
        session_id="s-after-clear",
        messages=({"role": "user", "content": "remember old fact"},),
    ))

    row = await plugin.db_handle.fetch_one(
        "SELECT COUNT(*) AS count FROM hindsight_finalize_jobs WHERE user_id = 42"
    )
    assert row["count"] == 0


async def test_new_session_after_clear_can_enqueue_finalize_job(plugin):
    await plugin._clear_memory(SimpleNamespace(), 42)
    await plugin.db_handle.execute(
        '''
        INSERT INTO conversation_context
        (user_id, session_id, context, model, parse_mode, temperature, created_at)
        VALUES (
            42,
            's-new',
            '{"messages":[{"role":"user","content":"new fact"}]}',
            'm',
            'HTML',
            0.8,
            datetime(CURRENT_TIMESTAMP, '+1 second')
        )
        '''
    )

    await plugin.on_session_before_delete(SimpleNamespace(
        user_id=42,
        session_id="s-new",
        messages=({"role": "user", "content": "new fact"},),
    ))

    row = await plugin.db_handle.fetch_one(
        "SELECT messages FROM hindsight_finalize_jobs WHERE user_id = 42"
    )
    assert row is not None
    assert '"clear_generation": 1' in row["messages"]


async def test_same_second_pre_clear_session_does_not_enqueue_finalize_job(plugin):
    await plugin.db_handle.execute(
        '''
        INSERT INTO conversation_context
        (user_id, session_id, context, model, parse_mode, temperature, created_at)
        VALUES (
            42,
            's-old-same-second',
            '{"messages":[{"role":"user","content":"old fact"}]}',
            'm',
            'HTML',
            0.8,
            strftime('%Y-%m-%d %H:%M:%f', 'now', '-0.100 seconds')
        )
        '''
    )
    await plugin._clear_memory(SimpleNamespace(), 42)

    await plugin.on_session_before_delete(SimpleNamespace(
        user_id=42,
        session_id="s-old-same-second",
        messages=({"role": "user", "content": "old fact"},),
    ))

    row = await plugin.db_handle.fetch_one(
        "SELECT COUNT(*) AS count FROM hindsight_finalize_jobs WHERE user_id = 42"
    )
    assert row["count"] == 0


async def test_assistant_event_for_pre_clear_request_is_not_appended(plugin):
    await plugin.on_user_message(SimpleNamespace(
        user_id=42,
        chat_id=10,
        request_id="pre-clear",
        text="old request",
        has_image=False,
        has_voice=False,
        is_command=False,
        ts=1.0,
    ))
    await plugin._clear_memory(SimpleNamespace(), 42)

    await plugin.on_assistant_response(SimpleNamespace(
        user_id=42,
        chat_id=10,
        request_id="pre-clear",
        text="old answer",
        tokens=3,
        model="m",
        ts=2.0,
    ))

    rows = await plugin.db_handle.fetch_all(
        "SELECT event_type, role FROM hindsight_memory_events WHERE user_id = 42"
    )
    assert rows == []


async def test_delayed_user_event_for_pre_clear_request_is_not_appended(plugin):
    clear_started = threading.Event()
    release_user_event = threading.Event()
    original_append = plugin._append_memory_event_sync

    def delayed_append(*args, **kwargs):
        clear_started.set()
        release_user_event.wait(timeout=5)
        return original_append(*args, **kwargs)

    plugin._append_memory_event_sync = delayed_append

    user_task = asyncio.create_task(plugin.on_user_message(SimpleNamespace(
        user_id=42,
        chat_id=10,
        request_id="delayed-pre-clear",
        text="old request",
        has_image=False,
        has_voice=False,
        is_command=False,
        ts=1.0,
    )))
    assert await asyncio.to_thread(clear_started.wait, 5)
    await plugin._clear_memory(SimpleNamespace(), 42)
    release_user_event.set()
    await user_task

    rows = await plugin.db_handle.fetch_all(
        "SELECT event_type, role FROM hindsight_memory_events WHERE user_id = 42"
    )
    assert rows == []
