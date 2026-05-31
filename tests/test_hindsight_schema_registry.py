from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from bot.database import Database
from bot.plugins.db_handle import DbHandle
from bot.plugins.hindsight_memory import HindsightMemoryPlugin


def test_register_schema_declares_memory_pipeline_tables():
    plugin = HindsightMemoryPlugin()
    stmts = plugin.register_schema()
    assert isinstance(stmts, list)
    joined = "\n".join(stmts)
    assert "hindsight_finalize_jobs" in joined
    assert "hindsight_memory_events" in joined
    assert "hindsight_dream_state" in joined
    assert "hindsight_memory_clear_state" in joined
    assert "hindsight_dream_runs" in joined
    assert "hindsight_memory_documents" in joined
    assert any("CREATE INDEX" in s for s in stmts)


def test_register_schema_creates_table_on_fresh_sqlite():
    plugin = HindsightMemoryPlugin()
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            for stmt in plugin.register_schema():
                cur.execute(stmt)
            conn.commit()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='hindsight_finalize_jobs'"
            )
            assert cur.fetchone() is not None
            cur.execute("PRAGMA table_info(hindsight_memory_documents)")
            columns = {row[1] for row in cur.fetchall()}
            assert "lesson_type" in columns
            assert "verified_at" in columns
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name='idx_hindsight_finalize_jobs_status_next_attempt'"
            )
            assert cur.fetchone() is not None
        finally:
            conn.close()


def test_initialize_migrates_existing_memory_documents_table(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "legacy.db"))
    Database._reset_singleton()
    db = Database()
    try:
        with db.get_connection() as conn:
            conn.execute(
                '''
                CREATE TABLE hindsight_memory_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    source_run_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    approved_at TIMESTAMP,
                    discarded_at TIMESTAMP
                )
                '''
            )

        plugin = HindsightMemoryPlugin()
        plugin.initialize(db=DbHandle(db), plugin_config={})

        with db.get_connection() as conn:
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(hindsight_memory_documents)").fetchall()
            }
        assert "lesson_type" in columns
        assert "verified_at" in columns
    finally:
        Database._reset_singleton()
