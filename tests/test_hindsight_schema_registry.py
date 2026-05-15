from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

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
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name='idx_hindsight_finalize_jobs_status_next_attempt'"
            )
            assert cur.fetchone() is not None
        finally:
            conn.close()
