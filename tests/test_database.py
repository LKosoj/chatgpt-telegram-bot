import sqlite3
import threading

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


def _assert_sqlite_connection_closed(connection):
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_database_local_storage_is_instance_attribute(db):
    assert "_local" in db.__dict__


def test_failed_init_does_not_cache_singleton_or_leave_thread_connection(tmp_path, monkeypatch):
    db_path = tmp_path / "failed-init.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._reset_singleton()

    original_init_db = Database.init_db
    captured_instances = []
    opened_connections = []
    attempts = 0

    def fail_once(self):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            captured_instances.append(self)
            with self.get_connection() as conn:
                opened_connections.append(conn)
                conn.execute("CREATE TABLE failed_init_probe (id INTEGER)")
                raise RuntimeError("init boom")
        return original_init_db(self)

    monkeypatch.setattr(Database, "init_db", fail_once)

    with pytest.raises(RuntimeError, match="init boom"):
        Database()

    assert Database._instance is None
    assert captured_instances
    assert "_local" in captured_instances[0].__dict__
    assert not hasattr(captured_instances[0]._local, "connection")
    _assert_sqlite_connection_closed(opened_connections[0])

    recovered = Database()

    assert recovered is Database._instance
    assert recovered is not captured_instances[0]
    Database._reset_singleton()


def test_reset_singleton_closes_current_instance_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "reset.db"))
    Database._reset_singleton()
    db = Database()
    with db.get_connection() as conn:
        conn.execute("SELECT 1")
    opened_connection = db._local.connection

    Database._reset_singleton()

    assert Database._instance is None
    assert not hasattr(db._local, "connection")
    _assert_sqlite_connection_closed(opened_connection)


def test_old_instance_del_does_not_close_new_singleton_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "first.db"))
    Database._reset_singleton()
    first = Database()
    with first.get_connection() as conn:
        conn.execute("SELECT 1")

    Database._reset_singleton()
    monkeypatch.setenv("DB_PATH", str(tmp_path / "second.db"))
    second = Database()
    with second.get_connection() as conn:
        conn.execute("SELECT 1")
    second_connection = second._local.connection

    first.__del__()

    assert second_connection.execute("SELECT 1").fetchone()[0] == 1
    with second.get_connection() as conn:
        assert conn is second_connection
    Database._reset_singleton()


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
async def test_save_conversation_context_async_does_not_call_llm(db):
    """После разрыва цикличной зависимости Database→OpenAIHelper, БД не должна
    вызывать LLM. Длинное сообщение оставляет имя «...», короткое — ставит
    fallback из первых 20 символов. Сгенерированное LLM имя — ответственность
    OpenAIHelper._ensure_session_name_with_llm (тестируется отдельно)."""
    helper = DummyOpenAI()
    long_context = {
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
    await db.save_conversation_context_async(1, long_context, "HTML", 0.8, 80, openai_helper=helper)
    sessions = db.list_user_sessions(1, is_active=1)
    assert sessions[0]["session_name"] == "..."

    short_context = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hi"},
        ]
    }
    await db.save_conversation_context_async(2, short_context, "HTML", 0.8, 80, openai_helper=helper)
    sessions2 = db.list_user_sessions(2, is_active=1)
    assert sessions2[0]["session_name"] == "hi"


@pytest.mark.asyncio
async def test_run_in_db_thread_accepts_kwargs_and_leaves_event_loop_thread(db):
    event_loop_thread = threading.get_ident()

    def capture_thread(*, value):
        return threading.get_ident(), value

    worker_thread, value = await db._run_in_db_thread(capture_thread, value=42)

    assert value == 42
    assert worker_thread != event_loop_thread


@pytest.mark.asyncio
async def test_async_db_facade_covers_session_settings_and_image_methods(db):
    await db.save_user_settings_async(42, {"language": "ru"})
    assert await db.get_user_settings_async(42) == {"language": "ru"}

    session_id = await db.create_session_async(42, openai_helper=DummyOpenAI())
    assert session_id

    await db.save_user_model_async(42, "llmgateway/high")
    context, _, _, _, resolved_session_id = await db.get_conversation_context_async(42, session_id)
    assert resolved_session_id == session_id
    assert context["messages"] == []

    image_id = await db.save_image_async(42, 42, "file-id")
    assert image_id
    images = await db.get_user_images_async(42, 42, limit=1)
    assert images[0]["file_id"] == "file-id"

    assert await db.prune_tool_call_events_async(30) == 0
    assert await db.prune_old_images_async(7) == 0


def test_max_sessions_enforced(db):
    helper = DummyOpenAI()
    for _ in range(6):
        db.create_session(1, max_sessions=3, openai_helper=helper)
    sessions = db.list_user_sessions(1)
    assert len(sessions) <= 3


def test_malformed_max_sessions_env_falls_back_for_real_session_paths(db, monkeypatch):
    helper = DummyOpenAI()
    monkeypatch.setenv("MAX_SESSIONS", "many")

    created = db.create_session(10, openai_helper=helper)
    context, _, _, _, auto_created = db.get_conversation_context(
        11,
        openai_helper=helper,
    )
    delete_target = db.create_session(12, openai_helper=helper)
    db.delete_session(12, delete_target, openai_helper=helper)

    assert created
    assert auto_created
    assert context == {"messages": []}
    remaining = db.list_user_sessions(12)
    assert remaining
    assert all(session["session_id"] != delete_target for session in remaining)


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


def test_create_session_without_helper_uses_first_openai_model_from_env(db, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "db-main,db-other")

    session_id = db.create_session(1)

    session = db.list_user_sessions(1)[0]
    assert session["session_id"] == session_id
    assert session["model"] == "db-main"


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

    remaining = {session["session_id"] for session in db.list_user_sessions(1)}
    # Активная (она же старейшая) сессия first вытесняется при прюнинге, но её
    # режим успевает скопироваться в новую сессию (см. ассерты ниже).
    assert first not in remaining
    assert remaining == {second, third, new_session}
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
    monkeypatch.setenv("OPENAI_MODEL", "db-main,db-other")
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
    assert sessions[0]["model"] == "db-main"
    with migrated.get_connection() as conn:
        indexes = conn.execute("PRAGMA index_list(conversation_context)").fetchall()
    assert any(row[1] == "idx_conversation_context_session" for row in indexes)


def test_fresh_database_records_ordered_schema_versions(db):
    with db.get_connection() as conn:
        versions = [
            row[0]
            for row in conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
        ]
    assert versions == [1, 2]


def test_schema_migration_registry_matches_target_version(db):
    versions = [version for version, _migration in db._schema_migrations()]

    assert versions == list(range(1, Database.TARGET_SCHEMA_VERSION + 1))
    assert len(versions) == len(set(versions))


def test_failed_migration_old_table_with_more_rows_is_recovered(tmp_path, monkeypatch):
    db_path = tmp_path / "recovery.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("OPENAI_MODEL", "db-main")
    Database._reset_singleton()

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (2)")
        conn.execute("""
            CREATE TABLE conversation_context_old (
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
            INSERT INTO conversation_context_old
            (user_id, context, parse_mode, temperature, max_tokens_percent)
            VALUES (1, '{"messages": [{"role": "user", "content": "keep"}]}', 'HTML', 0.8, 80)
        """)
        conn.execute("""
            CREATE TABLE conversation_context (
                user_id INTEGER,
                context TEXT NOT NULL,
                model TEXT NOT NULL,
                parse_mode TEXT NOT NULL,
                temperature FLOAT NOT NULL,
                max_tokens_percent INTEGER DEFAULT 100,
                session_id TEXT,
                session_name TEXT DEFAULT NULL,
                is_active INTEGER DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, session_id)
            )
        """)
        conn.commit()

    migrated = Database()

    sessions = migrated.list_user_sessions(1)
    assert len(sessions) == 1
    context, _, _, _, _ = migrated.get_conversation_context(1, sessions[0]["session_id"])
    assert context["messages"][0]["content"] == "keep"
    with migrated.get_connection() as conn:
        old_table = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'conversation_context_old'
        """).fetchone()
        versions = [
            row[0]
            for row in conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
        ]
    assert old_table is None
    assert versions == [1, 2]


def test_failed_migration_without_new_table_resets_schema_version(tmp_path, monkeypatch):
    db_path = tmp_path / "recovery-no-new.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._reset_singleton()

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (2)")
        conn.execute("""
            CREATE TABLE conversation_context_old (
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
            INSERT INTO conversation_context_old
            (user_id, context, parse_mode, temperature)
            VALUES (1, '{"messages": []}', 'HTML', 0.8)
        """)
        conn.commit()

    migrated = Database()

    sessions = migrated.list_user_sessions(1)
    assert len(sessions) == 1
    with migrated.get_connection() as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(conversation_context)").fetchall()]
        versions = [
            row[0]
            for row in conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
        ]
    assert "session_id" in cols
    assert "version" in cols
    assert versions == [1, 2]


def test_recovery_reconciles_stale_schema_version_when_new_table_kept(tmp_path, monkeypatch):
    db_path = tmp_path / "recovery-new-kept.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._reset_singleton()

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (2)")
        conn.execute("""
            CREATE TABLE conversation_context_old (
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
            INSERT INTO conversation_context_old
            (user_id, context, parse_mode, temperature)
            VALUES (1, '{"messages": []}', 'HTML', 0.8)
        """)
        conn.execute("""
            CREATE TABLE conversation_context (
                user_id INTEGER,
                context TEXT NOT NULL,
                model TEXT NOT NULL,
                parse_mode TEXT NOT NULL,
                temperature FLOAT NOT NULL,
                max_tokens_percent INTEGER DEFAULT 100,
                session_id TEXT,
                session_name TEXT DEFAULT NULL,
                is_active INTEGER DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, session_id)
            )
        """)
        conn.execute("""
            INSERT INTO conversation_context
            (user_id, session_id, context, model, parse_mode, temperature, is_active)
            VALUES (1, 's1', '{"messages": []}', 'llmgateway/high', 'HTML', 0.8, 1)
        """)
        conn.commit()

    migrated = Database()

    with migrated.get_connection() as conn:
        old_table = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'conversation_context_old'
        """).fetchone()
        cols = [row[1] for row in conn.execute("PRAGMA table_info(conversation_context)").fetchall()]
        versions = [
            row[0]
            for row in conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
        ]
    assert old_table is None
    assert "version" in cols
    assert versions == [1, 2]

    migrated.save_conversation_context(
        1,
        {"messages": [{"role": "user", "content": "after recovery"}]},
        "HTML",
        0.8,
        80,
        session_id="s1",
        openai_helper=DummyOpenAI(),
    )


def test_pragmas_enabled(db):
    with db.get_connection() as conn:
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1


def test_invalid_journal_mode_falls_back_to_wal(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "journal-invalid.db"))
    monkeypatch.setenv("SQLITE_JOURNAL_MODE", "WAL; DROP TABLE user_settings;")
    Database._reset_singleton()

    migrated = Database()

    with migrated.get_connection() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_valid_journal_mode_is_normalized(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "journal-valid.db"))
    monkeypatch.setenv("SQLITE_JOURNAL_MODE", " delete ")
    Database._reset_singleton()

    migrated = Database()

    with migrated.get_connection() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"


def test_invalid_sqlite_numeric_envs_fall_back(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "sqlite-numeric-invalid.db"))
    monkeypatch.setenv("SQLITE_TIMEOUT", "soon")
    monkeypatch.setenv("SQLITE_BUSY_TIMEOUT_MS", "later")
    caplog.set_level("WARNING")
    Database._reset_singleton()

    migrated = Database()

    try:
        with migrated.get_connection() as conn:
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        messages = [record.getMessage() for record in caplog.records]
        assert any("Invalid SQLITE_TIMEOUT='soon'" in message for message in messages)
        assert any("Invalid SQLITE_BUSY_TIMEOUT_MS='later'" in message for message in messages)
    finally:
        Database._reset_singleton()


def test_non_finite_sqlite_timeout_env_falls_back(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "sqlite-timeout-nan.db"))
    monkeypatch.setenv("SQLITE_TIMEOUT", "nan")
    caplog.set_level("WARNING")
    Database._reset_singleton()

    migrated = Database()

    try:
        with migrated.get_connection() as conn:
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert any(
            "Invalid SQLITE_TIMEOUT='nan'" in record.getMessage()
            for record in caplog.records
        )
    finally:
        Database._reset_singleton()


def test_outer_commit_failure_rolls_back(db, tmp_path):
    class CommitFailConnection(sqlite3.Connection):
        fail_commit = False
        rollback_called = False

        def commit(self):
            if self.fail_commit:
                raise sqlite3.OperationalError("commit failed")
            return super().commit()

        def rollback(self):
            self.rollback_called = True
            return super().rollback()

    failing_path = tmp_path / "commit-fail.db"
    conn = sqlite3.connect(failing_path, factory=CommitFailConnection)
    conn.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.fail_commit = True
    db.db_path = str(failing_path)
    db._local.connection = conn
    db._local.depth = 0

    with pytest.raises(sqlite3.OperationalError, match="commit failed"):
        with db.get_connection() as failing_conn:
            failing_conn.execute("INSERT INTO probe (value) VALUES ('pending')")

    assert conn.rollback_called is True
    conn.fail_commit = False
    assert conn.execute("SELECT COUNT(*) FROM probe").fetchone()[0] == 0

    worker_errors = []

    def write_after_failure():
        try:
            with db.get_connection() as recovered_conn:
                recovered_conn.execute("INSERT INTO probe (value) VALUES ('after')")
        except BaseException as exc:
            worker_errors.append(exc)

    worker = threading.Thread(target=write_after_failure)
    worker.start()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert worker_errors == []
    assert conn.execute("SELECT value FROM probe").fetchone()[0] == "after"
    assert getattr(db._local, "depth", None) == 0


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


def test_tool_call_events_can_be_pruned(db):
    db.record_tool_call_event(function_name="demo.old", status="success")
    db.record_tool_call_event(function_name="demo.fresh", status="success")
    with db.get_connection() as conn:
        conn.execute("""
            UPDATE tool_call_events
            SET created_at = datetime('now', '-45 days')
            WHERE function_name = 'demo.old'
        """)

    assert db.prune_tool_call_events(days=30) == 1

    events = db.list_tool_call_events()
    assert [event["function_name"] for event in events] == ["demo.fresh"]


def test_tool_call_event_prune_keeps_records_inside_cutoff(db):
    db.record_tool_call_event(function_name="demo.old", status="success")
    db.record_tool_call_event(function_name="demo.keep", status="success")
    with db.get_connection() as conn:
        conn.execute("""
            UPDATE tool_call_events
            SET created_at = datetime('now', '-30 days', '-1 hour')
            WHERE function_name = 'demo.old'
        """)
        conn.execute("""
            UPDATE tool_call_events
            SET created_at = datetime('now', '-29 days')
            WHERE function_name = 'demo.keep'
        """)

    assert db.prune_tool_call_events(days=30) == 1

    events = db.list_tool_call_events()
    assert [event["function_name"] for event in events] == ["demo.keep"]


def test_save_image_creates_missing_user_settings(db):
    image_id = db.save_image(42, 42, "telegram-file-id")

    assert image_id
    assert db.get_user_settings(42) == {}
    images = db.get_user_images(42, 42, limit=1)
    assert images[0]["file_id"] == "telegram-file-id"


def test_old_images_can_be_pruned(db):
    old_id = db.save_image(42, 42, "old-file-id")
    fresh_id = db.save_image(42, 42, "fresh-file-id")
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE images SET created_at = datetime('now', '-10 days') WHERE id = ?",
            (old_id,),
        )

    assert db.prune_old_images(days=7) == 1

    images = db.get_user_images(42, 42, limit=10)
    assert [image["id"] for image in images] == [fresh_id]


def test_old_image_prune_keeps_records_inside_cutoff(db):
    old_id = db.save_image(42, 42, "old-file-id")
    keep_id = db.save_image(42, 42, "keep-file-id")
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE images SET created_at = datetime('now', '-7 days', '-1 hour') WHERE id = ?",
            (old_id,),
        )
        conn.execute(
            "UPDATE images SET created_at = datetime('now', '-6 days') WHERE id = ?",
            (keep_id,),
        )

    assert db.prune_old_images(days=7) == 1

    images = db.get_user_images(42, 42, limit=10)
    assert [image["id"] for image in images] == [keep_id]


def test_cleanup_old_images_returns_prune_rowcount(db):
    old_id = db.save_image(42, 42, "old-file-id")
    db.save_image(42, 42, "fresh-file-id")
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE images SET created_at = datetime('now', '-10 days') WHERE id = ?",
            (old_id,),
        )

    assert db.cleanup_old_images(days=7) == 1


def test_retention_indexes_exist(db):
    with db.get_connection() as conn:
        tool_indexes = {
            row[1] for row in conn.execute("PRAGMA index_list(tool_call_events)").fetchall()
        }
        image_indexes = {
            row[1] for row in conn.execute("PRAGMA index_list(images)").fetchall()
        }

        def index_columns(index_name):
            return [row[2] for row in conn.execute(f"PRAGMA index_info({index_name})").fetchall()]

    assert "idx_tool_call_events_created_at" in tool_indexes
    assert "idx_tool_call_events_chat_created" in tool_indexes
    assert "idx_images_created_at" in image_indexes
    assert "idx_images_user_created" in image_indexes
    assert "idx_images_user_chat_created" in image_indexes
    assert index_columns("idx_tool_call_events_chat_created") == ["chat_id", "created_at"]
    assert index_columns("idx_tool_call_events_created_at") == ["created_at"]
    assert index_columns("idx_file_id_hash") == ["file_id_hash"]
    assert index_columns("idx_images_created_at") == ["created_at"]
    assert index_columns("idx_images_user_created") == ["user_id", "created_at"]
    assert index_columns("idx_images_user_chat_created") == ["user_id", "chat_id", "created_at"]


def test_oldest_session_excludes_protected_session_ids(db):
    helper = DummyOpenAI()
    session_ids = [db.create_session(1, max_sessions=10, openai_helper=helper) for _ in range(4)]
    with db.get_connection() as conn:
        for index, session_id in enumerate(session_ids):
            conn.execute(
                """
                UPDATE conversation_context
                SET created_at = datetime('2020-01-01', '+' || ? || ' seconds')
                WHERE user_id = ? AND session_id = ?
                """,
                (index, 1, session_id),
            )

    oldest = db.get_oldest_session_ids_for_limit(
        1,
        max_sessions=3,
        exclude_session_ids=[session_ids[0]],
    )

    assert oldest == [session_ids[1], session_ids[2]]


# ---------------------------------------------------------------------------
# Новые тесты: WP-A executor, CAS-locking, migration < 2
# ---------------------------------------------------------------------------

def test_version_column_increments_on_save(db):
    """version инкрементируется при каждом save_conversation_context."""
    helper = DummyOpenAI()
    context = {"messages": [{"role": "user", "content": "hello"}]}
    session_id = db.create_session(1, openai_helper=helper)

    db.save_conversation_context(1, context, "HTML", 0.8, 80, session_id=session_id)
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT version FROM conversation_context WHERE user_id = ? AND session_id = ?",
            (1, session_id),
        ).fetchone()
    assert row[0] == 1

    db.save_conversation_context(1, context, "HTML", 0.8, 80, session_id=session_id)
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT version FROM conversation_context WHERE user_id = ? AND session_id = ?",
            (1, session_id),
        ).fetchone()
    assert row[0] == 2


def test_save_bumps_version_regardless_of_existing_value(db):
    """save_conversation_context всегда успешно пишет и поднимает version на 1,
    каким бы ни было текущее значение version. Отдельный CAS-retry не нужен —
    атомарность обеспечивает write-lock транзакции (BEGIN IMMEDIATE)."""
    helper = DummyOpenAI()
    context_a = {"messages": [{"role": "user", "content": "initial"}]}
    session_id = db.create_session(1, openai_helper=helper)

    # Первый save — version становится 1.
    db.save_conversation_context(1, context_a, "HTML", 0.8, 80, session_id=session_id)

    # Вручную выставляем произвольную version.
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE conversation_context SET version = ? WHERE user_id = ? AND session_id = ?",
            (42, 1, session_id),
        )

    context_b = {"messages": [{"role": "user", "content": "updated"}]}
    db.save_conversation_context(1, context_b, "HTML", 0.8, 80, session_id=session_id)

    # Данные актуальны, version поднялась ровно на 1.
    details = db.get_session_details(1, session_id)
    assert details["context"]["messages"][0]["content"] == "updated"
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT version FROM conversation_context WHERE user_id = ? AND session_id = ?",
            (1, session_id),
        ).fetchone()
    assert row[0] == 43


def test_migration_adds_version_column_to_legacy_db(tmp_path, monkeypatch):
    """БД без колонки version (схема v1) получает её при init (миграция < 2)."""
    db_path = tmp_path / "v1.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._reset_singleton()

    # Создаём БД, имитирующую состояние после миграции v1:
    # таблица есть и session_id есть, но version отсутствует.
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.execute("""
            CREATE TABLE user_settings (
                user_id INTEGER PRIMARY KEY,
                settings TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE conversation_context (
                user_id INTEGER,
                context TEXT NOT NULL,
                model TEXT NOT NULL,
                parse_mode TEXT NOT NULL,
                temperature FLOAT NOT NULL,
                max_tokens_percent INTEGER DEFAULT 100,
                session_id TEXT,
                session_name TEXT DEFAULT NULL,
                is_active INTEGER DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, session_id)
            )
        """)
        conn.execute("""
            INSERT INTO conversation_context
            (user_id, session_id, context, model, parse_mode, temperature, is_active)
            VALUES (1, 'ses1', '{"messages": []}', 'llmgateway/high', 'HTML', 0.8, 1)
        """)
        conn.commit()

    migrated = Database()

    # Колонка version должна присутствовать.
    with migrated.get_connection() as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(conversation_context)").fetchall()]
    assert "version" in cols

    # Старые данные целы.
    sessions = migrated.list_user_sessions(1)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "ses1"

    # schema_version должна содержать запись 2.
    with migrated.get_connection() as conn:
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert ver == 2


def test_legacy_no_session_id_migration_creates_version_column(tmp_path, monkeypatch):
    """Legacy-БД без колонки session_id мигрирует через migrate_conversation_context
    и получает колонку version; последующий save не падает на 'no such column'."""
    db_path = tmp_path / "legacy.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    Database._reset_singleton()

    # Старая схема: conversation_context без session_id и без version.
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE conversation_context (
                user_id INTEGER PRIMARY KEY,
                context TEXT NOT NULL,
                model TEXT NOT NULL,
                parse_mode TEXT NOT NULL,
                temperature FLOAT NOT NULL,
                max_tokens_percent INTEGER DEFAULT 100,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            INSERT INTO conversation_context
            (user_id, context, model, parse_mode, temperature)
            VALUES (1, '{"messages": []}', 'llmgateway/high', 'HTML', 0.8)
        """)
        conn.commit()

    migrated = Database()

    # После миграции есть и session_id, и version.
    with migrated.get_connection() as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(conversation_context)").fetchall()]
    assert "session_id" in cols
    assert "version" in cols

    # save_conversation_context работает (не падает на отсутствующей колонке version).
    context = {"messages": [{"role": "user", "content": "after migration"}]}
    session_id = migrated.save_conversation_context(
        1, context, "HTML", 0.8, 80, openai_helper=DummyOpenAI()
    )
    assert session_id
    with migrated.get_connection() as conn:
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert ver == 2


def test_executor_single_worker(db):
    """После нескольких async-операций у executor ровно один поток."""
    import asyncio

    async def _run():
        for _ in range(5):
            await db._run_in_db_thread(lambda: None)
        executor = db._get_executor()
        return executor

    executor = asyncio.run(_run())
    # ThreadPoolExecutor._max_workers — публичный (CPython).
    assert executor._max_workers == 1


def test_shutdown_idempotent(db):
    """shutdown() не падает при двойном вызове."""
    db.shutdown()
    db.shutdown()  # не должно бросить

    # После shutdown executor=None; _get_executor создаёт новый.
    assert db._executor is None
    new_exec = db._get_executor()
    assert new_exec is not None
    # Финальная очистка — чтобы не мешать fixture teardown.
    db.shutdown()
