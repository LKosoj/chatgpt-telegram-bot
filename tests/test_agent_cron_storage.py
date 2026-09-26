"""SQLite storage + atomic claim behavior for AgentCronPlugin (T04).

Local ``cron_db`` fixture mirrors ``agent_db`` in ``tests/conftest.py`` but is
kept in this file since ``conftest.py`` is outside this task's ownership.
"""
import json
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from bot.database import Database
from bot.plugins.agent_cron import AGENT_CRON_JOB_LEASE_SECONDS, AgentCronPlugin
from bot.plugins.db_handle import DbHandle


@pytest.fixture()
def cron_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    Database._reset_singleton()
    db = Database()
    with db.get_connection() as conn:
        for stmt in AgentCronPlugin().register_schema():
            conn.execute(stmt)
    yield db
    Database._reset_singleton()


def _insert_job(db, **overrides):
    defaults = dict(
        id="job1", scope="s", chat_id=1, user_id=1, schedule="daily at 09:00", prompt="p",
        schedule_type="daily", next_run_at=None, interval_seconds=None, hour=9, minute=0,
        weekday=None, status="active", paused=0, reply_to_message_id=None, message_thread_id=None,
        created_at=datetime.now().isoformat(timespec="seconds"), last_started_at=None,
        last_finished_at=None, last_error=None, last_tokens=None, locked_at=None, locked_by=None,
    )
    defaults.update(overrides)
    with db.get_connection() as conn:
        conn.execute(
            '''INSERT INTO agent_cron_jobs (
                id, scope, chat_id, user_id, schedule, prompt, schedule_type, next_run_at,
                interval_seconds, hour, minute, weekday, status, paused, reply_to_message_id,
                message_thread_id, created_at, last_started_at, last_finished_at, last_error,
                last_tokens, locked_at, locked_by
            ) VALUES (:id, :scope, :chat_id, :user_id, :schedule, :prompt, :schedule_type,
                :next_run_at, :interval_seconds, :hour, :minute, :weekday, :status, :paused,
                :reply_to_message_id, :message_thread_id, :created_at, :last_started_at,
                :last_finished_at, :last_error, :last_tokens, :locked_at, :locked_by)''',
            defaults,
        )
    return defaults


def _row(db, job_id):
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM agent_cron_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


# --- register_schema ---------------------------------------------------


def test_register_schema_creates_table_and_indexes(cron_db):
    with cron_db.get_connection() as conn:
        names = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            ).fetchall()
        }
    assert "agent_cron_jobs" in names
    assert "idx_agent_cron_jobs_due" in names
    assert "idx_agent_cron_jobs_scope" in names


# --- JSON import ---------------------------------------------------


def test_import_json_jobs_moves_file_and_preserves_fields(tmp_path, cron_db):
    jobs_file = tmp_path / "agent_cron_jobs.json"
    data = {
        "scope-a": {
            "j1": {
                "id": "j1", "scope": "scope-a", "chat_id": 10, "user_id": 20,
                "schedule": "every 2 hours", "prompt": "check", "schedule_type": "interval",
                "next_run_at": "2030-01-01T10:00:00", "interval_seconds": 7200,
                "status": "active", "paused": False, "created_at": "2029-01-01T00:00:00",
                "reply_to_message_id": 5, "message_thread_id": None,
            },
            "j2": {
                "id": "j2", "scope": "scope-a", "chat_id": 10, "user_id": 20,
                "schedule": "weekly monday at 10:00", "prompt": "weekly check",
                "schedule_type": "weekly", "next_run_at": "2030-01-06T10:00:00",
                "hour": 10, "minute": 0, "weekday": 0, "status": "active", "paused": False,
                "created_at": "2029-01-02T00:00:00",
            },
        }
    }
    jobs_file.write_text(json.dumps(data), encoding="utf-8")

    plugin = AgentCronPlugin()
    plugin.initialize(db=DbHandle(cron_db), storage_root=str(tmp_path))

    row1 = _row(cron_db, "j1")
    row2 = _row(cron_db, "j2")
    assert row1["chat_id"] == 10 and row1["user_id"] == 20
    assert row1["schedule_type"] == "interval"
    assert row1["interval_seconds"] == 7200
    assert row1["reply_to_message_id"] == 5
    assert row2["schedule_type"] == "weekly"
    assert row2["hour"] == 10 and row2["weekday"] == 0

    assert not jobs_file.exists()
    assert (tmp_path / "agent_cron_jobs.json.migrated").exists()


def test_import_skips_when_table_not_empty(tmp_path, cron_db):
    _insert_job(cron_db, id="existing")
    jobs_file = tmp_path / "agent_cron_jobs.json"
    jobs_file.write_text(json.dumps({"s": {"j1": {"id": "j1", "chat_id": 1, "user_id": 1}}}), encoding="utf-8")

    plugin = AgentCronPlugin()
    plugin.initialize(db=DbHandle(cron_db), storage_root=str(tmp_path))

    # N2: table already had data, so the JSON is never re-imported — but this
    # setup is indistinguishable from a crash between the import commit and
    # the rename, so the rename to .migrated must still complete (idempotent).
    assert not jobs_file.exists()
    assert (tmp_path / "agent_cron_jobs.json.migrated").exists()
    assert _row(cron_db, "j1") is None
    assert _row(cron_db, "existing") is not None


def test_import_running_status_normalized_to_active(tmp_path, cron_db):
    jobs_file = tmp_path / "agent_cron_jobs.json"
    data = {"s": {"j1": {
        "id": "j1", "chat_id": 1, "user_id": 1, "schedule": "daily at 09:00", "prompt": "p",
        "schedule_type": "daily", "hour": 9, "minute": 0, "status": "running", "paused": False,
    }}}
    jobs_file.write_text(json.dumps(data), encoding="utf-8")

    plugin = AgentCronPlugin()
    plugin.initialize(db=DbHandle(cron_db), storage_root=str(tmp_path))

    row = _row(cron_db, "j1")
    assert row["status"] == "active"
    assert row["locked_at"] is None


# --- claim: mass (checker) path ---------------------------------------------------


def test_claim_due_jobs_locks_and_returns_only_due(cron_db):
    plugin = AgentCronPlugin()
    now = datetime.now()
    due_iso = (now - timedelta(minutes=1)).isoformat(timespec="seconds")
    future_iso = (now + timedelta(hours=1)).isoformat(timespec="seconds")
    _insert_job(cron_db, id="due", next_run_at=due_iso)
    _insert_job(cron_db, id="future", next_run_at=future_iso)
    _insert_job(cron_db, id="paused-due", next_run_at=due_iso, paused=1)

    now_iso = now.isoformat(timespec="seconds")
    lease_cutoff_iso = (now - timedelta(seconds=AGENT_CRON_JOB_LEASE_SECONDS)).isoformat(timespec="seconds")
    claimed = plugin._claim_due_jobs_sync(cron_db, now_iso, lease_cutoff_iso, "worker-1")

    assert [job["id"] for job in claimed] == ["due"]
    row = _row(cron_db, "due")
    assert row["status"] == "running"
    assert row["locked_at"] == now_iso
    assert row["locked_by"] == "worker-1"


def test_concurrent_claim_only_one_thread_wins(cron_db):
    plugin = AgentCronPlugin()
    now = datetime.now()
    due_iso = (now - timedelta(minutes=1)).isoformat(timespec="seconds")
    _insert_job(cron_db, id="due", next_run_at=due_iso)

    now_iso = now.isoformat(timespec="seconds")
    lease_cutoff_iso = (now - timedelta(seconds=AGENT_CRON_JOB_LEASE_SECONDS)).isoformat(timespec="seconds")

    results = []
    results_lock = threading.Lock()

    def worker(idx):
        claimed = plugin._claim_due_jobs_sync(cron_db, now_iso, lease_cutoff_iso, f"worker-{idx}")
        with results_lock:
            results.append(claimed)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    non_empty = [r for r in results if r]
    empty = [r for r in results if not r]
    assert len(non_empty) == 1
    assert len(empty) == 1

    with cron_db.get_connection() as conn:
        rows = conn.execute("SELECT * FROM agent_cron_jobs").fetchall()
    assert len(rows) == 1
    assert dict(rows[0])["locked_by"] in {"worker-0", "worker-1"}


def test_stale_lease_job_reclaimable(cron_db):
    plugin = AgentCronPlugin()
    now = datetime.now()
    stale_locked_at = (now - timedelta(seconds=AGENT_CRON_JOB_LEASE_SECONDS + 10)).isoformat(timespec="seconds")
    due_iso = (now - timedelta(minutes=1)).isoformat(timespec="seconds")
    _insert_job(cron_db, id="stale", next_run_at=due_iso, status="running", locked_at=stale_locked_at, locked_by="old-worker")

    now_iso = now.isoformat(timespec="seconds")
    lease_cutoff_iso = (now - timedelta(seconds=AGENT_CRON_JOB_LEASE_SECONDS)).isoformat(timespec="seconds")
    claimed = plugin._claim_due_jobs_sync(cron_db, now_iso, lease_cutoff_iso, "new-worker")

    assert [job["id"] for job in claimed] == ["stale"]


def test_fresh_lease_job_not_reclaimable(cron_db):
    plugin = AgentCronPlugin()
    now = datetime.now()
    fresh_locked_at = (now - timedelta(seconds=5)).isoformat(timespec="seconds")
    due_iso = (now - timedelta(minutes=1)).isoformat(timespec="seconds")
    _insert_job(cron_db, id="fresh", next_run_at=due_iso, status="running", locked_at=fresh_locked_at, locked_by="active-worker")

    now_iso = now.isoformat(timespec="seconds")
    lease_cutoff_iso = (now - timedelta(seconds=AGENT_CRON_JOB_LEASE_SECONDS)).isoformat(timespec="seconds")
    claimed = plugin._claim_due_jobs_sync(cron_db, now_iso, lease_cutoff_iso, "new-worker")

    assert claimed == []


def test_claim_due_jobs_excludes_ids_already_running(cron_db):
    """W1: a due job whose id is already tracked as in-flight (e.g. a manual
    /cron run queued just before this tick) must not be claimed by the bulk
    checker path — claiming it here would leave it locked but never run."""
    plugin = AgentCronPlugin()
    now = datetime.now()
    due_iso = (now - timedelta(minutes=1)).isoformat(timespec="seconds")
    _insert_job(cron_db, id="due", next_run_at=due_iso)

    now_iso = now.isoformat(timespec="seconds")
    lease_cutoff_iso = (now - timedelta(seconds=AGENT_CRON_JOB_LEASE_SECONDS)).isoformat(timespec="seconds")
    claimed = plugin._claim_due_jobs_sync(
        cron_db, now_iso, lease_cutoff_iso, "worker-1", exclude_ids=frozenset({"due"})
    )

    assert claimed == []
    row = _row(cron_db, "due")
    assert row["status"] == "active"
    assert row["locked_at"] is None


@pytest.mark.asyncio
async def test_check_due_jobs_does_not_claim_job_already_in_running_tasks(cron_db):
    """W1 integration: _check_due_jobs must pass the current _running_tasks ids
    through to the claim, so a manually-queued job isn't claimed-then-skipped."""
    plugin = AgentCronPlugin()
    plugin.db_handle = DbHandle(cron_db)
    now = datetime.now()
    due_iso = (now - timedelta(minutes=1)).isoformat(timespec="seconds")
    _insert_job(cron_db, id="due", next_run_at=due_iso)
    plugin._running_tasks["due"] = object()

    await plugin._check_due_jobs(SimpleNamespace())

    row = _row(cron_db, "due")
    assert row["status"] == "active"
    assert row["locked_at"] is None


# --- claim: manual (/cron run) path ---------------------------------------------------


def test_manual_run_claims_ignoring_paused(cron_db):
    plugin = AgentCronPlugin()
    now = datetime.now()
    due_iso = (now - timedelta(minutes=1)).isoformat(timespec="seconds")
    _insert_job(cron_db, id="paused", scope="s", next_run_at=due_iso, paused=1)

    now_iso = now.isoformat(timespec="seconds")
    lease_cutoff_iso = (now - timedelta(seconds=AGENT_CRON_JOB_LEASE_SECONDS)).isoformat(timespec="seconds")
    claimed = plugin._claim_job_by_id_sync(cron_db, "paused", "s", now_iso, lease_cutoff_iso, "worker-1")

    assert claimed is not None
    assert claimed["id"] == "paused"


def test_manual_run_respects_scope_isolation(cron_db):
    plugin = AgentCronPlugin()
    _insert_job(cron_db, id="j1", scope="a")

    now_iso = datetime.now().isoformat(timespec="seconds")
    lease_cutoff_iso = (datetime.now() - timedelta(seconds=AGENT_CRON_JOB_LEASE_SECONDS)).isoformat(timespec="seconds")
    claimed = plugin._claim_job_by_id_sync(cron_db, "j1", "b", now_iso, lease_cutoff_iso, "worker-1")

    assert claimed is None


# --- integration: _run_job persists next_run_at ---------------------------------------------------


class _FakeHelper:
    def __init__(self):
        self.config = {}

    async def get_chat_response(self, **kwargs):
        return "cron result", 5


class _FakeBot:
    async def send_message(self, **kwargs):
        return SimpleNamespace(message_id=1)

    async def _post(self, endpoint, data=None, **kwargs):
        return SimpleNamespace(message_id=1)


@pytest.mark.asyncio
async def test_advance_job_persists_next_run_at_after_success(cron_db):
    plugin = AgentCronPlugin()
    plugin.db_handle = DbHandle(cron_db)
    plugin.openai = _FakeHelper()
    parsed = plugin._parse_schedule("every 2 hours")
    job = await plugin._create_job(chat_id=100, user_id=42, schedule="every 2 hours", prompt="check", parsed=parsed)

    await plugin._run_job(_FakeBot(), job["scope"], job["id"], manual=False)

    row = _row(cron_db, job["id"])
    assert row["status"] == "active"
    assert row["locked_at"] is None
    assert row["locked_by"] is None
    next_run = datetime.fromisoformat(row["next_run_at"])
    assert next_run > datetime.now()
