"""Stage 2 — RemindersPlugin migrated to BackgroundTask framework."""
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.database import Database
from bot.plugin_manager import PluginManager
from bot.plugins.background import BackgroundTask
from bot.plugins.db_handle import DbHandle
from bot.plugins.reminders import RemindersPlugin


@pytest.fixture()
def reminders_db(tmp_path, monkeypatch):
    """Local fixture (mirrors cron_db / reminders_db elsewhere): temp SQLite DB
    with the reminders schema applied, for the reminders-only tests in this file."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    Database._reset_singleton()
    db = Database()
    with db.get_connection() as conn:
        for stmt in RemindersPlugin().register_schema():
            conn.execute(stmt)
    yield db
    Database._reset_singleton()


def _make_plugin(tmp_path: Path, db=None) -> RemindersPlugin:
    plugin = RemindersPlugin()
    plugin.initialize(storage_root=str(tmp_path), db=DbHandle(db) if db is not None else None)
    return plugin


async def _seed_reminder(plugin: RemindersPlugin, user_id: str, when: datetime,
                          message: str = "ping", reminder_id: str = "r1") -> None:
    await plugin.db_handle.execute(
        '''INSERT INTO reminders (id, owner_id, target_chat_id, time, fire_at_utc, message,
            integration, reply_to_message_id, send_attempts, status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
        (reminder_id, user_id, user_id, when.isoformat(), None, message, "telegram", None, 0,
         "pending", datetime.now().isoformat(timespec="seconds")),
    )


def test_reminders_plugin_declares_background_task(tmp_path):
    plugin = _make_plugin(tmp_path)
    tasks = plugin.get_background_tasks()
    assert len(tasks) == 1
    t = tasks[0]
    assert isinstance(t, BackgroundTask)
    assert t.name == "check"
    assert t.interval_seconds == 60.0
    assert callable(t.coroutine_factory)


async def test_reminders_tick_invokes_check_reminders_with_application_bot(tmp_path, reminders_db):
    plugin = _make_plugin(tmp_path, reminders_db)
    past = datetime.now() - timedelta(minutes=1)
    await _seed_reminder(plugin, user_id="42", when=past)

    fake_bot = SimpleNamespace(send_message=AsyncMock())
    fake_app = SimpleNamespace(bot=fake_bot)
    await plugin._check_reminders_tick(application=fake_app)

    fake_bot.send_message.assert_called_once()
    # Past-due reminder consumed and persisted (deleted after send).
    remaining = await plugin.db_handle.fetch_all("SELECT id FROM reminders")
    assert remaining == []


async def test_reminders_tick_skips_future_reminders(tmp_path, reminders_db):
    plugin = _make_plugin(tmp_path, reminders_db)
    future = datetime.now() + timedelta(hours=1)
    await _seed_reminder(plugin, user_id="42", when=future)

    fake_bot = SimpleNamespace(send_message=AsyncMock())
    fake_app = SimpleNamespace(bot=fake_bot)
    await plugin._check_reminders_tick(application=fake_app)

    fake_bot.send_message.assert_not_called()
    rows = await plugin.db_handle.fetch_all("SELECT id FROM reminders WHERE owner_id = ?", ("42",))
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# (4c) send-failure / poisoned-entry / bad-time tests
# ---------------------------------------------------------------------------

async def test_send_failure_increments_attempts_no_duplicate(tmp_path, reminders_db):
    """A failed send increments send_attempts and does NOT re-send on the next tick."""
    plugin = _make_plugin(tmp_path, reminders_db)
    past = datetime.now() - timedelta(minutes=1)
    await _seed_reminder(plugin, user_id="42", when=past, reminder_id="r_fail")

    error_bot = SimpleNamespace(send_message=AsyncMock(side_effect=RuntimeError("network error")))
    await plugin.check_reminders(error_bot)

    # Reminder still present with attempts=1
    row = await plugin.db_handle.fetch_one("SELECT send_attempts FROM reminders WHERE id = ?", ("r_fail",))
    assert row is not None
    assert row["send_attempts"] == 1

    # Persisted — reopen a plugin over the same DB and verify
    plugin2 = _make_plugin(tmp_path, reminders_db)
    row2 = await plugin2.db_handle.fetch_one("SELECT send_attempts FROM reminders WHERE id = ?", ("r_fail",))
    assert row2["send_attempts"] == 1

    # Second tick: attempts=2, no message sent
    await plugin.check_reminders(error_bot)
    row = await plugin.db_handle.fetch_one("SELECT send_attempts FROM reminders WHERE id = ?", ("r_fail",))
    assert row["send_attempts"] == 2
    assert error_bot.send_message.call_count == 2  # tried twice total, never succeeded


async def test_poisoned_reminder_removed_after_max_attempts(tmp_path, reminders_db):
    """After _MAX_SEND_ATTEMPTS failures the reminder is deleted and not retried."""
    plugin = _make_plugin(tmp_path, reminders_db)
    past = datetime.now() - timedelta(minutes=1)
    await _seed_reminder(plugin, user_id="7", when=past, reminder_id="r_poison")

    error_bot = SimpleNamespace(send_message=AsyncMock(side_effect=RuntimeError("always fails")))

    for _ in range(RemindersPlugin._MAX_SEND_ATTEMPTS):
        await plugin.check_reminders(error_bot)

    # Reminder must be gone after max attempts
    row = await plugin.db_handle.fetch_one("SELECT id FROM reminders WHERE id = ?", ("r_poison",))
    assert row is None

    # Persisted (same DB) too
    plugin2 = _make_plugin(tmp_path, reminders_db)
    row2 = await plugin2.db_handle.fetch_one("SELECT id FROM reminders WHERE id = ?", ("r_poison",))
    assert row2 is None

    # No further send attempts on next tick
    call_count_before = error_bot.send_message.call_count
    await plugin.check_reminders(error_bot)
    assert error_bot.send_message.call_count == call_count_before


async def test_bad_time_does_not_kill_tick(tmp_path, reminders_db):
    """A reminder with a corrupt 'time' field is skipped; other reminders still fire."""
    plugin = _make_plugin(tmp_path, reminders_db)
    past = datetime.now() - timedelta(minutes=1)
    await _seed_reminder(plugin, user_id="99", when=past, reminder_id="r_good")

    # Inject a bad-time entry directly (bypassing _seed_reminder's isoformat)
    await plugin.db_handle.execute(
        '''INSERT INTO reminders (id, owner_id, target_chat_id, time, fire_at_utc, message,
            integration, reply_to_message_id, send_attempts, status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
        ("r_bad", "99", "99", "NOT-A-DATE", None, "corrupt", "telegram", None, 0, "pending",
         datetime.now().isoformat(timespec="seconds")),
    )

    ok_bot = SimpleNamespace(send_message=AsyncMock())
    await plugin.check_reminders(ok_bot)

    # Good reminder fired
    ok_bot.send_message.assert_called_once()
    good_row = await plugin.db_handle.fetch_one("SELECT id FROM reminders WHERE id = ?", ("r_good",))
    assert good_row is None

    # Bad entry still present (not deleted, not re-sent)
    bad_row = await plugin.db_handle.fetch_one("SELECT id FROM reminders WHERE id = ?", ("r_bad",))
    assert bad_row is not None


async def test_successful_send_persists_and_failed_persists_attempts(tmp_path, reminders_db):
    """When one reminder succeeds and another fails, both states persist atomically."""
    plugin = _make_plugin(tmp_path, reminders_db)
    past = datetime.now() - timedelta(minutes=1)

    await _seed_reminder(plugin, user_id="5", when=past, reminder_id="r_ok")
    # Add a second reminder that will fail
    await plugin.db_handle.execute(
        '''INSERT INTO reminders (id, owner_id, target_chat_id, time, fire_at_utc, message,
            integration, reply_to_message_id, send_attempts, status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
        ("r_fail", "5", "5", past.isoformat(), None, "will fail", "telegram", None, 0, "pending",
         datetime.now().isoformat(timespec="seconds")),
    )

    call_count = 0

    async def selective_send(chat_id, text, reply_to_message_id=None):
        nonlocal call_count
        call_count += 1
        if "will fail" in text:
            raise RuntimeError("selective failure")

    mixed_bot = SimpleNamespace(send_message=AsyncMock(side_effect=selective_send))
    await plugin.check_reminders(mixed_bot)

    # r_ok gone, r_fail still here with attempts=1
    ok_row = await plugin.db_handle.fetch_one("SELECT id FROM reminders WHERE id = ?", ("r_ok",))
    assert ok_row is None
    fail_row = await plugin.db_handle.fetch_one("SELECT send_attempts FROM reminders WHERE id = ?", ("r_fail",))
    assert fail_row["send_attempts"] == 1

    # Reopen plugin over the same DB: both states persisted
    plugin2 = _make_plugin(tmp_path, reminders_db)
    ok_row2 = await plugin2.db_handle.fetch_one("SELECT id FROM reminders WHERE id = ?", ("r_ok",))
    assert ok_row2 is None
    fail_row2 = await plugin2.db_handle.fetch_one("SELECT send_attempts FROM reminders WHERE id = ?", ("r_fail",))
    assert fail_row2["send_attempts"] == 1


# ---------------------------------------------------------------------------

async def test_reminders_task_registers_and_stops_within_timeout(tmp_path):
    plugin_dir = tmp_path / "p"
    plugin_dir.mkdir()
    pm = PluginManager(config={"plugins": []}, plugins_directory=str(plugin_dir))
    pm.plugins["reminders"] = RemindersPlugin
    plugin = _make_plugin(tmp_path)
    pm.plugin_instances["reminders"] = plugin

    # Sanity: PluginManager resolves to the same pre-seeded instance.
    assert pm.get_plugin("reminders") is plugin

    fake_app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))

    await pm.start_background_tasks(fake_app)
    assert "reminders.check" in pm._background_tasks

    await pm.stop_background_tasks(timeout=2.0)
    assert pm._background_tasks == {}
