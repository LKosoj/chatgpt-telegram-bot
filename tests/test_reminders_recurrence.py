from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.database import Database
from bot.plugins.db_handle import DbHandle
from bot.plugins.reminders import RemindersPlugin
import bot.plugins.reminders as reminders_module


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=datetime(2030, 1, 31, 10, 1))

    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now.replace(tzinfo=timezone.utc).astimezone(tz) if tz else clock.now

    monkeypatch.setattr(reminders_module, "datetime", ClockDateTime)
    return clock


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "reminders.db"))
    Database._reset_singleton()
    db = Database()
    plugin = RemindersPlugin()
    with db.get_connection() as conn:
        for statement in plugin.register_schema():
            conn.execute(statement)
    plugin.initialize(db=DbHandle(db), storage_root=str(tmp_path))
    yield plugin
    Database._reset_singleton()


async def create(plugin, clock, **kwargs):
    return await plugin.execute(
        "set_reminder", None, user_id=42, chat_id=42,
        time=clock.now.replace(minute=0).strftime("%Y-%m-%d %H:%M"),
        current_time=clock.now.strftime("%Y-%m-%d %H:%M"),
        message="water", integration="telegram", **kwargs,
    )


async def rows(plugin):
    return await plugin.db_handle.fetch_all("SELECT * FROM reminders")


async def test_one_time_sent_marker_precedes_delete_and_survives_restart(plugin, clock, monkeypatch):
    await create(plugin, clock)
    helper = SimpleNamespace(send_message=AsyncMock())
    execute = plugin.db_handle.execute
    delete_statuses = []

    async def fail_delete(sql, params=()):
        if sql.strip().startswith("DELETE FROM reminders"):
            delete_statuses.append((await rows(plugin))[0]["status"])
            raise RuntimeError("delete unavailable")
        return await execute(sql, params)

    with monkeypatch.context() as patcher:
        patcher.setattr(plugin.db_handle, "execute", fail_delete)
        await plugin.check_reminders(helper)
        await plugin.check_reminders(helper)

    assert (await rows(plugin))[0]["status"] == "sent"
    assert delete_statuses == ["sent", "sent"]
    Database._reset_singleton()
    restarted = RemindersPlugin()
    restarted.initialize(db=DbHandle(Database()), storage_root=str(Path(plugin.reminders_file).parent))
    await restarted.check_reminders(helper)
    assert await rows(restarted) == []
    helper.send_message.assert_awaited_once()


async def test_one_time_deleted_after_success(plugin, clock):
    await create(plugin, clock)
    helper = SimpleNamespace(send_message=AsyncMock())
    await plugin.check_reminders(helper)
    assert await rows(plugin) == []
    await plugin.check_reminders(helper)
    helper.send_message.assert_awaited_once()


@pytest.mark.parametrize("frequency,step", [("daily", 1), ("weekly", 7)])
async def test_count_includes_first_send_and_deletes_after_last(plugin, clock, frequency, step):
    result = await create(plugin, clock, repeat_frequency=frequency, repeat_count=3)
    assert "error" not in result
    helper = SimpleNamespace(send_message=AsyncMock())
    for remaining in (2, 1):
        await plugin.check_reminders(helper)
        row, = await rows(plugin)
        assert row["repeat_remaining"] == remaining
        assert row["status"] == "pending"
        assert datetime.fromisoformat(row["time"]) == clock.now.replace(minute=0) + timedelta(days=step)
        clock.now += timedelta(days=step)
    await plugin.check_reminders(helper)
    assert await rows(plugin) == []
    assert helper.send_message.await_count == 3


@pytest.mark.parametrize("year,feb_day", [(2030, 28), (2032, 29)])
async def test_monthly_preserves_original_day_after_short_month(plugin, clock, year, feb_day):
    clock.now = datetime(year, 1, 31, 10, 1)
    await create(plugin, clock, repeat_frequency="monthly", repeat_count=3)
    helper = SimpleNamespace(send_message=AsyncMock())
    await plugin.check_reminders(helper)
    row, = await rows(plugin)
    assert row["time"] == f"{year}-02-{feb_day}T10:00:00"
    clock.now = datetime(year, 2, feb_day, 10, 1)
    await plugin.check_reminders(helper)
    row, = await rows(plugin)
    assert row["time"] == f"{year}-03-31T10:00:00"
    clock.now = datetime(year, 3, 31, 10, 1)
    await plugin.check_reminders(helper)
    assert await rows(plugin) == []


@pytest.mark.parametrize("count", [None, 10])
async def test_until_date_includes_last_day(plugin, clock, count):
    await create(plugin, clock, repeat_frequency="daily", repeat_until="2030-02-01", repeat_count=count)
    helper = SimpleNamespace(send_message=AsyncMock())
    await plugin.check_reminders(helper)
    assert len(await rows(plugin)) == 1
    clock.now = datetime(2030, 2, 1, 10, 1)
    await plugin.check_reminders(helper)
    assert await rows(plugin) == []
    assert helper.send_message.await_count == 2


async def test_count_and_until_stop_at_earlier_limit(plugin, clock):
    await create(plugin, clock, repeat_frequency="daily", repeat_count=1, repeat_until="2030-12-31")
    helper = SimpleNamespace(send_message=AsyncMock())
    await plugin.check_reminders(helper)
    assert await rows(plugin) == []
    helper.send_message.assert_awaited_once()


async def test_expired_until_date_does_not_send_after_downtime(plugin, clock):
    await create(plugin, clock, repeat_frequency="daily", repeat_until="2030-02-01")
    clock.now = datetime(2030, 2, 2, 10, 1)
    helper = SimpleNamespace(send_message=AsyncMock())
    await plugin.check_reminders(helper)
    assert await rows(plugin) == []
    helper.send_message.assert_not_awaited()


async def test_downtime_skips_missed_occurrences_without_consuming_send_count(plugin, clock):
    await create(plugin, clock, repeat_frequency="daily", repeat_count=3)
    clock.now = datetime(2030, 2, 5, 10, 1)
    helper = SimpleNamespace(send_message=AsyncMock())
    await plugin.check_reminders(helper)
    row, = await rows(plugin)
    assert row["time"] == "2030-02-06T10:00:00"
    assert row["repeat_remaining"] == 2
    await plugin.check_reminders(helper)
    helper.send_message.assert_awaited_once()


async def test_send_failure_does_not_consume_repeat_count(plugin, clock):
    await create(plugin, clock, repeat_frequency="daily", repeat_count=2)
    helper = SimpleNamespace(send_message=AsyncMock(side_effect=RuntimeError("offline")))
    await plugin.check_reminders(helper)
    row, = await rows(plugin)
    assert row["repeat_remaining"] == 2
    assert row["send_attempts"] == 1
    helper.send_message.side_effect = None
    await plugin.check_reminders(helper)
    row, = await rows(plugin)
    assert row["repeat_remaining"] == 1
    assert row["send_attempts"] == 0


@pytest.mark.parametrize("options", [
    {"repeat_frequency": "hourly"},
    {"repeat_frequency": "daily", "repeat_count": 0},
    {"repeat_frequency": "daily", "repeat_count": True},
    {"repeat_frequency": "daily", "repeat_count": 1.5},
    {"repeat_frequency": "daily", "repeat_until": "bad date"},
    {"repeat_frequency": "daily", "repeat_until": "2030-01-30"},
    {"repeat_count": 3},
    {"repeat_until": "2030-02-01"},
])
async def test_invalid_recurrence_is_rejected_without_creating_row(plugin, clock, options):
    result = await create(plugin, clock, **options)
    assert "error" in result
    assert await rows(plugin) == []


async def test_recurrence_visible_in_confirmation_list_and_buttons(plugin, clock):
    result = await create(plugin, clock, repeat_frequency="weekly", repeat_count=3, repeat_until="2030-02-28")
    listing = await plugin.execute("list_reminders", None, user_id=42)
    row, = await rows(plugin)
    button = plugin._build_reminders_keyboard([row]).inline_keyboard[0][0].text
    for text in (result["direct_result"]["value"], listing["direct_result"]["value"], button):
        assert "weekly" in text.lower()
        assert "3" in text
        assert "2030-02-28" in text


async def test_reschedule_failure_survives_restart_without_resending_or_double_decrement(plugin, clock, monkeypatch):
    await create(plugin, clock, repeat_frequency="daily", repeat_count=3)
    helper = SimpleNamespace(send_message=AsyncMock())
    execute = plugin.db_handle.execute

    async def fail_reschedule(sql, params=()):
        if sql.strip().startswith("UPDATE reminders SET time"):
            raise RuntimeError("reschedule unavailable")
        return await execute(sql, params)

    with monkeypatch.context() as patcher:
        patcher.setattr(plugin.db_handle, "execute", fail_reschedule)
        await plugin.check_reminders(helper)
        await plugin.check_reminders(helper)

    sent_row, = await rows(plugin)
    assert sent_row["status"] == "sent"
    assert sent_row["repeat_remaining"] == 3
    Database._reset_singleton()
    restarted = RemindersPlugin()
    restarted.initialize(db=DbHandle(Database()), storage_root=str(Path(plugin.reminders_file).parent))
    await restarted.check_reminders(helper)
    await restarted._finish_sent_reminder(sent_row)
    row, = await rows(restarted)
    assert row["status"] == "pending"
    assert row["repeat_remaining"] == 2
    assert row["time"] == "2030-02-01T10:00:00"
    helper.send_message.assert_awaited_once()


async def test_recurrence_keeps_user_local_time_and_utc_offset(plugin, clock):
    await plugin.execute(
        "set_reminder", None, user_id=42, chat_id=42,
        time="2030-01-31 13:00", current_time="2030-01-31 13:01",
        message="water", integration="telegram", repeat_frequency="daily", repeat_count=2,
    )
    helper = SimpleNamespace(send_message=AsyncMock())
    await plugin.check_reminders(helper)
    row, = await rows(plugin)
    assert row["time"] == "2030-02-01T13:00:00"
    assert row["fire_at_utc"] == "2030-02-01T10:00:00"
    clock.now = datetime(2030, 2, 1, 9, 59)
    await plugin.check_reminders(helper)
    helper.send_message.assert_awaited_once()
    clock.now = datetime(2030, 2, 1, 10, 0)
    await plugin.check_reminders(helper)
    assert helper.send_message.await_count == 2
    assert await rows(plugin) == []


async def test_monthly_without_limits_crosses_year_and_persists(plugin, clock):
    clock.now = datetime(2030, 12, 31, 10, 1)
    await create(plugin, clock, repeat_frequency="monthly")
    helper = SimpleNamespace(send_message=AsyncMock())
    restarted = RemindersPlugin()
    restarted.db_handle = plugin.db_handle
    await restarted.check_reminders(helper)
    row, = await rows(plugin)
    assert row["time"] == "2031-01-31T10:00:00"
    assert row["repeat_remaining"] is None
    assert row["repeat_until"] is None
    helper.send_message.assert_awaited_once()


async def test_legacy_schema_migration_is_idempotent_and_preserves_reminders(plugin, clock):
    def legacy_schema(db):
        with db.get_connection() as conn:
            conn.execute("DROP TABLE reminders")
            conn.execute('''CREATE TABLE reminders (
                id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, target_chat_id TEXT NOT NULL,
                time TEXT NOT NULL, fire_at_utc TEXT, message TEXT NOT NULL, integration TEXT NOT NULL,
                reply_to_message_id INTEGER, send_attempts INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending', locked_at TEXT, locked_by TEXT,
                created_at TEXT NOT NULL
            )''')
            conn.execute('''INSERT INTO reminders (id, owner_id, target_chat_id, time, message, integration, created_at)
                VALUES ('legacy', '42', '42', '2030-01-31T10:00:00', 'water', 'telegram', '2030-01-30')''')

    plugin.db_handle.run_sync_blocking(legacy_schema)
    for _ in range(2):
        restarted = RemindersPlugin()
        restarted.initialize(db=plugin.db_handle, storage_root=str(Path(plugin.reminders_file).parent))
    row, = await rows(plugin)
    assert row["id"] == "legacy"
    assert row["repeat_frequency"] is None
    helper = SimpleNamespace(send_message=AsyncMock())
    await restarted.check_reminders(helper)
    assert await rows(plugin) == []
    helper.send_message.assert_awaited_once()
