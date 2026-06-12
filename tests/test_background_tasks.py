"""Stage 2 — RemindersPlugin migrated to BackgroundTask framework."""
import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from bot.plugin_manager import PluginManager
from bot.plugins.background import BackgroundTask
from bot.plugins.reminders import RemindersPlugin


def _make_plugin(tmp_path: Path) -> RemindersPlugin:
    plugin = RemindersPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    return plugin


def _seed_reminder(plugin: RemindersPlugin, user_id: str, when: datetime,
                   message: str = "ping", reminder_id: str = "r1") -> None:
    plugin.reminders.setdefault(user_id, {})[reminder_id] = {
        "id": reminder_id,
        "user_id": user_id,
        "time": when.isoformat(),
        "message": message,
        "integration": "telegram",
        "reply_to_message_id": None,
    }
    plugin.save_reminders()


def test_reminders_plugin_declares_background_task(tmp_path):
    plugin = _make_plugin(tmp_path)
    tasks = plugin.get_background_tasks()
    assert len(tasks) == 1
    t = tasks[0]
    assert isinstance(t, BackgroundTask)
    assert t.name == "check"
    assert t.interval_seconds == 60.0
    assert callable(t.coroutine_factory)


async def test_reminders_tick_invokes_check_reminders_with_application_bot(tmp_path):
    plugin = _make_plugin(tmp_path)
    past = datetime.now() - timedelta(minutes=1)
    _seed_reminder(plugin, user_id="42", when=past)

    fake_bot = SimpleNamespace(send_message=AsyncMock())
    fake_app = SimpleNamespace(bot=fake_bot)
    await plugin._check_reminders_tick(application=fake_app)

    fake_bot.send_message.assert_called_once()
    # Past-due reminder consumed and persisted.
    assert plugin.reminders == {}


async def test_reminders_tick_skips_future_reminders(tmp_path):
    plugin = _make_plugin(tmp_path)
    future = datetime.now() + timedelta(hours=1)
    _seed_reminder(plugin, user_id="42", when=future)

    fake_bot = SimpleNamespace(send_message=AsyncMock())
    fake_app = SimpleNamespace(bot=fake_bot)
    await plugin._check_reminders_tick(application=fake_app)

    fake_bot.send_message.assert_not_called()
    assert "42" in plugin.reminders


# ---------------------------------------------------------------------------
# (4c) send-failure / poisoned-entry / bad-time tests
# ---------------------------------------------------------------------------

async def test_send_failure_increments_attempts_no_duplicate(tmp_path):
    """A failed send increments send_attempts and does NOT re-send on the next tick."""
    plugin = _make_plugin(tmp_path)
    past = datetime.now() - timedelta(minutes=1)
    _seed_reminder(plugin, user_id="42", when=past, reminder_id="r_fail")

    error_bot = SimpleNamespace(send_message=AsyncMock(side_effect=RuntimeError("network error")))
    await plugin.check_reminders(error_bot)

    # Reminder still present with attempts=1
    assert "42" in plugin.reminders
    assert "r_fail" in plugin.reminders["42"]
    assert plugin.reminders["42"]["r_fail"]["send_attempts"] == 1

    # Persisted — reload from disk and verify
    plugin2 = _make_plugin(tmp_path)
    assert plugin2.reminders["42"]["r_fail"]["send_attempts"] == 1

    # Second tick: attempts=2, no message sent
    await plugin.check_reminders(error_bot)
    assert plugin.reminders["42"]["r_fail"]["send_attempts"] == 2
    assert error_bot.send_message.call_count == 2  # tried twice total, never succeeded


async def test_poisoned_reminder_removed_after_max_attempts(tmp_path):
    """After _MAX_SEND_ATTEMPTS failures the reminder is deleted and not retried."""
    plugin = _make_plugin(tmp_path)
    past = datetime.now() - timedelta(minutes=1)
    _seed_reminder(plugin, user_id="7", when=past, reminder_id="r_poison")

    error_bot = SimpleNamespace(send_message=AsyncMock(side_effect=RuntimeError("always fails")))

    for _ in range(RemindersPlugin._MAX_SEND_ATTEMPTS):
        await plugin.check_reminders(error_bot)

    # Reminder must be gone after max attempts
    assert "7" not in plugin.reminders or "r_poison" not in plugin.reminders.get("7", {})

    # Persisted on disk too
    plugin2 = _make_plugin(tmp_path)
    assert "7" not in plugin2.reminders or "r_poison" not in plugin2.reminders.get("7", {})

    # No further send attempts on next tick
    call_count_before = error_bot.send_message.call_count
    await plugin.check_reminders(error_bot)
    assert error_bot.send_message.call_count == call_count_before


async def test_bad_time_does_not_kill_tick(tmp_path):
    """A reminder with a corrupt 'time' field is skipped; other reminders still fire."""
    plugin = _make_plugin(tmp_path)
    past = datetime.now() - timedelta(minutes=1)
    _seed_reminder(plugin, user_id="99", when=past, reminder_id="r_good")

    # Inject a bad-time entry directly (bypassing _seed_reminder's isoformat)
    plugin.reminders.setdefault("99", {})["r_bad"] = {
        "id": "r_bad",
        "user_id": "99",
        "time": "NOT-A-DATE",
        "message": "corrupt",
        "integration": "telegram",
        "reply_to_message_id": None,
    }
    plugin.save_reminders()

    ok_bot = SimpleNamespace(send_message=AsyncMock())
    await plugin.check_reminders(ok_bot)

    # Good reminder fired
    ok_bot.send_message.assert_called_once()
    assert "r_good" not in plugin.reminders.get("99", {})

    # Bad entry still present (not deleted, not re-sent)
    assert "r_bad" in plugin.reminders.get("99", {})


async def test_successful_send_persists_and_failed_persists_attempts(tmp_path):
    """When one reminder succeeds and another fails, both states persist atomically."""
    plugin = _make_plugin(tmp_path)
    past = datetime.now() - timedelta(minutes=1)

    _seed_reminder(plugin, user_id="5", when=past, reminder_id="r_ok")
    # Add a second reminder that will fail
    plugin.reminders["5"]["r_fail"] = {
        "id": "r_fail",
        "user_id": "5",
        "time": past.isoformat(),
        "message": "will fail",
        "integration": "telegram",
        "reply_to_message_id": None,
    }
    plugin.save_reminders()

    call_count = 0

    async def selective_send(chat_id, text, reply_to_message_id=None):
        nonlocal call_count
        call_count += 1
        if "will fail" in text:
            raise RuntimeError("selective failure")

    mixed_bot = SimpleNamespace(send_message=AsyncMock(side_effect=selective_send))
    await plugin.check_reminders(mixed_bot)

    # r_ok gone, r_fail still here with attempts=1
    assert "r_ok" not in plugin.reminders.get("5", {})
    assert plugin.reminders["5"]["r_fail"]["send_attempts"] == 1

    # Reload from disk: both states persisted
    plugin2 = _make_plugin(tmp_path)
    assert "r_ok" not in plugin2.reminders.get("5", {})
    assert plugin2.reminders["5"]["r_fail"]["send_attempts"] == 1


def test_save_reminders_no_tmp_file_after_success(tmp_path):
    """save_reminders must not leave a .tmp file after a successful save."""
    plugin = _make_plugin(tmp_path)
    plugin.reminders = {"1": {"r1": {"id": "r1", "time": "2030-01-01T10:00:00",
                                      "message": "x", "integration": "telegram",
                                      "user_id": "1", "reply_to_message_id": None}}}
    plugin.save_reminders()

    tmp_file = plugin.reminders_file + ".tmp"
    assert not os.path.exists(tmp_file), ".tmp file must be removed after os.replace"
    assert os.path.exists(plugin.reminders_file)


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
