"""Stage 2 — RemindersPlugin migrated to BackgroundTask framework."""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
