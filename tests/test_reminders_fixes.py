"""
Тесты для двух исправлений в RemindersPlugin:
  #7 — owner-keying в группах (chat_id != user_id)
  #5 — корректный UTC через current_time / fire_at_utc
"""
import sys
import os
import json
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# Гарантируем, что корень проекта в sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.plugins.reminders import RemindersPlugin


# ---------------------------------------------------------------------------
# Вспомогательные объекты
# ---------------------------------------------------------------------------

class FakeHelper:
    """Минимальный мок helper, перехватывает send_message."""
    def __init__(self):
        self.sent = []  # [(chat_id, text, reply_to_message_id)]

    async def send_message(self, chat_id, text, reply_to_message_id=None):
        self.sent.append((chat_id, text, reply_to_message_id))


def _make_plugin(tmp_path) -> RemindersPlugin:
    """Создаёт плагин с изолированным хранилищем в tmp_path."""
    plugin = RemindersPlugin.__new__(RemindersPlugin)
    plugin.reminders = {}
    plugin.reminders_file = str(tmp_path / "reminders.json")
    # Вместо реального localized_text возвращаем ключ + kwargs
    plugin.openai = None
    plugin.bot = None
    return plugin


# ---------------------------------------------------------------------------
# #7 — owner-keying в группах
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_group_set_reminder_keyed_by_user_id(tmp_path):
    """В групповом сценарии запись создаётся под owner_id=user_id, а не chat_id."""
    plugin = _make_plugin(tmp_path)
    with patch.object(plugin, "t", side_effect=lambda key, **kw: key):
        await plugin.execute(
            "set_reminder",
            helper=None,
            chat_id="-100999",   # групповой chat_id
            user_id="42",        # from_user.id
            time="2030-01-01 10:00",
            message="встреча",
            integration="telegram",
            current_time="2030-01-01 09:00",
        )

    # Запись должна быть под owner_id="42", а не под "-100999"
    assert "42" in plugin.reminders, "owner_id=user_id должен быть ключом верхнего уровня"
    assert "-100999" not in plugin.reminders, "chat_id группы не должен быть ключом"


@pytest.mark.asyncio
async def test_group_list_reminders_found_by_owner_id(tmp_path):
    """list_reminders по owner_id=user_id находит созданные в группе напоминания."""
    plugin = _make_plugin(tmp_path)
    with patch.object(plugin, "t", side_effect=lambda key, **kw: key):
        await plugin.execute(
            "set_reminder",
            helper=None,
            chat_id="-100999",
            user_id="42",
            time="2030-06-01 12:00",
            message="тест",
            integration="telegram",
            current_time="2030-06-01 11:00",
        )

        result = await plugin.execute(
            "list_reminders",
            helper=None,
            chat_id="-100999",
            user_id="42",
        )

    assert "direct_result" in result
    value = result["direct_result"]["value"]
    # Должно найти напоминание — возвращает не "reminders_none"
    assert "reminders_none" not in value or "тест" in value


@pytest.mark.asyncio
async def test_group_target_chat_id_is_group_chat(tmp_path):
    """target_chat_id в записи указывает на группу, а не на user_id."""
    plugin = _make_plugin(tmp_path)
    with patch.object(plugin, "t", side_effect=lambda key, **kw: key):
        await plugin.execute(
            "set_reminder",
            helper=None,
            chat_id="-100999",
            user_id="42",
            time="2030-01-01 10:00",
            message="встреча",
            integration="telegram",
            current_time="2030-01-01 09:00",
        )

    records = list(plugin.reminders["42"].values())
    assert len(records) == 1
    assert records[0]["target_chat_id"] == "-100999"
    assert records[0]["user_id"] == "42"


@pytest.mark.asyncio
async def test_group_send_reminder_uses_target_chat_id(tmp_path):
    """send_reminder шлёт в target_chat_id (группу), а не в user_id."""
    plugin = _make_plugin(tmp_path)
    helper = FakeHelper()

    reminder = {
        "id": "r1",
        "user_id": "42",
        "target_chat_id": "-100999",
        "time": "2030-01-01T10:00:00",
        "fire_at_utc": None,
        "message": "встреча",
        "integration": "telegram",
        "reply_to_message_id": None,
    }
    with patch.object(plugin, "t", return_value="notification"):
        await plugin.send_reminder(reminder, helper)

    assert len(helper.sent) == 1
    assert helper.sent[0][0] == "-100999", "должны слать в группу, а не в user_id"


# ---------------------------------------------------------------------------
# Back-compat: старые записи без target_chat_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backcompat_send_without_target_chat_id(tmp_path):
    """Старые записи без target_chat_id: шлём в user_id."""
    plugin = _make_plugin(tmp_path)
    helper = FakeHelper()

    reminder = {
        "id": "r_old",
        "user_id": "77",
        # target_chat_id отсутствует
        "time": "2020-01-01T10:00:00",
        "message": "старое",
        "integration": "telegram",
        "reply_to_message_id": None,
    }
    with patch.object(plugin, "t", return_value="notification"):
        await plugin.send_reminder(reminder, helper)

    assert len(helper.sent) == 1
    assert helper.sent[0][0] == "77"


# ---------------------------------------------------------------------------
# #5 — fire_at_utc и TZ
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fire_at_utc_computed_correctly(tmp_path):
    """fire_at_utc = target_local - (current_local - utc_now) = target_local - offset."""
    plugin = _make_plugin(tmp_path)

    # Симулируем UTC+3: current_time (локальное) = UTC + 3ч
    # utc_now фиксируем через patch, чтобы тест был детерминированным
    fake_utc_now = datetime(2030, 1, 1, 6, 0, 0)  # 06:00 UTC
    fake_local_now = datetime(2030, 1, 1, 9, 0, 0)  # 09:00 по UTC+3
    target_local = datetime(2030, 1, 1, 10, 0, 0)   # 10:00 по UTC+3 → 07:00 UTC

    with patch.object(plugin, "t", side_effect=lambda key, **kw: key):
        with patch("bot.plugins.reminders.datetime") as mock_dt:
            # datetime.strptime должен работать как обычно
            mock_dt.strptime = datetime.strptime
            mock_dt.now.return_value = datetime(2020, 1, 1, 0, 0, 0)  # для reminder_id
            # datetime.now(timezone.utc).replace(tzinfo=None) → fake_utc_now
            mock_dt.now.side_effect = lambda tz=None: (
                datetime(2030, 1, 1, 6, 0, 0, tzinfo=timezone.utc) if tz is not None
                else datetime(2020, 1, 1, 0, 0, 0)
            )

            await plugin.execute(
                "set_reminder",
                helper=None,
                chat_id="42",
                user_id="42",
                time=target_local.strftime("%Y-%m-%d %H:%M"),
                message="тест tz",
                integration="telegram",
                current_time=fake_local_now.strftime("%Y-%m-%d %H:%M"),
            )

    records = list(plugin.reminders["42"].values())
    assert len(records) == 1
    fire_at_utc_str = records[0]["fire_at_utc"]
    assert fire_at_utc_str is not None, "fire_at_utc должен быть вычислен"
    fire_at_utc = datetime.fromisoformat(fire_at_utc_str)
    # Ожидаем 07:00 UTC (10:00 - 3ч смещения)
    expected = datetime(2030, 1, 1, 7, 0, 0)
    assert fire_at_utc == expected, f"Ожидали {expected}, получили {fire_at_utc}"


@pytest.mark.asyncio
async def test_check_reminders_fires_by_utc(tmp_path):
    """check_reminders срабатывает для записей с fire_at_utc по UTC-времени."""
    plugin = _make_plugin(tmp_path)
    helper = FakeHelper()

    # Запись с fire_at_utc в прошлом (уже должна сработать)
    past_utc = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)).isoformat()
    plugin.reminders = {
        "42": {
            "r1": {
                "id": "r1",
                "user_id": "42",
                "target_chat_id": "42",
                "time": "2030-01-01T10:00:00",  # далёкое будущее (legacy-поле не используется)
                "fire_at_utc": past_utc,
                "message": "ping",
                "integration": "telegram",
                "reply_to_message_id": None,
            }
        }
    }

    with patch.object(plugin, "t", return_value="notification"):
        with patch.object(plugin, "save_reminders"):
            with patch.object(plugin, "load_reminders"):
                await plugin.check_reminders(helper)

    assert len(helper.sent) == 1, "Напоминание должно было сработать по fire_at_utc"
    # После срабатывания запись должна быть удалена
    assert "42" not in plugin.reminders or "r1" not in plugin.reminders.get("42", {})


@pytest.mark.asyncio
async def test_check_reminders_does_not_fire_future_utc(tmp_path):
    """check_reminders не срабатывает если fire_at_utc в будущем."""
    plugin = _make_plugin(tmp_path)
    helper = FakeHelper()

    future_utc = (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=2)).isoformat()
    plugin.reminders = {
        "42": {
            "r2": {
                "id": "r2",
                "user_id": "42",
                "target_chat_id": "42",
                "time": "2020-01-01T00:00:00",  # давнее прошлое (legacy-поле)
                "fire_at_utc": future_utc,
                "message": "не сейчас",
                "integration": "telegram",
                "reply_to_message_id": None,
            }
        }
    }

    with patch.object(plugin, "t", return_value="notification"):
        with patch.object(plugin, "save_reminders"):
            with patch.object(plugin, "load_reminders"):
                await plugin.check_reminders(helper)

    assert len(helper.sent) == 0, "Не должно срабатывать — fire_at_utc в будущем"


@pytest.mark.asyncio
async def test_check_reminders_legacy_path_no_fire_at_utc(tmp_path):
    """Legacy-записи без fire_at_utc: check_reminders использует наивное время по 'time'."""
    plugin = _make_plugin(tmp_path)
    helper = FakeHelper()

    # time в прошлом — должно сработать
    past = (datetime.now() - timedelta(minutes=10)).isoformat()
    plugin.reminders = {
        "99": {
            "r_legacy": {
                "id": "r_legacy",
                "user_id": "99",
                # нет target_chat_id, нет fire_at_utc
                "time": past,
                "message": "legacy ping",
                "integration": "telegram",
                "reply_to_message_id": None,
            }
        }
    }

    with patch.object(plugin, "t", return_value="notification"):
        with patch.object(plugin, "save_reminders"):
            with patch.object(plugin, "load_reminders"):
                await plugin.check_reminders(helper)

    assert len(helper.sent) == 1, "Legacy-путь должен сработать по 'time'"
    # fallback: chat_id = user_id для старых записей
    assert helper.sent[0][0] == "99"


@pytest.mark.asyncio
async def test_set_reminder_without_current_time_leaves_fire_at_utc_none(tmp_path):
    """Если current_time не передан, fire_at_utc остаётся None (деградация к legacy)."""
    plugin = _make_plugin(tmp_path)
    with patch.object(plugin, "t", side_effect=lambda key, **kw: key):
        await plugin.execute(
            "set_reminder",
            helper=None,
            chat_id="42",
            user_id="42",
            time="2030-01-01 10:00",
            message="без tz",
            integration="telegram",
            # current_time НЕ передаём
        )

    records = list(plugin.reminders.get("42", {}).values())
    assert records, "Запись должна быть создана"
    assert records[0]["fire_at_utc"] is None


@pytest.mark.asyncio
async def test_set_reminder_bad_current_time_leaves_fire_at_utc_none(tmp_path):
    """Если current_time не парсится, fire_at_utc остаётся None, исключения нет."""
    plugin = _make_plugin(tmp_path)
    with patch.object(plugin, "t", side_effect=lambda key, **kw: key):
        result = await plugin.execute(
            "set_reminder",
            helper=None,
            chat_id="42",
            user_id="42",
            time="2030-01-01 10:00",
            message="сломанный tz",
            integration="telegram",
            current_time="не дата вообще",
        )

    assert "direct_result" in result
    records = list(plugin.reminders.get("42", {}).values())
    assert records[0]["fire_at_utc"] is None


# ---------------------------------------------------------------------------
# Группы: delete_reminder тоже работает по owner_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_group_delete_reminder_by_owner_id(tmp_path):
    """delete_reminder ищет по owner_id=user_id, работает в группах."""
    plugin = _make_plugin(tmp_path)
    with patch.object(plugin, "t", side_effect=lambda key, **kw: key):
        # Создаём напоминание
        await plugin.execute(
            "set_reminder",
            helper=None,
            chat_id="-100999",
            user_id="42",
            time="2030-01-01 10:00",
            message="удалить",
            integration="telegram",
            current_time="2030-01-01 09:00",
        )
        # Получаем reminder_id
        reminder_id = next(iter(plugin.reminders["42"]))

        # Удаляем
        result = await plugin.execute(
            "delete_reminder",
            helper=None,
            chat_id="-100999",
            user_id="42",
            reminder_id=reminder_id,
        )

    assert "direct_result" in result
    # Должно быть удалено
    assert "42" not in plugin.reminders or reminder_id not in plugin.reminders.get("42", {})
