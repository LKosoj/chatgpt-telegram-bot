"""
Тесты для RemindersPlugin (SQLite-хранилище с атомарным захватом, T04):
  #7 — owner-keying в группах (chat_id != user_id)
  #5 — корректный UTC через current_time / fire_at_utc
  T04 — импорт из JSON, атомарный захват «отправить и пометить», истёкшая аренда
"""
import json
import os
import sys
import threading
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

# Гарантируем, что корень проекта в sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.database import Database
from bot.plugins.db_handle import DbHandle
from bot.plugins.reminders import REMINDER_LEASE_SECONDS, RemindersPlugin


# ---------------------------------------------------------------------------
# Вспомогательные объекты
# ---------------------------------------------------------------------------

class FakeHelper:
    """Минимальный мок helper, перехватывает send_message."""
    def __init__(self):
        self.sent = []  # [(chat_id, text, reply_to_message_id)]

    async def send_message(self, chat_id, text, reply_to_message_id=None):
        self.sent.append((chat_id, text, reply_to_message_id))


@pytest.fixture()
def reminders_db(tmp_path, monkeypatch):
    """Локальная фикстура (не трогает tests/conftest.py, вне владения задачи):
    временная SQLite БД со схемой reminders, по образцу agent_db в conftest.py."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    Database._reset_singleton()
    db = Database()
    with db.get_connection() as conn:
        for stmt in RemindersPlugin().register_schema():
            conn.execute(stmt)
    yield db
    Database._reset_singleton()


def _make_plugin(tmp_path, db) -> RemindersPlugin:
    """Создаёт плагин с изолированным хранилищем в tmp_path."""
    plugin = RemindersPlugin.__new__(RemindersPlugin)
    plugin.reminders_file = str(tmp_path / "reminders.json")
    plugin.db_handle = DbHandle(db)
    # Вместо реального localized_text возвращаем ключ + kwargs
    plugin.openai = None
    plugin.bot = None
    return plugin


async def _insert_reminder(db_handle, **overrides):
    defaults = dict(
        id="r1", owner_id="42", target_chat_id="42", time="2030-01-01T10:00:00",
        fire_at_utc=None, message="ping", integration="telegram", reply_to_message_id=None,
        send_attempts=0, status="pending", locked_at=None, locked_by=None,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    defaults.update(overrides)
    columns = (
        "id", "owner_id", "target_chat_id", "time", "fire_at_utc", "message", "integration",
        "reply_to_message_id", "send_attempts", "status", "locked_at", "locked_by", "created_at",
    )
    placeholders = ",".join("?" for _ in columns)
    await db_handle.execute(
        f"INSERT INTO reminders ({','.join(columns)}) VALUES ({placeholders})",
        tuple(defaults[c] for c in columns),
    )
    return defaults


# ---------------------------------------------------------------------------
# #7 — owner-keying в группах
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_group_set_reminder_keyed_by_user_id(tmp_path, reminders_db):
    """В групповом сценарии запись создаётся под owner_id=user_id, а не chat_id."""
    plugin = _make_plugin(tmp_path, reminders_db)
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
    rows_owner = await plugin.db_handle.fetch_all("SELECT * FROM reminders WHERE owner_id = ?", ("42",))
    rows_group = await plugin.db_handle.fetch_all("SELECT * FROM reminders WHERE owner_id = ?", ("-100999",))
    assert rows_owner, "owner_id=user_id должен быть ключом верхнего уровня"
    assert not rows_group, "chat_id группы не должен быть ключом"


@pytest.mark.asyncio
async def test_group_list_reminders_found_by_owner_id(tmp_path, reminders_db):
    """list_reminders по owner_id=user_id находит созданные в группе напоминания."""
    plugin = _make_plugin(tmp_path, reminders_db)
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
async def test_group_target_chat_id_is_group_chat(tmp_path, reminders_db):
    """target_chat_id в записи указывает на группу, а не на user_id."""
    plugin = _make_plugin(tmp_path, reminders_db)
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

    records = await plugin.db_handle.fetch_all("SELECT * FROM reminders WHERE owner_id = ?", ("42",))
    assert len(records) == 1
    assert records[0]["target_chat_id"] == "-100999"
    assert records[0]["owner_id"] == "42"


@pytest.mark.asyncio
async def test_group_send_reminder_uses_target_chat_id(tmp_path, reminders_db):
    """send_reminder шлёт в target_chat_id (группу), а не в user_id."""
    plugin = _make_plugin(tmp_path, reminders_db)
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
async def test_backcompat_send_without_target_chat_id(tmp_path, reminders_db):
    """Старые записи без target_chat_id: шлём в user_id."""
    plugin = _make_plugin(tmp_path, reminders_db)
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
async def test_fire_at_utc_computed_correctly(tmp_path, reminders_db):
    """fire_at_utc = target_local - (current_local - utc_now) = target_local - offset."""
    plugin = _make_plugin(tmp_path, reminders_db)

    # Симулируем UTC+3: current_time (локальное) = UTC + 3ч
    # utc_now фиксируем через mock_dt.now(...), чтобы тест был детерминированным
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

    records = await plugin.db_handle.fetch_all("SELECT * FROM reminders WHERE owner_id = ?", ("42",))
    assert len(records) == 1
    fire_at_utc_str = records[0]["fire_at_utc"]
    assert fire_at_utc_str is not None, "fire_at_utc должен быть вычислен"
    fire_at_utc = datetime.fromisoformat(fire_at_utc_str)
    # Ожидаем 07:00 UTC (10:00 - 3ч смещения)
    expected = datetime(2030, 1, 1, 7, 0, 0)
    assert fire_at_utc == expected, f"Ожидали {expected}, получили {fire_at_utc}"


@pytest.mark.asyncio
async def test_check_reminders_fires_by_utc(tmp_path, reminders_db):
    """check_reminders срабатывает для записей с fire_at_utc по UTC-времени."""
    plugin = _make_plugin(tmp_path, reminders_db)
    helper = FakeHelper()

    # Запись с fire_at_utc в прошлом (уже должна сработать)
    past_utc = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)).isoformat()
    await _insert_reminder(
        plugin.db_handle, id="r1", owner_id="42", target_chat_id="42",
        time="2030-01-01T10:00:00",  # далёкое будущее (legacy-поле не используется)
        fire_at_utc=past_utc, message="ping",
    )

    with patch.object(plugin, "t", return_value="notification"):
        await plugin.check_reminders(helper)

    assert len(helper.sent) == 1, "Напоминание должно было сработать по fire_at_utc"
    # После срабатывания запись должна быть удалена
    remaining = await plugin.db_handle.fetch_all("SELECT id FROM reminders WHERE id = ?", ("r1",))
    assert remaining == []


@pytest.mark.asyncio
async def test_check_reminders_does_not_fire_future_utc(tmp_path, reminders_db):
    """check_reminders не срабатывает если fire_at_utc в будущем."""
    plugin = _make_plugin(tmp_path, reminders_db)
    helper = FakeHelper()

    future_utc = (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=2)).isoformat()
    await _insert_reminder(
        plugin.db_handle, id="r2", owner_id="42", target_chat_id="42",
        time="2020-01-01T00:00:00",  # давнее прошлое (legacy-поле)
        fire_at_utc=future_utc, message="не сейчас",
    )

    with patch.object(plugin, "t", return_value="notification"):
        await plugin.check_reminders(helper)

    assert len(helper.sent) == 0, "Не должно срабатывать — fire_at_utc в будущем"


@pytest.mark.asyncio
async def test_check_reminders_legacy_path_no_fire_at_utc(tmp_path, reminders_db):
    """Legacy-записи без fire_at_utc: check_reminders использует наивное время по 'time'."""
    plugin = _make_plugin(tmp_path, reminders_db)
    helper = FakeHelper()

    # time в прошлом — должно сработать
    past = (datetime.now() - timedelta(minutes=10)).isoformat()
    await _insert_reminder(
        plugin.db_handle, id="r_legacy", owner_id="99", target_chat_id="99",
        time=past, fire_at_utc=None, message="legacy ping",
    )

    with patch.object(plugin, "t", return_value="notification"):
        await plugin.check_reminders(helper)

    assert len(helper.sent) == 1, "Legacy-путь должен сработать по 'time'"
    # fallback: chat_id = user_id для старых записей
    assert helper.sent[0][0] == "99"


@pytest.mark.asyncio
async def test_set_reminder_without_current_time_leaves_fire_at_utc_none(tmp_path, reminders_db):
    """Если current_time не передан, fire_at_utc остаётся None (деградация к legacy)."""
    plugin = _make_plugin(tmp_path, reminders_db)
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

    records = await plugin.db_handle.fetch_all("SELECT * FROM reminders WHERE owner_id = ?", ("42",))
    assert records, "Запись должна быть создана"
    assert records[0]["fire_at_utc"] is None


@pytest.mark.asyncio
async def test_set_reminder_bad_current_time_leaves_fire_at_utc_none(tmp_path, reminders_db):
    """Если current_time не парсится, fire_at_utc остаётся None, исключения нет."""
    plugin = _make_plugin(tmp_path, reminders_db)
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
    records = await plugin.db_handle.fetch_all("SELECT * FROM reminders WHERE owner_id = ?", ("42",))
    assert records[0]["fire_at_utc"] is None


# ---------------------------------------------------------------------------
# Группы: delete_reminder тоже работает по owner_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_group_delete_reminder_by_owner_id(tmp_path, reminders_db):
    """delete_reminder ищет по owner_id=user_id, работает в группах."""
    plugin = _make_plugin(tmp_path, reminders_db)
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
        rows = await plugin.db_handle.fetch_all("SELECT id FROM reminders WHERE owner_id = ?", ("42",))
        reminder_id = rows[0]["id"]

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
    remaining = await plugin.db_handle.fetch_all(
        "SELECT id FROM reminders WHERE owner_id = ? AND id = ?", ("42", reminder_id)
    )
    assert remaining == []


# ---------------------------------------------------------------------------
# T04 — JSON import
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_import_json_reminders_moves_file_and_preserves_fields(tmp_path, reminders_db):
    reminders_file = tmp_path / "reminders.json"
    data = {
        "42": {
            "r1": {
                "id": "r1", "user_id": "42", "target_chat_id": "-100999",
                "time": "2030-01-01T10:00:00", "fire_at_utc": "2030-01-01T07:00:00",
                "message": "встреча", "integration": "telegram", "reply_to_message_id": 5,
            }
        }
    }
    reminders_file.write_text(json.dumps(data), encoding="utf-8")

    plugin = RemindersPlugin()
    plugin.initialize(db=DbHandle(reminders_db), storage_root=str(tmp_path))

    rows = await plugin.db_handle.fetch_all("SELECT * FROM reminders WHERE id = ?", ("r1",))
    assert len(rows) == 1
    assert rows[0]["owner_id"] == "42"
    assert rows[0]["target_chat_id"] == "-100999"
    assert rows[0]["fire_at_utc"] == "2030-01-01T07:00:00"
    assert rows[0]["reply_to_message_id"] == 5

    assert not reminders_file.exists()
    assert (tmp_path / "reminders.json.migrated").exists()


@pytest.mark.asyncio
async def test_import_skips_when_table_not_empty(tmp_path, reminders_db):
    plugin = RemindersPlugin()
    plugin.initialize(db=DbHandle(reminders_db), storage_root=str(tmp_path))
    await _insert_reminder(plugin.db_handle, id="existing", owner_id="1", target_chat_id="1")

    reminders_file = tmp_path / "reminders.json"
    reminders_file.write_text(
        json.dumps({"2": {"r_new": {"id": "r_new", "user_id": "2", "time": "2030-01-01T10:00:00", "message": "m", "integration": "telegram"}}}),
        encoding="utf-8",
    )

    plugin2 = RemindersPlugin()
    plugin2.initialize(db=DbHandle(reminders_db), storage_root=str(tmp_path))

    # N2: table already had data, so the JSON is never re-imported — but this
    # setup is indistinguishable from a crash between the import commit and
    # the rename, so the rename to .migrated must still complete (idempotent).
    assert not reminders_file.exists()
    assert (tmp_path / "reminders.json.migrated").exists()
    rows = await plugin.db_handle.fetch_all("SELECT id FROM reminders WHERE id = ?", ("r_new",))
    assert rows == []


@pytest.mark.asyncio
async def test_import_skips_record_with_missing_time_valid_records_still_imported(tmp_path, reminders_db):
    """Legacy record without 'time' (and without fire_at_utc) must not be imported: an
    empty/invalid 'time' would sort lexicographically before any real ISO date in
    _claim_due_reminders_sync and fire immediately, instead of HEAD's behavior of never
    sending such a record."""
    reminders_file = tmp_path / "reminders.json"
    data = {
        "42": {
            "no_time": {
                "id": "no_time", "user_id": "42",
                "message": "без времени", "integration": "telegram",
            },
            "ok": {
                "id": "ok", "user_id": "42", "time": "2030-01-01T10:00:00",
                "message": "валидная", "integration": "telegram",
            },
        }
    }
    reminders_file.write_text(json.dumps(data), encoding="utf-8")

    plugin = RemindersPlugin()
    plugin.initialize(db=DbHandle(reminders_db), storage_root=str(tmp_path))

    rows = await plugin.db_handle.fetch_all("SELECT id FROM reminders")
    assert {r["id"] for r in rows} == {"ok"}

    # Never fires on any tick either — the record is simply absent from the table.
    helper = FakeHelper()
    await plugin.check_reminders(helper)
    assert helper.sent == []


# ---------------------------------------------------------------------------
# T04 — атомарный захват «отправить и пометить»
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_claim_reminder_only_one_thread_wins(tmp_path, reminders_db):
    plugin = _make_plugin(tmp_path, reminders_db)
    past_utc = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)).isoformat()
    await _insert_reminder(plugin.db_handle, id="r1", owner_id="1", target_chat_id="1", fire_at_utc=past_utc)

    now_local_iso = datetime.now().isoformat(timespec="seconds")
    now_utc_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    lease_cutoff_iso = (datetime.now() - timedelta(seconds=REMINDER_LEASE_SECONDS)).isoformat(timespec="seconds")

    results = []
    results_lock = threading.Lock()

    def worker(idx):
        claimed = plugin._claim_due_reminders_sync(
            reminders_db, now_local_iso, now_utc_iso, lease_cutoff_iso, f"worker-{idx}"
        )
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


@pytest.mark.asyncio
async def test_reminder_sent_exactly_once_under_claim(tmp_path, reminders_db):
    """Due-напоминание, check_reminders дважды подряд — send_message вызван ровно 1 раз."""
    plugin = _make_plugin(tmp_path, reminders_db)
    helper = FakeHelper()
    past_utc = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)).isoformat()
    await _insert_reminder(plugin.db_handle, id="r1", owner_id="1", target_chat_id="1", fire_at_utc=past_utc)

    with patch.object(plugin, "t", return_value="notification"):
        await plugin.check_reminders(helper)
        await plugin.check_reminders(helper)

    assert len(helper.sent) == 1


@pytest.mark.asyncio
async def test_stale_processing_lease_reclaimable(tmp_path, reminders_db):
    plugin = _make_plugin(tmp_path, reminders_db)
    stale_locked_at = (datetime.now() - timedelta(seconds=REMINDER_LEASE_SECONDS + 10)).isoformat(timespec="seconds")
    past_utc = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)).isoformat()
    await _insert_reminder(
        plugin.db_handle, id="r1", owner_id="1", target_chat_id="1", fire_at_utc=past_utc,
        status="processing", locked_at=stale_locked_at, locked_by="old-worker",
    )

    now_local_iso = datetime.now().isoformat(timespec="seconds")
    now_utc_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    lease_cutoff_iso = (datetime.now() - timedelta(seconds=REMINDER_LEASE_SECONDS)).isoformat(timespec="seconds")
    claimed = plugin._claim_due_reminders_sync(reminders_db, now_local_iso, now_utc_iso, lease_cutoff_iso, "new-worker")

    assert [r["id"] for r in claimed] == ["r1"]


@pytest.mark.asyncio
async def test_delete_failure_after_successful_send_does_not_resend(tmp_path, reminders_db):
    """W3: a DELETE failure right after a successful send must not be treated
    as a send failure — the reminder must not be sent twice on a later tick."""
    plugin = _make_plugin(tmp_path, reminders_db)
    helper = FakeHelper()
    past_utc = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)).isoformat()
    await _insert_reminder(plugin.db_handle, id="r1", owner_id="1", target_chat_id="1", fire_at_utc=past_utc)

    real_execute = plugin.db_handle.execute
    state = {"failed_once": False}

    async def flaky_execute(sql, params=()):
        if not state["failed_once"] and sql.strip().startswith("DELETE FROM reminders"):
            state["failed_once"] = True
            raise RuntimeError("simulated transient DELETE failure")
        return await real_execute(sql, params)

    with patch.object(plugin, "t", return_value="notification"):
        with patch.object(plugin.db_handle, "execute", side_effect=flaky_execute):
            await plugin.check_reminders(helper)

        assert len(helper.sent) == 1, "send must succeed even though the follow-up DELETE fails"

        row = await plugin.db_handle.fetch_one("SELECT * FROM reminders WHERE id = ?", ("r1",))
        assert row is not None, "row must survive a DELETE failure so it can be cleaned up later"
        assert row["send_attempts"] == 0, (
            "a DELETE failure after a successful send must not be counted as a send failure"
        )

        # A later tick must retry only the deletion, never resend.
        await plugin.check_reminders(helper)

    assert len(helper.sent) == 1, "reminder must not be sent twice because of a transient DELETE failure"
    remaining = await plugin.db_handle.fetch_all("SELECT id FROM reminders WHERE id = ?", ("r1",))
    assert remaining == [], "row should be deleted once the retry DELETE succeeds"


@pytest.mark.asyncio
async def test_fresh_processing_lease_not_reclaimable(tmp_path, reminders_db):
    plugin = _make_plugin(tmp_path, reminders_db)
    fresh_locked_at = (datetime.now() - timedelta(seconds=5)).isoformat(timespec="seconds")
    past_utc = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)).isoformat()
    await _insert_reminder(
        plugin.db_handle, id="r1", owner_id="1", target_chat_id="1", fire_at_utc=past_utc,
        status="processing", locked_at=fresh_locked_at, locked_by="active-worker",
    )

    now_local_iso = datetime.now().isoformat(timespec="seconds")
    now_utc_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    lease_cutoff_iso = (datetime.now() - timedelta(seconds=REMINDER_LEASE_SECONDS)).isoformat(timespec="seconds")
    claimed = plugin._claim_due_reminders_sync(reminders_db, now_local_iso, now_utc_iso, lease_cutoff_iso, "new-worker")

    assert claimed == []


@pytest.mark.asyncio
async def test_list_reminders_excludes_sent_status(tmp_path, reminders_db):
    """W4: a 'sent' row (DELETE failed right after a successful send) must not
    show up in list_reminders — it already fired and must not look pending."""
    plugin = _make_plugin(tmp_path, reminders_db)
    await _insert_reminder(plugin.db_handle, id="r1", owner_id="42", target_chat_id="42", status="pending")
    await _insert_reminder(plugin.db_handle, id="r2", owner_id="42", target_chat_id="42", status="sent")

    with patch.object(plugin, "t", side_effect=lambda key, **kw: key):
        result = await plugin.execute("list_reminders", helper=None, user_id="42")

    value = result["direct_result"]["value"]
    assert "r1" in value, "pending reminder must still be listed"
    assert "r2" not in value, "sent reminder must be excluded from the listing"
