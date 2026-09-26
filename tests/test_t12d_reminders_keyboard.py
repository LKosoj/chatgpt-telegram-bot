"""T12d D8: the reminders-list keyboard (one view/delete button row per
reminder + a close-menu row) is built by an identical loop in
RemindersPlugin.handle_prompt_constructor (the /list_reminders command) and
in the "delete" branch of handle_reminder_callback (refreshing the list after
a deletion). No existing test covered either keyboard directly; this pins
down that both code paths produce the same button structure for the same
underlying reminder set, so the duplication can be safely collapsed into a
shared _build_reminders_keyboard helper.
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.database import Database
from bot.plugins.db_handle import DbHandle
from bot.plugins.reminders import RemindersPlugin


@pytest.fixture()
def reminders_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    Database._reset_singleton()
    db = Database()
    with db.get_connection() as conn:
        for stmt in RemindersPlugin().register_schema():
            conn.execute(stmt)
    yield db
    Database._reset_singleton()


def _make_plugin(db) -> RemindersPlugin:
    plugin = RemindersPlugin.__new__(RemindersPlugin)
    plugin.db_handle = DbHandle(db)
    plugin.openai = None
    plugin.bot = None
    return plugin


async def _insert_reminder(db_handle, **overrides):
    defaults = dict(
        id="r1", owner_id="42", target_chat_id="42", time="2030-01-01T10:00:00",
        fire_at_utc=None, message="ping", integration="telegram", reply_to_message_id=None,
        send_attempts=0, status="pending", locked_at=None, locked_by=None,
        created_at="2026-01-01T00:00:00",
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


class FakeMessage:
    def __init__(self, user_id):
        self.from_user = SimpleNamespace(id=user_id)
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append({"text": text, **kwargs})


class FakeQuery:
    def __init__(self, user_id, data):
        self.from_user = SimpleNamespace(id=user_id)
        self.data = data
        self.answers = []
        self.edits = []

    async def answer(self, *args, **kwargs):
        self.answers.append((args, kwargs))

    async def edit_message_text(self, text, **kwargs):
        self.edits.append({"text": text, **kwargs})


def _button_texts(markup):
    # Compares only button labels, not callback_data: the two scenarios below
    # use distinct reminder ids (SQLite PK is global), so callback_data
    # necessarily differs even when the keyboard *shape* should match.
    return [[b.text for b in row] for row in markup.inline_keyboard]


@pytest.mark.asyncio
async def test_prompt_constructor_and_delete_callback_agree_on_keyboard(reminders_db):
    # Scenario A: exactly r1 and r2 exist -> handle_prompt_constructor's keyboard.
    plugin_a = _make_plugin(reminders_db)
    await _insert_reminder(plugin_a.db_handle, id="r1", owner_id="42", message="first")
    await _insert_reminder(
        plugin_a.db_handle, id="r2", owner_id="42", message="second", created_at="2026-01-01T00:00:01",
    )

    message = FakeMessage(user_id=42)
    await plugin_a.handle_prompt_constructor(SimpleNamespace(message=message), context=None)
    keyboard_from_list = _button_texts(message.replies[0]["reply_markup"])

    # Scenario B: r1, r2, r3 exist for a different owner; delete r3 via the
    # callback -> the refreshed keyboard should cover the same remaining
    # (r1, r2) content as scenario A.
    plugin_b = _make_plugin(reminders_db)
    await _insert_reminder(plugin_b.db_handle, id="r1b", owner_id="43", message="first")
    await _insert_reminder(
        plugin_b.db_handle, id="r2b", owner_id="43", message="second", created_at="2026-01-01T00:00:01",
    )
    await _insert_reminder(
        plugin_b.db_handle, id="r3b", owner_id="43", message="third", created_at="2026-01-01T00:00:02",
    )

    query = FakeQuery(user_id=43, data="reminder:delete:r3b")
    await plugin_b.handle_reminder_callback(SimpleNamespace(callback_query=query), context=None)
    keyboard_after_delete = _button_texts(query.edits[0]["reply_markup"])

    assert keyboard_from_list == keyboard_after_delete
