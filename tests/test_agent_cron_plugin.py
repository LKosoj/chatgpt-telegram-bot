import importlib.util
import importlib.machinery
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.database import Database
from bot.plugins.agent_cron import AgentCronPlugin
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


class FakeBot:
    def __init__(self):
        self.messages = []
        self.posts = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.messages))

    async def _post(self, endpoint, data=None, **kwargs):
        self.posts.append((endpoint, data, kwargs))
        return SimpleNamespace(message_id=len(self.posts))


class FakeHelper:
    def __init__(self):
        self.requests = []
        self.config = {}

    async def get_chat_response(self, **kwargs):
        self.requests.append(kwargs)
        return "cron result", 5


def test_agent_cron_parses_supported_natural_schedules(tmp_path):
    plugin = AgentCronPlugin()
    plugin.initialize(storage_root=str(tmp_path))

    once = plugin._parse_schedule("in 10 minutes")
    daily = plugin._parse_schedule("daily at 09:30")
    weekly = plugin._parse_schedule("weekly monday at 10:00")

    assert once["schedule_type"] == "once"
    assert daily["schedule_type"] == "daily"
    assert daily["hour"] == 9
    assert daily["minute"] == 30
    assert weekly["schedule_type"] == "weekly"
    assert weekly["weekday"] == 0


@pytest.mark.asyncio
async def test_agent_cron_manual_run_delivers_result(tmp_path, cron_db):
    plugin = AgentCronPlugin()
    helper = FakeHelper()
    plugin.initialize(openai=helper, db=DbHandle(cron_db), storage_root=str(tmp_path))
    bot = FakeBot()
    parsed = plugin._parse_schedule("daily at 09:30")
    job = await plugin._create_job(
        chat_id=100,
        user_id=42,
        schedule="daily at 09:30",
        prompt="make a brief",
        parsed=parsed,
        reply_to_message_id=77,
    )

    await plugin._run_job(bot, job["scope"], job["id"], manual=True)

    stored = await plugin.db_handle.fetch_one("SELECT * FROM agent_cron_jobs WHERE id = ?", (job["id"],))
    assert stored["status"] == "active"
    assert stored["last_tokens"] == 5
    assert helper.requests[0]["chat_id"] == 100
    assert helper.requests[0]["request_context"].autonomous is True
    assert bot.messages[0]["chat_id"] == 100
    assert "Cron job" in bot.messages[0]["text"]
    assert "cron result" in bot.messages[0]["text"]


@pytest.mark.asyncio
async def test_agent_cron_failure_uses_rich_config(tmp_path, cron_db):
    class FailingHelper(FakeHelper):
        def __init__(self):
            super().__init__()
            self.config["telegram_rich_messages"] = "auto"

        async def get_chat_response(self, **kwargs):
            self.requests.append(kwargs)
            raise RuntimeError("cron failed")

    plugin = AgentCronPlugin()
    helper = FailingHelper()
    plugin.initialize(openai=helper, db=DbHandle(cron_db), storage_root=str(tmp_path))
    bot = FakeBot()
    parsed = plugin._parse_schedule("daily at 09:30")
    job = await plugin._create_job(
        chat_id=100,
        user_id=42,
        schedule="daily at 09:30",
        prompt="make a brief",
        parsed=parsed,
        reply_to_message_id=77,
    )

    await plugin._run_job(bot, job["scope"], job["id"], manual=True)

    stored = await plugin.db_handle.fetch_one("SELECT * FROM agent_cron_jobs WHERE id = ?", (job["id"],))
    assert stored["status"] == "failed"
    assert bot.messages == []
    assert bot.posts == [
        (
            "sendRichMessage",
            {
                "chat_id": 100,
                "rich_message": {
                    "markdown": f"Cron job `{job['id']}` failed: cron failed"
                },
                "reply_parameters": {"message_id": 77},
            },
            {"api_kwargs": None},
        )
    ]


# --- Job deleted mid-run must not send a message or dispatch the hook (E1) ---


@pytest.mark.asyncio
async def test_run_job_deleted_during_run_skips_completion_message_and_hook(tmp_path, cron_db, monkeypatch):
    """/cron remove racing a long-running job must not post a completion message
    or dispatch the autonomous-response hook once the row is gone."""
    monkeypatch.setenv("HINDSIGHT_AUTONOMOUS_CAPTURE_ENABLED", "true")

    class DeletingHelper(FakeHelper):
        def __init__(self, db_handle):
            super().__init__()
            self.plugin_manager = SimpleNamespace(dispatch_observe=AsyncMock())
            self._db_handle = db_handle

        async def get_chat_response(self, **kwargs):
            self.requests.append(kwargs)
            job_id = kwargs["request_id"][len("agent_cron_"):]
            await self._db_handle.execute("DELETE FROM agent_cron_jobs WHERE id = ?", (job_id,))
            return "cron result", 5

    plugin = AgentCronPlugin()
    db_handle = DbHandle(cron_db)
    helper = DeletingHelper(db_handle)
    plugin.initialize(openai=helper, db=db_handle, storage_root=str(tmp_path))
    bot = FakeBot()
    parsed = plugin._parse_schedule("daily at 09:30")
    job = await plugin._create_job(
        chat_id=100, user_id=42, schedule="daily at 09:30", prompt="make a brief",
        parsed=parsed, reply_to_message_id=77,
    )

    await plugin._run_job(bot, job["scope"], job["id"], manual=True)

    assert bot.messages == []
    assert bot.posts == []
    helper.plugin_manager.dispatch_observe.assert_not_awaited()
    assert await plugin.db_handle.fetch_one(
        "SELECT * FROM agent_cron_jobs WHERE id = ?", (job["id"],)
    ) is None


@pytest.mark.asyncio
async def test_run_job_deleted_during_run_skips_failure_message(tmp_path, cron_db):
    """Same as above for the failure branch: a job deleted while helper.get_chat_response
    is in flight must not get a "failed" message posted after it raises."""

    class DeletingFailingHelper(FakeHelper):
        def __init__(self, db_handle):
            super().__init__()
            self._db_handle = db_handle

        async def get_chat_response(self, **kwargs):
            self.requests.append(kwargs)
            job_id = kwargs["request_id"][len("agent_cron_"):]
            await self._db_handle.execute("DELETE FROM agent_cron_jobs WHERE id = ?", (job_id,))
            raise RuntimeError("boom")

    plugin = AgentCronPlugin()
    db_handle = DbHandle(cron_db)
    helper = DeletingFailingHelper(db_handle)
    plugin.initialize(openai=helper, db=db_handle, storage_root=str(tmp_path))
    bot = FakeBot()
    parsed = plugin._parse_schedule("daily at 09:30")
    job = await plugin._create_job(
        chat_id=100, user_id=42, schedule="daily at 09:30", prompt="make a brief",
        parsed=parsed, reply_to_message_id=77,
    )

    await plugin._run_job(bot, job["scope"], job["id"], manual=True)

    assert bot.messages == []
    assert bot.posts == []


# --- Autonomous capture hook (HINDSIGHT_AUTONOMOUS_CAPTURE_ENABLED) ---------


@pytest.mark.asyncio
async def test_agent_cron_default_does_not_dispatch_autonomous_hook(tmp_path, cron_db, monkeypatch):
    monkeypatch.delenv("HINDSIGHT_AUTONOMOUS_CAPTURE_ENABLED", raising=False)
    plugin = AgentCronPlugin()
    helper = FakeHelper()  # no .plugin_manager attribute
    plugin.initialize(openai=helper, db=DbHandle(cron_db), storage_root=str(tmp_path))
    bot = FakeBot()
    parsed = plugin._parse_schedule("daily at 09:30")
    job = await plugin._create_job(
        chat_id=100, user_id=42, schedule="daily at 09:30", prompt="make a brief",
        parsed=parsed, reply_to_message_id=77,
    )

    # If the hook were dispatched unconditionally, accessing helper.plugin_manager
    # on a helper without one would either raise or (if guarded) still change
    # behavior; assert the run completes normally with today's default (disabled).
    await plugin._run_job(bot, job["scope"], job["id"], manual=True)

    stored = await plugin.db_handle.fetch_one("SELECT * FROM agent_cron_jobs WHERE id = ?", (job["id"],))
    assert stored["status"] == "active"


@pytest.mark.asyncio
async def test_agent_cron_dispatches_autonomous_hook_when_enabled(tmp_path, cron_db, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_AUTONOMOUS_CAPTURE_ENABLED", "true")
    plugin = AgentCronPlugin()
    helper = FakeHelper()
    helper.plugin_manager = SimpleNamespace(dispatch_observe=AsyncMock())
    plugin.initialize(openai=helper, db=DbHandle(cron_db), storage_root=str(tmp_path))
    bot = FakeBot()
    parsed = plugin._parse_schedule("daily at 09:30")
    job = await plugin._create_job(
        chat_id=100, user_id=42, schedule="daily at 09:30", prompt="make a brief",
        parsed=parsed, reply_to_message_id=77,
    )

    await plugin._run_job(bot, job["scope"], job["id"], manual=True)

    helper.plugin_manager.dispatch_observe.assert_awaited_once()
    call = helper.plugin_manager.dispatch_observe.call_args
    assert call.args[0] == "on_assistant_response"
    payload = call.args[1]
    assert payload.autonomous is True
    assert payload.chat_id == 100
    assert payload.user_id == 42
    assert payload.text == "cron result"
    assert call.kwargs == {"user_id": 42}


# --- Tests for surgical bugfixes ---


def test_parse_schedule_every_0_minutes_returns_none(tmp_path):
    """4b: _parse_schedule must reject zero-interval schedules."""
    plugin = AgentCronPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    result = plugin._parse_schedule("every 0 minutes")
    assert result is None


def test_advance_job_zero_interval_pauses_job(tmp_path):
    """4b defence-in-depth: _advance_job with interval_seconds=0 must pause the job."""
    plugin = AgentCronPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    job = {
        "id": "testjob",
        "schedule_type": "interval",
        "interval_seconds": 0,
        "next_run_at": None,
    }
    plugin._advance_job(job)
    assert job["paused"] is True
    assert job["next_run_at"] is None
