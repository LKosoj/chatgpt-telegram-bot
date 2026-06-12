import importlib.util
import importlib.machinery
import os
import sys
import types
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.plugins.agent_cron import AgentCronPlugin


class FakeBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.messages))


class FakeHelper:
    def __init__(self):
        self.requests = []

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
async def test_agent_cron_manual_run_delivers_result(tmp_path):
    plugin = AgentCronPlugin()
    helper = FakeHelper()
    plugin.initialize(openai=helper, storage_root=str(tmp_path))
    bot = FakeBot()
    parsed = plugin._parse_schedule("daily at 09:30")
    job = plugin._create_job(
        chat_id=100,
        user_id=42,
        schedule="daily at 09:30",
        prompt="make a brief",
        parsed=parsed,
        reply_to_message_id=77,
    )

    await plugin._run_job(bot, job["scope"], job["id"], manual=True)

    stored = plugin.jobs[job["scope"]][job["id"]]
    assert stored["status"] == "active"
    assert stored["last_tokens"] == 5
    assert helper.requests[0]["chat_id"] == 100
    assert bot.messages[0]["chat_id"] == 100
    assert "Cron job" in bot.messages[0]["text"]
    assert "cron result" in bot.messages[0]["text"]


# --- Tests for surgical bugfixes ---


@pytest.mark.asyncio
async def test_run_job_updates_live_dict_after_load_replaces_jobs(tmp_path):
    """4a fix A: _run_job re-fetches job from self.jobs after await so that
    next_run_at is updated in the current dict even if _load_jobs replaced it mid-flight."""

    plugin = AgentCronPlugin()

    # Helper that replaces plugin.jobs mid-await (simulates _load_jobs called from checker)
    original_jobs_ref = None
    new_jobs_ref = None

    class ReplacingHelper:
        async def get_chat_response(self, **kwargs):
            nonlocal original_jobs_ref, new_jobs_ref
            # At this point plugin.jobs still points at the original dict
            original_jobs_ref = plugin.jobs
            # Simulate _check_due_jobs -> _load_jobs replacing the dict
            # We rebuild a deep copy of jobs so plugin.jobs is a NEW object
            import copy
            new_dict = copy.deepcopy(plugin.jobs)
            plugin.jobs = new_dict
            new_jobs_ref = plugin.jobs
            return "result", 7

    helper = ReplacingHelper()
    plugin.initialize(openai=helper, storage_root=str(tmp_path))
    bot = FakeBot()

    parsed = plugin._parse_schedule("every 2 hours")
    job = plugin._create_job(
        chat_id=200,
        user_id=99,
        schedule="every 2 hours",
        prompt="check things",
        parsed=parsed,
    )
    scope = job["scope"]
    job_id = job["id"]

    await plugin._run_job(bot, scope, job_id, manual=False)

    # After run: the live (new) dict should have next_run_at updated to a future time
    live_job = plugin.jobs[scope][job_id]
    assert live_job["status"] == "active", "status should be active after success"
    from datetime import datetime
    next_run = datetime.fromisoformat(live_job["next_run_at"])
    assert next_run > datetime.now(), "next_run_at must be in the future after advance"


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


def test_save_jobs_does_not_leave_tmp_file(tmp_path):
    """3e: _save_jobs must use atomic tmp+replace and leave no .tmp file behind."""
    plugin = AgentCronPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    parsed = plugin._parse_schedule("daily at 10:00")
    plugin._create_job(
        chat_id=1,
        user_id=1,
        schedule="daily at 10:00",
        prompt="test",
        parsed=parsed,
    )
    # _create_job calls _save_jobs internally; ensure no .tmp leftover
    tmp_file = plugin.jobs_file + ".tmp"
    assert not os.path.exists(tmp_file), ".tmp file must not exist after save"
    assert os.path.exists(plugin.jobs_file), "jobs file must exist after save"
