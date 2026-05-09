import importlib.util
import importlib.machinery
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
