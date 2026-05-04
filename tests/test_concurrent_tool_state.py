import asyncio
import importlib.util
import json
import sys
import types
from types import SimpleNamespace

import pytest


_INSERTED_MODULES = []


def _install_module_if_missing(name, module):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = module
        _INSERTED_MODULES.append(name)


_markdown2 = types.ModuleType("markdown2")
_markdown2.markdown = lambda text, *args, **kwargs: text
_install_module_if_missing("markdown2", _markdown2)

from bot.openai_tool_handler import handle_function_call
from bot.plugins.language_learning import LanguageLearningPlugin
from bot.plugins.reminders import RemindersPlugin
from bot.plugins.task_management import TaskManagementPlugin
from bot.request_context import RequestContext

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


CONCURRENT_STATE_BUG_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="Tool plugins currently read request-local data from shared OpenAIHelper state.",
)


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.content = ""


class FakeChoice:
    def __init__(self, tool_calls):
        self.message = FakeMessage(tool_calls)


class FakeResponse:
    def __init__(self, tool_name, arguments):
        self.choices = [FakeChoice([FakeToolCall(tool_name, json.dumps(arguments))])]


class RacingPluginManager:
    def __init__(self, task_plugin, language_plugin):
        self.plugins = {
            "task_management.create_task": task_plugin,
            "language_learning.track_progress": language_plugin,
        }
        self.first_call_started = asyncio.Event()
        self.second_call_started = asyncio.Event()
        self.call_count = 0

    def filter_allowed_plugins(self, allowed_plugins):
        return allowed_plugins

    def is_function_allowed(self, function_name, allowed_plugins):
        return True

    async def call_function(self, function_name, helper, arguments):
        self.call_count += 1
        if self.call_count == 1:
            self.first_call_started.set()
            await self.second_call_started.wait()
        else:
            self.second_call_started.set()

        args = json.loads(arguments)
        plugin = self.plugins[function_name]
        base_name = function_name.split(".", 1)[1]
        await plugin.execute(base_name, helper, **args)
        return json.dumps({
            "direct_result": {
                "kind": "text",
                "format": "markdown",
                "value": "ok",
            }
        })


class SharedHelper:
    def __init__(self, plugin_manager):
        self.plugin_manager = plugin_manager
        self.config = {"bot_language": "en", "functions_max_consecutive_calls": 1}
        self.user_id = None
        self.message_id = None
        self.history = []

    def get_current_model(self, user_id):
        return "gpt-test"

    def _add_function_call_to_history(self, chat_id, function_name, content):
        self.history.append((chat_id, function_name, content))

    def _localized_text(self, key, bot_language):
        return key


@pytest.mark.asyncio
@CONCURRENT_STATE_BUG_XFAIL
async def test_concurrent_tool_calls_keep_task_and_progress_owners_separate(tmp_path):
    task_plugin = TaskManagementPlugin()
    language_plugin = LanguageLearningPlugin()
    task_plugin.initialize(storage_root=str(tmp_path))
    language_plugin.initialize(storage_root=str(tmp_path))
    plugin_manager = RacingPluginManager(task_plugin, language_plugin)
    helper = SharedHelper(plugin_manager)

    first = asyncio.create_task(handle_function_call(
        helper,
        chat_id=1001,
        response=FakeResponse(
            "task_management.create_task",
            {"title": "first task", "priority": "high"},
        ),
        allowed_plugins=["All"],
        user_id=101,
    ))
    await plugin_manager.first_call_started.wait()
    second = asyncio.create_task(handle_function_call(
        helper,
        chat_id=1002,
        response=FakeResponse(
            "language_learning.track_progress",
            {"language": "english", "completed_exercise": True},
        ),
        allowed_plugins=["All"],
        user_id=202,
    ))

    await asyncio.gather(first, second)

    assert set(task_plugin.tasks) == {"101"}
    assert set(language_plugin.users_progress) == {"202"}


@pytest.mark.asyncio
@CONCURRENT_STATE_BUG_XFAIL
async def test_reminder_uses_request_context_message_id_not_shared_helper(tmp_path):
    plugin = RemindersPlugin()
    plugin.initialize(storage_root=str(tmp_path))
    request_context = RequestContext(
        chat_id=555,
        user_id=777,
        message_id=123,
        session_id="session-1",
    )
    helper = SimpleNamespace(message_id=999)

    await plugin.execute(
        "set_reminder",
        helper,
        chat_id=request_context.plugin_chat_id,
        time="2030-01-01 12:30",
        message="check state",
        integration="telegram",
        current_time="2026-05-04 10:00",
        request_context=request_context,
    )

    reminder = next(iter(plugin.reminders[request_context.plugin_chat_id].values()))
    assert reminder["reply_to_message_id"] == request_context.message_id
