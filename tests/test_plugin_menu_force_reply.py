import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import ForceReply


_INSERTED_MODULES = []


def _install_module_if_missing(name, module):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = module
        _INSERTED_MODULES.append(name)


class _FakeEncoding:
    def encode(self, value):
        return list(value)


_tiktoken = types.ModuleType("tiktoken")
_tiktoken.encoding_for_model = lambda _model: _FakeEncoding()
_tiktoken.get_encoding = lambda _name: _FakeEncoding()
_install_module_if_missing("tiktoken", _tiktoken)

_pydub = types.ModuleType("pydub")
_pydub.AudioSegment = object
_install_module_if_missing("pydub", _pydub)

_markdown2 = types.ModuleType("markdown2")
_markdown2.markdown = lambda text, *args, **kwargs: text
_install_module_if_missing("markdown2", _markdown2)


def _retry(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


_tenacity = types.ModuleType("tenacity")
_tenacity.retry = _retry
_tenacity.stop_after_attempt = lambda *args, **kwargs: None
_tenacity.wait_fixed = lambda *args, **kwargs: None
_tenacity.retry_if_exception_type = lambda *args, **kwargs: None
_install_module_if_missing("tenacity", _tenacity)

from bot.telegram_bot import ChatGPTTelegramBot  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class FakeCallbackMessage:
    def __init__(self):
        self.reply_text = AsyncMock(return_value=SimpleNamespace(message_id=777))
        self.delete = AsyncMock()


class FakeCallbackQuery:
    def __init__(self):
        self.data = "pluginmenu:input:notes:0"
        self.message = FakeCallbackMessage()
        self.answer = AsyncMock()
        self.edit_message_text = AsyncMock()


class FakePluginManager:
    def is_plugin_disabled_for_user(self, plugin_name, user_id):
        return False

    def disabled_plugins_for_user(self, user_id):
        return set()


def _make_bot():
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {"allowed_user_ids": "*", "bot_language": "en"}
    bot.openai = SimpleNamespace(plugin_manager=FakePluginManager())
    bot.plugin_menu_page_size = 6
    bot.plugin_menu_entries = [
        {
            "plugin_name": "notes",
            "command": "note",
            "args": "title body",
            "description": "Create a note",
        }
    ]
    return bot


@pytest.mark.asyncio
async def test_plugin_menu_input_imports_force_reply():
    bot = _make_bot()
    update = SimpleNamespace(callback_query=FakeCallbackQuery())
    context = SimpleNamespace(user_data={})

    await bot.handle_plugin_menu_callback(update, context)

    query = update.callback_query
    query.answer.assert_awaited_once()
    query.edit_message_text.assert_not_called()
    query.message.reply_text.assert_awaited_once()

    reply_args = query.message.reply_text.await_args.args
    reply_kwargs = query.message.reply_text.await_args.kwargs
    assert reply_args == ("Enter parameters for /note title body",)
    reply_markup = reply_kwargs["reply_markup"]
    assert isinstance(reply_markup, ForceReply)
    assert reply_markup.force_reply is True
    assert reply_markup.selective is True
    assert context.user_data["plugin_menu_pending"] == {
        "plugin": "notes",
        "cmd_id": "0",
        "prompt_message_id": 777,
    }


@pytest.mark.asyncio
async def test_plugin_menu_close_deletes_menu_and_clears_pending_state():
    bot = _make_bot()
    query = FakeCallbackQuery()
    query.data = "pluginmenu:close"
    update = SimpleNamespace(callback_query=query)
    context = SimpleNamespace(user_data={"plugin_menu_pending": {"plugin": "notes"}})

    await bot.handle_plugin_menu_callback(update, context)

    query.answer.assert_awaited_once()
    query.message.delete.assert_awaited_once()
    query.edit_message_text.assert_not_called()
    assert "plugin_menu_pending" not in context.user_data


def test_plugins_menu_markup_has_close_button():
    bot = _make_bot()

    markup = bot._build_plugins_menu(page=0, plugin=None)

    close_button = markup.inline_keyboard[-1][0]
    assert close_button.text == "❌ Close"
    assert close_button.callback_data == "pluginmenu:close"


@pytest.mark.asyncio
async def test_plugin_menu_command_usage_view_has_close_button():
    bot = _make_bot()
    query = FakeCallbackQuery()
    query.data = "pluginmenu:cmd:notes:0"
    update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=42))
    context = SimpleNamespace(user_data={})

    await bot.handle_plugin_menu_callback(update, context)

    reply_markup = query.edit_message_text.await_args.kwargs["reply_markup"]
    close_button = reply_markup.inline_keyboard[-1][0]
    assert close_button.text == "❌ Close"
    assert close_button.callback_data == "pluginmenu:close"
