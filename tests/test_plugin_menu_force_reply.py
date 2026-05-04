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


class FakeCallbackQuery:
    def __init__(self):
        self.data = "pluginmenu:input:notes:0"
        self.message = FakeCallbackMessage()
        self.answer = AsyncMock()
        self.edit_message_text = AsyncMock()


def _make_bot():
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {"bot_language": "en"}
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
