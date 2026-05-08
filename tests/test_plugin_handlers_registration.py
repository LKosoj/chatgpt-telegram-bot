import importlib.util
import sys
import types

import pytest
from telegram.ext import MessageHandler, filters


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

from bot import telegram_bot  # noqa: E402
from bot.telegram_bot import ChatGPTTelegramBot  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


async def plugin_message_callback(update, context):
    return None


class FakeTelegramBot:
    def __init__(self):
        self.set_my_commands_calls = []

    async def set_my_commands(self, *args, **kwargs):
        self.set_my_commands_calls.append((args, kwargs))


class FakeApplication:
    def __init__(self):
        self.bot = FakeTelegramBot()
        self.handlers = []
        self.error_handlers = []
        self.post_init_callback = None
        self.owner = None
        self.run_polling_calls = 0

    def add_handler(self, handler, group=0):
        self.handlers.append(handler)

    def add_error_handler(self, handler):
        self.error_handlers.append(handler)

    def run_polling(self):
        self.run_polling_calls += 1
        if self.post_init_callback:
            self.owner._background_tasks = [object()]
            telegram_bot.asyncio.get_event_loop().run_until_complete(
                self.post_init_callback(self)
            )


class FakeApplicationBuilder:
    def __init__(self, application):
        self.application = application

    def token(self, token):
        return self

    def post_init(self, callback):
        self.application.post_init_callback = callback
        return self

    def post_shutdown(self, callback):
        return self

    def concurrent_updates(self, enabled):
        return self

    def local_mode(self, enabled):
        return self

    def base_url(self, url):
        return self

    def build(self):
        return self.application


class FakePluginManager:
    def __init__(self, message_handlers):
        self.message_handlers = message_handlers
        self.get_message_handlers_calls = 0

    def set_openai(self, openai):
        self.openai = openai

    def get_message_handlers(self):
        self.get_message_handlers_calls += 1
        return self.message_handlers

    def build_bot_commands(self):
        return {"plugin_commands": [], "menu_entries": []}

    def close_all(self):
        return None


class FakeOpenAI:
    def __init__(self, plugin_manager):
        self.plugin_manager = plugin_manager
        self.bot = None

    async def close(self):
        return None


def _message_handler_index(handlers, callback_name):
    for index, handler in enumerate(handlers):
        if (
            isinstance(handler, MessageHandler)
            and getattr(handler.callback, "__name__", None) == callback_name
        ):
            return index
    raise AssertionError(f"Missing MessageHandler callback {callback_name}")


def _make_bot(message_handlers):
    plugin_manager = FakePluginManager(message_handlers)
    openai = FakeOpenAI(plugin_manager)
    bot = ChatGPTTelegramBot(
        {
            "token": "test-token",
            "bot_language": "en",
            "enable_image_generation": False,
            "enable_tts_generation": False,
        },
        openai,
        db=object(),
    )
    return bot, plugin_manager


def test_plugin_message_handlers_registered_once_and_before_builtin_handlers(
    monkeypatch,
):
    ready_handler = MessageHandler(
        filters.Document.ALL,
        plugin_message_callback,
    )
    message_handlers = [
        {"handler": ready_handler},
        {
            "filters": filters.TEXT,
            "handler": plugin_message_callback,
            "handler_kwargs": {},
        },
    ]
    bot, plugin_manager = _make_bot(message_handlers)
    application = FakeApplication()
    application.owner = bot
    monkeypatch.setattr(
        telegram_bot,
        "ApplicationBuilder",
        lambda: FakeApplicationBuilder(application),
    )

    async def cleanup():
        return None

    monkeypatch.setattr(bot, "cleanup", cleanup)

    bot.run()

    generated_plugin_handlers = [
        handler
        for handler in application.handlers
        if (
            isinstance(handler, MessageHandler)
            and getattr(handler.callback, "__name__", None)
            == "plugin_message_handler"
        )
    ]
    plugin_indexes = [
        index
        for index, handler in enumerate(application.handlers)
        if handler is ready_handler or handler in generated_plugin_handlers
    ]
    document_index = _message_handler_index(
        application.handlers,
        "handle_document",
    )
    text_index = _message_handler_index(application.handlers, "prompt")

    ready_handler_count = sum(
        handler is ready_handler for handler in application.handlers
    )
    order_is_preserved = max(plugin_indexes) < document_index < text_index

    assert (
        application.run_polling_calls,
        plugin_manager.get_message_handlers_calls,
        ready_handler_count,
        len(generated_plugin_handlers),
        order_is_preserved,
    ) == (1, 1, 1, 1, True)


@pytest.mark.asyncio
async def test_post_init_guard_skips_plugin_message_handlers_on_second_call():
    ready_handler = MessageHandler(
        filters.Document.ALL,
        plugin_message_callback,
    )
    message_handlers = [
        {"handler": ready_handler},
        {
            "filters": filters.TEXT,
            "handler": plugin_message_callback,
            "handler_kwargs": {},
        },
    ]
    bot, plugin_manager = _make_bot(message_handlers)
    bot._background_tasks = [object()]
    application = FakeApplication()

    await bot.post_init(application)
    await bot.post_init(application)

    generated_plugin_handlers = [
        handler
        for handler in application.handlers
        if (
            isinstance(handler, MessageHandler)
            and getattr(handler.callback, "__name__", None)
            == "plugin_message_handler"
        )
    ]
    ready_handler_count = sum(
        handler is ready_handler for handler in application.handlers
    )

    assert (
        plugin_manager.get_message_handlers_calls,
        ready_handler_count,
        len(generated_plugin_handlers),
    ) == (1, 1, 1)
