import asyncio
import importlib.util
import sys
import types
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


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
from bot.request_context import RequestContext  # noqa: E402
from bot.telegram_bot import ChatGPTTelegramBot  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class FakePluginManager:
    db = None

    def get_plugin(self, name):
        return None

    def set_db(self, db):
        self.db = db

    def disabled_plugins_for_user(self, user_id):
        return set()

    def is_plugin_disabled_for_user(self, plugin_name, user_id):
        return False


class SequencingDB:
    def __init__(self):
        self.active_by_chat = defaultdict(int)
        self.overlaps = []
        self.load_calls = []
        self.save_calls = []

    def get_conversation_context(self, chat_id, *args, **kwargs):
        self.load_calls.append(chat_id)
        if self.active_by_chat[chat_id]:
            self.overlaps.append(chat_id)
        self.active_by_chat[chat_id] += 1
        return {"messages": []}, "HTML", 0.8, 100, "session-1"

    def save_conversation_context(self, chat_id, *args, **kwargs):
        self.save_calls.append(chat_id)
        self.active_by_chat[chat_id] -= 1


class DelayedOpenAI:
    def __init__(self, db):
        self.db = db
        self.plugin_manager = FakePluginManager()
        self.active_calls = 0
        self.max_parallel_calls = 0
        self.started = asyncio.Event()
        self.requests = []

    def get_current_model(self, user_id):
        return "gpt-test"

    async def get_chat_response(self, *, chat_id, query, **kwargs):
        self.requests.append({"chat_id": chat_id, "query": query, **kwargs})
        self.active_calls += 1
        self.max_parallel_calls = max(self.max_parallel_calls, self.active_calls)
        self.started.set()
        self.db.get_conversation_context(chat_id)
        try:
            await asyncio.sleep(0.05)
            return f"response for {query}", 1
        finally:
            self.db.save_conversation_context(chat_id)
            self.active_calls -= 1


class FakeMessage:
    def __init__(self, chat_id, user_id, message_id, text):
        self.chat_id = chat_id
        self.message_id = message_id
        self.text = text
        self.reply_to_message = None
        self.from_user = SimpleNamespace(id=user_id, name=f"User {user_id}")
        self.is_topic_message = False
        self.message_thread_id = None
        self.reply_text_calls = []

    async def reply_text(self, **kwargs):
        self.reply_text_calls.append(kwargs)
        return SimpleNamespace(chat_id=self.chat_id, message_id=900 + len(self.reply_text_calls))


class FakeUpdate:
    def __init__(self, message):
        self.message = message
        self.effective_message = message
        self.effective_chat = SimpleNamespace(id=message.chat_id, type="private")
        self.effective_user = message.from_user
        self.callback_query = None


def _make_context():
    return SimpleNamespace(bot=SimpleNamespace(id=999))


def _make_bot():
    db = SequencingDB()
    openai = DelayedOpenAI(db)
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {
        "allowed_user_ids": "*",
        "bot_language": "en",
        "enable_image_generation": False,
        "enable_quoting": False,
        "enable_vision": False,
        "stream": False,
        "token_price": 0.0,
    }
    bot.db = db
    bot.openai = openai
    bot.last_message = {}
    bot.usage = {}
    bot._classify_reply_intent = AsyncMock(return_value=None)
    return bot, db, openai


async def _run_without_indicator(update, context, coroutine, chat_action, is_inline=False):
    return await coroutine()


@pytest.mark.asyncio
async def test_reentrant_process_buffer_does_not_clear_active_processing_flag():
    bot = object.__new__(ChatGPTTelegramBot)
    bot.buffer_lock = asyncio.Lock()
    bot.message_buffer = {
        1234: {
            "messages": [],
            "processing": True,
            "timer": None,
        }
    }

    await bot.process_buffer(1234)

    assert bot.message_buffer[1234]["processing"] is True


@pytest.mark.asyncio
async def test_same_conversation_key_updates_are_serialized(monkeypatch):
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", _run_without_indicator)
    bot, db, openai = _make_bot()
    first = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=1, text="first"))
    second = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=2, text="second"))

    await asyncio.gather(
        bot.process_message("first", first, _make_context()),
        bot.process_message("second", second, _make_context()),
    )

    assert db.overlaps == []
    request_contexts = [request["request_context"] for request in openai.requests]
    assert all(isinstance(context, RequestContext) for context in request_contexts)
    assert {
        (context.message_id, context.user_id)
        for context in request_contexts
    } == {(1, 42), (2, 42)}


@pytest.mark.asyncio
async def test_different_conversation_keys_can_process_independently(monkeypatch):
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", _run_without_indicator)
    bot, db, openai = _make_bot()
    first = FakeUpdate(FakeMessage(chat_id=1001, user_id=101, message_id=1, text="first"))
    second = FakeUpdate(FakeMessage(chat_id=2002, user_id=202, message_id=1, text="second"))

    await asyncio.gather(
        bot.process_message("first", first, _make_context()),
        bot.process_message("second", second, _make_context()),
    )

    assert db.overlaps == []
    assert set(db.load_calls) == {1001, 2002}
    assert set(db.save_calls) == {1001, 2002}
    assert openai.max_parallel_calls >= 2
