import importlib.util
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


_INSERTED_MODULES = []


def _install_module_if_missing(name, module):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = module
        _INSERTED_MODULES.append(name)


_tiktoken = types.ModuleType("tiktoken")
_tiktoken.encoding_for_model = lambda _model: None
_tiktoken.get_encoding = lambda _name: None
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

from bot import telegram_bot
from bot.telegram_bot import ChatGPTTelegramBot

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class FakePluginManager:
    def get_plugin(self, name):
        return None


class FakeOpenAI:
    def __init__(self, chunks):
        self.chunks = chunks
        self.plugin_manager = FakePluginManager()
        self.stream_requests = []

    def get_current_model(self, user_id):
        return "gpt-test"

    def get_chat_response_stream(self, **kwargs):
        self.stream_requests.append(kwargs)

        async def stream():
            for chunk in self.chunks:
                yield chunk

        return stream()


class FakeDB:
    def __init__(self, conversation_context):
        self.conversation_context = conversation_context
        self.calls = []

    def get_conversation_context(self, chat_id):
        self.calls.append(chat_id)
        return self.conversation_context


class FakeMessage:
    def __init__(self, chat_id=1234, user_id=42, message_id=7, reply_side_effects=None):
        self.chat_id = chat_id
        self.message_id = message_id
        self.text = "hello"
        self.reply_to_message = None
        self.from_user = SimpleNamespace(id=user_id, name="Alice")
        self.is_topic_message = False
        self.message_thread_id = None
        self.reply_text_calls = []
        self.reply_chat_action_calls = []
        self._reply_side_effects = list(reply_side_effects or [])

    async def reply_chat_action(self, **kwargs):
        self.reply_chat_action_calls.append(kwargs)

    async def reply_text(self, **kwargs):
        self.reply_text_calls.append(kwargs)
        if self._reply_side_effects:
            effect = self._reply_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        return SimpleNamespace(chat_id=self.chat_id, message_id=900 + len(self.reply_text_calls))


class FakeUpdate:
    def __init__(self, message):
        self.message = message
        self.effective_message = message
        self.effective_chat = SimpleNamespace(id=message.chat_id, type="private")


def _make_context():
    return SimpleNamespace(
        bot=SimpleNamespace(
            id=999,
            delete_message=AsyncMock(),
        )
    )


def _make_bot(chunks, conversation_context):
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {
        "allowed_user_ids": "*",
        "bot_language": "en",
        "enable_image_generation": False,
        "enable_quoting": False,
        "enable_vision": False,
        "stream": True,
        "token_price": 0.0,
    }
    bot.db = FakeDB(conversation_context)
    bot.openai = FakeOpenAI(chunks)
    bot.last_message = {}
    bot.usage = {}
    bot._classify_reply_intent = AsyncMock(return_value=None)
    return bot


@pytest.mark.asyncio
async def test_streaming_unpacks_conversation_context_5_tuple(monkeypatch):
    edit_message = AsyncMock()
    monkeypatch.setattr(telegram_bot, "edit_message_with_retry", edit_message)

    bot = _make_bot(
        chunks=[
            ("Hello", "not_finished"),
            ("Hello world", "3"),
        ],
        conversation_context=({"messages": []}, "HTML", 0.8, 80, "session-1"),
    )
    update = FakeUpdate(FakeMessage())

    await bot.process_message("hello", update, _make_context())

    assert bot.db.calls == [1234]
    assert len(update.effective_message.reply_text_calls) == 1
    assert update.effective_message.reply_text_calls[0]["text"] == "Hello"
    assert update.effective_message.reply_text_calls[0]["parse_mode"] == "HTML"
    edit_message.assert_awaited()
    assert edit_message.await_args.kwargs["text"] == "Hello world"


@pytest.mark.asyncio
async def test_streaming_falls_back_to_html_when_parse_mode_is_none(monkeypatch):
    edit_message = AsyncMock()
    monkeypatch.setattr(telegram_bot, "edit_message_with_retry", edit_message)

    bot = _make_bot(
        chunks=[
            ("Hello", "1"),
        ],
        conversation_context=({"messages": []}, None, 0.8, 80, "session-1"),
    )
    update = FakeUpdate(FakeMessage())

    await bot.process_message("hello", update, _make_context())

    assert len(update.effective_message.reply_text_calls) == 1
    assert (
        update.effective_message.reply_text_calls[0]["parse_mode"]
        == telegram_bot.constants.ParseMode.HTML
    )
    edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_reply_text_failure_is_logged_and_does_not_retry_each_chunk(
    monkeypatch,
    caplog,
):
    edit_message = AsyncMock()
    monkeypatch.setattr(telegram_bot, "edit_message_with_retry", edit_message)

    first_send_error = RuntimeError("telegram send failed")
    fallback_message = SimpleNamespace(chat_id=1234, message_id=901)
    message = FakeMessage(reply_side_effects=[first_send_error, fallback_message])
    bot = _make_bot(
        chunks=[
            ("Hello", "not_finished"),
            ("Hello again", "not_finished"),
            ("Hello final", "3"),
        ],
        conversation_context=({"messages": []}, "HTML", 0.8, 80, "session-1"),
    )

    with caplog.at_level(logging.DEBUG, logger="bot.telegram_bot"):
        await bot.process_message("hello", FakeUpdate(message), _make_context())

    assert 1 <= len(message.reply_text_calls) <= 2
    assert len(message.reply_text_calls) < 3
    assert not edit_message.await_args_list
    assert any(record.exc_info for record in caplog.records)
