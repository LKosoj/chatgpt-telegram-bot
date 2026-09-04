"""
Tests for T13 finding 2: user-controlled/exception text sent with
parse_mode=MARKDOWN must be escaped via escape_markdown() before being sent,
otherwise Telegram raises BadRequest on unbalanced markdown entities.

Covers:
- image() generation-failure reply (finding 2a)
- tts() generation-failure reply (finding 2b)
- handle_callback_inline_query()'s "loading" edit, which embeds the raw
  cached inline query text (finding 2f)
"""
import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from telegram import constants

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


async def immediate_indicator(update, context, coroutine, chat_action="", is_inline=False):
    await coroutine()


class FakeMessage:
    def __init__(self, text, user_id=42, chat_id=1001):
        self.text = text
        self.chat_id = chat_id
        self.message_id = 7
        self.is_topic_message = False
        self.from_user = SimpleNamespace(id=user_id, name="Alice")
        self.reply_text = AsyncMock()
        self.reply_photo = AsyncMock()
        self.reply_document = AsyncMock()
        self.reply_voice = AsyncMock()
        self.reply_audio = AsyncMock()

    def parse_entities(self, *_args, **_kwargs):
        return {}


class FakeUpdate:
    def __init__(self, message):
        self.message = message
        self.effective_message = message
        self.effective_chat = SimpleNamespace(id=message.chat_id, type=constants.ChatType.PRIVATE)
        self.effective_user = message.from_user
        self.edited_message = None
        self.callback_query = None
        self.inline_query = None


def _make_bot():
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {
        "enable_image_generation": True,
        "enable_tts_generation": True,
        "bot_language": "en",
        "enable_quoting": False,
        "image_receive_mode": "photo",
    }
    bot.usage = {}
    bot.check_allowed_and_within_budget = AsyncMock(return_value=True)
    return bot


@pytest.mark.asyncio
async def test_image_generation_failure_escapes_markdown(monkeypatch):
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", immediate_indicator)
    bot = _make_bot()
    bot.openai = SimpleNamespace(generate_image=AsyncMock(side_effect=Exception("bad[chars]_here*")))
    update = FakeUpdate(FakeMessage("draw a cat"))

    await bot.image(update, SimpleNamespace())

    update.message.reply_text.assert_awaited_once()
    _, kwargs = update.message.reply_text.call_args
    assert kwargs["parse_mode"] == constants.ParseMode.MARKDOWN
    assert "bad[chars]_here*" not in kwargs["text"]
    assert "bad\\[chars\\]\\_here\\*" in kwargs["text"]


@pytest.mark.asyncio
async def test_tts_generation_failure_escapes_markdown(monkeypatch):
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", immediate_indicator)
    bot = _make_bot()
    bot.openai = SimpleNamespace(
        get_user_tts_model_async=AsyncMock(return_value="tts-1"),
        generate_speech=AsyncMock(side_effect=Exception("bad[chars]_here*")),
        config={},
    )
    update = FakeUpdate(FakeMessage("say hello"))

    await bot.tts(update, SimpleNamespace())

    update.message.reply_text.assert_awaited_once()
    _, kwargs = update.message.reply_text.call_args
    assert kwargs["parse_mode"] == constants.ParseMode.MARKDOWN
    assert "bad[chars]_here*" not in kwargs["text"]
    assert "bad\\[chars\\]\\_here\\*" in kwargs["text"]


@pytest.mark.asyncio
async def test_inline_callback_loading_message_escapes_cached_query(monkeypatch):
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", immediate_indicator)
    bot = _make_bot()
    bot.config["stream"] = False
    bot.usage = {42: SimpleNamespace()}
    bot.openai = SimpleNamespace(
        plugin_manager=SimpleNamespace(dispatch_observe=AsyncMock()),
        should_force_non_stream_first_turn_async=AsyncMock(return_value=False),
        get_chat_response=AsyncMock(side_effect=RuntimeError("stop-after-loading-edit")),
    )
    bot.inline_queries_cache = {"uid-1": "hello_world *bold*"}

    callback_query = SimpleNamespace(
        data="gpt:uid-1",
        from_user=SimpleNamespace(id=42, name="Alice"),
        inline_message_id="inline-msg-1",
    )
    update = SimpleNamespace(
        callback_query=callback_query,
        effective_chat=SimpleNamespace(id=42, type=constants.ChatType.PRIVATE),
        effective_user=SimpleNamespace(id=42),
    )
    context = SimpleNamespace(bot=SimpleNamespace(edit_message_text=AsyncMock()))

    await bot.handle_callback_inline_query(update, context)

    first_call_kwargs = context.bot.edit_message_text.call_args_list[0].kwargs
    assert first_call_kwargs["parse_mode"] == constants.ParseMode.MARKDOWN
    assert "hello_world *bold*" not in first_call_kwargs["text"]
    assert "hello\\_world \\*bold\\*" in first_call_kwargs["text"]


def _make_transcribe_bot():
    bot = _make_bot()
    bot.config.update({
        "enable_transcription": True,
        "ignore_group_transcriptions": False,
        "allowed_user_ids": "42",
        "voice_reply_transcript": True,
        "voice_reply_prompts": [],
    })
    bot.usage = {42: Mock()}
    bot._pinned_session_id = AsyncMock(return_value=None)
    bot._remember_inflight_session = Mock()
    bot._forget_inflight_session = Mock()
    return bot


def _make_audio_update():
    message = FakeMessage("")
    message.effective_attachment = SimpleNamespace(file_id="audio-file", file_unique_id="audio-unique")
    return FakeUpdate(message)


@pytest.mark.asyncio
async def test_transcribe_download_failure_escapes_markdown(monkeypatch):
    # T13 finding 2c: media_download_fail in transcribe() is sent with MARKDOWN.
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", immediate_indicator)
    monkeypatch.setattr(telegram_bot.asyncio, "sleep", AsyncMock())
    bot = _make_transcribe_bot()
    bot.application = SimpleNamespace(
        bot=SimpleNamespace(get_file=AsyncMock(side_effect=Exception("bad[chars]_here*"))),
    )
    bot.openai = SimpleNamespace(transcribe=AsyncMock())
    update = _make_audio_update()

    await bot.transcribe(update, SimpleNamespace())

    update.message.reply_text.assert_awaited_once()
    _, kwargs = update.message.reply_text.call_args
    assert kwargs["parse_mode"] == constants.ParseMode.MARKDOWN
    assert "bad[chars]_here*" not in kwargs["text"]
    assert "bad\\[chars\\]\\_here\\*" in kwargs["text"]
    bot.openai.transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_transcribe_response_failure_escapes_markdown(monkeypatch):
    # T13 finding 2d: transcribe_fail reply is sent with MARKDOWN.
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", immediate_indicator)

    class FakeAudioSegment:
        @staticmethod
        def from_file(_path):
            return SimpleNamespace(export=lambda *args, **kwargs: None, duration_seconds=3.0)

    monkeypatch.setattr(telegram_bot, "_get_audio_segment_cls", lambda: FakeAudioSegment)
    bot = _make_transcribe_bot()
    bot.application = SimpleNamespace(
        bot=SimpleNamespace(
            get_file=AsyncMock(return_value=SimpleNamespace(download_to_drive=AsyncMock())),
        ),
    )
    bot.openai = SimpleNamespace(transcribe=AsyncMock(side_effect=Exception("bad[chars]_here*")))
    update = _make_audio_update()

    await bot.transcribe(update, SimpleNamespace())

    update.message.reply_text.assert_awaited_once()
    _, kwargs = update.message.reply_text.call_args
    assert kwargs["parse_mode"] == constants.ParseMode.MARKDOWN
    assert "bad[chars]_here*" not in kwargs["text"]
    assert "bad\\[chars\\]\\_here\\*" in kwargs["text"]


@pytest.mark.asyncio
async def test_describe_image_from_context_splits_long_interpretation():
    # T13 finding 3a: a vision description above 4096 characters must be split
    # into several reply_text calls instead of one message Telegram would reject.
    bot = _make_bot()
    bot.config["allowed_user_ids"] = "42"
    bot.usage = {42: Mock()}
    bot.usage[42].add_vision_tokens_async = AsyncMock(side_effect=bot.usage[42].add_vision_tokens)
    bot._telegram_image_as_png = AsyncMock(return_value=b"png")
    bot.openai = SimpleNamespace(interpret_image=AsyncMock(return_value=("x" * 5000, 10)))
    update = FakeUpdate(FakeMessage("describe"))

    await bot._describe_image_from_context(update, "describe", "file-1")

    texts = [call.kwargs["text"] for call in update.message.reply_text.await_args_list]
    assert len(texts) > 1
    assert all(len(text) <= 4096 for text in texts)
    assert "".join(texts) == "x" * 5000
    assert update.message.reply_text.await_args_list[0].kwargs["reply_to_message_id"] is None
    bot.usage[42].add_vision_tokens.assert_called_once_with(10)
