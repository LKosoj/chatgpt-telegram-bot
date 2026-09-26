"""
Tests for T12b cluster B1: the markdown-then-plain-fallback send pattern
duplicated in ``_describe_image_from_context`` (single-image vision reply)
and ``_process_vision_media_group`` (multi-image vision reply). Both loops
send each chunk with ``parse_mode=MARKDOWN`` first and retry without a
parse mode on ``BadRequest``.

Written against the pre-extraction inline code first (this file, before the
``_send_markdown_or_plain`` helper existed) via the lighter of the two call
sites, ``_describe_image_from_context``; after extraction, a direct test of
the shared helper covers the multi-chunk/reply-to-message-id behavior common
to both call sites.
"""
import importlib.util
import io
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from telegram import constants
from telegram.error import BadRequest

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


class FakeMessage:
    def __init__(self, user_id=42, chat_id=1001):
        self.chat_id = chat_id
        self.message_id = 7
        self.is_topic_message = False
        self.from_user = SimpleNamespace(id=user_id, name="Alice")
        self.reply_text = AsyncMock()


class FakeUpdate:
    def __init__(self, message):
        self.message = message
        self.effective_message = message
        self.effective_chat = SimpleNamespace(id=message.chat_id, type=constants.ChatType.PRIVATE)
        self.effective_user = message.from_user


def _make_bot():
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {"bot_language": "en", "enable_quoting": False, "allowed_user_ids": "42"}
    bot.usage = {42: SimpleNamespace(add_vision_tokens_async=AsyncMock())}
    return bot


@pytest.mark.asyncio
async def test_describe_image_sends_markdown_first():
    bot = _make_bot()
    bot._telegram_image_as_png = AsyncMock(return_value=object())
    bot.openai = SimpleNamespace(interpret_image=AsyncMock(return_value=("**bold** result", 12)))
    update = FakeUpdate(FakeMessage())

    await bot._describe_image_from_context(update, "describe it", "file-1")

    update.message.reply_text.assert_awaited_once()
    _, kwargs = update.message.reply_text.call_args
    assert kwargs["text"] == "**bold** result"
    assert kwargs["parse_mode"] == constants.ParseMode.MARKDOWN


@pytest.mark.asyncio
async def test_describe_image_falls_back_to_plain_on_bad_request():
    bot = _make_bot()
    bot._telegram_image_as_png = AsyncMock(return_value=object())
    bot.openai = SimpleNamespace(interpret_image=AsyncMock(return_value=("broken *markdown", 12)))
    update = FakeUpdate(FakeMessage())
    update.message.reply_text = AsyncMock(side_effect=[BadRequest("bad entities"), None])

    await bot._describe_image_from_context(update, "describe it", "file-1")

    assert update.message.reply_text.await_count == 2
    first_kwargs = update.message.reply_text.await_args_list[0].kwargs
    second_kwargs = update.message.reply_text.await_args_list[1].kwargs
    assert first_kwargs["parse_mode"] == constants.ParseMode.MARKDOWN
    assert "parse_mode" not in second_kwargs
    assert second_kwargs["text"] == "broken *markdown"


@pytest.mark.asyncio
async def test_send_markdown_or_plain_multi_chunk_reply_to_only_first_chunk(monkeypatch):
    """Direct test of the extracted helper: reply_to_message_id is only set
    on the first chunk, and a BadRequest on one chunk falls back to a plain
    retry for just that chunk (later chunks are unaffected)."""
    bot = _make_bot()
    monkeypatch.setattr(telegram_bot, "split_into_chunks", lambda text: ["chunk one", "chunk two"])
    update = FakeUpdate(FakeMessage())
    update.message.reply_text = AsyncMock(side_effect=[BadRequest("bad entities"), None, None])

    await bot._send_markdown_or_plain(update, "chunk one\nchunk two")

    assert update.message.reply_text.await_count == 3
    calls = update.message.reply_text.await_args_list
    assert calls[0].kwargs["parse_mode"] == constants.ParseMode.MARKDOWN
    assert calls[0].kwargs["reply_to_message_id"] is None  # enable_quoting False, private chat
    assert "parse_mode" not in calls[1].kwargs
    assert calls[1].kwargs["text"] == "chunk one"
    assert calls[2].kwargs["parse_mode"] == constants.ParseMode.MARKDOWN
    assert calls[2].kwargs["reply_to_message_id"] is None
    assert calls[2].kwargs["text"] == "chunk two"


class FakeBusyStatus:
    async def start(self):
        return self

    async def stop(self):
        return None


def _make_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_media_group_execute_sends_markdown_first_then_falls_back_on_bad_request():
    """Direct test of call site 2 (media-group ``_process_vision_media_group``'s
    ``_execute``): behaves identically to call site 1 -- Markdown first, plain-text
    retry on BadRequest."""
    bot = _make_bot()
    bot.config["enable_vision"] = True
    bot.config["enable_vision_follow_up_questions"] = False
    bot.application = None
    bot.db = SimpleNamespace(
        save_image_async=AsyncMock(),
        get_active_session_id_async=AsyncMock(return_value="sess-1"),
    )
    bot._build_busy_status = lambda *args, **kwargs: FakeBusyStatus()
    bot.openai = SimpleNamespace(
        interpret_images=AsyncMock(return_value=("broken *markdown", 12)),
        plugin_manager=SimpleNamespace(dispatch_observe=AsyncMock()),
    )

    png_bytes = _make_png_bytes()

    async def get_file(file_id):
        return SimpleNamespace(download_as_bytearray=AsyncMock(return_value=png_bytes))

    context = SimpleNamespace(bot=SimpleNamespace(id=999, get_file=get_file))
    message = FakeMessage()
    message.reply_text = AsyncMock(side_effect=[BadRequest("bad entities"), None])
    update = FakeUpdate(message)
    item = {
        "update": update,
        "context": context,
        "chat_id": message.chat_id,
        "user_id": message.from_user.id,
        "message_id": message.message_id,
        "message_timestamp": 1000.0,
        "caption": "Describe this",
        "is_forwarded": False,
        "file_id": "file-1",
    }

    await bot._process_vision_media_group([item])

    assert message.reply_text.await_count == 2
    first_kwargs = message.reply_text.await_args_list[0].kwargs
    second_kwargs = message.reply_text.await_args_list[1].kwargs
    assert first_kwargs["parse_mode"] == constants.ParseMode.MARKDOWN
    assert "parse_mode" not in second_kwargs
    assert second_kwargs["text"] == "broken *markdown"


def test_no_duplicate_markdown_or_plain_inline_send_remains():
    """Regression guard for the B1 extraction: the markdown-then-plain
    fallback pattern must be defined once, and both original call sites
    (single-image vision reply, media-group vision reply) must route
    through it rather than re-duplicating the inline send/retry logic."""
    import inspect

    source = inspect.getsource(telegram_bot)
    assert source.count("def _send_markdown_or_plain(") == 1
    assert source.count("self._send_markdown_or_plain(") == 2
