"""T14: UsageTracker call sites in async telegram_bot handlers must not block
the event loop. These tests patch ``asyncio.to_thread`` (in both
bot.telegram_bot and bot.usage_tracker) with a recorder -- like
tests/test_utils_send_long_response_file.py's fake_to_thread -- and assert
that the underlying sync UsageTracker methods are actually routed through it
for each handler, using a real UsageTracker (not a hand-rolled fake) so the
recorded call names come from the genuine code path.
"""
import asyncio
import importlib.util
import io
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

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
from bot import usage_tracker  # noqa: E402
from bot.telegram_bot import ChatGPTTelegramBot  # noqa: E402
from bot.usage_tracker import UsageTracker  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


def _fake_to_thread_factory(calls):
    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    return fake_to_thread


def _make_config(**overrides):
    config = {
        "enable_vision": True,
        "enable_image_generation": True,
        "enable_tts_generation": True,
        "enable_transcription": True,
        "ignore_group_transcriptions": False,
        "allowed_user_ids": "*",
        "admin_user_ids": "-",
        "user_budgets": "*",
        "budget_period": "monthly",
        "guest_budget": 0.0,
        "bot_language": "en",
        "image_receive_mode": "photo",
        "voice_reply_prompts": [],
        "voice_reply_transcript": True,
        "enable_quoting": False,
        "token_price": 0.002,
        "image_prices": [0.016, 0.018, 0.02],
        "vision_token_price": 0.01,
        "tts_prices": [0.015, 0.030],
        "transcription_price": 0.006,
    }
    config.update(overrides)
    return config


class FakeMessage:
    def __init__(self, text="hello", user_id=42):
        self.text = text
        self.caption = None
        self.photo = []
        self.document = None
        self.media_group_id = None
        self.message_id = 7
        self.is_topic_message = False
        self.message_thread_id = None
        self.from_user = SimpleNamespace(id=user_id, name="user")
        self.reply_text = AsyncMock(return_value=SimpleNamespace(message_id=900))
        self.reply_photo = AsyncMock(return_value=SimpleNamespace(message_id=901))
        self.reply_audio = AsyncMock(return_value=SimpleNamespace(message_id=902))
        self.reply_voice = AsyncMock(return_value=SimpleNamespace(message_id=903))

    def parse_entities(self, types=None):
        return {}


class FakeUpdate:
    def __init__(self, message):
        self.message = message
        self.effective_message = message
        self.effective_chat = SimpleNamespace(id=1001, type="private")
        self.effective_user = message.from_user
        self.edited_message = None
        self.callback_query = None
        self.inline_query = None


def _make_bot(tmp_path, config=None):
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = config or _make_config()
    bot.usage = {42: UsageTracker(42, "user", logs_dir=str(tmp_path))}
    bot._conversation_locks = {}
    bot._conversation_locks_guard = asyncio.Lock()
    return bot


async def _immediate_indicator(update, context, coroutine, chat_action="", is_inline=False):
    await coroutine()


@pytest.mark.asyncio
async def test_record_chat_usage_routes_add_chat_tokens_through_to_thread(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", _fake_to_thread_factory(calls))

    bot = _make_bot(tmp_path)
    bot.openai = SimpleNamespace(
        get_last_chat_usage_split=lambda chat_id: None,
        get_last_chat_model=lambda chat_id: "gpt-test",
    )

    result = await bot._record_chat_usage(chat_id=1001, user_id=42, total_tokens=50)

    assert result is True
    assert "add_chat_tokens" in calls
    assert sum(bot.usage[42].usage["usage_history"]["chat_tokens"].values()) == 50


@pytest.mark.asyncio
async def test_telegram_image_as_png_routes_pil_conversion_through_to_thread(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(telegram_bot.asyncio, "to_thread", _fake_to_thread_factory(calls))

    bot = _make_bot(tmp_path)
    source = io.BytesIO()
    Image.new("RGB", (2, 2)).save(source, format="PNG")
    bot.openai = SimpleNamespace(download_file_as_bytes=AsyncMock(return_value=source.getvalue()))

    result = await bot._telegram_image_as_png("file-id")

    assert calls == ["_convert"]
    decoded = Image.open(result)
    assert decoded.format == "PNG"


@pytest.mark.asyncio
async def test_check_allowed_and_within_budget_routes_get_current_cost_through_to_thread(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", _fake_to_thread_factory(calls))

    bot = _make_bot(tmp_path)
    update = FakeUpdate(FakeMessage())

    allowed = await bot.check_allowed_and_within_budget(update, SimpleNamespace())

    assert allowed is True
    assert "get_current_cost" in calls


@pytest.mark.asyncio
async def test_describe_image_from_context_routes_add_vision_tokens_through_to_thread(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(telegram_bot.asyncio, "to_thread", _fake_to_thread_factory(calls))
    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", _fake_to_thread_factory(calls))

    bot = _make_bot(tmp_path)
    source = io.BytesIO()
    Image.new("RGB", (2, 2)).save(source, format="PNG")
    bot.openai = SimpleNamespace(
        download_file_as_bytes=AsyncMock(return_value=source.getvalue()),
        interpret_image=AsyncMock(return_value=("a description", 42)),
    )
    message = FakeMessage()
    update = FakeUpdate(message)

    await bot._describe_image_from_context(update, "describe this", "file-id")

    assert "_convert" in calls
    assert "add_vision_tokens" in calls
    message.reply_text.assert_awaited()
    assert sum(bot.usage[42].usage["usage_history"]["vision_tokens"].values()) == 42


@pytest.mark.asyncio
async def test_image_generation_routes_add_image_request_through_to_thread(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", _fake_to_thread_factory(calls))
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", _immediate_indicator)

    bot = _make_bot(tmp_path)
    bot.openai = SimpleNamespace(
        generate_image=AsyncMock(return_value=("http://example.com/img.png", "1024x1024")),
    )
    message = FakeMessage(text="a cat")
    update = FakeUpdate(message)

    await bot.image(update, SimpleNamespace())

    assert "add_image_request" in calls
    message.reply_photo.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_generation_routes_add_tts_request_through_to_thread(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", _fake_to_thread_factory(calls))
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", _immediate_indicator)

    bot = _make_bot(tmp_path)
    speech_file = SimpleNamespace(close=lambda: None)
    bot.openai = SimpleNamespace(
        get_user_tts_model_async=AsyncMock(return_value="tts-1"),
        generate_speech=AsyncMock(return_value=(speech_file, 123)),
        config={"tts_response_format": "wav"},
    )
    message = FakeMessage(text="say hi")
    update = FakeUpdate(message)

    await bot.tts(update, SimpleNamespace())

    assert "add_tts_request" in calls
    message.reply_audio.assert_awaited_once()


class _FakeDownloadedFile:
    async def download_to_drive(self, path):
        with open(path, "wb") as output:
            output.write(b"voice-bytes")


class _FakeTrack:
    duration_seconds = 61

    def export(self, path, format):
        with open(path, "wb") as output:
            output.write(b"mp3-bytes")


class _FakeAudioSegment:
    @staticmethod
    def from_file(path):
        return _FakeTrack()


@pytest.mark.asyncio
async def test_transcribe_routes_add_transcription_seconds_through_to_thread(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(usage_tracker.asyncio, "to_thread", _fake_to_thread_factory(calls))
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", _immediate_indicator)
    monkeypatch.setattr(telegram_bot, "AudioSegment", _FakeAudioSegment)
    monkeypatch.chdir(tmp_path)
    fake_module = tmp_path / "fake_bot" / "telegram_bot.py"
    fake_module.parent.mkdir()
    monkeypatch.setattr(telegram_bot, "__file__", str(fake_module))

    bot = _make_bot(tmp_path)
    message = FakeMessage()
    message.effective_attachment = SimpleNamespace(file_unique_id="voice-file", file_id="telegram-file-id")
    update = FakeUpdate(message)
    bot.db = SimpleNamespace(
        get_active_session_id=lambda *a, **k: None,
        get_active_session_id_async=AsyncMock(return_value=None),
    )
    bot.openai = SimpleNamespace(transcribe=AsyncMock(return_value="hello transcript"))
    bot.application = SimpleNamespace(bot=SimpleNamespace(get_file=AsyncMock(return_value=_FakeDownloadedFile())))

    await bot.transcribe(update, SimpleNamespace())

    assert "add_transcription_seconds" in calls
    assert sum(bot.usage[42].usage["usage_history"]["transcription_seconds"].values()) == 61
