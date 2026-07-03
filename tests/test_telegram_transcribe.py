import asyncio
import inspect
import importlib.util
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


class FakeUsageTracker:
    def __init__(self):
        self.transcription_seconds = []

    def get_current_cost(self):
        return {
            "cost_month": 0.0,
            "cost_week": 0.0,
            "cost_today": 0.0,
            "cost_all_time": 0.0,
        }

    def add_transcription_seconds(self, seconds):
        self.transcription_seconds.append(seconds)


class FakeDownloadedFile:
    async def download_to_drive(self, path):
        with open(path, "wb") as output:
            output.write(b"voice-bytes")


class FakeTrack:
    duration_seconds = 61

    def export(self, path, format):
        with open(path, "wb") as output:
            output.write(b"mp3-bytes")


class FakeAudioSegment:
    @staticmethod
    def from_file(path):
        return FakeTrack()


class FakeMessage:
    def __init__(self):
        self.from_user = SimpleNamespace(id=42, name="voice-user")
        self.message_id = 7
        self.is_topic_message = False
        self.effective_attachment = SimpleNamespace(
            file_unique_id="voice-file",
            file_id="telegram-file-id",
        )
        self.reply_text = AsyncMock()


class FakeUpdate:
    def __init__(self):
        self.message = FakeMessage()
        self.effective_message = self.message
        self.effective_chat = SimpleNamespace(id=1001, type="private")
        self.effective_user = self.message.from_user
        self.edited_message = None
        self.callback_query = None
        self.inline_query = None


def _make_bot(tmp_path):
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {
        "enable_transcription": True,
        "ignore_group_transcriptions": False,
        "allowed_user_ids": "*",
        "admin_user_ids": "-",
        "user_budgets": "*",
        "budget_period": "monthly",
        "guest_budget": 0.0,
        "bot_language": "en",
        "voice_reply_prompts": [],
        "voice_reply_transcript": True,
        "enable_quoting": False,
        "token_price": 0.002,
        "image_prices": [0.016, 0.018, 0.02],
        "vision_token_price": 0.01,
        "tts_prices": [0.015, 0.030],
        "transcription_price": 0.006,
    }
    bot.usage = {42: FakeUsageTracker()}
    bot.db = SimpleNamespace(
        get_active_session_id=MagicMock(return_value=None),
        get_active_session_id_async=AsyncMock(return_value=None),
    )
    bot.openai = SimpleNamespace(transcribe=AsyncMock(return_value="hello transcript"))
    bot.application = SimpleNamespace(bot=SimpleNamespace(get_file=AsyncMock(return_value=FakeDownloadedFile())))
    bot._conversation_locks = {}
    bot._conversation_locks_guard = asyncio.Lock()
    return bot


@pytest.mark.asyncio
async def test_transcribe_records_audio_duration_after_conversion(tmp_path, monkeypatch):
    async def immediate_indicator(update, context, coroutine, chat_action="", is_inline=False):
        await coroutine()

    monkeypatch.chdir(tmp_path)
    fake_module = tmp_path / "fake_bot" / "telegram_bot.py"
    fake_module.parent.mkdir()
    monkeypatch.setattr(telegram_bot, "__file__", str(fake_module))
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", immediate_indicator)
    monkeypatch.setattr(telegram_bot, "AudioSegment", FakeAudioSegment)

    bot = _make_bot(tmp_path)
    update = FakeUpdate()

    await bot.transcribe(update, SimpleNamespace())

    assert bot.usage[42].transcription_seconds == [61]
    bot.openai.transcribe.assert_awaited_once()
    update.message.reply_text.assert_awaited_once()
    assert list((fake_module.parent / "temp").iterdir()) == []


@pytest.mark.asyncio
async def test_transcribe_model_failure_logs_provider_text(tmp_path, monkeypatch, caplog):
    async def immediate_indicator(update, context, coroutine, chat_action="", is_inline=False):
        await coroutine()

    monkeypatch.chdir(tmp_path)
    fake_module = tmp_path / "fake_bot" / "telegram_bot.py"
    fake_module.parent.mkdir()
    monkeypatch.setattr(telegram_bot, "__file__", str(fake_module))
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", immediate_indicator)
    monkeypatch.setattr(telegram_bot, "AudioSegment", FakeAudioSegment)

    bot = _make_bot(tmp_path)
    bot.openai.transcribe = AsyncMock(side_effect=RuntimeError("secret provider transcript text"))
    update = FakeUpdate()

    caplog.set_level(logging.INFO, logger="bot.telegram_bot")
    await bot.transcribe(update, SimpleNamespace())

    update.message.reply_text.assert_awaited_once()
    assert "Transcribe response handling failed" in caplog.text
    assert "RuntimeError: secret provider transcript text" in caplog.text
    assert "telegram-file-id" not in caplog.text


def test_telegram_media_handlers_do_not_log_exception_objects_directly():
    sources = [
        inspect.getsource(ChatGPTTelegramBot.transcribe),
        inspect.getsource(ChatGPTTelegramBot.vision),
    ]

    for source in sources:
        assert "logger.exception(" not in source
        assert "exc_info=" not in source
