"""
Test for T12b cluster B4: the "validate content fits Telegram's rich
markdown byte limit, then send it once the stream is finished" logic
duplicated between the ``rich_stream_active`` and ``rich_stream_final_only``
branches inside ``ChatGPTTelegramBot._process_message_locked`` (both call
``rich_markdown_fits`` and ``send_rich_markdown`` the same way). Extracted
into ``_send_rich_markdown_if_fits(context, chat_id, content, tokens,
update)``.
"""
import importlib.util
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

import bot.telegram_bot as telegram_bot_module  # noqa: E402
from bot.telegram_bot import ChatGPTTelegramBot  # noqa: E402
from bot.telegram_rich import MAX_RICH_MARKDOWN_BYTES  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


def _make_bot():
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {"enable_quoting": False}
    return bot


def _make_update():
    return SimpleNamespace(
        effective_message=SimpleNamespace(is_topic_message=False),
        effective_chat=SimpleNamespace(type="private"),
        message=SimpleNamespace(message_id=42),
        callback_query=None,
    )


@pytest.mark.asyncio
async def test_send_rich_markdown_if_fits_raises_when_content_too_large(monkeypatch):
    bot = _make_bot()
    fake_send = AsyncMock()
    monkeypatch.setattr(telegram_bot_module, "send_rich_markdown", fake_send)
    context = SimpleNamespace(bot=SimpleNamespace())
    update = _make_update()
    oversized = "a" * (MAX_RICH_MARKDOWN_BYTES + 1)

    with pytest.raises(ValueError, match="exceeds"):
        await bot._send_rich_markdown_if_fits(context, chat_id=1, content=oversized, tokens="5", update=update)

    fake_send.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_rich_markdown_if_fits_returns_none_while_stream_not_finished(monkeypatch):
    bot = _make_bot()
    fake_send = AsyncMock()
    monkeypatch.setattr(telegram_bot_module, "send_rich_markdown", fake_send)
    context = SimpleNamespace(bot=SimpleNamespace())
    update = _make_update()

    result = await bot._send_rich_markdown_if_fits(
        context, chat_id=1, content="hello", tokens="not_finished", update=update
    )

    assert result == (None, None)
    fake_send.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_rich_markdown_if_fits_sends_and_returns_total_tokens_when_finished(monkeypatch):
    bot = _make_bot()
    sent_message = SimpleNamespace(message_id=99)
    fake_send = AsyncMock(return_value=sent_message)
    monkeypatch.setattr(telegram_bot_module, "send_rich_markdown", fake_send)
    context = SimpleNamespace(bot=SimpleNamespace())
    update = _make_update()

    result = await bot._send_rich_markdown_if_fits(
        context, chat_id=1, content="hello world", tokens="7", update=update
    )

    assert result == (sent_message, 7)
    fake_send.assert_awaited_once_with(
        context.bot,
        chat_id=1,
        markdown="hello world",
        message_thread_id=None,
        reply_to_message_id=None,
    )


@pytest.mark.asyncio
async def test_no_duplicate_rich_markdown_fits_check_remains(monkeypatch):
    """Regression guard: both original inline duplicate blocks in
    ``_process_message_locked`` (rich_stream_active / rich_stream_final_only)
    now route through the shared helper."""
    import inspect

    source = inspect.getsource(telegram_bot_module)
    assert source.count("def _send_rich_markdown_if_fits(") == 1
    assert source.count("self._send_rich_markdown_if_fits(") == 2
