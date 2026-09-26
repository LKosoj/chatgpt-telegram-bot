"""
Tests for T12b cluster B2: the ``dispatch_observe("on_session_reset", ...)``
/ ``dispatch_observe("on_user_message", ...)`` pattern duplicated across
several call sites in ``bot/telegram_bot.py``.

``test_handle_direct_result_dispatches_session_reset_final_delivery`` is
written against the pre-extraction inline code (the ``getattr``-based
variant in ``_handle_direct_result``, reason="final_delivery") and passes
before ``_dispatch_session_reset``/``_dispatch_user_message`` exist. The
remaining tests exercise the extracted helpers directly and a structural
guard confirming no duplicate inline payload construction remains.
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

from bot import telegram_bot  # noqa: E402
from bot.telegram_bot import ChatGPTTelegramBot  # noqa: E402
from bot.plugins.hooks import SessionResetPayload, UserMessagePayload  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


def _make_bot():
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {"bot_language": "en"}
    bot.openai = SimpleNamespace(plugin_manager=SimpleNamespace(dispatch_observe=AsyncMock()))
    return bot


@pytest.mark.asyncio
async def test_handle_direct_result_dispatches_session_reset_final_delivery(monkeypatch):
    bot = _make_bot()
    sent_message = SimpleNamespace(message_id=1)
    monkeypatch.setattr(telegram_bot, "handle_direct_result", AsyncMock(return_value=[sent_message]))
    bot._remember_sent_image_messages = AsyncMock()
    bot._run_post_delivery_cleanup = AsyncMock()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=111),
        effective_user=SimpleNamespace(id=222),
        effective_message=SimpleNamespace(reply_text=AsyncMock()),
    )

    await bot._handle_direct_result(update, {"direct_result": {"kind": "photo"}})

    bot.openai.plugin_manager.dispatch_observe.assert_awaited_once_with(
        "on_session_reset",
        SessionResetPayload(chat_id=111, user_id=222, reason="final_delivery", terminal_only=False),
        user_id=222,
    )


@pytest.mark.asyncio
async def test_dispatch_session_reset_builds_expected_payload():
    bot = _make_bot()

    await bot._dispatch_session_reset(555, 666, reason="request_start")

    bot.openai.plugin_manager.dispatch_observe.assert_awaited_once_with(
        "on_session_reset",
        SessionResetPayload(chat_id=555, user_id=666, reason="request_start", terminal_only=False),
        user_id=666,
    )


@pytest.mark.asyncio
async def test_dispatch_user_message_builds_expected_payload():
    bot = _make_bot()

    await bot._dispatch_user_message(
        555, 666,
        request_id="555_1",
        text="hello",
        has_image=True,
        has_voice=False,
        is_command=False,
        ts=123.0,
    )

    bot.openai.plugin_manager.dispatch_observe.assert_awaited_once_with(
        "on_user_message",
        UserMessagePayload(
            chat_id=555, user_id=666, request_id="555_1", text="hello",
            has_image=True, has_voice=False, is_command=False, ts=123.0,
        ),
        user_id=666,
    )


def test_no_duplicate_inline_payload_construction_remains():
    """Regression guard for the B2 extraction: SessionResetPayload/
    UserMessagePayload should only be constructed inside the two dispatch
    helpers, not re-duplicated inline at any of the original call sites."""
    import inspect

    source = inspect.getsource(telegram_bot)
    assert source.count("SessionResetPayload(") == 1
    assert source.count("UserMessagePayload(") == 1
