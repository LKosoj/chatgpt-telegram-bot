"""
Tests for the T13a require_message/require_query/require_accessible_message
helpers (bot/telegram_bot.py) — thin, pure accessors used across
telegram_bot.py so that mypy can narrow Optional Telegram attributes via a
local variable (attribute-expression narrowing does not propagate into
nested closures, local-variable narrowing does).
"""
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

from bot.telegram_bot import (  # noqa: E402
    ChatGPTTelegramBot,
    _warn_vision_document_without_mime_type,
    require_accessible_message,
    require_message,
    require_query,
)

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


def test_require_message_returns_update_message():
    message = SimpleNamespace()
    update = SimpleNamespace(message=message)
    assert require_message(update) is message


def test_require_message_returns_none_when_absent():
    update = SimpleNamespace(message=None)
    assert require_message(update) is None


def test_require_query_returns_update_callback_query():
    query = SimpleNamespace()
    update = SimpleNamespace(callback_query=query)
    assert require_query(update) is query


def test_require_query_returns_none_when_absent():
    update = SimpleNamespace(callback_query=None)
    assert require_query(update) is None


def test_require_accessible_message_returns_query_message():
    # Дублёр без общего базового класса с telegram.Message — намеренно:
    # require_accessible_message не делает рантайм isinstance-проверку (см. её
    # докстринг в bot/telegram_bot.py), только typing-уровневый cast, поэтому
    # дак-тайпинг объекты (в т.ч. тестовые дублёры вроде этого) проходят как
    # есть, а не отсекаются как MaybeInaccessibleMessage.
    message = SimpleNamespace(reply_text=lambda *a, **k: None)
    query = SimpleNamespace(message=message)
    assert require_accessible_message(query) is message


def test_require_accessible_message_returns_none_when_query_message_absent():
    query = SimpleNamespace(message=None)
    assert require_accessible_message(query) is None


# --- T13a review round 1: None-path must stay observable (WARNING log), not silent. ---
# PII-safe by construction: only the immediate caller's function name and update_id are
# logged, never message text or user names (see tests/test_pii_safe_logging.py).


def test_require_message_logs_warning_when_absent(caplog):
    update = SimpleNamespace(message=None, update_id=777)
    caplog.set_level(logging.WARNING)

    assert require_message(update) is None

    assert "update has no message (update_id=777)" in caplog.text
    assert "test_require_message_logs_warning_when_absent" in caplog.text


def test_require_query_logs_warning_when_absent(caplog):
    update = SimpleNamespace(callback_query=None, update_id=778)
    caplog.set_level(logging.WARNING)

    assert require_query(update) is None

    assert "update has no callback_query (update_id=778)" in caplog.text
    assert "test_require_query_logs_warning_when_absent" in caplog.text


def test_require_accessible_message_logs_warning_when_absent(caplog):
    query = SimpleNamespace(message=None)
    caplog.set_level(logging.WARNING)

    assert require_accessible_message(query) is None

    assert "callback_query has no accessible message" in caplog.text
    assert "test_require_accessible_message_logs_warning_when_absent" in caplog.text


def test_warn_vision_document_without_mime_type_logs_warning(caplog):
    update = SimpleNamespace(update_id=555)
    caplog.set_level(logging.WARNING)

    _warn_vision_document_without_mime_type(update)

    assert "document attachment has no mime_type (update_id=555)" in caplog.text


@pytest.mark.asyncio
async def test_settings_replies_via_effective_message_when_update_message_is_none():
    # Regression test for the T13a review round-1 ERROR: settings() lost the working
    # reply path for updates without update.message (e.g. delivered as edited_message).
    # Fix mirrors reset()'s existing pattern — assert on update.effective_message instead
    # of narrowing through require_message, which only ever reads update.message.
    reply_text = AsyncMock()
    effective_message = SimpleNamespace(reply_text=reply_text, is_topic_message=False)
    update = SimpleNamespace(
        message=None,
        effective_message=effective_message,
        effective_user=SimpleNamespace(id=42),
    )
    context = SimpleNamespace()
    fake_self = SimpleNamespace(
        _ensure_allowed=AsyncMock(return_value=True),
        _get_user_language_async=AsyncMock(return_value="en"),
        _settings_text_async=AsyncMock(return_value="settings text"),
        _build_settings_menu=AsyncMock(return_value="menu"),
    )

    await ChatGPTTelegramBot.settings(fake_self, update, context)

    reply_text.assert_awaited_once()
