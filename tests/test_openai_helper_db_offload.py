"""Tests for WP-B: SessionLogger.drain() called in close(), and async DB offload in _save_conversation_context."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("tiktoken")

from bot.openai_helper import OpenAIHelper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_helper_for_save():
    """Минимальный объект для тестирования _save_conversation_context."""
    helper = object.__new__(OpenAIHelper)
    helper._closed = False
    helper._background_tasks = set()
    # _ensure_session_name_with_llm не нужен — сессия не создаётся (save вернёт None)
    return helper


def _make_helper_for_close():
    """Минимальный объект для тестирования close()."""
    helper = object.__new__(OpenAIHelper)
    helper._closed = False
    helper._background_tasks = set()
    # HTTP-клиенты: заглушки с async close
    helper.client = SimpleNamespace(close=AsyncMock())
    helper.gateway_client = None
    helper._http_client = None
    return helper


# ---------------------------------------------------------------------------
# _save_conversation_context — async DB offload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_uses_async_method_when_available():
    """Если на db есть save_conversation_context_async — вызывается только он."""
    helper = _make_helper_for_save()
    async_save = AsyncMock(return_value=None)
    sync_save = MagicMock()
    helper.db = SimpleNamespace(
        save_conversation_context_async=async_save,
        save_conversation_context=sync_save,
    )

    await OpenAIHelper._save_conversation_context(
        helper,
        chat_id=1,
        context={"messages": []},
        parse_mode="HTML",
        temperature=0.7,
        max_tokens_percent=80,
        session_id=None,
    )

    async_save.assert_awaited_once()
    sync_save.assert_not_called()


@pytest.mark.asyncio
async def test_save_falls_back_to_sync_when_no_async_method():
    """Если async-метода нет — вызывается sync-вариант."""
    helper = _make_helper_for_save()
    sync_save = MagicMock(return_value=None)
    helper.db = SimpleNamespace(save_conversation_context=sync_save)

    await OpenAIHelper._save_conversation_context(
        helper,
        chat_id=1,
        context={"messages": []},
        parse_mode="HTML",
        temperature=0.7,
        max_tokens_percent=80,
        session_id=None,
    )

    sync_save.assert_called_once()


# ---------------------------------------------------------------------------
# close() — drain вызывается
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_close_calls_drain_on_session_logger():
    """close() должен вызвать session_logger.drain() после отмены фоновых задач."""
    helper = _make_helper_for_close()
    drain_mock = AsyncMock()
    helper.session_logger = SimpleNamespace(drain=drain_mock)

    await OpenAIHelper.close(helper)

    drain_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_swallows_drain_exception():
    """Исключение из drain не должно прерывать close()."""
    helper = _make_helper_for_close()

    async def _failing_drain():
        raise RuntimeError("drain exploded")

    helper.session_logger = SimpleNamespace(drain=_failing_drain)

    # Не должно бросить исключение
    await OpenAIHelper.close(helper)


@pytest.mark.asyncio
async def test_close_without_session_logger():
    """close() не падает, если session_logger отсутствует."""
    helper = _make_helper_for_close()
    # нет атрибута session_logger вообще

    await OpenAIHelper.close(helper)  # не должно бросить


@pytest.mark.asyncio
async def test_close_drain_called_before_http_client_close():
    """drain должен вызываться ДО закрытия HTTP-клиента."""
    call_order = []

    helper = _make_helper_for_close()

    async def _drain():
        call_order.append("drain")

    async def _http_close():
        call_order.append("http_close")

    helper.session_logger = SimpleNamespace(drain=_drain)
    helper.client = SimpleNamespace(close=AsyncMock(side_effect=lambda: call_order.append("http_close") or None))

    await OpenAIHelper.close(helper)

    assert call_order.index("drain") < call_order.index("http_close")
