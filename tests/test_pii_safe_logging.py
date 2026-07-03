import asyncio
import ast
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import telegram

from bot.utils import (
    BusyStatusMessage,
    edit_message_with_retry,
    error_handler,
    log_json_shape,
    wrap_with_indicator,
)


_LOG_METHODS = {"debug", "info", "warning", "error", "critical"}
_LOG_SHAPE_WRAPPERS = {"log_exception_shape", "log_value_shape"}
_RAW_EXCEPTION_NAMES = {"e", "exc"}


def _is_logging_call(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in _LOG_METHODS:
        return False
    value = func.value
    return isinstance(value, ast.Name) and value.id in {"logger", "logging"}


def _is_shape_wrapper(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _LOG_SHAPE_WRAPPERS
    )


def _raw_exception_reference(node: ast.AST) -> str | None:
    if _is_shape_wrapper(node):
        return None
    if isinstance(node, ast.Name) and node.id in _RAW_EXCEPTION_NAMES:
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "error"
        and isinstance(node.value, ast.Name)
        and node.value.id == "context"
    ):
        return "context.error"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if (
            node.func.id == "str"
            and len(node.args) == 1
            and _raw_exception_reference(node.args[0])
        ):
            return f"str({ast.unparse(node.args[0])})"
    if isinstance(node, ast.JoinedStr):
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                reference = _raw_exception_reference(value.value)
                if reference:
                    return reference
    return None


def test_log_json_shape_preserves_payload_values():
    secret_prompt = "prompt with private details"
    secret_argument = '{"token":"private-token"}'
    secret_result = "private tool result"

    shape = log_json_shape(
        {
            "messages": [{"role": "user", "content": secret_prompt}],
            "tool_calls": [{
                "id": "call-1",
                "function": {
                    "name": "p1.do",
                    "arguments": secret_argument,
                },
            }],
            "direct_result": {
                "kind": "text",
                "value": secret_result,
            },
        },
        max_depth=5,
    )

    assert "private details" in shape
    assert "private-token" in shape
    assert "private tool result" in shape
    assert "messages" in shape
    assert "arguments" in shape
    assert "redacted" not in shape


def test_touched_runtime_logs_do_not_use_raw_exception_logging_patterns():
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "bot" / "__main__.py",
        root / "bot" / "utils.py",
        root / "bot" / "openai_helper.py",
        root / "bot" / "openai_tool_handler.py",
        root / "bot" / "telegram_bot.py",
    ]

    for path in sources:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"(?:logger|logging)\.exception\(", source)
        assert "exc_info=" not in source

        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_logging_call(node):
                continue
            for arg in node.args:
                reference = _raw_exception_reference(arg)
                assert reference is None, f"{path}:{node.lineno}: raw {reference}"


@pytest.mark.asyncio
async def test_busy_status_message_failures_log_exception_value(caplog):
    class FailingMessage:
        async def delete(self):
            raise RuntimeError("private-delete-secret")

    status = BusyStatusMessage(
        update=SimpleNamespace(),
        context=SimpleNamespace(),
        description="Loading",
    )
    status.message = FailingMessage()
    caplog.set_level(logging.WARNING)

    await status.stop()

    assert "Failed to delete busy status message error=RuntimeError: private-delete-secret" in caplog.text


@pytest.mark.asyncio
async def test_wrap_with_indicator_chat_action_failure_logs_exception_value(caplog):
    class FailingChat:
        async def send_action(self, *_args, **_kwargs):
            raise RuntimeError("private-chat-action-secret")

    class Application:
        def create_task(self, coroutine, update=None):
            return asyncio.create_task(coroutine)

    async def work():
        await asyncio.sleep(0.01)
        return "ok"

    update = SimpleNamespace(
        effective_chat=FailingChat(),
        effective_message=SimpleNamespace(is_topic_message=False),
    )
    context = SimpleNamespace(application=Application())
    caplog.set_level(logging.WARNING)

    result = await wrap_with_indicator(update, context, work, chat_action="typing")

    assert result == "ok"
    assert "Error sending chat action error=RuntimeError: private-chat-action-secret" in caplog.text


@pytest.mark.asyncio
async def test_edit_message_with_retry_failure_logs_exception_value(caplog):
    secret = "private-edit-secret"
    bot = SimpleNamespace(
        edit_message_text=AsyncMock(
            side_effect=[
                telegram.error.BadRequest("render failed"),
                RuntimeError(secret),
            ]
        )
    )
    context = SimpleNamespace(bot=bot)
    caplog.set_level(logging.WARNING)

    with pytest.raises(RuntimeError):
        await edit_message_with_retry(
            context,
            chat_id=1,
            message_id="2",
            text="private message text",
        )

    assert "Failed to edit message error=RuntimeError: private-edit-secret" in caplog.text
    assert "private message text" not in caplog.text


@pytest.mark.asyncio
async def test_error_handler_logs_exception_value(caplog):
    caplog.set_level(logging.ERROR)

    await error_handler(
        object(),
        SimpleNamespace(error=RuntimeError("private-error-handler-secret")),
    )

    assert "Exception while handling an update error=RuntimeError: private-error-handler-secret" in caplog.text
