from dataclasses import FrozenInstanceError

import pytest

from bot.request_context import RequestContext


def test_request_context_exposes_plugin_chat_id_as_int():
    context = RequestContext(
        chat_id=1234,
        user_id=42,
        message_id=7,
        session_id="session-1",
        request_id="1234_7",
    )

    assert context.chat_id == 1234
    assert context.user_id == 42
    assert context.message_id == 7
    assert context.session_id == "session-1"
    assert context.request_id == "1234_7"
    assert context.plugin_chat_id == 1234
    assert isinstance(context.plugin_chat_id, int)


def test_request_context_autonomous_defaults_to_false():
    context = RequestContext(chat_id=1234, user_id=42)

    assert context.autonomous is False


def test_request_context_accepts_autonomous_true():
    context = RequestContext(chat_id=1234, user_id=42, autonomous=True)

    assert context.autonomous is True


def test_request_context_is_immutable():
    context = RequestContext(chat_id=1234, user_id=42)

    with pytest.raises(FrozenInstanceError):
        context.chat_id = 5678
