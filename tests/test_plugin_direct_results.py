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


_markdown2 = types.ModuleType("markdown2")
_markdown2.markdown = lambda text, *args, **kwargs: text
_install_module_if_missing("markdown2", _markdown2)

from bot.utils import handle_direct_result

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


def _config():
    return {"enable_quoting": False}


def _reaction_response(value="👍"):
    return {
        "direct_result": {
            "kind": "reaction",
            "value": value,
        }
    }


class FakeReactionTarget:
    def __init__(self):
        self.set_reaction = AsyncMock(return_value=True)


class FakeMessage:
    def __init__(self, reply_to_message=None):
        self.message_id = 100
        self.is_topic_message = False
        self.message_thread_id = None
        self.reply_to_message = reply_to_message
        self.reply_text = AsyncMock(return_value=SimpleNamespace(message_id=200))


class FakeUpdate:
    def __init__(self, message):
        self.message = message
        self.effective_message = message
        self.callback_query = None
        self.effective_chat = SimpleNamespace(type="private")


@pytest.mark.xfail(
    strict=True,
    reason="handle_direct_result reads direct_result['format'] before reaction branch exists",
)
@pytest.mark.asyncio
async def test_handle_direct_result_reaction_without_format_sets_reply_target_reaction():
    target = FakeReactionTarget()
    message = FakeMessage(reply_to_message=target)

    await handle_direct_result(_config(), FakeUpdate(message), _reaction_response("🔥"))

    target.set_reaction.assert_awaited_once()
    reaction_call = target.set_reaction.await_args
    assert reaction_call.args == ("🔥",) or reaction_call.kwargs.get("reaction") == "🔥"
    message.reply_text.assert_not_called()


@pytest.mark.xfail(
    strict=True,
    reason="handle_direct_result reads direct_result['format'] before reaction fallback exists",
)
@pytest.mark.asyncio
async def test_handle_direct_result_reaction_without_reply_target_sends_fallback_message():
    message = FakeMessage()

    sent_messages = await handle_direct_result(_config(), FakeUpdate(message), _reaction_response())

    message.reply_text.assert_awaited_once()
    assert message.reply_text.await_args.kwargs["text"].strip()
    assert sent_messages == [message.reply_text.return_value]
