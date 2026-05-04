import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import constants


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

from bot.telegram_bot import ChatGPTTelegramBot  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class FakeCallbackMessage:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.message_id = 55
        self.delete = AsyncMock()
        self.reply_document = AsyncMock()


class FakeCallbackQuery:
    def __init__(self, data, user_id, chat_id):
        self.data = data
        self.from_user = SimpleNamespace(id=user_id, name=f"user-{user_id}")
        self.message = FakeCallbackMessage(chat_id=chat_id)
        self.answer = AsyncMock()
        self.edit_message_text = AsyncMock()


class FakeCallbackUpdate:
    def __init__(self, data, user_id, chat_id, chat_type):
        self.callback_query = FakeCallbackQuery(data, user_id=user_id, chat_id=chat_id)
        self.message = None
        self.effective_chat = SimpleNamespace(id=chat_id, type=chat_type)
        self.effective_user = self.callback_query.from_user
        self.effective_message = self.callback_query.message


def _make_context():
    return SimpleNamespace(bot=SimpleNamespace(get_chat_member=AsyncMock()))


def _make_db(active_sessions=None):
    return SimpleNamespace(
        list_user_sessions=MagicMock(return_value=active_sessions or []),
        create_session=MagicMock(return_value="session-new"),
        switch_active_session=MagicMock(),
        delete_session=MagicMock(),
        save_conversation_context=MagicMock(),
        get_active_session_id=MagicMock(return_value="session-active"),
        get_conversation_context=MagicMock(return_value=(
            {"messages": [{"role": "user", "content": "group history"}]},
            "HTML",
            0.1,
            80,
            "session-1",
        )),
    )


def _make_openai():
    return SimpleNamespace(
        config={"temperature": 0.1},
        conversations={},
        reset_chat_history=MagicMock(),
    )


def _make_bot(active_sessions=None):
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {
        "allowed_user_ids": "*",
        "admin_user_ids": "-",
        "bot_language": "en",
        "max_sessions": 5,
        "MAX_SESSIONS": 5,
    }
    bot.db = _make_db(active_sessions=active_sessions)
    bot.openai = _make_openai()
    bot.usage = {}
    bot.get_chat_modes = MagicMock(return_value={
        "assistant": {
            "name": "Assistant",
            "prompt_start": "Assistant prompt",
            "parse_mode": "HTML",
            "temperature": 0.1,
            "max_tokens_percent": 80,
        }
    })
    bot._schedule_hindsight_session_finalize = MagicMock()
    bot.reset = AsyncMock()
    return bot


def _group_update(data):
    return FakeCallbackUpdate(
        data,
        user_id=42,
        chat_id=-100123,
        chat_type=constants.ChatType.SUPERGROUP,
    )


def _private_update(data):
    return FakeCallbackUpdate(
        data,
        user_id=42,
        chat_id=42,
        chat_type=constants.ChatType.PRIVATE,
    )


@pytest.mark.xfail(
    strict=True,
    reason="group prompt selection still uses actor user_id for session DB lookup",
)
@pytest.mark.asyncio
async def test_group_prompt_selection_uses_group_conversation_key_for_session_db():
    bot = _make_bot(active_sessions=[])
    update = _group_update("prompt:assistant")

    await bot.handle_prompt_selection(update, _make_context())

    bot.db.list_user_sessions.assert_called_once_with(-100123, is_active=1)
    bot.db.create_session.assert_called_once_with(
        user_id=-100123,
        max_sessions=5,
        openai_helper=bot.openai,
    )
    bot.db.save_conversation_context.assert_called_once()
    assert bot.db.save_conversation_context.call_args.args[0] == -100123
    assert -100123 in bot.openai.conversations


@pytest.mark.xfail(
    strict=True,
    reason="group session switch still uses actor user_id instead of group conversation_key",
)
@pytest.mark.asyncio
async def test_group_session_switch_uses_group_conversation_key():
    bot = _make_bot()
    update = _group_update("session:switch:session-2")

    await bot.handle_session_callback(update, _make_context())

    bot.db.switch_active_session.assert_called_once_with(-100123, "session-2")
    bot.db.get_conversation_context.assert_called_once_with(-100123, "session-2")
    assert bot.openai.conversations[-100123] == [{"role": "user", "content": "group history"}]


@pytest.mark.xfail(
    strict=True,
    reason="group session delete still uses actor user_id instead of group conversation_key",
)
@pytest.mark.asyncio
async def test_group_session_delete_uses_group_conversation_key():
    bot = _make_bot()
    update = _group_update("session:delete:session-2")

    await bot.handle_session_callback(update, _make_context())

    bot._schedule_hindsight_session_finalize.assert_called_once_with(-100123, "session-2")
    bot.db.delete_session.assert_called_once_with(
        -100123,
        "session-2",
        openai_helper=bot.openai,
    )
    bot.db.get_active_session_id.assert_called_once_with(-100123)
    bot.db.get_conversation_context.assert_called_once_with(
        -100123,
        "session-active",
        openai_helper=bot.openai,
    )
    assert bot.openai.conversations[-100123] == [{"role": "user", "content": "group history"}]


@pytest.mark.asyncio
async def test_private_session_switch_keeps_user_id_conversation_key():
    bot = _make_bot()
    update = _private_update("session:switch:session-2")

    await bot.handle_session_callback(update, _make_context())

    bot.db.switch_active_session.assert_called_once_with(42, "session-2")
    bot.db.get_conversation_context.assert_called_once_with(42, "session-2")
    assert bot.openai.conversations[42] == [{"role": "user", "content": "group history"}]
