import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import ChatMember


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

from bot.telegram_bot import ChatGPTTelegramBot
from bot.i18n import localized_text
from bot.utils import is_allowed

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class FakeCallbackMessage:
    def __init__(self, chat_id=1234):
        self.chat_id = chat_id
        self.message_id = 55
        self.delete = AsyncMock()
        self.reply_document = AsyncMock()


class FakeCallbackQuery:
    def __init__(self, data, user_id=999, chat_id=1234):
        self.data = data
        self.from_user = SimpleNamespace(id=user_id, name=f"user-{user_id}")
        self.message = FakeCallbackMessage(chat_id=chat_id)
        self.answer = AsyncMock()
        self.edit_message_text = AsyncMock()


class FakeCallbackUpdate:
    def __init__(self, data, user_id=999, chat_id=1234):
        self.callback_query = FakeCallbackQuery(data, user_id=user_id, chat_id=chat_id)
        self.message = None
        self.effective_chat = SimpleNamespace(id=chat_id, type="private")
        self.effective_user = self.callback_query.from_user
        self.effective_message = self.callback_query.message


def _make_context():
    return SimpleNamespace(bot=SimpleNamespace(get_chat_member=AsyncMock()))


def _make_db():
    return SimpleNamespace(
        list_user_sessions=MagicMock(return_value=[]),
        create_session=MagicMock(return_value="session-1"),
        switch_active_session=MagicMock(),
        delete_session=MagicMock(),
        export_sessions_to_yaml=MagicMock(return_value=None),
        save_conversation_context=MagicMock(),
        get_conversation_context=MagicMock(return_value=({"messages": []}, "HTML", 0.1, 80, "session-1")),
        get_active_session_id=MagicMock(return_value="session-1"),
        get_session_details=MagicMock(return_value=None),
        create_hindsight_finalize_job=MagicMock(return_value=1),
    )


def _make_openai():
    return SimpleNamespace(
        config={"temperature": 0.1},
        conversations={},
        is_hindsight_enabled=MagicMock(return_value=False),
        get_current_model=MagicMock(return_value="gpt-test"),
        reset_chat_history=MagicMock(),
    )


def _make_bot(allowed_user_ids):
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {
        "allowed_user_ids": allowed_user_ids,
        "admin_user_ids": "-",
        "bot_language": "en",
        "max_sessions": 5,
    }
    bot.db = _make_db()
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
    bot._enqueue_hindsight_session_finalize_before_delete = AsyncMock()
    bot._enqueue_hindsight_and_delete_oldest_sessions_for_limit = AsyncMock()
    return bot


@pytest.mark.asyncio
async def test_callback_reset_allows_wildcard_allowed_users():
    bot = _make_bot(allowed_user_ids="*")
    update = FakeCallbackUpdate("session:back", user_id=999)

    await bot.reset(update, _make_context())

    bot.db.list_user_sessions.assert_called_once_with(999)
    text = update.callback_query.edit_message_text.await_args.kwargs["text"]
    assert text != localized_text("access_denied_command", "en")


@pytest.mark.asyncio
async def test_callback_reset_rejects_restricted_non_member():
    bot = _make_bot(allowed_user_ids="111")
    update = FakeCallbackUpdate("session:back", user_id=999)

    await bot.reset(update, _make_context())

    bot.db.list_user_sessions.assert_not_called()
    update.callback_query.edit_message_text.assert_awaited_once_with(
        text=localized_text("access_denied_command", "en")
    )


@pytest.mark.asyncio
async def test_unauthorized_prompt_selection_does_not_mutate_mode_or_context():
    bot = _make_bot(allowed_user_ids="111")
    bot.reset = AsyncMock()
    update = FakeCallbackUpdate("prompt:assistant", user_id=999)

    await bot.handle_prompt_selection(update, _make_context())

    update.callback_query.answer.assert_awaited_once()
    bot.db.list_user_sessions.assert_not_called()
    bot.db.create_session.assert_not_called()
    bot.db.save_conversation_context.assert_not_called()
    assert bot.openai.conversations == {}
    bot.reset.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback_data",
    [
        "session:new",
        "session:switch:session-1",
        "session:delete:session-1",
        "session:export",
    ],
)
async def test_unauthorized_session_callback_does_not_mutate_db(callback_data):
    bot = _make_bot(allowed_user_ids="111")
    bot.reset = AsyncMock()
    update = FakeCallbackUpdate(callback_data, user_id=999)

    await bot.handle_session_callback(update, _make_context())

    update.callback_query.answer.assert_awaited_once()
    bot.db.create_session.assert_not_called()
    bot.db.switch_active_session.assert_not_called()
    bot.db.delete_session.assert_not_called()
    bot.db.export_sessions_to_yaml.assert_not_called()
    bot.openai.reset_chat_history.assert_not_called()
    bot.reset.assert_not_called()


@pytest.mark.asyncio
async def test_restricted_group_callback_checks_membership_without_message():
    update = FakeCallbackUpdate("session:back", user_id=999, chat_id=-100123)
    update.effective_chat.type = "supergroup"
    context = _make_context()
    context.bot.get_chat_member.return_value = SimpleNamespace(status=ChatMember.MEMBER)

    allowed = await is_allowed(
        {"allowed_user_ids": "111", "admin_user_ids": "-", "bot_language": "en"},
        update,
        context,
    )

    assert allowed is True
    context.bot.get_chat_member.assert_awaited_once_with(-100123, "111")


@pytest.mark.asyncio
async def test_unauthorized_plugin_callback_query_does_not_call_plugin_handler():
    bot = _make_bot(allowed_user_ids="111")
    plugin_handler = AsyncMock()
    update = FakeCallbackUpdate("plugin:run", user_id=999)

    await bot.handle_plugin_callback_query(
        update,
        _make_context(),
        {"callback_query_handler": plugin_handler},
    )

    plugin_handler.assert_not_called()
    update.callback_query.edit_message_text.assert_awaited_once_with(
        text=localized_text("access_denied_command", "en")
    )
