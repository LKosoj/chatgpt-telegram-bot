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
        get_oldest_session_ids_for_limit=MagicMock(return_value=[]),
        delete_sessions_by_ids=MagicMock(return_value=True),
        create_hindsight_finalize_job=MagicMock(return_value=10),
        create_hindsight_finalize_jobs_for_sessions=MagicMock(return_value=[]),
        claim_hindsight_finalize_jobs=MagicMock(return_value=[]),
        mark_hindsight_finalize_job_done=MagicMock(return_value=True),
        mark_hindsight_finalize_job_failed=MagicMock(return_value=True),
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


class _FakeHindsightPlugin:
    """Stand-in for HindsightMemoryPlugin in group-flow tests.

    Stage 4A: bot code checks plugin.is_active and plugin.auto_save_enabled
    instead of helper.is_hindsight_enabled / helper.config[hindsight_auto_save].
    """

    def __init__(self):
        self.is_active = False
        self.auto_save_enabled = True

    def bank_id_for(self, user_id):
        return f"telegram-{user_id}"


class _FakeGroupPluginManager:
    def __init__(self):
        self.hindsight = _FakeHindsightPlugin()

    def has_plugin(self, name):
        return name == "hindsight_memory"

    def get_plugin(self, name):
        return self.hindsight if name == "hindsight_memory" else None

    def is_plugin_disabled_for_user(self, name, user_id):
        return False


def _make_openai():
    plugin_manager = _FakeGroupPluginManager()
    return SimpleNamespace(
        config={"temperature": 0.1, "hindsight_auto_save": True},
        conversations={},
        loaded_conversation_sessions={},
        is_hindsight_enabled=MagicMock(return_value=False),
        finalize_hindsight_session_memory=AsyncMock(return_value=0),
        reset_chat_history=MagicMock(),
        plugin_manager=plugin_manager,
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
    bot._enqueue_hindsight_session_finalize_before_delete = AsyncMock(return_value=10)
    bot._enqueue_hindsight_and_delete_oldest_sessions_for_limit = AsyncMock()
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
        prune_old_sessions=False,
    )
    bot.db.save_conversation_context.assert_called_once()
    assert bot.db.save_conversation_context.call_args.args[0] == -100123
    assert -100123 in bot.openai.conversations


@pytest.mark.asyncio
async def test_group_session_switch_uses_group_conversation_key():
    bot = _make_bot()
    update = _group_update("session:switch:session-2")

    await bot.handle_session_callback(update, _make_context())

    bot.db.switch_active_session.assert_called_once_with(-100123, "session-2")
    bot.db.get_conversation_context.assert_called_once_with(-100123, "session-2")
    assert bot.openai.conversations[-100123] == [{"role": "user", "content": "group history"}]


@pytest.mark.asyncio
async def test_group_session_new_uses_group_key_and_lowercase_max_sessions_config():
    bot = _make_bot()
    bot.config["max_sessions"] = 3
    bot.config["MAX_SESSIONS"] = 99
    update = _group_update("session:new")

    await bot.handle_session_callback(update, _make_context())

    bot.db.create_session.assert_called_once_with(
        user_id=-100123,
        max_sessions=3,
        openai_helper=bot.openai,
        prune_old_sessions=False,
    )
    bot.openai.reset_chat_history.assert_called_once_with(
        chat_id=-100123,
        content='',
        session_id="session-new",
    )


@pytest.mark.asyncio
async def test_group_session_delete_uses_group_conversation_key():
    bot = _make_bot()
    update = _group_update("session:delete:session-2")

    await bot.handle_session_callback(update, _make_context())

    bot._enqueue_hindsight_session_finalize_before_delete.assert_awaited_once_with(-100123, "session-2")
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


@pytest.mark.asyncio
async def test_group_session_delete_enqueues_hindsight_before_db_delete():
    order = []
    bot = _make_bot()
    bot._enqueue_hindsight_session_finalize_before_delete = AsyncMock(side_effect=lambda *_: order.append("enqueue") or 10)
    bot.db.delete_session.side_effect = lambda *_args, **_kwargs: order.append("delete")
    update = _group_update("session:delete:session-2")

    await bot.handle_session_callback(update, _make_context())

    assert order == ["enqueue", "delete"]


@pytest.mark.asyncio
async def test_group_session_delete_keeps_session_when_hindsight_enqueue_fails():
    bot = _make_bot()
    bot._enqueue_hindsight_session_finalize_before_delete = AsyncMock(side_effect=RuntimeError("enqueue failed"))
    update = _group_update("session:delete:session-2")

    await bot.handle_session_callback(update, _make_context())

    bot.db.delete_session.assert_not_called()
    update.callback_query.edit_message_text.assert_awaited_once()
    assert "enqueue failed" in update.callback_query.edit_message_text.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_enqueue_hindsight_session_finalize_before_delete_persists_snapshot():
    bot = _make_bot()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "remember this"},
    ]
    bot.db.get_conversation_context.return_value = (
        {"messages": messages},
        "Markdown",
        0.1,
        50,
        "session-2",
    )
    bot.openai.is_hindsight_enabled.return_value = True
    bot.openai.config["hindsight_auto_save"] = True
    bot.openai.plugin_manager.hindsight.is_active = True
    bot.openai.plugin_manager.hindsight.auto_save_enabled = True

    job_id = await ChatGPTTelegramBot._enqueue_hindsight_session_finalize_before_delete(
        bot,
        -100123,
        "session-2",
    )

    assert job_id == 10
    bot.db.create_hindsight_finalize_job.assert_called_once_with(-100123, "session-2", messages=messages)
    bot.openai.finalize_hindsight_session_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_and_delete_oldest_sessions_waits_for_snapshot_before_delete():
    order = []
    bot = _make_bot()
    bot.db.get_oldest_session_ids_for_limit.return_value = ["old-1", "old-2"]
    bot.db.create_hindsight_finalize_jobs_for_sessions.side_effect = lambda *_args: order.append("enqueue") or [10, 11]
    bot.db.delete_sessions_by_ids.side_effect = lambda *_args: order.append("delete") or True

    await ChatGPTTelegramBot._enqueue_hindsight_and_delete_oldest_sessions_for_limit(bot, -100123, 3)

    assert order == ["enqueue", "delete"]
    bot.db.create_hindsight_finalize_jobs_for_sessions.assert_called_once_with(-100123, ["old-1", "old-2"])
    bot.db.delete_sessions_by_ids.assert_called_once_with(-100123, ["old-1", "old-2"])


@pytest.mark.asyncio
async def test_enqueue_and_delete_oldest_sessions_keeps_sessions_when_enqueue_fails():
    bot = _make_bot()
    bot.db.get_oldest_session_ids_for_limit.return_value = ["old-1"]
    bot.db.create_hindsight_finalize_jobs_for_sessions.side_effect = RuntimeError("enqueue failed")

    with pytest.raises(RuntimeError, match="enqueue failed"):
        await ChatGPTTelegramBot._enqueue_hindsight_and_delete_oldest_sessions_for_limit(bot, -100123, 3)

    bot.db.delete_sessions_by_ids.assert_not_called()


@pytest.mark.asyncio
async def test_process_pending_hindsight_finalize_jobs_marks_done_after_save():
    bot = _make_bot()
    messages = [{"role": "user", "content": "remember this"}]
    bot.openai.is_hindsight_enabled.return_value = True
    bot.openai.config["hindsight_auto_save"] = True
    bot.openai.plugin_manager.hindsight.is_active = True
    bot.openai.plugin_manager.hindsight.auto_save_enabled = True
    bot.openai.finalize_hindsight_session_memory = AsyncMock(return_value=2)
    bot.db.claim_hindsight_finalize_jobs.return_value = [{
        "id": 11,
        "user_id": -100123,
        "session_id": "old-1",
        "messages": messages,
    }]

    processed = await ChatGPTTelegramBot._process_pending_hindsight_finalize_jobs(bot, limit=5)

    assert processed == 1
    bot.openai.finalize_hindsight_session_memory.assert_awaited_once_with(
        -100123,
        "old-1",
        messages=messages,
        raise_on_error=True,
        async_store=False,
    )
    bot.db.mark_hindsight_finalize_job_done.assert_called_once_with(11, 2)
    bot.db.mark_hindsight_finalize_job_failed.assert_not_called()


@pytest.mark.asyncio
async def test_process_pending_hindsight_finalize_jobs_marks_failed_for_retry():
    bot = _make_bot()
    bot.openai.is_hindsight_enabled.return_value = True
    bot.openai.config["hindsight_auto_save"] = True
    bot.openai.plugin_manager.hindsight.is_active = True
    bot.openai.plugin_manager.hindsight.auto_save_enabled = True
    bot.openai.finalize_hindsight_session_memory = AsyncMock(side_effect=RuntimeError("retain failed"))
    bot.db.claim_hindsight_finalize_jobs.return_value = [{
        "id": 11,
        "user_id": -100123,
        "session_id": "old-1",
        "messages": [{"role": "user", "content": "remember this"}],
    }]

    processed = await ChatGPTTelegramBot._process_pending_hindsight_finalize_jobs(bot, limit=5)

    assert processed == 1
    bot.db.mark_hindsight_finalize_job_done.assert_not_called()
    bot.db.mark_hindsight_finalize_job_failed.assert_called_once()
    assert bot.db.mark_hindsight_finalize_job_failed.call_args.args[:2] == (11, "retain failed")


@pytest.mark.asyncio
async def test_group_session_new_prunes_old_sessions_after_hindsight_enqueue():
    order = []
    bot = _make_bot()
    bot._enqueue_hindsight_and_delete_oldest_sessions_for_limit = AsyncMock(
        side_effect=lambda *_: order.append("prune")
    )
    bot.db.create_session.side_effect = lambda **_kwargs: order.append("create") or "session-new"
    bot.config["max_sessions"] = 3
    update = _group_update("session:new")

    await bot.handle_session_callback(update, _make_context())

    assert order == ["prune", "create"]
    bot._enqueue_hindsight_and_delete_oldest_sessions_for_limit.assert_awaited_once_with(-100123, 3)
    bot.db.create_session.assert_called_once_with(
        user_id=-100123,
        max_sessions=3,
        openai_helper=bot.openai,
        prune_old_sessions=False,
    )
    bot.openai.reset_chat_history.assert_called_once_with(
        chat_id=-100123,
        content='',
        session_id="session-new",
    )


@pytest.mark.asyncio
async def test_private_session_switch_keeps_user_id_conversation_key():
    bot = _make_bot()
    update = _private_update("session:switch:session-2")

    await bot.handle_session_callback(update, _make_context())

    bot.db.switch_active_session.assert_called_once_with(42, "session-2")
    bot.db.get_conversation_context.assert_called_once_with(42, "session-2")
    assert bot.openai.conversations[42] == [{"role": "user", "content": "group history"}]
