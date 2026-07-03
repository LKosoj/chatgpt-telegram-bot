import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import ChatMember
from telegram.error import BadRequest


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
import bot.utils as utils_module

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class FakeCallbackMessage:
    def __init__(self, chat_id=1234):
        self.chat_id = chat_id
        self.message_id = 55
        self.is_topic_message = False
        self.delete = AsyncMock()
        self.reply_text = AsyncMock()
        self.reply_document = AsyncMock()


class FakeCallbackQuery:
    def __init__(self, data, user_id=999, chat_id=1234, language_code="en"):
        self.data = data
        self.from_user = SimpleNamespace(id=user_id, name=f"user-{user_id}", language_code=language_code)
        self.inline_message_id = "inline-message-1"
        self.message = FakeCallbackMessage(chat_id=chat_id)
        self.answer = AsyncMock()
        self.edit_message_text = AsyncMock()


class FakeCallbackUpdate:
    def __init__(self, data, user_id=999, chat_id=1234, language_code="en"):
        self.callback_query = FakeCallbackQuery(
            data,
            user_id=user_id,
            chat_id=chat_id,
            language_code=language_code,
        )
        self.message = None
        self.effective_chat = SimpleNamespace(id=chat_id, type="private")
        self.effective_user = self.callback_query.from_user
        self.effective_message = self.callback_query.message


class FakeMessage:
    def __init__(self, user_id=999, chat_id=1234, language_code="en"):
        self.from_user = SimpleNamespace(id=user_id, name=f"user-{user_id}", language_code=language_code)
        self.chat_id = chat_id
        self.message_id = 56
        self.text = "/help"
        self.reply_to_message = None
        self.is_topic_message = False
        self.message_thread_id = None
        self.reply_text = AsyncMock()


class FakeMessageUpdate:
    def __init__(self, user_id=999, chat_id=1234, language_code="en"):
        self.message = FakeMessage(user_id=user_id, chat_id=chat_id, language_code=language_code)
        self.callback_query = None
        self.inline_query = None
        self.effective_chat = SimpleNamespace(id=chat_id, type="private")
        self.effective_user = self.message.from_user
        self.effective_message = self.message


class FakeInlineCallbackQuery:
    def __init__(self, data, user_id=999, language_code="en"):
        self.data = data
        self.from_user = SimpleNamespace(id=user_id, name=f"user-{user_id}", language_code=language_code)
        self.inline_message_id = "inline-message-1"
        self.message = None
        self.answer = AsyncMock()
        self.edit_message_text = AsyncMock()


class FakeInlineCallbackUpdate:
    def __init__(self, data, user_id=999, language_code="en"):
        self.callback_query = FakeInlineCallbackQuery(data, user_id=user_id, language_code=language_code)
        self.message = None
        self.inline_query = None
        self.effective_chat = None
        self.effective_user = self.callback_query.from_user
        self.effective_message = None


def _make_context():
    return SimpleNamespace(bot=SimpleNamespace(get_chat_member=AsyncMock()))


def _make_db():
    db = SimpleNamespace(
        list_user_sessions=MagicMock(return_value=[]),
        create_session=MagicMock(return_value="session-1"),
        switch_active_session=MagicMock(),
        delete_session=MagicMock(),
        export_sessions_to_yaml=MagicMock(return_value=None),
        get_user_settings=MagicMock(return_value=None),
        save_user_settings=MagicMock(),
        save_user_model=MagicMock(),
        save_conversation_context=MagicMock(),
        get_conversation_context=MagicMock(return_value=({"messages": []}, "HTML", 0.1, 80, "session-1")),
        get_active_session_id=MagicMock(return_value="session-1"),
        get_session_details=MagicMock(return_value=None),
    )
    db.list_user_sessions_async = AsyncMock(side_effect=lambda *args, **kwargs: db.list_user_sessions(*args, **kwargs))
    db.create_session_async = AsyncMock(side_effect=lambda *args, **kwargs: db.create_session(*args, **kwargs))
    db.switch_active_session_async = AsyncMock(side_effect=lambda *args, **kwargs: db.switch_active_session(*args, **kwargs))
    db.delete_session_async = AsyncMock(side_effect=lambda *args, **kwargs: db.delete_session(*args, **kwargs))
    db.export_sessions_to_yaml_async = AsyncMock(side_effect=lambda *args, **kwargs: db.export_sessions_to_yaml(*args, **kwargs))
    db.get_user_settings_async = AsyncMock(side_effect=lambda *args, **kwargs: db.get_user_settings(*args, **kwargs))
    db.save_user_settings_async = AsyncMock(side_effect=lambda *args, **kwargs: db.save_user_settings(*args, **kwargs))
    db.save_user_model_async = AsyncMock(side_effect=lambda *args, **kwargs: db.save_user_model(*args, **kwargs))
    db.save_conversation_context_async = AsyncMock(
        side_effect=lambda *args, **kwargs: db.save_conversation_context(*args, **kwargs)
    )
    db.get_conversation_context_async = AsyncMock(
        side_effect=lambda *args, **kwargs: db.get_conversation_context(*args, **kwargs)
    )
    db.get_active_session_id_async = AsyncMock(side_effect=lambda *args, **kwargs: db.get_active_session_id(*args, **kwargs))
    db.get_session_details_async = AsyncMock(side_effect=lambda *args, **kwargs: db.get_session_details(*args, **kwargs))
    db.delete_sessions_by_ids = MagicMock(return_value=True)
    db.delete_sessions_by_ids_async = AsyncMock(side_effect=lambda *args, **kwargs: db.delete_sessions_by_ids(*args, **kwargs))
    db.get_oldest_session_ids_for_limit = MagicMock(return_value=[])
    db.get_oldest_session_ids_for_limit_async = AsyncMock(
        side_effect=lambda *args, **kwargs: db.get_oldest_session_ids_for_limit(*args, **kwargs)
    )
    db.save_image = MagicMock()
    db.save_image_async = AsyncMock(side_effect=lambda *args, **kwargs: db.save_image(*args, **kwargs))
    db.get_user_images = MagicMock(return_value=[])
    db.get_user_images_async = AsyncMock(side_effect=lambda *args, **kwargs: db.get_user_images(*args, **kwargs))
    db.cleanup_old_images = MagicMock()
    db.cleanup_old_images_async = AsyncMock(side_effect=lambda *args, **kwargs: db.cleanup_old_images(*args, **kwargs))
    return db


def _make_openai():
    return SimpleNamespace(
        config={"temperature": 0.1, "tts_model": "tts-a", "tts_voice": "alice"},
        conversations={},
        get_current_model=MagicMock(return_value="gpt-test"),
        get_current_model_async=AsyncMock(return_value="gpt-test"),
        get_user_tts_model=MagicMock(return_value="tts-a"),
        get_user_tts_voice=MagicMock(return_value="alice"),
        get_user_tts_model_async=AsyncMock(return_value="tts-a"),
        get_user_tts_voice_async=AsyncMock(return_value="alice"),
        get_available_tts_models=AsyncMock(return_value=["tts-a", "tts-b"]),
        get_available_tts_voices=AsyncMock(return_value=["alice", "bob"]),
        reset_chat_history=AsyncMock(),
        get_chat_response=AsyncMock(return_value=("answer", 1)),
        get_chat_response_stream=MagicMock(),
        should_force_non_stream_first_turn=MagicMock(return_value=True),
        should_force_non_stream_first_turn_async=AsyncMock(return_value=True),
        plugin_manager=FakeSettingsPluginManager(),
    )


class FakeDescribedPlugin:
    def get_source_name(self):
        return "Alpha source"

    def get_help_text(self):
        return "Alpha help text"

    def get_commands(self):
        return [{"command": "alpha", "description": "Run alpha command"}]

    def get_spec(self):
        return [{"name": "alpha.search", "description": "Search alpha records"}]


class FakeSettingsPluginManager:
    plugins = {"alpha": object(), "beta": object(), "skills": object()}

    def __init__(self):
        self.alpha_plugin = FakeDescribedPlugin()
        self.skills_plugin = SimpleNamespace(available_skills={
            "docx": {
                "name": "docx",
                "description": "Work with Word documents",
                "scripts": [],
                "agents": [],
                "references": [],
            },
            "pptx": {
                "name": "PowerPoint",
                "description": "Create and edit presentations",
                "scripts": ["qa-pptx.sh"],
                "agents": [{"id": "deck_generator"}],
                "references": ["references/checklist.md"],
            },
        })

    def has_plugin(self, plugin_name):
        return plugin_name in self.plugins

    def get_plugin(self, plugin_name):
        if plugin_name == "alpha":
            return self.alpha_plugin
        if plugin_name == "skills":
            return self.skills_plugin
        return None

    def get_plugin_help_texts(self):
        return [{"plugin_name": "alpha", "text": "alpha secret help"}]

    def is_plugin_disabled_for_user(self, plugin_name, user_id):
        return False

    async def collect_objects(self, slot, payload, *, user_id=None):
        return []


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
    bot.commands = [SimpleNamespace(command="help", description="Help")]
    bot.group_commands = bot.commands
    bot.inline_queries_cache = {}
    bot._user_language_cache = {}
    bot.get_chat_modes = MagicMock(return_value={
        "assistant": {
            "name": "Assistant",
            "prompt_start": "Assistant prompt",
            "parse_mode": "HTML",
            "temperature": 0.1,
            "max_tokens_percent": 80,
        }
    })
    bot._dispatch_session_before_delete = AsyncMock()
    bot._dispatch_and_delete_oldest_sessions_for_limit = AsyncMock()
    return bot


@pytest.mark.asyncio
async def test_unauthorized_help_is_denied_before_plugin_help():
    bot = _make_bot(allowed_user_ids="111")
    bot.openai.plugin_manager.get_plugin_help_texts = MagicMock(return_value=[
        {"plugin_name": "alpha", "text": "alpha secret help"},
    ])
    update = FakeMessageUpdate(user_id=999)

    await bot.help(update, _make_context())

    update.message.reply_text.assert_awaited_once()
    text = update.message.reply_text.await_args.kwargs["text"]
    assert text == localized_text("disallowed", "en")
    bot.openai.plugin_manager.get_plugin_help_texts.assert_not_called()


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
        "session:set_model:0",
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
    bot.db.save_user_model.assert_not_called()
    bot.openai.reset_chat_history.assert_not_called()
    bot.reset.assert_not_called()


@pytest.mark.asyncio
async def test_session_change_model_shows_openai_model_choices():
    bot = _make_bot(allowed_user_ids="*")
    bot.openai.config = {
        "model": "model-a",
        "model_choices": ["model-a", "model-b"],
    }
    bot.openai.get_current_model.return_value = "model-a"
    bot.openai.get_current_model_async.return_value = "model-a"
    update = FakeCallbackUpdate("session:change_model", user_id=999)

    await bot.handle_session_callback(update, _make_context())

    text = update.callback_query.edit_message_text.await_args.kwargs["text"]
    reply_markup = update.callback_query.edit_message_text.await_args.kwargs["reply_markup"]
    assert "OPENAI_MODEL" in text
    assert reply_markup.inline_keyboard[0][0].text == "✓ model-a"
    assert reply_markup.inline_keyboard[1][0].text == "model-b"
    assert reply_markup.inline_keyboard[1][0].callback_data == "session:set_model:1"


@pytest.mark.asyncio
async def test_session_set_model_saves_selected_openai_model_choice():
    bot = _make_bot(allowed_user_ids="*")
    bot.openai.config = {
        "model": "model-a",
        "model_choices": ["model-a", "model-b"],
    }
    bot.reset = AsyncMock()
    update = FakeCallbackUpdate("session:set_model:1", user_id=999)
    context = _make_context()

    await bot.handle_session_callback(update, context)

    bot.db.save_user_model.assert_called_once_with(999, "model-b")
    bot.reset.assert_awaited_once_with(update, context)


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
async def test_restricted_group_membership_check_uses_ttl_cache(monkeypatch):
    utils_module._GROUP_MEMBERSHIP_CACHE.clear()
    now = 1000.0
    monkeypatch.setattr(utils_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(utils_module, "_GROUP_MEMBERSHIP_CACHE_TTL_SECONDS", 10.0)
    update = FakeCallbackUpdate("session:back", user_id=999, chat_id=-100456)
    update.effective_chat.type = "supergroup"
    context = _make_context()
    context.bot.get_chat_member.return_value = SimpleNamespace(status=ChatMember.MEMBER)
    config = {"allowed_user_ids": "111", "admin_user_ids": "-", "bot_language": "en"}

    assert await is_allowed(config, update, context) is True
    assert await is_allowed(config, update, context) is True
    context.bot.get_chat_member.assert_awaited_once_with(-100456, "111")

    now = 1011.0

    assert await is_allowed(config, update, context) is True
    assert context.bot.get_chat_member.await_count == 2
    utils_module._GROUP_MEMBERSHIP_CACHE.clear()


@pytest.mark.asyncio
async def test_restricted_group_membership_unexpected_bad_request_is_not_cached(monkeypatch):
    utils_module._GROUP_MEMBERSHIP_CACHE.clear()
    monkeypatch.setattr(utils_module, "_GROUP_MEMBERSHIP_CACHE_TTL_SECONDS", 10.0)
    update = FakeCallbackUpdate("session:back", user_id=999, chat_id=-100789)
    update.effective_chat.type = "supergroup"
    context = _make_context()
    context.bot.get_chat_member.side_effect = [
        BadRequest("Chat not found"),
        SimpleNamespace(status=ChatMember.MEMBER),
    ]
    config = {"allowed_user_ids": "111", "admin_user_ids": "-", "bot_language": "en"}

    with pytest.raises(BadRequest, match="Chat not found"):
        await is_allowed(config, update, context)

    assert await is_allowed(config, update, context) is True
    assert context.bot.get_chat_member.await_count == 2
    assert utils_module._GROUP_MEMBERSHIP_CACHE.get((-100789, "111")) is not None
    utils_module._GROUP_MEMBERSHIP_CACHE.clear()


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


@pytest.mark.asyncio
async def test_unauthorized_plugin_menu_callback_is_denied_before_menu_actions():
    bot = _make_bot(allowed_user_ids="111")
    bot._build_plugins_menu = MagicMock()
    update = FakeCallbackUpdate("pluginmenu:page:root:0", user_id=999)

    await bot.handle_plugin_menu_callback(update, _make_context())

    update.callback_query.answer.assert_awaited_once()
    update.callback_query.edit_message_text.assert_awaited_once_with(
        text=localized_text("access_denied_command", "en")
    )
    bot._build_plugins_menu.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_inline_gpt_callback_does_not_call_openai_or_consume_cache():
    bot = _make_bot(allowed_user_ids="111")
    bot.inline_queries_cache["abc"] = "expensive prompt"
    update = FakeInlineCallbackUpdate("gpt:abc", user_id=999)

    await bot.handle_callback_inline_query(update, _make_context())

    update.callback_query.answer.assert_awaited()
    bot.openai.get_chat_response.assert_not_called()
    bot.openai.get_chat_response_stream.assert_not_called()
    assert bot.inline_queries_cache == {"abc": "expensive prompt"}


@pytest.mark.asyncio
async def test_unauthorized_settings_callback_does_not_mutate_user_settings():
    bot = _make_bot(allowed_user_ids="111")
    update = FakeCallbackUpdate("settings:lang_set:ru", user_id=999)

    await bot.handle_settings_callback(update, _make_context())

    update.callback_query.answer.assert_awaited_once()
    update.callback_query.edit_message_text.assert_awaited_once_with(
        text=localized_text("access_denied_command", "en")
    )
    bot.db.get_user_settings.assert_not_called()
    bot.db.save_user_settings.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_busy_callback_does_not_touch_pending_queue():
    bot = _make_bot(allowed_user_ids="111")
    bot._cleanup_pending_busy_messages = MagicMock()
    update = FakeCallbackUpdate("busymsg:queue:token-1", user_id=999)

    await bot.handle_busy_message_callback(update, _make_context())

    update.callback_query.answer.assert_awaited_once()
    update.callback_query.edit_message_text.assert_awaited_once_with(
        text=localized_text("access_denied_command", "en")
    )
    bot._cleanup_pending_busy_messages.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_plugins_command_does_not_build_menu():
    bot = _make_bot(allowed_user_ids="111")
    bot.openai.plugin_manager.build_bot_commands = MagicMock(return_value={
        "plugin_commands": [],
        "menu_entries": [],
    })
    bot._build_plugins_menu = MagicMock()
    update = FakeMessageUpdate(user_id=999)

    await bot.handle_plugins_menu(update, _make_context())

    update.message.reply_text.assert_awaited_once()
    bot.openai.plugin_manager.build_bot_commands.assert_not_called()
    bot._build_plugins_menu.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_stats_command_does_not_read_usage_or_sessions():
    bot = _make_bot(allowed_user_ids="111")
    update = FakeMessageUpdate(user_id=999)

    await bot.stats(update, _make_context())

    update.message.reply_text.assert_awaited_once()
    bot.db.list_user_sessions.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_resend_command_does_not_replay_prompt():
    bot = _make_bot(allowed_user_ids="111")
    bot.last_message = {1234: "repeat me"}
    bot.prompt = AsyncMock()
    update = FakeMessageUpdate(user_id=999)

    await bot.resend(update, _make_context())

    update.message.reply_text.assert_awaited_once()
    bot.prompt.assert_not_called()
    assert bot.last_message == {1234: "repeat me"}


@pytest.mark.asyncio
async def test_unauthorized_plugin_command_does_not_call_handler():
    bot = _make_bot(allowed_user_ids="111")
    plugin_handler = AsyncMock()
    update = FakeMessageUpdate(user_id=999)

    await bot.handle_plugin_command(
        update,
        _make_context(),
        {"handler": plugin_handler, "plugin_name": "alpha"},
    )

    update.message.reply_text.assert_awaited_once()
    plugin_handler.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_plugin_menu_args_reply_does_not_resolve_command():
    bot = _make_bot(allowed_user_ids="111")
    bot._get_plugin_command = MagicMock()
    update = FakeMessageUpdate(user_id=999)
    update.message.reply_to_message = SimpleNamespace(message_id=77)
    context = _make_context()
    context.user_data = {
        "plugin_menu_pending": {
            "plugin": "alpha",
            "cmd_id": "0",
            "prompt_message_id": 77,
        },
    }

    await bot.handle_plugin_menu_args_reply(update, context)

    update.message.reply_text.assert_awaited_once()
    bot._get_plugin_command.assert_not_called()
    assert "plugin_menu_pending" in context.user_data


@pytest.mark.asyncio
async def test_authorized_command_wrapper_rejects_unauthorized_user_before_callback():
    bot = _make_bot(allowed_user_ids="111")
    callback = AsyncMock()
    handler = bot._authorized_command_handler("demo", callback)
    update = FakeMessageUpdate(user_id=999)

    await handler.callback(update, _make_context())

    update.message.reply_text.assert_awaited_once()
    callback.assert_not_called()


@pytest.mark.asyncio
async def test_authorized_callback_wrapper_rejects_unauthorized_user_before_callback():
    bot = _make_bot(allowed_user_ids="111")
    callback = AsyncMock()
    handler = bot._authorized_callback_query_handler(callback, pattern="^demo")
    update = FakeCallbackUpdate("demo", user_id=999)

    await handler.callback(update, _make_context())

    update.callback_query.answer.assert_awaited_once()
    update.callback_query.edit_message_text.assert_awaited_once_with(
        text=localized_text("access_denied_command", "en")
    )
    callback.assert_not_called()


def test_auto_language_detects_persists_and_caches_first_contact():
    bot = _make_bot(allowed_user_ids="*")
    bot.config["bot_language"] = "auto"
    update = FakeCallbackUpdate("settings:root", user_id=999, language_code="de-DE")

    assert bot._get_user_language(update) == "de"
    bot.db.save_user_settings.assert_called_once_with(999, {"language": "de"})

    bot.db.get_user_settings.reset_mock()
    update.effective_user.language_code = "ru"

    assert bot._get_user_language(update) == "de"
    bot.db.get_user_settings.assert_not_called()


def test_explicit_bot_language_is_default_and_user_setting_can_override():
    bot = _make_bot(allowed_user_ids="*")
    bot.config["bot_language"] = "ru"
    update = FakeCallbackUpdate("settings:root", user_id=999, language_code="de")

    assert bot._get_user_language(update) == "ru"
    bot.db.save_user_settings.assert_not_called()

    bot._user_language_cache.clear()
    bot.db.get_user_settings.return_value = {"language": "de"}

    assert bot._get_user_language(update) == "de"


def test_language_settings_menu_uses_two_languages_per_row():
    bot = _make_bot(allowed_user_ids="*")
    bot.config["bot_language"] = "auto"

    markup = bot._build_language_settings_menu(page=0, current_language="ru")
    rows = markup.inline_keyboard
    language_rows = rows[:5]

    assert all(len(row) == 2 for row in language_rows)
    assert rows[-1][0].callback_data == "settings:root"


@pytest.mark.asyncio
async def test_settings_language_selection_saves_user_language():
    bot = _make_bot(allowed_user_ids="*")
    bot.config["bot_language"] = "en"
    bot.db.get_user_settings.return_value = {}
    update = FakeCallbackUpdate("settings:lang_set:ru", user_id=999, language_code="en")

    await bot.handle_settings_callback(update, _make_context())

    assert bot._user_language_cache[999] == "ru"
    bot.db.save_user_settings.assert_called_with(999, {"language": "ru"})
    update.callback_query.answer.assert_any_await("Язык сохранён.")
    text = update.callback_query.edit_message_text.await_args.kwargs["text"]
    assert "Настройки" in text


@pytest.mark.asyncio
async def test_settings_tts_model_selection_saves_user_model():
    bot = _make_bot(allowed_user_ids="*")
    bot.db.get_user_settings.return_value = {}
    update = FakeCallbackUpdate("settings:tts_model_set:0:1", user_id=999)

    await bot.handle_settings_callback(update, _make_context())

    bot.db.save_user_settings.assert_called_with(999, {"tts_model": "tts-b"})
    update.callback_query.answer.assert_any_await("TTS model saved.")


@pytest.mark.asyncio
async def test_settings_tts_voice_selection_saves_user_voice():
    bot = _make_bot(allowed_user_ids="*")
    bot.db.get_user_settings.return_value = {}
    update = FakeCallbackUpdate("settings:tts_voice_set:0:1", user_id=999)

    await bot.handle_settings_callback(update, _make_context())

    bot.db.save_user_settings.assert_called_with(999, {"tts_voice": "bob"})
    update.callback_query.answer.assert_any_await("TTS voice saved.")


@pytest.mark.asyncio
async def test_settings_plugin_toggle_saves_disabled_plugins():
    bot = _make_bot(allowed_user_ids="*")
    bot.openai.plugin_manager = FakeSettingsPluginManager()
    bot.db.get_user_settings.return_value = {}
    update = FakeCallbackUpdate("settings:plugin_toggle:0:0", user_id=999)

    await bot.handle_settings_callback(update, _make_context())

    bot.db.save_user_settings.assert_called_with(999, {"disabled_plugins": ["alpha"]})
    update.callback_query.answer.assert_any_await("Plugin disabled: alpha")


@pytest.mark.asyncio
async def test_settings_plugin_menu_has_description_button():
    bot = _make_bot(allowed_user_ids="*")
    bot.openai.plugin_manager = FakeSettingsPluginManager()
    bot.db.get_user_settings.return_value = {}

    markup = await bot._build_plugin_settings_menu(page=0, user_id=999)

    first_row = markup.inline_keyboard[0]
    assert first_row[0].text == "On: alpha"
    assert first_row[0].callback_data == "settings:plugin_toggle:0:0"
    assert first_row[1].text == "ℹ️"
    assert first_row[1].callback_data == "settings:plugin_info:0:0"


@pytest.mark.asyncio
async def test_settings_plugin_info_shows_description_with_back_button():
    bot = _make_bot(allowed_user_ids="*")
    bot.openai.plugin_manager = FakeSettingsPluginManager()
    update = FakeCallbackUpdate("settings:plugin_info:0:0", user_id=999)

    await bot.handle_settings_callback(update, _make_context())

    text = update.callback_query.edit_message_text.await_args.kwargs["text"]
    reply_markup = update.callback_query.edit_message_text.await_args.kwargs["reply_markup"]
    assert "Plugin: alpha" in text
    assert "Alpha source" in text
    assert "Alpha help text" in text
    assert "/alpha: Run alpha command" in text
    assert "alpha.search: Search alpha records" in text
    assert reply_markup.inline_keyboard[0][0].callback_data == "settings:plugins:0"


@pytest.mark.asyncio
async def test_settings_plugin_toggle_can_reenable_plugin():
    bot = _make_bot(allowed_user_ids="*")
    bot.openai.plugin_manager = FakeSettingsPluginManager()
    bot.db.get_user_settings.return_value = {"disabled_plugins": ["alpha"]}
    update = FakeCallbackUpdate("settings:plugin_toggle:0:0", user_id=999)

    await bot.handle_settings_callback(update, _make_context())

    bot.db.save_user_settings.assert_called_with(999, {"disabled_plugins": []})
    update.callback_query.answer.assert_any_await("Plugin enabled: alpha")


@pytest.mark.asyncio
async def test_settings_skill_toggle_saves_disabled_skills():
    bot = _make_bot(allowed_user_ids="*")
    bot.openai.plugin_manager = FakeSettingsPluginManager()
    bot.db.get_user_settings.return_value = {}
    update = FakeCallbackUpdate("settings:skill_toggle:0:1", user_id=999)

    await bot.handle_settings_callback(update, _make_context())

    bot.db.save_user_settings.assert_called_with(999, {"disabled_skills": ["pptx"]})
    update.callback_query.answer.assert_any_await("Skill disabled: pptx")


@pytest.mark.asyncio
async def test_settings_skill_menu_has_description_button():
    bot = _make_bot(allowed_user_ids="*")
    bot.openai.plugin_manager = FakeSettingsPluginManager()
    bot.db.get_user_settings.return_value = {}

    markup = await bot._build_skill_settings_menu(page=0, user_id=999)

    first_row = markup.inline_keyboard[0]
    assert first_row[0].text == "On: docx"
    assert first_row[0].callback_data == "settings:skill_toggle:0:0"
    assert first_row[1].text == "ℹ️"
    assert first_row[1].callback_data == "settings:skill_info:0:0"


@pytest.mark.asyncio
async def test_settings_skill_info_shows_description_with_back_button():
    bot = _make_bot(allowed_user_ids="*")
    bot.openai.plugin_manager = FakeSettingsPluginManager()
    update = FakeCallbackUpdate("settings:skill_info:0:1", user_id=999)

    await bot.handle_settings_callback(update, _make_context())

    text = update.callback_query.edit_message_text.await_args.kwargs["text"]
    reply_markup = update.callback_query.edit_message_text.await_args.kwargs["reply_markup"]
    assert "Skill: PowerPoint (pptx)" in text
    assert "Create and edit presentations" in text
    assert "qa-pptx.sh" in text
    assert "deck_generator" in text
    assert "references/checklist.md" in text
    assert reply_markup.inline_keyboard[0][0].callback_data == "settings:skill_page:0"
