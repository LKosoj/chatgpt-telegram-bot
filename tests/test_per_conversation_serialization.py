import asyncio
import importlib.util
import sys
import types
from collections import defaultdict
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

from bot import telegram_bot  # noqa: E402
from bot.request_context import RequestContext  # noqa: E402
from bot.telegram_bot import ChatGPTTelegramBot  # noqa: E402

for _module_name in _INSERTED_MODULES:
    sys.modules.pop(_module_name, None)


class FakePluginManager:
    db = None

    def get_plugin(self, name):
        return None

    def set_db(self, db):
        self.db = db

    def disabled_plugins_for_user(self, user_id):
        return set()

    def is_plugin_disabled_for_user(self, plugin_name, user_id):
        return False

    async def dispatch_observe(self, event_name, payload, *, user_id=None):
        return None

    async def dispatch_blocking(self, event_name, payload, *, user_id=None):
        return []


class SequencingDB:
    def __init__(self):
        self.active_by_chat = defaultdict(int)
        self.overlaps = []
        self.load_calls = []
        self.save_calls = []
        self.created_sessions = []
        self.active_session_id = "session-1"
        self.create_session_result = "session-new"
        self.oldest_session_ids = []
        self.oldest_session_requests = []
        self.oldest_session_excludes = []
        self.deleted_session_ids = []

    def get_active_session_id(self, user_id):
        return self.active_session_id

    def get_conversation_context(self, chat_id, session_id=None, *args, **kwargs):
        self.load_calls.append(chat_id)
        if self.active_by_chat[chat_id]:
            self.overlaps.append(chat_id)
        self.active_by_chat[chat_id] += 1
        return {"messages": []}, "HTML", 0.8, 100, session_id or self.active_session_id

    def save_conversation_context(self, chat_id, *args, **kwargs):
        self.save_calls.append(chat_id)
        self.active_by_chat[chat_id] -= 1

    def get_user_images(self, user_id, chat_id, limit=1):
        return []

    def get_oldest_session_ids_for_limit(self, user_id, max_sessions=None, exclude_session_ids=None):
        excluded = set(exclude_session_ids or [])
        self.oldest_session_requests.append({
            "user_id": user_id,
            "max_sessions": max_sessions,
            "exclude_session_ids": tuple(exclude_session_ids or ()),
        })
        self.oldest_session_excludes.append(tuple(exclude_session_ids or ()))
        return [session_id for session_id in self.oldest_session_ids if session_id not in excluded]

    def delete_sessions_by_ids(self, user_id, session_ids):
        self.deleted_session_ids.append((user_id, list(session_ids)))
        deleted = set(session_ids)
        self.oldest_session_ids = [
            session_id for session_id in self.oldest_session_ids
            if session_id not in deleted
        ]
        return True

    def create_session(self, user_id, **kwargs):
        self.created_sessions.append({"user_id": user_id, **kwargs})
        if self.create_session_result:
            self.active_session_id = self.create_session_result
        return self.create_session_result


class DelayedOpenAI:
    def __init__(self, db):
        self.db = db
        self.plugin_manager = FakePluginManager()
        self.active_calls = 0
        self.max_parallel_calls = 0
        self.started = asyncio.Event()
        self.requests = []
        self.model_calls = []
        self.cleared_states = []

    def get_current_model(self, user_id, session_id=None):
        self.model_calls.append((user_id, session_id))
        return "gpt-test"

    async def get_chat_response(self, *, chat_id, query, **kwargs):
        self.requests.append({"chat_id": chat_id, "query": query, **kwargs})
        self.active_calls += 1
        self.max_parallel_calls = max(self.max_parallel_calls, self.active_calls)
        self.started.set()
        self.db.get_conversation_context(chat_id)
        try:
            await asyncio.sleep(0.05)
            return f"response for {query}", 1
        finally:
            self.db.save_conversation_context(chat_id)
            self.active_calls -= 1

    def _clear_chat_state(self, state_key):
        self.cleared_states.append(state_key)


class FakeMessage:
    def __init__(self, chat_id, user_id, message_id, text):
        self.chat_id = chat_id
        self.message_id = message_id
        self.text = text
        self.via_bot = None
        self.reply_to_message = None
        self.from_user = SimpleNamespace(id=user_id, name=f"User {user_id}")
        self.is_topic_message = False
        self.message_thread_id = None
        self.reply_text_calls = []

    async def reply_text(self, **kwargs):
        self.reply_text_calls.append(kwargs)
        return SimpleNamespace(chat_id=self.chat_id, message_id=900 + len(self.reply_text_calls))

    def parse_entities(self, types=None):
        return {}


class FakeUpdate:
    def __init__(self, message):
        self.edited_message = None
        self.message = message
        self.effective_message = message
        self.effective_chat = SimpleNamespace(id=message.chat_id, type="private")
        self.effective_user = message.from_user
        self.callback_query = None


class FakeCallbackQuery:
    def __init__(self, data, chat_id=1234, user_id=42):
        self.data = data
        self.from_user = SimpleNamespace(id=user_id, name=f"User {user_id}")
        self.message = SimpleNamespace(chat_id=chat_id)
        self.answer_calls = []
        self.edit_message_text_calls = []

    async def answer(self, *args, **kwargs):
        self.answer_calls.append({"args": args, "kwargs": kwargs})

    async def edit_message_text(self, **kwargs):
        self.edit_message_text_calls.append(kwargs)


class FakeCallbackUpdate:
    def __init__(self, data, chat_id=1234, user_id=42):
        self.callback_query = FakeCallbackQuery(data, chat_id=chat_id, user_id=user_id)
        self.message = None
        self.effective_message = self.callback_query.message
        self.effective_chat = SimpleNamespace(id=chat_id, type="private")
        self.effective_user = self.callback_query.from_user


def _make_context():
    return SimpleNamespace(bot=SimpleNamespace(id=999))


def _make_bot():
    db = SequencingDB()
    openai = DelayedOpenAI(db)
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {
        "allowed_user_ids": "*",
        "bot_language": "en",
        "enable_image_generation": False,
        "enable_quoting": False,
        "enable_vision": False,
        "stream": False,
        "token_price": 0.0,
        "max_sessions": 5,
    }
    bot.db = db
    bot.openai = openai
    bot.last_message = {}
    bot.usage = {}
    bot.buffer_lock = asyncio.Lock()
    bot.message_buffer = {}
    bot.pending_busy_messages = {}
    bot.pending_busy_message_ttl = 600
    bot._conversation_locks = {}
    bot._conversation_locks_guard = asyncio.Lock()
    bot._classify_reply_intent = AsyncMock(return_value=None)
    return bot, db, openai


async def _run_without_indicator(update, context, coroutine, chat_action, is_inline=False):
    return await coroutine()


@pytest.mark.asyncio
async def test_reentrant_process_buffer_does_not_clear_active_processing_flag():
    bot = object.__new__(ChatGPTTelegramBot)
    bot.buffer_lock = asyncio.Lock()
    bot.message_buffer = {
        1234: {
            "messages": [],
            "processing": True,
            "timer": None,
        }
    }

    await bot.process_buffer(1234)

    assert bot.message_buffer[1234]["processing"] is True


@pytest.mark.asyncio
async def test_same_conversation_key_updates_are_serialized(monkeypatch):
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", _run_without_indicator)
    bot, db, openai = _make_bot()
    first = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=1, text="first"))
    second = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=2, text="second"))

    await asyncio.gather(
        bot.process_message("first", first, _make_context()),
        bot.process_message("second", second, _make_context()),
    )

    assert db.overlaps == []
    request_contexts = [request["request_context"] for request in openai.requests]
    assert all(isinstance(context, RequestContext) for context in request_contexts)
    assert {
        (context.message_id, context.user_id)
        for context in request_contexts
    } == {(1, 42), (2, 42)}


@pytest.mark.asyncio
async def test_different_conversation_keys_can_process_independently(monkeypatch):
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", _run_without_indicator)
    bot, db, openai = _make_bot()
    first = FakeUpdate(FakeMessage(chat_id=1001, user_id=101, message_id=1, text="first"))
    second = FakeUpdate(FakeMessage(chat_id=2002, user_id=202, message_id=1, text="second"))

    await asyncio.gather(
        bot.process_message("first", first, _make_context()),
        bot.process_message("second", second, _make_context()),
    )

    assert db.overlaps == []
    assert set(db.load_calls) == {1001, 2002}
    assert set(db.save_calls) == {1001, 2002}
    assert openai.max_parallel_calls >= 2


@pytest.mark.asyncio
async def test_process_message_pins_active_session_before_later_session_change(monkeypatch):
    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", _run_without_indicator)
    bot, db, openai = _make_bot()

    async def classify_and_switch_session(update, prompt):
        db.active_session_id = "session-new"
        return None

    bot._classify_reply_intent = AsyncMock(side_effect=classify_and_switch_session)
    update = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=3, text="first"))

    await bot.process_message("first", update, _make_context())

    request = openai.requests[0]
    assert request["session_id"] == "session-1"
    assert request["request_context"].session_id == "session-1"
    assert openai.model_calls == [(42, "session-1")]


@pytest.mark.asyncio
async def test_prompt_asks_for_busy_message_action_instead_of_buffering():
    bot, _db, _openai = _make_bot()
    bot.check_allowed_and_within_budget = AsyncMock(return_value=True)
    bot.message_buffer[1234] = {
        "messages": [],
        "processing": True,
        "timer": None,
    }
    update = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=7, text="second"))

    await bot.prompt(update, _make_context())

    assert bot.message_buffer[1234]["messages"] == []
    assert len(bot.pending_busy_messages) == 1
    reply = update.message.reply_text_calls[0]
    assert "processed" in reply["text"]
    keyboard = reply["reply_markup"].inline_keyboard
    assert keyboard[0][0].callback_data.startswith("busymsg:queue:")
    assert keyboard[0][1].callback_data.startswith("busymsg:new_session:")


@pytest.mark.asyncio
async def test_busy_message_queue_callback_adds_pending_message_to_buffer():
    bot, _db, _openai = _make_bot()
    pending_update = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=8, text="queued"))
    token = bot._store_pending_busy_message(
        chat_id=1234,
        user_id=42,
        prompt="queued",
        update=pending_update,
        context=_make_context(),
        message_id=8,
    )
    bot.message_buffer[1234] = {
        "messages": [],
        "processing": True,
        "timer": None,
    }

    callback_update = FakeCallbackUpdate(f"busymsg:queue:{token}", chat_id=1234, user_id=42)

    await bot.handle_busy_message_callback(callback_update, _make_context())

    assert token not in bot.pending_busy_messages
    assert bot.message_buffer[1234]["messages"][0]["text"] == "queued"
    assert callback_update.callback_query.edit_message_text_calls[-1]["text"] == "Your message has been queued."


@pytest.mark.asyncio
async def test_busy_message_new_session_callback_runs_with_separate_session_keys():
    bot, db, openai = _make_bot()
    bot.process_message = AsyncMock()
    pending_update = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=9, text="parallel"))
    token = bot._store_pending_busy_message(
        chat_id=1234,
        user_id=42,
        prompt="parallel",
        update=pending_update,
        context=_make_context(),
        message_id=9,
    )
    callback_update = FakeCallbackUpdate(f"busymsg:new_session:{token}", chat_id=1234, user_id=42)

    await bot.handle_busy_message_callback(callback_update, _make_context())
    for _ in range(5):
        await asyncio.sleep(0)
        if token not in bot.pending_busy_messages:
            break

    assert token not in bot.pending_busy_messages
    assert db.created_sessions[0]["user_id"] == 42
    assert callback_update.callback_query.edit_message_text_calls[-1]["text"] == (
        "Started processing your message in a new session."
    )
    bot.process_message.assert_awaited_once()
    _, kwargs = bot.process_message.await_args
    assert kwargs["session_id"] == "session-new"
    assert kwargs["conversation_lock_key"] == (42, "session-new")
    assert kwargs["conversation_state_key"] == (42, "session-new")
    assert openai.cleared_states == [(42, "session-new")]


@pytest.mark.asyncio
async def test_busy_message_new_session_serializes_prune_and_create():
    bot, db, _openai = _make_bot()
    bot.process_message = AsyncMock()
    active_prunes = 0
    max_parallel_prunes = 0

    async def dispatch_and_prune(*args, **kwargs):
        nonlocal active_prunes, max_parallel_prunes
        active_prunes += 1
        max_parallel_prunes = max(max_parallel_prunes, active_prunes)
        await asyncio.sleep(0.01)
        active_prunes -= 1

    session_counter = 0

    def create_session(user_id, **kwargs):
        nonlocal session_counter
        session_counter += 1
        session_id = f"session-new-{session_counter}"
        db.created_sessions.append({"user_id": user_id, **kwargs})
        return session_id

    bot._dispatch_and_delete_oldest_sessions_for_limit = dispatch_and_prune
    db.create_session = create_session

    pending_items = []
    for message_id in (20, 21):
        pending_update = FakeUpdate(
            FakeMessage(chat_id=1234, user_id=42, message_id=message_id, text=f"parallel {message_id}")
        )
        token = bot._store_pending_busy_message(
            chat_id=1234,
            user_id=42,
            prompt=f"parallel {message_id}",
            update=pending_update,
            context=_make_context(),
            message_id=message_id,
        )
        pending_items.append(dict(bot.pending_busy_messages[token]))

    session_ids = await asyncio.gather(
        *(bot._start_pending_busy_message_in_new_session(pending) for pending in pending_items)
    )
    await asyncio.sleep(0)

    assert session_ids == ["session-new-1", "session-new-2"]
    assert max_parallel_prunes == 1


@pytest.mark.asyncio
async def test_busy_message_new_session_pruning_excludes_running_parallel_sessions():
    bot, db, _openai = _make_bot()
    bot.config["max_sessions"] = 1
    release_processing = asyncio.Event()
    started_processing = asyncio.Event()

    async def process_message(*args, **kwargs):
        started_processing.set()
        await release_processing.wait()

    session_counter = 0

    def create_session(user_id, **kwargs):
        nonlocal session_counter
        session_counter += 1
        session_id = f"session-new-{session_counter}"
        db.created_sessions.append({"user_id": user_id, **kwargs})
        return session_id

    bot.process_message = AsyncMock(side_effect=process_message)
    db.create_session = create_session

    pending_items = []
    for message_id in (30, 31):
        pending_update = FakeUpdate(
            FakeMessage(chat_id=1234, user_id=42, message_id=message_id, text=f"parallel {message_id}")
        )
        token = bot._store_pending_busy_message(
            chat_id=1234,
            user_id=42,
            prompt=f"parallel {message_id}",
            update=pending_update,
            context=_make_context(),
            message_id=message_id,
        )
        pending_items.append(dict(bot.pending_busy_messages[token]))

    first_session_id = await bot._start_pending_busy_message_in_new_session(pending_items[0])
    await started_processing.wait()

    db.oldest_session_ids = ["session-1", first_session_id, "session-old"]
    second_session_id = await bot._start_pending_busy_message_in_new_session(pending_items[1])

    assert second_session_id == "session-new-2"
    assert set(db.oldest_session_excludes[-1]) == {"session-1", first_session_id}
    assert db.deleted_session_ids[-1] == (42, ["session-old"])

    release_processing.set()
    for _ in range(10):
        if len(db.deleted_session_ids) >= 2 and not bot._protected_parallel_session_ids(42):
            break
        await asyncio.sleep(0)

    deleted_after_completion = [
        session_id
        for _user_id, session_ids in db.deleted_session_ids[1:]
        for session_id in session_ids
    ]
    assert first_session_id in deleted_after_completion
    assert any(
        request["max_sessions"] == bot.config["max_sessions"] + 1
        for request in db.oldest_session_requests[2:]
    )
    assert not bot._protected_parallel_session_ids(42)


@pytest.mark.asyncio
async def test_busy_message_completion_prune_excludes_original_inflight_session():
    bot, db, _openai = _make_bot()
    bot.config["max_sessions"] = 1
    original_started = asyncio.Event()
    release_original = asyncio.Event()
    parallel_started = asyncio.Event()
    release_parallel = asyncio.Event()

    async def process_message_locked(*args, **kwargs):
        session_id = kwargs["session_id"]
        if session_id == "session-1":
            original_started.set()
            await release_original.wait()
            return None
        parallel_started.set()
        await release_parallel.wait()
        return None

    session_counter = 0

    def create_session(user_id, **kwargs):
        nonlocal session_counter
        session_counter += 1
        session_id = f"session-new-{session_counter}"
        db.created_sessions.append({"user_id": user_id, **kwargs})
        return session_id

    bot._process_message_locked = AsyncMock(side_effect=process_message_locked)
    db.create_session = create_session

    original_update = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=40, text="original"))
    original_task = asyncio.create_task(
        bot.process_message("original", original_update, _make_context())
    )
    await original_started.wait()

    pending_update = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=41, text="parallel"))
    token = bot._store_pending_busy_message(
        chat_id=1234,
        user_id=42,
        prompt="parallel",
        update=pending_update,
        context=_make_context(),
        message_id=41,
    )
    pending = dict(bot.pending_busy_messages[token])

    db.oldest_session_ids = ["session-1", "session-old"]
    session_id = await bot._start_pending_busy_message_in_new_session(pending)
    await parallel_started.wait()
    db.oldest_session_ids = ["session-1", session_id]
    release_parallel.set()

    for _ in range(10):
        await asyncio.sleep(0)
        completion_prunes = [
            request
            for request in db.oldest_session_requests
            if request["max_sessions"] == bot.config["max_sessions"] + 1
        ]
        if completion_prunes:
            break

    completion_prunes = [
        request
        for request in db.oldest_session_requests
        if request["max_sessions"] == bot.config["max_sessions"] + 1
    ]
    assert completion_prunes[-1]["exclude_session_ids"] == ("session-1",)
    deleted_after_completion = [
        session_id
        for _user_id, session_ids in db.deleted_session_ids[1:]
        for session_id in session_ids
    ]
    assert "session-1" not in deleted_after_completion

    release_original.set()
    await original_task
    assert not bot._protected_session_ids(42)


def test_inflight_session_protection_is_reference_counted():
    bot, _db, _openai = _make_bot()

    bot._remember_inflight_session(42, "session-1")
    bot._remember_inflight_session(42, "session-1")
    bot._forget_inflight_session(42, "session-1")

    assert bot._protected_session_ids(42) == {"session-1"}

    bot._forget_inflight_session(42, "session-1")

    assert not bot._protected_session_ids(42)


@pytest.mark.asyncio
async def test_busy_message_new_session_pruning_excludes_active_session():
    bot, db, _openai = _make_bot()
    db.oldest_session_ids = ["session-1", "session-old"]
    bot.process_message = AsyncMock()
    pending_update = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=10, text="parallel"))
    token = bot._store_pending_busy_message(
        chat_id=1234,
        user_id=42,
        prompt="parallel",
        update=pending_update,
        context=_make_context(),
        message_id=10,
    )
    db.active_session_id = "session-other"
    callback_update = FakeCallbackUpdate(f"busymsg:new_session:{token}", chat_id=1234, user_id=42)

    await bot.handle_busy_message_callback(callback_update, _make_context())
    await asyncio.sleep(0)

    assert db.oldest_session_excludes[0] == ("session-1",)
    assert db.deleted_session_ids[0] == (42, ["session-old"])


@pytest.mark.asyncio
async def test_busy_message_new_session_reports_background_failure():
    bot, _db, _openai = _make_bot()
    bot.process_message = AsyncMock(side_effect=RuntimeError("boom"))
    pending_update = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=12, text="parallel"))
    token = bot._store_pending_busy_message(
        chat_id=1234,
        user_id=42,
        prompt="parallel",
        update=pending_update,
        context=_make_context(),
        message_id=12,
    )
    callback_update = FakeCallbackUpdate(f"busymsg:new_session:{token}", chat_id=1234, user_id=42)

    await bot.handle_busy_message_callback(callback_update, _make_context())
    for _ in range(3):
        await asyncio.sleep(0)

    assert callback_update.callback_query.edit_message_text_calls[-1]["text"] == (
        "Started processing your message in a new session."
    )
    assert pending_update.message.reply_text_calls[-1]["text"].startswith("Failed to get response")
    assert token in bot.pending_busy_messages
    assert "claimed" not in bot.pending_busy_messages[token]
    retry_keyboard = pending_update.message.reply_text_calls[-1]["reply_markup"].inline_keyboard
    assert retry_keyboard[0][0].callback_data == f"busymsg:queue:{token}"
    assert retry_keyboard[0][1].callback_data == f"busymsg:new_session:{token}"


@pytest.mark.asyncio
async def test_busy_message_new_session_failure_keeps_pending_message_retryable():
    bot, db, _openai = _make_bot()
    db.create_session_result = None
    pending_update = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=10, text="retry"))
    token = bot._store_pending_busy_message(
        chat_id=1234,
        user_id=42,
        prompt="retry",
        update=pending_update,
        context=_make_context(),
        message_id=10,
    )
    callback_update = FakeCallbackUpdate(f"busymsg:new_session:{token}", chat_id=1234, user_id=42)

    await bot.handle_busy_message_callback(callback_update, _make_context())

    assert token in bot.pending_busy_messages
    assert "claimed" not in bot.pending_busy_messages[token]
    assert callback_update.callback_query.edit_message_text_calls[-1]["text"] == (
        "Failed to create a new session. Please try again later."
    )


@pytest.mark.asyncio
async def test_parallel_session_keeps_early_plugin_prompt_route_enabled():
    bot, _db, openai = _make_bot()
    update = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=11, text="plugin prompt"))
    bot._try_handle_plugin_prompt = AsyncMock(return_value=True)

    await bot.process_message(
        "plugin prompt",
        update,
        _make_context(),
        session_id="session-new",
        conversation_lock_key=(42, "session-new"),
        conversation_state_key=(42, "session-new"),
    )

    bot._try_handle_plugin_prompt.assert_awaited_once()
    assert openai.requests == []


@pytest.mark.asyncio
async def test_early_plugin_prompt_route_does_not_create_empty_session():
    bot, db, openai = _make_bot()
    db.active_session_id = None
    update = FakeUpdate(FakeMessage(chat_id=1234, user_id=42, message_id=12, text="plugin prompt"))
    bot._try_handle_plugin_prompt = AsyncMock(return_value=True)

    await bot.process_message("plugin prompt", update, _make_context())

    bot._try_handle_plugin_prompt.assert_awaited_once()
    assert db.load_calls == []
    assert db.created_sessions == []
    assert openai.requests == []
