import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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


class FakeAgentTools:
    def __init__(self, terminal_clear_result=True):
        self.clear_calls = []
        self.terminal_clear_calls = []
        self.plan_calls = []
        self.terminal_clear_result = terminal_clear_result

    def clear_plan_tasks(self, chat_id=None, user_id=None):
        self.clear_calls.append((chat_id, user_id))
        return True

    def clear_terminal_plan_tasks(self, chat_id=None, user_id=None):
        self.terminal_clear_calls.append((chat_id, user_id))
        return self.terminal_clear_result

    def get_plan_tasks(self, chat_id=None, user_id=None):
        self.plan_calls.append((chat_id, user_id))
        return []


class FakePluginManager:
    def __init__(self, agent_tools=None):
        self.agent_tools = agent_tools

    def get_plugin(self, name):
        if name == "agent_tools":
            return self.agent_tools
        return None


class FakeOpenAI:
    def __init__(self, chunks, agent_tools=None):
        self.chunks = chunks
        self.plugin_manager = FakePluginManager(agent_tools)
        self.stream_requests = []

    def get_current_model(self, user_id):
        return "gpt-test"

    def get_chat_response_stream(self, **kwargs):
        self.stream_requests.append(kwargs)

        async def stream():
            for chunk in self.chunks:
                yield chunk

        return stream()


class FakeDB:
    def __init__(self, conversation_context):
        self.conversation_context = conversation_context
        self.calls = []

    def get_conversation_context(self, chat_id):
        self.calls.append(chat_id)
        return self.conversation_context


class FakeTelegramFile:
    def __init__(self, content=b"file-content"):
        self.content = content
        self.downloaded_paths = []

    async def download_to_drive(self, path):
        self.downloaded_paths.append(path)
        Path(path).write_bytes(self.content)


class FakeMessage:
    def __init__(self, chat_id=1234, user_id=42, message_id=7, reply_side_effects=None):
        self.chat_id = chat_id
        self.message_id = message_id
        self.text = "hello"
        self.reply_to_message = None
        self.from_user = SimpleNamespace(id=user_id, name="Alice")
        self.is_topic_message = False
        self.message_thread_id = None
        self.reply_text_calls = []
        self.reply_chat_action_calls = []
        self._reply_side_effects = list(reply_side_effects or [])

    async def reply_chat_action(self, **kwargs):
        self.reply_chat_action_calls.append(kwargs)

    async def reply_text(self, **kwargs):
        self.reply_text_calls.append(kwargs)
        if self._reply_side_effects:
            effect = self._reply_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        return SimpleNamespace(chat_id=self.chat_id, message_id=900 + len(self.reply_text_calls))


class FakeUpdate:
    def __init__(self, message):
        self.message = message
        self.effective_message = message
        self.effective_chat = SimpleNamespace(id=message.chat_id, type="private")
        self.effective_user = message.from_user


def _make_context(telegram_file=None):
    bot = SimpleNamespace(
        id=999,
        delete_message=AsyncMock(),
    )
    if telegram_file is not None:
        bot.get_file = AsyncMock(return_value=telegram_file)
    return SimpleNamespace(bot=bot, user_data={})


def _make_bot(chunks, conversation_context, agent_tools=None):
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {
        "allowed_user_ids": "*",
        "bot_language": "en",
        "enable_image_generation": False,
        "enable_quoting": False,
        "enable_vision": False,
        "stream": True,
        "token_price": 0.0,
    }
    bot.db = FakeDB(conversation_context)
    bot.openai = FakeOpenAI(chunks, agent_tools)
    bot.last_message = {}
    bot.usage = {}
    bot._classify_reply_intent = AsyncMock(return_value=None)
    return bot


@pytest.mark.asyncio
async def test_process_message_clears_terminal_plan_before_agent_request(monkeypatch):
    edit_message = AsyncMock()
    monkeypatch.setattr(telegram_bot, "edit_message_with_retry", edit_message)
    agent_tools = FakeAgentTools()
    bot = _make_bot(
        chunks=[
            ("Hello", "1"),
        ],
        conversation_context=({"messages": []}, "HTML", 0.8, 80, "session-1"),
        agent_tools=agent_tools,
    )

    await bot.process_message("hello", FakeUpdate(FakeMessage()), _make_context())

    assert agent_tools.terminal_clear_calls == [(1234, 42)]
    assert agent_tools.clear_calls == []


@pytest.mark.asyncio
async def test_handle_direct_result_clears_plan_after_success(monkeypatch):
    agent_tools = FakeAgentTools()
    bot = object.__new__(ChatGPTTelegramBot)
    bot.config = {"bot_language": "en"}
    bot.openai = SimpleNamespace(plugin_manager=FakePluginManager(agent_tools))
    bot._remember_sent_image_messages = Mock()
    bot._run_post_delivery_cleanup = AsyncMock()
    sent_messages = [SimpleNamespace(message_id=200)]
    monkeypatch.setattr(
        telegram_bot,
        "handle_direct_result",
        AsyncMock(return_value=sent_messages),
    )

    await bot._handle_direct_result(
        FakeUpdate(FakeMessage()),
        {"direct_result": {"kind": "final", "format": "mixed", "text": "ok"}},
    )

    assert agent_tools.clear_calls == [(1234, 42)]


@pytest.mark.asyncio
async def test_streaming_unpacks_conversation_context_5_tuple(monkeypatch):
    edit_message = AsyncMock()
    monkeypatch.setattr(telegram_bot, "edit_message_with_retry", edit_message)

    bot = _make_bot(
        chunks=[
            ("Hello", "not_finished"),
            ("Hello world", "3"),
        ],
        conversation_context=({"messages": []}, "HTML", 0.8, 80, "session-1"),
    )
    update = FakeUpdate(FakeMessage())

    await bot.process_message("hello", update, _make_context())

    assert bot.db.calls == [1234]
    assert len(update.effective_message.reply_text_calls) == 1
    assert update.effective_message.reply_text_calls[0]["text"] == "Hello"
    assert update.effective_message.reply_text_calls[0]["parse_mode"] == "HTML"
    request_context = bot.openai.stream_requests[0]["request_context"]
    assert isinstance(request_context, RequestContext)
    assert request_context.chat_id == 1234
    assert request_context.user_id == 42
    assert request_context.message_id == 7
    assert request_context.request_id == "1234_7"
    edit_message.assert_awaited()
    assert edit_message.await_args.kwargs["text"] == "Hello world"


@pytest.mark.asyncio
async def test_streaming_falls_back_to_html_when_parse_mode_is_none(monkeypatch):
    edit_message = AsyncMock()
    monkeypatch.setattr(telegram_bot, "edit_message_with_retry", edit_message)

    bot = _make_bot(
        chunks=[
            ("Hello", "1"),
        ],
        conversation_context=({"messages": []}, None, 0.8, 80, "session-1"),
    )
    update = FakeUpdate(FakeMessage())

    await bot.process_message("hello", update, _make_context())

    assert len(update.effective_message.reply_text_calls) == 1
    assert (
        update.effective_message.reply_text_calls[0]["parse_mode"]
        == telegram_bot.constants.ParseMode.HTML
    )
    edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_plain_text_image_edit_request_uses_normal_chat(monkeypatch):
    edit_message = AsyncMock()
    monkeypatch.setattr(telegram_bot, "edit_message_with_retry", edit_message)

    bot = _make_bot(
        chunks=[
            ("Plain chat response", "1"),
        ],
        conversation_context=({"messages": []}, "HTML", 0.8, 80, "session-1"),
    )
    bot.config["enable_image_generation"] = True
    bot._is_image_edit_request = Mock(side_effect=AssertionError("text-only edit routing was used"))
    bot._edit_image_from_context = AsyncMock()
    update = FakeUpdate(FakeMessage())

    await bot.process_message("отредактируй это изображение", update, _make_context())

    bot._is_image_edit_request.assert_not_called()
    bot._edit_image_from_context.assert_not_awaited()
    assert bot.openai.stream_requests[0]["query"] == "отредактируй это изображение"


@pytest.mark.asyncio
async def test_reply_to_image_edit_request_still_edits_image(monkeypatch):
    async def immediate_wrap(update, context, coroutine, chat_action="", is_inline=False):
        await coroutine()

    monkeypatch.setattr(telegram_bot, "wrap_with_indicator", immediate_wrap)

    bot = _make_bot(
        chunks=[],
        conversation_context=({"messages": []}, "HTML", 0.8, 80, "session-1"),
    )
    bot.config["enable_image_generation"] = True
    bot._classify_reply_intent = AsyncMock(return_value="image_edit")
    bot._edit_image_from_context = AsyncMock()
    message = FakeMessage()
    message.reply_to_message = SimpleNamespace(
        photo=[SimpleNamespace(file_id="telegram-image-file")],
        document=None,
    )
    update = FakeUpdate(message)

    await bot.process_message("добавь шапку", update, _make_context())

    bot._edit_image_from_context.assert_awaited_once_with(
        update,
        "добавь шапку",
        "telegram-image-file",
    )
    assert bot.openai.stream_requests == []


@pytest.mark.asyncio
async def test_reply_to_document_adds_downloaded_file_context_to_chat_request(monkeypatch):
    edit_message = AsyncMock()
    monkeypatch.setattr(telegram_bot, "edit_message_with_retry", edit_message)

    bot = _make_bot(
        chunks=[
            ("Done", "1"),
        ],
        conversation_context=({"messages": []}, "HTML", 0.8, 80, "session-1"),
    )
    telegram_file = FakeTelegramFile(b"pptx-bytes")
    context = _make_context(telegram_file)
    message = FakeMessage()
    message.reply_to_message = SimpleNamespace(
        photo=[],
        document=SimpleNamespace(
            file_id="telegram-doc-file",
            file_unique_id="telegram-doc-unique",
            file_name="deck.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            file_size=10,
        ),
        text=None,
        caption=None,
    )
    update = FakeUpdate(message)

    await bot.process_message("сделай заголовок черным", update, context)

    context.bot.get_file.assert_awaited_once_with("telegram-doc-file")
    assert telegram_file.downloaded_paths
    local_path = telegram_file.downloaded_paths[0]
    query = bot.openai.stream_requests[0]["query"]
    assert "сделай заголовок черным" in query
    assert "Telegram reply context:" in query
    assert f"- local_path: {local_path}" in query
    assert "- file_name: deck.pptx" in query
    assert "- size_bytes: 10" in query
    assert not Path(local_path).exists()
    assert not Path(local_path).parent.exists()


@pytest.mark.asyncio
async def test_streaming_reply_text_failure_is_logged_and_does_not_retry_each_chunk(
    monkeypatch,
    caplog,
):
    edit_message = AsyncMock()
    monkeypatch.setattr(telegram_bot, "edit_message_with_retry", edit_message)

    first_send_error = RuntimeError("telegram send failed")
    fallback_message = SimpleNamespace(chat_id=1234, message_id=901)
    message = FakeMessage(reply_side_effects=[first_send_error, fallback_message])
    bot = _make_bot(
        chunks=[
            ("Hello", "not_finished"),
            ("Hello again", "not_finished"),
            ("Hello final", "3"),
        ],
        conversation_context=({"messages": []}, "HTML", 0.8, 80, "session-1"),
    )

    with caplog.at_level(logging.DEBUG, logger="bot.telegram_bot"):
        await bot.process_message("hello", FakeUpdate(message), _make_context())

    assert 1 <= len(message.reply_text_calls) <= 2
    assert len(message.reply_text_calls) < 3
    assert not edit_message.await_args_list
    assert any(record.exc_info for record in caplog.records)
