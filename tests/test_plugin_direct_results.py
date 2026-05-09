import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import constants
from telegram.constants import ReactionEmoji


_INSERTED_MODULES = []


def _install_module_if_missing(name, module):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = module
        _INSERTED_MODULES.append(name)


_markdown2 = types.ModuleType("markdown2")
_markdown2.markdown = lambda text, *args, **kwargs: text
_install_module_if_missing("markdown2", _markdown2)

from bot.utils import direct_result_inline_fallback_text, handle_direct_result, is_direct_result, should_send_text_as_file
from bot.plugins.reaction import ReactionPlugin

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
        self.reply_document = AsyncMock(return_value=SimpleNamespace(message_id=201))
        self.reply_animation = AsyncMock(return_value=SimpleNamespace(message_id=202))


class FakeUpdate:
    def __init__(self, message):
        self.message = message
        self.effective_message = message
        self.callback_query = None
        self.effective_chat = SimpleNamespace(type="private")


@pytest.mark.asyncio
async def test_handle_direct_result_reaction_without_format_sets_reply_target_reaction():
    target = FakeReactionTarget()
    message = FakeMessage(reply_to_message=target)
    response = await ReactionPlugin().execute(
        "react_with_emoji",
        helper=None,
        reaction=ReactionEmoji.FIRE.value,
    )
    assert "format" not in response["direct_result"]

    await handle_direct_result(_config(), FakeUpdate(message), response)

    target.set_reaction.assert_awaited_once()
    reaction_call = target.set_reaction.await_args
    assert (
        reaction_call.args == (ReactionEmoji.FIRE.value,)
        or reaction_call.kwargs.get("reaction") == ReactionEmoji.FIRE.value
    )
    message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_handle_direct_result_reaction_without_reply_target_sends_fallback_message():
    message = FakeMessage()

    sent_messages = await handle_direct_result(_config(), FakeUpdate(message), _reaction_response())

    message.reply_text.assert_awaited_once()
    assert message.reply_text.await_args.kwargs["text"].strip()
    assert sent_messages == [message.reply_text.return_value]


@pytest.mark.asyncio
async def test_handle_direct_result_text_branch_still_replies_with_markdown():
    message = FakeMessage()

    sent_messages = await handle_direct_result(
        _config(),
        FakeUpdate(message),
        {
            "direct_result": {
                "kind": "text",
                "format": "markdown",
                "value": "hello",
            }
        },
    )

    message.reply_text.assert_awaited_once_with(
        message_thread_id=None,
        reply_to_message_id=None,
        text="hello",
        parse_mode=constants.ParseMode.MARKDOWN,
    )
    assert sent_messages == [message.reply_text.return_value]


@pytest.mark.asyncio
async def test_handle_direct_result_final_sends_text_and_multiple_artifacts(tmp_path):
    first_artifact = tmp_path / "one.pptx"
    second_artifact = tmp_path / "two.xlsx"
    first_artifact.write_bytes(b"one")
    second_artifact.write_bytes(b"two")
    message = FakeMessage()

    sent_messages = await handle_direct_result(
        _config(),
        FakeUpdate(message),
        {
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "summary",
                "artifacts": [
                    {"kind": "file", "format": "path", "value": str(first_artifact)},
                    {"kind": "file", "format": "path", "value": str(second_artifact)},
                ],
            }
        },
    )

    message.reply_text.assert_awaited_once()
    assert message.reply_text.await_args.kwargs["text"] == "summary"
    assert message.reply_document.await_count == 2
    assert sent_messages == [
        message.reply_text.return_value,
        message.reply_document.return_value,
        message.reply_document.return_value,
    ]


@pytest.mark.asyncio
async def test_handle_direct_result_file_path_passes_caption(tmp_path):
    artifact = tmp_path / "report.pptx"
    artifact.write_bytes(b"data")
    message = FakeMessage()

    await handle_direct_result(
        _config(),
        FakeUpdate(message),
        {
            "direct_result": {
                "kind": "file",
                "format": "path",
                "value": str(artifact),
                "caption": "Sales overview",
            }
        },
    )

    message.reply_document.assert_awaited_once()
    assert message.reply_document.await_args.kwargs["caption"] == "Sales overview"


@pytest.mark.asyncio
async def test_handle_direct_result_gif_path_uses_reply_animation(tmp_path):
    gif_path = tmp_path / "anim.gif"
    gif_path.write_bytes(b"GIF89a")
    message = FakeMessage()

    sent_messages = await handle_direct_result(
        _config(),
        FakeUpdate(message),
        {
            "direct_result": {
                "kind": "gif",
                "format": "path",
                "value": str(gif_path),
            }
        },
    )

    message.reply_animation.assert_awaited_once()
    message.reply_document.assert_not_called()
    assert sent_messages == [message.reply_animation.return_value]


@pytest.mark.asyncio
async def test_handle_direct_result_gif_url_uses_reply_animation():
    message = FakeMessage()

    await handle_direct_result(
        _config(),
        FakeUpdate(message),
        {
            "direct_result": {
                "kind": "gif",
                "format": "url",
                "value": "https://example.com/anim.gif",
            }
        },
    )

    message.reply_animation.assert_awaited_once()
    message.reply_document.assert_not_called()


def test_is_direct_result_requires_dict_payload_with_kind():
    assert is_direct_result({"direct_result": {"kind": "text", "value": "hi"}}) is True
    assert is_direct_result({"direct_result": True}) is False
    assert is_direct_result({"direct_result": {"value": "no kind"}}) is False
    assert is_direct_result({"other": "x"}) is False
    assert is_direct_result("not json at all") is False
    assert is_direct_result('{"direct_result": {"kind": "text", "value": "hi"}}') is True


def test_direct_result_inline_fallback_preserves_text_and_artifact_names(tmp_path):
    artifact = tmp_path / "report.pdf"
    artifact.write_bytes(b"pdf")

    text = direct_result_inline_fallback_text(
        {
            "direct_result": {
                "kind": "final",
                "text": "summary",
                "artifacts": [{"kind": "file", "format": "path", "value": str(artifact)}],
            }
        },
        "unavailable",
    )

    assert "summary" in text
    assert "report.pdf" in text
    assert "inline mode cannot attach files" in text


def test_markdown_tables_are_sent_as_html_file_when_large():
    table = "| A | B |\n| --- | --- |\n" + "\n".join("| x | y |" for _ in range(300))
    assert should_send_text_as_file(table)


@pytest.mark.asyncio
async def test_handle_direct_result_skips_payload_without_kind():
    message = FakeMessage()
    sent = await handle_direct_result(
        _config(),
        FakeUpdate(message),
        {"direct_result": {"value": "missing kind"}},
    )
    message.reply_text.assert_not_called()
    assert sent == []


@pytest.mark.asyncio
async def test_handle_direct_result_final_long_text_goes_as_html_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    message = FakeMessage()

    sent_messages = await handle_direct_result(
        _config(),
        FakeUpdate(message),
        {
            "direct_result": {
                "kind": "final",
                "format": "mixed",
                "text": "x" * 4097,
                "artifacts": [],
            }
        },
    )

    message.reply_text.assert_not_called()
    message.reply_document.assert_awaited_once()
    assert message.reply_document.await_args.kwargs["filename"].endswith(".html")
    assert sent_messages == [message.reply_document.return_value]
